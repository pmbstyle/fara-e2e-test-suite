"""Fara Agent for LM Studio with E2E test instrumentation."""
from __future__ import annotations

import asyncio
import io
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from PIL import Image
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from browser import SimpleBrowser, BrowserType
from exceptions import ActionParseError, LLMError, LLMResponseError
from message_types import ImageObj, SystemMessage, UserMessage, message_to_openai_format
from prompts import get_computer_use_system_prompt
from test_types import ActionTrace, TestCase, TestRunResult
from utils import get_trimmed_url


class FaraAgent:
    """Fara agent optimized for LM Studio with multi-browser support."""

    MLM_PROCESSOR_IM_CFG = {
        "min_pixels": 3136,
        "max_pixels": 12845056,
        "patch_size": 14,
        "merge_size": 2,
    }

    def __init__(
        self,
        config: Dict[str, Any],
        headless: bool = True,
        browser_type: BrowserType = "firefox",
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config
        self.headless = headless
        self.browser_type = browser_type
        self.logger = logger or logging.getLogger("fara_agent")
        self.viewport_width = config.get("viewport_width", 1440)
        self.viewport_height = config.get("viewport_height", 900)
        self.last_im_size: tuple[int, int] | None = None
        self.facts: list[str] = []
        self.max_n_images = config.get("max_n_images", 1)
        self.downloads_folder = config.get("downloads_folder")
        self.message_history: list[UserMessage] = []
        self.show_overlay = config.get("show_overlay", not headless)
        self.show_click_markers = config.get("show_click_markers", not headless)

        self.browser = SimpleBrowser(
            browser_type=browser_type,
            headless=headless,
            viewport_width=self.viewport_width,
            viewport_height=self.viewport_height,
            downloads_folder=self.downloads_folder,
            show_overlay=self.show_overlay,
            show_click_markers=self.show_click_markers,
            slow_mo=config.get("slow_mo", 0),
            logger=self.logger,
        )

        self.client = AsyncOpenAI(
            api_key=config.get("api_key", "lm-studio"),
            base_url=config.get("base_url", "http://localhost:1234/v1"),
        )

        self.history: List[Any] = []
        self.max_rounds = config.get("max_rounds", 15)
        self.max_tokens = config.get("max_tokens", 768)
        self.save_screenshots = config.get("save_screenshots", True)
        self.screenshots_folder = config.get("screenshots_folder", "./screenshots")
        self.round_count = 0
        self._is_lm_studio = "1234" in str(config.get("base_url", "")) or "lm-studio" in str(
            config.get("api_key", "")
        )
        self.scroll_history: list[dict[str, Any]] = []
        self._click_counts: dict[tuple[int, int], int] = {}
        self._type_counts: dict[tuple[int, int], int] = {}
        self._just_submitted: bool = False
        self._page_changed: bool = False
        self._last_url_norm: str | None = None
        self._console_errors: list[str] = []

    async def start(self) -> None:
        """Initialize the agent."""
        await self.browser.start()
        self.logger.info(f"Agent started with {self.browser_type} browser")

    async def close(self) -> None:
        """Close the agent."""
        await self.browser.close()
        self.logger.info("Agent closed")

    async def _get_screenshot(self) -> Image.Image:
        """Capture and return screenshot as PIL Image."""
        screenshot_bytes = await self.browser.screenshot()
        return Image.open(io.BytesIO(screenshot_bytes))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2.0, min=2.0, max=10),
        reraise=True,
    )
    async def _call_model(self, messages: List[Any]) -> str:
        """Call the LLM with retry logic."""
        try:
            openai_messages = [message_to_openai_format(msg) for msg in messages]
            create_kwargs = {
                "model": self.config.get("model", "microsoft_fara-7b"),
                "messages": openai_messages,
                "temperature": self.config.get("temperature", 0.1),
                "max_tokens": self.max_tokens,
                "stop": ["</tool_call>", "<|im_end|>", "<|endoftext|>"],
            }
            if self._is_lm_studio:
                create_kwargs["top_p"] = 0.85

            response = await self.client.chat.completions.create(**create_kwargs)
            content = response.choices[0].message.content

            if not content:
                raise LLMResponseError("Empty response from model")

            return content
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"Model call failed: {e}") from e

    def _parse_action(self, response: str) -> Dict[str, Any] | None:
        """Parse tool call from model response."""
        if "<tool_call>" not in response:
            return None

        try:
            start = response.find("<tool_call>") + len("<tool_call>")
            end = response.find("</tool_call>", start)
            if end == -1:
                end = len(response)

            json_str = response[start:end].strip()
            tool_call = json.loads(json_str)

            if tool_call.get("name") == "computer_use":
                return tool_call.get("arguments", {})
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse action JSON: {e}")
            raise ActionParseError(f"Invalid JSON in tool call: {e}", raw_response=response)
        except Exception as e:
            self.logger.error(f"Failed to parse action: {e}")

        return None

    def _convert_resized_coords_to_viewport(self, coords: List[float]) -> List[float]:
        """Scale coordinates from resized prompt image back to browser viewport."""
        if not self.last_im_size:
            return coords
        im_w, im_h = self.last_im_size
        scale_x = self.viewport_width / im_w
        scale_y = self.viewport_height / im_h
        return [coords[0] * scale_x, coords[1] * scale_y]

    def _normalize_url_or_search(self, raw: str) -> str:
        """Return a URL, performing search fallback when the input isn't a full URL."""
        if raw.startswith(("https://", "http://", "file://", "about:")):
            return raw
        if " " in raw:
            return f"https://www.bing.com/search?q={raw}"
        return f"https://{raw}"

    def _prune_user_messages(self) -> list[UserMessage]:
        """Keep a short text history while sending only one image (latest turn) to the model."""
        if not self.message_history:
            return []
        pruned: list[UserMessage] = []
        for msg in self.message_history[:-1]:
            if isinstance(msg.content, list):
                text_parts = [item for item in msg.content if not isinstance(item, ImageObj)]
                pruned.append(UserMessage(content=text_parts or ["[previous screenshot omitted]"]))
            else:
                pruned.append(msg)
        pruned.append(self.message_history[-1])
        return pruned

    async def _execute_action(self, action_args: Dict[str, Any]) -> str:
        """Execute a browser action with expanded action set."""
        action = action_args.get("action")
        # Normalize action names
        if action == "click":
            action = "left_click"
        if action == "input_text":
            action = "type"

        try:
            if action == "visit_url":
                url = action_args.get("url")
                if not url:
                    return "No URL provided."
                target = self._normalize_url_or_search(str(url))
                await self.browser.goto(target)
                # Use intelligent waiting
                await self.browser.wait_for_load_state("networkidle", timeout=10000)
                return f"I navigated to '{url}'."

            elif action == "left_click":
                coord = action_args.get("coordinate", [0, 0])
                scaled = self._convert_resized_coords_to_viewport(coord)
                self._record_action_coord(action, scaled)
                
                # Get element info before click
                element_info = await self.browser.click(scaled[0], scaled[1])
                
                label = action_args.get("label", "") or ""
                if any(
                    term in label.lower()
                    for term in ("sign up", "sign-in", "signin", "login", "submit", "confirm", "continue")
                ):
                    self._just_submitted = True

                info_str = self._format_element_info(element_info)
                if self.show_click_markers:
                    await self.browser.show_click_marker(scaled[0], scaled[1], "click")
                return f"I clicked at coordinates ({scaled[0]:.1f}, {scaled[1]:.1f}).{info_str}"

            elif action == "double_click":
                coord = action_args.get("coordinate", [0, 0])
                scaled = self._convert_resized_coords_to_viewport(coord)
                element_info = await self.browser.double_click(scaled[0], scaled[1])
                info_str = self._format_element_info(element_info)
                if self.show_click_markers:
                    await self.browser.show_click_marker(scaled[0], scaled[1], "dblclick")
                return f"I double-clicked at ({scaled[0]:.1f}, {scaled[1]:.1f}).{info_str}"

            elif action == "right_click":
                coord = action_args.get("coordinate", [0, 0])
                scaled = self._convert_resized_coords_to_viewport(coord)
                element_info = await self.browser.right_click(scaled[0], scaled[1])
                info_str = self._format_element_info(element_info)
                if self.show_click_markers:
                    await self.browser.show_click_marker(scaled[0], scaled[1], "right")
                return f"I right-clicked at ({scaled[0]:.1f}, {scaled[1]:.1f}).{info_str}"

            elif action in ("mouse_move", "hover"):
                coord = action_args.get("coordinate", [0, 0])
                scaled = self._convert_resized_coords_to_viewport(coord)
                await self.browser.hover(scaled[0], scaled[1])
                if self.show_click_markers:
                    await self.browser.show_click_marker(scaled[0], scaled[1], "hover")
                return f"I moved the cursor to ({scaled[0]:.1f}, {scaled[1]:.1f})."

            elif action == "drag_and_drop":
                start_coord = action_args.get("start_coordinate", [0, 0])
                end_coord = action_args.get("end_coordinate", [0, 0])
                start_scaled = self._convert_resized_coords_to_viewport(start_coord)
                end_scaled = self._convert_resized_coords_to_viewport(end_coord)
                await self.browser.drag_and_drop(
                    start_scaled[0], start_scaled[1], end_scaled[0], end_scaled[1]
                )
                return f"I dragged from ({start_scaled[0]:.1f}, {start_scaled[1]:.1f}) to ({end_scaled[0]:.1f}, {end_scaled[1]:.1f})."

            elif action == "type":
                coord = action_args.get("coordinate")
                text = action_args.get("text", "")
                press_enter = action_args.get("press_enter", False)
                delete_existing_text = action_args.get("delete_existing_text", False)

                if coord:
                    scaled = self._convert_resized_coords_to_viewport(coord)
                    self._record_action_coord(action, scaled)
                    await self.browser.click(scaled[0], scaled[1])
                    if self.show_click_markers:
                        await self.browser.show_click_marker(scaled[0], scaled[1], "type")

                await self.browser.type_text(text, press_enter, delete_existing_text)
                if press_enter:
                    self._just_submitted = True
                return f"I typed '{text}'."

            elif action == "select_option":
                coord = action_args.get("coordinate", [0, 0])
                scaled = self._convert_resized_coords_to_viewport(coord)
                value = action_args.get("value")
                label = action_args.get("label")
                index = action_args.get("index")
                selected = await self.browser.select_option(
                    scaled[0], scaled[1], value=value, label=label, index=index
                )
                return f"I selected option: {selected}."

            elif action == "file_upload":
                coord = action_args.get("coordinate", [0, 0])
                scaled = self._convert_resized_coords_to_viewport(coord)
                files = action_args.get("files", [])
                if isinstance(files, str):
                    files = [files]
                await self.browser.file_upload(scaled[0], scaled[1], files)
                return f"I uploaded {len(files)} file(s)."

            elif action == "scroll":
                pixels = action_args.get("pixels", 0)
                direction = "up" if pixels > 0 else "down"
                if pixels > 0:
                    await self.browser.page_up()
                elif pixels < 0:
                    await self.browser.page_down()
                else:
                    await self.browser.scroll(pixels)

                scroll_state = await self.browser.get_scroll_position()
                self.scroll_history.append(
                    {
                        "direction": direction,
                        "y": scroll_state.get("y", 0),
                        "scrollHeight": scroll_state.get("scrollHeight", 0),
                        "timestamp": asyncio.get_event_loop().time(),
                    }
                )
                return f"I scrolled {direction} one page."

            elif action in ("key", "keypress"):
                keys = action_args.get("keys", [])
                if not keys:
                    return "No keys provided."
                await self.browser.press_keys(keys)
                return f"I pressed keys: {', '.join(keys)}."

            elif action == "history_back":
                await self.browser.go_back()
                return "I went back to the previous page."

            elif action == "history_forward":
                await self.browser.go_forward()
                return "I went forward to the next page."

            elif action == "reload":
                await self.browser.reload()
                return "I reloaded the page."

            elif action == "web_search":
                query = action_args.get("query", "")
                search_url = f"https://www.bing.com/search?q={query}"
                await self.browser.goto(search_url)
                return f"I searched for '{query}'."

            elif action == "wait":
                time_secs = action_args.get("time", action_args.get("duration", 1)) or 1
                await asyncio.sleep(time_secs)
                return f"I waited for {time_secs} seconds."

            elif action == "wait_for_element":
                selector = action_args.get("selector", "")
                timeout = action_args.get("timeout", 10000)
                found = await self.browser.wait_for_selector(selector, timeout=timeout)
                return f"Element {'found' if found else 'not found'}: {selector}."

            elif action == "switch_frame":
                frame_selector = action_args.get("frame", "")
                success = await self.browser.switch_to_frame(frame_selector)
                return f"Switched to frame: {frame_selector}" if success else f"Failed to switch to frame: {frame_selector}"

            elif action == "switch_tab":
                index = action_args.get("index", 0)
                success = await self.browser.switch_to_page(index)
                return f"Switched to tab {index}" if success else f"Failed to switch to tab {index}"

            elif action == "pause_and_memorize_fact":
                fact = action_args.get("fact") or ""
                if fact:
                    self.facts.append(str(fact))
                return "I memorized a fact."

            elif action == "terminate":
                status = action_args.get("status", "success")
                if self.facts:
                    return f"Task completed with status: {status}. Facts: {self.facts}"
                return f"Task completed with status: {status}"

            else:
                return f"Unknown action: {action}"

        except Exception as e:
            self.logger.error(f"Action execution failed: {e}")
            return f"Action failed: {str(e)}"

    def _format_element_info(self, element_info: dict[str, Any]) -> str:
        """Format element info for action result."""
        if not element_info or not element_info.get("found"):
            return ""
        pieces = []
        if element_info.get("tag"):
            pieces.append(f"tag={element_info.get('tag')}")
        if element_info.get("type"):
            pieces.append(f"type={element_info.get('type')}")
        if element_info.get("role"):
            pieces.append(f"role={element_info.get('role')}")
        if element_info.get("ariaChecked"):
            pieces.append(f"aria-checked={element_info.get('ariaChecked')}")
        if element_info.get("checked") is not None:
            pieces.append(f"checked={element_info.get('checked')}")
        if element_info.get("text"):
            pieces.append(f"text='{element_info.get('text')[:50]}'")
        return " Element: " + ", ".join(pieces) if pieces else ""

    def _normalize_for_compare(self, url: str) -> str:
        """Normalize URL for change detection."""
        return get_trimmed_url(url or "", 300).lower()

    def _build_context_text(
        self, test_case: TestCase, action_history: list[str], rounds_left: int
    ) -> str:
        """Assemble concise context for the model."""
        current_url = self.browser.get_url()
        lines = [
            f"Objective: {test_case.objective}",
            f"Current URL: {current_url}",
            f"Rounds left: {rounds_left}",
            "",
            "PASS criteria:",
            *[f"- {item}" for item in test_case.pass_criteria],
            "",
            "FAIL criteria:",
            *[f"- {item}" for item in test_case.fail_criteria],
        ]
        if test_case.credentials:
            lines.append("")
            lines.append("Credentials:")
            for k, v in test_case.credentials.items():
                lines.append(f"- {k}: {v}")
        if test_case.notes:
            lines.append("")
            lines.append(f"Notes: {test_case.notes}")
        if test_case.start_url and self.browser.get_url() != test_case.start_url:
            lines.append("")
            lines.append(
                "You have left the starting page. Do NOT restart or refill the form unless an error requires it."
            )
            lines.append(
                "Compare the current page to the pass criteria. If it matches, terminate with success."
            )
        if self._page_changed:
            lines.append("")
            lines.append(
                "The page changed. Evaluate for PASS/FAIL now. Do NOT redo earlier steps unless an error demands it."
            )
        if self._just_submitted:
            lines.append("")
            lines.append(
                "You just submitted. Do NOT refill the form. Evaluate and terminate success/failure."
            )
        lines.append("")
        lines.append(
            "If you clicked submit/confirm or pressed Enter, evaluate the new page and terminate."
        )
        lines.append(
            "If stuck clicking the same element, change approach or terminate with failure."
        )
        if action_history:
            lines.append("")
            lines.append("Recent actions:")
            lines.extend(action_history[-6:])
        repeat_warnings = self._get_repeat_warnings()
        if repeat_warnings:
            lines.append("")
            lines.append("Avoid redundant actions:")
            lines.extend(repeat_warnings)
        if self.scroll_history:
            last_scroll = self.scroll_history[-1]
            sh = last_scroll.get("scrollHeight", 0) or 1
            y = last_scroll.get("y", 0)
            pct = (y / sh) * 100
            lines.append("")
            lines.append(f"Scroll position: {y:.0f}/{sh:.0f} ({pct:.1f}%).")
            recent_dirs = [s["direction"] for s in self.scroll_history[-6:]]
            if "up" in recent_dirs and "down" in recent_dirs and len(recent_dirs) >= 4:
                lines.append(
                    "Loop warning: Scrolling up/down repeatedly. Click a result instead."
                )
        lines.append("")
        lines.append(
            "When confident, call terminate with status 'success' or 'failure' and a reason."
        )
        return "\n".join(lines)

    def _bucket_coord(self, coord: list[float] | tuple[float, float]) -> tuple[int, int]:
        """Bucket coordinates to reduce noise in repeat detection."""
        return (int(round(coord[0] / 20) * 20), int(round(coord[1] / 20) * 20))

    def _record_action_coord(
        self, action: str, coord: list[float] | tuple[float, float] | None
    ) -> None:
        """Track repeated interactions on the same screen region."""
        if coord is None:
            return
        bucket = self._bucket_coord(coord)
        if action in ("left_click", "click"):
            self._click_counts[bucket] = self._click_counts.get(bucket, 0) + 1
        if action == "type":
            self._type_counts[bucket] = self._type_counts.get(bucket, 0) + 1

    def _get_repeat_warnings(self) -> list[str]:
        """Return hints to avoid flapping on the same element."""
        warnings: list[str] = []
        for (x, y), count in self._click_counts.items():
            if count >= 2:
                note = f"- Clicked near ({x},{y}) {count} times; avoid unless visibly unchecked."
                if count >= 3:
                    note += " Change approach or terminate failure."
                warnings.append(note)
        for (x, y), count in self._type_counts.items():
            if count >= 2:
                note = f"- Typed near ({x},{y}) {count} times; avoid unless an error is visible."
                if count >= 3:
                    note += " Change approach or terminate failure."
                warnings.append(note)
        return warnings

    def _detect_loop_blocker(self) -> str | None:
        """Return a reason to abort if we are looping on the same element."""
        max_clicks = max(self._click_counts.values(), default=0)
        max_types = max(self._type_counts.values(), default=0)
        if max_clicks >= 4:
            return "Loop detected: clicked the same region 4+ times without progress."
        if max_types >= 3:
            return "Loop detected: typed in the same field 3+ times without progress."
        return None

    def _check_auto_verdict(
        self,
        *,
        test_case: TestCase,
        current_url: str,
        page_title: str,
        page_text: str,
    ) -> tuple[Optional[bool], Optional[str]]:
        """Conservative heuristics: auto-FAIL on obvious error pages only."""
        url_lower = (current_url or "").lower()
        title_lower = (page_title or "").lower()
        text_lower = (page_text or "").lower()
        fail_blob = " ".join(test_case.fail_criteria).lower()
        start_url = (test_case.start_url or "").rstrip("/")
        error_terms = ("404", "not-found", "not found", "error", "fail", "oops", "uh-oh")

        if any(term in url_lower for term in error_terms) or any(
            term in title_lower for term in error_terms
        ):
            return False, "Landed on an error/404 page."
        if any(term in text_lower for term in ("404", "not found")):
            return False, "Page body shows 404/not found."
        if (
            "login" in fail_blob
            and "login" in url_lower
            and start_url
            and url_lower == start_url.lower()
        ):
            return False, "Still on login page; fail criteria mention login."

        return None, None

    async def run_test_case(
        self,
        *,
        test_case: TestCase,
        run_id: str,
        screenshots_root: Path,
    ) -> TestRunResult:
        """Run a TestCase and collect traces."""
        self.logger.info(f"Running test case: {test_case.id}")
        original_max = self.max_rounds
        if test_case.max_rounds:
            self.max_rounds = test_case.max_rounds

        # Reset state for new test
        self.message_history.clear()
        self.scroll_history.clear()
        self._click_counts.clear()
        self._type_counts.clear()
        self.facts.clear()
        self._console_errors.clear()

        if test_case.start_url:
            await self.browser.goto(test_case.start_url)
            # Use intelligent waiting
            try:
                await self.browser.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                pass  # Continue even if network doesn't become idle

        screenshot_dir = screenshots_root / run_id
        if self.save_screenshots:
            screenshot_dir.mkdir(parents=True, exist_ok=True)

        start_time = datetime.utcnow()
        actions: list[ActionTrace] = []
        success = False
        fail_reason = "Max rounds reached."

        # Initial wait for page to stabilize
        await asyncio.sleep(1.5)
        screenshot = await self._get_screenshot()
        page_title = await self.browser.get_title()
        page_text = await self.browser.get_body_text()
        self._last_url_norm = self._normalize_for_compare(self.browser.get_url())
        prompt_data = get_computer_use_system_prompt(screenshot, self.MLM_PROCESSOR_IM_CFG)
        system_prompt = SystemMessage(
            content=prompt_data["content"]
            + "\n\nYou are executing an end-to-end test case. Be decisive and avoid loops."
        )
        action_history: list[str] = []

        for round_num in range(self.max_rounds):
            self.round_count = round_num + 1
            self.logger.info(f"Round {self.round_count}/{self.max_rounds}")

            current_url_norm = self._normalize_for_compare(self.browser.get_url())
            self._page_changed = current_url_norm != (self._last_url_norm or "")
            if self._page_changed:
                self._click_counts.clear()
                self._type_counts.clear()

            context_text = self._build_context_text(
                test_case, action_history, self.max_rounds - round_num
            )
            context_text = f"{context_text}\n\nPage title: {page_title}\nPage text snippet: {page_text[:220]}"
            user_content = [
                ImageObj.from_pil(screenshot.resize(prompt_data["im_size"])),
                context_text,
            ]
            self.last_im_size = prompt_data["im_size"]
            user_message = UserMessage(content=user_content)
            self.message_history.append(user_message)
            pruned_users = self._prune_user_messages()

            messages_for_model = [system_prompt, *pruned_users]
            
            try:
                response = await self._call_model(messages_for_model)
            except LLMError as e:
                fail_reason = f"LLM error: {e}"
                break
                
            self.logger.info(f"Model response: {response[:200]}...")
            if self.show_overlay:
                await self.browser.update_overlay(f"[INFO] Model response: {response[:300]}")

            action_args = self._parse_action(response)
            if not action_args:
                fail_reason = "Model did not return a valid tool call."
                break

            if action_args.get("action") == "terminate":
                status = action_args.get("status", "").lower() or "failure"
                reason = action_args.get("reason") or action_args.get("message") or ""
                success = status == "success"
                fail_reason = reason or f"Model terminated with status: {status}"
                actions.append(
                    ActionTrace(
                        round_index=round_num + 1,
                        action="terminate",
                        arguments=action_args,
                        model_response=response,
                        result=reason or "terminate",
                        page_url=self.browser.get_url(),
                        screenshot_path=None,
                        timestamp=datetime.utcnow(),
                    )
                )
                break

            action_start = datetime.utcnow()
            result = await self._execute_action(action_args)
            action_duration = (datetime.utcnow() - action_start).total_seconds() * 1000
            self.logger.info(f"Action result: {result}")

            loop_reason = self._detect_loop_blocker()
            if loop_reason:
                fail_reason = loop_reason
                actions.append(
                    ActionTrace(
                        round_index=round_num + 1,
                        action=action_args.get("action", "unknown"),
                        arguments=action_args,
                        model_response=response,
                        result=loop_reason,
                        page_url=self.browser.get_url(),
                        screenshot_path=None,
                        timestamp=datetime.utcnow(),
                        duration_ms=action_duration,
                    )
                )
                break

            # Wait for page to stabilize
            await asyncio.sleep(1.5)
            try:
                await self.browser.wait_for_load_state("domcontentloaded", timeout=3000)
            except Exception:
                pass

            screenshot = await self._get_screenshot()
            page_title = await self.browser.get_title()
            page_text = await self.browser.get_body_text()
            self._last_url_norm = self._normalize_for_compare(self.browser.get_url())
            self._page_changed = False
            self._just_submitted = False

            # Capture console errors
            console_msgs = self.browser.get_console_messages()
            console_errors = [m["text"] for m in console_msgs if m.get("type") == "error"]
            if console_errors:
                self._console_errors.extend(console_errors[-5:])

            screenshot_path = None
            if self.save_screenshots:
                screenshot_path = screenshot_dir / f"step-{round_num + 1:02d}.png"
                screenshot.save(screenshot_path)

            # Get element info for trace
            coord = action_args.get("coordinate")
            element_info = None
            if coord:
                scaled = self._convert_resized_coords_to_viewport(coord)
                element_info = await self.browser.get_element_at(scaled[0], scaled[1])

            actions.append(
                ActionTrace(
                    round_index=round_num + 1,
                    action=action_args.get("action", "unknown"),
                    arguments=action_args,
                    model_response=response,
                    result=result,
                    page_url=self.browser.get_url(),
                    screenshot_path=screenshot_path,
                    timestamp=datetime.utcnow(),
                    duration_ms=action_duration,
                    element_info=element_info,
                    console_errors=console_errors[-3:] if console_errors else None,
                )
            )

            action_summary = f"{round_num + 1}. {action_args.get('action')}: {result}"
            action_history.append(action_summary)

            # Auto-verdict if obviously done
            auto_status, auto_reason = self._check_auto_verdict(
                test_case=test_case,
                current_url=self.browser.get_url(),
                page_title=page_title,
                page_text=page_text,
            )
            if auto_status is not None:
                success = bool(auto_status)
                fail_reason = auto_reason or ("Auto pass" if success else "Auto fail")
                actions.append(
                    ActionTrace(
                        round_index=round_num + 1,
                        action="auto_terminate",
                        arguments={"auto": True, "status": "success" if success else "failure"},
                        model_response=response,
                        result=fail_reason,
                        page_url=self.browser.get_url(),
                        screenshot_path=screenshot_path,
                        timestamp=datetime.utcnow(),
                    )
                )
                break

            prompt_data = get_computer_use_system_prompt(screenshot, self.MLM_PROCESSOR_IM_CFG)

        end_time = datetime.utcnow()
        self.max_rounds = original_max
        reason = "Completed successfully." if success else fail_reason

        return TestRunResult(
            case=test_case,
            success=success,
            started_at=start_time,
            finished_at=end_time,
            reason=reason,
            actions=actions,
            facts=self.facts.copy(),
            browser_type=self.browser_type,
            final_url=self.browser.get_url(),
            console_errors=self._console_errors.copy(),
        )

    async def run(self, task: str) -> None:
        """Legacy runner that wraps a simple string task."""
        test_case = TestCase(
            id="adhoc",
            objective=task,
            pass_criteria=["The objective is reached."],
            fail_criteria=["The objective is not reached."],
        )
        await self.run_test_case(
            test_case=test_case,
            run_id=f"adhoc-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
            screenshots_root=Path(self.screenshots_folder),
        )
