"""Fara Agent for LM Studio with E2E test instrumentation."""
from __future__ import annotations

import asyncio
import io
import json
import logging
import re
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
        self.reasoning_history: list[str] = []
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
        self._visit_counts: dict[str, int] = {}
        self._just_submitted: bool = False
        self._page_changed: bool = False
        self._last_url_norm: str | None = None
        self._console_errors: list[str] = []
        self._current_run_id: Optional[str] = None
        self._last_action_signature: Optional[str] = None
        self._repeat_action_streak: int = 0
        self._last_page_text: str = ""
        self._visited_url_norms: set[str] = set()
        self._page_changed_since_last_action: bool = False
        self._verified_expectations: list[tuple[str, str]] = []
        self._transitions: list[dict[str, Any]] = []

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

            # Optional request logging for debugging context/loops.
            if self.config.get("debug_log_requests"):
                ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
                run_dir = self._current_run_id or "unknown-run"
                base_dir = Path(str(self.config.get("reports_folder") or "./reports"))
                log_dir = base_dir / "model_requests" / run_dir
                log_dir.mkdir(parents=True, exist_ok=True)
                # Truncate any base64 image URLs to avoid huge logs
                def _truncate_images(msgs: list[dict]) -> list[dict]:
                    out = []
                    for m in msgs:
                        content = m.get("content")
                        if isinstance(content, list):
                            new_items = []
                            for item in content:
                                if isinstance(item, dict) and item.get("type") == "image_url":
                                    # Keep a placeholder marker
                                    new_items.append({"type": "image_url", "image_url": {"url": "<image_base64_truncated>"}})
                                else:
                                    new_items.append(item)
                            m = {**m, "content": new_items}
                        out.append(m)
                    return out

                payload = {
                    "messages": _truncate_images(openai_messages),
                    "create_kwargs": {k: v for k, v in create_kwargs.items() if k != "messages"},
                }
                log_path = log_dir / f"request-{ts}.json"
                try:
                    log_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                except Exception as exc:
                    self.logger.warning(f"Failed to log request payload: {exc}")

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
        def _parse_tool_obj(obj: Any) -> Dict[str, Any] | None:
            if not isinstance(obj, dict):
                return None
            # Preferred envelope
            if obj.get("name") == "computer_use" and isinstance(obj.get("arguments"), dict):
                return obj["arguments"]
            # Sometimes models emit only arguments
            if "action" in obj and isinstance(obj.get("action"), str):
                return obj
            return None

        def _extract_between(tag: str) -> str | None:
            open_tag = f"<{tag}>"
            if open_tag not in response:
                return None
            start = response.find(open_tag) + len(open_tag)
            end = response.find(f"</{tag}>", start)
            if end == -1:
                end = len(response)
            return response[start:end].strip()

        # 1) Tagged tool call
        for tag in ("tool_call", "function_call"):
            payload = _extract_between(tag)
            if payload:
                try:
                    obj = json.loads(payload)
                except Exception:
                    obj = None
                parsed = _parse_tool_obj(obj)
                if parsed:
                    return parsed

        # 2) Raw JSON fallback: attempt to parse the largest {...} block
        try:
            start = response.find("{")
            end = response.rfind("}")
            if start != -1 and end != -1 and end > start:
                obj = json.loads(response[start : end + 1])
                parsed = _parse_tool_obj(obj)
                if parsed:
                    return parsed
        except Exception:
            pass

        return None

    def _allowed_actions(self) -> set[str]:
        """Contract actions exposed in the system prompt (keep Fara's API small)."""
        return {
            "key",
            "type",
            "mouse_move",
            "left_click",
            "scroll",
            "visit_url",
            "web_search",
            "history_back",
            "wait",
            "terminate",
        }

    def _is_action_allowed(self, action_args: dict[str, Any] | None) -> bool:
        if not action_args:
            return False
        action = str(action_args.get("action") or "")
        return action in self._allowed_actions()

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

    def _latest_user_message(self) -> list[UserMessage]:
        """Return only the latest user message (screenshot + state context)."""
        if not self.message_history:
            return []
        return [self.message_history[-1]]

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
                # Avoid wasting rounds re-visiting the exact same URL.
                if self._normalize_for_compare(target) == self._normalize_for_compare(self.browser.get_url()):
                    return "Already at that URL; no navigation performed."
                await self.browser.goto(target)
                # Use intelligent waiting
                await self.browser.wait_for_load_state("networkidle", timeout=10000)
                self._record_visit(target)
                return f"I navigated to '{url}'."

            elif action == "left_click":
                coord = action_args.get("coordinate", [0, 0])
                scaled = self._convert_resized_coords_to_viewport(coord)
                self._record_action_coord(action, scaled)
                
                # Get element info before click
                element_info = await self.browser.click(scaled[0], scaled[1])
                try:
                    clicked_text = (
                        str(element_info.get("text")).strip()
                        if element_info and element_info.get("found") and element_info.get("text")
                        else None
                    )
                    if clicked_text:
                        self.facts.append(f"Clicked: {clicked_text}" if len(clicked_text) < 80 else "Clicked: <text>")
                        self.facts = self.facts[-10:]
                    action_args["_clicked_text"] = clicked_text
                except Exception:
                    pass
                
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
                if abs(int(pixels)) < 300:
                    await self.browser.scroll(int(pixels))
                else:
                    if pixels > 0:
                        await self.browser.page_up()
                    elif pixels < 0:
                        await self.browser.page_down()
                    else:
                        await self.browser.scroll(0)

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
                self._record_visit(self.browser.get_url())
                return "I went back to the previous page."

            elif action == "history_forward":
                await self.browser.go_forward()
                self._record_visit(self.browser.get_url())
                return "I went forward to the next page."

            elif action == "reload":
                await self.browser.reload()
                self._record_visit(self.browser.get_url())
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
        """Assemble minimal context for the model (keep it a policy, not a planner)."""
        current_url = self.browser.get_url()
        lines: list[str] = [f"Current URL: {current_url}", f"Rounds left: {rounds_left}"]

        if self._page_changed:
            lines.append("")
            lines.append("Page changed since last round. Evaluate pass/fail on this page before new actions.")
        if self._just_submitted:
            lines.append("")
            lines.append("You just submitted. Do NOT resubmit. Evaluate and terminate success/failure.")

        suggestion = self._suggest_visible_click_target(test_case=test_case, page_text=self._last_page_text or "")
        if suggestion:
            lines.append("")
            lines.append(f"If you can see \"{suggestion}\", click it now to proceed.")

        # Lightweight "what's satisfied here" hint for the policy model (no big text dumps).
        try:
            scoped = self._extract_scoped_expectations_for_url(test_case, current_url)
            lower_text = (self._last_page_text or "").lower()
            satisfied = [e for e in scoped if e.lower() in lower_text]
            if satisfied:
                lines.append("")
                lines.append("Current page satisfies:")
                for expected in satisfied[:2]:
                    lines.append(f"- \"{expected}\"")
        except Exception:
            pass

        if self._verified_expectations:
            lines.append("")
            lines.append("Verified so far:")
            for url, expected in self._verified_expectations[-3:]:
                lines.append(f"- {url} contains \"{expected}\"")

        if action_history:
            lines.append("")
            lines.append("Recent actions:")
            lines.extend(action_history[-2:])

        repeat_warnings = self._get_repeat_warnings()
        if repeat_warnings:
            lines.append("")
            lines.append("Avoid redundant actions:")
            lines.extend(repeat_warnings)

        lines.append("")
        lines.append("If you cannot make progress in 2-3 actions, terminate with FAILURE and explain the blocker.")
        lines.append("Avoid re-opening the same URL repeatedly if it is already loaded and verified.")
        lines.append("When confident, call terminate with status 'success' or 'failure' and a reason.")
        return "\n".join(lines)

    def _suggest_visible_click_target(self, *, test_case: TestCase, page_text: str) -> str | None:
        """
        Very small, task-derived hint: if an objective step includes a click target in quotes and that exact
        text is currently visible on the page, suggest clicking it. Avoid repeating targets we've already clicked.
        """
        text_lower = (page_text or "").lower()
        if not text_lower:
            return None

        already_clicked = {
            (tr.get("clicked_text") or "").strip().lower()
            for tr in self._transitions
            if tr.get("action") == "left_click" and tr.get("clicked_text")
        }
        for step in test_case.objective_steps or []:
            if "click" not in step.lower():
                continue
            target = self._extract_expected_text(step)
            if not target:
                continue
            target_lower = target.lower()
            if target_lower in already_clicked:
                continue
            if target_lower in text_lower:
                return target
        return None

    def _extract_scoped_expectations_for_url(self, test_case: TestCase, url: str) -> list[str]:
        """Return quoted expectations that are explicitly scoped to a URL in the task text."""
        url_norm = self._normalize_for_compare(url)

        def _url_in_text(s: str) -> str | None:
            match = re.search(r"https?://\S+", s)
            if not match:
                return None
            return self._normalize_for_compare(match.group(0))

        def _is_verify_step(s: str) -> bool:
            sl = s.lower()
            return any(term in sl for term in ("verify", "assert", "confirm", "check", "ensure", "validate", "expect"))

        def _looks_like_page_content_assertion(s: str) -> bool:
            """
            Only treat quoted text in steps as a page-content expectation when the step explicitly
            talks about content/text/labels, not just navigation (e.g. 'browser is at URL').
            """
            sl = s.lower()
            if any(phrase in sl for phrase in ("browser is at", "url is", "address is")):
                return False
            return any(term in sl for term in ("contains", "shows", "visible", "text", "label", "heading"))

        def _scoped_step_url(step: str) -> str | None:
            """
            Extract a URL that is explicitly presented as the page-under-test for this step.
            This avoids mis-scoping expectations in steps like:
              'Click CTA \"X\" and verify the browser returns to http://...'
            """
            m = re.search(
                r"\b(?:on|open|visit|at|navigate to)\s+(https?://\S+)",
                step,
                flags=re.IGNORECASE,
            )
            if not m:
                return None
            return self._normalize_for_compare(m.group(1))

        out: list[str] = []
        for step in test_case.objective_steps or []:
            step_url = _scoped_step_url(step)
            if step_url and step_url == url_norm and _is_verify_step(step) and _looks_like_page_content_assertion(step):
                expected = self._extract_expected_text(step)
                if expected:
                    out.append(expected)
        for crit in test_case.pass_criteria or []:
            crit_url = _url_in_text(crit)
            if crit_url and crit_url == url_norm:
                expected = self._extract_expected_text(crit)
                if expected:
                    out.append(expected)
        # de-dupe, stable order
        seen: set[str] = set()
        uniq: list[str] = []
        for e in out:
            if e not in seen:
                seen.add(e)
                uniq.append(e)
        return uniq

    def _parse_url_in_text(self, text: str) -> str | None:
        match = re.search(r"https?://\S+", text or "")
        if not match:
            return None
        return self._normalize_for_compare(match.group(0))

    def _extract_click_target_texts(self, test_case: TestCase) -> list[str]:
        """Extract quoted click target texts from objective steps (best-effort)."""
        out: list[str] = []
        for step in test_case.objective_steps or []:
            sl = step.lower()
            if "click" not in sl:
                continue
            expected = self._extract_expected_text(step)
            if expected:
                out.append(expected)
        return out

    def _check_auto_pass(self, *, test_case: TestCase, current_url: str) -> tuple[bool, str | None]:
        """
        Auto-pass when:
        - All URL-scoped quoted expectations in pass_criteria were observed on their URLs
        - If pass_criteria mentions a final URL (e.g. 'browser is at http://...'), we are there
        - If the task includes a click step with a quoted target that should lead to the final URL,
          we have evidence of a click-driven transition to that final URL.
        """
        pass_criteria = test_case.pass_criteria or []
        if not pass_criteria:
            return False, None

        # 1) Check URL-scoped quote expectations (url + quote).
        required_pairs: list[tuple[str, str]] = []
        final_url_norm: str | None = None
        for crit in pass_criteria:
            crit = str(crit)
            url_norm = self._parse_url_in_text(crit)
            expected = self._extract_expected_text(crit)
            if url_norm and expected:
                required_pairs.append((url_norm, expected))
            # Try to infer an explicit final URL requirement.
            if url_norm and expected is None and any(
                phrase in crit.lower() for phrase in ("browser is at", "is at", "returns to", "at ")
            ):
                final_url_norm = url_norm

        if required_pairs:
            observed = {(self._normalize_for_compare(u), e) for (u, e) in self._verified_expectations}
            for url_norm, expected in required_pairs:
                if (url_norm, expected) in observed:
                    continue
                # allow current page too
                if self._normalize_for_compare(current_url) == url_norm and expected.lower() in (
                    self._last_page_text or ""
                ).lower():
                    continue
                return False, None

        # If no explicit final URL in criteria, we can't safely auto-pass.
        if not final_url_norm:
            return False, None

        if self._normalize_for_compare(current_url) != final_url_norm:
            return False, None

        # 2) Evidence of click-driven transition to final URL.
        click_targets = [t.lower() for t in self._extract_click_target_texts(test_case)]
        if click_targets:
            for tr in reversed(self._transitions):
                if tr.get("to") != final_url_norm:
                    continue
                if tr.get("action") != "left_click":
                    continue
                clicked = (tr.get("clicked_text") or "").strip().lower()
                if clicked and clicked in click_targets:
                    return True, f"All URL-scoped pass criteria satisfied and clicked '{clicked}' navigated to final URL."
            # Fallback: element text may be missing; accept any recent click-driven transition to the final URL.
            for tr in list(reversed(self._transitions))[:2]:
                if tr.get("to") == final_url_norm and tr.get("action") == "left_click":
                    return True, "All URL-scoped pass criteria satisfied and a recent click navigated to the final URL."
            return False, None

        # No click targets; don't guess.
        return False, None

    def _build_task_brief(self, test_case: TestCase) -> str:
        """Static task definition sent as system message once."""
        lines: list[str] = [f"Objective: {test_case.objective}"]
        if test_case.objective_steps:
            lines.append("")
            lines.append("Steps to follow:")
            for idx, step in enumerate(test_case.objective_steps, start=1):
                lines.append(f"{idx}. {step}")
        if test_case.pass_criteria:
            lines.append("")
            lines.append("Pass criteria:")
            lines.extend([f"- {item}" for item in test_case.pass_criteria])
        if test_case.fail_criteria:
            lines.append("")
            lines.append("Fail criteria:")
            lines.extend([f"- {item}" for item in test_case.fail_criteria])
        if test_case.notes:
            lines.append("")
            lines.append(f"Notes: {test_case.notes}")
        if test_case.credentials:
            lines.append("")
            lines.append("Credentials (keep safe):")
            for k, v in test_case.credentials.items():
                lines.append(f"- {k}: {v}")
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

    def _record_visit(self, url: str) -> None:
        """Track repeated visits to the same URL to catch navigation loops."""
        norm = self._normalize_for_compare(url)
        self._visit_counts[norm] = self._visit_counts.get(norm, 0) + 1

    def _action_signature(self, action_args: dict[str, Any]) -> str:
        """Create a coarse signature for repeat-action detection."""
        action = str(action_args.get("action") or "")
        url = self._normalize_for_compare(self.browser.get_url())
        coord = action_args.get("coordinate")
        bucket = None
        if isinstance(coord, list) and len(coord) == 2:
            scaled = self._convert_resized_coords_to_viewport(coord)
            bucket = self._bucket_coord(scaled)
        pixels = action_args.get("pixels")
        if pixels is not None:
            pixels = int(pixels)
        return f"{action}|{url}|{bucket}|{pixels}"

    def _write_trace(
        self,
        trace_path: Optional[Path],
        *,
        test_case: TestCase,
        started_at: datetime,
        actions: list[ActionTrace],
        reason: str | None = None,
        success: bool | None = None,
    ) -> None:
        """Persist a lightweight partial trace for debugging even if cancelled."""
        if not trace_path:
            return
        try:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "task_id": test_case.id,
                "objective": test_case.objective,
                "success": success,
                "reason": reason,
                "started_at": started_at.isoformat(),
                "actions": [
                    {
                        "round": a.round_index,
                        "action": a.action,
                        "result": a.result,
                        "url": a.page_url,
                        "model_response": a.model_response,
                        "screenshot": str(a.screenshot_path) if a.screenshot_path else None,
                    }
                    for a in actions
                ],
            }
            trace_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            self.logger.warning(f"Failed to write trace file {trace_path}: {exc}")

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
        max_visits = max(self._visit_counts.values(), default=0)
        if max_clicks >= 4:
            return "Loop detected: clicked the same region 4+ times without progress."
        if max_types >= 3:
            return "Loop detected: typed in the same field 3+ times without progress."
        # Visiting the same page repeatedly is often benign in SPA flows unless we are failing to satisfy
        # URL-scoped expectations; rely on the more specific loop_missing_text breaker instead.
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

    def _extract_expected_text(self, criteria: str) -> Optional[str]:
        """Extract quoted expectation from criteria."""
        match = re.search(r'"([^"]+)"', criteria) or re.search(r"'([^']+)'", criteria)
        if match:
            return match.group(1).strip()
        return None

    def _extract_expected_texts(self, criteria: list[str]) -> list[str]:
        """Extract all quoted expectations from a list of strings."""
        out: list[str] = []
        for item in criteria:
            expected = self._extract_expected_text(str(item))
            if expected:
                out.append(expected)
        return out

    def _check_text_expectations(
        self,
        *,
        test_case: TestCase,
        current_url: str,
        page_text: str,
        page_changed: bool,
    ) -> tuple[Optional[bool], Optional[str]]:
        """
        Fast-fail:
        - Only evaluate immediately after a navigation/page-change.
        - Only enforce quoted expectations that are explicitly scoped to the current page via a URL
          in objective_steps or pass_criteria (best practice for machine-checkable tasks).
        """
        if not page_changed:
            return None, None

        url_norm = self._normalize_for_compare(current_url)
        text_lower = (page_text or "").lower()

        expectations = self._extract_scoped_expectations_for_url(test_case, current_url)

        if not expectations:
            return None, None

        missing = [e for e in expectations if e.lower() not in text_lower]
        if missing:
            return False, f"Expected text '{missing[0]}' missing after navigation."

        return None, None

    async def run_test_case(
        self,
        *,
        test_case: TestCase,
        run_id: str,
        screenshots_root: Path,
        trace_path: Optional[Path] = None,
    ) -> TestRunResult:
        """Run a TestCase and collect traces."""
        self.logger.info(f"Running test case: {test_case.id}")
        self._current_run_id = run_id
        original_max = self.max_rounds
        if test_case.max_rounds:
            self.max_rounds = test_case.max_rounds

        # Reset state for new test
        self.message_history.clear()
        self.scroll_history.clear()
        self._click_counts.clear()
        self._type_counts.clear()
        self._visit_counts.clear()
        self.reasoning_history.clear()
        self._last_action_signature = None
        self._repeat_action_streak = 0
        self.facts.clear()
        self._console_errors.clear()
        self._visited_url_norms.clear()
        self._page_changed_since_last_action = False
        self._verified_expectations.clear()
        self._transitions.clear()

        if test_case.start_url:
            await self.browser.goto(test_case.start_url)
            # Use intelligent waiting
            try:
                await self.browser.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                pass  # Continue even if network doesn't become idle
            self._visit_counts.clear()
            self._record_visit(test_case.start_url)
            self._visited_url_norms.add(self._normalize_for_compare(test_case.start_url))

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
        self._last_page_text = page_text or ""
        self._last_url_norm = self._normalize_for_compare(self.browser.get_url())
        if self._last_url_norm:
            self._visited_url_norms.add(self._last_url_norm)
        # Populate verified expectations visible on the initial page.
        try:
            for expected in self._extract_scoped_expectations_for_url(test_case, self.browser.get_url()):
                if expected.lower() in (page_text or "").lower():
                    pair = (self.browser.get_url(), expected)
                    if pair not in self._verified_expectations:
                        self._verified_expectations.append(pair)
        except Exception:
            pass
        prompt_data = get_computer_use_system_prompt(screenshot, self.MLM_PROCESSOR_IM_CFG)
        # Keep the model coordinate space 1:1 with the real browser viewport when possible.
        if self.config.get("sync_viewport_to_prompt", True):
            try:
                target_w, target_h = prompt_data["im_size"]
                if (self.viewport_width, self.viewport_height) != (target_w, target_h):
                    await self.browser.set_viewport_size(target_w, target_h)
                    self.viewport_width, self.viewport_height = target_w, target_h
                    # Ensure scaling is 1:1 if we sync successfully.
                    self.last_im_size = (target_w, target_h)
                    await asyncio.sleep(0.2)
                    screenshot = await self._get_screenshot()
                    prompt_data = get_computer_use_system_prompt(screenshot, self.MLM_PROCESSOR_IM_CFG)
            except Exception as exc:
                self.logger.warning(f"Viewport sync failed; falling back to coordinate scaling: {exc}")
        system_prompt = SystemMessage(
            content=prompt_data["content"]
            + "\n\nYou are executing an end-to-end test case. Be decisive and avoid loops."
        )
        task_brief_msg = SystemMessage(content=self._build_task_brief(test_case))
        action_history: list[str] = []

        for round_num in range(self.max_rounds):
            self.round_count = round_num + 1
            self.logger.info(f"Round {self.round_count}/{self.max_rounds}")

            # page_changed refers to a navigation that happened since the previous model action.
            self._page_changed = bool(self._page_changed_since_last_action)
            if self._page_changed:
                self._click_counts.clear()
                self._type_counts.clear()

            # Auto-check simple textual expectations when the target block is visible.
            text_status, text_reason = self._check_text_expectations(
                test_case=test_case,
                current_url=self.browser.get_url(),
                page_text=page_text,
                page_changed=self._page_changed,
            )
            if text_status is not None:
                # This check is intentionally auto-FAIL only.
                success = False
                fail_reason = text_reason or "Auto fail: expected text missing after navigation."
                screenshot_path = None
                if self.save_screenshots:
                    screenshot_path = screenshot_dir / f"step-{round_num + 1:02d}.png"
                    if not screenshot_path.exists():
                        screenshot.save(screenshot_path)
                actions.append(
                    ActionTrace(
                        round_index=round_num + 1,
                        action="auto_terminate",
                        arguments={
                            "auto": True,
                            "status": "failure",
                            "source": "text_expectations",
                        },
                        model_response="",
                        result=fail_reason,
                        page_url=self.browser.get_url(),
                        screenshot_path=screenshot_path,
                        timestamp=datetime.utcnow(),
                    )
                )
                self._write_trace(
                    trace_path,
                    test_case=test_case,
                    started_at=start_time,
                    actions=actions,
                    reason=fail_reason,
                    success=success,
                )
                break

            # Supervisor auto-pass: if URL-scoped pass criteria are satisfied and the required
            # navigation transition (e.g., CTA click) is evidenced, terminate successfully.
            auto_pass, auto_pass_reason = self._check_auto_pass(test_case=test_case, current_url=self.browser.get_url())
            if auto_pass:
                success = True
                fail_reason = auto_pass_reason or "Pass criteria satisfied."
                screenshot_path = None
                if self.save_screenshots:
                    screenshot_path = screenshot_dir / f"step-{round_num + 1:02d}.png"
                    if not screenshot_path.exists():
                        screenshot.save(screenshot_path)
                actions.append(
                    ActionTrace(
                        round_index=round_num + 1,
                        action="auto_terminate",
                        arguments={"auto": True, "status": "success", "source": "auto_pass"},
                        model_response="",
                        result=fail_reason,
                        page_url=self.browser.get_url(),
                        screenshot_path=screenshot_path,
                        timestamp=datetime.utcnow(),
                    )
                )
                self._write_trace(
                    trace_path,
                    test_case=test_case,
                    started_at=start_time,
                    actions=actions,
                    reason=fail_reason,
                    success=True,
                )
                break


            context_text = self._build_context_text(
                test_case, action_history, self.max_rounds - round_num
            )
            user_content = [
                ImageObj.from_pil(screenshot.resize(prompt_data["im_size"])),
                context_text,
            ]
            self.last_im_size = prompt_data["im_size"]
            user_message = UserMessage(content=user_content)
            self.message_history.append(user_message)
            latest_user = self._latest_user_message()

            messages_for_model = [system_prompt, task_brief_msg, *latest_user]
            
            try:
                response = await self._call_model(messages_for_model)
            except LLMError as e:
                fail_reason = f"LLM error: {e}"
                break
            
            self.logger.info(f"Model response: {response[:200]}...")
            if self.show_overlay:
                await self.browser.update_overlay(f"[INFO] Model response: {response[:300]}")

            action_args = self._parse_action(response)
            if not action_args or not self._is_action_allowed(action_args):
                # One repair turn: ask for a strict tool_call only.
                repair_prompt = UserMessage(
                    content=(
                        "Return ONLY:\n<tool_call>\n{...}\n</tool_call>\n"
                        "No whitespace or text outside the tags. No markdown. No explanations.\n"
                        f"Inside {{…}} return JSON like {{\"name\":\"computer_use\",\"arguments\":{{...}}}}.\n"
                        f"Allowed actions: {sorted(self._allowed_actions())}"
                    )
                )
                try:
                    response = await self._call_model([system_prompt, task_brief_msg, *latest_user, repair_prompt])
                    action_args = self._parse_action(response)
                except Exception:
                    action_args = None
                if not action_args or not self._is_action_allowed(action_args):
                    fail_reason = "Model did not return a valid tool call."
                    break

            # Clear one-shot flags after they've been included in the prompt.
            self._page_changed_since_last_action = False
            self._page_changed = False
            self._just_submitted = False

            # Record concise reasoning (strip tool_call JSON).
            reason_text = response.split("<tool_call>")[0].strip() or response.strip()
            if reason_text:
                trimmed = reason_text[:400]
                self.reasoning_history.append(trimmed)
                self.reasoning_history = self.reasoning_history[-4:]

            # Auto-memorize stable hints.
            try:
                if test_case.start_url:
                    url_norm = self._normalize_for_compare(self.browser.get_url())
                    if url_norm == self._normalize_for_compare(test_case.start_url):
                        fact = f"Visited start_url: {test_case.start_url}"
                        if fact not in self.facts:
                            self.facts.append(fact)
            except Exception:
                pass

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
                self._write_trace(
                    trace_path,
                    test_case=test_case,
                    started_at=start_time,
                    actions=actions,
                    reason=fail_reason,
                    success=success,
                )
                break

            action_start = datetime.utcnow()
            before_url_norm = self._normalize_for_compare(self.browser.get_url())
            result = await self._execute_action(action_args)
            action_duration = (datetime.utcnow() - action_start).total_seconds() * 1000
            self.logger.info(f"Action result: {result}")

            # Repeat-action streak detector (helps break nav/scroll loops early).
            action_sig = self._action_signature(action_args)
            if action_sig == self._last_action_signature:
                self._repeat_action_streak += 1
            else:
                self._repeat_action_streak = 1
                self._last_action_signature = action_sig

            if self._repeat_action_streak >= 3:
                fail_reason = "Loop detected: repeated the same action 3 times in a row without progress."
                actions.append(
                    ActionTrace(
                        round_index=round_num + 1,
                        action=action_args.get("action", "unknown"),
                        arguments=action_args,
                        model_response=response,
                        result=fail_reason,
                        page_url=self.browser.get_url(),
                        screenshot_path=None,
                        timestamp=datetime.utcnow(),
                        duration_ms=action_duration,
                    )
                )
                break

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
                self._write_trace(
                    trace_path,
                    test_case=test_case,
                    started_at=start_time,
                    actions=actions,
                    reason=fail_reason,
                    success=False,
                )
                break

            # Wait for page to stabilize (action-aware: SPA-friendly but faster).
            action_name = str(action_args.get("action") or "")
            if action_name in ("scroll", "mouse_move", "hover"):
                await asyncio.sleep(0.2)
            else:
                try:
                    await self.browser.wait_for_load_state("domcontentloaded", timeout=2500)
                except Exception:
                    pass
                await asyncio.sleep(0.35)

            screenshot = await self._get_screenshot()
            page_title = await self.browser.get_title()
            page_text = await self.browser.get_body_text()
            self._last_page_text = page_text or ""
            after_url = self.browser.get_url()
            after_url_norm = self._normalize_for_compare(after_url)
            self._last_url_norm = after_url_norm
            self._visited_url_norms.add(after_url_norm)
            self._page_changed_since_last_action = after_url_norm != (before_url_norm or "")
            # Only count visits when we actually navigated (click/back/etc).
            if self._page_changed_since_last_action and action_args.get("action") != "visit_url":
                self._record_visit(after_url)

            # Update supervisor memory of verified URL-scoped expectations.
            try:
                scoped = self._extract_scoped_expectations_for_url(test_case, after_url)
                lower_text = (page_text or "").lower()
                for expected in scoped:
                    if expected.lower() in lower_text:
                        pair = (after_url, expected)
                        if pair not in self._verified_expectations:
                            self._verified_expectations.append(pair)
            except Exception:
                pass

            # Record navigation transitions for evidence of click-driven navigation.
            try:
                if self._page_changed_since_last_action:
                    self._transitions.append(
                        {
                            "from": before_url_norm,
                            "to": after_url_norm,
                            "action": action_args.get("action"),
                            "clicked_text": action_args.get("_clicked_text"),
                        }
                    )
                    self._transitions = self._transitions[-20:]
            except Exception:
                pass

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
            self._write_trace(
                trace_path,
                test_case=test_case,
                started_at=start_time,
                actions=actions,
                reason=None,
                success=None,
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
                self._write_trace(
                    trace_path,
                    test_case=test_case,
                    started_at=start_time,
                    actions=actions,
                    reason=fail_reason,
                    success=success,
                )
                break

            # Hard loop breaker: if we've visited this URL 3+ times and its *URL-scoped*
            # expected text is still missing, fail.
            norm_url = self._normalize_for_compare(self.browser.get_url())
            url_visits = self._visit_counts.get(norm_url, 0)
            scoped_expectations = self._extract_scoped_expectations_for_url(test_case, self.browser.get_url())
            if url_visits >= 3 and scoped_expectations:
                page_lower = (page_text or "").lower()
                missing = [e for e in scoped_expectations if e.lower() not in page_lower]
                if missing:
                    success = False
                    fail_reason = f"Loop detected: visited same page 3+ times and missing expected text '{missing[0]}'."
                    actions.append(
                        ActionTrace(
                            round_index=round_num + 1,
                            action="auto_terminate",
                            arguments={"auto": True, "status": "failure", "source": "loop_missing_text"},
                            model_response=response,
                            result=fail_reason,
                            page_url=self.browser.get_url(),
                            screenshot_path=screenshot_path,
                            timestamp=datetime.utcnow(),
                        )
                    )
                    self._write_trace(
                        trace_path,
                        test_case=test_case,
                        started_at=start_time,
                        actions=actions,
                        reason=fail_reason,
                        success=False,
                    )
                    break

            prompt_data = get_computer_use_system_prompt(screenshot, self.MLM_PROCESSOR_IM_CFG)

        end_time = datetime.utcnow()
        self.max_rounds = original_max
        reason = "Completed successfully." if success else fail_reason

        # Final trace flush
        self._write_trace(
            trace_path,
            test_case=test_case,
            started_at=start_time,
            actions=actions,
            reason=reason,
            success=success,
        )

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
