"""Browser controller for Fara agent with multi-browser support and intelligent waiting."""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Literal, Optional

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
    TimeoutError as PlaywrightTimeout,
)

from exceptions import (
    BrowserError,
    BrowserNotStartedError,
    ElementNotFoundError,
    ElementNotInteractableError,
    NavigationError,
    ScreenshotError,
)

BrowserType = Literal["chromium", "firefox", "webkit"]


class SimpleBrowser:
    """Browser manager using Playwright with multi-browser support and intelligent waiting."""

    def __init__(
        self,
        browser_type: BrowserType = "firefox",
        headless: bool = True,
        viewport_width: int = 1440,
        viewport_height: int = 900,
        downloads_folder: str | Path | None = None,
        show_overlay: bool = False,
        show_click_markers: bool = False,
        slow_mo: int = 0,
        logger: Optional[logging.Logger] = None,
    ):
        self.browser_type = browser_type
        self.headless = headless
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.logger = logger or logging.getLogger("browser")
        self.downloads_folder = str(downloads_folder) if downloads_folder else None
        self.show_overlay = show_overlay
        self.show_click_markers = show_click_markers
        self.slow_mo = slow_mo
        self._overlay_created = False
        self._marker_created = False
        self._last_overlay_text: str | None = None

        self._playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.last_download_path: str | None = None
        self._console_messages: list[dict[str, Any]] = []

    def _ensure_started(self) -> None:
        """Raise if browser not started."""
        if self.page is None:
            raise BrowserNotStartedError()

    async def start(self) -> None:
        """Start the browser with specified engine."""
        self._playwright = await async_playwright().start()

        # Select browser engine
        browser_launcher = getattr(self._playwright, self.browser_type)
        launch_options: dict[str, Any] = {"headless": self.headless}
        if self.slow_mo > 0:
            launch_options["slow_mo"] = self.slow_mo

        self.browser = await browser_launcher.launch(**launch_options)
        self.context = await self.browser.new_context(
            viewport={"width": self.viewport_width, "height": self.viewport_height}
        )
        self.page = await self.context.new_page()

        # Capture console messages
        self.page.on("console", self._handle_console)

        if self.downloads_folder:
            os.makedirs(self.downloads_folder, exist_ok=True)
            self.page.on("download", self._handle_download)

        if self.show_overlay:
            await self._setup_overlay_init_script()
            await self._inject_overlay()

        if self.show_click_markers:
            await self._setup_click_marker_init_script()
            await self._inject_click_marker()

        self.logger.info(f"Browser started: {self.browser_type} (headless={self.headless})")

    def _handle_console(self, msg: Any) -> None:
        """Capture console messages."""
        self._console_messages.append({
            "type": msg.type,
            "text": msg.text,
            "location": msg.location,
        })
        # Keep only last 100 messages
        if len(self._console_messages) > 100:
            self._console_messages = self._console_messages[-100:]

    async def _handle_download(self, download: Any) -> None:
        """Handle file downloads."""
        fname = download.suggested_filename
        target = os.path.join(self.downloads_folder or ".", fname)
        try:
            await download.save_as(target)
            self.last_download_path = target
            self.logger.info(f"Download saved to {target}")
        except Exception as e:
            self.logger.error(f"Download save failed: {e}")

    async def close(self) -> None:
        """Close the browser and clean up resources."""
        if self.page:
            await self.page.close()
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self._playwright:
            await self._playwright.stop()
        self.logger.info("Browser closed")

    # ─────────────────────────────────────────────────────────────────────────
    # Navigation with intelligent waiting
    # ─────────────────────────────────────────────────────────────────────────

    async def goto(
        self,
        url: str,
        wait_until: Literal["load", "domcontentloaded", "networkidle", "commit"] = "load",
        timeout: float = 30000,
    ) -> None:
        """Navigate to a URL with configurable wait strategy."""
        self._ensure_started()
        try:
            await self.page.goto(url, wait_until=wait_until, timeout=timeout)
            if self.show_overlay and self._last_overlay_text:
                await self.restore_overlay_text()
        except PlaywrightTimeout as e:
            raise NavigationError(f"Navigation timed out: {url}", url=url, timeout=timeout) from e
        except Exception as e:
            raise NavigationError(f"Navigation failed: {e}", url=url) from e

    async def wait_for_load_state(
        self,
        state: Literal["load", "domcontentloaded", "networkidle"] = "networkidle",
        timeout: float = 30000,
    ) -> None:
        """Wait for page to reach specified load state."""
        self._ensure_started()
        await self.page.wait_for_load_state(state, timeout=timeout)

    async def wait_for_selector(
        self,
        selector: str,
        state: Literal["attached", "detached", "visible", "hidden"] = "visible",
        timeout: float = 10000,
    ) -> bool:
        """Wait for an element matching selector. Returns True if found."""
        self._ensure_started()
        try:
            await self.page.wait_for_selector(selector, state=state, timeout=timeout)
            return True
        except PlaywrightTimeout:
            return False

    async def wait_for_navigation(self, timeout: float = 30000) -> None:
        """Wait for navigation to complete."""
        self._ensure_started()
        async with self.page.expect_navigation(timeout=timeout):
            pass

    async def wait_for_url(self, url_pattern: str, timeout: float = 30000) -> bool:
        """Wait for URL to match pattern. Returns True if matched."""
        self._ensure_started()
        try:
            await self.page.wait_for_url(url_pattern, timeout=timeout)
            return True
        except PlaywrightTimeout:
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # Screenshots
    # ─────────────────────────────────────────────────────────────────────────

    async def screenshot(self, full_page: bool = False) -> bytes:
        """Take a screenshot, hiding debug overlays."""
        self._ensure_started()
        overlay_was_visible = False
        marker_was_visible = False

        try:
            # Hide overlays for clean screenshot
            if self.show_overlay and self._overlay_created:
                overlay_was_visible = await self._toggle_overlay(False)
            if self.show_click_markers and self._marker_created:
                marker_was_visible = await self._toggle_marker(False)

            shot = await self.page.screenshot(full_page=full_page)

            # Restore overlays
            if overlay_was_visible:
                await self._toggle_overlay(True)
            if marker_was_visible:
                await self._toggle_marker(True)

            return shot
        except Exception as e:
            raise ScreenshotError(f"Screenshot failed: {e}") from e

    async def _toggle_overlay(self, visible: bool) -> bool:
        """Toggle overlay visibility, returns previous state."""
        try:
            return await self.page.evaluate(
                f"""() => {{
                    const el = document.getElementById('fara-debug-overlay');
                    if (!el) return false;
                    const wasVisible = el.style.display !== 'none';
                    el.style.display = '{'flex' if visible else 'none'}';
                    return wasVisible;
                }}"""
            )
        except Exception:
            return False

    async def _toggle_marker(self, visible: bool) -> bool:
        """Toggle click marker visibility, returns previous state."""
        try:
            return await self.page.evaluate(
                f"""() => {{
                    const el = document.getElementById('fara-click-marker');
                    if (!el) return false;
                    const wasVisible = el.style.display !== 'none';
                    el.style.display = '{'block' if visible else 'none'}';
                    return wasVisible;
                }}"""
            )
        except Exception:
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # Element targeting and interaction
    # ─────────────────────────────────────────────────────────────────────────

    async def get_element_at(self, x: float, y: float) -> dict[str, Any]:
        """Get detailed info about element at coordinates for pre-flight validation."""
        self._ensure_started()
        try:
            return await self.page.evaluate(
                """([vx, vy]) => {
                    const el = document.elementFromPoint(vx, vy);
                    if (!el) return { found: false };
                    
                    const rect = el.getBoundingClientRect();
                    const style = window.getComputedStyle(el);
                    const isVisible = style.display !== 'none' 
                        && style.visibility !== 'hidden' 
                        && style.opacity !== '0'
                        && rect.width > 0 
                        && rect.height > 0;
                    
                    const isDisabled = el.disabled === true 
                        || el.getAttribute('aria-disabled') === 'true';
                    
                    const isInteractable = isVisible && !isDisabled 
                        && style.pointerEvents !== 'none';
                    
                    return {
                        found: true,
                        tag: (el.tagName || '').toLowerCase(),
                        type: (el.type || '').toLowerCase(),
                        id: el.id || '',
                        className: el.className || '',
                        role: el.getAttribute('role') || '',
                        ariaLabel: el.getAttribute('aria-label') || '',
                        ariaChecked: el.getAttribute('aria-checked'),
                        checked: 'checked' in el ? !!el.checked : null,
                        disabled: isDisabled,
                        text: (el.innerText || '').trim().slice(0, 200),
                        placeholder: el.placeholder || '',
                        value: el.value || '',
                        href: el.href || '',
                        isVisible,
                        isInteractable,
                        rect: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
                        selector: _buildSelector(el),
                    };
                    
                    function _buildSelector(elem) {
                        if (elem.id) return '#' + elem.id;
                        let path = [];
                        while (elem && elem.nodeType === Node.ELEMENT_NODE) {
                            let selector = elem.tagName.toLowerCase();
                            if (elem.id) {
                                selector = '#' + elem.id;
                                path.unshift(selector);
                                break;
                            }
                            let sib = elem, nth = 1;
                            while (sib = sib.previousElementSibling) {
                                if (sib.tagName === elem.tagName) nth++;
                            }
                            if (nth > 1) selector += ':nth-of-type(' + nth + ')';
                            path.unshift(selector);
                            elem = elem.parentElement;
                        }
                        return path.join(' > ');
                    }
                }""",
                [x, y],
            )
        except Exception as e:
            self.logger.warning(f"Failed to get element at ({x}, {y}): {e}")
            return {"found": False}

    async def validate_click_target(self, x: float, y: float) -> tuple[bool, str]:
        """Validate that coordinates point to a clickable element."""
        element = await self.get_element_at(x, y)
        if not element.get("found"):
            return False, "No element found at coordinates"
        if not element.get("isVisible"):
            return False, f"Element is not visible: {element.get('tag')}"
        if not element.get("isInteractable"):
            return False, f"Element is not interactable: {element.get('tag')}, disabled={element.get('disabled')}"
        return True, f"Valid target: {element.get('tag')} {element.get('text', '')[:30]}"

    async def click(
        self,
        x: float,
        y: float,
        validate: bool = False,
        retry_offsets: list[tuple[float, float]] | None = None,
    ) -> dict[str, Any]:
        """Click at coordinates with optional validation and self-healing retries."""
        self._ensure_started()

        if validate:
            valid, reason = await self.validate_click_target(x, y)
            if not valid:
                # Try nearby offsets if provided
                if retry_offsets:
                    for dx, dy in retry_offsets:
                        alt_x, alt_y = x + dx, y + dy
                        valid, reason = await self.validate_click_target(alt_x, alt_y)
                        if valid:
                            x, y = alt_x, alt_y
                            self.logger.info(f"Self-healing: adjusted click to ({x}, {y})")
                            break
                if not valid:
                    raise ElementNotInteractableError(reason, coordinates=(x, y))

        await self.page.mouse.click(x, y)

        # Return element info for logging
        return await self.get_element_at(x, y)

    async def double_click(self, x: float, y: float) -> dict[str, Any]:
        """Double-click at coordinates."""
        self._ensure_started()
        await self.page.mouse.dblclick(x, y)
        return await self.get_element_at(x, y)

    async def right_click(self, x: float, y: float) -> dict[str, Any]:
        """Right-click (context menu) at coordinates."""
        self._ensure_started()
        await self.page.mouse.click(x, y, button="right")
        return await self.get_element_at(x, y)

    async def hover(self, x: float, y: float) -> None:
        """Move cursor without clicking."""
        self._ensure_started()
        await self.page.mouse.move(x, y)

    async def drag_and_drop(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        steps: int = 10,
    ) -> None:
        """Drag from start coordinates to end coordinates."""
        self._ensure_started()
        await self.page.mouse.move(start_x, start_y)
        await self.page.mouse.down()
        await self.page.mouse.move(end_x, end_y, steps=steps)
        await self.page.mouse.up()

    # ─────────────────────────────────────────────────────────────────────────
    # Keyboard input
    # ─────────────────────────────────────────────────────────────────────────

    async def type_text(
        self,
        text: str,
        press_enter: bool = False,
        delete_existing_text: bool = False,
        delay: int = 0,
    ) -> None:
        """Type text, optionally clearing existing input."""
        self._ensure_started()
        if delete_existing_text:
            await self.page.keyboard.press("Control+A")
            await self.page.keyboard.press("Backspace")
        await self.page.keyboard.type(text, delay=delay)
        if press_enter:
            await self.page.keyboard.press("Enter")

    async def press_key(self, key: str) -> None:
        """Press a keyboard key."""
        self._ensure_started()
        await self.page.keyboard.press(key)

    async def press_keys(self, keys: list[str]) -> None:
        """Press multiple keys in sequence."""
        self._ensure_started()
        for key in keys:
            await self.page.keyboard.press(key)

    # ─────────────────────────────────────────────────────────────────────────
    # Form interactions
    # ─────────────────────────────────────────────────────────────────────────

    async def select_option(
        self,
        x: float,
        y: float,
        value: str | None = None,
        label: str | None = None,
        index: int | None = None,
    ) -> list[str]:
        """Select option from dropdown at coordinates."""
        self._ensure_started()
        element = await self.get_element_at(x, y)
        selector = element.get("selector")

        if not selector:
            raise ElementNotFoundError("Could not build selector for element", coordinates=(x, y))

        select_args: dict[str, Any] = {}
        if value is not None:
            select_args["value"] = value
        elif label is not None:
            select_args["label"] = label
        elif index is not None:
            select_args["index"] = index

        return await self.page.select_option(selector, **select_args)

    async def file_upload(self, x: float, y: float, file_paths: list[str]) -> None:
        """Upload files to file input at coordinates."""
        self._ensure_started()
        element = await self.get_element_at(x, y)

        if element.get("tag") != "input" or element.get("type") != "file":
            raise ElementNotInteractableError(
                "Element is not a file input",
                coordinates=(x, y),
                reason=f"Found {element.get('tag')} type={element.get('type')}",
            )

        selector = element.get("selector")
        if not selector:
            raise ElementNotFoundError("Could not build selector for file input", coordinates=(x, y))

        await self.page.set_input_files(selector, file_paths)

    # ─────────────────────────────────────────────────────────────────────────
    # Scrolling
    # ─────────────────────────────────────────────────────────────────────────

    async def scroll(self, pixels: int) -> None:
        """Scroll the page (positive=up, negative=down)."""
        self._ensure_started()
        await self.page.mouse.wheel(0, -pixels)

    async def page_up(self) -> None:
        """Scroll up one page via keyboard."""
        self._ensure_started()
        await self.page.keyboard.press("PageUp")

    async def page_down(self) -> None:
        """Scroll down one page via keyboard."""
        self._ensure_started()
        await self.page.keyboard.press("PageDown")

    async def scroll_to_top(self) -> None:
        """Scroll to top of page."""
        self._ensure_started()
        await self.page.evaluate("window.scrollTo(0, 0)")

    async def scroll_to_bottom(self) -> None:
        """Scroll to bottom of page."""
        self._ensure_started()
        await self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")

    async def get_scroll_position(self) -> dict[str, Any]:
        """Return scroll position info for the current page."""
        self._ensure_started()
        try:
            return await self.page.evaluate(
                """() => {
                    const y = window.scrollY || 0;
                    const x = window.scrollX || 0;
                    const h = Math.max(
                        document.body.scrollHeight || 0,
                        document.documentElement.scrollHeight || 0
                    );
                    const w = Math.max(
                        document.body.scrollWidth || 0,
                        document.documentElement.scrollWidth || 0
                    );
                    const vh = window.innerHeight || 1;
                    const vw = window.innerWidth || 1;
                    return { x, y, scrollHeight: h, scrollWidth: w, viewportH: vh, viewportW: vw };
                }"""
            )
        except Exception:
            return {"x": 0, "y": 0, "scrollHeight": 0, "scrollWidth": 0, "viewportH": 0, "viewportW": 0}

    # ─────────────────────────────────────────────────────────────────────────
    # Navigation controls
    # ─────────────────────────────────────────────────────────────────────────

    async def go_back(self) -> None:
        """Go back in history."""
        self._ensure_started()
        await self.page.go_back()

    async def go_forward(self) -> None:
        """Go forward in history."""
        self._ensure_started()
        await self.page.go_forward()

    async def reload(self) -> None:
        """Reload the page."""
        self._ensure_started()
        await self.page.reload()

    def get_url(self) -> str:
        """Get current URL."""
        self._ensure_started()
        return self.page.url

    async def get_title(self) -> str:
        """Get current page title."""
        self._ensure_started()
        try:
            return await self.page.title()
        except Exception:
            return ""

    # ─────────────────────────────────────────────────────────────────────────
    # Frame and tab management
    # ─────────────────────────────────────────────────────────────────────────

    async def switch_to_frame(self, frame_selector: str) -> bool:
        """Switch to an iframe by selector. Returns True if successful."""
        self._ensure_started()
        try:
            frame = self.page.frame_locator(frame_selector)
            # Verify frame exists by trying to get its content
            await frame.locator("body").count()
            # Store reference for later use
            self._current_frame = frame
            return True
        except Exception as e:
            self.logger.warning(f"Failed to switch to frame {frame_selector}: {e}")
            return False

    async def switch_to_main_frame(self) -> None:
        """Switch back to main frame."""
        self._current_frame = None

    async def get_pages(self) -> list[Page]:
        """Get all open pages/tabs."""
        self._ensure_started()
        return self.context.pages

    async def switch_to_page(self, index: int) -> bool:
        """Switch to page/tab by index. Returns True if successful."""
        self._ensure_started()
        pages = self.context.pages
        if 0 <= index < len(pages):
            self.page = pages[index]
            return True
        return False

    async def new_page(self) -> Page:
        """Open a new page/tab and switch to it."""
        self._ensure_started()
        self.page = await self.context.new_page()
        return self.page

    # ─────────────────────────────────────────────────────────────────────────
    # Page content extraction
    # ─────────────────────────────────────────────────────────────────────────

    async def get_body_text(self, max_len: int = 800) -> str:
        """Return a snippet of the page body text."""
        self._ensure_started()
        try:
            text = await self.page.evaluate(
                """() => {
                    const t = document.body?.innerText || "";
                    return t.slice(0, 1200);
                }"""
            )
            return text[:max_len]
        except Exception:
            return ""

    async def get_accessibility_tree(self) -> dict[str, Any]:
        """Get accessibility tree snapshot for semantic element identification."""
        self._ensure_started()
        try:
            snapshot = await self.page.accessibility.snapshot()
            return snapshot or {}
        except Exception as e:
            self.logger.warning(f"Failed to get accessibility tree: {e}")
            return {}

    def get_console_messages(self) -> list[dict[str, Any]]:
        """Get captured console messages."""
        return self._console_messages.copy()

    def clear_console_messages(self) -> None:
        """Clear captured console messages."""
        self._console_messages.clear()

    # Legacy method for backward compatibility
    async def describe_element_at(self, x: float, y: float) -> dict[str, Any]:
        """Return a lightweight description of the element at viewport coords."""
        return await self.get_element_at(x, y)

    # ─────────────────────────────────────────────────────────────────────────
    # Debug overlay (kept for backward compatibility)
    # ─────────────────────────────────────────────────────────────────────────

    async def _setup_overlay_init_script(self) -> None:
        """Add init script for overlay persistence across navigations."""
        await self.page.add_init_script(
            """() => {
                if (document.getElementById('fara-debug-overlay')) return;
                const el = document.createElement('div');
                el.id = 'fara-debug-overlay';
                el.style.cssText = `
                    position: fixed; bottom: 8px; right: 8px; max-width: 42vw;
                    padding: 10px 12px; border-radius: 10px;
                    font: 12px/1.45 "Fira Code", Menlo, Consolas, monospace;
                    color: #e9f5ff;
                    background: linear-gradient(145deg, rgba(12,17,28,0.92), rgba(20,32,52,0.9));
                    border: 1px solid rgba(255,255,255,0.14);
                    z-index: 2147483647; pointer-events: none;
                    box-shadow: 0 8px 20px rgba(0,0,0,0.45);
                    white-space: pre-wrap; backdrop-filter: blur(6px);
                    max-height: 42vh; overflow: hidden;
                    display: flex; flex-direction: column; gap: 4px; text-align: left;
                `;
                el.textContent = 'Fara debug overlay ready.';
                document.body?.appendChild(el);
            }"""
        )

    async def _inject_overlay(self) -> None:
        """Inject a debug overlay for headful debugging."""
        try:
            created = await self.page.evaluate(
                """() => {
                    const existing = document.getElementById('fara-debug-overlay');
                    if (existing) return true;
                    const el = document.createElement('div');
                    el.id = 'fara-debug-overlay';
                    el.style.cssText = `
                        position: fixed; bottom: 8px; right: 8px; max-width: 42vw;
                        padding: 10px 12px; border-radius: 10px;
                        font: 12px/1.45 "Fira Code", Menlo, Consolas, monospace;
                        color: #e9f5ff;
                        background: linear-gradient(145deg, rgba(12,17,28,0.92), rgba(20,32,52,0.9));
                        border: 1px solid rgba(255,255,255,0.14);
                        z-index: 2147483647; pointer-events: none;
                        box-shadow: 0 8px 20px rgba(0,0,0,0.45);
                        white-space: pre-wrap; backdrop-filter: blur(6px);
                        max-height: 42vh; overflow: hidden;
                        display: flex; flex-direction: column; gap: 4px; text-align: left;
                    `;
                    el.textContent = 'Fara debug overlay ready.';
                    document.body?.appendChild(el);
                    return true;
                }"""
            )
            self._overlay_created = bool(created)
        except Exception as e:
            self.logger.warning(f"Failed to inject overlay: {e}")

    async def update_overlay(self, text: str) -> None:
        """Update debug overlay text."""
        if not self.show_overlay:
            return
        self._last_overlay_text = text
        if not self._overlay_created:
            await self._inject_overlay()
        try:
            await self.page.evaluate(
                """(msg) => {
                    let el = document.getElementById('fara-debug-overlay');
                    if (el) el.textContent = msg;
                }""",
                text[:800],
            )
        except Exception as e:
            self.logger.warning(f"Failed to update overlay: {e}")

    async def restore_overlay_text(self) -> None:
        """Reapply the last overlay text after navigation."""
        if self.show_overlay and self._last_overlay_text:
            await self.update_overlay(self._last_overlay_text)

    async def _setup_click_marker_init_script(self) -> None:
        """Add init script for click marker persistence."""
        await self.page.add_init_script(
            """() => {
                if (document.getElementById('fara-click-marker')) return;
                const el = document.createElement('div');
                el.id = 'fara-click-marker';
                el.style.cssText = `
                    position: fixed; width: 30px; height: 30px; border-radius: 50%;
                    border: 2px solid #5bd1ff;
                    box-shadow: 0 0 12px rgba(91,209,255,0.65);
                    background: rgba(91,209,255,0.15);
                    z-index: 2147483647; pointer-events: none;
                    transform: translate(-50%, -50%); display: none;
                `;
                const label = document.createElement('div');
                label.id = 'fara-click-marker-label';
                label.style.cssText = `
                    position: absolute; bottom: -14px; left: 50%;
                    transform: translateX(-50%);
                    font: 11px/1.2 "Fira Code", Menlo, Consolas, monospace;
                    padding: 2px 6px; border-radius: 6px;
                    background: rgba(0,0,0,0.7); color: #e9f5ff;
                    white-space: nowrap; box-shadow: 0 2px 6px rgba(0,0,0,0.35);
                `;
                label.textContent = 'click';
                el.appendChild(label);
                document.body?.appendChild(el);
            }"""
        )

    async def _inject_click_marker(self) -> None:
        """Ensure the click marker element exists."""
        try:
            created = await self.page.evaluate(
                """() => {
                    let el = document.getElementById('fara-click-marker');
                    if (el) return true;
                    el = document.createElement('div');
                    el.id = 'fara-click-marker';
                    el.style.cssText = `
                        position: fixed; width: 30px; height: 30px; border-radius: 50%;
                        border: 2px solid #5bd1ff;
                        box-shadow: 0 0 12px rgba(91,209,255,0.65);
                        background: rgba(91,209,255,0.15);
                        z-index: 2147483647; pointer-events: none;
                        transform: translate(-50%, -50%); display: none;
                    `;
                    const label = document.createElement('div');
                    label.id = 'fara-click-marker-label';
                    label.style.cssText = `
                        position: absolute; bottom: -14px; left: 50%;
                        transform: translateX(-50%);
                        font: 11px/1.2 "Fira Code", Menlo, Consolas, monospace;
                        padding: 2px 6px; border-radius: 6px;
                        background: rgba(0,0,0,0.7); color: #e9f5ff;
                        white-space: nowrap; box-shadow: 0 2px 6px rgba(0,0,0,0.35);
                    `;
                    label.textContent = 'click';
                    el.appendChild(label);
                    document.body?.appendChild(el);
                    return true;
                }"""
            )
            self._marker_created = bool(created)
        except Exception as e:
            self.logger.warning(f"Failed to inject click marker: {e}")

    async def show_click_marker(self, x: float, y: float, label: str = "click") -> None:
        """Show a transient click marker at viewport coords."""
        if not self.show_click_markers:
            return
        if not self._marker_created:
            await self._inject_click_marker()
        try:
            await self.page.evaluate(
                """([vx, vy, lbl]) => {
                    const el = document.getElementById('fara-click-marker');
                    if (!el) return;
                    const labelEl = el.querySelector('#fara-click-marker-label');
                    if (labelEl) labelEl.textContent = lbl || 'click';
                    el.style.left = `${vx}px`;
                    el.style.top = `${vy}px`;
                    el.style.display = 'block';
                    setTimeout(() => {
                        const el2 = document.getElementById('fara-click-marker');
                        if (el2) el2.style.display = 'none';
                    }, 1000);
                }""",
                [x, y, label[:24]],
            )
        except Exception as e:
            self.logger.warning(f"Failed to show click marker: {e}")
