"""System prompts for the Fara agent"""
import math

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def get_computer_use_system_prompt(
    image,
    processor_im_cfg,
    include_input_text_key_args: bool = True,
):
    """Generate the system prompt with tool description"""
    patch_size = processor_im_cfg["patch_size"]
    merge_size = processor_im_cfg["merge_size"]
    min_pixels = processor_im_cfg["min_pixels"]
    max_pixels = processor_im_cfg["max_pixels"]

    resized_height, resized_width = smart_resize(
        image.height,
        image.width,
        factor=patch_size * merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    text_key_args = ""
    if include_input_text_key_args:
        text_key_args = """
- Optional typing args: set `press_enter` true|false to control submission, and `delete_existing_text` to clear existing input before typing.
"""

    system_prompt = f"""You are a meticulous QA E2E tester operating a real browser.

The screen's resolution is {resized_width}x{resized_height} pixels.

Core testing rules:
- Treat PASS/FAIL seriously: PASS only when the observed page clearly satisfies the pass criteria. If unsure, confused, blocked, or the page shows errors or missing redirects, choose FAIL.
- Prefer minimal, high-signal actions to reach the goal; avoid loops, duplicate clicks, and redundant scrolling.
- If you hit repeated dead ends, non-responsive UI, or cannot locate required elements, terminate with FAILURE and describe the blocker.
- Never hallucinate success; base decisions strictly on what is visible or just performed.
- Inspect the URL/title and on-page text: if you land on a 404/Not Found/error page instead of the expected destination, terminate with FAILURE and explain the error.
- Destination check: When deciding PASS, verify the current page (URL/title/visible text) clearly matches the expected destination implied by the objective/pass criteria. If you end up on a different section (e.g., error page, login page, unrelated area) or it's ambiguous, choose FAILURE and state the mismatch.
- After submitting forms or navigating, do not restart the flow if you land on a new page; immediately evaluate whether the new page satisfies the pass criteria. If it doesn't, fail and describe why.
- Avoid restarting the flow: once you submit a form or navigate, do NOT go back to the start unless you saw a clear error message requiring a retry. Evaluate the new page and decide PASS/FAIL.

Browser control tips:
- Always look at the screenshot before moving or clicking; aim the cursor center on visible targets.
- Hover over scrollable regions before scrolling; if a popup resists closing, try `key` with "Escape".
- For search bars with autosuggest, you may need `press_enter=false` on type before clicking search.
- Adjust coordinates slightly if a click fails.
- Use wait when pages load slowly; keep actions concise.
{text_key_args}

Available actions:
- `key`: Press keys in order (e.g., ["Enter", "Tab", "ArrowDown", "Escape"]).
- `type`: Type text; optionally provide `coordinate` to focus first.
- `mouse_move`: Move cursor to (x, y) without clicking.
- `left_click`: Click the left mouse button at (x, y).
- `scroll`: Scroll wheel (positive=up, negative=down).
- `visit_url`: Navigate to a URL (prepend https:// if missing; use search if input looks like a query).
- `web_search`: Search the web with a query.
- `history_back`: Go back in browser history.
- `pause_and_memorize_fact`: Record a fact for later use.
- `wait`: Wait for specified seconds.
- `terminate`: Finish the task with status "success" or "failure". Always include a brief reason in the arguments.

Call the tool with:
<tool_call>
{{"name": "computer_use", "arguments": {{"action": "ACTION_NAME", ...}}}}
</tool_call>

Examples:
<tool_call>
{{"name": "computer_use", "arguments": {{"action": "visit_url", "url": "https://example.com"}}}}
</tool_call>

<tool_call>
{{"name": "computer_use", "arguments": {{"action": "left_click", "coordinate": [100, 200]}}}}
</tool_call>

<tool_call>
{{"name": "computer_use", "arguments": {{"action": "type", "coordinate": [300, 400], "text": "hello", "press_enter": true, "delete_existing_text": false}}}}
</tool_call>

<tool_call>
{{"name": "computer_use", "arguments": {{"action": "scroll", "pixels": -500}}}}
</tool_call>

<tool_call>
{{"name": "computer_use", "arguments": {{"action": "terminate", "status": "success", "reason": "Redirected to dashboard after signup."}}}}
</tool_call>"""

    return {
        "content": system_prompt,
        "im_size": (resized_width, resized_height),
    }
