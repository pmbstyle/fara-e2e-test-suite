# Fara E2E Test Suite

A production-ready AI-powered end-to-end testing framework that uses vision-language models to interact with web applications through natural language test definitions. Configured for [Microsoft Fara 7b GGUF](https://huggingface.co/bartowski/microsoft_Fara-7B-GGUF) model and [LM Studio](https://lmstudio.ai/) inference.

<a href="https://www.youtube.com/watch?v=ptI_Tz1mams"><img alt="image" src="https://github.com/user-attachments/assets/e7eccf41-8028-4444-b07b-db980dbf4129" /></a>



## Features

- **AI-Powered Testing**: Uses vision-language models (via OpenAI-compatible API) to understand and interact with web pages.
- **Natural Language Tests**: Define tests in simple YAML/JSON with objectives, pass/fail criteria
- **Multi-Browser Support**: Chromium, Firefox, and WebKit via Playwright
- **Parallel Execution**: Run multiple tests concurrently with configurable workers
- **Rich Reporting**: HTML (with timeline view), JSON, and JUnit XML reports
- **Test Tagging**: Organize tests with tags for selective execution
- **Intelligent Waiting**: Smart page load detection instead of fixed delays
- **Self-Healing**: Pre-flight element validation and retry mechanisms

## Quick Start

### Installation

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
playwright install chromium firefox webkit
```

### Configuration

Create a `config.json` or use environment variables:

```json
{
  "agent": {
    "model": "microsoft_fara-7b",
    "base_url": "http://localhost:1234/v1",
    "api_key": "lm-studio",
    "temperature": 0.1,
    "max_rounds": 20
  },
  "browser": {
    "browser": "firefox",
    "headless": true
  },
  "reporting": {
    "output_format": "html",
    "embed_screenshots": false
  }
}
```

Or use environment variables:
- `FARA_BASE_URL` - LLM API endpoint
- `FARA_API_KEY` - API key for the LLM service
- `FARA_MODEL` - Model name

### Defining Tests

Create YAML files in the `tasks/` directory:

```yaml
id: login-test
objective: Log in and verify dashboard is visible
pass_criteria:
  - User is logged in successfully
  - Dashboard/home view is visible
fail_criteria:
  - Login error is displayed
  - Still on login page after submit
start_url: https://example.com/login
credentials:
  email: test@example.com
  password: "TestPassword123"
tags:
  - smoke
  - auth
  - p0
priority: 1
max_rounds: 10
retry_count: 2
```

### Running Tests

```bash
# Run all tests
python test_runner.py

# Run specific test
python test_runner.py --task login-test

# Run tests with specific tags
python test_runner.py --tag smoke --exclude-tag slow

# Parallel execution with Chrome
python test_runner.py --parallel 4 --browser chromium

# Headful mode for debugging
python test_runner.py --task login-test --headful

# Generate all report formats
python test_runner.py --output-format all
```

## CLI Options

### Task Selection
- `--tasks-dir DIR` - Directory containing task files (default: `tasks`)
- `--task ID` - Run specific task by ID (repeatable)
- `--tag TAG` - Include tests with this tag (repeatable)
- `--exclude-tag TAG` - Exclude tests with this tag (repeatable)
- `--include-skipped` - Include tests marked as skip=true
- `--sort-by-priority` - Run higher priority tests first

### Browser Options
- `--browser [chromium|firefox|webkit]` - Browser engine (default: firefox)
- `--headful` - Show browser window
- `--base-url URL` - Override start URLs

### Execution Options
- `--parallel N` - Number of parallel workers (default: 1)
- `--config FILE` - Path to config file

### Output Options
- `--reports-dir DIR` - Reports directory (default: `reports`)
- `--output-format [html|json|junit|all]` - Report format
- `--verbose, -v` - Verbose output
- `--quiet, -q` - Minimal output

## Test Case Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes* | Unique identifier (defaults to filename) |
| `objective` | string | Yes | What the test should accomplish |
| `pass_criteria` | list | Yes | Conditions for test success |
| `fail_criteria` | list | Yes | Conditions for test failure |
| `start_url` | string | No | Initial page URL |
| `credentials` | object | No | Login credentials as key-value pairs |
| `notes` | string | No | Additional context for the AI agent |
| `max_rounds` | int | No | Maximum action rounds (default: 20) |
| `tags` | list | No | Tags for filtering tests |
| `skip` | bool | No | Skip this test (default: false) |
| `skip_reason` | string | No | Reason for skipping |
| `retry_count` | int | No | Retry attempts on failure (default: 0) |
| `priority` | int | No | 1-10, lower = higher priority (default: 5) |
| `timeout_seconds` | float | No | Test timeout |
| `owner` | string | No | Test owner/team |

## Reports

### HTML Report
Interactive report with:
- Timeline view of actions
- Filterable actions table
- Embedded or linked screenshots
- Model response inspection

### JSON Report
Machine-readable format for:
- CI/CD integration
- Custom dashboards
- Historical analysis

### JUnit XML
Compatible with:
- Jenkins
- GitHub Actions
- GitLab CI
- Azure DevOps

## Available Actions

The AI agent can perform:

| Action | Description |
|--------|-------------|
| `visit_url` | Navigate to a URL |
| `left_click` | Single click at coordinates |
| `double_click` | Double-click at coordinates |
| `right_click` | Context menu click |
| `hover` | Move mouse without clicking |
| `type` | Enter text with optional Enter key |
| `select_option` | Select dropdown option |
| `file_upload` | Upload files to input |
| `drag_and_drop` | Drag element to location |
| `scroll` | Scroll page up/down |
| `key` | Press keyboard keys |
| `wait` | Wait for specified time |
| `wait_for_element` | Wait for selector |
| `switch_frame` | Switch to iframe |
| `switch_tab` | Switch browser tab |
| `history_back/forward` | Navigate history |
| `reload` | Refresh page |
| `terminate` | End test with verdict |

## Running Tests

```bash
# Install dev dependencies
pip install -r requirements.txt

# Run unit tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

## License

MIT
