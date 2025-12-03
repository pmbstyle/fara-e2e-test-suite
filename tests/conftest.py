"""Pytest fixtures for Fara E2E tests."""
from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

from test_types import ActionTrace, TestCase, TestRunResult


@pytest.fixture
def sample_test_case() -> TestCase:
    """Create a sample test case for testing."""
    return TestCase(
        id="test-login",
        objective="Log in to the application",
        pass_criteria=[
            "User is logged in successfully",
            "Dashboard is visible",
        ],
        fail_criteria=[
            "Login error is displayed",
            "Still on login page after submit",
        ],
        start_url="https://example.com/login",
        credentials={
            "email": "test@example.com",
            "password": "testpass123",
        },
        notes="Test login flow",
        max_rounds=10,
        tags={"smoke", "auth"},
        priority=1,
    )


@pytest.fixture
def sample_test_result(sample_test_case: TestCase) -> TestRunResult:
    """Create a sample test result for testing."""
    return TestRunResult(
        case=sample_test_case,
        success=True,
        started_at=datetime(2024, 1, 1, 10, 0, 0),
        finished_at=datetime(2024, 1, 1, 10, 0, 30),
        reason="Login successful",
        actions=[
            ActionTrace(
                round_index=1,
                action="type",
                arguments={"coordinate": [100, 200], "text": "test@example.com"},
                model_response="<tool_call>...</tool_call>",
                result="I typed 'test@example.com'.",
                page_url="https://example.com/login",
                timestamp=datetime(2024, 1, 1, 10, 0, 5),
            ),
            ActionTrace(
                round_index=2,
                action="type",
                arguments={"coordinate": [100, 250], "text": "testpass123"},
                model_response="<tool_call>...</tool_call>",
                result="I typed 'testpass123'.",
                page_url="https://example.com/login",
                timestamp=datetime(2024, 1, 1, 10, 0, 10),
            ),
            ActionTrace(
                round_index=3,
                action="left_click",
                arguments={"coordinate": [150, 300]},
                model_response="<tool_call>...</tool_call>",
                result="I clicked at coordinates (150.0, 300.0).",
                page_url="https://example.com/dashboard",
                timestamp=datetime(2024, 1, 1, 10, 0, 15),
            ),
            ActionTrace(
                round_index=4,
                action="terminate",
                arguments={"status": "success", "reason": "Dashboard visible"},
                model_response="<tool_call>...</tool_call>",
                result="terminate",
                page_url="https://example.com/dashboard",
                timestamp=datetime(2024, 1, 1, 10, 0, 20),
            ),
        ],
        facts=["User logged in as test@example.com"],
        browser_type="firefox",
        final_url="https://example.com/dashboard",
    )


@pytest.fixture
def temp_dir() -> Path:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_task_yaml() -> str:
    """Sample YAML task definition."""
    return """
id: signup-test
objective: Complete the signup flow
pass_criteria:
  - User is registered successfully
  - Dashboard is visible after signup
fail_criteria:
  - Registration error is displayed
  - Still on signup page
credentials:
  name: TestUser
  email: test@example.com
  password: "Test123!"
start_url: https://example.com/signup
tags:
  - smoke
  - signup
priority: 1
max_rounds: 15
"""


@pytest.fixture
def sample_task_json() -> Dict[str, Any]:
    """Sample JSON task definition."""
    return {
        "id": "login-test",
        "objective": "Log in with valid credentials",
        "pass_criteria": ["User logged in", "Dashboard visible"],
        "fail_criteria": ["Login error", "Still on login page"],
        "credentials": {"email": "user@example.com", "password": "pass123"},
        "start_url": "https://example.com/login",
        "tags": ["auth", "p0"],
        "retry_count": 2,
    }


@pytest.fixture
def mock_browser() -> MagicMock:
    """Create a mock browser for testing."""
    browser = MagicMock()
    browser.start = AsyncMock()
    browser.close = AsyncMock()
    browser.goto = AsyncMock()
    browser.screenshot = AsyncMock(return_value=b"fake_screenshot_data")
    browser.click = AsyncMock(return_value={"found": True, "tag": "button"})
    browser.type_text = AsyncMock()
    browser.get_url = MagicMock(return_value="https://example.com")
    browser.get_title = AsyncMock(return_value="Example Page")
    browser.get_body_text = AsyncMock(return_value="Page content here")
    browser.get_element_at = AsyncMock(return_value={"found": True, "tag": "div"})
    browser.wait_for_load_state = AsyncMock()
    browser.get_scroll_position = AsyncMock(return_value={"y": 0, "scrollHeight": 1000})
    browser.get_console_messages = MagicMock(return_value=[])
    browser.update_overlay = AsyncMock()
    browser.show_click_marker = AsyncMock()
    return browser


@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Create a mock LLM client for testing."""
    client = MagicMock()
    
    async def mock_create(**kwargs):
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = """
I will click the login button.
<tool_call>
{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [150, 300]}}
</tool_call>
"""
        return response
    
    client.chat.completions.create = mock_create
    return client

