"""Custom exception hierarchy for Fara E2E agent."""
from __future__ import annotations

from typing import Any, Optional


class FaraError(Exception):
    """Base exception for all Fara-related errors."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


# Browser-related exceptions
class BrowserError(FaraError):
    """Base exception for browser automation errors."""

    pass


class NavigationError(BrowserError):
    """Raised when page navigation fails."""

    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        details = {}
        if url:
            details["url"] = url
        if timeout:
            details["timeout"] = timeout
        super().__init__(message, details)
        self.url = url
        self.timeout = timeout


class ElementNotFoundError(BrowserError):
    """Raised when an element cannot be found at coordinates or by selector."""

    def __init__(
        self,
        message: str,
        coordinates: Optional[tuple[float, float]] = None,
        selector: Optional[str] = None,
    ):
        details = {}
        if coordinates:
            details["coordinates"] = coordinates
        if selector:
            details["selector"] = selector
        super().__init__(message, details)
        self.coordinates = coordinates
        self.selector = selector


class ElementNotInteractableError(BrowserError):
    """Raised when an element exists but cannot be interacted with."""

    def __init__(
        self,
        message: str,
        coordinates: Optional[tuple[float, float]] = None,
        reason: Optional[str] = None,
    ):
        details = {}
        if coordinates:
            details["coordinates"] = coordinates
        if reason:
            details["reason"] = reason
        super().__init__(message, details)
        self.coordinates = coordinates
        self.reason = reason


class BrowserNotStartedError(BrowserError):
    """Raised when attempting to use browser before starting."""

    def __init__(self):
        super().__init__("Browser has not been started. Call start() first.")


class ScreenshotError(BrowserError):
    """Raised when screenshot capture fails."""

    pass


# LLM-related exceptions
class LLMError(FaraError):
    """Base exception for LLM/model-related errors."""

    pass


class LLMConnectionError(LLMError):
    """Raised when unable to connect to the LLM service."""

    def __init__(self, message: str, base_url: Optional[str] = None):
        details = {"base_url": base_url} if base_url else {}
        super().__init__(message, details)
        self.base_url = base_url


class LLMResponseError(LLMError):
    """Raised when LLM returns an invalid or unparseable response."""

    def __init__(self, message: str, response: Optional[str] = None):
        details = {"response_preview": response[:200] if response else None}
        super().__init__(message, details)
        self.response = response


class ActionParseError(LLMError):
    """Raised when unable to parse action from model response."""

    def __init__(self, message: str, raw_response: Optional[str] = None):
        details = {"raw_response": raw_response[:500] if raw_response else None}
        super().__init__(message, details)
        self.raw_response = raw_response


class ModelTimeoutError(LLMError):
    """Raised when model call times out."""

    def __init__(self, timeout: float):
        super().__init__(f"Model call timed out after {timeout}s", {"timeout": timeout})
        self.timeout = timeout


# Test definition exceptions
class TestDefinitionError(FaraError):
    """Base exception for test definition/loading errors."""

    pass


class TaskLoadError(TestDefinitionError):
    """Raised when a task file cannot be loaded or parsed."""

    def __init__(self, message: str, file_path: Optional[str] = None):
        details = {"file_path": file_path} if file_path else {}
        super().__init__(message, details)
        self.file_path = file_path


class TaskValidationError(TestDefinitionError):
    """Raised when a task definition is invalid."""

    def __init__(self, message: str, task_id: Optional[str] = None, field: Optional[str] = None):
        details = {}
        if task_id:
            details["task_id"] = task_id
        if field:
            details["field"] = field
        super().__init__(message, details)
        self.task_id = task_id
        self.field = field


# Test execution exceptions
class TestExecutionError(FaraError):
    """Base exception for test execution errors."""

    pass


class MaxRoundsExceededError(TestExecutionError):
    """Raised when test exceeds maximum allowed rounds."""

    def __init__(self, max_rounds: int, task_id: Optional[str] = None):
        message = f"Test exceeded maximum rounds ({max_rounds})"
        details = {"max_rounds": max_rounds}
        if task_id:
            details["task_id"] = task_id
        super().__init__(message, details)
        self.max_rounds = max_rounds
        self.task_id = task_id


class LoopDetectedError(TestExecutionError):
    """Raised when agent detects it's stuck in a loop."""

    def __init__(self, message: str, action: Optional[str] = None, count: Optional[int] = None):
        details = {}
        if action:
            details["action"] = action
        if count:
            details["repeat_count"] = count
        super().__init__(message, details)
        self.action = action
        self.count = count


# Configuration exceptions
class ConfigurationError(FaraError):
    """Raised when configuration is invalid."""

    pass


class ConfigFileNotFoundError(ConfigurationError):
    """Raised when a required config file is not found."""

    def __init__(self, file_path: str):
        super().__init__(f"Configuration file not found: {file_path}", {"file_path": file_path})
        self.file_path = file_path

