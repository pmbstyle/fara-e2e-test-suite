"""Unit tests for test_types module."""
from __future__ import annotations

from datetime import datetime

import pytest

from test_types import ActionTrace, TestCase, TestRunResult, TestSuiteResult


class TestTestCase:
    """Tests for TestCase dataclass."""

    def test_default_values(self):
        case = TestCase(
            id="test-1",
            objective="Test objective",
            pass_criteria=["Pass"],
            fail_criteria=["Fail"],
        )
        assert case.tags == set()
        assert case.skip is False
        assert case.retry_count == 0
        assert case.priority == 5

    def test_has_tag(self):
        case = TestCase(
            id="test-1",
            objective="Test",
            pass_criteria=["Pass"],
            fail_criteria=["Fail"],
            tags={"smoke", "auth"},
        )
        assert case.has_tag("smoke")
        assert case.has_tag("SMOKE")  # Case insensitive
        assert not case.has_tag("unknown")

    def test_has_any_tag(self):
        case = TestCase(
            id="test-1",
            objective="Test",
            pass_criteria=["Pass"],
            fail_criteria=["Fail"],
            tags={"smoke", "auth"},
        )
        assert case.has_any_tag({"smoke", "other"})
        assert not case.has_any_tag({"unknown", "other"})

    def test_matches_filter_include(self):
        case = TestCase(
            id="test-1",
            objective="Test",
            pass_criteria=["Pass"],
            fail_criteria=["Fail"],
            tags={"smoke"},
        )
        assert case.matches_filter(include_tags={"smoke"})
        assert not case.matches_filter(include_tags={"regression"})
        assert case.matches_filter()  # No filter = match

    def test_matches_filter_exclude(self):
        case = TestCase(
            id="test-1",
            objective="Test",
            pass_criteria=["Pass"],
            fail_criteria=["Fail"],
            tags={"smoke"},
        )
        assert not case.matches_filter(exclude_tags={"smoke"})
        assert case.matches_filter(exclude_tags={"regression"})

    def test_matches_filter_combined(self):
        case = TestCase(
            id="test-1",
            objective="Test",
            pass_criteria=["Pass"],
            fail_criteria=["Fail"],
            tags={"smoke", "fast"},
        )
        # Include smoke, exclude slow
        assert case.matches_filter(include_tags={"smoke"}, exclude_tags={"slow"})
        # Include smoke, but exclude fast (which it has)
        assert not case.matches_filter(include_tags={"smoke"}, exclude_tags={"fast"})


class TestActionTrace:
    """Tests for ActionTrace dataclass."""

    def test_creation(self):
        trace = ActionTrace(
            round_index=1,
            action="left_click",
            arguments={"coordinate": [100, 200]},
            model_response="<tool_call>...</tool_call>",
            result="Clicked at (100, 200)",
            page_url="https://example.com",
        )
        assert trace.round_index == 1
        assert trace.action == "left_click"
        assert trace.screenshot_path is None
        assert trace.timestamp is None

    def test_with_optional_fields(self):
        trace = ActionTrace(
            round_index=1,
            action="type",
            arguments={"text": "hello"},
            model_response="...",
            result="Typed 'hello'",
            page_url="https://example.com",
            timestamp=datetime.now(),
            duration_ms=150.5,
            element_info={"tag": "input"},
            console_errors=["Error 1"],
        )
        assert trace.timestamp is not None
        assert trace.duration_ms == 150.5
        assert trace.element_info == {"tag": "input"}
        assert trace.console_errors == ["Error 1"]


class TestTestRunResult:
    """Tests for TestRunResult dataclass."""

    def test_duration_seconds(self):
        result = TestRunResult(
            case=TestCase(
                id="test",
                objective="Test",
                pass_criteria=["Pass"],
                fail_criteria=["Fail"],
            ),
            success=True,
            started_at=datetime(2024, 1, 1, 10, 0, 0),
            finished_at=datetime(2024, 1, 1, 10, 0, 30),
            reason="Done",
            actions=[],
            facts=[],
        )
        assert result.duration_seconds == 30.0

    def test_action_count(self):
        result = TestRunResult(
            case=TestCase(
                id="test",
                objective="Test",
                pass_criteria=["Pass"],
                fail_criteria=["Fail"],
            ),
            success=True,
            started_at=datetime.now(),
            finished_at=datetime.now(),
            reason="Done",
            actions=[
                ActionTrace(
                    round_index=1,
                    action="click",
                    arguments={},
                    model_response="",
                    result="",
                    page_url="",
                ),
                ActionTrace(
                    round_index=2,
                    action="type",
                    arguments={},
                    model_response="",
                    result="",
                    page_url="",
                ),
            ],
            facts=[],
        )
        assert result.action_count == 2

    def test_status_property(self):
        result = TestRunResult(
            case=TestCase(
                id="test",
                objective="Test",
                pass_criteria=["Pass"],
                fail_criteria=["Fail"],
            ),
            success=True,
            started_at=datetime.now(),
            finished_at=datetime.now(),
            reason="Done",
            actions=[],
            facts=[],
        )
        assert result.status == "passed"
        
        result.success = False
        assert result.status == "failed"


class TestTestSuiteResult:
    """Tests for TestSuiteResult dataclass."""

    def test_statistics(self):
        case = TestCase(
            id="test",
            objective="Test",
            pass_criteria=["Pass"],
            fail_criteria=["Fail"],
        )
        
        results = [
            TestRunResult(
                case=case,
                success=True,
                started_at=datetime(2024, 1, 1, 10, 0, 0),
                finished_at=datetime(2024, 1, 1, 10, 0, 10),
                reason="Pass",
                actions=[],
                facts=[],
            ),
            TestRunResult(
                case=case,
                success=True,
                started_at=datetime(2024, 1, 1, 10, 0, 10),
                finished_at=datetime(2024, 1, 1, 10, 0, 20),
                reason="Pass",
                actions=[],
                facts=[],
            ),
            TestRunResult(
                case=case,
                success=False,
                started_at=datetime(2024, 1, 1, 10, 0, 20),
                finished_at=datetime(2024, 1, 1, 10, 0, 30),
                reason="Fail",
                actions=[],
                facts=[],
            ),
        ]
        
        suite = TestSuiteResult(
            results=results,
            started_at=datetime(2024, 1, 1, 10, 0, 0),
            finished_at=datetime(2024, 1, 1, 10, 0, 30),
        )
        
        assert suite.total == 3
        assert suite.passed == 2
        assert suite.failed == 1
        assert suite.pass_rate == pytest.approx(66.67, rel=0.1)
        assert suite.duration_seconds == 30.0

    def test_failed_tests_property(self):
        case = TestCase(
            id="test",
            objective="Test",
            pass_criteria=["Pass"],
            fail_criteria=["Fail"],
        )
        
        passed = TestRunResult(
            case=case,
            success=True,
            started_at=datetime.now(),
            finished_at=datetime.now(),
            reason="Pass",
            actions=[],
            facts=[],
        )
        failed = TestRunResult(
            case=case,
            success=False,
            started_at=datetime.now(),
            finished_at=datetime.now(),
            reason="Fail",
            actions=[],
            facts=[],
        )
        
        suite = TestSuiteResult(
            results=[passed, failed],
            started_at=datetime.now(),
            finished_at=datetime.now(),
        )
        
        assert len(suite.failed_tests) == 1
        assert suite.failed_tests[0].success is False
        assert len(suite.passed_tests) == 1

