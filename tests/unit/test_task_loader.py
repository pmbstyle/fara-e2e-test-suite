"""Unit tests for task_loader module."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from exceptions import TaskLoadError, TaskValidationError
from task_loader import (
    _as_list,
    _as_set,
    _parse_task,
    discover_tasks,
    load_task_file,
    validate_task,
)
from test_types import TestCase


class TestAsListFunction:
    """Tests for _as_list helper function."""

    def test_none_returns_empty_list(self):
        assert _as_list(None) == []

    def test_string_returns_single_item_list(self):
        assert _as_list("test") == ["test"]

    def test_list_returns_stringified_items(self):
        assert _as_list(["a", "b", 123]) == ["a", "b", "123"]

    def test_invalid_type_raises_error(self):
        with pytest.raises(TaskLoadError):
            _as_list({"dict": "value"})


class TestAsSetFunction:
    """Tests for _as_set helper function."""

    def test_none_returns_empty_set(self):
        assert _as_set(None) == set()

    def test_string_returns_single_item_set(self):
        assert _as_set("tag1") == {"tag1"}

    def test_list_returns_set(self):
        assert _as_set(["tag1", "tag2"]) == {"tag1", "tag2"}

    def test_invalid_type_raises_error(self):
        with pytest.raises(TaskLoadError):
            _as_set(123)


class TestParseTask:
    """Tests for _parse_task function."""

    def test_parses_minimal_task(self):
        data = {
            "objective": "Test objective",
            "pass_criteria": ["Pass condition"],
            "fail_criteria": ["Fail condition"],
        }
        task = _parse_task(data, "fallback-id")
        assert task.id == "fallback-id"
        assert task.objective == "Test objective"
        assert task.pass_criteria == ["Pass condition"]
        assert task.fail_criteria == ["Fail condition"]

    def test_parses_full_task(self):
        data = {
            "id": "my-task",
            "objective": "Complete signup",
            "pass_criteria": ["User registered", "Dashboard visible"],
            "fail_criteria": ["Error shown"],
            "start_url": "https://example.com",
            "credentials": {"email": "test@test.com"},
            "notes": "Some notes",
            "max_rounds": 20,
            "tags": ["smoke", "signup"],
            "skip": True,
            "skip_reason": "Flaky",
            "retry_count": 3,
            "priority": 2,
            "owner": "team@example.com",
        }
        task = _parse_task(data, "fallback")
        assert task.id == "my-task"
        assert task.start_url == "https://example.com"
        assert task.credentials == {"email": "test@test.com"}
        assert task.notes == "Some notes"
        assert task.max_rounds == 20
        assert task.tags == {"smoke", "signup"}
        assert task.skip is True
        assert task.skip_reason == "Flaky"
        assert task.retry_count == 3
        assert task.priority == 2
        assert task.owner == "team@example.com"

    def test_missing_objective_raises_error(self):
        data = {
            "pass_criteria": ["Pass"],
            "fail_criteria": ["Fail"],
        }
        with pytest.raises(TaskValidationError) as exc_info:
            _parse_task(data, "test-id")
        assert "objective" in str(exc_info.value)

    def test_missing_pass_criteria_raises_error(self):
        data = {
            "objective": "Test",
            "fail_criteria": ["Fail"],
        }
        with pytest.raises(TaskValidationError) as exc_info:
            _parse_task(data, "test-id")
        assert "pass_criteria" in str(exc_info.value)

    def test_missing_fail_criteria_raises_error(self):
        data = {
            "objective": "Test",
            "pass_criteria": ["Pass"],
        }
        with pytest.raises(TaskValidationError) as exc_info:
            _parse_task(data, "test-id")
        assert "fail_criteria" in str(exc_info.value)

    def test_accepts_task_as_objective_alias(self):
        data = {
            "task": "Alternative objective",
            "pass_criteria": ["Pass"],
            "fail_criteria": ["Fail"],
        }
        task = _parse_task(data, "test")
        assert task.objective == "Alternative objective"

    def test_priority_clamped_to_valid_range(self):
        data = {
            "objective": "Test",
            "pass_criteria": ["Pass"],
            "fail_criteria": ["Fail"],
            "priority": 0,
        }
        task = _parse_task(data, "test")
        assert task.priority == 1

        data["priority"] = 15
        task = _parse_task(data, "test")
        assert task.priority == 10


class TestLoadTaskFile:
    """Tests for load_task_file function."""

    def test_loads_yaml_file(self, temp_dir: Path, sample_task_yaml: str):
        task_file = temp_dir / "test.yaml"
        task_file.write_text(sample_task_yaml)
        
        task = load_task_file(task_file)
        assert task.id == "signup-test"
        assert task.objective == "Complete the signup flow"
        assert "smoke" in task.tags

    def test_loads_json_file(self, temp_dir: Path, sample_task_json: dict):
        task_file = temp_dir / "test.json"
        task_file.write_text(json.dumps(sample_task_json))
        
        task = load_task_file(task_file)
        assert task.id == "login-test"
        assert task.retry_count == 2

    def test_uses_filename_as_fallback_id(self, temp_dir: Path):
        task_file = temp_dir / "my-custom-task.yaml"
        task_file.write_text("""
objective: Test objective
pass_criteria: [Pass]
fail_criteria: [Fail]
""")
        task = load_task_file(task_file)
        assert task.id == "my-custom-task"


class TestDiscoverTasks:
    """Tests for discover_tasks function."""

    def test_discovers_all_tasks(self, temp_dir: Path):
        # Create multiple task files
        (temp_dir / "task1.yaml").write_text("""
objective: Task 1
pass_criteria: [Pass]
fail_criteria: [Fail]
""")
        (temp_dir / "task2.json").write_text(json.dumps({
            "objective": "Task 2",
            "pass_criteria": ["Pass"],
            "fail_criteria": ["Fail"],
        }))
        
        tasks = discover_tasks(temp_dir)
        assert len(tasks) == 2

    def test_filters_by_id(self, temp_dir: Path):
        (temp_dir / "task1.yaml").write_text("""
id: target-task
objective: Task 1
pass_criteria: [Pass]
fail_criteria: [Fail]
""")
        (temp_dir / "task2.yaml").write_text("""
id: other-task
objective: Task 2
pass_criteria: [Pass]
fail_criteria: [Fail]
""")
        
        tasks = discover_tasks(temp_dir, only_ids=["target-task"])
        assert len(tasks) == 1
        assert tasks[0].id == "target-task"

    def test_filters_by_include_tags(self, temp_dir: Path):
        (temp_dir / "task1.yaml").write_text("""
id: smoke-test
objective: Smoke test
pass_criteria: [Pass]
fail_criteria: [Fail]
tags: [smoke, p0]
""")
        (temp_dir / "task2.yaml").write_text("""
id: regression-test
objective: Regression test
pass_criteria: [Pass]
fail_criteria: [Fail]
tags: [regression]
""")
        
        tasks = discover_tasks(temp_dir, include_tags={"smoke"})
        assert len(tasks) == 1
        assert tasks[0].id == "smoke-test"

    def test_filters_by_exclude_tags(self, temp_dir: Path):
        (temp_dir / "task1.yaml").write_text("""
id: fast-test
objective: Fast test
pass_criteria: [Pass]
fail_criteria: [Fail]
tags: [fast]
""")
        (temp_dir / "task2.yaml").write_text("""
id: slow-test
objective: Slow test
pass_criteria: [Pass]
fail_criteria: [Fail]
tags: [slow]
""")
        
        tasks = discover_tasks(temp_dir, exclude_tags={"slow"})
        assert len(tasks) == 1
        assert tasks[0].id == "fast-test"

    def test_excludes_skipped_by_default(self, temp_dir: Path):
        (temp_dir / "task1.yaml").write_text("""
id: active-test
objective: Active test
pass_criteria: [Pass]
fail_criteria: [Fail]
""")
        (temp_dir / "task2.yaml").write_text("""
id: skipped-test
objective: Skipped test
pass_criteria: [Pass]
fail_criteria: [Fail]
skip: true
skip_reason: Temporarily disabled
""")
        
        tasks = discover_tasks(temp_dir)
        assert len(tasks) == 1
        assert tasks[0].id == "active-test"

    def test_includes_skipped_when_requested(self, temp_dir: Path):
        (temp_dir / "task1.yaml").write_text("""
id: skipped-test
objective: Skipped test
pass_criteria: [Pass]
fail_criteria: [Fail]
skip: true
""")
        
        tasks = discover_tasks(temp_dir, include_skipped=True)
        assert len(tasks) == 1

    def test_sorts_by_priority(self, temp_dir: Path):
        (temp_dir / "low.yaml").write_text("""
id: low-priority
objective: Low priority
pass_criteria: [Pass]
fail_criteria: [Fail]
priority: 5
""")
        (temp_dir / "high.yaml").write_text("""
id: high-priority
objective: High priority
pass_criteria: [Pass]
fail_criteria: [Fail]
priority: 1
""")
        
        tasks = discover_tasks(temp_dir, sort_by_priority=True)
        assert tasks[0].id == "high-priority"
        assert tasks[1].id == "low-priority"

    def test_raises_error_for_missing_task_id(self, temp_dir: Path):
        (temp_dir / "existing.yaml").write_text("""
id: existing
objective: Existing test
pass_criteria: [Pass]
fail_criteria: [Fail]
""")
        
        with pytest.raises(TaskLoadError) as exc_info:
            discover_tasks(temp_dir, only_ids=["nonexistent"])
        assert "nonexistent" in str(exc_info.value)

    def test_raises_error_for_missing_directory(self, temp_dir: Path):
        with pytest.raises(TaskLoadError):
            discover_tasks(temp_dir / "nonexistent")


class TestValidateTask:
    """Tests for validate_task function."""

    def test_valid_task_returns_empty_list(self):
        data = {
            "objective": "Test",
            "pass_criteria": ["Pass"],
            "fail_criteria": ["Fail"],
        }
        errors = validate_task(data)
        assert errors == []

    def test_missing_objective_returns_error(self):
        data = {
            "pass_criteria": ["Pass"],
            "fail_criteria": ["Fail"],
        }
        errors = validate_task(data)
        assert any("objective" in e for e in errors)

    def test_invalid_max_rounds_returns_error(self):
        data = {
            "objective": "Test",
            "pass_criteria": ["Pass"],
            "fail_criteria": ["Fail"],
            "max_rounds": "invalid",
        }
        errors = validate_task(data)
        assert any("max_rounds" in e for e in errors)

    def test_negative_retry_count_returns_error(self):
        data = {
            "objective": "Test",
            "pass_criteria": ["Pass"],
            "fail_criteria": ["Fail"],
            "retry_count": -1,
        }
        errors = validate_task(data)
        assert any("retry_count" in e for e in errors)


class TestTestCaseTagMethods:
    """Tests for TestCase tag filtering methods."""

    def test_has_tag_case_insensitive(self, sample_test_case: TestCase):
        assert sample_test_case.has_tag("SMOKE")
        assert sample_test_case.has_tag("smoke")

    def test_has_any_tag(self, sample_test_case: TestCase):
        assert sample_test_case.has_any_tag({"smoke", "unknown"})
        assert not sample_test_case.has_any_tag({"unknown1", "unknown2"})

    def test_matches_filter_with_include(self, sample_test_case: TestCase):
        assert sample_test_case.matches_filter(include_tags={"smoke"})
        assert not sample_test_case.matches_filter(include_tags={"unknown"})

    def test_matches_filter_with_exclude(self, sample_test_case: TestCase):
        assert not sample_test_case.matches_filter(exclude_tags={"smoke"})
        assert sample_test_case.matches_filter(exclude_tags={"unknown"})

