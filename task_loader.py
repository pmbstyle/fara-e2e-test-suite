"""Filesystem-backed task loader for natural-language E2E tests."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import yaml

from exceptions import TaskLoadError, TaskValidationError
from test_types import TestCase


def _as_list(value: Any) -> List[str]:
    """Convert value to list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [value]
    raise TaskLoadError(f"Expected string or list, got {type(value).__name__}")


def _as_set(value: Any) -> Set[str]:
    """Convert value to set of strings."""
    if value is None:
        return set()
    if isinstance(value, (list, set, tuple)):
        return {str(item) for item in value}
    if isinstance(value, str):
        return {value}
    raise TaskLoadError(f"Expected string, list, or set, got {type(value).__name__}")


def _parse_task(data: Dict[str, Any], fallback_id: str) -> TestCase:
    """Parse a dictionary into a TestCase."""
    if not isinstance(data, dict):
        raise TaskLoadError("Task payload must be a mapping")

    task_id = str(data.get("id") or fallback_id)
    objective = data.get("objective") or data.get("task") or ""
    
    if not objective:
        raise TaskValidationError(f"Task is missing an 'objective' field", task_id=task_id, field="objective")

    objective_steps = _as_list(data.get("objective_steps") or data.get("steps"))
    pass_criteria = _as_list(data.get("pass_criteria") or data.get("pass"))
    fail_criteria = _as_list(data.get("fail_criteria") or data.get("fail"))

    if not pass_criteria:
        raise TaskValidationError(
            "Task must define at least one pass_criteria item",
            task_id=task_id,
            field="pass_criteria"
        )
    if not fail_criteria:
        raise TaskValidationError(
            "Task must define at least one fail_criteria item",
            task_id=task_id,
            field="fail_criteria"
        )

    credentials = data.get("credentials") or {}
    if credentials and not isinstance(credentials, dict):
        raise TaskValidationError(
            "Credentials must be a mapping",
            task_id=task_id,
            field="credentials"
        )

    # Parse tags
    tags = _as_set(data.get("tags"))
    
    # Parse skip settings
    skip = bool(data.get("skip", False))
    skip_reason = data.get("skip_reason")
    
    # Parse retry count
    retry_count = int(data.get("retry_count", 0))
    if retry_count < 0:
        retry_count = 0
    
    # Parse timeout
    timeout_seconds = data.get("timeout_seconds") or data.get("timeout")
    if timeout_seconds is not None:
        timeout_seconds = float(timeout_seconds)
    
    # Parse priority
    priority = int(data.get("priority", 5))
    if priority < 1:
        priority = 1
    elif priority > 10:
        priority = 10
    
    # Parse owner
    owner = data.get("owner")

    return TestCase(
        id=task_id,
        objective=str(objective),
        objective_steps=objective_steps,
        pass_criteria=pass_criteria,
        fail_criteria=fail_criteria,
        start_url=data.get("start_url"),
        credentials=credentials,
        notes=data.get("notes"),
        max_rounds=data.get("max_rounds"),
        tags=tags,
        skip=skip,
        skip_reason=skip_reason,
        retry_count=retry_count,
        timeout_seconds=timeout_seconds,
        priority=priority,
        owner=owner,
    )


def load_task_file(path: Path) -> TestCase:
    """Load a single task file (YAML or JSON)."""
    try:
        raw = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".yml", ".yaml"}:
            data = yaml.safe_load(raw)
        else:
            data = json.loads(raw)
        return _parse_task(data, fallback_id=path.stem)
    except (TaskLoadError, TaskValidationError):
        raise
    except Exception as exc:
        raise TaskLoadError(f"Failed to load task file: {exc}", file_path=str(path)) from exc


def discover_tasks(
    tasks_dir: Path,
    only_ids: Optional[Iterable[str]] = None,
    include_tags: Optional[Set[str]] = None,
    exclude_tags: Optional[Set[str]] = None,
    include_skipped: bool = False,
    sort_by_priority: bool = False,
) -> List[TestCase]:
    """
    Discover and load tasks from a directory.
    
    Args:
        tasks_dir: Directory containing task YAML/JSON files
        only_ids: If provided, only load tasks with these IDs
        include_tags: If provided, only include tasks with at least one of these tags
        exclude_tags: If provided, exclude tasks with any of these tags
        include_skipped: If True, include tasks marked as skip=true
        sort_by_priority: If True, sort tasks by priority (1=highest first)
    
    Returns:
        List of TestCase objects
    """
    tasks_dir = tasks_dir.expanduser().resolve()
    
    if not tasks_dir.exists():
        raise TaskLoadError(f"Tasks directory does not exist: {tasks_dir}")
    
    id_filter = {tid for tid in (only_ids or [])}
    found: List[TestCase] = []
    
    # Collect YAML and JSON files
    yaml_files = sorted(tasks_dir.glob("*.yaml")) + sorted(tasks_dir.glob("*.yml"))
    json_files = sorted(tasks_dir.glob("*.json"))
    all_files = yaml_files + json_files
    
    for path in all_files:
        task = load_task_file(path)
        
        # Filter by ID if specified
        if id_filter and task.id not in id_filter:
            continue
        
        # Filter by skip status
        if task.skip and not include_skipped:
            continue
        
        # Filter by tags
        if not task.matches_filter(include_tags, exclude_tags):
            continue
        
        found.append(task)
    
    # Check for missing IDs
    if id_filter:
        found_ids = {t.id for t in found}
        missing = id_filter - found_ids
        if missing:
            raise TaskLoadError(f"Tasks not found: {', '.join(sorted(missing))}")
    
    # Sort by priority if requested
    if sort_by_priority:
        found.sort(key=lambda t: t.priority)
    
    return found


def validate_task(data: Dict[str, Any]) -> List[str]:
    """
    Validate task data without loading.
    
    Returns list of validation errors (empty if valid).
    """
    errors = []
    
    if not isinstance(data, dict):
        return ["Task must be a dictionary/mapping"]
    
    if not data.get("objective") and not data.get("task"):
        errors.append("Missing required field: objective")
    
    pass_criteria = data.get("pass_criteria") or data.get("pass")
    if not pass_criteria:
        errors.append("Missing required field: pass_criteria")
    elif not isinstance(pass_criteria, (str, list)):
        errors.append("pass_criteria must be a string or list")
    
    fail_criteria = data.get("fail_criteria") or data.get("fail")
    if not fail_criteria:
        errors.append("Missing required field: fail_criteria")
    elif not isinstance(fail_criteria, (str, list)):
        errors.append("fail_criteria must be a string or list")

    objective_steps = data.get("objective_steps") or data.get("steps")
    if objective_steps is None:
        errors.append("Missing required field: objective_steps (ordered steps the agent should follow)")
    elif not isinstance(objective_steps, (str, list)):
        errors.append("objective_steps must be a string or list")
    
    credentials = data.get("credentials")
    if credentials is not None and not isinstance(credentials, dict):
        errors.append("credentials must be a dictionary")
    
    max_rounds = data.get("max_rounds")
    if max_rounds is not None:
        try:
            val = int(max_rounds)
            if val < 1:
                errors.append("max_rounds must be at least 1")
        except (ValueError, TypeError):
            errors.append("max_rounds must be an integer")
    
    tags = data.get("tags")
    if tags is not None and not isinstance(tags, (str, list, set)):
        errors.append("tags must be a string or list")
    
    retry_count = data.get("retry_count")
    if retry_count is not None:
        try:
            val = int(retry_count)
            if val < 0:
                errors.append("retry_count cannot be negative")
        except (ValueError, TypeError):
            errors.append("retry_count must be an integer")
    
    priority = data.get("priority")
    if priority is not None:
        try:
            val = int(priority)
            if val < 1 or val > 10:
                errors.append("priority must be between 1 and 10")
        except (ValueError, TypeError):
            errors.append("priority must be an integer")
    
    return errors
