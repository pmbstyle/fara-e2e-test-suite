"""Typed objects for natural-language E2E tests."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


@dataclass
class TestCase:
    """Single natural-language test description."""

    id: str
    objective: str
    objective_steps: List[str]
    pass_criteria: List[str]
    fail_criteria: List[str]
    start_url: Optional[str] = None
    credentials: Dict[str, str] = field(default_factory=dict)
    notes: Optional[str] = None
    max_rounds: Optional[int] = None
    
    # New fields for tagging and execution control
    tags: Set[str] = field(default_factory=set)
    skip: bool = False
    skip_reason: Optional[str] = None
    retry_count: int = 0
    timeout_seconds: Optional[float] = None
    
    # Test metadata
    priority: int = field(default=5)  # 1 = highest, 10 = lowest
    owner: Optional[str] = None
    
    def has_tag(self, tag: str) -> bool:
        """Check if test has a specific tag."""
        return tag.lower() in {t.lower() for t in self.tags}
    
    def has_any_tag(self, tags: Set[str]) -> bool:
        """Check if test has any of the specified tags."""
        lower_tags = {t.lower() for t in tags}
        return bool(lower_tags & {t.lower() for t in self.tags})
    
    def matches_filter(
        self,
        include_tags: Optional[Set[str]] = None,
        exclude_tags: Optional[Set[str]] = None,
    ) -> bool:
        """Check if test matches tag filters."""
        if include_tags and not self.has_any_tag(include_tags):
            return False
        if exclude_tags and self.has_any_tag(exclude_tags):
            return False
        return True


@dataclass
class ActionTrace:
    """One model decision plus the screenshot captured afterwards."""

    round_index: int
    action: str
    arguments: Dict[str, Any]
    model_response: str
    result: str
    page_url: str
    screenshot_path: Optional[Path] = None
    timestamp: Optional[datetime] = None
    duration_ms: Optional[float] = None
    
    # Element info captured at action time
    element_info: Optional[Dict[str, Any]] = None
    console_errors: Optional[List[str]] = None


@dataclass
class TestRunResult:
    """Outcome of a test case execution."""

    case: TestCase
    success: bool
    started_at: datetime
    finished_at: datetime
    reason: str
    actions: List[ActionTrace] = field(default_factory=list)
    facts: List[str] = field(default_factory=list)
    
    # Additional metadata
    retry_attempt: int = 0
    browser_type: Optional[str] = None
    final_url: Optional[str] = None
    console_errors: List[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        return max(0.0, (self.finished_at - self.started_at).total_seconds())
    
    @property
    def action_count(self) -> int:
        return len(self.actions)
    
    @property
    def status(self) -> str:
        return "passed" if self.success else "failed"


@dataclass 
class TestSuiteResult:
    """Aggregated results for a test suite run."""
    
    results: List[TestRunResult]
    started_at: datetime
    finished_at: datetime
    
    @property
    def total(self) -> int:
        return len(self.results)
    
    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.success)
    
    @property
    def failed(self) -> int:
        return self.total - self.passed
    
    @property
    def pass_rate(self) -> float:
        return (self.passed / self.total * 100) if self.total else 0.0
    
    @property
    def duration_seconds(self) -> float:
        return max(0.0, (self.finished_at - self.started_at).total_seconds())
    
    @property
    def failed_tests(self) -> List[TestRunResult]:
        return [r for r in self.results if not r.success]
    
    @property
    def passed_tests(self) -> List[TestRunResult]:
        return [r for r in self.results if r.success]
