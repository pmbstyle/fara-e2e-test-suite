"""Base reporter interface for E2E test runs."""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import List

from test_types import TestRunResult


class ReportFormat(str, Enum):
    """Supported report formats."""
    HTML = "html"
    JSON = "json"
    JUNIT = "junit"
    ALL = "all"


class BaseReporter(ABC):
    """Abstract base class for report generators."""

    @abstractmethod
    def generate(self, result: TestRunResult, output_dir: Path) -> Path:
        """
        Generate a report for a single test result.
        
        Args:
            result: Test execution result
            output_dir: Directory to write report to
            
        Returns:
            Path to the generated report file
        """
        pass

    @abstractmethod
    def generate_suite(self, results: List[TestRunResult], output_dir: Path) -> Path:
        """
        Generate a combined report for multiple test results.
        
        Args:
            results: List of test execution results
            output_dir: Directory to write report to
            
        Returns:
            Path to the generated report file
        """
        pass

    @property
    @abstractmethod
    def format(self) -> ReportFormat:
        """Return the report format this reporter generates."""
        pass

