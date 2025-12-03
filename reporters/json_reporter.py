"""JSON report generator for E2E test runs."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from reporters.base import BaseReporter, ReportFormat
from test_types import ActionTrace, TestRunResult


class JSONReporter(BaseReporter):
    """Generate machine-readable JSON reports."""

    @property
    def format(self) -> ReportFormat:
        return ReportFormat.JSON

    def _action_to_dict(self, action: ActionTrace) -> Dict[str, Any]:
        """Convert ActionTrace to JSON-serializable dict."""
        return {
            "round": action.round_index,
            "action": action.action,
            "arguments": action.arguments,
            "result": action.result,
            "url": action.page_url,
            "model_response": action.model_response,
            "screenshot": str(action.screenshot_path) if action.screenshot_path else None,
        }

    def _result_to_dict(self, result: TestRunResult) -> Dict[str, Any]:
        """Convert TestRunResult to JSON-serializable dict."""
        return {
            "test_case": {
                "id": result.case.id,
                "objective": result.case.objective,
                "start_url": result.case.start_url,
                "pass_criteria": result.case.pass_criteria,
                "fail_criteria": result.case.fail_criteria,
                "credentials": {k: "***" for k in result.case.credentials},  # Mask credentials
                "notes": result.case.notes,
                "max_rounds": result.case.max_rounds,
            },
            "result": {
                "success": result.success,
                "reason": result.reason,
                "started_at": result.started_at.isoformat(),
                "finished_at": result.finished_at.isoformat(),
                "duration_seconds": result.duration_seconds,
                "total_actions": len(result.actions),
                "facts": result.facts,
            },
            "actions": [self._action_to_dict(a) for a in result.actions],
        }

    def generate(self, result: TestRunResult, output_dir: Path) -> Path:
        """Generate JSON report for a single test result."""
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        filename = f"{result.case.id}-{timestamp}.json"
        target = output_dir / filename

        report_data = {
            "generated_at": datetime.utcnow().isoformat(),
            "report_version": "1.0",
            "tests": [self._result_to_dict(result)],
            "summary": {
                "total": 1,
                "passed": 1 if result.success else 0,
                "failed": 0 if result.success else 1,
                "pass_rate": 100.0 if result.success else 0.0,
            },
        }

        target.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
        return target

    def generate_suite(self, results: List[TestRunResult], output_dir: Path) -> Path:
        """Generate combined JSON report for multiple test results."""
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        filename = f"suite-{timestamp}.json"
        target = output_dir / filename

        passed = sum(1 for r in results if r.success)
        failed = len(results) - passed
        pass_rate = (passed / len(results) * 100) if results else 0.0

        # Calculate timing stats
        durations = [r.duration_seconds for r in results]
        total_duration = sum(durations)
        avg_duration = total_duration / len(durations) if durations else 0
        min_duration = min(durations) if durations else 0
        max_duration = max(durations) if durations else 0

        report_data = {
            "generated_at": datetime.utcnow().isoformat(),
            "report_version": "1.0",
            "tests": [self._result_to_dict(r) for r in results],
            "summary": {
                "total": len(results),
                "passed": passed,
                "failed": failed,
                "pass_rate": round(pass_rate, 2),
                "total_duration_seconds": round(total_duration, 2),
                "avg_duration_seconds": round(avg_duration, 2),
                "min_duration_seconds": round(min_duration, 2),
                "max_duration_seconds": round(max_duration, 2),
            },
            "failed_tests": [
                {"id": r.case.id, "reason": r.reason}
                for r in results if not r.success
            ],
        }

        target.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
        return target

