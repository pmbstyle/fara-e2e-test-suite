"""JUnit XML report generator for CI integration."""
from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import List

from reporters.base import BaseReporter, ReportFormat
from test_types import TestRunResult


class JUnitReporter(BaseReporter):
    """Generate JUnit XML reports for CI/CD integration."""

    @property
    def format(self) -> ReportFormat:
        return ReportFormat.JUNIT

    def _escape_xml(self, text: str) -> str:
        """Escape special XML characters."""
        return html.escape(str(text), quote=True)

    def _format_timestamp(self, dt: datetime) -> str:
        """Format datetime for JUnit XML."""
        return dt.strftime("%Y-%m-%dT%H:%M:%S")

    def _build_testcase_xml(self, result: TestRunResult) -> str:
        """Build XML for a single test case."""
        lines = []
        
        classname = "fara.e2e"
        name = self._escape_xml(result.case.id)
        time_sec = f"{result.duration_seconds:.3f}"
        
        if result.success:
            lines.append(
                f'    <testcase classname="{classname}" name="{name}" time="{time_sec}">'
            )
            # Add system-out with action summary
            if result.actions:
                lines.append("      <system-out><![CDATA[")
                lines.append(f"Objective: {result.case.objective}")
                lines.append(f"URL: {result.case.start_url or 'N/A'}")
                lines.append(f"Actions: {len(result.actions)}")
                lines.append("")
                for action in result.actions[-5:]:  # Last 5 actions
                    lines.append(f"  [{action.round_index}] {action.action}: {action.result[:100]}")
                lines.append("]]></system-out>")
            lines.append("    </testcase>")
        else:
            lines.append(
                f'    <testcase classname="{classname}" name="{name}" time="{time_sec}">'
            )
            
            # Add failure element
            failure_msg = self._escape_xml(result.reason)
            failure_type = "AssertionError" if "criteria" in result.reason.lower() else "TestFailure"
            
            lines.append(f'      <failure message="{failure_msg}" type="{failure_type}"><![CDATA[')
            lines.append(f"Test Case: {result.case.id}")
            lines.append(f"Objective: {result.case.objective}")
            lines.append(f"Failure Reason: {result.reason}")
            lines.append("")
            lines.append("Pass Criteria:")
            for criterion in result.case.pass_criteria:
                lines.append(f"  - {criterion}")
            lines.append("")
            lines.append("Fail Criteria:")
            for criterion in result.case.fail_criteria:
                lines.append(f"  - {criterion}")
            lines.append("")
            lines.append(f"Total Actions: {len(result.actions)}")
            if result.actions:
                lines.append("")
                lines.append("Last Actions:")
                for action in result.actions[-5:]:
                    lines.append(f"  [{action.round_index}] {action.action} @ {action.page_url}")
                    lines.append(f"      Result: {action.result[:150]}")
            lines.append("]]></failure>")
            
            # Add system-err with model responses for failed tests
            lines.append("      <system-err><![CDATA[")
            lines.append("Model Responses (last 3):")
            for action in result.actions[-3:]:
                lines.append(f"\n--- Round {action.round_index} ---")
                lines.append(action.model_response[:500])
            lines.append("]]></system-err>")
            
            lines.append("    </testcase>")
        
        return "\n".join(lines)

    def generate(self, result: TestRunResult, output_dir: Path) -> Path:
        """Generate JUnit XML report for a single test result."""
        return self.generate_suite([result], output_dir)

    def generate_suite(self, results: List[TestRunResult], output_dir: Path) -> Path:
        """Generate combined JUnit XML report for multiple test results."""
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        filename = f"junit-{timestamp}.xml"
        target = output_dir / filename

        # Calculate statistics
        tests = len(results)
        failures = sum(1 for r in results if not r.success)
        total_time = sum(r.duration_seconds for r in results)
        
        # Get earliest start and latest end
        if results:
            earliest = min(r.started_at for r in results)
            timestamp_str = self._format_timestamp(earliest)
        else:
            timestamp_str = self._format_timestamp(datetime.utcnow())

        # Build XML
        lines = ['<?xml version="1.0" encoding="UTF-8"?>']
        lines.append(
            f'<testsuite name="Fara E2E Tests" '
            f'tests="{tests}" '
            f'failures="{failures}" '
            f'errors="0" '
            f'skipped="0" '
            f'time="{total_time:.3f}" '
            f'timestamp="{timestamp_str}">'
        )
        
        # Add properties
        lines.append("  <properties>")
        lines.append('    <property name="reporter" value="fara-e2e-junit"/>')
        lines.append(f'    <property name="generated_at" value="{datetime.utcnow().isoformat()}"/>')
        lines.append("  </properties>")
        
        # Add test cases
        for result in results:
            lines.append(self._build_testcase_xml(result))
        
        lines.append("</testsuite>")

        target.write_text("\n".join(lines), encoding="utf-8")
        return target

