"""Unit tests for reporters module."""
from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree

import pytest

from reporters import HTMLReporter, JSONReporter, JUnitReporter
from test_types import TestRunResult


class TestJSONReporter:
    """Tests for JSON reporter."""

    def test_generates_valid_json(self, temp_dir: Path, sample_test_result: TestRunResult):
        reporter = JSONReporter()
        report_path = reporter.generate(sample_test_result, temp_dir)
        
        assert report_path.exists()
        assert report_path.suffix == ".json"
        
        data = json.loads(report_path.read_text())
        assert "tests" in data
        assert "summary" in data
        assert len(data["tests"]) == 1

    def test_json_structure(self, temp_dir: Path, sample_test_result: TestRunResult):
        reporter = JSONReporter()
        report_path = reporter.generate(sample_test_result, temp_dir)
        data = json.loads(report_path.read_text())
        
        test = data["tests"][0]
        assert "test_case" in test
        assert "result" in test
        assert "actions" in test
        
        assert test["test_case"]["id"] == "test-login"
        assert test["result"]["success"] is True
        assert len(test["actions"]) == 4

    def test_credentials_masked(self, temp_dir: Path, sample_test_result: TestRunResult):
        reporter = JSONReporter()
        report_path = reporter.generate(sample_test_result, temp_dir)
        data = json.loads(report_path.read_text())
        
        creds = data["tests"][0]["test_case"]["credentials"]
        assert all(v == "***" for v in creds.values())

    def test_suite_report(self, temp_dir: Path, sample_test_result: TestRunResult):
        reporter = JSONReporter()
        report_path = reporter.generate_suite([sample_test_result, sample_test_result], temp_dir)
        data = json.loads(report_path.read_text())
        
        assert data["summary"]["total"] == 2
        assert data["summary"]["passed"] == 2
        assert data["summary"]["failed"] == 0
        assert data["summary"]["pass_rate"] == 100.0


class TestJUnitReporter:
    """Tests for JUnit XML reporter."""

    def test_generates_valid_xml(self, temp_dir: Path, sample_test_result: TestRunResult):
        reporter = JUnitReporter()
        report_path = reporter.generate(sample_test_result, temp_dir)
        
        assert report_path.exists()
        assert report_path.suffix == ".xml"
        
        # Should be valid XML
        tree = ElementTree.parse(report_path)
        root = tree.getroot()
        assert root.tag == "testsuite"

    def test_xml_attributes(self, temp_dir: Path, sample_test_result: TestRunResult):
        reporter = JUnitReporter()
        report_path = reporter.generate(sample_test_result, temp_dir)
        
        tree = ElementTree.parse(report_path)
        root = tree.getroot()
        
        assert root.get("tests") == "1"
        assert root.get("failures") == "0"
        assert root.get("errors") == "0"

    def test_failing_test_has_failure_element(self, temp_dir: Path, sample_test_result: TestRunResult):
        sample_test_result.success = False
        sample_test_result.reason = "Test failed"
        
        reporter = JUnitReporter()
        report_path = reporter.generate(sample_test_result, temp_dir)
        
        tree = ElementTree.parse(report_path)
        root = tree.getroot()
        
        assert root.get("failures") == "1"
        failure = root.find(".//failure")
        assert failure is not None
        assert "Test failed" in failure.get("message", "")

    def test_suite_statistics(self, temp_dir: Path, sample_test_result: TestRunResult):
        results = [sample_test_result]
        
        # Add a failing result
        failing = TestRunResult(
            case=sample_test_result.case,
            success=False,
            started_at=sample_test_result.started_at,
            finished_at=sample_test_result.finished_at,
            reason="Failed test",
            actions=[],
            facts=[],
        )
        results.append(failing)
        
        reporter = JUnitReporter()
        report_path = reporter.generate_suite(results, temp_dir)
        
        tree = ElementTree.parse(report_path)
        root = tree.getroot()
        
        assert root.get("tests") == "2"
        assert root.get("failures") == "1"


class TestHTMLReporter:
    """Tests for HTML reporter."""

    def test_generates_html_file(self, temp_dir: Path, sample_test_result: TestRunResult):
        reporter = HTMLReporter()
        report_path = reporter.generate(sample_test_result, temp_dir)
        
        assert report_path.exists()
        assert report_path.suffix == ".html"
        
        content = report_path.read_text()
        assert "<!DOCTYPE html>" in content
        assert sample_test_result.case.id in content

    def test_contains_test_info(self, temp_dir: Path, sample_test_result: TestRunResult):
        reporter = HTMLReporter()
        report_path = reporter.generate(sample_test_result, temp_dir)
        content = report_path.read_text()
        
        assert sample_test_result.case.objective in content
        assert "PASS" in content  # Verdict badge
        assert "Pass Criteria" in content

    def test_contains_actions(self, temp_dir: Path, sample_test_result: TestRunResult):
        reporter = HTMLReporter()
        report_path = reporter.generate(sample_test_result, temp_dir)
        content = report_path.read_text()
        
        # Check for action types in the report
        assert "type" in content.lower()
        assert "left_click" in content or "click" in content.lower()
        assert "terminate" in content.lower()

    def test_embed_screenshots_option(self, temp_dir: Path, sample_test_result: TestRunResult):
        reporter_embed = HTMLReporter(embed_screenshots=True)
        reporter_link = HTMLReporter(embed_screenshots=False)
        
        # Both should generate successfully
        path1 = reporter_embed.generate(sample_test_result, temp_dir / "embed")
        path2 = reporter_link.generate(sample_test_result, temp_dir / "link")
        
        assert path1.exists()
        assert path2.exists()

    def test_suite_report(self, temp_dir: Path, sample_test_result: TestRunResult):
        reporter = HTMLReporter()
        report_path = reporter.generate_suite([sample_test_result], temp_dir)
        content = report_path.read_text()
        
        assert "Test Suite Report" in content
        assert "Total Tests" in content
        assert "Passed" in content

