"""CLI-friendly orchestrator for running natural-language E2E tests."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Set

from agent import FaraAgent
from config import FaraConfig, load_config
from exceptions import FaraError, TaskLoadError
from reporters import HTMLReporter, JSONReporter, JUnitReporter, ReportFormat
from reporters.html import build_report  # backward compatibility
from task_loader import discover_tasks
from test_types import TestCase, TestRunResult, TestSuiteResult


class E2ETestRunner:
    """High-level runner that manages browser lifecycle per task."""

    def __init__(
        self,
        config: FaraConfig,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config
        self.logger = logger or logging.getLogger("e2e_runner")

    async def run_task(
        self,
        case: TestCase,
        retry_attempt: int = 0,
        trace_path: Optional[Path] = None,
    ) -> TestRunResult:
        """Run a single test case."""
        run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        
        # Build agent config from FaraConfig
        agent_config = {
            "model": self.config.agent.model,
            "base_url": self.config.agent.base_url,
            "api_key": self.config.agent.api_key,
            "temperature": self.config.agent.temperature,
            "max_rounds": case.max_rounds or self.config.agent.max_rounds,
            "max_tokens": self.config.agent.max_tokens,
            "max_n_images": self.config.agent.max_n_images,
            "save_screenshots": self.config.reporting.save_screenshots,
            "screenshots_folder": str(self.config.reporting.screenshots_folder),
            "reports_folder": str(self.config.reporting.reports_folder),
            "downloads_folder": str(self.config.reporting.downloads_folder),
            "show_overlay": self.config.browser.show_overlay,
            "show_click_markers": self.config.browser.show_click_markers,
            "debug_log_requests": self.config.agent.debug_log_requests,
        }
        
        agent = FaraAgent(
            config=agent_config,
            headless=self.config.browser.headless,
            browser_type=self.config.browser.browser,
            logger=self.logger,
        )
        
        start = datetime.utcnow()
        try:
            await agent.start()
            result = await agent.run_test_case(
                test_case=case,
                run_id=f"{case.id}-{run_id}",
                screenshots_root=self.config.reporting.screenshots_folder,
                trace_path=trace_path,
            )
            result.retry_attempt = retry_attempt
            result.browser_type = self.config.browser.browser
            return result
        except Exception as exc:
            self.logger.error(f"Task {case.id} crashed: {exc}", exc_info=True)
            end = datetime.utcnow()
            return TestRunResult(
                case=case,
                success=False,
                started_at=start,
                finished_at=end,
                reason=f"Runner exception: {exc}",
                actions=[],
                facts=[],
                retry_attempt=retry_attempt,
                browser_type=self.config.browser.browser,
            )
        finally:
            await agent.close()

    async def run_task_with_retries(self, case: TestCase, trace_path: Optional[Path] = None) -> TestRunResult:
        """Run a task with configured retries."""
        retry_count = case.retry_count
        last_result = None
        
        for attempt in range(retry_count + 1):
            if attempt > 0:
                self.logger.info(f"Retrying task {case.id} (attempt {attempt + 1}/{retry_count + 1})")
            
            result = await self.run_task(case, retry_attempt=attempt, trace_path=trace_path)
            last_result = result
            
            if result.success:
                return result
        
        return last_result

    async def run_sequential(self, cases: Sequence[TestCase]) -> List[TestRunResult]:
        """Run test cases sequentially."""
        results: List[TestRunResult] = []
        
        for i, case in enumerate(cases, 1):
            self.logger.info(f"=== Running task {case.id} ({i}/{len(cases)}) ===")
            
            if case.skip:
                self.logger.info(f"Skipping {case.id}: {case.skip_reason or 'marked as skip'}")
                results.append(TestRunResult(
                    case=case,
                    success=False,
                    started_at=datetime.utcnow(),
                    finished_at=datetime.utcnow(),
                    reason=f"Skipped: {case.skip_reason or 'marked as skip'}",
                    actions=[],
                    facts=[],
                ))
                continue
            
            result = await self.run_task_with_retries(case)
            results.append(result)
            
            # Generate individual report
            self._generate_report(result)
        
        return results

    async def run_parallel(
        self,
        cases: Sequence[TestCase],
        max_workers: int = 4,
    ) -> List[TestRunResult]:
        """Run test cases in parallel with limited concurrency."""
        semaphore = asyncio.Semaphore(max_workers)
        
        async def run_with_limit(case: TestCase, index: int) -> TestRunResult:
            async with semaphore:
                self.logger.info(f"=== Starting task {case.id} ({index}/{len(cases)}) ===")
                
                if case.skip:
                    self.logger.info(f"Skipping {case.id}: {case.skip_reason or 'marked as skip'}")
                    return TestRunResult(
                        case=case,
                        success=False,
                        started_at=datetime.utcnow(),
                        finished_at=datetime.utcnow(),
                        reason=f"Skipped: {case.skip_reason or 'marked as skip'}",
                        actions=[],
                        facts=[],
                    )
                
                result = await self.run_task_with_retries(case)
                self._generate_report(result)
                return result
        
        tasks = [run_with_limit(case, i + 1) for i, case in enumerate(cases)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Task failed with exception: {result}")
                final_results.append(TestRunResult(
                    case=cases[i],
                    success=False,
                    started_at=datetime.utcnow(),
                    finished_at=datetime.utcnow(),
                    reason=f"Exception: {result}",
                    actions=[],
                    facts=[],
                ))
            else:
                final_results.append(result)
        
        return final_results

    async def run_all(self, cases: Sequence[TestCase]) -> TestSuiteResult:
        """Run all test cases with configured parallelism."""
        start_time = datetime.utcnow()
        
        if self.config.parallel_workers > 1:
            self.logger.info(f"Running {len(cases)} tests with {self.config.parallel_workers} parallel workers")
            results = await self.run_parallel(cases, self.config.parallel_workers)
        else:
            results = await self.run_sequential(cases)
        
        end_time = datetime.utcnow()
        
        suite_result = TestSuiteResult(
            results=results,
            started_at=start_time,
            finished_at=end_time,
        )
        
        # Generate suite-level reports
        self._generate_suite_reports(suite_result)
        
        return suite_result

    def _generate_report(self, result: TestRunResult) -> None:
        """Generate reports for a single test result."""
        output_dir = self.config.reporting.reports_folder
        output_format = self.config.reporting.output_format
        embed = self.config.reporting.embed_screenshots
        
        if output_format in (ReportFormat.HTML, ReportFormat.ALL, "html", "all"):
            reporter = HTMLReporter(embed_screenshots=embed)
            path = reporter.generate(result, output_dir)
            self.logger.info(f"HTML report: {path}")
        
        if output_format in (ReportFormat.JSON, ReportFormat.ALL, "json", "all"):
            reporter = JSONReporter()
            path = reporter.generate(result, output_dir)
            self.logger.info(f"JSON report: {path}")
        
        if output_format in (ReportFormat.JUNIT, ReportFormat.ALL, "junit", "all"):
            reporter = JUnitReporter()
            path = reporter.generate(result, output_dir)
            self.logger.info(f"JUnit report: {path}")

    def _generate_suite_reports(self, suite: TestSuiteResult) -> None:
        """Generate suite-level reports."""
        output_dir = self.config.reporting.reports_folder
        output_format = self.config.reporting.output_format
        embed = self.config.reporting.embed_screenshots
        
        if output_format in (ReportFormat.HTML, ReportFormat.ALL, "html", "all"):
            reporter = HTMLReporter(embed_screenshots=embed)
            path = reporter.generate_suite(suite.results, output_dir)
            self.logger.info(f"Suite HTML report: {path}")
        
        if output_format in (ReportFormat.JSON, ReportFormat.ALL, "json", "all"):
            reporter = JSONReporter()
            path = reporter.generate_suite(suite.results, output_dir)
            self.logger.info(f"Suite JSON report: {path}")
        
        if output_format in (ReportFormat.JUNIT, ReportFormat.ALL, "junit", "all"):
            reporter = JUnitReporter()
            path = reporter.generate_suite(suite.results, output_dir)
            self.logger.info(f"Suite JUnit report: {path}")


async def run_from_cli_args(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Entry point shared by the CLI script."""
    tasks_dir = Path(args.tasks_dir)
    
    # Build tag filters
    include_tags: Optional[Set[str]] = None
    exclude_tags: Optional[Set[str]] = None
    
    if args.tag:
        include_tags = set(args.tag)
    if args.exclude_tag:
        exclude_tags = set(args.exclude_tag)
    
    try:
        cases = discover_tasks(
            tasks_dir,
            only_ids=args.task if args.task else None,
            include_tags=include_tags,
            exclude_tags=exclude_tags,
            include_skipped=args.include_skipped,
            sort_by_priority=args.sort_by_priority,
        )
    except TaskLoadError as exc:
        logger.error(str(exc))
        return 1

    if not cases:
        logger.warning("No test cases found matching filters")
        return 0

    # Load config with CLI overrides
    config_path = Path(args.config) if args.config else None
    cli_overrides = {
        "browser": args.browser,
        "headful": args.headful,
        "parallel": args.parallel,
        "verbose": args.verbose,
        "output_format": args.output_format,
        "base_url": args.base_url,
    }
    # Remove None values
    cli_overrides = {k: v for k, v in cli_overrides.items() if v is not None}
    
    try:
        config = load_config(config_path, cli_overrides)
    except Exception as exc:
        logger.error(f"Failed to load config: {exc}")
        return 1
    
    # Override reports dir from CLI if provided
    if args.reports_dir:
        config.reporting.reports_folder = Path(args.reports_dir)

    logger.info(f"Loaded {len(cases)} test case(s)")
    if config.verbose:
        logger.info(f"Browser: {config.browser.browser}, Headless: {config.browser.headless}")
        logger.info(f"Parallel workers: {config.parallel_workers}")
        logger.info(f"Output format: {config.reporting.output_format}")

    runner = E2ETestRunner(config=config, logger=logger)
    suite_result = await runner.run_all(cases)

    # Print summary
    print("\n" + "=" * 60)
    print(f"TEST SUITE SUMMARY")
    print("=" * 60)
    print(f"Total:  {suite_result.total}")
    print(f"Passed: {suite_result.passed}")
    print(f"Failed: {suite_result.failed}")
    print(f"Pass Rate: {suite_result.pass_rate:.1f}%")
    print(f"Duration: {suite_result.duration_seconds:.1f}s")
    print("=" * 60)

    if suite_result.failed_tests:
        print("\nFailed Tests:")
        for result in suite_result.failed_tests:
            print(f"  - {result.case.id}: {result.reason[:80]}")
    
    return 1 if suite_result.failed > 0 else 0


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Run natural-language browser E2E tests with AI agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Run all tests
  %(prog)s --task login-test            # Run specific test
  %(prog)s --tag smoke                  # Run tests tagged 'smoke'
  %(prog)s --parallel 4 --browser chromium  # Parallel with Chrome
  %(prog)s --output-format all          # Generate all report formats
        """,
    )
    
    # Task selection
    task_group = parser.add_argument_group("Task Selection")
    task_group.add_argument(
        "--tasks-dir",
        default="tasks",
        help="Directory containing task YAML/JSON files (default: tasks)",
    )
    task_group.add_argument(
        "--task",
        action="append",
        help="Specific task ID to run (can be used multiple times)",
    )
    task_group.add_argument(
        "--tag",
        action="append",
        help="Only run tests with this tag (can be used multiple times)",
    )
    task_group.add_argument(
        "--exclude-tag",
        action="append",
        help="Exclude tests with this tag (can be used multiple times)",
    )
    task_group.add_argument(
        "--include-skipped",
        action="store_true",
        help="Include tests marked as skip=true",
    )
    task_group.add_argument(
        "--sort-by-priority",
        action="store_true",
        help="Sort tests by priority (1=highest first)",
    )
    
    # Browser options
    browser_group = parser.add_argument_group("Browser Options")
    browser_group.add_argument(
        "--browser",
        choices=["chromium", "firefox", "webkit"],
        help="Browser engine to use (default: firefox)",
    )
    browser_group.add_argument(
        "--headful",
        action="store_true",
        help="Run browser in headful mode (show GUI)",
    )
    browser_group.add_argument(
        "--base-url",
        help="Override start URLs with this base URL",
    )
    
    # Execution options
    exec_group = parser.add_argument_group("Execution Options")
    exec_group.add_argument(
        "--parallel",
        type=int,
        metavar="N",
        help="Number of parallel test workers (default: 1)",
    )
    exec_group.add_argument(
        "--config",
        help="Path to config file (default: config.json if exists)",
    )
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--reports-dir",
        default="reports",
        help="Directory for saving reports (default: reports)",
    )
    output_group.add_argument(
        "--output-format",
        choices=["html", "json", "junit", "all"],
        help="Report output format (default: html)",
    )
    output_group.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    output_group.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress non-essential output",
    )
    
    return parser


def main() -> None:
    """Main entry point."""
    parser = _build_arg_parser()
    args = parser.parse_args()
    
    # Configure logging based on verbosity
    if args.quiet:
        log_level = logging.WARNING
    elif args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format="[%(levelname)s] %(message)s" if not args.verbose else "[%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("e2e_runner")
    
    try:
        exit_code = asyncio.run(run_from_cli_args(args, logger))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        exit_code = 130
    except FaraError as exc:
        logger.error(f"Error: {exc}")
        exit_code = 1
    except Exception as exc:
        logger.exception(f"Unexpected error: {exc}")
        exit_code = 1
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
