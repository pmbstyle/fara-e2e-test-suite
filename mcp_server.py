"""MCP server bridge for the Fara E2E runner.

This exposes a small tool surface for the coding agent to:
- list and create tasks
- run a task (headless by default, optional headful debug)
- list runs and fetch compact/full reports (with file:// links)
- retry a previous run

Transport: stdio (local-first).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import mcp.types as types
from mcp.server import InitializationOptions, Server
from mcp.server.stdio import stdio_server
from mcp.server.sse import SseServerTransport
import anyio
from logging.handlers import RotatingFileHandler
import asyncio
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.responses import Response
import uvicorn

from config import load_config
from reporters import HTMLReporter, JSONReporter
from task_loader import discover_tasks, load_task_file
from test_runner import E2ETestRunner
from test_types import TestCase, TestRunResult

logger = logging.getLogger("fara_mcp")
logger.propagate = False
LOG_FILE = Path(__file__).with_name("mcp_server.log")

ROOT = Path(__file__).parent
TASKS_DIR = ROOT / "tasks"
REPORTS_DIR = ROOT / "reports"
SCREENSHOTS_DIR = ROOT / "screenshots"
RUN_INDEX_PATH = REPORTS_DIR / "run_index.json"


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _file_uri(path: Path | None) -> str | None:
    if not path:
        return None
    return path.resolve().as_uri()


def _compact_actions(result: TestRunResult, limit: int = 5) -> list[dict[str, Any]]:
    actions = result.actions[-limit:]
    return [
        {
            "round": a.round_index,
            "action": a.action,
            "result": a.result,
            "url": a.page_url,
            "screenshot": _file_uri(a.screenshot_path),
        }
        for a in actions
    ]


def _build_summary(result: TestRunResult, limit_actions: int = 5) -> dict[str, Any]:
    return {
        "task_id": result.case.id,
        "objective": result.case.objective,
        "success": result.success,
        "reason": result.reason,
        "started_at": result.started_at.isoformat(),
        "finished_at": result.finished_at.isoformat(),
        "duration_seconds": result.duration_seconds,
        "final_url": result.final_url,
        "actions": _compact_actions(result, limit_actions),
        "console_errors": result.console_errors,
    }


@dataclass
class RunRecord:
    run_id: str
    task_id: str
    status: str  # queued|running|passed|failed|error
    success: bool | None
    reason: str | None
    started_at: str
    finished_at: str | None
    report_paths: dict[str, str] | None


class RunIndex:
    """Persistent run index so the agent can fetch by id or retry."""

    def __init__(self, path: Path):
        self.path = path
        self._runs: dict[str, RunRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            for run_id, data in raw.items():
                self._runs[run_id] = RunRecord(**data)
        except Exception as exc:
            logger.warning(f"Failed to load run index: {exc}")

    def _save(self) -> None:
        payload = {rid: asdict(run) for rid, run in self._runs.items()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def list_runs(self) -> list[RunRecord]:
        return list(self._runs.values())

    def get(self, run_id: str) -> RunRecord | None:
        return self._runs.get(run_id)

    def put(self, record: RunRecord) -> None:
        self._runs[record.run_id] = record
        self._save()


class TaskStore:
    """Lightweight task store backed by the existing tasks directory."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def list_tasks(self) -> list[dict[str, Any]]:
        tasks = discover_tasks(self.root)
        return [
            {
                "id": t.id,
                "objective": t.objective,
                "tags": sorted(t.tags),
                "priority": t.priority,
                "skip": t.skip,
            }
            for t in tasks
        ]

    def load(self, task_id: str) -> TestCase:
        path = self.root / f"{task_id}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Task not found: {task_id}")
        return load_task_file(path)

    def create(self, task: dict[str, Any]) -> str:
        task_id = str(task.get("id") or task.get("name") or f"task-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}")
        path = self.root / f"{task_id}.yaml"
        if path.exists():
            raise FileExistsError(f"Task already exists: {task_id}")
        # Persist as YAML to stay compatible
        import yaml

        path.write_text(yaml.safe_dump(task), encoding="utf-8")
        return task_id


class E2EMCPServer:
    """Glue layer between MCP and the E2E runner."""

    def __init__(self) -> None:
        self.server = Server("fara-e2e-mcp", instructions="Run and inspect UI E2E tests")
        self.task_store = TaskStore(TASKS_DIR)
        self.run_index = RunIndex(RUN_INDEX_PATH)
        self._active_tasks: dict[str, asyncio.Task] = {}
        self._semaphore = asyncio.Semaphore(1)  # limit concurrent Playwright runs
        self._register_handlers()

    def _register_handlers(self) -> None:
        @self.server.list_tools()
        async def list_tools() -> list[types.Tool]:
            return [
                types.Tool(
                    name="list_tasks",
                    description="List available E2E tasks",
                    inputSchema={"type": "object", "properties": {}},
                ),
                types.Tool(
                    name="create_task",
                    description="Create a new task (YAML-compatible payload)",
                    inputSchema={"type": "object", "properties": {}, "additionalProperties": True},
                ),
                types.Tool(
                    name="run_task",
                    description="Run a task by id",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "task_id": {"type": "string"},
                            "headful_debug": {"type": "boolean"},
                            "retries": {"type": "integer"},
                            "output_format": {"type": "string", "enum": ["json", "html", "all"]},
                            "max_actions": {"type": "integer"},
                        },
                        "required": ["task_id"],
                    },
                ),
                types.Tool(
                    name="list_runs",
                    description="List recent runs",
                    inputSchema={"type": "object", "properties": {}},
                ),
                types.Tool(
                    name="get_run_status",
                    description="Get compact status for a run_id",
                    inputSchema={"type": "object", "properties": {"run_id": {"type": "string"}}, "required": ["run_id"]},
                ),
                types.Tool(
                    name="get_report",
                    description="Fetch report paths and details for a run_id",
                    inputSchema={"type": "object", "properties": {"run_id": {"type": "string"}}, "required": ["run_id"]},
                ),
                types.Tool(
                    name="retry_run",
                    description="Retry the task from a previous run_id",
                    inputSchema={"type": "object", "properties": {"run_id": {"type": "string"}}, "required": ["run_id"]},
                ),
                types.Tool(
                    name="cancel_run",
                    description="Attempt to cancel an in-progress run",
                    inputSchema={"type": "object", "properties": {"run_id": {"type": "string"}}, "required": ["run_id"]},
                ),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
            def _wrap(payload: dict[str, Any]) -> list[types.TextContent]:
                return [types.TextContent(type="text", text=json.dumps(payload, indent=2))]

            logger.info("call_tool start: %s args=%s", name, arguments)
            try:
                if name == "list_tasks":
                    payload = self.task_store.list_tasks()
                elif name == "create_task":
                    task_id = self.task_store.create(arguments)
                    payload = {"task_id": task_id, "status": "created"}
                elif name == "run_task":
                    payload = await self._run_task(arguments)
                elif name == "list_runs":
                    payload = [asdict(r) for r in self.run_index.list_runs()]
                elif name == "get_run_status":
                    payload = self._get_run(arguments.get("run_id"))
                elif name == "get_report":
                    payload = self._get_run(arguments.get("run_id"), include_paths=True)
                elif name == "retry_run":
                    payload = await self._retry_run(arguments.get("run_id"))
                elif name == "cancel_run":
                    payload = await self._cancel_run(arguments.get("run_id"))
                else:
                    payload = {"error": f"Unknown tool: {name}"}
            except Exception as exc:
                logger.exception("Tool call failed: %s", name)
                payload = {"error": str(exc), "tool": name}

            logger.info("call_tool done: %s", name)
            return _wrap(payload)

    def _get_run(self, run_id: Optional[str], include_paths: bool = False) -> dict[str, Any]:
        if not run_id:
            raise ValueError("run_id is required")
        record = self.run_index.get(run_id)
        if not record:
            raise FileNotFoundError(f"Run not found: {run_id}")
        data = asdict(record)
        if not include_paths:
            data.pop("report_paths", None)
        return data

    async def _run_task(self, args: dict[str, Any]) -> dict[str, Any]:
        task_id = args["task_id"]
        headful_debug = bool(args.get("headful_debug", False))
        retries = args.get("retries")
        output_format = args.get("output_format") or "json"
        max_actions = int(args.get("max_actions") or 5)

        case = self.task_store.load(task_id)
        if retries is not None:
            case.retry_count = max(0, int(retries))

        logger.info("run_task enqueue: %s", task_id)
        run_id = f"{task_id}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        record = RunRecord(
            run_id=run_id,
            task_id=task_id,
            status="queued",
            success=None,
            reason=None,
            started_at=_now_iso(),
            finished_at=None,
            report_paths=None,
        )
        self.run_index.put(record)

        async def worker() -> None:
            logger.info("run_task worker start: %s", task_id)
            record.status = "running"
            record.started_at = _now_iso()
            self.run_index.put(record)

            config = load_config()
            config.browser.headless = not headful_debug
            config.reporting.reports_folder = REPORTS_DIR
            config.reporting.screenshots_folder = SCREENSHOTS_DIR
            config.reporting.output_format = output_format

            try:
                runner = E2ETestRunner(config=config, logger=logger)
                result = await runner.run_task_with_retries(case)

                reporter = JSONReporter()
                json_path = reporter.generate(result, REPORTS_DIR)
                html_path = None
                if output_format in ("html", "all"):
                    html_path = HTMLReporter(embed_screenshots=config.reporting.embed_screenshots).generate(
                        result, REPORTS_DIR
                    )

                record.status = "passed" if result.success else "failed"
                record.success = result.success
                record.reason = result.reason
                record.finished_at = _now_iso()
                record.report_paths = {
                    "json": _file_uri(json_path),
                    "html": _file_uri(html_path) if html_path else None,
                }
                self.run_index.put(record)
            except Exception as exc:
                logger.exception("run_task worker failed")
                record.status = "error"
                record.success = False
                record.reason = str(exc)
                record.finished_at = _now_iso()
                self.run_index.put(record)
            finally:
                logger.info("run_task worker finished: %s", task_id)

        async def wrapped_worker():
            async with self._semaphore:
                await worker()

        task = asyncio.create_task(wrapped_worker(), name=f"run-{run_id}")
        self._active_tasks[run_id] = task
        task.add_done_callback(lambda t: self._active_tasks.pop(run_id, None))

        # Respond immediately; clients should poll get_run_status/get_report
        return {
            "run_id": run_id,
            "task_id": task_id,
            "status": "queued",
            "queued_at": record.started_at,
        }

    async def _retry_run(self, run_id: Optional[str]) -> dict[str, Any]:
        if not run_id:
            raise ValueError("run_id is required")
        record = self.run_index.get(run_id)
        if not record:
            raise FileNotFoundError(f"Run not found: {run_id}")
        args = {"task_id": record.task_id}
        return await self._run_task(args)

    async def _cancel_run(self, run_id: Optional[str]) -> dict[str, Any]:
        if not run_id:
            raise ValueError("run_id is required")
        task = self._active_tasks.get(run_id)
        record = self.run_index.get(run_id)
        if not record:
            raise FileNotFoundError(f"Run not found: {run_id}")
        if not task:
            return {"run_id": run_id, "status": record.status, "message": "Not running"}
        task.cancel()
        record.status = "error"
        record.reason = "Cancelled"
        record.finished_at = _now_iso()
        record.success = False
        self.run_index.put(record)
        return {"run_id": run_id, "status": "cancelled"}

    async def serve(self) -> None:
        init_opts = InitializationOptions(
            server_name="fara-e2e-mcp",
            server_version="0.1.0",
            capabilities=self.server.get_capabilities(
                notification_options=self.server.notification_options,
                experimental_capabilities={},
            ),
            instructions=self.server.instructions,
        )
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                initialization_options=init_opts,
            )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Fara MCP server (stdio or HTTP SSE)")
    parser.add_argument("--http", help="Run HTTP SSE server on host:port (e.g., 127.0.0.1:8765)")
    args = parser.parse_args()

    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
        handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"))
        # Replace any existing handlers to avoid writing to stdout/stderr (which breaks MCP stdio)
        logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
        logger.handlers = [handler]
        logging.getLogger("mcp").setLevel(logging.DEBUG)
        logging.getLogger("anyio").setLevel(logging.WARNING)
        logger.info("MCP server logging to %s", LOG_FILE)
    except Exception:
        logger.exception("Failed to set up file logging")

    srv = E2EMCPServer()

    async def run_stdio():
        init_opts = InitializationOptions(
            server_name="fara-e2e-mcp",
            server_version="0.1.0",
            capabilities=srv.server.get_capabilities(
                notification_options=srv.server.notification_options,
                experimental_capabilities={},
            ),
            instructions=srv.server.instructions,
        )
        async with stdio_server() as (read_stream, write_stream):
            await srv.server.run(
                read_stream,
                write_stream,
                initialization_options=init_opts,
            )

    async def run_http(bind: str):
        host, port = bind.split(":")
        transport = SseServerTransport("/messages")

        async def handle_sse(request):
            async with transport.connect_sse(request.scope, request.receive, request._send) as streams:
                init_opts = InitializationOptions(
                    server_name="fara-e2e-mcp",
                    server_version="0.1.0",
                    capabilities=srv.server.get_capabilities(
                        notification_options=srv.server.notification_options,
                        experimental_capabilities={},
                    ),
                    instructions=srv.server.instructions,
                )
                await srv.server.run(
                    streams[0],
                    streams[1],
                    initialization_options=init_opts,
                    stateless=True,
                )
            return Response()

        async def handle_root(request):
            return Response("fara-e2e MCP server", media_type="text/plain")

        async def post_message(request):
            session_id = request.query_params.get("session_id")
            if not session_id:
                return Response("Accepted", status_code=202)
            await transport.handle_post_message(request.scope, request.receive, request._send)
            return Response("Accepted", status_code=202)

        async def handle_oauth(request):
            return Response(status_code=404)

        routes = [
            Route("/", endpoint=handle_root, methods=["GET"]),
            Route("/sse", endpoint=handle_sse, methods=["GET"]),
            Route("/.well-known/oauth-authorization-server", endpoint=handle_oauth, methods=["GET"]),
            Route("/", endpoint=post_message, methods=["POST"]),
            Route("/messages", endpoint=post_message, methods=["POST"]),
        ]

        app = Starlette(routes=routes)
        logger.info("HTTP SSE server listening on http://%s:%s", host, port)
        config = uvicorn.Config(app, host=host, port=int(port), log_level="info")
        server = uvicorn.Server(config)
        await server.serve()

    try:
        if args.http:
            anyio.run(run_http, args.http)
        else:
            anyio.run(run_stdio)
    except Exception:
        logger.exception("MCP server crashed")
        raise


if __name__ == "__main__":
    main()
