"""Enhanced HTML report generator for E2E test runs."""
from __future__ import annotations

import base64
import html
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from reporters.base import BaseReporter, ReportFormat
from test_types import ActionTrace, TestRunResult


class HTMLReporter(BaseReporter):
    """Generate enhanced HTML reports with embedded screenshots and timeline."""

    def __init__(self, embed_screenshots: bool = False):
        """
        Initialize HTML reporter.
        
        Args:
            embed_screenshots: If True, embed screenshots as base64 in the HTML
        """
        self.embed_screenshots = embed_screenshots

    @property
    def format(self) -> ReportFormat:
        return ReportFormat.HTML

    def _render_list(self, items: List[str]) -> str:
        """Render a list of items as HTML."""
        if not items:
            return "<p class='empty'>None</p>"
        inner = "".join(f"<li>{html.escape(item)}</li>" for item in items)
        return f"<ul>{inner}</ul>"

    def _get_screenshot_src(self, screenshot_path: Optional[Path], report_root: Path) -> str:
        """Get screenshot source - either base64 or relative path."""
        if not screenshot_path:
            return ""
        
        path = Path(screenshot_path)
        if not path.exists():
            return ""
        
        if self.embed_screenshots:
            try:
                with open(path, "rb") as f:
                    data = base64.b64encode(f.read()).decode("utf-8")
                return f"data:image/png;base64,{data}"
            except Exception:
                pass
        
        return os.path.relpath(path, start=report_root)

    def _render_result_message(self, result: TestRunResult) -> str:
        """Render the result message with appropriate styling."""
        if result.success:
            return f"""
            <div class="meta" style="margin-top: 10px;">
                <div class="pill" style="grid-column: 1 / -1; background: rgba(15, 81, 50, 0.3); border-color: var(--success-border);">
                    <strong>✓ Result:</strong> {html.escape(result.reason)}
                </div>
            </div>"""
        else:
            return f"""
            <div style="margin-top: 12px;">
                <div class="error-label">⚠ Error Details</div>
                <div class="error-details">{html.escape(result.reason)}</div>
            </div>"""

    def _render_timeline(self, actions: List[ActionTrace], report_root: Path) -> str:
        """Render an interactive timeline of actions."""
        if not actions:
            return """<div class='empty-state'>
                <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="12" y1="8" x2="12" y2="12"></line>
                    <line x1="12" y1="16" x2="12.01" y2="16"></line>
                </svg>
                <p>No actions were recorded during this test run.</p>
                <p class='empty-hint'>This usually means the test failed before any actions could be executed.</p>
            </div>"""

        items = []
        for act in actions:
            status_class = "success" if act.action != "auto_terminate" or "success" in act.result.lower() else "failure"
            if act.action == "terminate":
                status_class = "success" if "success" in act.arguments.get("status", "").lower() else "failure"
            
            screenshot_html = ""
            if act.screenshot_path:
                src = self._get_screenshot_src(act.screenshot_path, report_root)
                if src:
                    screenshot_html = f'''
                    <div class="screenshot-preview">
                        <img src="{src}" alt="Step {act.round_index}" loading="lazy" 
                             onclick="openModal(this.src)" />
                    </div>'''
            
            # Parse model response for display
            model_preview = act.model_response[:300].replace("<", "&lt;").replace(">", "&gt;")
            if len(act.model_response) > 300:
                model_preview += "..."
            
            args_str = html.escape(str(act.arguments))
            
            items.append(f'''
            <div class="timeline-item {status_class}" data-round="{act.round_index}">
                <div class="timeline-marker">
                    <span class="round-num">{act.round_index}</span>
                </div>
                <div class="timeline-content">
                    <div class="timeline-header">
                        <span class="action-name">{html.escape(act.action)}</span>
                        <span class="action-url">{html.escape(act.page_url)}</span>
                    </div>
                    <div class="timeline-body">
                        <div class="action-args"><code>{args_str}</code></div>
                        <div class="action-result">{html.escape(act.result)}</div>
                        <details class="model-response">
                            <summary>Model Response</summary>
                            <pre>{model_preview}</pre>
                        </details>
                    </div>
                    {screenshot_html}
                </div>
            </div>''')

        return f'<div class="timeline">{"".join(items)}</div>'

    def _render_actions_table(self, actions: List[ActionTrace], report_root: Path) -> str:
        """Render actions as a filterable table."""
        if not actions:
            return """<div class='empty-state'>
                <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                    <line x1="9" y1="9" x2="15" y2="9"></line>
                    <line x1="9" y1="15" x2="15" y2="15"></line>
                </svg>
                <p>No actions were recorded during this test run.</p>
                <p class='empty-hint'>Check the error message above for details on what went wrong.</p>
            </div>"""

        rows = []
        for act in actions:
            screenshot_cell = "—"
            if act.screenshot_path:
                src = self._get_screenshot_src(act.screenshot_path, report_root)
                if src:
                    if self.embed_screenshots:
                        screenshot_cell = f'<img src="{src}" class="thumb" onclick="openModal(this.src)" />'
                    else:
                        screenshot_cell = f'<a href="{src}" target="_blank">view</a>'

            rows.append(f"""
            <tr data-action="{html.escape(act.action)}">
                <td>{act.round_index}</td>
                <td><span class="action-badge">{html.escape(act.action)}</span></td>
                <td><code class="url">{html.escape(act.page_url)}</code></td>
                <td><code class="args">{html.escape(str(act.arguments))}</code></td>
                <td class="result">{html.escape(act.result)}</td>
                <td class="model-col">{html.escape(act.model_response[:400])}</td>
                <td class="screenshot-col">{screenshot_cell}</td>
            </tr>""")

        return f"""
        <div class="table-controls">
            <input type="text" id="actionFilter" placeholder="Filter actions..." onkeyup="filterTable()">
            <select id="actionTypeFilter" onchange="filterTable()">
                <option value="">All Actions</option>
                <option value="click">Clicks</option>
                <option value="type">Type</option>
                <option value="scroll">Scroll</option>
                <option value="terminate">Terminate</option>
            </select>
        </div>
        <table class="actions" id="actionsTable">
            <thead>
                <tr>
                    <th>#</th>
                    <th>Action</th>
                    <th>URL</th>
                    <th>Arguments</th>
                    <th>Result</th>
                    <th>Model Snippet</th>
                    <th>Screenshot</th>
                </tr>
            </thead>
            <tbody>{"".join(rows)}</tbody>
        </table>"""

    def _get_css(self) -> str:
        """Return the CSS styles for the report."""
        return """
        :root {
            --bg-primary: #0b1220;
            --bg-card: #111a2d;
            --bg-input: #0f1729;
            --border-color: #1f2a44;
            --text-primary: #e6edf7;
            --text-secondary: #c3cee6;
            --accent-blue: #9dd0ff;
            --success-bg: #0f5132;
            --success-text: #b6f6d8;
            --success-border: #1e7a46;
            --fail-bg: #5b1a1a;
            --fail-text: #f6c6c6;
            --fail-border: #8a2f2f;
        }
        
        * { box-sizing: border-box; }
        
        body {
            font-family: "Inter", "Segoe UI", -apple-system, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            margin: 0;
            padding: 24px;
            line-height: 1.5;
        }
        
        .container { max-width: 1400px; margin: 0 auto; }
        
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 18px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.4);
        }
        
        .header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 20px;
            flex-wrap: wrap;
        }
        
        .header-content { flex: 1; min-width: 300px; }
        
        h1 {
            margin: 0 0 8px;
            font-size: 1.5rem;
            font-weight: 600;
            color: #f5f7ff;
        }
        
        h2 {
            margin: 0 0 12px;
            font-size: 1.1rem;
            font-weight: 600;
            color: #f5f7ff;
        }
        
        p { margin: 4px 0; color: var(--text-secondary); }
        
        .badge {
            padding: 8px 16px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.9rem;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }
        
        .badge.pass {
            background: var(--success-bg);
            color: var(--success-text);
            border: 1px solid var(--success-border);
        }
        
        .badge.fail {
            background: var(--fail-bg);
            color: var(--fail-text);
            border: 1px solid var(--fail-border);
        }
        
        .meta {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-top: 16px;
        }
        
        .pill {
            background: var(--bg-input);
            border: 1px solid var(--border-color);
            padding: 10px 12px;
            border-radius: 8px;
            color: var(--text-secondary);
            font-size: 0.875rem;
        }
        
        .pill strong { color: var(--text-primary); }
        
        code {
            background: #0c1424;
            padding: 2px 6px;
            border-radius: 4px;
            color: #d3e1ff;
            font-family: "JetBrains Mono", "Fira Code", monospace;
            font-size: 0.85em;
        }
        
        ul {
            margin: 8px 0 0 20px;
            padding: 0;
            color: var(--text-secondary);
        }
        
        li { margin: 4px 0; }
        
        .empty { color: #666; font-style: italic; }
        
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: var(--text-secondary);
        }
        
        .empty-state svg {
            opacity: 0.3;
            margin-bottom: 20px;
        }
        
        .empty-state p {
            margin: 8px 0;
            font-size: 1rem;
        }
        
        .empty-hint {
            font-size: 0.875rem;
            color: #888;
            font-style: italic;
        }
        
        .back-button {
            display: inline-block;
            margin-bottom: 16px;
            padding: 8px 16px;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-primary);
            text-decoration: none;
            font-size: 0.9rem;
            transition: all 0.2s;
        }
        
        .back-button:hover {
            background: var(--bg-input);
            border-color: var(--accent-blue);
            color: var(--accent-blue);
            transform: translateX(-4px);
        }
        
        .error-details {
            background: rgba(139, 26, 26, 0.2);
            border: 1px solid var(--fail-border);
            border-radius: 8px;
            padding: 16px;
            margin-top: 12px;
            font-family: "JetBrains Mono", "Fira Code", monospace;
            font-size: 0.85rem;
            line-height: 1.6;
            color: var(--fail-text);
            white-space: pre-wrap;
            word-break: break-word;
        }
        
        .error-label {
            font-weight: 600;
            color: var(--fail-text);
            margin-bottom: 8px;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.5px;
        }
        
        .criteria-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
        }
        
        @media (max-width: 768px) {
            .criteria-grid {
                grid-template-columns: 1fr;
            }
        }
        
        .criteria-grid h2 {
            font-size: 1rem;
            margin-bottom: 12px;
        }
        
        .criteria-grid ul {
            margin: 0;
            padding-left: 20px;
        }
        
        /* Timeline Styles */
        .timeline {
            position: relative;
            padding-left: 40px;
        }
        
        .timeline::before {
            content: '';
            position: absolute;
            left: 15px;
            top: 0;
            bottom: 0;
            width: 2px;
            background: var(--border-color);
        }
        
        .timeline-item {
            position: relative;
            margin-bottom: 20px;
            padding: 16px;
            background: var(--bg-input);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .timeline-item:hover {
            transform: translateX(4px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        
        .timeline-item.success { border-left: 3px solid var(--success-border); }
        .timeline-item.failure { border-left: 3px solid var(--fail-border); }
        
        .timeline-marker {
            position: absolute;
            left: -33px;
            top: 16px;
            width: 24px;
            height: 24px;
            background: var(--bg-card);
            border: 2px solid var(--accent-blue);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .round-num {
            font-size: 0.7rem;
            font-weight: 700;
            color: var(--accent-blue);
        }
        
        .timeline-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
            flex-wrap: wrap;
            gap: 8px;
        }
        
        .action-name {
            font-weight: 600;
            color: var(--accent-blue);
            text-transform: uppercase;
            font-size: 0.8rem;
            letter-spacing: 0.5px;
        }
        
        .action-url {
            font-size: 0.75rem;
            color: #888;
            max-width: 400px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        
        .timeline-body { font-size: 0.9rem; }
        
        .action-args {
            margin: 8px 0;
            padding: 8px;
            background: rgba(0,0,0,0.2);
            border-radius: 4px;
            overflow-x: auto;
        }
        
        .action-result {
            color: var(--text-secondary);
            padding: 8px 0;
        }
        
        .model-response {
            margin-top: 8px;
            font-size: 0.85rem;
        }
        
        .model-response summary {
            cursor: pointer;
            color: #888;
            padding: 4px;
        }
        
        .model-response pre {
            margin: 8px 0 0;
            padding: 12px;
            background: rgba(0,0,0,0.3);
            border-radius: 4px;
            overflow-x: auto;
            white-space: pre-wrap;
            word-break: break-word;
            font-size: 0.8rem;
            color: #aaa;
        }
        
        .screenshot-preview {
            margin-top: 12px;
        }
        
        .screenshot-preview img {
            max-width: 300px;
            max-height: 200px;
            border-radius: 6px;
            border: 1px solid var(--border-color);
            cursor: pointer;
            transition: transform 0.2s;
        }
        
        .screenshot-preview img:hover {
            transform: scale(1.02);
        }
        
        /* Table Styles */
        .table-controls {
            display: flex;
            gap: 12px;
            margin-bottom: 16px;
            flex-wrap: wrap;
        }
        
        .table-controls input,
        .table-controls select {
            background: var(--bg-input);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 8px 12px;
            color: var(--text-primary);
            font-size: 0.875rem;
        }
        
        .table-controls input { flex: 1; min-width: 200px; }
        
        table.actions {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.8rem;
        }
        
        table.actions th,
        table.actions td {
            border: 1px solid var(--border-color);
            padding: 10px;
            text-align: left;
            vertical-align: top;
        }
        
        table.actions th {
            background: #16233b;
            color: #dce9ff;
            font-weight: 600;
            position: sticky;
            top: 0;
        }
        
        table.actions tbody tr:nth-child(odd) { background: #0f1626; }
        table.actions tbody tr:nth-child(even) { background: #10192b; }
        table.actions tbody tr:hover { background: #1a2744; }
        
        .action-badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            background: rgba(157, 208, 255, 0.15);
            color: var(--accent-blue);
            font-size: 0.75rem;
            font-weight: 600;
        }
        
        .url { 
            max-width: 200px;
            display: block;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .args {
            max-width: 200px;
            display: block;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .result { max-width: 200px; }
        .model-col { max-width: 300px; font-size: 0.75rem; color: #888; }
        
        .thumb {
            width: 60px;
            height: 40px;
            object-fit: cover;
            border-radius: 4px;
            cursor: pointer;
        }
        
        a { color: var(--accent-blue); text-decoration: none; }
        a:hover { text-decoration: underline; }
        
        /* Modal */
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.9);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }
        
        .modal.active { display: flex; }
        
        .modal img {
            max-width: 95%;
            max-height: 95%;
            border-radius: 8px;
        }
        
        .modal-close {
            position: absolute;
            top: 20px;
            right: 30px;
            font-size: 2rem;
            color: white;
            cursor: pointer;
        }
        
        /* Tabs */
        .tabs {
            display: flex;
            gap: 4px;
            margin-bottom: 16px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 4px;
        }
        
        .tab {
            padding: 8px 16px;
            background: transparent;
            border: none;
            color: var(--text-secondary);
            cursor: pointer;
            font-size: 0.9rem;
            border-radius: 6px 6px 0 0;
            transition: all 0.2s;
        }
        
        .tab:hover { background: rgba(255,255,255,0.05); }
        .tab.active {
            background: var(--bg-input);
            color: var(--accent-blue);
            font-weight: 600;
        }
        
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        
        @media (max-width: 768px) {
            body { padding: 12px; }
            .header { flex-direction: column; }
            .timeline { padding-left: 30px; }
            .timeline-marker { left: -26px; }
        }
        """

    def _get_js(self) -> str:
        """Return the JavaScript for interactivity."""
        return """
        function filterTable() {
            const filter = document.getElementById('actionFilter').value.toLowerCase();
            const typeFilter = document.getElementById('actionTypeFilter').value.toLowerCase();
            const rows = document.querySelectorAll('#actionsTable tbody tr');
            
            rows.forEach(row => {
                const text = row.textContent.toLowerCase();
                const action = row.dataset.action?.toLowerCase() || '';
                const matchesText = text.includes(filter);
                const matchesType = !typeFilter || action.includes(typeFilter);
                row.style.display = matchesText && matchesType ? '' : 'none';
            });
        }
        
        function openModal(src) {
            const modal = document.getElementById('imageModal');
            const img = document.getElementById('modalImage');
            img.src = src;
            modal.classList.add('active');
        }
        
        function closeModal() {
            document.getElementById('imageModal').classList.remove('active');
        }
        
        function switchTab(tabName) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
            document.getElementById(tabName).classList.add('active');
        }
        
        document.addEventListener('keydown', e => {
            if (e.key === 'Escape') closeModal();
        });
        
        document.getElementById('imageModal')?.addEventListener('click', e => {
            if (e.target.id === 'imageModal') closeModal();
        });
        """

    def generate(self, result: TestRunResult, output_dir: Path) -> Path:
        """Generate HTML report for a single test result."""
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        filename = f"{result.case.id}-{timestamp}.html"
        target = output_dir / filename

        verdict_text = "PASS" if result.success else "FAIL"
        verdict_class = "pass" if result.success else "fail"

        # Check if there's a suite report to link back to
        suite_reports = list(output_dir.glob("suite-*.html"))
        back_button = ""
        if suite_reports:
            latest_suite = max(suite_reports, key=lambda p: p.stat().st_mtime)
            back_button = f'<a href="{latest_suite.name}" class="back-button">← Back to Suite</a>'

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>E2E Report - {html.escape(result.case.id)}</title>
    <style>{self._get_css()}</style>
</head>
<body>
    <div class="container">
        {back_button}
        <div class="card">
            <div class="header">
                <div class="header-content">
                    <h1>{html.escape(result.case.objective)}</h1>
                    <p>Task ID: <code>{html.escape(result.case.id)}</code></p>
                    {f'<p>Start URL: <code>{html.escape(result.case.start_url)}</code></p>' if result.case.start_url else ''}
                </div>
                <span class="badge {verdict_class}">{verdict_text}</span>
            </div>
            <div class="meta">
                <div class="pill"><strong>Started:</strong> {result.started_at.strftime('%Y-%m-%d %H:%M:%S')} UTC</div>
                <div class="pill"><strong>Finished:</strong> {result.finished_at.strftime('%Y-%m-%d %H:%M:%S')} UTC</div>
                <div class="pill"><strong>Duration:</strong> {result.duration_seconds:.1f}s</div>
                <div class="pill"><strong>Actions:</strong> {len(result.actions)}</div>
            </div>
            {self._render_result_message(result)}
        </div>

        <div class="card">
            <div class="criteria-grid">
                <div>
                    <h2>✓ Pass Criteria</h2>
                    {self._render_list(result.case.pass_criteria)}
                </div>
                <div>
                    <h2>✗ Fail Criteria</h2>
                    {self._render_list(result.case.fail_criteria)}
                </div>
            </div>
        </div>

        {f'''<div class="card">
            <h2>Credentials</h2>
            {self._render_list([f"{k}: {v}" for k, v in result.case.credentials.items()])}
        </div>''' if result.case.credentials else ''}

        {f'''<div class="card">
            <h2>Notes</h2>
            <p>{html.escape(result.case.notes)}</p>
        </div>''' if result.case.notes else ''}

        <div class="card">
            <h2>Actions</h2>
            <div class="tabs">
                <button class="tab active" data-tab="timelineView" onclick="switchTab('timelineView')">Timeline</button>
                <button class="tab" data-tab="tableView" onclick="switchTab('tableView')">Table</button>
            </div>
            <div id="timelineView" class="tab-content active">
                {self._render_timeline(result.actions, target.parent)}
            </div>
            <div id="tableView" class="tab-content">
                {self._render_actions_table(result.actions, target.parent)}
            </div>
        </div>
    </div>

    <div class="modal" id="imageModal" onclick="closeModal()">
        <span class="modal-close" onclick="closeModal()">&times;</span>
        <img id="modalImage" src="" alt="Screenshot">
    </div>

    <script>{self._get_js()}</script>
</body>
</html>"""

        target.write_text(html_content, encoding="utf-8")
        return target

    def generate_suite(self, results: List[TestRunResult], output_dir: Path) -> Path:
        """Generate combined HTML report for multiple test results."""
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        filename = f"suite-{timestamp}.html"
        target = output_dir / filename

        passed = sum(1 for r in results if r.success)
        failed = len(results) - passed
        pass_rate = (passed / len(results) * 100) if results else 0

        # Generate individual reports and collect their paths
        individual_reports = {}
        for result in results:
            report_path = self.generate(result, output_dir)
            individual_reports[result.case.id] = report_path.name

        # Build test cards with links
        test_cards = []
        for result in results:
            verdict_class = "pass" if result.success else "fail"
            verdict_text = "PASS" if result.success else "FAIL"
            report_link = individual_reports.get(result.case.id, "#")
            
            # Format reason with proper line breaks
            reason_html = html.escape(result.reason)
            if len(reason_html) > 200:
                reason_html = reason_html[:200] + "..."
            reason_html = reason_html.replace('\n', '<br>')
            
            test_cards.append(f'''
            <div class="test-card {verdict_class}">
                <div class="test-header">
                    <a href="{report_link}" class="test-name-link">
                        <span class="test-name">{html.escape(result.case.id)}</span>
                    </a>
                    <span class="badge {verdict_class}">{verdict_text}</span>
                </div>
                <p class="test-objective">{html.escape(result.case.objective)}</p>
                <div class="test-meta">
                    <span>Duration: {result.duration_seconds:.1f}s</span>
                    <span>Actions: {len(result.actions)}</span>
                </div>
                <p class="test-reason">{reason_html}</p>
                <a href="{report_link}" class="view-details">View Details →</a>
            </div>''')

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>E2E Test Suite Report</title>
    <style>
        {self._get_css()}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 16px;
            margin: 20px 0;
        }}
        .summary-stat {{
            text-align: center;
            padding: 20px;
            background: var(--bg-input);
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }}
        .summary-stat .value {{
            font-size: 2rem;
            font-weight: 700;
            color: var(--accent-blue);
        }}
        .summary-stat.passed .value {{ color: var(--success-text); }}
        .summary-stat.failed .value {{ color: var(--fail-text); }}
        .summary-stat .label {{
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-top: 4px;
        }}
        .test-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 16px;
        }}
        .test-card {{
            background: var(--bg-input);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 16px;
            border-left: 3px solid var(--border-color);
        }}
        .test-card.pass {{ border-left-color: var(--success-border); }}
        .test-card.fail {{ border-left-color: var(--fail-border); }}
        .test-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }}
        .test-name {{
            font-weight: 600;
            color: var(--text-primary);
        }}
        .test-objective {{
            font-size: 0.875rem;
            color: var(--text-secondary);
            margin: 8px 0;
        }}
        .test-meta {{
            font-size: 0.75rem;
            color: #888;
            display: flex;
            gap: 16px;
        }}
        .test-reason {{
            font-size: 0.8rem;
            color: #888;
            margin-top: 8px;
            font-style: italic;
            line-height: 1.4;
        }}
        .test-name-link {{
            text-decoration: none;
            color: inherit;
            transition: color 0.2s;
        }}
        .test-name-link:hover {{
            color: var(--accent-blue);
        }}
        .view-details {{
            display: inline-block;
            margin-top: 12px;
            padding: 6px 12px;
            background: rgba(157, 208, 255, 0.1);
            border: 1px solid var(--accent-blue);
            border-radius: 6px;
            color: var(--accent-blue);
            text-decoration: none;
            font-size: 0.85rem;
            transition: all 0.2s;
        }}
        .view-details:hover {{
            background: rgba(157, 208, 255, 0.2);
            transform: translateX(4px);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <h1>E2E Test Suite Report</h1>
            <p>Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
            
            <div class="summary-grid">
                <div class="summary-stat">
                    <div class="value">{len(results)}</div>
                    <div class="label">Total Tests</div>
                </div>
                <div class="summary-stat passed">
                    <div class="value">{passed}</div>
                    <div class="label">Passed</div>
                </div>
                <div class="summary-stat failed">
                    <div class="value">{failed}</div>
                    <div class="label">Failed</div>
                </div>
                <div class="summary-stat">
                    <div class="value">{pass_rate:.0f}%</div>
                    <div class="label">Pass Rate</div>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>Test Results</h2>
            <div class="test-grid">
                {"".join(test_cards)}
            </div>
        </div>
    </div>
</body>
</html>"""

        target.write_text(html_content, encoding="utf-8")
        return target


# Backward compatibility function
def build_report(result: TestRunResult, output_dir: Path) -> Path:
    """Legacy function for backward compatibility."""
    reporter = HTMLReporter(embed_screenshots=False)
    return reporter.generate(result, output_dir)

