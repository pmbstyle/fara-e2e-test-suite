"""HTML report generator - backward compatibility module.

This module is kept for backward compatibility.
New code should use `from reporters import HTMLReporter` directly.
"""
from reporters.html import HTMLReporter, build_report

__all__ = ["HTMLReporter", "build_report"]
