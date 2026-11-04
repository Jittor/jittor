"""Emulated ``torch_xla.debug.metrics``.

XLA collects compiled-graph counters; Jittor runs eagerly and has no such
metrics.  ``metrics_report`` returns a statement of that fact instead of an
empty string so a printed debug report does not look like a real (empty) XLA
report.
"""

from __future__ import absolute_import

_REPORT = (
    "torch_xla emulation: XLA metrics are unavailable; Jittor executes "
    "eagerly and collects no XLA counters.\n"
)


def metrics_report(show_region_info=False):
    del show_region_info
    return _REPORT


def clear_metrics():
    return None


__all__ = ["metrics_report", "clear_metrics"]
