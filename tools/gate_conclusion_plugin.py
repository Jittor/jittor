"""Record what a pytest session *concluded*, per nodeid, as machine-readable JSON.

Loaded by :mod:`tools.gate_conclusion_diff` (``-p gate_conclusion_plugin``); it
is not imported by the test suite itself.

Why not parse the terminal summary or ``--junitxml``: both answer "how many"
before they answer "which". The optimisation this exists to police -- making a
gate faster -- fails by *losing* a conclusion, and a lost conclusion keeps the
counts plausible. So two separate facts are written down:

* ``collected``  -- the nodeids the session decided to run, and
* ``conclusions`` -- the nodeids it actually reported an outcome for.

The difference between those two sets is the whole point. A worker that dies
mid-run, a crash that takes the session with it, a distribution mode that drops
an item: each of them leaves a collected nodeid with no conclusion, and none of
them changes the exit status in a way that a "N passed" line makes obvious.

The output path comes from ``GATE_CONCLUSION_OUT``; without it the plugin is
inert, so a stray ``-p`` cannot silently overwrite an earlier record.
"""

import json
import os


#: Where to write the record. Unset -> the plugin does nothing.
OUT_VARIABLE = "GATE_CONCLUSION_OUT"


class _Recorder:
    def __init__(self, path):
        self.path = path
        self.collected = []
        self.conclusions = {}
        self.durations = {}

    # -- collection ------------------------------------------------------
    def pytest_collection_modifyitems(self, items):
        # After deselection: what this session intends to run.
        self.collected = [item.nodeid for item in items]

    # -- outcomes --------------------------------------------------------
    def pytest_runtest_logreport(self, report):
        status, reason = _status_of(report)
        self.durations[report.nodeid] = round(
            self.durations.get(report.nodeid, 0.0) + report.duration, 4)
        if status is None:
            return
        previous = self.conclusions.get(report.nodeid)
        # A test that passes its call phase and then errors in teardown has
        # concluded twice; the worse outcome is the conclusion.
        if previous is None or _SEVERITY[status] > _SEVERITY[previous["status"]]:
            self.conclusions[report.nodeid] = {"status": status, "reason": reason}

    # -- collection errors ------------------------------------------------
    def pytest_collectreport(self, report):
        if report.outcome == "failed":
            self.conclusions[report.nodeid] = {
                "status": "collect-error",
                "reason": _first_line(report.longreprtext),
            }

    def pytest_sessionfinish(self, session, exitstatus):
        payload = {
            "exit_status": int(exitstatus),
            "collected": sorted(self.collected),
            "conclusions": self.conclusions,
            "durations": self.durations,
        }
        _write_atomic(self.path, payload)


#: Which phase outcome wins when a nodeid reports more than once.
_SEVERITY = {
    "passed": 0,
    "xpassed": 1,
    "skipped": 2,
    "xfailed": 3,
    "failed": 4,
    "error": 5,
    "collect-error": 6,
}


def _first_line(text):
    if not text:
        return ""
    return text.strip().splitlines()[0][:300]


def _status_of(report):
    """The conclusion this report carries, or ``None`` if it carries none.

    A passing setup/teardown is not a conclusion; a *failing* one is, and it is
    an error rather than a failure -- the test never ran.
    """
    if report.when in ("setup", "teardown"):
        if report.failed:
            return "error", _first_line(report.longreprtext)
        if report.skipped and report.when == "setup":
            return "skipped", _skip_reason(report)
        return None, ""
    if report.passed:
        return ("xpassed" if getattr(report, "wasxfail", None) is not None
                else "passed"), ""
    if report.skipped:
        if getattr(report, "wasxfail", None) is not None:
            return "xfailed", str(report.wasxfail)[:300]
        return "skipped", _skip_reason(report)
    return "failed", _first_line(report.longreprtext)


def _skip_reason(report):
    """The recorded reason, not just "skipped".

    "Skipped for a different reason than last time" is a changed conclusion:
    that is how a gate gets faster by quietly excluding more (0.15's red line).
    """
    longrepr = getattr(report, "longrepr", None)
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        return str(longrepr[2])[:300]
    return _first_line(getattr(report, "longreprtext", "") or "")


def _write_atomic(path, payload):
    """Write via a temporary file in the same directory, then ``os.replace``.

    A reader that finds a half-written record cannot tell it from a short one,
    and this file is the evidence a gate change is judged on (9.20, 9.22).
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    temporary = os.path.join(
        directory, ".%s.%d.partial" % (os.path.basename(path), os.getpid()))
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def pytest_configure(config):
    path = os.environ.get(OUT_VARIABLE, "").strip()
    if not path:
        return
    # Under xdist only the controller writes: workers get their own
    # `pytest_configure`, and every logreport reaches the controller anyway.
    if getattr(config, "workerinput", None) is not None:
        return
    config.pluginmanager.register(_Recorder(path), "gate-conclusion-recorder")
