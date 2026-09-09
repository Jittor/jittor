# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Prove a test session reached its end, so a dead one cannot read as a clean one.

A native crash inside a test does not fail that test. It takes the interpreter
with it: pytest prints the nodeid it was about to run and the process is gone --
no result line, no traceback, no summary, and the tests after it never run. A
Torch session was observed disappearing at 48% this way, and the log showed no
failure anywhere, because there was nothing left alive to write one.

That is worse than the skip problem this repository already knows about. A skip
at least occupies a line in the summary. A truncated session produces a log that
*ends*, and a reader has to notice an absence to catch it.

So the session states its own completion. ``pytest_sessionfinish`` runs after the
last test whether the run passed, failed, or was interrupted by pytest itself --
but not if the process died, which is exactly the discrimination wanted. The
sentinel carries the collected and executed counts as well, so a session that
finished while silently collecting fewer tests than expected is also visible.

``tools/check_session_completed.py`` is the consumer. It also accepts a bare log
for runs taken before this plugin existed: a completed pytest always writes a
summary line, so its absence is the same finding by weaker evidence.
"""

import json
import os
import pathlib


#: Printed on the terminal even when no sentinel path is configured, so a log
#: kept by hand still carries the evidence.
MARKER = "JITTOR-SESSION-COMPLETE"

_STATE = {"collected": 0, "executed": 0}


def sentinel_path():
    value = os.environ.get("JITTOR_SESSION_SENTINEL", "").strip()
    return pathlib.Path(value) if value else None


def record_collected(count):
    _STATE["collected"] = int(count)


def record_executed():
    _STATE["executed"] += 1


def summary():
    return dict(_STATE)


def write_sentinel(path=None, extra=None):
    """Record that this session reached its end. Returns the path, or None."""
    path = path or sentinel_path()
    if path is None:
        return None
    payload = summary()
    if extra:
        payload.update(extra)
    payload["marker"] = MARKER
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


def marker_line():
    state = summary()
    return "%s collected=%d executed=%d" % (
        MARKER, state["collected"], state["executed"])


# -- the pytest side ---------------------------------------------------------

def pytest_collection_finish(session):
    record_collected(len(session.items))


def pytest_runtest_logreport(report):
    if report.when == "call":
        record_executed()


def pytest_sessionfinish(session, exitstatus):
    write_sentinel(extra={"exitstatus": int(exitstatus)})


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    # Printed last so the marker is the final thing a truncated log would be
    # missing, which is what makes its absence readable at a glance.
    terminalreporter.write_line(marker_line())
