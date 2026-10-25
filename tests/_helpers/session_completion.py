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

import pytest


#: Printed on the terminal even when no sentinel path is configured, so a log
#: kept by hand still carries the evidence.
MARKER = "JITTOR-SESSION-COMPLETE"

_STATE = {"collected": 0, "executed": 0}

#: Where an xdist worker hands its collected count to the controller. The
#: controller collects no items itself, so ``pytest_collection_finish`` there
#: would record zero and the marker would say ``collected=0 executed=N``.
_XDIST_COLLECTED_KEY = "jittor_session_collected"


def _xdist_worker_id(config):
    return getattr(config, "workerinput", {}).get("workerid")


def _xdist_controller(config):
    return (getattr(config.option, "numprocesses", None) not in (None, 0)
            and _xdist_worker_id(config) is None)


def sentinel_path():
    value = os.environ.get("JITTOR_SESSION_SENTINEL", "").strip()
    return pathlib.Path(value) if value else None


def record_collected(count):
    _STATE["collected"] = int(count)


def record_executed():
    _STATE["executed"] += 1


def summary():
    return dict(_STATE)


def restore(state):
    """Put the counters back, for callers that had to disturb them.

    The plugin is registered for the same session its own tests run in, so a
    test that writes a count is writing the live session's count. Restoring is
    what keeps the sentinel describing the run rather than the last test that
    touched it.
    """
    _STATE.update(state)


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
    count = len(session.items)
    if _xdist_worker_id(session.config) is not None:
        # Collection happens in the workers; the controller only receives the
        # reports. Publish this worker's share so the controller can sum it.
        record_collected(count)
        workeroutput = getattr(session.config, "workeroutput", None)
        if workeroutput is not None:
            workeroutput[_XDIST_COLLECTED_KEY] = count
        return
    if _xdist_controller(session.config):
        # Nothing was collected here; ``pytest_testnodedown`` fills it in
        # before the marker is printed.
        return
    record_collected(count)


def pytest_runtest_logreport(report):
    # A test reaches a terminal outcome at its call phase, or at setup when it
    # was skipped or errored before the body ran. Counting only ``call`` made a
    # finished session with setup-level skips look like one that lost tests;
    # the counts are compared, so they have to mean the same thing.
    if report.when == "call" or (
            report.when == "setup" and (report.skipped or report.failed)):
        record_executed()


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error):
    count = (getattr(node, "workeroutput", None) or {}).get(
        _XDIST_COLLECTED_KEY)
    if count is not None:
        # Every xdist worker collects the same selection and then runs its
        # share, so summing would report N times the tree. The largest count is
        # the selection size; a worker that collected less saw a narrower one.
        record_collected(max(_STATE["collected"], int(count)))


def pytest_sessionfinish(session, exitstatus):
    # One sentinel describes one process. Under xdist only the controller has
    # the merged counts, and a worker writing the shared path would race the
    # others; a worker's disappearance is xdist's crash to report.
    if _xdist_worker_id(session.config) is not None:
        return
    write_sentinel(extra={"exitstatus": int(exitstatus)})


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    # Printed last so the marker is the final thing a truncated log would be
    # missing, which is what makes its absence readable at a glance.
    terminalreporter.write_line(marker_line())
