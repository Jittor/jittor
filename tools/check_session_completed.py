#!/usr/bin/env python3
"""Fail when a test session died instead of finishing.

A crash inside a native kernel does not fail the test that caused it -- it ends
the interpreter. pytest prints the nodeid it was about to run and the process is
gone: no result line, no traceback, no summary, and every test after it never
runs. The log simply stops. Nothing in it says "failed", so a reader who scans
for failures finds none, and a script that greps for them agrees.

This is the check that turns that absence into a finding. Preference order:

1. the sentinel written by ``tests/_helpers/session_completion.py`` -- positive
   evidence that ``pytest_sessionfinish`` ran;
2. the marker line in the captured log, same evidence, weaker medium;
3. a pytest summary line, for logs taken before the plugin existed.

Exit code 0 only when the session is shown to have completed. A missing sentinel
is *not* treated as "probably fine": the entire point is that the failure mode
looks like nothing at all.

Usage::

    python tools/check_session_completed.py --log <pytest.log>
    python tools/check_session_completed.py --sentinel <sentinel.json>
    python tools/check_session_completed.py --log <pytest.log> --expect-collected 8318
"""

import argparse
import json
import pathlib
import re
import sys


MARKER = "JITTOR-SESSION-COMPLETE"

#: What a finished pytest writes. Any of these means the run reached its end.
_SUMMARY = re.compile(
    r"(=+\s*(no tests ran|.*\b\d+ (passed|failed|error|errors|skipped|xfailed|xpassed)\b).*)"
    r"|(^\d+ (passed|failed|error|skipped|xfailed|xpassed))",
    re.IGNORECASE | re.MULTILINE)

#: The last nodeid pytest printed before it stopped, for the failure message.
_NODEID = re.compile(r"^(\S+::\S+)", re.MULTILINE)


def evidence(log_text, sentinel):
    """Return ``(completed, how, detail)``."""
    if sentinel is not None:
        data = json.loads(sentinel.read_text(encoding="utf-8"))
        if data.get("marker") == MARKER:
            return True, "sentinel", data
        return False, "sentinel", data
    if log_text is None:
        return False, "nothing", {}
    if MARKER in log_text:
        return True, "marker", {}
    if _SUMMARY.search(log_text):
        return True, "summary", {}
    return False, "none", {}


def last_nodeid(log_text):
    if not log_text:
        return None
    found = _NODEID.findall(log_text)
    return found[-1] if found else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", help="captured pytest output")
    parser.add_argument("--sentinel", help="JSON written by JITTOR_SESSION_SENTINEL")
    parser.add_argument("--expect-collected", type=int, default=None,
                        help="fail if the session collected fewer than this")
    args = parser.parse_args(argv)

    if not args.log and not args.sentinel:
        parser.error("give --log or --sentinel")

    log_text = None
    if args.log:
        path = pathlib.Path(args.log)
        if not path.is_file():
            print("no log at %s" % path)
            return 2
        log_text = path.read_text(encoding="utf-8", errors="replace")

    sentinel = pathlib.Path(args.sentinel) if args.sentinel else None
    if sentinel is not None and not sentinel.is_file():
        print("FAIL: the session never wrote its completion sentinel (%s).\n"
              "      That is what a process death looks like: the run stopped "
              "without reaching pytest_sessionfinish, so the tests after it "
              "never ran and nothing recorded a failure." % sentinel)
        node = last_nodeid(log_text)
        if node:
            print("      last nodeid printed: %s" % node)
        return 1

    completed, how, detail = evidence(log_text, sentinel)
    if not completed:
        print("FAIL: no evidence this session reached its end.\n"
              "      A completed run writes a summary; this log just stops. "
              "A native crash ends the interpreter without failing any test, "
              "so the absence is the finding.")
        node = last_nodeid(log_text)
        if node:
            print("      last nodeid printed: %s" % node)
        return 1

    collected = detail.get("collected")
    if args.expect_collected is not None and collected is not None:
        if collected < args.expect_collected:
            print("FAIL: session finished but collected %d of an expected %d; "
                  "a run that quietly collects less is the same loss of "
                  "coverage by another route."
                  % (collected, args.expect_collected))
            return 1

    print("session completed (evidence: %s)%s"
          % (how, "" if collected is None else ", collected=%d" % collected))
    return 0


if __name__ == "__main__":
    sys.exit(main())
