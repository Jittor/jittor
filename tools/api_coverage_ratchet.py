#!/usr/bin/env python3
"""Hold the untested public surface to a set that may only shrink.

``tests/_helpers/api_coverage.py`` says which public entry points a run
actually called. That number is only useful if something acts on it: a
measurement nobody compares is a number nobody checks.

This is the comparison. The checked-in baseline lists the entry points that
were *not* exercised when it was taken. A later run may cover more of them --
that is the point -- but it may not add new ones. Adding a public API without a
test therefore fails here instead of being noticed years later, which is how
`bitwise_not` kept a Critical bool defect while reading as covered.

Two rules, and the second is the one that makes it a ratchet rather than a
report:

1. every uncovered name must already be in the baseline;
2. names the baseline lists but the run covered are *progress* -- reported, and
   the baseline should be rewritten with ``--update`` so the gap cannot drift
   back open.

A name that has left the manifest entirely is dropped rather than treated as a
violation: deleting an API is not a coverage regression.

One baseline per surface
------------------------
Native and Torch mode publish different object graphs and run in different
processes, so each has its own manifest and its own baseline, and a run
measures exactly one of them. The surface comes from the coverage report itself
unless ``--surface`` names it, and ``--update`` rewrites *that* surface only:
regenerating one baseline must not empty the other, which it would if the two
shared a file or if the update path wrote whatever the report happened to hold.

Usage::

    # after a coverage run
    python tools/api_coverage_ratchet.py --report <coverage.json>
    python tools/api_coverage_ratchet.py --report <coverage.json> --update
    python tools/api_coverage_ratchet.py --report <coverage.json> --surface torch
    python tools/api_coverage_ratchet.py --report part1.json part2.json --update
"""

import argparse
import json
import pathlib
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# The surface declaration lives with the wrapper that produces the reports, so
# the tool, the gate and the measurement read one copy of the manifest and
# baseline paths instead of three that can disagree. The path entry is taken
# back afterwards: a command-line tool that leaves ``tests/`` on ``sys.path``
# changes what its caller's later imports resolve to.
sys.path.insert(0, str(REPO_ROOT / "tests"))
try:
    from _helpers.api_coverage import SURFACES  # noqa: E402
finally:
    sys.path.remove(str(REPO_ROOT / "tests"))

#: How the baseline was produced, recorded in the file so a later maintainer
#: can reproduce it rather than guess which session the numbers came from.
COMMANDS = {
    "native": "JITTOR_TORCH_SHIM=0 JITTOR_API_COVERAGE=1 "
              "JITTOR_API_COVERAGE_REPORT=<path> python -m pytest <native gate selection>",
    "torch": "JITTOR_TORCH_SHIM=1 JITTOR_API_COVERAGE=1 "
             "JITTOR_API_COVERAGE_REPORT=<path> python -m pytest <torch gate selection>",
}


def _display(path):
    """A repository-relative path when it is one, and the path when it is not.

    A baseline redirected somewhere else -- which is how the update-isolation
    test checks that one surface's rewrite leaves the others alone -- is not
    under the checkout, and reporting where the file went must not be the thing
    that raises.
    """
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError:
        return path


def manifest_path(surface):
    return SURFACES[surface]["manifest"]


def baseline_path(surface):
    return SURFACES[surface]["baseline"]


def manifest_names(surface):
    data = json.loads(manifest_path(surface).read_text(encoding="utf-8"))
    names = set()
    for module_key, entries in data.items():
        for name in entries:
            names.add("%s.%s" % (module_key, name))
    return names


def load_baseline(surface):
    """The recorded untested set, or an empty one before the first ``--update``.

    Absent is a real state only while a surface is being bootstrapped, and only
    for this tool: ``tests/structure/test_api_coverage_ratchet.py`` fails when a
    declared surface has no baseline file, because a declared surface without
    one is a missing artefact, not a fact about the machine.
    """
    path = baseline_path(surface)
    if not path.is_file():
        return {"uncovered": [], "unwrappable": []}
    return json.loads(path.read_text(encoding="utf-8"))


def write_baseline(surface, uncovered, unwrappable, note=None):
    """Rewrite one surface's baseline. The other surface's file is not touched."""
    path = baseline_path(surface)
    payload = {}
    if note:
        # What the run could not include. A baseline taken from a session that
        # was missing part of itself is still usable -- it can only be looser
        # than the truth, and the ratchet only tightens -- but the reader has
        # to be told, or the number reads as the whole session's result.
        payload["_taken_with"] = note
    payload.update({
        "_comment": (
            "%s public entry points no maintained run exercised, and the ones "
            "the coverage wrapper cannot reach. Regenerate with "
            "tools/api_coverage_ratchet.py --surface %s --update. This set may "
            "only shrink; adding a public API without a test grows it and fails "
            "the ratchet." % (surface.capitalize(), surface)
        ),
        "_surface": surface,
        "_produced_by": COMMANDS[surface],
        "uncovered": sorted(uncovered),
        "unwrappable": sorted(unwrappable),
    })
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def compare(report, baseline, known_names):
    """Return ``(added, removed, stale)`` against the baseline."""
    uncovered = {n for n in report.get("uncalled_names", []) if n in known_names}
    recorded = {n for n in baseline.get("uncovered", []) if n in known_names}
    stale = {n for n in baseline.get("uncovered", []) if n not in known_names}
    return sorted(uncovered - recorded), sorted(recorded - uncovered), sorted(stale)


def merge_reports(reports):
    """One report from several, for a session that cannot run in one process.

    The design is one session, one report. Reality here is that two Torch cases
    terminate the interpreter (KI-TEST-002), so the maintained selection has to
    be run in parts, and a part that never ran would leave its entry points
    looking untested. Merging is a union of what was *called*: an entry point is
    uncovered only when no part called it.

    The denominator is not merged, it is checked. Every part wraps the same
    manifest against the same surface, so ``wrapped`` and ``unwrappable`` must
    already agree; if they do not, the parts measured different things and
    silently picking one would invent a number.
    """
    reports = list(reports)
    if not reports:
        raise SystemExit("no coverage report given")
    if len(reports) == 1:
        return reports[0]
    surfaces = {report.get("surface") for report in reports}
    if len(surfaces) != 1:
        raise SystemExit("the reports measured different surfaces: %s"
                         % ", ".join(sorted(str(name) for name in surfaces)))
    wrapped = {report.get("wrapped") for report in reports}
    unwrappable = {tuple(sorted(report.get("unwrappable_names", []))) for report in reports}
    if len(wrapped) != 1 or len(unwrappable) != 1:
        raise SystemExit(
            "the reports disagree about what could be wrapped, so they did not "
            "measure the same surface; re-take them in the same configuration")
    uncalled = None
    for report in reports:
        names = set(report.get("uncalled_names", []))
        uncalled = names if uncalled is None else (uncalled & names)
    merged = dict(reports[0])
    merged["uncalled_names"] = sorted(uncalled)
    merged["uncalled"] = len(uncalled)
    merged["called"] = merged.get("wrapped", 0) - len(uncalled)
    return merged


def resolve_surface(report, requested):
    """Which surface this report describes.

    A report carries the surface its run measured. Comparing a Torch report
    against the native baseline would report every Torch name as newly untested
    and every native name as progress -- nonsense in both directions -- so a
    mismatch is refused rather than resolved by precedence.
    """
    recorded = report.get("surface")
    if requested is None:
        if recorded is None:
            raise SystemExit(
                "the report does not say which surface it measured; re-run the "
                "coverage session, or pass --surface {%s}"
                % ",".join(sorted(SURFACES)))
        if recorded not in SURFACES:
            raise SystemExit("the report names an unknown surface %r" % recorded)
        return recorded
    if recorded is not None and recorded != requested:
        raise SystemExit(
            "--surface %s was asked for but the report measured the %s surface"
            % (requested, recorded))
    return requested


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, nargs="+",
                        help="JSON written by JITTOR_API_COVERAGE_REPORT; give "
                             "several when the session had to be run in parts, "
                             "and an entry point counts as covered when any "
                             "part called it")
    parser.add_argument("--surface", choices=sorted(SURFACES), default=None,
                        help="which surface the report measured "
                             "(default: read from the report)")
    parser.add_argument("--update", action="store_true",
                        help="rewrite this surface's baseline from this report")
    parser.add_argument("--note", default=None,
                        help="what the run could not include, recorded in the "
                             "baseline so a looser set is not read as the whole "
                             "session's result")
    args = parser.parse_args(argv)

    report = merge_reports(
        json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        for path in args.report)
    surface = resolve_surface(report, args.surface)
    known = manifest_names(surface)
    baseline = load_baseline(surface)
    added, removed, stale = compare(report, baseline, known)

    print("%s surface: run covered %d of %d wrapped entry points" %
          (surface, report.get("called", 0), report.get("wrapped", 0)))
    if removed:
        print("progress: %d entry point(s) now exercised that the baseline "
              "listed as untested" % len(removed))
        for name in removed[:20]:
            print("  + %s" % name)
        if len(removed) > 20:
            print("  ... and %d more" % (len(removed) - 20))
    if stale:
        print("note: %d baseline entry/entries no longer in the manifest "
              "(dropped, not a regression)" % len(stale))

    if args.update:
        path = write_baseline(
            surface,
            {n for n in report.get("uncalled_names", []) if n in known},
            set(report.get("unwrappable_names", [])),
            note=args.note,
        )
        print("baseline rewritten:", _display(path))
        return 0

    if added:
        print("\nFAIL: %d public entry point(s) went untested and are not in the "
              "%s baseline:" % (len(added), surface))
        for name in added:
            print("  - %s" % name)
        print("\nAdd a test, or run --update with a reason if the coverage "
              "genuinely cannot be had.")
        return 1
    print("ratchet holds: nothing newly untested")
    return 0


if __name__ == "__main__":
    sys.exit(main())
