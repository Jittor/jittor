# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""The untested-surface baselines, and the rule that they may only shrink.

Measuring which public entry points a run exercises is worth nothing on its
own: a number nobody compares is a number nobody checks. ``api_coverage_ratchet``
is the comparison, and this file is what keeps the comparison honest -- each
baseline has to stay in step with its manifest, and the rule has to be shown to
fail on the thing it exists to catch.

The comparison itself needs a full coverage run, which is a nightly-class job,
not a PR gate. What runs here is only what can be checked without one: every
declared surface has a baseline, the baseline parses and stays inside its own
surface, every name in it is still a public name, and the ratchet reports a
violation when handed one.

The rule cases build their own baseline out of manifest names instead of
reading the checked-in one. They are testing ``compare``, not the data: a rule
case that depended on today's recorded set would start passing or failing for
reasons that have nothing to do with the rule.

Every declared surface, and no skipping
---------------------------------------
The surfaces come from ``api_coverage.SURFACES`` rather than a list written
here, so a surface added there cannot quietly go unchecked. A declared surface
whose baseline file is missing **fails**. It is tempting to skip instead, and
that is exactly the failure this repository keeps paying for: an entry that
only ever skips is indistinguishable from one that passes -- four device-method
cases sat behind a two-GPU precondition and never executed on a one-GPU box, a
guard scanned a directory a layout move had deleted so its counter was
permanently zero, and an allocator contract failed on a stale path without ever
having checked a runtime behaviour. Nor would a skip be honest here: a surface
is declared in code, so a missing baseline is a forgotten step or a bad
configuration, never something this machine lacks.
"""

import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.structure

REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(REPO_ROOT / "tools"))
import api_coverage_ratchet as ratchet  # noqa: E402

SURFACES = sorted(ratchet.SURFACES)


def _require_baseline(surface):
    """The parsed baseline, or an actionable failure naming what to run."""
    path = ratchet.baseline_path(surface)
    assert path.is_file(), (
        "the %s API surface is declared in tests/_helpers/api_coverage.py but has "
        "no baseline at %s. Take a coverage run in the session that owns the "
        "surface and record it:\n  %s\n  python tools/api_coverage_ratchet.py "
        "--report <path> --surface %s --update"
        % (surface, path.relative_to(REPO_ROOT), ratchet.COMMANDS[surface], surface))
    return json.loads(path.read_text(encoding="utf-8"))


def _synthetic_baseline(surface, size=3):
    """A recorded set built from the manifest, for the rule cases."""
    known = ratchet.manifest_names(surface)
    recorded = sorted(known)[:size]
    assert len(recorded) == size, "the %s manifest is too small to test the rule" % surface
    return known, recorded


@pytest.mark.parametrize("surface", SURFACES)
def test_the_baseline_exists_and_parses(surface):
    data = _require_baseline(surface)
    assert "uncovered" in data
    assert "unwrappable" in data


@pytest.mark.parametrize("surface", SURFACES)
def test_the_lists_are_sorted_and_deduplicated(surface):
    # A set written in a stable order is what makes the diff of a coverage
    # change readable; an unordered dump would show the whole file moving.
    data = _require_baseline(surface)
    for key in ("uncovered", "unwrappable"):
        names = data[key]
        assert names == sorted(names), "%s %s is not sorted" % (surface, key)
        assert len(names) == len(set(names)), "%s %s has duplicates" % (surface, key)


@pytest.mark.parametrize("surface", SURFACES)
def test_every_recorded_name_is_still_public(surface):
    # A baseline that drifts from its manifest stops meaning anything: a name
    # that left the API is not a coverage debt, and one that the baseline never
    # heard of would slip past the ratchet.
    data = _require_baseline(surface)
    unknown = sorted(set(data["uncovered"]) - ratchet.manifest_names(surface))
    assert unknown == [], (
        "the %s baseline lists names that are not in %s; regenerate with "
        "tools/api_coverage_ratchet.py --surface %s --update"
        % (surface, ratchet.manifest_path(surface).relative_to(REPO_ROOT), surface))


@pytest.mark.parametrize("surface", SURFACES)
def test_the_two_lists_do_not_overlap(surface):
    # An entry point is either measurable and untested, or unreachable by the
    # wrapper. Counting one in both would let it be dismissed twice.
    data = _require_baseline(surface)
    assert sorted(set(data["uncovered"]) & set(data["unwrappable"])) == []


@pytest.mark.parametrize("surface", SURFACES)
def test_the_baseline_stays_inside_its_own_surface(surface):
    # The surfaces name different object graphs. A native name in the Torch
    # baseline (or the reverse) would mean an update ran against the wrong
    # manifest, and the ratchet would then police a set nothing measures.
    data = _require_baseline(surface)
    assert data.get("_surface", surface) == surface
    mine = ratchet.manifest_names(surface)
    others = set()
    for name in SURFACES:
        if name != surface:
            others |= ratchet.manifest_names(name)
    strays = sorted((set(data["uncovered"]) | set(data["unwrappable"])) & (others - mine))
    assert strays == [], (
        "the %s baseline holds names that only exist on another surface: %s"
        % (surface, strays[:10]))


@pytest.mark.parametrize("surface", SURFACES)
def test_a_newly_untested_entry_point_is_a_violation(surface):
    """The rule must fail on the case it exists to catch."""
    known, recorded = _synthetic_baseline(surface)
    fresh = sorted(known - set(recorded))[0]
    report = {"uncalled_names": sorted(set(recorded) | {fresh})}
    added, _removed, _stale = ratchet.compare(report, {"uncovered": recorded}, known)
    assert added == [fresh]


@pytest.mark.parametrize("surface", SURFACES)
def test_covering_a_recorded_entry_point_is_progress_not_a_violation(surface):
    known, recorded = _synthetic_baseline(surface)
    report = {"uncalled_names": recorded[1:]}
    added, removed, _stale = ratchet.compare(report, {"uncovered": recorded}, known)
    assert added == []
    assert removed == [recorded[0]]


@pytest.mark.parametrize("surface", SURFACES)
def test_a_name_that_left_the_api_is_dropped_not_blamed(surface):
    known, recorded = _synthetic_baseline(surface)
    departed = "%s.gone_forever" % surface
    baseline = {"uncovered": recorded + [departed]}
    added, _removed, stale = ratchet.compare({"uncalled_names": recorded}, baseline, known)
    assert added == []
    assert stale == [departed]


@pytest.mark.parametrize("surface", SURFACES)
def test_updating_one_surface_leaves_the_other_baselines_alone(surface, tmp_path, monkeypatch):
    """``--update`` writes one file. The other surfaces must be byte-identical.

    Worth its own case because the failure is silent: a shared write path would
    rewrite the other baseline from a report that never measured it, emptying
    the set that surface's ratchet compares against, and every later run would
    pass while policing nothing.
    """
    for name in SURFACES:
        target = tmp_path / ("%s.json" % name)
        target.write_text(
            json.dumps({"uncovered": ["%s.sentinel" % name], "unwrappable": []}) + "\n",
            encoding="utf-8")
        monkeypatch.setitem(ratchet.SURFACES[name], "baseline", target)
    before = {name: (tmp_path / ("%s.json" % name)).read_bytes()
              for name in SURFACES if name != surface}

    uncalled = sorted(ratchet.manifest_names(surface))[:3]
    report = tmp_path / "report.json"
    report.write_text(json.dumps({
        "surface": surface, "called": 1, "wrapped": 2,
        "uncalled_names": uncalled, "unwrappable_names": [],
    }), encoding="utf-8")
    assert ratchet.main(["--report", str(report), "--update"]) == 0

    written = json.loads((tmp_path / ("%s.json" % surface)).read_text(encoding="utf-8"))
    assert written["uncovered"] == uncalled
    for name, content in before.items():
        assert (tmp_path / ("%s.json" % name)).read_bytes() == content, (
            "updating the %s baseline rewrote the %s one" % (surface, name))


@pytest.mark.parametrize("surface", SURFACES)
def test_parts_of_one_session_merge_by_union_of_what_was_called(surface):
    """A name is uncovered only when no part called it."""
    known, recorded = _synthetic_baseline(surface, size=4)
    common = {"surface": surface, "wrapped": 4, "unwrappable_names": []}
    first = dict(common, uncalled_names=recorded[:3])   # part one called the last
    second = dict(common, uncalled_names=recorded[1:])  # part two called the first
    merged = ratchet.merge_reports([first, second])
    assert merged["uncalled_names"] == recorded[1:3]
    assert merged["called"] == 2


def test_parts_that_measured_different_denominators_are_refused():
    # Merging reports whose wrapped sets disagree would produce a denominator
    # neither run had, and the uncovered set would be missing whatever the
    # smaller run never wrapped.
    first = {"surface": SURFACES[0], "wrapped": 10, "unwrappable_names": [],
             "uncalled_names": []}
    second = dict(first, wrapped=9)
    with pytest.raises(SystemExit):
        ratchet.merge_reports([first, second])


def test_a_report_is_refused_against_the_wrong_surface():
    # Comparing one surface's report against another's baseline would report
    # every name as newly untested and every recorded name as progress.
    assert len(SURFACES) > 1, "a single surface cannot exercise the mismatch rule"
    with pytest.raises(SystemExit):
        ratchet.resolve_surface({"surface": SURFACES[0]}, SURFACES[1])


def test_a_report_without_a_surface_is_refused_rather_than_guessed():
    with pytest.raises(SystemExit):
        ratchet.resolve_surface({}, None)
