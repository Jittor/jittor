# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""The coverage wrapper's own contract, on a surface built for the purpose.

``api_coverage`` is a measuring instrument, and the two ways it can lie are
both silent. It can *drop* an entry point it cannot wrap, which shrinks the
denominator and flatters every number computed from it -- the failure the
module's own docstring is written against. And it can *change* what it
measures, which would make the diagnostic's result differ from the run it is
supposed to describe.

Neither is visible from a real session: a real manifest has no name whose
wrapping is expected to fail, so the accounting path that matters is never
taken. The surface here is a stub module built to contain exactly those cases
-- a class, a non-callable, a callable module, an absent name, an owner that
does not resolve, and a slot that refuses assignment -- so the invariant can be
asserted as an equation: wrapped + unwrappable equals every name the manifest
declared.
"""

import json
import sys
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.structure

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _helpers import api_coverage  # noqa: E402


STUB_ROOT = "jittor_api_coverage_stub_root"

#: The manifest the stub surface declares: four wrappable names and six that
#: cannot be wrapped, for six different reasons.
STUB_MANIFEST = {
    "stub": ["plain", "boom", "published", "Klass", "not_callable",
             "callable_module", "absent"],
    "stub.sub": ["f"],
    "stub.no_such_owner": ["x"],
    "Frozen": ["upper"],
}
STUB_TOTAL = sum(len(v) for v in STUB_MANIFEST.values())


class _Klass:
    pass


class _CallableModule(types.ModuleType):
    def __call__(self):
        return "called"


def _build_stub_module():
    root = types.ModuleType(STUB_ROOT)
    root.plain = lambda value: value + 1
    def boom():
        raise ValueError("delegated")
    root.boom = boom
    root.Klass = _Klass
    root.not_callable = 3
    # ``torch.random`` has this exact shape: a module that also answers to a
    # call. Replacing it would take the published namespace out of its parent.
    root.callable_module = _CallableModule("stub.callable_module")
    # Published by composition from an owner module that still holds it -- the
    # shape the Torch frontend contracts on, and the one the wrapper has to
    # keep identical on both sides.
    owner = types.ModuleType(STUB_ROOT + ".owner")
    def published():
        return "published"
    published.__module__ = owner.__name__
    owner.published = published
    root.owner = owner
    root.published = published
    # A built-in type refuses attribute assignment, which is the "slot that
    # will not take a wrapper" case.
    root.frozen = str
    sub = types.ModuleType(STUB_ROOT + ".sub")
    sub.f = lambda: "f"
    root.sub = sub
    return root


@pytest.fixture
def stub_surface(tmp_path, monkeypatch):
    """A declared surface whose whole object graph is built here."""
    manifest = tmp_path / "stub_manifest.json"
    manifest.write_text(json.dumps(STUB_MANIFEST), encoding="utf-8")
    monkeypatch.setitem(api_coverage.SURFACES, "stub", {
        "module": STUB_ROOT,
        "alias": "stub",
        "extra_owners": {"Frozen": ("frozen",)},
        "manifest": manifest,
        "baseline": tmp_path / "stub_baseline.json",
    })
    root = _build_stub_module()
    monkeypatch.setitem(sys.modules, STUB_ROOT, root)
    # The owner module has to be reachable by name: that is how the wrapper
    # finds the second binding to move.
    monkeypatch.setitem(sys.modules, root.owner.__name__, root.owner)

    # The wrapper keeps its tallies in module-level sets, so a case has to
    # borrow them and give them back; leaving a stub name behind would show up
    # in whatever a later session reports.
    saved = {name: set(getattr(api_coverage, name))
             for name in ("CALLED", "WRAPPED", "UNWRAPPABLE")}
    saved_surface = api_coverage.SURFACE
    for name in saved:
        getattr(api_coverage, name).clear()
    try:
        yield sys.modules[STUB_ROOT]
    finally:
        for name, content in saved.items():
            recorded = getattr(api_coverage, name)
            recorded.clear()
            recorded.update(content)
        api_coverage.SURFACE = saved_surface


def test_nothing_declared_leaves_the_accounting(stub_surface):
    """wrapped + unwrappable == every name the manifest declared.

    A name dropped here would be a name missing from the denominator, and a
    smaller denominator reads as better coverage than the run achieved.
    """
    wrapped, unwrappable = api_coverage.install("stub")
    assert wrapped + unwrappable == STUB_TOTAL
    assert api_coverage.WRAPPED == {"stub.plain", "stub.boom", "stub.published",
                                    "stub.sub.f"}
    assert api_coverage.UNWRAPPABLE == {
        "stub.Klass",            # wrapping it would replace the type
        "stub.not_callable",     # not an entry point this instrument can fire
        "stub.callable_module",  # a namespace, whatever else it answers to
        "stub.absent",           # declared but gone
        "stub.no_such_owner.x",  # the owner itself does not resolve
        "Frozen.upper",          # the slot refuses assignment
    }


def test_the_wrapper_delegates_without_changing_the_result(stub_surface):
    api_coverage.install("stub")
    assert stub_surface.plain(1) == 2
    assert "stub.plain" in api_coverage.CALLED
    # The callable module keeps its identity as well as its behaviour.
    assert stub_surface.callable_module is sys.modules[STUB_ROOT].callable_module
    assert stub_surface.callable_module() == "called"


def test_the_wrapper_does_not_swallow_an_exception(stub_surface):
    api_coverage.install("stub")
    with pytest.raises(ValueError, match="delegated"):
        stub_surface.boom()
    assert "stub.boom" in api_coverage.CALLED


def test_an_uncalled_entry_point_is_reported_as_uncalled(stub_surface):
    api_coverage.install("stub")
    stub_surface.plain(1)
    report = api_coverage.report()
    assert report["surface"] == "stub"
    assert report["called"] == 1
    assert report["wrapped"] == 4
    assert sorted(report["uncalled_names"]) == [
        "stub.boom", "stub.published", "stub.sub.f"]


def test_the_wrapper_is_visible_to_an_identity_contract(stub_surface):
    """The one thing this instrument cannot hide, pinned rather than hoped about.

    The wrapper replaces the published binding and nothing else, so a caller
    that holds the object somewhere else -- an owner module, an object-keyed
    registry -- can tell. On the Torch surface 86 fidelity cases do exactly
    that, and they are listed in ``api_coverage.IDENTITY_CONTRACT_FILES``.

    Two ways of hiding were tried and both made it worse: rebinding the
    defining module took the count to 101, rebinding every alias to 97, with
    the failures moving from ``assertIs`` to the fidelity registry, which is
    keyed by the object. The conclusion is a property of mutation, not a bug to
    retry: this case exists so that if someone teaches the wrapper to preserve
    identity, this assertion is where they find out it worked, and the
    exclusion list can then go.
    """
    api_coverage.install("stub")
    owner = sys.modules[STUB_ROOT + ".owner"]
    assert stub_surface.published is not owner.published
    # Behaviour is identical either way; only identity differs.
    assert stub_surface.published() == owner.published() == "published"
    assert "stub.published" in api_coverage.CALLED


def test_installing_twice_does_not_double_wrap(stub_surface):
    api_coverage.install("stub")
    first = stub_surface.plain
    api_coverage.install("stub")
    assert stub_surface.plain is first
    assert stub_surface.plain(1) == 2


def test_a_key_from_another_surface_resolves_to_nothing(stub_surface):
    # The prefix is what says which surface a key belongs to. Resolving
    # ``jt.nn`` against the Torch root would wrap whatever happened to answer
    # to those attribute names, and record it under a name nothing else uses.
    assert api_coverage._owner("jt.nn", "stub", stub_surface) is None
    assert api_coverage._owner("stub.sub", "stub", stub_surface) is stub_surface.sub


@pytest.mark.parametrize("surface", sorted(
    name for name in api_coverage.SURFACES if name != "stub"))
def test_each_declared_manifest_stays_under_its_own_prefix(surface):
    """A manifest key must belong to the surface whose file it sits in.

    This is the static half of the same rule: a stray ``jt.*`` key in the Torch
    manifest would silently resolve to nothing and go straight into the
    unwrappable bucket, where it would look like an entry point out of reach
    rather than a mis-filed one.
    """
    spec = api_coverage.declaration(surface)
    manifest = json.loads(spec["manifest"].read_text(encoding="utf-8"))
    assert manifest, "%s manifest is empty" % surface
    allowed = set(spec["extra_owners"])
    prefix = spec["alias"] + "."
    stray = sorted(key for key in manifest
                   if key not in allowed and key != spec["alias"]
                   and not key.startswith(prefix))
    assert stray == [], "%s manifest holds foreign keys: %s" % (surface, stray)


def test_the_files_the_wrapper_disturbs_are_named_and_still_exist():
    """The exclusion list has to keep pointing at real files.

    It is the record of where this instrument is visible to its subject. A
    renamed or deleted file would leave a line that excludes nothing, and the
    next coverage run would quietly go back to reporting 86 failures the
    comparison is supposed to have accounted for.
    """
    root = Path(__file__).resolve().parents[2]
    missing = [name for name in api_coverage.IDENTITY_CONTRACT_FILES
               if not (root / name).is_file()]
    assert missing == [], (
        "api_coverage.IDENTITY_CONTRACT_FILES names files that are gone: %s. "
        "Re-take the on/off comparison and rewrite the list from what it "
        "reports." % missing)
    assert len(set(api_coverage.IDENTITY_CONTRACT_FILES)) == \
        len(api_coverage.IDENTITY_CONTRACT_FILES)


def test_an_unknown_surface_is_named_rather_than_a_bare_key_error():
    with pytest.raises(KeyError, match="unknown API surface"):
        api_coverage.declaration("no_such_surface")
