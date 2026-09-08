import pytest
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace

from _helpers import state_leaks
from _helpers.state_leaks import assert_rss_growth_bounded


def _snapshot(startup):
    """A snapshot with only the field under test populated."""
    return {"counters": {}, "flags": {}, "startup": startup,
            "autograd_policy": None, "caches": {}, "modules": {}}


def test_a_changed_frozen_startup_path_is_reported():
    """The survey must name the file that corrupted the startup config.

    Driven with synthetic snapshots rather than by corrupting the real module:
    the only ways to actually change a frozen attribute are to bypass the
    freeze or to lose a race with a restore path, and a test that did either
    would be the leak it is testing for.
    """
    before = _snapshot({"jittor_path": "/checkout/python/jittor"})
    after = _snapshot({"jittor_path": "/tmp/deleted-by-a-test"})
    report = state_leaks.differences(before, after)
    assert len(report) == 1, report
    assert "compiler.jittor_path" in report[0]
    assert "/tmp/deleted-by-a-test" in report[0]
    assert "frozen" in report[0]


def test_an_unchanged_startup_config_is_not_reported():
    """Otherwise every file in the tree reports a leak and the survey is noise."""
    same = {"jittor_path": "/checkout/python/jittor", "cc_type": "g++"}
    assert state_leaks.differences(_snapshot(same), _snapshot(dict(same))) == []


def test_the_startup_watch_covers_every_frozen_flag(monkeypatch):
    """A watch list that names a subset would go quiet on the rest.

    The list is derived rather than written out, so a flag added to the freeze
    is watched without anyone remembering to come back here.
    """
    source = Path(__file__).resolve().parents[2] / "python/jittor/_runtime/flag_policy.py"
    STARTUP_FLAGS = runpy.run_path(str(source))["STARTUP_FLAGS"]
    compiler = ModuleType("jittor.compiler")
    for name in STARTUP_FLAGS:
        setattr(compiler, name, [80, 89] if name == "cuda_archs" else "configured")
    monkeypatch.setitem(sys.modules, "jittor.compiler", compiler)
    monkeypatch.setitem(sys.modules, "jittor._runtime.flag_policy", SimpleNamespace(STARTUP_FLAGS=STARTUP_FLAGS))

    watched = state_leaks._startup_config()
    expected = {name for name in STARTUP_FLAGS if hasattr(compiler, name)}
    unwatched = {name for name in expected
                 if not isinstance(getattr(compiler, name), (list, tuple))}
    assert unwatched <= set(watched), sorted(unwatched - set(watched))
    assert watched["jittor_path"] == compiler.jittor_path


def test_startup_watch_detects_module_bypass_not_just_correct_config(monkeypatch):
    class FrozenCompiler(ModuleType):
        def __setattr__(self, name, value):
            raise AttributeError("frozen")
    compiler = FrozenCompiler("jittor.compiler")
    ModuleType.__setattr__(compiler, "jittor_path", "/good")
    readonly_config = SimpleNamespace(jittor_path="/good")
    monkeypatch.setitem(sys.modules, "jittor.compiler", compiler)
    monkeypatch.setitem(sys.modules, "jittor._runtime.flag_policy",
                        SimpleNamespace(STARTUP_FLAGS={"jittor_path"}))
    before = _snapshot(state_leaks._startup_config())
    ModuleType.__setattr__(compiler, "jittor_path", "/corrupt")
    assert readonly_config.jittor_path == "/good"
    after = _snapshot(state_leaks._startup_config())
    assert "compiler.jittor_path" in state_leaks.differences(before, after)[0]


def test_snapshot_uses_public_observations_and_preserves_report_keys(monkeypatch):
    values = {name: 0 for name in state_leaks.WATCHED_FLAGS}
    counters = SimpleNamespace(held_vars=2, live_vars=3, live_ops=1)
    owner = SimpleNamespace(introspection=SimpleNamespace(
        counters=counters, policy=SimpleNamespace(runtime=SimpleNamespace(snapshot=lambda: dict(values)))))
    monkeypatch.setitem(sys.modules, "jittor", owner)
    before = state_leaks.snapshot(collect=False)
    assert before["counters"] == {"number_of_hold_vars": 2, "number_of_lived_vars": 3, "number_of_lived_ops": 1}
    values["use_cuda"] = 1
    counters.live_vars = 4
    after = state_leaks.snapshot(collect=False)
    assert before["flags"]["use_cuda"] == 0
    assert after["flags"]["use_cuda"] == 1
    report = state_leaks.differences(before, after)
    assert any("number_of_lived_vars 3 -> 4" in line for line in report)
    assert any("flags.use_cuda 0 -> 1" in line for line in report)


def test_snapshot_does_not_hide_an_observation_failure(monkeypatch):
    class Counters:
        held_vars = 0
        @property
        def live_vars(self):
            raise RuntimeError("counter probe failed")
    owner = SimpleNamespace(introspection=SimpleNamespace(counters=Counters()))
    monkeypatch.setitem(sys.modules, "jittor", owner)
    with pytest.raises(RuntimeError, match="counter probe failed"):
        state_leaks.snapshot(collect=False)


def test_snapshot_without_a_runtime_never_imports_jittor(monkeypatch):
    import builtins
    original = builtins.__import__
    monkeypatch.delitem(sys.modules, "jittor", raising=False)
    def guarded(name, *args, **kwargs):
        assert name.split(".", 1)[0] != "jittor", "unexpected native bootstrap"
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    assert state_leaks.snapshot() is None


def test_snapshot_refuses_a_missing_service_after_initialization(monkeypatch):
    owner = SimpleNamespace(runtime=object())
    monkeypatch.setitem(sys.modules, "jittor", owner)
    with pytest.raises(RuntimeError, match="no public introspection"):
        state_leaks.snapshot(collect=False)
    owner.__spec__ = SimpleNamespace(_initializing=True)
    assert state_leaks.snapshot(collect=False) is None


def test_rss_bound_rejects_an_intentional_retained_allocation():
    retained = []

    def leak_one_mebibyte():
        retained.append(bytearray(1 << 20))

    with pytest.raises(AssertionError, match="RSS grew"):
        assert_rss_growth_bounded(
            leak_one_mebibyte,
            warmup=0,
            iterations=8,
            max_growth_bytes=4 << 20,
        )
