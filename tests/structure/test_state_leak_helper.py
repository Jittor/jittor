import pytest

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


def test_the_startup_watch_covers_every_frozen_flag():
    """A watch list that names a subset would go quiet on the rest.

    The list is derived rather than written out, so a flag added to the freeze
    is watched without anyone remembering to come back here.
    """
    import jittor  # noqa: F401  -- the snapshot only reads imported modules
    from jittor._runtime.flag_policy import STARTUP_FLAGS
    import jittor.compiler as compiler

    watched = state_leaks._startup_config()
    expected = {name for name in STARTUP_FLAGS if hasattr(compiler, name)}
    unwatched = {name for name in expected
                 if not isinstance(getattr(compiler, name), (list, tuple))}
    assert unwatched <= set(watched), sorted(unwatched - set(watched))
    assert watched["jittor_path"] == compiler.jittor_path


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
