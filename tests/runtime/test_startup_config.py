"""Startup options freeze across every native Flags instance."""

import os
from unittest import mock

import numpy as np
import pytest

import jittor as jt
from jittor._runtime.flag_policy import (
    FLAG_ALIASES, READONLY_FLAGS, RUNTIME_FLAGS, STARTUP_FLAGS,
)


@pytest.mark.parametrize("name", sorted(STARTUP_FLAGS))
def test_all_native_flag_instances_reject_late_startup_writes(name):
    if not hasattr(jt.flags, name):
        pytest.skip("startup field unavailable in this backend build")
    before = getattr(jt.flags, name)
    for owner in (jt.flags, jt.compiler.flags, jt.core.Flags(), jt.Flags()):
        with pytest.raises(RuntimeError, match="immutable startup configuration"):
            setattr(owner, name, before)
        assert getattr(owner, name) == before
    with pytest.raises(AttributeError, match="startup configuration"):
        setattr(jt.runtime, name, before)
    with pytest.raises(AttributeError, match="immutable"):
        setattr(jt.config, name, before)
    if hasattr(jt.compiler, name):
        with pytest.raises(AttributeError, match="immutable startup configuration"):
            setattr(jt.compiler, name, getattr(jt.compiler, name))


@pytest.mark.parametrize("name", sorted(STARTUP_FLAGS))
def test_a_refused_startup_write_refuses_before_mutating(name):
    """The refusal must come *before* the assignment, not after it.

    ``test_all_native_flag_instances_reject_late_startup_writes`` above
    already asserts that the write raises, but it writes back the value that
    is already there -- so it passes just as well against an implementation
    that assigns first and raises afterwards. This one writes a *different*
    value, which is the only way the difference is observable.

    The difference matters because of what a half-done write costs. It is not
    a failure here: it is a ``jittor_path`` naming a deleted temporary
    directory for the rest of the session, surfacing as unrelated failures in
    whichever tests happen to be ordered after the one that tried the patch.
    A refusal that corrupts what it refuses to change is worse than no
    refusal, because the exception makes it look handled.
    """
    if not hasattr(jt.compiler, name):
        pytest.skip("startup field unavailable in this backend build")
    before = getattr(jt.compiler, name)
    probe = "/jittor-freeze-probe"
    assert before != probe
    with pytest.raises(AttributeError, match="immutable startup configuration"):
        setattr(jt.compiler, name, probe)
    assert getattr(jt.compiler, name) == before
    with pytest.raises(AttributeError, match="immutable startup configuration"):
        delattr(jt.compiler, name)
    assert getattr(jt.compiler, name) == before


@pytest.mark.parametrize("name", sorted(STARTUP_FLAGS))
def test_a_refused_patch_reports_from_its_own_rollback(name):
    """Where the confusing half of this lands: ``unittest.mock``.

    ``patch.object.__enter__`` assigns the replacement and, when that raises,
    calls its own ``__exit__`` to roll back -- which "restores" by writing the
    original value back, a write this module also refuses. So the error a
    patcher sees is raised from the rollback and names the *original* value,
    reading as though restoring were the forbidden part rather than patching.

    Pinned rather than fixed: the cleanup path failing is ``mock``'s, and the
    refusal is loud (the patching test goes red) and leaves nothing behind, as
    the case above establishes. Recorded here so the next person to read that
    traceback does not go looking for a bug in the restore path.
    """
    if not hasattr(jt.compiler, name):
        pytest.skip("startup field unavailable in this backend build")
    before = getattr(jt.compiler, name)
    with pytest.raises(AttributeError, match="immutable startup configuration"):
        with mock.patch.object(jt.compiler, name, "/jittor-freeze-probe"):
            pass
    assert getattr(jt.compiler, name) == before


def test_the_frozen_paths_still_point_at_this_checkout():
    """The observable end of the property above, stated in one place.

    Asserted separately from the parametrised case because this is what a
    later test actually trips over: not "the attribute changed" but "the
    directory it names is gone".
    """
    assert os.path.isdir(jt.compiler.jittor_path)
    assert os.path.isdir(os.path.join(jt.compiler.jittor_path, "src"))
    assert os.path.isdir(jt.compiler.cache_path)


def test_the_core_source_walk_can_be_pointed_elsewhere_without_patching():
    """Why no test needs to patch ``jittor_path`` in the first place.

    ``core_source_signature`` cannot be shown to notice a new or a same-size
    edited file by walking the real checkout, so it takes an explicit root.
    That parameter is the supported alternative to patching a frozen
    attribute; if it is ever dropped, the next person to need it reaches for
    ``mock.patch.object`` and gets the rollback behaviour described above.
    """
    signature = jt.compiler.core_source_signature(root=jt.compiler.jittor_path)
    assert signature == jt.compiler.core_source_signature()
    assert jt.compiler.core_source_signature(root=os.devnull) == {}


@pytest.mark.parametrize("name", sorted(READONLY_FLAGS))
def test_native_runtime_counters_cannot_be_assigned(name):
    before = getattr(jt.flags, name)
    with pytest.raises(RuntimeError, match="read-only runtime counter"):
        setattr(jt.core.Flags(), name, before)
    with pytest.raises(AttributeError, match="read-only runtime counter"):
        setattr(jt.runtime, name, before)


def test_runtime_scope_changes_native_execution_and_restores_after_error():
    before = jt.runtime.no_grad
    with pytest.raises(ValueError, match="scope exit"):
        with jt.runtime.scope(no_grad=True, use_cuda=0):
            assert jt.flags.no_grad
            result = jt.array([2.0, 3.0]).sqr()
            np.testing.assert_array_equal(result.numpy(), [4.0, 9.0])
            raise ValueError("scope exit")
    assert jt.runtime.no_grad == before
    with pytest.raises(AttributeError, match="startup configuration"):
        with jt.runtime.scope(no_grad=not before, cc_flags="invalid"):
            pass
    assert jt.flags.no_grad == before


def test_config_is_detached_and_every_native_flag_has_one_owner():
    assert not (STARTUP_FLAGS & RUNTIME_FLAGS or STARTUP_FLAGS & READONLY_FLAGS
                or RUNTIME_FLAGS & READONLY_FLAGS)
    config = jt.config.snapshot()
    expected = {name for name in STARTUP_FLAGS if hasattr(jt.flags, name)}
    assert set(config) == expected
    assert "config" in jt.__all__
    if "cuda_archs" in config:
        assert isinstance(jt.config.cuda_archs, tuple)
        config["cuda_archs"].append(-1)
        assert -1 not in jt.config.cuda_archs
        assert -1 not in jt.flags.cuda_archs
    runtime = jt.runtime.snapshot()
    expected_runtime = {name for name in RUNTIME_FLAGS | READONLY_FLAGS
                        if hasattr(jt.flags, name)}
    assert set(runtime) == expected_runtime
    assert set(config).isdisjoint(runtime)
    native_fields = {name for name, member in vars(jt.core.Flags).items()
                     if isinstance(member, property) or type(member).__name__ == "getset_descriptor"}
    assert {"cc_flags", "sync_run", "exec_called"} <= native_fields
    assert native_fields <= STARTUP_FLAGS | RUNTIME_FLAGS | READONLY_FLAGS | FLAG_ALIASES.keys()


def test_flag_policy_changes_invalidate_the_native_binding_build_stamp():
    files = jt.compiler.core_generator_signature()["files"]
    policy = files["_runtime/flag_policy.py"]
    assert policy["size"] > 0
    assert len(policy["sha256"]) == 64


def test_startup_seal_does_not_prevent_late_custom_operators():
    value = jt.code([3], "int32", cpu_src="@out(0)=3; @out(1)=5; @out(2)=8;")
    np.testing.assert_array_equal(value.numpy(), [3, 5, 8])
