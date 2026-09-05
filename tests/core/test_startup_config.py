"""Startup options freeze across every native Flags instance."""

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
