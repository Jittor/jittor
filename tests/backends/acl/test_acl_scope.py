"""A failed ACL graph must not mask scope errors or corrupt device policy."""

import pytest

from _helpers.capability import require_accelerator
from _helpers.child_process import run_child_script


@pytest.mark.parametrize("boundary", ["body", "sync", "exit"])
def test_failed_graph_restores_scope_without_resubmission(boundary):
    require_accelerator("acl")
    # Expected fallback rejections belong to the child. The parent's autouse
    # zero-fallback guard remains intact, including its attempt counter.
    result = run_child_script(
        r'''
import gc
import os

import jittor as jt

assert jt.compiler.has_acl
boundary = os.environ["JITTOR_TEST_SCOPE_BOUNDARY"]
jt.runtime.use_cuda = 0
saved_grad = jt.flags.no_grad
pending = []
marker = ValueError("original-scope-body")

with jt.runtime.scope(backend_fallback="error"):
    before = jt.core.backend_fallback_count()
    try:
        with jt.flag_scope(use_acl=1, use_cuda=1, no_grad=1):
            # ACL rejects float64 random through its fallback counter. Retain
            # the failed graph so a second submission is observable exactly.
            pending.append(jt.core.ops.random((16,), "float64", "normal"))
            if boundary == "body":
                raise marker
            if boundary == "sync":
                pending[0].sync()
    except (ValueError, RuntimeError) as error:
        if boundary == "body":
            assert error is marker, repr(error)
        else:
            assert isinstance(error, RuntimeError), repr(error)
            message = str(error).lower()
            assert "fallback" in message and "random" in message, message
    else:
        raise AssertionError("expected scope or graph failure")

    assert jt.runtime.use_cuda == 0, jt.runtime.use_cuda
    assert jt.flags.no_grad == saved_grad
    attempts = jt.core.backend_fallback_count() - before
    assert attempts == (0 if boundary == "body" else 1), attempts
    pending.clear()
    gc.collect()

    # Once the caller releases the failed graph, normal device execution must
    # remain usable. This also detects pending work leaked by error cleanup.
    with jt.runtime.scope(use_cuda=1):
        output = jt.array([2.0, 3.0]) + 1.0
        output.sync()
        assert output.location() == "device"
        assert output.device_id >= 0
        assert output.placement_backend in (-1, 2)
        assert output.numpy().tolist() == [3.0, 4.0]
    assert jt.runtime.use_cuda == 0
    assert jt.core.backend_fallback_count() - before == attempts
print("ACL-SCOPE-RECOVERY-PASS", boundary, flush=True)
''',
        env={"JITTOR_TEST_SCOPE_BOUNDARY": boundary},
        text=True,
        crash_isolated=True,
        without_torch_mode=True,
        name="acl_scope_" + boundary,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ACL-SCOPE-RECOVERY-PASS " + boundary in result.stdout
