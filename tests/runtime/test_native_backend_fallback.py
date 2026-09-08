"""Native fallback decisions and Python flag scopes share one runtime owner."""

import pytest

import jittor as jt


def test_backend_fallback_has_one_runtime_owner():
    jt.tests.backend_fallback_has_one_runtime_owner()


def test_invalid_backend_fallback_preserves_policy():
    from contextlib import ExitStack as _TestPolicyStack
    with _TestPolicyStack() as _test_policy_stack:
        original = jt.introspection.policy.runtime.backend_fallback
        try:
            with jt.flag_scope(backend_fallback="error"):
                with pytest.raises(RuntimeError, match="backend_fallback must be error, warn, or allow"):
                    _test_policy_stack.enter_context(jt.runtime.scope(backend_fallback="silent"))
                assert jt.introspection.policy.runtime.backend_fallback == "error"
                with jt.flag_scope(backend_fallback="allow"):
                    assert jt.introspection.policy.runtime.backend_fallback == "allow"
                assert jt.introspection.policy.runtime.backend_fallback == "error"
        finally:
            _test_policy_stack.enter_context(jt.runtime.scope(backend_fallback=original))


@pytest.mark.parametrize("policy", ["error", "warn", "allow"])
def test_backend_fallback_decision_and_diagnostics(policy, capfd):
    before = jt.core.backend_fallback_count()
    with jt.flag_scope(backend_fallback=policy):
        if policy == "error":
            with pytest.raises(RuntimeError, match=(
                    "Backend fallback: op=fallback_policy_probe backend=.* "
                    "target=cpu reason=test kernel is unavailable")):
                jt.tests.backend_fallback_checks_current_policy()
        else:
            jt.tests.backend_fallback_checks_current_policy()
    assert jt.core.backend_fallback_count() == before + 1
    captured = capfd.readouterr()
    output = captured.out + captured.err
    if policy == "warn":
        assert "op=fallback_policy_probe" in output
        assert "target=cpu reason=test kernel is unavailable" in output
    elif policy == "allow":
        assert "op=fallback_policy_probe" not in output
