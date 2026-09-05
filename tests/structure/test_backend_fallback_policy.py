import pytest

from jittor._runtime import (
    BackendFallbackPolicy,
    FallbackError,
    FallbackMode,
)


def test_policy_defaults_to_warn_and_reports_decision():
    policy = BackendFallbackPolicy()
    assert policy.mode == "warn"
    decision = policy.decide("conv2d", "cuda", "kernel unavailable", "cpu")
    assert decision.mode is FallbackMode.WARN
    assert decision.allowed
    assert decision.fallback_backend == "cpu"


def test_error_policy_fails_closed_with_context():
    policy = BackendFallbackPolicy("error")
    with pytest.raises(FallbackError, match="cuda/conv2d"):
        policy.decide("conv2d", "cuda", "kernel unavailable", "cpu")


def test_policy_validates_modes_and_self_fallback():
    policy = BackendFallbackPolicy("allow")
    policy.set_mode(FallbackMode.ERROR)
    assert policy.mode == "error"
    with pytest.raises(ValueError, match="one of"):
        policy.set_mode("silent")
    with pytest.raises(ValueError, match="differ"):
        policy.decide("add", "cpu", "missing", "cpu")

