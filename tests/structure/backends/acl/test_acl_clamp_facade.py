"""The optional ACL clamp hook has an explicit runtime-owned contract."""

import pytest

from jittor._runtime import acl_clamp, dispatch


@pytest.fixture(autouse=True)
def isolated_acl_registration(monkeypatch):
    monkeypatch.setattr(dispatch, "dispatch_context", lambda *args, **kwargs:
                        dispatch.DispatchContext("acl", 0, ("float32",)))
    with dispatch.override_kernel("clamp.scalar", "acl", None):
        with dispatch.override_kernel("clamp.scalar", "*", None):
            yield


def test_acl_clamp_falls_back_without_a_registered_backend():
    acl_clamp.unregister_acl_clamp()
    assert acl_clamp.dispatch_acl_clamp("input", 0, 1) is None


def test_acl_clamp_registration_and_removal_are_replaceable():
    calls = []

    def first(*args):
        calls.append(("first", args))
        return "first-result"

    def second(*args):
        calls.append(("second", args))
        return "second-result"

    assert acl_clamp.register_acl_clamp(first) is None
    assert dispatch.registered_kernel("clamp.scalar", "acl") is first
    assert acl_clamp.register_acl_clamp(first) is first
    assert acl_clamp.dispatch_acl_clamp("x", 0, 1) == "first-result"
    assert acl_clamp.register_acl_clamp(second) is first
    assert dispatch.registered_kernel("clamp.scalar", "acl") is second
    assert acl_clamp.unregister_acl_clamp(first) is None
    assert acl_clamp.dispatch_acl_clamp("x", 0, 1) == "second-result"
    assert acl_clamp.unregister_acl_clamp(second) is second
    assert dispatch.registered_kernel("clamp.scalar", "acl") is None
    assert calls == [("first", ("x", 0, 1)), ("second", ("x", 0, 1))]


def test_acl_clamp_backend_exceptions_are_not_silently_fallback():
    def broken(*_args):
        raise ValueError("ACL clamp failed")

    acl_clamp.register_acl_clamp(broken)
    try:
        acl_clamp.dispatch_acl_clamp("x", 0, 1)
    except ValueError as error:
        assert str(error) == "ACL clamp failed"
    else:
        raise AssertionError("ACL backend errors must propagate")


def test_acl_clamp_does_not_select_acl_on_cpu(monkeypatch):
    def unexpected(*_args):
        raise AssertionError("ACL kernel selected for CPU input")

    acl_clamp.register_acl_clamp(unexpected)
    monkeypatch.setattr(dispatch, "dispatch_context", lambda *args, **kwargs:
                        dispatch.DispatchContext("cpu", -1, ("float32",)))
    with dispatch.override_kernel("clamp.scalar", "cpu", None):
        assert acl_clamp.dispatch_acl_clamp("x", 0, 1) is None


def test_acl_clamp_observes_registry_removal():
    handler = lambda *_args: "result"
    acl_clamp.register_acl_clamp(handler)
    dispatch.unregister_kernel("clamp.scalar", "acl", handler)
    assert acl_clamp.dispatch_acl_clamp("x", 0, 1) is None
    assert acl_clamp.unregister_acl_clamp(handler) is None


def test_acl_clamp_invalid_replacement_preserves_registration():
    handler = lambda *_args: "result"
    acl_clamp.register_acl_clamp(handler)
    with pytest.raises(TypeError, match="handler must be callable"):
        acl_clamp.register_acl_clamp(None)
    assert dispatch.registered_kernel("clamp.scalar", "acl") is handler
