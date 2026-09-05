"""The optional ACL clamp hook has an explicit runtime-owned contract."""

from jittor._runtime import acl_clamp


def teardown_function(_function):
    acl_clamp.unregister_acl_clamp()


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
    assert acl_clamp.dispatch_acl_clamp("x", 0, 1) == "first-result"
    assert acl_clamp.register_acl_clamp(second) is first
    assert acl_clamp.unregister_acl_clamp(first) is None
    assert acl_clamp.dispatch_acl_clamp("x", 0, 1) == "second-result"
    assert acl_clamp.unregister_acl_clamp(second) is second
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
