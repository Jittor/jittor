"""torch._C._disabled_torch_function_impl runs a function with dispatch off."""

import sys

import pytest
import torch


def _overrides():
    return sys.modules["torch.overrides"]


class _Hooked:
    """A tensor-like whose hook intercepts every call."""

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        return "hooked"


class _OptedOut:
    """Uses the disabled implementation as its hook, as nn.Parameter does."""

    __torch_function__ = torch._C._disabled_torch_function_impl


class _Delegating:
    """A hook that records the call, then delegates with unchanged arguments."""

    calls = []

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        cls.calls.append(func)
        return torch._C._disabled_torch_function_impl(func, types, args, kwargs)


def _dispatching(x):
    overrides = _overrides()
    if overrides.has_torch_function((x,)):
        return overrides.handle_torch_function(_dispatching, (x,), x)
    return "plain"


def test_disabled_torch_function_impl_calls_with_args_and_kwargs():
    calls = []

    def function(value, *, scale):
        calls.append((value, scale))
        return value * scale

    result = torch._C._disabled_torch_function_impl(
        function, (torch.Tensor,), (3,), {"scale": 4})

    assert result == 12
    assert calls == [(3, 4)]


def test_disabled_torch_function_impl_accepts_missing_or_none_kwargs():
    def function(value=5):
        return value + 1

    assert torch._C._disabled_torch_function_impl(function, ()) == 6
    assert torch._C._disabled_torch_function_impl(function, (), (2,), None) == 3


def test_list_arguments_are_accepted_and_other_containers_rejected():
    impl = torch._C._disabled_torch_function_impl

    assert impl(lambda a, b: a + b, (), [1, 2]) == 3
    with pytest.raises(TypeError, match="List or Tuple"):
        impl(lambda *a: a, (), {"a": 1})


def test_a_type_using_the_disabled_impl_has_no_override():
    overrides = _overrides()

    assert not overrides.has_torch_function((_OptedOut(),))
    assert overrides.has_torch_function((_Hooked(),))
    assert _dispatching(_OptedOut()) == "plain"


def test_dispatch_is_off_while_the_function_runs_and_restored_after():
    overrides = _overrides()
    impl = torch._C._disabled_torch_function_impl
    hooked = (_Hooked(),)

    assert impl(overrides.has_torch_function, (), (hooked,)) is False
    nested = impl(lambda: (impl(overrides.has_torch_function, (), (hooked,)),
                           overrides.has_torch_function(hooked)), ())
    assert nested == (False, False)
    assert overrides.has_torch_function(hooked)


def test_dispatch_is_restored_after_the_function_raises():
    def fail():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        torch._C._disabled_torch_function_impl(fail, ())
    assert _overrides().has_torch_function((_Hooked(),))


def test_a_delegating_hook_runs_once_instead_of_recursing():
    _Delegating.calls.clear()

    assert _dispatching(_Delegating()) == "plain"
    assert _Delegating.calls == [_dispatching]


def test_cat_dispatches_only_while_dispatch_is_on():
    assert torch.cat([_Hooked()]) == "hooked"
    with pytest.raises((TypeError, AttributeError)):
        torch._C._disabled_torch_function_impl(torch.cat, (), ([_Hooked()],))
