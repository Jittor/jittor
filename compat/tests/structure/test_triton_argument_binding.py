"""Compare cached launch binding with inspect, including invalid calls."""
import gc
import importlib.util
import inspect
from pathlib import Path
import types
import weakref
from unittest import mock

import pytest


@pytest.fixture
def binding():
    path = Path(__file__).resolve().parents[2] / "triton/argument_binding.py"
    spec = importlib.util.spec_from_file_location("argument_binding", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.bind_arguments


@pytest.mark.parametrize("args,kwargs", [
    ((1,), {"required": 9}),
    ((), {"a": 1, "required": 9}),
    ((1, 2), {"required": 9, "option": 5}),
    ((1, 2, 3), {"required": 9}),
    ((1,), {}),
    ((1,), {"a": 2, "required": 9}),
    ((1,), {"required": 9, "unknown": 1}),
    ((), {"required": 9}),
])
def test_fixed_signature_matches_inspect_on_first_and_cached_calls(binding, args, kwargs):
    def function(a, b=7, *, required, option=None):
        pass

    signature = inspect.signature(function)
    kernel = types.SimpleNamespace(signature=signature)
    for _ in range(2):
        try:
            expected = signature.bind(*args, **kwargs)
        except TypeError:
            with pytest.raises(TypeError):
                binding(kernel, signature, args, kwargs)
        else:
            expected.apply_defaults()
            assert binding(kernel, signature, args, kwargs) == expected.arguments


def test_defaults_are_objects_and_cached_calls_skip_inspect(binding):
    class NoRepr:
        def __repr__(self):
            raise AssertionError("Default value must not be serialized into code")

    default = NoRepr()
    def function(x=default):
        pass

    signature = inspect.signature(function)
    kernel = types.SimpleNamespace(signature=signature)
    assert binding(kernel, signature, (), {})["x"] is default
    with mock.patch.object(inspect.Signature, "bind", side_effect=AssertionError("slow bind")):
        assert binding(kernel, signature, (3,), {}) == {"x": 3}
    function.__defaults__ = (4,)
    assert binding(kernel, signature, (), {})["x"] is default
    kernel.signature = inspect.signature(function)
    assert binding(kernel, kernel.signature, (), {}) == {"x": 4}


def test_unusual_signatures_keep_inspect_semantics(binding):
    def function(a, /, *values, required=1, **extra):
        pass

    signature = inspect.signature(function)
    kernel = types.SimpleNamespace(signature=signature)
    for args, kwargs in [((1,), {}), ((1, 2, 3), {"a": 4, "required": 5})]:
        expected = signature.bind(*args, **kwargs)
        expected.apply_defaults()
        assert binding(kernel, signature, args, kwargs) == expected.arguments


def test_bound_values_are_not_retained_by_kernel_cache(binding):
    class Operand:
        pass

    def function(pointer):
        pass

    signature = inspect.signature(function)
    kernel = types.SimpleNamespace(signature=signature)
    operand = Operand()
    reference = weakref.ref(operand)
    assert binding(kernel, signature, (operand,), {}) == {"pointer": operand}
    del operand
    gc.collect()
    assert reference() is None
