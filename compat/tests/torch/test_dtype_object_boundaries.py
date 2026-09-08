"""Real dtype identities cross native allocation, JIT and NumPy boundaries."""

import pickle

import jittor as jt
import numpy as np
import pytest

from jittor.compat.torch.tensor_state import compatibility_owner
from jittor.compat.torch.types import dtype


@pytest.fixture
def torch():
    return compatibility_owner(jt)


def test_dtype_objects_are_immutable_singletons(torch, monkeypatch):
    assert not isinstance(torch.float32, str)
    assert str(torch.float32) == "torch.float32"
    assert f"{torch.int64}" == "torch.int64"
    assert torch.float32 != "float32"
    assert torch.float32 != "torch.float32"
    assert torch.long is torch.int64
    assert pickle.loads(pickle.dumps(torch.float32)) is torch.float32
    assert {torch.float32: "real"}[torch.float32] == "real"
    with pytest.raises(AttributeError, match="immutable"):
        torch.float32.name = "float8_e4m3fn"
    # Exercise the actual old str-subclass pickle protocol with its state dict.
    import sys
    class OldDtype(str):
        pass
    OldDtype.__name__ = OldDtype.__qualname__ = "dtype"
    OldDtype.__module__ = dtype.__module__
    legacy = OldDtype("float32")
    legacy.name = "float32"
    legacy._is_fp = True
    with monkeypatch.context() as patch:
        patch.setattr(sys.modules[dtype.__module__], "dtype", OldDtype)
        payload = pickle.dumps(legacy)
    assert pickle.loads(payload) is torch.float32


@pytest.mark.parametrize("name", ["float32", "float64", "int32", "int64", "bool"])
def test_dtype_objects_reach_native_overloads_and_numpy_construction(torch, name):
    selected = getattr(torch, name)
    values = np.array([0, 1, 2], dtype=name)
    python_array = jt.array(values, dtype=selected)
    native_array = jt.core.ops.array(values)
    cast = jt.core.ops.unary(native_array, selected)
    assert jt.core._checked_dtype_name(selected) == name
    assert jt.core._checked_dtype_name(str(selected)) == name
    np.testing.assert_array_equal(python_array.numpy(), values)
    np.testing.assert_array_equal(cast.numpy(), values)
    assert jt.core._checked_dtype_name(cast.dtype) == name


def test_placeholder_dtypes_are_metadata_only_at_every_boundary(torch):
    value = torch.tensor([1.0])
    placeholders = [item for name, item in dtype._registry.items()
                    if name not in dtype._supported]
    assert placeholders
    for selected in placeholders:
        assert {selected: selected.name}[selected] == selected.name
        for operation in (
            lambda: torch.empty((2,), dtype=selected),
            lambda: torch.empty_like(value, dtype=selected),
            lambda: torch.tensor([1.0], dtype=selected),
            lambda: torch.hann_window(2, dtype=selected),
            lambda: torch.sparse_coo_tensor([[0]], [1.0], (1,), dtype=selected),
            lambda: jt.array([1.0], dtype=selected),
            lambda: jt.random((2,), dtype=selected),
            lambda: value.cast(selected),
            lambda: jt.core.ops.empty((2,), selected),
            lambda: jt.core.ops.empty((2,), selected.name),
            lambda: jt.core.ops.empty((2,), str(selected)),
        ):
            with pytest.raises(NotImplementedError, match=selected.name):
                operation()
    np.testing.assert_array_equal((value + 1).numpy(), [2.0])


def test_default_dtype_precision_matmul_and_gradient(torch):
    previous = torch.get_default_dtype()
    a = None
    try:
        torch.set_default_dtype(torch.float64)
        precise = 1.0000000000001
        value = torch.tensor([precise])
        assert value.dtype is torch.float64
        assert value.numpy()[0] == precise
        assert torch.zeros_like(value).dtype is torch.float64
        assert torch.empty_like(value, dtype=torch.int64).dtype is torch.int64
        window = torch.hann_window(3, periodic=False, dtype=torch.float64)
        assert window.dtype is torch.float64
        np.testing.assert_array_equal(window.numpy(), [0., 1., 0.])
        a = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
        b = torch.tensor([[5., 6.], [7., 8.]])
        result = torch.matmul(a, b)
        np.testing.assert_allclose(result.numpy(), [[19., 22.], [43., 50.]])
        gradient = torch.autograd.grad(result.sum(), a)[0]
        assert gradient.dtype is torch.float64
        np.testing.assert_allclose(gradient.numpy(), [[11., 15.], [11., 15.]])
    finally:
        if a is not None:
            a.requires_grad_(False)
            from jittor.compat.torch.nested import _torch_prune_leaf_registry
            _torch_prune_leaf_registry()
        torch.set_default_dtype(previous)


def test_registered_kernel_dtype_filter_uses_canonical_names(torch):
    from jittor._runtime.dispatch import register_kernel, select_kernel, unregister_kernel
    def identity(value):
        return value
    register_kernel("test.dtype_object_identity", "*", identity, dtypes=(torch.float32,))
    try:
        value = torch.tensor([1.0], dtype=torch.float32)
        assert select_kernel("test.dtype_object_identity", value) is identity
    finally:
        unregister_kernel("test.dtype_object_identity", "*", identity)
