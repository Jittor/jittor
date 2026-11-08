"""Specified-dimension extrema select the first extremum and its full gradient.

The native Torch interpreter and the shim run this same file independently.
Set JITTOR_TEST_DEVICE=npu only under an explicitly assigned accelerator lease.
"""

from contextlib import nullcontext
from functools import lru_cache
import os
import numpy as np
import pytest
import torch

DEVICE = os.environ.get("JITTOR_TEST_DEVICE", "cpu")
DTYPES = ["float16", "float32"] + (["bfloat16"] if DEVICE == "npu" else [])
PATTERNS = ["ties", "inf"] + (["nan_inf"] if DEVICE == "npu" else [])
IS_SHIM = hasattr(torch, "_torch_compat_install_context")
if DEVICE == "npu" and (not IS_SHIM):
    import torch_npu  # noqa: F401 -- registers the real NPU backend


def _scope(device):
    if not IS_SHIM:
        return nullcontext()
    import jittor as jt
    from jittor._runtime.fallback import forbid_backend_fallbacks

    assert bool(jt.compiler.has_acl) == (device == "npu")
    return forbid_backend_fallbacks()


def _materialize_and_check(device, *tensors):
    jt = None
    if IS_SHIM:
        import jittor as jt

        jt.sync_all(device_sync=True)
    elif device == "npu":
        torch.npu.synchronize()
    if device == "npu":
        _canonical_npu_guard()(
            torch, [(str(index), value) for index, value in enumerate(tensors)], jt=jt
        )
    else:
        for value in tensors:
            assert value.device.type == "cpu"
            if jt is not None:
                assert value.location() == "cpu" and jt.core.dispatch_context([value])[0] == "cpu"


@lru_cache(maxsize=1)
def _canonical_npu_guard():
    import importlib.util
    from pathlib import Path

    root = Path(
        os.environ.get("JITTOR_SOURCE_ROOT", str(Path(__file__).resolve().parents[3]))
    ).resolve()
    source = root / "benchmarks/profiling/benchmark_qwen3_ascend.py"
    spec = importlib.util.spec_from_file_location("_public_extrema_npu_placement", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._assert_npu_values


def _array(value):
    return value.detach().float().cpu().numpy().copy()


@pytest.mark.parametrize("device", [DEVICE])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("which", ["max", "min"])
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("geometry", ["last", "first", "vector_last", "vector_first"])
@pytest.mark.parametrize("pattern", PATTERNS)
def test_specified_dim_extreme_first_index_and_full_source_gradient(
    device, dtype, which, keepdim, geometry, pattern
):
    if device == "npu":
        assert torch.npu.is_available() and torch.npu.device_count() == 1
    assert device in ("cpu", "npu")
    data = np.array([[1, 5, 5, -2], [4, -3, -3, 4], [2, 5, 0, 4]], dtype=np.float32)
    if pattern == "inf":
        data = np.array(
            [[np.inf, 5, 5, -np.inf], [4, -np.inf, -np.inf, 4], [2, 5, np.inf, 4]], dtype=np.float32
        )
    if pattern == "nan_inf":
        data = np.array(
            [[np.nan, 5, np.nan, -np.inf], [np.inf, -3, -3, np.nan], [2, np.nan, np.inf, -np.inf]],
            dtype=np.float32,
        )
    dim = 0 if geometry in ("first", "vector_first") else -1
    if geometry.startswith("vector"):
        data = data[0].copy()
    axis = dim % data.ndim
    indices = getattr(np, "argmax" if which == "max" else "argmin")(data, axis=axis)
    indices_kept = np.expand_dims(indices, axis)
    expected_values = np.take_along_axis(data, indices_kept, axis=axis)
    if not keepdim:
        expected_values = np.squeeze(expected_values, axis)
    expected_indices = indices_kept if keepdim else indices
    weights = (np.arange(expected_values.size, dtype=np.float32) + 1).reshape(
        expected_values.shape
    ) / 8
    expected_gradient = np.zeros_like(data)
    np.put_along_axis(
        expected_gradient,
        indices_kept,
        weights if keepdim else np.expand_dims(weights, axis),
        axis=axis,
    )
    with _scope(device):
        source = torch.tensor(data, dtype=getattr(torch, dtype), device=device, requires_grad=True)
        result = getattr(torch, which)(source, dim=dim, keepdim=keepdim)
        (gradient,) = torch.autograd.grad(
            result.values,
            source,
            grad_outputs=torch.tensor(weights, dtype=getattr(torch, dtype), device=device),
        )
        _materialize_and_check(device, source, result.values, result.indices, gradient)
        assert result.values.dtype == source.dtype and gradient.dtype == source.dtype
        assert result.indices.dtype == torch.int64
        assert tuple(gradient.shape) == tuple(source.shape)
        np.testing.assert_array_equal(result.indices.detach().cpu().numpy(), expected_indices)
        np.testing.assert_array_equal(_array(result.values), expected_values)
        np.testing.assert_array_equal(_array(gradient), expected_gradient)


@pytest.mark.parametrize("device", [DEVICE])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dim", [0, 1])
def test_gather_smaller_index_shape_and_duplicate_source_gradient(device, dtype, dim):
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    indices = (
        np.array([[1, 0, 2, 2], [1, 0, 2, 1]], dtype=np.int64)
        if dim == 0
        else np.array([[1, 1], [3, 0], [0, 0]], dtype=np.int64)
    )
    weights = (np.arange(indices.size, dtype=np.float32) + 1).reshape(indices.shape) / 8
    expected = np.zeros_like(data)
    coordinates = list(np.indices(indices.shape))
    coordinates[dim] = indices
    np.add.at(expected, tuple(coordinates), weights)
    with _scope(device):
        source = torch.tensor(data, dtype=getattr(torch, dtype), device=device, requires_grad=True)
        index = torch.tensor(indices, dtype=torch.int64, device=device)
        result = torch.gather(source, dim, index)
        (gradient,) = torch.autograd.grad(
            result,
            source,
            grad_outputs=torch.tensor(weights, dtype=getattr(torch, dtype), device=device),
        )
        _materialize_and_check(device, source, index, result, gradient)
        assert tuple(gradient.shape) == tuple(source.shape)
        assert result.dtype == gradient.dtype == source.dtype
        np.testing.assert_array_equal(_array(result), np.take_along_axis(data, indices, axis=dim))
        np.testing.assert_array_equal(_array(gradient), expected)


if DEVICE == "npu":

    @pytest.mark.parametrize("device", [DEVICE])
    @pytest.mark.parametrize("operation", ["gather", "max", "min"])
    def test_npu_input_backward_keeps_actual_device_under_global_cpu(device, operation):
        data = np.array([[1, 5, 5, -2], [4, -3, -3, 4], [2, 5, 0, 4]], dtype=np.float32)
        with _scope(device):
            source = torch.tensor(data, device=device, requires_grad=True)
            if operation == "gather":
                dim = 1
                indices = np.array([[1, 1], [3, 0], [0, 0]], dtype=np.int64)
                index = torch.tensor(indices, device=device)
                result = torch.gather(source, dim, index)
            else:
                dim = -1 if operation == "max" else 0
                pair = getattr(torch, operation)(source, dim=dim, keepdim=True)
                result, index = pair.values, pair.indices
                indices = index.detach().cpu().numpy().copy()
            weights = (np.arange(result.numel(), dtype=np.float32) + 1).reshape(
                tuple(result.shape)
            ) / 8
            output_gradient = torch.tensor(weights, device=device)
            _materialize_and_check(device, source, index, result, output_gradient)
            if IS_SHIM:
                import jittor as jt

                backward_scope = jt.flag_scope(use_cuda=0)
            else:
                backward_scope = nullcontext()
            with backward_scope:
                if IS_SHIM:
                    assert jt.flags.use_cuda == 0
                (gradient,) = torch.autograd.grad(result, source, grad_outputs=output_gradient)
                _materialize_and_check(device, gradient)
            expected = np.zeros_like(data)
            coordinates = list(np.indices(indices.shape))
            coordinates[dim] = indices
            np.add.at(expected, tuple(coordinates), weights)
            assert tuple(gradient.shape) == tuple(source.shape)
            assert gradient.dtype == source.dtype
            np.testing.assert_array_equal(_array(gradient), expected)
