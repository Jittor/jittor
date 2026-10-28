"""Actual CPU BF16 GEMM providers and differentiation/storage contracts."""
import importlib
import os
from unittest.mock import patch

import numpy as np
import torch
import jittor as jt
from jittor._runtime import forbid_backend_fallbacks
from jittor._runtime.backend_libraries import get_library_ops
from _helpers.device_types import instantiate_device_type_tests


# Independent binary Torch 2.11 CPU observations, including all HVP inputs.
_ORDINARY = (
    [[1.2734375, -.94140625], [.103515625, -.78125]],
    ([[3.265625, -.138671875, -2.984375], [.79296875, -1.0703125, -2.]],
     [[2.71875, .404296875, -1.1953125], [-3.0625, 1.2890625, .35546875]],
     [2.75, -3.4375]),
    ([[4.21875, 8.75, 7.59375], [1.203125, 4.8125, 4.0625]],
     [[12.1875, -.8125, 1.421875], [7.3125, -7.65625, -4.84375]], [10.5, 12.]),
)
_REPEATED = (
    _ORDINARY[0],
    ([[4.15625, -.134765625, -3.734375], [.96875, -1.4375, -2.625]],
     [[3.453125, .515625, -1.515625], [-3.890625, 1.6796875, .427734375]],
     [4.21875, -5.3125]),
    ([[6.3125, 12.375, 10.8125], [2.359375, 7.28125, 6.1875]],
     [[16.5, -1.5625, 1.75], [10.25, -10.25, -6.21875]], [18.5, 20.75]),
)


def _observe(value, expected):
    value.sync()
    assert value.location() == "cpu" and value.dtype == torch.bfloat16
    np.testing.assert_array_equal(value.detach().float().numpy(), expected)


class TestBFloat16CpuAccumulators:
    def test_empty_contracted_dimension_preserves_bias_and_gradients(self, device):
        with forbid_backend_fallbacks():
            for enabled in ("0", "1"):
                with patch.dict(os.environ, {"JT_BUILD_USE_MKL": enabled}):
                    for shape in ((2, 0), (2, 1, 0), (0,)):
                        x = torch.zeros(shape, dtype=torch.bfloat16, requires_grad=True)
                        weight = torch.zeros((3, 0), dtype=torch.bfloat16, requires_grad=True)
                        bias = torch.tensor([1., 2., 3.], dtype=torch.bfloat16, requires_grad=True)
                        out = torch.nn.functional.linear(x, weight, bias)
                        expected = np.broadcast_to([1., 2., 3.], shape[:-1] + (3,))
                        _observe(out, expected)
                        dx, dw, db = torch.autograd.grad(out.float().sum(), (x, weight, bias))
                        assert dx.shape == x.shape and dw.shape == weight.shape
                        assert dx.numel() == dw.numel() == 0
                        rows = int(np.prod(shape[:-1]))
                        _observe(db, [float(rows)] * 3)

    def test_native_opmath_hvp_rebinding_and_repeated_function(self, device):
        owner = importlib.import_module("jittor.nn.functional.matrix")
        initial = jt.core.backend_fallback_count()
        with patch.dict(os.environ, {"JT_BUILD_USE_MKL": "0"}), forbid_backend_fallbacks():
            assert get_library_ops("mkl", load=True) is None
            for mode in ("ordinary", "same-instance", "rebind-input", "rebind-weight"):
                x = torch.tensor([[1.0078125, .25, -.5], [.75, -1.125, .375]],
                                 dtype=torch.bfloat16, requires_grad=True)
                weight = torch.tensor([[1.0078125, .5, -.25], [-.375, .75, 1.25]],
                                      dtype=torch.bfloat16, requires_grad=True)
                bias = torch.tensor([.00390625, -.125], dtype=torch.bfloat16, requires_grad=True)
                xholder, wholder = (jt.ops.reshape(v, v.shape) for v in (x, weight))
                instance = owner._BFloat16MatmulAccumulator()
                output = (instance(xholder, wholder) + bias.float()).bfloat16()
                if mode == "same-instance":
                    extra = (instance(x * .5, weight + .125) + bias.float()).bfloat16()
                    loss = output.float().square().sum() + extra.float().square().sum()
                else:
                    loss = output.float().square().sum()
                if mode == "rebind-input":
                    xholder.assign(torch.full(x.shape, 7., dtype=torch.bfloat16))
                if mode == "rebind-weight":
                    wholder.assign(torch.full(weight.shape, 7., dtype=torch.bfloat16))
                gradients = torch.autograd.grad(loss, (x, weight, bias), create_graph=True)
                hessian = torch.autograd.grad(sum(g.float().sum() for g in gradients), (x, weight, bias))
                expected = _REPEATED if mode == "same-instance" else _ORDINARY
                _observe(output, expected[0])
                for values, references in ((gradients, expected[1]), (hessian, expected[2])):
                    for value, reference in zip(values, references):
                        _observe(value, reference)
            assert jt.core.backend_fallback_count() == initial

    def test_onednn_accumulator_is_real_bf16_and_keeps_default_output(self, device):
        with patch.dict(os.environ, {"JT_BUILD_USE_MKL": "1"}), forbid_backend_fallbacks():
            ops = get_library_ops("mkl", load=True)
            if ops is None:
                self.skipTest("optional oneDNN library is unavailable")
            original = ops.mkl_matmul
            calls = []

            def observe(a, b, trans_a, trans_b, fp32_output=False):
                result = original(a, b, trans_a, trans_b, fp32_output)
                if fp32_output:
                    for operand in (a, b):
                        operand.sync()
                        assert operand.dtype == torch.bfloat16 and operand.location() == "cpu"
                    result.sync()
                    assert result.dtype == torch.float32 and result.location() == "cpu"
                    calls.append((trans_a, trans_b))
                return result

            x = torch.tensor([[1.0078125]], dtype=torch.bfloat16, requires_grad=True)
            weight = torch.tensor([[1.0078125]], dtype=torch.bfloat16, requires_grad=True)
            bias = torch.tensor([.00390625], dtype=torch.bfloat16, requires_grad=True)
            with patch.object(ops, "mkl_matmul", observe):
                output = torch.nn.functional.linear(x, weight, bias)
                _observe(output, [[1.0234375]])
            assert calls == [(False, True)]
            # Equal extents with three different dtype/output descriptors must
            # not reuse a cached oneDNN plan for the wrong storage width.
            wide = x.float()
            for a, fp32, expected_dtype, expected in (
                    (wide, False, torch.float32, 1.01568603515625),
                    (x, True, torch.float32, 1.01568603515625),
                    (x, False, torch.bfloat16, 1.015625),
                    (wide, False, torch.float32, 1.01568603515625)):
                result = original(a, a, False, True, fp32)
                result.sync()
                assert result.dtype == expected_dtype and result.location() == "cpu"
                np.testing.assert_array_equal(result.float().numpy(), [[expected]])
            with self.assertRaisesRegex(RuntimeError, "BF16 compute operands"):
                original(wide, wide, False, True, True)
            with self.assertRaisesRegex(RuntimeError, "matching float32 or bfloat16"):
                original(x, wide, False, True)


instantiate_device_type_tests(TestBFloat16CpuAccumulators, globals(), only_for=("cpu",))
