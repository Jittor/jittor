"""Numerical AMP boundaries measured against independent PyTorch CPU/CUDA.

The large values distinguish bfloat16's exponent range from float16; every
case executes its outputs and gradients on the selected device. Expected
scaler values are from PyTorch 2.11, including CUDA's float-valued finite check
for a double gradient and overflow introduced by unscale itself.
"""

from functools import wraps
import importlib
import unittest
from unittest import mock

import numpy as np

import jittor as jt
import torch

from jittor._runtime import forbid_backend_fallbacks
from _helpers import common as cu
from _helpers.device_types import instantiate_device_type_tests


def _region(device):
    return "cpu" if device == "cpu" else "cuda"


def _no_fallbacks(function):
    @wraps(function)
    def run(self, device):
        with forbid_backend_fallbacks():
            try:
                return function(self, device)
            finally:
                jt.sync_all(True)
    return run


def _resident(test, value, device):
    value.sync()
    test.assertEqual(value.location(), "cpu" if device == "cpu" else "device")


def _scaler_case(device, scale, gradient, dtype=torch.float32):
    parameter = torch.nn.Parameter(torch.tensor([1.], dtype=dtype, device=device))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scaler = torch.amp.GradScaler(_region(device), init_scale=scale,
                                  growth_factor=4., growth_interval=1)
    scaler.scale(parameter.sum())
    parameter.grad = torch.tensor([gradient], dtype=dtype, device=device)
    return parameter, optimizer, scaler


class TestAmpBoundaries(cu.JittorTestCase):
    @_no_fallbacks
    def test_inplace_unary_rejects_integer_targets_without_mutation(self, device):
        for dtype in (torch.bool, torch.int32, torch.int64):
            for enabled in (False, True):
                for name in ('log_', 'rsqrt_', 'exp_'):
                    with self.subTest(dtype=str(dtype), enabled=enabled, operation=name):
                        x = torch.tensor([1, 2], dtype=dtype, device=device)
                        before = x.cpu().numpy().copy()
                        with torch.autocast(_region(device), dtype=torch.bfloat16, enabled=enabled):
                            with self.assertRaisesRegex(RuntimeError, 'cast'):
                                getattr(x, name)()
                        self.assertEqual(str(x.dtype), str(dtype))
                        _resident(self, x, device)
                        np.testing.assert_array_equal(x.cpu().numpy(), before)
        if device == 'cpu':
            for name in ('log_', 'rsqrt_', 'exp_'):
                x = torch.tensor([1., 4.], dtype=torch.complex64, device=device)
                self.assertIs(getattr(x, name)(), x)
                self.assertTrue(x.is_complex())
                self.assertTrue(np.isfinite(x.numpy()).all())

    @_no_fallbacks
    def test_exp_inplace_preserves_dtype_identity_and_aliases(self, device):
        expected = {
            'torch.float16': [1.2841796875, 1.6484375, 2.720703125, 7.40234375],
            'torch.bfloat16': [1.28125, 1.6484375, 2.71875, 7.375],
        }
        for precision in (torch.float16, torch.bfloat16):
            for enabled in (False, True):
                for dtype in (torch.float16, torch.bfloat16):
                    with self.subTest(dtype=str(dtype), precision=str(precision), enabled=enabled):
                        x = torch.tensor([.2501, .5001, 1.001, 2.001], dtype=dtype, device=device)
                        alias = x.view(-1)
                        _resident(self, x, device)
                        _resident(self, alias, device)
                        with torch.autocast(_region(device), dtype=precision, enabled=enabled):
                            self.assertIs(x.exp_(), x)
                        self.assertEqual(str(x.dtype), str(dtype))
                        self.assertEqual(str(alias.dtype), str(dtype))
                        for value in (x, alias):
                            _resident(self, value, device)
                            np.testing.assert_array_equal(value.float().cpu().numpy(), expected[str(dtype)])

    @_no_fallbacks
    def test_cpu_exp_output_dtype_and_cuda_autocast_policy(self, device):
        expected_low = {
            'torch.float16': [1.2841796875, 1.6484375, 2.720703125, 7.40234375],
            'torch.bfloat16': [1.28125, 1.6484375, 2.71875, 7.375],
        }
        for dtype in (torch.float16, torch.bfloat16):
            for precision in (torch.float16, torch.bfloat16):
                for enabled in ((False, True) if device == 'cpu' else (True,)):
                    for function in (torch.exp, lambda x: x.exp(), lambda x: torch.exp(input=x)):
                        with self.subTest(dtype=str(dtype), precision=str(precision), enabled=enabled):
                            x = torch.tensor([.2501, .5001, 1.001, 2.001], dtype=dtype, device=device)
                            with torch.autocast(_region(device), dtype=precision, enabled=enabled):
                                output = function(x)
                            expected_dtype = str(dtype) if device == 'cpu' else 'torch.float32'
                            self.assertEqual(str(output.dtype), expected_dtype)
                            _resident(self, output, device)
                            if device == 'cpu':
                                np.testing.assert_array_equal(output.float().cpu().numpy(), expected_low[str(dtype)])
                            with self.assertRaises(TypeError):
                                torch.exp(x, dtype=torch.float32)

    @_no_fallbacks
    def test_autograd_grad_returns_input_dtypes_and_preserves_unused_entries(self, device):
        for enabled in (False, True):
            dtypes = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
            inputs = tuple(torch.zeros(2, dtype=dtype, device=device).requires_grad_(True)
                           for dtype in dtypes)
            with torch.autocast(_region(device), dtype=torch.bfloat16, enabled=enabled):
                outputs = tuple(torch.exp(value) for value in inputs)
            gradients = torch.autograd.grad(outputs, inputs,
                                             grad_outputs=tuple(torch.ones_like(value) for value in outputs))
            for value, gradient in zip(inputs, gradients):
                self.assertEqual(str(gradient.dtype), str(value.dtype))
                self.assertFalse(gradient.requires_grad)
                _resident(self, gradient, device)
                np.testing.assert_array_equal(gradient.float().cpu().numpy(), [1., 1.])
            for materialize in (False, True):
                for create_graph in (False, True):
                    x = torch.zeros(2, dtype=torch.float16, device=device).requires_grad_(True)
                    unused = torch.zeros(2, dtype=torch.float64, device=device).requires_grad_(True)
                    with torch.autocast(_region(device), dtype=torch.bfloat16, enabled=enabled):
                        y = torch.exp(x)
                    first, other = torch.autograd.grad(y, (x, unused), grad_outputs=torch.ones_like(y),
                                                        allow_unused=True, materialize_grads=materialize,
                                                        create_graph=create_graph)
                    self.assertEqual(str(first.dtype), str(x.dtype))
                    self.assertEqual(first.requires_grad, create_graph)
                    _resident(self, first, device)
                    if materialize:
                        # Native materialized zeros and frontend tensors
                        # use different dtype spelling for the same width.
                        self.assertEqual(str(other.dtype).split('.')[-1],
                                         str(unused.dtype).split('.')[-1])
                        _resident(self, other, device)
                        np.testing.assert_array_equal(other.cpu().numpy(), [0., 0.])
                    else:
                        self.assertIsNone(other)

    @_no_fallbacks
    def test_autograd_grad_dtype_cast_retains_second_derivative(self, device):
        for dtype in (torch.float16, torch.bfloat16):
            for enabled in (False, True):
                x = torch.zeros(2, dtype=dtype, device=device).requires_grad_(True)
                with torch.autocast(_region(device), dtype=torch.bfloat16, enabled=enabled):
                    y = torch.exp(x)
                first, = torch.autograd.grad(y, (x,), grad_outputs=torch.ones_like(y), create_graph=True)
                self.assertEqual(str(first.dtype), str(dtype))
                self.assertTrue(first.requires_grad)
                second, = torch.autograd.grad(first, (x,), grad_outputs=torch.ones_like(first))
                self.assertEqual(str(second.dtype), str(dtype))
                self.assertFalse(second.requires_grad)
                for value in (first, second):
                    _resident(self, value, device)
                    np.testing.assert_array_equal(value.float().cpu().numpy(), [1., 1.])

    @_no_fallbacks
    def test_exp_autocast_keeps_wide_input_gradient_dtype(self, device):
        for precision in (torch.float16, torch.bfloat16):
            for dtype in (torch.float32, torch.float64):
                for function in (torch.exp, lambda value: value.exp()):
                    x = torch.tensor([.25, .5, 1., 2.], dtype=dtype, device=device).requires_grad_(True)
                    with torch.autocast(_region(device), dtype=precision):
                        output = function(x)
                    gradient, = torch.autograd.grad(output, (x,), grad_outputs=torch.ones_like(output))
                    self.assertEqual(str(output.dtype), str(dtype))
                    self.assertEqual(str(gradient.dtype), str(dtype))
                    _resident(self, gradient, device)
                    np.testing.assert_array_equal(output.cpu().numpy(), gradient.cpu().numpy())

    @_no_fallbacks
    def test_fp32_autocast_policy_is_shared_by_public_entry_points(self, device):
        for precision in (torch.float16, torch.bfloat16):
            x = torch.tensor([.25, .5, 1., 2.], dtype=precision, device=device).requires_grad_(True)
            functions = (torch.log, lambda value: value.log(), torch.rsqrt, lambda value: value.rsqrt(),
                         torch.sum, lambda value: value.sum(), lambda value: torch.cumsum(value, 0),
                         lambda value: value.cumsum(0), lambda value: torch.cumprod(value, 0),
                         lambda value: value.cumprod(0), lambda value: torch.softmax(value, 0),
                         lambda value: value.softmax(0), lambda value: torch.nn.functional.softmax(value, 0),
                         torch.nn.Softmax(dim=0), lambda value: torch.log_softmax(value, 0),
                         lambda value: value.log_softmax(0), lambda value: torch.nn.functional.log_softmax(value, 0),
                         torch.nn.LogSoftmax(dim=0))
            for index, function in enumerate(functions):
                with self.subTest(precision=str(precision), entry=index):
                    with torch.autocast(_region(device), dtype=precision):
                        output = function(x)
                    expected_dtype = 'torch.float32' if device == 'cuda' else str(precision)
                    self.assertEqual(str(output.dtype), expected_dtype)
                    _resident(self, output, device)
                    gradient, = torch.autograd.grad(output, (x,), grad_outputs=torch.ones_like(output))
                    self.assertEqual(str(gradient.dtype), str(precision))
                    _resident(self, gradient, device)
                    self.assertTrue(np.isfinite(gradient.float().cpu().numpy()).all())
            with torch.autocast(_region(device), dtype=precision, enabled=False):
                self.assertEqual(str(torch.log(x).dtype), str(precision))
                self.assertEqual(str(torch.sum(x).dtype), str(precision))
            double = torch.tensor([.25, 1., 4.], dtype=torch.float64, device=device).requires_grad_(True)
            with torch.autocast(_region(device), dtype=precision):
                inverse = double.rsqrt()
                logarithm = double.log()
            self.assertEqual(str(inverse.dtype), 'torch.float64')
            self.assertEqual(str(logarithm.dtype), 'torch.float64')
            dx, = torch.autograd.grad(inverse, (double,), grad_outputs=torch.ones_like(inverse))
            np.testing.assert_array_equal(dx.cpu().numpy(), [-4., -.5, -.0625])

    @_no_fallbacks
    def test_explicit_autocast_dtype_and_out_apply_before_reduction(self, device):
        for precision in (torch.float16, torch.bfloat16):
            x = torch.tensor([10000., 1., -10000., .5], dtype=precision, device=device)
            expected = np.cumsum(x.float().cpu().numpy().astype(np.float32))
            with torch.autocast(_region(device), dtype=precision):
                explicit = torch.cumsum(x, 0, dtype=torch.float32)
                out = torch.empty_like(x, dtype=torch.float32)
                returned = torch.cumsum(x, 0, out=out)
                self.assertIs(returned, out)
                for dtype in (precision, torch.float32, torch.float64):
                    for function in (lambda: torch.sum(x, dtype=dtype),
                                     lambda: torch.cumsum(x, 0, dtype=dtype),
                                     lambda: torch.cumprod(x / 10000., 0, dtype=dtype),
                                     lambda: torch.softmax(x, 0, dtype=dtype),
                                     lambda: torch.nn.functional.log_softmax(x, 0, dtype=dtype)):
                        value = function()
                        self.assertEqual(str(value.dtype), str(dtype))
                        _resident(self, value, device)
                total = torch.empty((), dtype=torch.float64, device=device)
                self.assertIs(torch.sum(x, dim=0, out=total), total)
            for value in (explicit, out):
                _resident(self, value, device)
                np.testing.assert_array_equal(value.cpu().numpy(), expected)
            np.testing.assert_array_equal(total.cpu().numpy(), expected[-1])
            with torch.autocast(_region(device), dtype=precision):
                inplace = torch.tensor([.25, 1., 4.], dtype=precision, device=device)
                self.assertIs(inplace.rsqrt_(), inplace)
            self.assertEqual(str(inplace.dtype), str(precision))
            np.testing.assert_array_equal(inplace.float().cpu().numpy(), [2., 1., .5])

    @_no_fallbacks
    def test_explicit_float32_softmax_backward_preserves_opmath(self, device):
        values = np.random.RandomState(82).normal(0., 2., (2, 257)).astype('float32')
        seed_values = np.random.RandomState(83).normal(0., .3, values.shape).astype('float32')
        probabilities = np.exp(values.astype(np.float64) - values.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        for precision in (torch.float16, torch.bfloat16):
            for logarithmic in (False, True):
                with self.subTest(precision=str(precision), logarithmic=logarithmic):
                    x = torch.tensor(values, device=device).requires_grad_(True)
                    seed = torch.tensor(seed_values, device=device)
                    function = torch.nn.functional.log_softmax if logarithmic else torch.nn.functional.softmax
                    with torch.autocast(_region(device), dtype=precision):
                        output = function(x, dim=-1, dtype=torch.float32)
                    gradient, = torch.autograd.grad(output, (x,), grad_outputs=seed)
                    if logarithmic:
                        expected = seed_values - probabilities * seed_values.sum(axis=1, keepdims=True)
                    else:
                        expected = probabilities * (seed_values - (probabilities * seed_values).sum(axis=1, keepdims=True))
                    self.assertEqual(str(output.dtype), 'torch.float32')
                    self.assertEqual(str(gradient.dtype), 'torch.float32')
                    _resident(self, gradient, device)
                    np.testing.assert_allclose(gradient.cpu().numpy(), expected, rtol=1e-5, atol=1e-7)

    @_no_fallbacks
    def test_cross_entropy_backward_keeps_wide_opmath_after_autocast(self, device):
        values = np.random.RandomState(81).normal(0., 3., (4, 257)).astype('float32')
        labels = torch.tensor([3, 17, -100, 101], dtype=torch.int64, device=device)
        for precision in (torch.float16, torch.bfloat16):
            scale = 1024. if precision == torch.float16 else 1.
            seed = torch.tensor([.375, .25, .5, 1.], device=device) * scale
            for input_dtype in (torch.float32, precision):
                with self.subTest(precision=str(precision), input_dtype=str(input_dtype)):
                    x = torch.tensor(values, dtype=input_dtype, device=device).requires_grad_(True)
                    rounded = x.detach().float().cpu().numpy().astype(np.float64)
                    probabilities = np.exp(rounded - rounded.max(axis=1, keepdims=True))
                    probabilities /= probabilities.sum(axis=1, keepdims=True)
                    probabilities[np.arange(4), [3, 17, 0, 101]] -= 1.
                    probabilities[2] = 0.
                    expected = probabilities * np.array([.375, .25, .5, 1.])[:, None] * scale
                    expected = torch.tensor(expected.astype(np.float32), device=device).to(input_dtype)
                    with torch.autocast(_region(device), dtype=precision):
                        losses = torch.nn.functional.cross_entropy(x, labels, reduction='none')
                    self.assertEqual(str(losses.dtype), 'torch.float32')
                    _resident(self, losses, device)
                    # The real Trainer leaves autocast before backward. Its
                    # taped CE node must not re-enable narrowing on FP32
                    # statistics or cotangents while building this derivative.
                    actual, = torch.autograd.grad(losses, (x,), grad_outputs=seed, retain_graph=True)
                    self.assertEqual(str(actual.dtype), str(input_dtype))
                    _resident(self, actual, device)
                    np.testing.assert_allclose(actual.float().cpu().numpy() / scale,
                                               expected.float().cpu().numpy() / scale,
                                               rtol=1e-5, atol=1e-7)
                    repeated, = torch.autograd.grad(losses, (x,), grad_outputs=seed)
                    _resident(self, repeated, device)
                    np.testing.assert_array_equal(actual.float().cpu().numpy(),
                                                  repeated.float().cpu().numpy())

    @_no_fallbacks
    def test_portable_grouped_convolution_rounds_autocast_operands(self, device):
        owner = importlib.import_module('jittor.nn.functional.convolution')
        for rank, groups in ((2, 1), (2, 2), (3, 1), (3, 2)):
            with self.subTest(rank=rank, groups=groups):
                shape = (1, 2) + (2,) * rank
                # Grouped BF16 scatter reductions have a separate accumulation
                # boundary. Keep their partial sums exact while still testing
                # that an FP32 operand is rounded before the contraction.
                source_value, rounded_value = ((65664., 65536.) if (rank, groups) == (3, 2)
                                                else (1e5, 99840.))
                x = torch.nn.Parameter(torch.full(shape, source_value, device=device))
                taps = np.eye(2, dtype=np.float32) if groups == 1 else np.ones((2, 1), dtype=np.float32)
                weight = torch.nn.Parameter(torch.tensor(taps.reshape((2, 2 // groups) + (1,) * rank),
                                                         device=device))
                fn = getattr(torch.nn.functional, 'conv%dd' % rank)
                with mock.patch.object(owner, 'select_kernel', return_value=None), jt.flag_scope(enable_tuner=0):
                    with torch.autocast(_region(device), dtype=torch.bfloat16):
                        output = fn(x, weight, groups=groups)
                    self.assertEqual(str(output.dtype), 'torch.bfloat16')
                    _resident(self, output, device)
                    np.testing.assert_array_equal(output.float().cpu().numpy(), np.full(shape, rounded_value))
                    output.float().sum().backward()
                    jt.sync([x.grad, weight.grad])
                for grad in (x.grad, weight.grad):
                    self.assertEqual(str(grad.dtype), 'torch.float32')
                    _resident(self, grad, device)
                np.testing.assert_array_equal(x.grad.cpu().numpy(), np.ones(shape))
                np.testing.assert_array_equal(weight.grad.cpu().numpy(),
                                               np.full(tuple(weight.shape), rounded_value * (2 ** rank)))

    @_no_fallbacks
    def test_portable_transposed_convolution_keeps_the_compute_dtype(self, device):
        owner = importlib.import_module('jittor.nn.functional.convolution_transpose')
        for rank, groups in ((2, 1), (2, 2), (3, 1)):
            with self.subTest(rank=rank, groups=groups):
                shape = (1, 2) + (2,) * rank
                x = torch.nn.Parameter(torch.full(shape, 1e5, device=device))
                taps = np.eye(2, dtype=np.float32) if groups == 1 else np.ones((2, 1), dtype=np.float32)
                weight = torch.nn.Parameter(torch.tensor(taps.reshape((2, 2 // groups) + (1,) * rank),
                                                         device=device))
                fn = getattr(torch.nn.functional, 'conv_transpose%dd' % rank)
                with mock.patch.object(owner, 'select_kernel', return_value=None), jt.flag_scope(enable_tuner=0):
                    with torch.autocast(_region(device), dtype=torch.bfloat16):
                        output = fn(x, weight, groups=groups)
                    self.assertEqual(str(output.dtype), 'torch.bfloat16')
                    _resident(self, output, device)
                    np.testing.assert_array_equal(output.float().cpu().numpy(), np.full(shape, 99840.))
                    output.float().sum().backward()
                    jt.sync([x.grad, weight.grad])
                for grad in (x.grad, weight.grad):
                    self.assertEqual(str(grad.dtype), 'torch.float32')
                    _resident(self, grad, device)
                np.testing.assert_array_equal(x.grad.cpu().numpy(), np.ones(shape))
                np.testing.assert_array_equal(weight.grad.cpu().numpy(),
                                               np.full(tuple(weight.shape), 99840. * (2 ** rank)))

    @_no_fallbacks
    def test_explicit_fp32_rmsnorm_intermediates_stay_fp32(self, device):
        source = np.linspace(-0.01, 0.01, 1024, dtype=np.float32).reshape(2, 4, 128)
        for precision in (torch.float16, torch.bfloat16):
            with self.subTest(precision=str(precision)):
                x = torch.tensor(source, device=device).to(precision)
                weight = torch.linspace(0.5, 1.5, 128, device=device)
                full = x.float()
                expected = full * torch.rsqrt(full.pow(2).mean(-1, keepdim=True) + 1e-6)
                with torch.autocast(_region(device), dtype=precision):
                    square = full.pow(2)
                    mean = square.mean(-1, keepdim=True)
                    epsilon = mean + 1e-6
                    inverse = torch.rsqrt(epsilon)
                    normalized = full * inverse
                    weighted = weight * normalized.to(precision)
                    for value in (square, mean, epsilon, inverse, normalized, weighted, full * full):
                        self.assertEqual(str(value.dtype), 'torch.float32')
                        _resident(self, value, device)
                np.testing.assert_allclose(normalized.cpu().numpy(), expected.cpu().numpy(),
                                           rtol=1e-6, atol=1e-6)

    @_no_fallbacks
    def test_autocast_elementwise_promotion_keeps_inputs_and_backward(self, device):
        for precision in (torch.float16, torch.bfloat16):
            with self.subTest(precision=str(precision)):
                full = torch.nn.Parameter(torch.tensor([0.25, 0.5], device=device))
                low = torch.tensor([0.5, 1.0], device=device).to(precision)
                with torch.autocast(_region(device), dtype=precision):
                    product = full * low
                    output = (product + full) / 2.0
                    self.assertEqual(str(product.dtype), 'torch.float32')
                    self.assertEqual(str(output.dtype), 'torch.float32')
                    self.assertEqual(str(low.dtype), str(precision))
                    self.assertEqual(str((low * low).dtype), str(precision))
                output.sum().backward()
                self.assertEqual(str(full.grad.dtype), 'torch.float32')
                _resident(self, full.grad, device)
                np.testing.assert_array_equal(full.grad.cpu().numpy(), [0.75, 1.0])

    @_no_fallbacks
    def test_autocast_power_and_rsqrt_keyword_inputs_follow_the_device_policy(self, device):
        for precision in (torch.float16, torch.bfloat16):
            with self.subTest(precision=str(precision)):
                low = torch.tensor([0.25, 0.5], device=device).to(precision)
                with torch.autocast(_region(device), dtype=precision):
                    expected_dtype = 'torch.float32' if device == 'cuda' else str(precision)
                    outputs = (low.pow(exponent=2), low ** 2, low.square(),
                               torch.pow(input=low, exponent=2), torch.rsqrt(input=low))
                    for value in outputs:
                        self.assertEqual(str(value.dtype), expected_dtype)
                        _resident(self, value, device)
                    self.assertEqual(str(torch.sqrt(input=low.float()).dtype), 'torch.float32')
                with torch.autocast(_region(device), dtype=precision, enabled=False):
                    self.assertEqual(str(low.pow(exponent=2).dtype), str(precision))
                    self.assertEqual(str(torch.rsqrt(input=low).dtype), str(precision))
                full = low.float()
                with torch.autocast(_region(device), dtype=precision):
                    total = full.sum(dtype=torch.float64)
                self.assertEqual(str(total.dtype), 'torch.float64')
                np.testing.assert_array_equal(total.cpu().numpy(), 0.75)

    @_no_fallbacks
    def test_all_fp32_matmul_really_uses_bfloat16_in_forward_and_backward(self, device):
        x = torch.nn.Parameter(torch.full((2, 2), 1e5, device=device))
        weight = torch.nn.Parameter(torch.eye(2, device=device))
        optimizer = torch.optim.SGD([x, weight], lr=0.1)
        optimizer.zero_grad()
        with torch.autocast(_region(device), dtype=torch.bfloat16):
            output = x @ weight
        self.assertEqual(str(output.dtype), "torch.bfloat16")
        _resident(self, output, device)
        np.testing.assert_array_equal(output.float().numpy(), np.full((2, 2), 99840.))
        # Backward runs after autocast exited. The op's AMP snapshot must retain
        # the bfloat16 bit, including through its shared high node flag bits.
        output.float().sum().backward()
        for grad in (x.grad, weight.grad):
            self.assertEqual(str(grad.dtype), "torch.float32")
            _resident(self, grad, device)
        np.testing.assert_array_equal(x.grad.numpy(), np.ones((2, 2)))
        np.testing.assert_array_equal(weight.grad.numpy(), np.full((2, 2), 199680.))

    @_no_fallbacks
    def test_fp32_linear_and_convolution_bias_keep_bfloat16(self, device):
        for operation in ("linear", "conv2d", "conv_transpose2d"):
            convolution = operation != "linear"
            with self.subTest(operation=operation):
                shape = (1, 2, 3, 3) if convolution else (2, 2)
                x = torch.nn.Parameter(torch.full(shape, 1e5, device=device))
                weight = torch.eye(2, device=device)
                if convolution:
                    weight = weight.reshape(2, 2, 1, 1)
                weight = torch.nn.Parameter(weight)
                bias = torch.nn.Parameter(torch.zeros(2, device=device))
                optimizer = torch.optim.SGD([x, weight, bias], lr=0.1)
                optimizer.zero_grad()
                fn = getattr(torch.nn.functional, operation)
                with torch.autocast(_region(device), dtype=torch.bfloat16):
                    output = fn(x, weight, bias)
                self.assertEqual(str(output.dtype), "torch.bfloat16")
                _resident(self, output, device)
                np.testing.assert_array_equal(output.float().numpy(), np.full(shape, 99840.))
                output.float().sum().backward()
                for grad in (x.grad, weight.grad, bias.grad):
                    self.assertEqual(str(grad.dtype), "torch.float32")
                    _resident(self, grad, device)
                    self.assertTrue(np.isfinite(grad.numpy()).all())
                np.testing.assert_array_equal(x.grad.numpy(), np.ones(shape))
                count = 9 if convolution else 2
                if convolution:
                    # The portable CPU contraction rounds an intermediate in
                    # bf16: 892928 versus the independent Torch value 897024.
                    np.testing.assert_allclose(weight.grad.numpy(), 897024., rtol=0.01)
                else:
                    np.testing.assert_array_equal(weight.grad.numpy(), np.full((2, 2), 199680.))
                np.testing.assert_array_equal(bias.grad.numpy(), np.full(2, count))

    @_no_fallbacks
    def test_bfloat16_request_overrides_a_float16_operand(self, device):
        x = torch.full((2, 2), 1e5, device=device)
        weight = torch.eye(2, dtype=torch.float16, device=device)
        with torch.autocast(_region(device), dtype=torch.bfloat16):
            output = torch.nn.functional.linear(x, weight)
        self.assertEqual(str(output.dtype), "torch.bfloat16")
        _resident(self, output, device)
        np.testing.assert_array_equal(output.float().numpy(), np.full((2, 2), 99840.))

    @_no_fallbacks
    def test_zero_scale_uses_ieee_reciprocal_and_checks_incoming_gradients(self, device):
        parameter, optimizer, scaler = _scaler_case(device, 0., 0.)
        scaler.unscale_(optimizer)
        self.assertTrue(np.isnan(parameter.grad.numpy()).all())
        scaler.step(optimizer)
        scaler.update()
        _resident(self, parameter, device)
        self.assertTrue(np.isnan(parameter.numpy()).all())
        self.assertEqual(scaler.get_scale(), 0.)

    @_no_fallbacks
    def test_backoff_may_go_below_one_and_keeps_nonfinite_gradients(self, device):
        parameter, optimizer, scaler = _scaler_case(device, 0.25, np.inf)
        scaler.step(optimizer)
        scaler.update()
        self.assertEqual(parameter.item(), 1.)
        self.assertIsNotNone(parameter.grad)
        self.assertTrue(np.isinf(parameter.grad.numpy()).all())
        self.assertEqual(scaler.get_scale(), 0.125)

    @_no_fallbacks
    def test_unscale_overflow_does_not_reclassify_a_finite_gradient(self, device):
        parameter, optimizer, scaler = _scaler_case(device, 1e-10, 1e38)
        scaler.step(optimizer)
        scaler.update()
        self.assertTrue(np.isneginf(parameter.numpy()).all())
        self.assertAlmostEqual(scaler.get_scale() / 4.000000053405728e-10, 1.)

    @_no_fallbacks
    def test_float64_check_domain_matches_each_device(self, device):
        parameter, optimizer, scaler = _scaler_case(device, 2., 1e300, torch.float64)
        scaler.unscale_(optimizer)
        np.testing.assert_allclose(parameter.grad.numpy(), [5e299], rtol=1e-14)
        scaler.step(optimizer)
        scaler.update()
        if device == "cuda":
            # PyTorch CUDA's isfinite_ensure_cuda_math takes a float.
            self.assertEqual(parameter.item(), 1.)
            self.assertEqual(scaler.get_scale(), 1.)
        else:
            np.testing.assert_allclose(parameter.numpy(), [-5e298], rtol=3e-8)
            self.assertEqual(scaler.get_scale(), 8.)

    @_no_fallbacks
    def test_growth_preserves_the_old_scale_if_float32_would_overflow(self, device):
        parameter, optimizer, scaler = _scaler_case(device, 1e38, 1.)
        scaler.step(optimizer)
        scaler.update()
        self.assertEqual(scaler.get_scale(), float(np.float32(1e38)))
        self.assertEqual(scaler.state_dict()["_growth_tracker"], 0)
        self.assertEqual(parameter.item(), 1.)

    @_no_fallbacks
    def test_nonfinite_initial_scales_follow_ieee_arithmetic(self, device):
        for initial in (np.inf, -np.inf, np.nan):
            with self.subTest(initial=initial):
                parameter, optimizer, scaler = _scaler_case(device, initial, 1.)
                scaler.step(optimizer)
                scaler.update()
                _resident(self, parameter, device)
                if np.isnan(initial):
                    self.assertTrue(np.isnan(parameter.item()))
                    self.assertTrue(np.isnan(scaler.get_scale()))
                else:
                    self.assertEqual(parameter.item(), 1.)
                    self.assertEqual(scaler.get_scale(), initial)

    @_no_fallbacks
    def test_manual_scale_update_accepts_scalars_and_checks_tensor_shape(self, device):
        parameter, _, scaler = _scaler_case(device, 2., 1.)
        for value in (0., 0.25, np.inf, np.nan):
            scaler.update(new_scale=float(value))
            if np.isnan(value):
                self.assertTrue(np.isnan(scaler.get_scale()))
            else:
                self.assertEqual(scaler.get_scale(), value)
        scaler.update(new_scale=torch.tensor(0.5, device=device))
        self.assertEqual(scaler.get_scale(), 0.5)
        with self.assertRaisesRegex(RuntimeError, "broadcast shape"):
            scaler.update(new_scale=torch.tensor([0.5], device=device))
        self.assertEqual(scaler.get_scale(), 0.5)
        with self.assertRaisesRegex(AssertionError, "1-element tensor"):
            scaler.update(new_scale=torch.tensor([0.5, 1.], device=device))
        with self.assertRaisesRegex(AssertionError, "requires_grad=False"):
            scaler.update(new_scale=torch.tensor(0.5, device=device, requires_grad=True))
        with self.assertRaisesRegex(TypeError, "new_scale"):
            scaler.update(new_scale=1)
        _resident(self, parameter, device)

    @_no_fallbacks
    def test_fp16_gradients_are_rejected_before_mutation(self, device):
        parameter, optimizer, scaler = _scaler_case(device, 2., 1., torch.float16)
        with self.assertRaisesRegex(ValueError, "unscale FP16 gradients"):
            scaler.unscale_(optimizer)
        self.assertEqual(parameter.grad.item(), 1.)

    @_no_fallbacks
    def test_each_optimizer_has_its_own_unscale_and_step_stage(self, device):
        p, first, scaler = _scaler_case(device, 4., 8.)
        q = torch.nn.Parameter(torch.tensor([1.], device=device))
        second = torch.optim.SGD([q], lr=0.1)
        q.grad = torch.tensor([12.], device=device)
        scaler.unscale_(first)
        scaler.unscale_(second)
        scaler.step(first)
        scaler.step(second)
        np.testing.assert_allclose([p.item(), q.item()], [0.8, 0.7], rtol=1e-6)
        with self.assertRaisesRegex(RuntimeError, "step.*already been called"):
            scaler.step(first)
        with self.assertRaisesRegex(RuntimeError, "after step"):
            scaler.unscale_(first)
        scaler.update()
        self.assertEqual(scaler.get_scale(), 16.)

    @_no_fallbacks
    def test_initialization_and_duplicate_unscale_are_checked(self, device):
        parameter = torch.nn.Parameter(torch.tensor([1.], device=device))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        for operation in ("update", "step", "unscale_"):
            with self.subTest(operation=operation):
                scaler = torch.amp.GradScaler(_region(device))
                with self.assertRaisesRegex(AssertionError, "_scale is None"):
                    getattr(scaler, operation)(*(() if operation == "update" else (optimizer,)))
        scaler.scale(parameter.sum())
        with self.assertRaisesRegex(AssertionError, "No inf checks"):
            scaler.update()
        parameter.grad = torch.ones_like(parameter)
        scaler.unscale_(optimizer)
        with self.assertRaisesRegex(RuntimeError, "unscale_.*already been called"):
            scaler.unscale_(optimizer)
        scaler.update(new_scale=0.25)
        self.assertEqual(scaler.get_scale(), 0.25)
        parameter.grad = torch.ones_like(parameter)
        scaler.step(optimizer)


instantiate_device_type_tests(TestAmpBoundaries, globals())


if __name__ == "__main__":
    unittest.main()
