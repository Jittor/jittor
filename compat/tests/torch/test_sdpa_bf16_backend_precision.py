"""Torch default BF16 attention can use admitted native fused precision."""
import contextlib
import importlib.util
import unittest
from unittest import mock

import numpy as np
import torch
import jittor as jt

from _fixtures.sdpa_bf16_flash import cases
from _helpers import capability
from _helpers.sdpa_reference import reference
from jittor._runtime.fallback import forbid_backend_fallbacks
from jittor.compat.torch.installers.nn import attention as frontend
from jittor.nn.functional import attention as native


def _usable():
    return (capability.check_accelerator("cuda", backend=jt).enabled
            and importlib.util.find_spec("cudnn") is not None)


@unittest.skipUnless(_usable(), "needs CUDA and nvidia-cudnn-frontend headers")
class TestBFloat16BackendPrecision(unittest.TestCase):
    def setUp(self):
        self.scope = contextlib.ExitStack()
        self.scope.enter_context(jt.flag_scope(use_cuda=1, amp_reg=0))
        self.scope.enter_context(forbid_backend_fallbacks())
        self.addCleanup(self.scope.close)
        self.before = jt.core.backend_fallback_count()
        self.addCleanup(lambda: self.assertEqual(jt.core.backend_fallback_count(), self.before))
        self.addCleanup(frontend._ROUTES.clear)

    def _physical(self, values):
        for value in values:
            value.sync()
            self.assertEqual(value.device.type, "cuda")
            self.assertEqual(value.location(), "device")
            self.assertEqual(jt.core.dispatch_context([value])[0], "cuda")
            self.assertEqual(str(value.dtype).split(".")[-1], "bfloat16")

    def _close(self, value, expected, budget):
        observed = value.detach().float().cpu().numpy()
        self.assertTrue(np.isfinite(observed).all())
        error = np.max(abs(observed - expected)) / max(1e-8, float(np.max(abs(expected))))
        self.assertLessEqual(error, budget)

    def test_default_matches_independent_flash_outputs_and_all_input_gradients(self):
        for seed, cotangent_seed, data in cases():
            for training in (False, True):
                with self.subTest(seed=seed, shape=data["q"].shape, training=training):
                    tensors = [torch.tensor(data[name], device="cuda", dtype=torch.bfloat16,
                                            requires_grad=training) for name in ("q", "k", "v")]
                    # Force a new selection for each fixture. A valid inference
                    # cache hit calls cuDNN directly without publishing LAST_FUSED.
                    frontend._ROUTES.clear()
                    native.LAST_FUSED_KERNEL[0] = None
                    with contextlib.nullcontext() if training else torch.no_grad():
                        output = torch.nn.functional.scaled_dot_product_attention(*tensors)
                    self.assertIsNotNone(native.LAST_FUSED_KERNEL[0])
                    self.assertTrue(native.LAST_FUSED_KERNEL[0].__module__.endswith("cudnn_attention_cuda"))
                    self._physical([*tensors, output])
                    self._close(output, data["out"], .002)
                    if training:
                        cotangent = np.random.RandomState(cotangent_seed).uniform(
                            .2, 1.1, data["out"].shape).astype("float32")
                        gradients = torch.autograd.grad(
                            (output.float() * torch.tensor(cotangent, device="cuda")).sum(), tensors)
                        self._physical(gradients)
                        for name, value in zip(("dq", "dk", "dv"), gradients):
                            # The public default keeps its original .01 budget;
                            # CuDNN forward uses the native rounded backward.
                            self._close(value, data[name], .01)

    def test_reduced_training_executes_cudnn_forward_and_native_backward(self):
        from jittor.backends.cuda.kernels.nn import cudnn_attention_cuda as provider

        _, cotangent_seed, data = list(cases())[2]
        tensors = [torch.tensor(data[name], device="cuda", dtype=torch.bfloat16,
                                requires_grad=True) for name in ("q", "k", "v")]
        cotangent = np.random.RandomState(cotangent_seed).uniform(
            .2, 1.1, data["out"].shape).astype("float32")
        with mock.patch.object(provider, "_forward", wraps=provider._forward) as forward, \
                mock.patch.object(provider, "_backward", wraps=provider._backward) as fused_grad, \
                mock.patch.object(native, "_attention_gradients",
                                  wraps=native._attention_gradients) as native_grad:
            output = torch.nn.functional.scaled_dot_product_attention(*tensors)
            gradients = torch.autograd.grad(
                (output.float() * torch.tensor(cotangent, device="cuda")).sum(), tensors)
            self._physical([output, *gradients])
            self.assertEqual(forward.call_count, 1)
            self.assertEqual(native_grad.call_count, 1)
            self.assertEqual(fused_grad.call_count, 0)
            self._physical([native_grad.call_args.kwargs["rounded_output"]])
        for name, gradient in zip(("dq", "dk", "dv"), gradients):
            self._close(gradient, data[name], .01)

    def test_sequence_major_inputs_keep_all_gradient_layouts(self):
        for seed, cotangent_seed, data in cases():
            with self.subTest(seed=seed, shape=data["q"].shape):
                leaves = [torch.tensor(data[name].transpose(0, 2, 1, 3).copy(), device="cuda",
                                       dtype=torch.bfloat16, requires_grad=True)
                          for name in ("q", "k", "v")]
                inputs = [leaf.permute(0, 2, 1, 3) for leaf in leaves]
                cotangent = np.random.RandomState(cotangent_seed).uniform(
                    .2, 1.1, data["out"].shape).astype("float32")
                output = torch.nn.functional.scaled_dot_product_attention(*inputs)
                gradients = torch.autograd.grad(
                    (output.float() * torch.tensor(cotangent, device="cuda")).sum(), leaves)
                self._physical([output, *gradients])
                self._close(output, data["out"], .002)
                for name, gradient in zip(("dq", "dk", "dv"), gradients):
                    self._close(gradient, data[name].transpose(0, 2, 1, 3), .01)

    def test_rebinding_inputs_preserves_the_original_backward(self):
        _, cotangent_seed, data = list(cases())[2]
        cotangent = np.random.RandomState(cotangent_seed).uniform(
            .2, 1.1, data["out"].shape).astype("float32")
        for rebound in range(3):
            with self.subTest(rebound=rebound):
                leaves = [torch.tensor(data[name], device="cuda", dtype=torch.bfloat16,
                                       requires_grad=True) for name in ("q", "k", "v")]
                inputs = [leaf * 1. for leaf in leaves]
                output = torch.nn.functional.scaled_dot_product_attention(*inputs)
                output.sync()
                inputs[rebound].assign(inputs[rebound] + .3)
                inputs[rebound].sync()
                gradients = torch.autograd.grad(
                    (output.float() * torch.tensor(cotangent, device="cuda")).sum(), leaves)
                self._physical([output, *gradients])
                self._close(output, data["out"], .002)
                for name, gradient in zip(("dq", "dk", "dv"), gradients):
                    self._close(gradient, data[name], .01)

    def test_cached_route_preserves_its_precision_request(self):
        from jittor.backends.cuda.kernels.nn import cudnn_attention_cuda as provider

        _, _, data = next(cases())
        tensors = [torch.tensor(data[name], dtype=torch.bfloat16, device="cuda")
                   for name in ("q", "k", "v")]
        # The first real flash lookup publishes its availability token. Cache
        # reuse is measured after that discovery, with the stable token.
        with torch.no_grad():
            torch.nn.functional.scaled_dot_product_attention(*tensors).sync()
        frontend._ROUTES.clear()
        with torch.no_grad(), mock.patch.object(
                frontend, "_native_scaled_dot_product_attention",
                wraps=frontend._native_scaled_dot_product_attention) as walk, mock.patch.object(
                    provider, "_forward", wraps=provider._forward) as execute:
            first = torch.nn.functional.scaled_dot_product_attention(*tensors)
            first.sync()
            self.assertEqual(len(frontend._ROUTES), 1)
            second = torch.nn.functional.scaled_dot_product_attention(*tensors)
            second.sync()
            self.assertEqual(walk.call_count, 1)
            self.assertEqual(execute.call_count, 2)
            for call in execute.call_args_list:
                self._physical(call.args[:3])
        self._physical([first, second])
        self._close(first, data["out"], .002)
        np.testing.assert_array_equal(first.float().cpu().numpy(), second.float().cpu().numpy())

    def test_unadmitted_mask_causal_and_long_shapes_keep_native_opmath(self):
        rng = np.random.RandomState(19)
        for kind in ("mask", "causal", "long", "single_key"):
            with self.subTest(kind=kind):
                q = rng.randn(1, 4, 2, 8).astype("float32")
                kv_shape = (1, 4, 65 if kind == "long" else 1 if kind == "single_key" else 2, 8)
                k, v = (rng.randn(*kv_shape).astype("float32") for _ in range(2))
                tensors = [torch.tensor(value, device="cuda", dtype=torch.bfloat16,
                                        requires_grad=True) for value in (q, k, v)]
                quantized = [value.detach().float().cpu().numpy() for value in tensors]
                mask = np.array([[True, False], [True, True]]) if kind == "mask" else None
                cotangent = rng.uniform(.2, 1.1, q.shape).astype("float32")
                grad_out = jt.array(cotangent).bfloat16().float32().numpy()
                expected = reference(*quantized, mask, kind == "causal", grad_out)
                native.LAST_FUSED_KERNEL[0] = None
                output = native.scaled_dot_product_attention(
                    *tensors, attn_mask=None if mask is None else torch.tensor(mask, device="cuda"),
                    is_causal=kind == "causal", allow_reduced_precision=True)
                gradients = jt.grad((output.float32() * jt.array(cotangent)).sum(), tensors)
                self.assertIsNone(native.LAST_FUSED_KERNEL[0])
                self._physical([output, *gradients])
                for index, (value, expected_value) in enumerate(zip([output, *gradients], expected)):
                    bits = np.asarray(expected_value, dtype=np.float32).view(np.uint32)
                    rounded = ((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000).view(np.float32)
                    self._close(value, rounded, .002 if index == 0 else .01)

    def test_admitted_head_and_sequence_boundaries_reach_cudnn(self):
        rng = np.random.RandomState(19)
        for dimension in (8, 16, 24):
            for query_length in (1, 64):
                with self.subTest(dimension=dimension, query_length=query_length):
                    values = [rng.normal(0, .3, (1, 4, length, dimension)).astype("float32")
                              for length in (query_length, 64, 64)]
                    tensors = [torch.tensor(value, device="cuda", dtype=torch.bfloat16,
                                            requires_grad=True) for value in values]
                    quantized = [value.detach().float().cpu().numpy() for value in tensors]
                    cotangent = rng.uniform(.2, 1.1, values[0].shape).astype("float32")
                    grad_out = jt.array(cotangent).bfloat16().float32().numpy()
                    expected = reference(*quantized, None, False, grad_out)
                    native.LAST_FUSED_KERNEL[0] = None
                    output = torch.nn.functional.scaled_dot_product_attention(*tensors)
                    gradients = torch.autograd.grad(
                        (output.float() * torch.tensor(cotangent, device="cuda")).sum(), tensors)
                    self.assertIsNotNone(native.LAST_FUSED_KERNEL[0])
                    self.assertTrue(native.LAST_FUSED_KERNEL[0].__module__.endswith("cudnn_attention_cuda"))
                    self._physical([output, *gradients])
                    for value, expected_value in zip([output, *gradients], expected):
                        self._close(value, expected_value, .02)

    def test_dropout_and_gqa_requests_keep_the_original_math_calculation(self):
        rng = np.random.RandomState(7)
        for kind in ("dropout", "gqa"):
            with self.subTest(kind=kind):
                shapes = [(1, 4, 2, 8), *([(1, 2 if kind == "gqa" else 4, 2, 8)] * 2)]
                tensors = [torch.tensor(rng.randn(*shape).astype("float32"), device="cuda",
                                        dtype=torch.bfloat16, requires_grad=True) for shape in shapes]
                cotangent = torch.tensor(rng.uniform(.2, 1.1, shapes[0]).astype("float32"), device="cuda")
                observed = []
                for opt_in in (False, True):
                    torch.manual_seed(29)
                    native.LAST_FUSED_KERNEL[0] = None
                    output = native.scaled_dot_product_attention(
                        *tensors, dropout_p=.25 if kind == "dropout" else 0.,
                        enable_gqa=kind == "gqa", allow_reduced_precision=opt_in)
                    gradients = jt.grad((output.float32() * cotangent).sum(), tensors)
                    self.assertIsNone(native.LAST_FUSED_KERNEL[0])
                    self._physical([output, *gradients])
                    observed.append([value.detach().float().cpu().numpy().copy()
                                     for value in [output, *gradients]])
                    jt.sync_all()
                for old, new in zip(*observed):
                    np.testing.assert_array_equal(new, old)
