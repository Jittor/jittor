"""Low-precision attention rounds at its public output and gradient boundary."""
import importlib
import os
import unittest

import numpy as np

import jittor as jt
from jittor._runtime.fallback import forbid_backend_fallbacks

from _helpers.sdpa_reference import reference as _reference

attention = importlib.import_module("jittor.nn.functional.attention")


def _round_reference(value, dtype):
    if dtype == "float16":
        return value.astype(np.float16).astype(np.float64)
    bits = value.astype(np.float32).view(np.uint32)
    rounded = (bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)
    return rounded.view(np.float32).astype(np.float64)


class TestAttentionOpmath(unittest.TestCase):
    def _check(self, dtype, training, amp_reg=0, head_dim=8, public=False):
        device = os.environ.get("JITTOR_TEST_DEVICE", "cpu")
        self.assertIn(device, ("cpu", "cuda"))
        rng = np.random.RandomState(7)
        arrays = [rng.randn(1, 4, 2, head_dim).astype("float32") for _ in range(3)]
        cotangent = rng.uniform(.2, 1.1, arrays[0].shape).astype("float32")
        masks = (None, np.array([[True, False], [False, False]]),
                 np.array([[0., -.3], [.2, -.1]], dtype="float32"))
        with jt.flag_scope(use_cuda=int(device == "cuda"), amp_reg=0), forbid_backend_fallbacks():
            for mask in masks:
                with self.subTest(dtype=dtype, training=training, mask=str(mask)):
                    tensors = [jt.array(value).cast(dtype) for value in arrays]
                    quantized = [value.float32().numpy().copy() for value in tensors]
                    expected = _reference(*quantized, mask, False, _round_reference(cotangent, dtype))
                    native_mask = None if mask is None else jt.array(mask).stop_grad()
                    scale = head_dim ** -.5
                    attention.LAST_FUSED_KERNEL[0] = None
                    if training:
                        with jt.flag_scope(amp_reg=amp_reg):
                            output = (attention.scaled_dot_product_attention(
                                *tensors, attn_mask=native_mask, scale=scale) if public else
                                attention._MemoryEfficientAttention.apply(
                                    *tensors, native_mask, False, scale))
                            self.assertEqual(jt.flags.amp_reg, amp_reg)
                        loss = (output.float32() * jt.array(cotangent)).sum()
                        with jt.flag_scope(amp_reg=amp_reg):
                            gradients = jt.grad(loss, tensors)
                            self.assertEqual(jt.flags.amp_reg, amp_reg)
                        actual = [output, *gradients]
                    else:
                        with jt.no_grad(), jt.flag_scope(amp_reg=amp_reg):
                            output = (attention.scaled_dot_product_attention(
                                *tensors, attn_mask=native_mask, scale=scale) if public else
                                attention._composite(
                                    *tensors, native_mask, 0., False, scale, dtype))
                            self.assertEqual(jt.flags.amp_reg, amp_reg)
                        actual = [output]
                    if public:
                        self.assertIsNone(attention.LAST_FUSED_KERNEL[0])
                    for index, (value, reference) in enumerate(zip(actual, expected)):
                        reference = _round_reference(reference, dtype)
                        self.assertEqual(str(value.dtype), dtype)
                        value.sync()
                        self.assertEqual(value.location(), "device" if device == "cuda" else "cpu")
                        observed = value.float32().numpy()
                        self.assertTrue(np.isfinite(observed).all())
                        scale = max(1e-8, float(np.max(np.abs(reference))))
                        error = float(np.max(np.abs(observed - reference))) / scale
                        # Same frozen forward/backward budgets as ecosystem
                        # attention, including the necessary final dtype cast.
                        self.assertLessEqual(error, .002 if index == 0 else .01)

    def test_bfloat16_training(self):
        self._check("bfloat16", True)

    def test_float16_training(self):
        self._check("float16", True)

    def test_bfloat16_inference(self):
        self._check("bfloat16", False)

    def test_float16_inference(self):
        self._check("float16", False)

    def test_low_precision_opmath_is_preserved_inside_autocast(self):
        self._check("bfloat16", True, jt.amp_flags.prefer_bfloat16)
        self._check("float16", False, jt.amp_flags.prefer16)

    def test_small_bfloat16_heads_use_the_public_opmath_path(self):
        for head_dim in (8, 16, 24):
            for training in (False, True):
                self._check("bfloat16", training, head_dim=head_dim, public=True)

    def test_reduced_precision_opt_in_keeps_cpu_opmath(self):
        rng = np.random.RandomState(19)
        arrays = [rng.randn(1, 2, 3, 8).astype("float32") for _ in range(3)]
        cotangent = rng.uniform(.2, 1.1, arrays[0].shape).astype("float32")
        with jt.flag_scope(use_cuda=0, amp_reg=0), forbid_backend_fallbacks():
            tensors = [jt.array(value).bfloat16() for value in arrays]
            expected = _reference(*(value.float32().numpy() for value in tensors),
                                  None, False, _round_reference(cotangent, "bfloat16"))
            attention.LAST_FUSED_KERNEL[0] = None
            output = attention.scaled_dot_product_attention(
                *tensors, allow_reduced_precision=True)
            gradients = jt.grad((output.float32() * jt.array(cotangent)).sum(), tensors)
            self.assertIsNone(attention.LAST_FUSED_KERNEL[0])
            for index, (value, reference) in enumerate(zip([output, *gradients], expected)):
                value.sync()
                self.assertEqual(value.location(), "cpu")
                self.assertEqual(str(value.dtype), "bfloat16")
                reference = _round_reference(reference, "bfloat16")
                error = np.max(abs(value.float32().numpy() - reference))
                error /= max(1e-8, float(np.max(abs(reference))))
                self.assertLessEqual(error, .002 if index == 0 else .01)

    def test_reduced_precision_request_requires_a_bool(self):
        with jt.flag_scope(use_cuda=0):
            q = jt.ones((1, 1, 2, 8), dtype="bfloat16")
            for invalid in (None, 1, "backend"):
                with self.subTest(invalid=invalid), self.assertRaisesRegex(TypeError, "must be bool"):
                    attention.scaled_dot_product_attention(q, q, q, allow_reduced_precision=invalid)

    def test_float64_first_and_second_derivatives(self):
        rng = np.random.RandomState(19)
        # Exactly representable scale isolates attention precision from the
        # separately tracked Python-scalar narrowing limitation.
        arrays = [rng.randn(1, 1, 2, 4) for _ in range(3)]
        cotangent, direction = rng.randn(2, 1, 1, 2, 4)
        with jt.flag_scope(use_cuda=0, auto_convert_64_to_32=0, amp_reg=0), forbid_backend_fallbacks():
            tensors = [jt.array(value, dtype="float64") for value in arrays]
            output = attention._MemoryEfficientAttention.apply(*tensors, None, False, .5)
            gradients = jt.grad((output * jt.array(cotangent, dtype="float64")).sum(), tensors)
            expected = _reference(*arrays, None, False, cotangent)
            for actual, reference in zip([output, *gradients], expected):
                self.assertEqual(str(actual.dtype), "float64")
                np.testing.assert_allclose(actual.numpy(), reference, atol=1e-12, rtol=1e-12)
            second = jt.grad((gradients[0] * jt.array(direction, dtype="float64")).sum(), tensors)
            epsilon = 1e-4
            plus = _reference(arrays[0] + epsilon * direction, *arrays[1:], None, False, cotangent)[1:]
            minus = _reference(arrays[0] - epsilon * direction, *arrays[1:], None, False, cotangent)[1:]
            for actual, high, low in zip(second, plus, minus):
                np.testing.assert_allclose(actual.numpy(), (high - low) / (2 * epsilon),
                                           atol=2e-8, rtol=2e-6)

    def test_rebinding_inputs_and_reusing_an_instance_retains_each_forward(self):
        rng = np.random.RandomState(23)
        arrays = [rng.randn(1, 1, 2, 4) for _ in range(3)]
        cotangent = rng.randn(1, 1, 2, 4)
        expected = _reference(*arrays, None, False, cotangent)
        with jt.flag_scope(use_cuda=0, auto_convert_64_to_32=0, amp_reg=0), forbid_backend_fallbacks():
            for index in range(3):
                with self.subTest(rebound_input=index):
                    leaves = [jt.array(value, dtype="float64") for value in arrays]
                    tensors = [leaf * 1. for leaf in leaves]
                    instance = attention._MemoryEfficientAttention()
                    output = instance(*tensors, None, False, .5)
                    output.sync()
                    tensors[index].assign(tensors[index] + .3)
                    tensors[index].sync()
                    instance(*(tensor + .7 for tensor in tensors), None, False, .5).sync()
                    gradients = jt.grad((output * jt.array(cotangent, dtype="float64")).sum(), leaves)
                    for actual, reference in zip([output, *gradients], expected):
                        np.testing.assert_allclose(actual.numpy(), reference, atol=1e-12, rtol=1e-12)
