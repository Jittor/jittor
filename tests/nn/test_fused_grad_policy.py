"""Fusion dispatch follows output autograd semantics, not process mode alone."""

import unittest

import numpy as np

import jittor as jt
from jittor._runtime.core_api import _output_requires_grad, _stop_grad_outputs
from jittor.nn.backends.cudnn import _try_cudnn_conv2d
from jittor.nn.rms_norm_cuda import multihead_rms_norm_cuda
from jittor.nn.swiglu_cuda import _silu_and_mul_cuda


class TestOutputGradPolicy(unittest.TestCase):
    def test_combines_grad_mode_with_nested_tensor_inputs(self):
        trainable = jt.array([1.0])
        frozen = jt.array([2.0]).stop_grad()

        self.assertTrue(_output_requires_grad(trainable))
        self.assertTrue(_output_requires_grad([frozen, (trainable,)]))
        self.assertFalse(_output_requires_grad(frozen, {"value": frozen}))
        with jt.no_grad():
            self.assertFalse(_output_requires_grad(trainable))

    def test_stops_nested_fusion_outputs(self):
        outputs = (jt.array([1.0]), [jt.array([2.0])])
        returned = _stop_grad_outputs(outputs)
        self.assertIs(returned, outputs)
        self.assertFalse(returned[0].requires_grad)
        self.assertFalse(returned[1][0].requires_grad)


@unittest.skipUnless(jt.has_cuda, "fusion policy CUDA checks need CUDA")
class TestCudaFusionGradPolicy(unittest.TestCase):
    def test_stopped_input_uses_inference_fusion_without_no_grad_scope(self):
        raw = np.random.RandomState(0).randn(3, 16).astype("float32")
        with jt.flag_scope(use_cuda=1):
            x = jt.array(raw).stop_grad()
            output = _silu_and_mul_cuda(x)
            self.assertIsNotNone(output)
            self.assertFalse(output.requires_grad)
            expected = raw[:, :8] / (1.0 + np.exp(-raw[:, :8])) * raw[:, 8:]
            np.testing.assert_allclose(
                output.numpy(), expected, rtol=1e-5, atol=1e-5)

    def test_low_precision_cudnn_conv_keeps_training_gradients(self):
        rng = np.random.RandomState(1)
        x_raw = rng.randn(1, 2, 5, 5).astype("float32")
        weight_raw = rng.randn(3, 2, 3, 3).astype("float32")

        with jt.flag_scope(use_cuda=0):
            x_ref = jt.array(x_raw)
            weight_ref = jt.array(weight_raw)
            y_ref = jt.nn.conv2d(x_ref, weight_ref, padding=1)
            reference = jt.fetch_sync(
                [y_ref] + jt.grad(y_ref.sum(), [x_ref, weight_ref]))

        for dtype, tolerance in (("float16", 3e-2), ("bfloat16", 8e-2)):
            with self.subTest(dtype=dtype), jt.flag_scope(use_cuda=1):
                x = jt.array(x_raw).cast(dtype)
                weight = jt.array(weight_raw).cast(dtype)
                x.requires_grad = True
                weight.requires_grad = True
                output = _try_cudnn_conv2d(
                    x, weight, None, 1, 1, 1, 1)
                self.assertIsNotNone(output)
                actual = jt.fetch_sync(
                    [output] + jt.grad(output.sum(), [x, weight]))
                for got, expected in zip(actual, reference):
                    np.testing.assert_allclose(
                        got, expected, rtol=tolerance, atol=tolerance)

    def test_short_head_rms_norm_marks_inference_output_stopped(self):
        rng = np.random.RandomState(2)
        with jt.flag_scope(use_cuda=1):
            x = jt.array(rng.randn(2, 3, 96).astype("float32")).bfloat16()
            gamma = jt.ones((3, 96), dtype="float32")
            x.stop_grad()
            gamma.stop_grad()
            output = multihead_rms_norm_cuda(x, gamma)
            self.assertIsNotNone(output)
            self.assertFalse(output.requires_grad)
            self.assertTrue(np.isfinite(output.float32().numpy()).all())


if __name__ == "__main__":
    unittest.main()
