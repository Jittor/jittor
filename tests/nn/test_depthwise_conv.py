
from _helpers import capability as _test_capability
# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers:
#     Guoye Yang <498731903@qq.com>
#     Dun Liang <randonlang@gmail.com>.
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import unittest.mock

import jittor as jt
import numpy as np
import jittor.models as jtmodels


class TestDepthwiseConvCPU(unittest.TestCase):
    @staticmethod
    def _reference(x, weight, stride, padding, dilation):
        n, channels, height, width = x.shape
        kernel_height, kernel_width = weight.shape[2:]
        output_height = (
            height + 2 * padding[0] - dilation[0] * (kernel_height - 1) - 1
        ) // stride[0] + 1
        output_width = (
            width + 2 * padding[1] - dilation[1] * (kernel_width - 1) - 1
        ) // stride[1] + 1
        output = np.zeros((n, channels, output_height, output_width), dtype=np.float32)
        for batch in range(n):
            for channel in range(channels):
                for row in range(output_height):
                    for column in range(output_width):
                        total = 0.0
                        for kernel_row in range(kernel_height):
                            input_row = row * stride[0] - padding[0] + kernel_row * dilation[0]
                            for kernel_column in range(kernel_width):
                                input_column = (
                                    column * stride[1]
                                    - padding[1]
                                    + kernel_column * dilation[1]
                                )
                                if 0 <= input_row < height and 0 <= input_column < width:
                                    total += (
                                        x[batch, channel, input_row, input_column]
                                        * weight[channel, 0, kernel_row, kernel_column]
                                    )
                        output[batch, channel, row, column] = total
        return output

    def test_cpu_fallback_matches_grouped_convolution_baseline(self):
        x_array = (
            np.arange(40, dtype=np.float32).reshape(1, 2, 4, 5) - 17.0
        ) / 8.0
        weight_array = (
            np.arange(12, dtype=np.float32).reshape(2, 1, 3, 2) - 4.0
        ) / 7.0
        stride = (1, 2)
        padding = (1, 0)
        dilation = (1, 1)
        with jt.flag_scope(use_cuda=0):
            output = jt.nn.DepthwiseConv(stride, padding, dilation)(
                jt.array(x_array), jt.array(weight_array)
            )
            actual = output.numpy()
        expected = self._reference(x_array, weight_array, stride, padding, dilation)
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
        self.assertEqual(actual.shape, (1, 2, 4, 2))

    def test_cpu_backward_matches_independent_reference(self):
        x_array = np.arange(18, dtype=np.float32).reshape(1, 2, 3, 3) / 5.0
        weight_array = np.array(
            [[[[1.0, -2.0], [0.5, 3.0]]],
             [[[2.0, 1.0], [-1.0, 0.25]]]], dtype=np.float32
        )
        upstream = np.arange(8, dtype=np.float32).reshape(1, 2, 2, 2) / 7.0
        expected_x = np.zeros_like(x_array)
        expected_weight = np.zeros_like(weight_array)
        for channel in range(2):
            for row in range(2):
                for column in range(2):
                    cotangent = upstream[0, channel, row, column]
                    for kernel_row in range(2):
                        for kernel_column in range(2):
                            input_row = row + kernel_row
                            input_column = column + kernel_column
                            expected_x[0, channel, input_row, input_column] += (
                                cotangent * weight_array[channel, 0, kernel_row, kernel_column]
                            )
                            expected_weight[channel, 0, kernel_row, kernel_column] += (
                                cotangent * x_array[0, channel, input_row, input_column]
                            )
        with jt.flag_scope(use_cuda=0):
            x = jt.array(x_array)
            weight = jt.array(weight_array)
            output = jt.nn.DepthwiseConv()(x, weight)
            grads = jt.grad((output * jt.array(upstream)).sum(), [x, weight])
            actual_x, actual_weight = jt.fetch_sync(grads)
        np.testing.assert_allclose(actual_x, expected_x, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(actual_weight, expected_weight, atol=1e-6, rtol=1e-6)


@unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "Cuda not found")
class TestDepthwiseConvFp32Fma(unittest.TestCase):
    """Public CUDA Conv1d cancellation/VJP regression, not a PPO acceptance."""

    def test_strict_fp32_accumulated_input_vjp(self):
        a, b = np.float32(1 + 2 ** -23), np.float32(1 - 2 ** -23)
        scale, residual = np.float32(2 ** 30), np.float32(-2 ** -16)
        scaled_b = np.float32(scale * b)
        x_array = np.array([[[-1, b], [-1, b]]], dtype=np.float32)
        weight_array = np.array([[[a, 1]], [[1, a]]], dtype=np.float32)
        dy_array = np.array([[[-scale, scaled_b, 0],
                              [scaled_b, -scale, 0]]], dtype=np.float32)
        with jt.runtime.scope(use_cuda=1, device_id=0, amp_reg=0, no_grad=0,
                              backend_fallback="error", cuda_kernel_math="strict"):
            self.assertEqual(jt.core.backend_fallback_count(), 0)
            for repeat in range(2):
                with self.subTest(repeat=repeat):
                    x, weight, dy = map(jt.array, (x_array, weight_array, dy_array))
                    output = jt.nn.conv1d(x, weight, padding=1, groups=2)
                    dx = jt.grad((output * dy).sum(), x)
                    jt.sync_all()
                    for value in (x, weight, dy, output, dx):
                        self.assertEqual(jt.core.dispatch_context([value]), ("cuda", 0))
                        self.assertEqual(value.location(), "device")
                        self.assertEqual(int(value.device_id), 0)
                    actual = dx.numpy().copy()
                    centers = actual[0, :, 0]
                    self.assertTrue(bool(np.logical_or(centers == 0, centers == residual).all()))
                    self.assertTrue(bool((np.abs(centers - residual) <=
                                          (1e-5 + 1e-5 * np.abs(residual))).any()),
                                    "FMA input VJP residual must survive one accumulation order")
                    self.assertTrue(bool(np.array_equal(actual[0, :, 1],
                        np.array([scaled_b, np.float32(-scale * a)], dtype=np.float32))))
                    self.assertTrue(bool(np.array_equal(x.numpy(), x_array)))
                    self.assertTrue(bool(np.array_equal(weight.numpy(), weight_array)))
                    self.assertTrue(bool(np.array_equal(dy.numpy(), dy_array)))
            self.assertEqual(jt.core.backend_fallback_count(), 0)

    def test_strict_fp32_conv1d_cancellation_and_vjp(self):
        a = np.float32(1.0 + 2.0 ** -23)
        b = np.float32(1.0 - 2.0 ** -23)
        residual = np.float32(-2.0 ** -46)
        with jt.runtime.scope(use_cuda=1, device_id=0, amp_reg=0, no_grad=0,
                              backend_fallback="error", cuda_kernel_math="strict"):
            self.assertEqual(jt.core.backend_fallback_count(), 0,
                             "pre-existing backend fallback")
            for batch in (1, 2):
                with self.subTest(batch=batch):
                    x_array = np.broadcast_to(
                        np.array([-1.0, b], dtype=np.float32), (batch, 2, 2)).copy()
                    w_array = np.broadcast_to(
                        np.array([1.0, a], dtype=np.float32), (2, 1, 2)).copy()
                    x, weight = jt.array(x_array), jt.array(w_array)
                    flags = (bool(x.requires_grad), bool(weight.requires_grad))
                    self.assertTrue(all(flags), "native public operands must be trainable")
                    output = jt.nn.conv1d(x, weight, padding=1, groups=2)
                    cropped = output[:, :, :2]
                    dx, dw = jt.grad(cropped[:, :, 1].sum(), [x, weight])
                    jt.sync_all()
                    for tensor in (x, weight, output, cropped, dx, dw):
                        self.assertEqual(jt.core.dispatch_context([tensor]), ("cuda", 0),
                                         "actual CUDA0 dispatch required")
                        self.assertEqual(int(tensor.device_id), 0, "actual CUDA0 ordinal required")
                        self.assertEqual(tensor.location(), "device", "actual CUDA storage required")
                    actual_y = output.numpy().copy()
                    actual_crop = cropped.numpy().copy()
                    actual_dx, actual_dw = dx.numpy().copy(), dw.numpy().copy()
                    expected_y = np.broadcast_to(
                        np.array([-a, residual, b], dtype=np.float32), (batch, 2, 3))
                    expected_dx = np.broadcast_to(
                        np.array([1.0, a], dtype=np.float32), (batch, 2, 2))
                    expected_dw = np.broadcast_to(
                        np.array([-batch, np.float32(batch) * b], dtype=np.float32), (2, 1, 2))
                    # Full-value exact supplemental precision check, stronger than
                    # the unchanged 1e-5 independent public CUDA comparison.
                    self.assertTrue(bool(np.array_equal(actual_y, expected_y)),
                                    "complete fused forward")
                    self.assertTrue(bool(np.array_equal(actual_crop, expected_y[:, :, :2])),
                                    "complete crop")
                    self.assertTrue(bool(np.array_equal(actual_dx, expected_dx)),
                                    "complete analytic input VJP")
                    self.assertTrue(bool(np.array_equal(actual_dw, expected_dw)),
                                    "complete analytic weight VJP")
                    self.assertTrue(bool(np.array_equal(x.numpy(), x_array)),
                                    "original input bytes unchanged")
                    self.assertTrue(bool(np.array_equal(weight.numpy(), w_array)),
                                    "original weight bytes unchanged")
                    self.assertEqual((bool(x.requires_grad), bool(weight.requires_grad)), flags,
                                     "original input flags unchanged")
            self.assertEqual(jt.core.backend_fallback_count(), 0, "CUDA test backend fallback")


def load_parameters(m1, m2):
    m1.save('/tmp/temp.pk')
    m2.load('/tmp/temp.pk')

def compare_parameters(m1, m2):
    ps1 = m1.parameters()
    ps2 = m2.parameters()
    for i in range(len(ps1)):
        x = ps1[i].data + 1e-8
        y = ps2[i].data + 1e-8
        relative_error = abs(x - y) / abs(y)
        diff = relative_error.mean()
        assert diff < 1e-4, (diff, 'backward', ps2[i].name(), ps1[i].mean(), ps1[i].std(), ps2[i].mean(), ps2[i].std())

class TestDepthwiseConv(unittest.TestCase):
    @unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "Cuda not found")
    @jt.flag_scope(use_cuda=1)
    def test_data(self):
        test_img = np.random.random((64,3,224,224)).astype('float32')
        jittor_test_img = jt.array(test_img)
        lr = 100

        jittor_model = jtmodels.__dict__['mobilenet_v2']()
        jittor_model2 = jtmodels.__dict__['mobilenet_v2']()
        # Set eval to avoid dropout layer & bn errors
        jittor_model.train()
        jittor_model.classifier[0].eval()
        for m in jittor_model.modules():
            if isinstance(m, jt.nn.BatchNorm):
                m.eval()

        jittor_model2.train()
        jittor_model2.classifier[0].eval()
        for m in jittor_model2.modules():
            if isinstance(m, jt.nn.BatchNorm):
                m.eval()

        load_parameters(jittor_model2, jittor_model)
        # Model 1 runs the generic grouped path, model 2 the depthwise CUDA
        # kernel, and the two are compared below. The switch used to be the
        # module attribute ``is_depthwise_conv``; the choice now belongs to
        # jt.nn.conv2d (so that a layer built before `use_cuda = 1` still gets
        # the fast path), and the private keyword is the way to turn it off.
        # ``is_depthwise_conv`` stays as the descriptive attribute it names.
        original_conv2d = jt.nn.conv2d
        def _generic_conv2d(*args, **kwargs):
            kwargs["_depthwise_fast_path"] = False
            return original_conv2d(*args, **kwargs)
        cnt = 0
        for m in jittor_model2.modules():
            if isinstance(m, jt.nn.Conv):
                if (m.is_depthwise_conv):
                    cnt += 1
        assert cnt == 17, (cnt, '!=', 17)
        jt_optimizer = jt.nn.SGD(jittor_model.parameters(), lr = lr)
        jt_optimizer2 = jt.nn.SGD(jittor_model2.parameters(), lr = lr)

        with unittest.mock.patch.object(jt.nn, "conv2d", _generic_conv2d):
            jittor_result = jittor_model(jittor_test_img)
        mask = jt.random(jittor_result.shape, jittor_result.dtype)
        loss = jittor_result * mask
        jt_optimizer.step(loss)
        jt.sync_all(True)

        jittor_result2 = jittor_model2(jittor_test_img)
        loss = jittor_result2 * mask

        x = jittor_result2.data + 1e-8
        y = jittor_result.data + 1e-8
        relative_error = abs(x - y) / abs(y)
        diff = relative_error.mean()
        assert diff < 1e-4, (diff, 'forword')

        jt_optimizer2.step(loss)
        jt.sync_all(True)
        compare_parameters(jittor_model, jittor_model2)


        jt.clean()
        jt.gc()

if __name__ == "__main__":
    unittest.main()
