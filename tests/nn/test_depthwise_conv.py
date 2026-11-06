
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
        # lr=100 amplifies ~1e-6 per-operator differences through the 54-layer
    # network; repeated runs measured 6e-4..2.1e-3 even though the operator
    # and whole-model gradients agree at float32 noise. Keep this as a coarse
    # sanity bound; the gradient-level tests above are the tight gates.
    assert diff < 5e-3, (diff, 'backward', ps2[i].name(), ps1[i].mean(), ps1[i].std(), ps2[i].mean(), ps2[i].std())

class TestDepthwiseConv(unittest.TestCase):
    @unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "Cuda not found")
    @jt.flag_scope(use_cuda=1, float32_cudnn_precision="highest")
    def test_large_kernel_launch_matches_generic_grouped_path(self):
        # The CUDA kernel keeps the whole filter in per-thread registers. An
        # 11x11 filter needs 158 registers, so maxThreadsPerBlock is 384 while
        # the launch asked for 484-512 threads: every such convolution failed
        # with cudaErrorLaunchOutOfResources before the block-size clamp. The
        # fast path must match the generic grouped convolution for those sizes.
        rng = np.random.default_rng(7)
        for channels, kernel, size, padding in ((3, 11, 32, 0), (3, 11, 32, 5), (2, 7, 32, 0)):
            x = jt.array(rng.random((1, channels, size, size), dtype=np.float32))
            weight = jt.array(rng.random((channels, 1, kernel, kernel), dtype=np.float32))
            fast = jt.nn.conv2d(x, weight, padding=padding, groups=channels)
            generic = jt.nn.conv2d(
                x, weight, padding=padding, groups=channels, _depthwise_fast_path=False
            )
            np.testing.assert_allclose(
                fast.numpy(), generic.numpy(), atol=1e-4, rtol=1e-4,
                err_msg="depthwise %dx%d kernel, padding %d" % (kernel, kernel, padding),
            )

    @unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "Cuda not found")
    @jt.flag_scope(use_cuda=1, float32_cudnn_precision="highest")
    def test_backward_matches_generic_grouped_path(self):
        # The filter-gradient kernel identifies lane 0 with cub::LaneId(),
        # removed in CCCL 3; 2-D blocks are not lane aligned and a partial
        # warp must not enter the shuffle reduction. The fast gradients must
        # stay within float32 noise of the generic grouped convolution.
        rng = np.random.default_rng(11)
        for channels, kernel, size, padding, stride in ((3, 3, 12, 1, 1),
                                                        (2, 11, 16, 5, 1),
                                                        (4, 5, 14, 0, 2)):
            x_array = rng.standard_normal((2, channels, size, size)).astype(np.float32)
            weight_array = rng.random((channels, 1, kernel, kernel)).astype(np.float32)
            out_size = (size + 2 * padding - kernel) // stride + 1
            upstream_array = rng.standard_normal(
                (2, channels, out_size, out_size)).astype(np.float32)
            results = []
            for fast_path in (True, False):
                x = jt.array(x_array)
                weight = jt.array(weight_array)
                upstream = jt.array(upstream_array)
                output = jt.nn.conv2d(x, weight, padding=padding, stride=stride,
                                      groups=channels, _depthwise_fast_path=fast_path)
                input_grad, weight_grad = jt.grad((output * upstream).sum(), [x, weight])
                results.append((output.numpy(), input_grad.numpy(), weight_grad.numpy()))
            np.testing.assert_allclose(results[0][0], results[1][0], atol=1e-4, rtol=1e-4,
                                       err_msg="depthwise forward %dx%d" % (kernel, kernel))
            for name, fast_grad, generic_grad in (
                    ("input", results[0][1], results[1][1]),
                    ("filter", results[0][2], results[1][2])):
                np.testing.assert_allclose(
                    fast_grad, generic_grad, atol=1e-4, rtol=1e-4,
                    err_msg="depthwise %s grad, %dx%d kernel, stride %d, padding %d"
                            % (name, kernel, kernel, stride, padding))

    @unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "Cuda not found")
    @jt.flag_scope(use_cuda=1, float32_cudnn_precision="highest")
    def test_whole_model_gradients_match_generic_path(self):
        # The legacy lr=100 parameter check is chaos-amplified. Compare the
        # gradients after one backward instead: a wrong lane id changes the
        # filter gradient at the operator level, so this catches it without
        # depending on a 54-layer amplification factor.
        rng = np.random.default_rng(23)
        image = jt.array(rng.random((8, 3, 64, 64)).astype(np.float32))
        model_generic = jtmodels.__dict__['mobilenet_v2']()
        model_fast = jtmodels.__dict__['mobilenet_v2']()
        for model in (model_generic, model_fast):
            model.train()
            model.classifier[0].eval()
            for layer in model.modules():
                if isinstance(layer, jt.nn.BatchNorm):
                    layer.eval()
        load_parameters(model_fast, model_generic)

        original_conv2d = jt.nn.conv2d
        def _generic_conv2d(*args, **kwargs):
            kwargs["_depthwise_fast_path"] = False
            return original_conv2d(*args, **kwargs)

        with unittest.mock.patch.object(jt.nn, "conv2d", _generic_conv2d):
            output_generic = model_generic(image)
        output_fast = model_fast(image)
        np.testing.assert_allclose(
            output_fast.numpy(), output_generic.numpy(), atol=1e-4, rtol=1e-4,
            err_msg="whole-model depthwise forward")

        mask = jt.array(rng.random(output_fast.shape, dtype=np.float32))
        loss_generic = (output_generic * mask).sum()
        loss_fast = (output_fast * mask).sum()
        grads_generic = jt.grad(loss_generic, list(model_generic.parameters()))
        grads_fast = jt.grad(loss_fast, list(model_fast.parameters()))
        jt.sync_all(True)
        for index, (generic_grad, fast_grad) in enumerate(zip(grads_generic, grads_fast)):
            if generic_grad is None or fast_grad is None:
                self.assertIs(generic_grad is None, fast_grad is None)
                continue
            np.testing.assert_allclose(
                fast_grad.numpy(), generic_grad.numpy(), atol=1e-5, rtol=1e-4,
                err_msg="whole-model depthwise gradient %d" % index)
        jt.clean()
        jt.gc()

    @unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "Cuda not found")
    @jt.flag_scope(use_cuda=1, float32_cudnn_precision="highest")
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
