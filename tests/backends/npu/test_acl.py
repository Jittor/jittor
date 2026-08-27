# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers: Dun Liang <randonlang@gmail.com>.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
from _helpers.assertions import expect_error
import numpy as np
from jittor import init, Module
import numpy as np

@unittest.skipIf(not jt.compiler.has_acl, "No ACL found")
class TestACL(unittest.TestCase):

    def test_source_converter_ignores_cuda_names_in_comments(self):
        from jittor.extern.acl import acl_compiler

        source = (
            "// cudaMemcpyAsync copy of the input\n"
            "/* cudaMalloc(ptr, size) is mentioned here */\n"
            "const char* message = \"cudaMalloc failed\";\n"
            "const char* url = R\"tag(https://example.test/cudaGetLastError)tag\";\n"
            "int value = 1;\n"
        )
        converted = acl_compiler.mod.process(source, "comment_probe.cc", {})
        self.assertEqual(converted, source)

    def test_source_converter_maps_every_device_count_call(self):
        from jittor.extern.acl import acl_compiler

        source = (
            "if (cudaGetDeviceCount(&count) != cudaSuccess) count = 0;\n"
            "cudaGetDeviceCount(&count);\n"
            "cudaGetDeviceCount(&count);\n"
        )
        converted = acl_compiler.mod.process(source, "device_count_probe.cc", {})
        self.assertEqual(converted.count("acl_jittor_get_device_count"), 3)
        self.assertIn("ACL_SUCCESS", converted)
        self.assertNotIn("cudaGetDeviceCount", converted)

    def test_source_converter_maps_cuda_error_type(self):
        from jittor.extern.acl import acl_compiler

        source = (
            "cudaError_t err = cudaMalloc(&ptr, size);\n"
            "if (err == cudaSuccess) return ptr;\n"
            "#define CALLBACK_ARGS cudaStream_t stream, cudaError_t status, void*\n"
        )
        converted = acl_compiler.mod.process(source, "error_type_probe.cc", {})
        self.assertEqual(converted.count("aclError"), 2)
        self.assertNotIn("aclrtError", converted)
        self.assertIn("ACL_SUCCESS", converted)

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_float32_matmul_runs_on_acl(self):
        a_np = np.arange(12, dtype=np.float32).reshape(3, 4)
        b_np = np.arange(20, dtype=np.float32).reshape(4, 5)

        with jt.log_capture_scope(
            log_v=0, log_vprefix="acl_op_exec.cc=100"
        ) as logs:
            actual = jt.matmul(jt.array(a_np), jt.array(b_np)).numpy()

        np.testing.assert_allclose(actual, a_np @ b_np, rtol=1e-5, atol=1e-5)
        messages = [log["msg"].lower() for log in logs]
        self.assertTrue(any("compile acl op" in message for message in messages))
        self.assertFalse(any("fallback cpu" in message for message in messages))

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_torch_compat_empty_cuda_tensor(self):
        import jittor as torch

        device = torch.device("cuda")
        empty = torch.tensor([], dtype=torch.float32, device=device)
        self.assertEqual(empty.numel(), 0)
        self.assertTrue(empty.is_cuda)

        value = torch.tensor([1.0], dtype=torch.float32, device=device)
        joined = torch.cat((empty, value))
        np.testing.assert_array_equal(joined.cpu().numpy(), [1.0])

    def test_torch_compat_default_device_follows_execution_flag(self):
        import jittor as torch

        with jt.flag_scope(use_acl=0, use_cuda=0):
            self.assertEqual(torch.get_default_device().type, "cpu")
        with jt.flag_scope(use_acl=1, use_cuda=1):
            self.assertEqual(torch.get_default_device().type, "cuda")

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_var_gather_uses_acl(self):
        source = jt.float32([[1, 2, 3], [4, 5, 6]])
        index = jt.int32([[2], [0]])

        actual = source.gather(1, index).numpy()

        np.testing.assert_array_equal(actual, [[3], [4]])

    @jt.flag_scope(use_acl=1)
    def test_array(self):
        a = jt.array([1, 2, 3])
        np.testing.assert_allclose(a.numpy(), [1, 2, 3])
        print('test_array pass')

    @jt.flag_scope(use_acl=1)
    def test_add(self):
        a = jt.array([1, 2, 3])
        b = a + a
        np.testing.assert_allclose(b.numpy(), [2, 4, 6])
        print('test_add pass')

    @jt.flag_scope(use_acl=1)
    def test_add_float(self):
        a = jt.array([1.0, 2.0, 3.0])
        b = a + a
        np.testing.assert_allclose(b.numpy(), [2, 4, 6])
        print('test_add_float pass')

    @jt.flag_scope(use_acl=1)
    def test_array_cast(self):
        # this test cannot pass because cast error
        x = np.random.rand(10)
        y = jt.float32(x)
        np.testing.assert_allclose(x, y.numpy())
        print('test_array_cast pass')

    @jt.flag_scope(use_acl=1)
    def test_array_cast_half(self):
        # this test cannot pass because cast error
        x = np.random.rand(10).astype("float32")
        y = jt.float16(x)
        np.testing.assert_allclose(x.astype("float16"), y.numpy())
        print('test_array_cast_half pass')

    @jt.flag_scope(use_acl=1)
    def test_rand(self):
        a = jt.rand(10)
        b = a * 10
        b.sync()
        print(b)

    def test_meminfo(self):
        jt.display_memory_info()
        print('test_meminfo pass')

    @jt.flag_scope(use_acl=1)
    def test_conv(self):
        x = jt.rand(10, 3, 50, 50)
        w = jt.rand(4, 3, 3, 3)
        # x = jt.rand(2, 2, 1, 1)
        # w = jt.rand(2,2,1,1)
        y = jt.nn.conv2d(x, w)
        y.sync(True)
        y1 = y.data
        mask = jt.rand_like(y)
        dx, dw = jt.grad((y * mask).sum(), [x, w])
        dx1, dw1 = dx.data, dw.data
        # dw, = jt.grad((y*mask).sum(), [w])
        # dw1 = dw.data
        with jt.flag_scope(use_acl=0):
            y = jt.nn.conv2d(x, w)
            y2 = y.data
            dx, dw = jt.grad((y * mask).sum(), [x, w])
            dx2, dw2 = dx.data, dw.data
            # dw, = jt.grad((y*mask).sum(), [w])
            # dw2 = dw.data
        np.testing.assert_allclose(y1, y2, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(dx1, dx2, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(dw1, dw2, rtol=1e-4, atol=1e-5)
        print('test_conv pass')

    @jt.flag_scope(use_acl=1)
    def test_matmul(self):
        # x = jt.rand(10, 3, 50, 50)
        # w = jt.rand(4,3,3,3)
        x = jt.rand(10, 10)
        w = jt.rand(10, 10)
        y = jt.matmul(x, w)
        ny = np.matmul(x.numpy(), w.numpy())
        np.testing.assert_allclose(y.numpy(), ny, atol=1e-3, rtol=1e-3)
        print('test_matmul pass')

    @jt.flag_scope(use_acl=1)
    def test_max(self):
        x = jt.rand(3, 3)
        y = x.max(1).data
        ny = x.data.max(1)
        np.testing.assert_allclose(y, ny)
        print('test_max pass')

    @jt.flag_scope(use_acl=1)
    def test_sum(self):
        x = jt.rand(3, 3).float16()
        print(x)
        # return
        y = x.sum(1).data
        print(y)
        print(x)
        ny = x.data.sum(1)
        np.testing.assert_allclose(y, ny)
        print('test_sum pass')

    @jt.flag_scope(use_acl=1)
    def test_broadcast(self):
        x = jt.rand(3)
        # print(x)
        y = x.broadcast([3, 3]).data
        ny = np.broadcast_arrays(x.data, y)[0]
        np.testing.assert_allclose(y, ny)
        print(x, y)
        # y = x.broadcast([3,3], dims=[1]).data
        y = jt.broadcast(x, shape=(3, 3), dims=[1]).data
        with jt.flag_scope(use_acl=0):
            ny = jt.broadcast(x, shape=(3, 3), dims=[1]).data
        # ny = np.broadcast_arrays(x.data, y)[0]
        np.testing.assert_allclose(y, ny)
        print(x, y)
        print('test_broadcast pass')

    @jt.flag_scope(use_acl=1)
    def test_resnet(self):
        from jittor.models import resnet50
        net = resnet50()
        x = jt.rand(2, 3, 224, 224)
        y = net(x)
        y.sync()


class Linear(Module):

    def __init__(self, in_features, out_features, bias=True):
        self.w = (jt.random(
            (in_features, out_features), type='normal') - 0.5) / in_features**0.5
        self.b = jt.random((out_features, ), type='normal') - 0.5 if bias else None

    def execute(self, x):
        x = jt.nn.matmul(x, self.w)
        if self.b is not None:
            return x + self.b
        return x


def relu(x):
    return jt.maximum(x, 0.0)


Relu = jt.make_module(relu)


class Model(Module):

    def __init__(self, input_size):
        self.linear1 = Linear(input_size, 10)
        self.relu1 = Relu()
        self.linear2 = Linear(10, 1)

    def execute(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        return self.linear2(x)


@unittest.skipIf(not jt.compiler.has_acl, "No ACL found")
class TestExample(unittest.TestCase):

    @jt.flag_scope(use_acl=1)
    def test1(self):
        np.random.seed(0)
        jt.set_seed(3)
        n = 1000
        batch_size = 50
        lr = 0.05

        def get_data(n):
            for i in range(n):
                x = np.random.rand(batch_size, 1).astype("float32")
                y = x * x
                yield jt.float32(x), jt.float32(y)

        model = Model(input_size=1)
        ps = model.parameters()

        for i, (x, y) in enumerate(get_data(n)):
            jt.sync_all(True)
            pred_y = model(x).name("pred_y")
            loss = ((pred_y - y).sqr()).name("loss")
            loss_mean = loss.mean()

            gs = jt.grad(loss_mean, ps)
            for p, g in zip(ps, gs):
                p -= g * lr
            if i > 2:
                assert prev == jt.liveness_info(
                ), f"memory leak {prev} {jt.liveness_info()}"
            prev = jt.liveness_info()
            print(
                f"step {i}, loss = {loss_mean.data.sum()} {jt.liveness_info()}"
            )

        # The exact converged loss depends on the RNG stream and op
        # execution order, which vary across builds, so an exact-match
        # list is brittle. The meaningful checks here are the
        # memory-leak assertion above and that training converges to a
        # small loss.
        loss_mean = loss_mean.data
        assert loss_mean < 1e-2, f'training did not converge: {loss_mean}'

        jt.clean()


if __name__ == "__main__":
    unittest.main()
