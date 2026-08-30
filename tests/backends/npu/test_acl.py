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
    def test_float_arg_reduce_runs_on_acl(self):
        cases = [
            (jt.float32([[1, 5, 3, 5], [-2, -4, 7, 0]]), "max", 1, False,
             [1, 2], [5, 7]),
            (jt.float32([[1, 5, 3, 5], [-2, -4, 7, 0]]), "min", 0, True,
             [[1, 1, 0, 1]], [[-2, -4, 3, 0]]),
            (jt.float16([3, -1, -1, 4]), "min", 0, False, [1], [-1]),
        ]

        actual = []
        with jt.log_capture_scope(
            log_v=0, log_vprefix="acl_op_exec.cc=100"
        ) as logs:
            for x, op, dim, keepdims, _indices, _values in cases:
                indices, values = jt.arg_reduce(x, op, dim, keepdims)
                actual.append((indices.numpy(), values.numpy()))

        for (indices, values), case in zip(actual, cases):
            np.testing.assert_array_equal(indices, case[4])
            np.testing.assert_allclose(values, case[5], rtol=0, atol=0)

        messages = [log["msg"].lower() for log in logs]
        self.assertTrue(any(
            "exec acl op" in message and "arg_reduce" in message
            for message in messages
        ))
        self.assertFalse(any("compile cpu" in message for message in messages))
        self.assertFalse(any("fallback cpu" in message for message in messages))

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_float_arg_reduce_backward_runs_on_acl(self):
        cases = [
            (
                jt.float32([[1, 5, 3, 5], [-2, -4, 7, 0]]),
                "max", 1, False, jt.float32([2, 3]),
                [[0, 2, 0, 0], [0, 0, 3, 0]],
            ),
            (
                jt.float32([[1, -4, 3], [1, 8, -2]]),
                "min", 0, True, jt.float32([[4, 5, 6]]),
                [[4, 5, 0], [0, 0, 6]],
            ),
            (
                jt.float16([3, -1, -1, 4]),
                "min", -1, False, jt.float16([7]),
                [0, 7, 0, 0],
            ),
        ]

        actual = []
        with jt.log_capture_scope(
            log_v=0, log_vprefix="acl_op_exec.cc=100"
        ) as logs:
            for x, op, dim, keepdims, weight, _expected in cases:
                x.start_grad()
                _indices, values = jt.arg_reduce(x, op, dim, keepdims)
                grad = jt.grad((values * weight).sum(), x)
                actual.append(grad.numpy())

        for grad, case in zip(actual, cases):
            np.testing.assert_allclose(grad, case[5], rtol=0, atol=0)

        messages = [log["msg"].lower() for log in logs]
        self.assertTrue(any(
            "exec acl op" in message and "arg_reduce" in message
            for message in messages
        ))
        self.assertTrue(any("compile acl op" in message for message in messages))
        self.assertFalse(any("compile cpu" in message for message in messages))
        self.assertFalse(any("fallback cpu" in message for message in messages))

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_item_waits_for_acl_stream(self):
        actual = []
        with jt.log_capture_scope(
                log_v=0, log_vprefix="acl_op_exec.cc=100") as logs:
            for value in range(32):
                source = jt.array([value], dtype="int64")
                actual.append(int((source * 3 + 7).item()))

        self.assertEqual(actual, [value * 3 + 7 for value in range(32)])
        messages = [entry["msg"].lower() for entry in logs]
        self.assertTrue(any(
            "compile acl op" in message or "compile op(" in message
            for message in messages
        ))
        self.assertFalse(any("compile cpu" in message for message in messages))
        self.assertFalse(any("fallback cpu" in message for message in messages))

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

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_bfloat16_add_sub_run_on_acl(self):
        left_np = np.array([1.5, -2.0], dtype=np.float32)
        right_np = np.array([0.5, 4.0], dtype=np.float32)
        left = jt.array(left_np).bfloat16()
        right = jt.array(right_np).bfloat16()

        with jt.log_capture_scope(
                log_v=0, log_vprefix="acl_op_exec.cc=100") as logs:
            added = left + right
            subtracted = left - right
            self.assertEqual(str(added.dtype), "bfloat16")
            self.assertEqual(str(subtracted.dtype), "bfloat16")
            added, subtracted = jt.fetch_sync(
                [added.float32(), subtracted.float32()])

        np.testing.assert_allclose(added, left_np + right_np, atol=0, rtol=0)
        np.testing.assert_allclose(
            subtracted, left_np - right_np, atol=0, rtol=0)
        messages = [entry["msg"].lower() for entry in logs]
        self.assertTrue(any(
            "compile acl op" in message or "compile op(" in message
            for message in messages
        ))
        self.assertFalse(any("compile cpu" in message for message in messages))
        self.assertFalse(any("fallback cpu" in message for message in messages))

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
    def test_inference_rms_norm(self):
        rng = np.random.RandomState(2026)
        x_np = rng.randn(2, 3, 1024).astype("float32")
        gamma_np = rng.uniform(
            0.5, 1.5, size=(1024,)).astype("float32")
        expected = x_np / np.sqrt(
            np.mean(x_np * x_np, axis=-1, keepdims=True) + 1e-6)
        expected *= gamma_np

        for dtype, atol, rtol in (
            ("float32", 2e-5, 2e-5),
            ("bfloat16", 2e-2, 2e-2),
        ):
            x = getattr(jt.array(x_np), dtype)()
            gamma = jt.array(gamma_np)
            with jt.no_grad():
                actual = jt.nn._rms_norm_cuda(x, gamma, 1e-6)
                self.assertIsNotNone(actual, dtype)
                self.assertEqual(str(actual.dtype), dtype)
                actual = actual.float32().numpy()
            np.testing.assert_allclose(
                actual, expected, atol=atol, rtol=rtol)

        x = jt.array(x_np)
        gamma = jt.array(gamma_np)
        self.assertIsNone(jt.nn._rms_norm_cuda(x, gamma, 1e-6))
        with jt.no_grad():
            self.assertIsNone(jt.nn._rms_norm_cuda(x, gamma, object()))

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_inference_rotary_embedding(self):
        rng = np.random.RandomState(20260829)
        q_np = rng.randn(1, 16, 7, 128).astype("float32")
        k_np = rng.randn(1, 8, 7, 128).astype("float32")
        angles = rng.randn(1, 1, 7, 128).astype("float32")
        cos_np = np.cos(angles)
        sin_np = np.sin(angles)
        dtype_cases = (
            ("float32", 2e-5, 2e-5),
            ("float16", 3e-3, 3e-3),
            ("bfloat16", 3e-2, 3e-2),
        )

        def expected(x, cos, sin):
            half = x.shape[-1] // 2
            rotated = np.concatenate((-x[..., half:], x[..., :half]), axis=-1)
            return x * cos + rotated * sin

        with jt.no_grad(), jt.log_capture_scope(
                log_v=0, log_vprefix="acl_op_exec.cc=100") as logs:
            results = []
            for dtype, atol, rtol in dtype_cases:
                q = getattr(jt.array(q_np), dtype)()
                k = getattr(jt.array(k_np), dtype)()
                cos = getattr(jt.array(cos_np), dtype)()
                sin = getattr(jt.array(sin_np), dtype)()
                actual_q, actual_k = jt.nn.rotary_emb(
                    q, k, freq_cos=cos, freq_sin=sin)
                self.assertEqual(str(actual_q.dtype), dtype)
                self.assertEqual(str(actual_k.dtype), dtype)
                results.append((
                    dtype, atol, rtol,
                    actual_q.float32(), actual_k.float32(),
                    q.float32(), k.float32(), cos.float32(), sin.float32()))

            results = [jt.fetch_sync(list(result[3:])) for result in results]

        for result, (dtype, atol, rtol) in zip(results, dtype_cases):
            actual_q, actual_k, q, k, cos, sin = result
            np.testing.assert_allclose(
                actual_q, expected(q, cos, sin), atol=atol, rtol=rtol,
                err_msg=dtype)
            np.testing.assert_allclose(
                actual_k, expected(k, cos, sin), atol=atol, rtol=rtol,
                err_msg=dtype)
        messages = [entry["msg"].lower() for entry in logs]
        self.assertTrue(any("compile acl op" in message for message in messages))
        self.assertFalse(any("compile cpu" in message for message in messages))
        self.assertFalse(any("fallback cpu" in message for message in messages))

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_rotary_embedding_composite_gradient(self):
        rng = np.random.RandomState(20260830)
        q_np = rng.randn(1, 3, 4, 32).astype("float32")
        k_np = rng.randn(1, 2, 4, 32).astype("float32")
        cos_np = rng.randn(1, 1, 4, 32).astype("float32")
        sin_np = rng.randn(1, 1, 4, 32).astype("float32")
        q_weight = rng.randn(*q_np.shape).astype("float32")
        k_weight = rng.randn(*k_np.shape).astype("float32")

        def expected_grad(weight):
            half = weight.shape[-1] // 2
            first = (weight[..., :half] * cos_np[..., :half]
                     + weight[..., half:] * sin_np[..., half:])
            second = (weight[..., half:] * cos_np[..., half:]
                      - weight[..., :half] * sin_np[..., :half])
            return np.concatenate((first, second), axis=-1)

        q, k = jt.array(q_np), jt.array(k_np)
        with jt.log_capture_scope(
                log_v=0, log_vprefix="acl_op_exec.cc=100") as logs:
            actual_q, actual_k = jt.nn.rotary_emb(
                q, k, freq_cos=jt.array(cos_np), freq_sin=jt.array(sin_np))
            loss = ((actual_q * jt.array(q_weight)).sum()
                    + (actual_k * jt.array(k_weight)).sum())
            grad_q, grad_k = jt.grad(loss, [q, k])
            grad_q, grad_k = jt.fetch_sync([grad_q, grad_k])

        np.testing.assert_allclose(
            grad_q, expected_grad(q_weight), atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(
            grad_k, expected_grad(k_weight), atol=2e-5, rtol=2e-5)
        messages = [entry["msg"].lower() for entry in logs]
        self.assertFalse(any("compile cpu" in message for message in messages))
        self.assertFalse(any("fallback cpu" in message for message in messages))

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_inference_scaled_dot_product_attention(self):
        rng = np.random.RandomState(20260831)
        q_np = rng.randn(1, 4, 7, 64).astype("float32") * 0.1
        k_np = rng.randn(1, 2, 7, 64).astype("float32") * 0.1
        v_np = rng.randn(1, 2, 7, 64).astype("float32") * 0.1
        additive_np = rng.randn(1, 1, 7, 7).astype("float32") * 0.03

        def expected(query, key, value, causal, additive_mask=None):
            key = np.repeat(key, 2, axis=1)
            value = np.repeat(value, 2, axis=1)
            scores = np.matmul(query, np.swapaxes(key, -1, -2)) / 8.0
            if causal:
                blocked = np.triu(np.ones(scores.shape[-2:], dtype=bool), 1)
                scores = np.where(blocked, -1e30, scores)
            if additive_mask is not None:
                scores = scores + additive_mask
            weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
            weights /= weights.sum(axis=-1, keepdims=True)
            return np.matmul(weights, value)

        q, k, v = jt.array(q_np), jt.array(k_np), jt.array(v_np)
        with jt.no_grad(), jt.log_capture_scope(
                log_v=0, log_vprefix="acl_op_exec.cc=100") as logs:
            prefill = jt.nn._acl_scaled_dot_product_attention(
                q, k, v, is_causal=True, enable_gqa=True)
            decode = jt.nn._acl_scaled_dot_product_attention(
                q[:, :, :1, :], k, v, enable_gqa=True)
            additive = jt.nn._acl_scaled_dot_product_attention(
                q, k, v, attn_mask=jt.array(additive_np), enable_gqa=True)
            self.assertIsNotNone(prefill)
            self.assertIsNotNone(decode)
            self.assertIsNotNone(additive)
            prefill, decode, additive = jt.fetch_sync(
                [prefill, decode, additive])

            q_bf16 = q.bfloat16()
            k_bf16 = k.bfloat16()
            v_bf16 = v.bfloat16()
            prefill_bf16 = jt.nn._acl_scaled_dot_product_attention(
                q_bf16, k_bf16, v_bf16,
                is_causal=True, enable_gqa=True)
            decode_bf16 = jt.nn._acl_scaled_dot_product_attention(
                q_bf16[:, :, :1, :], k_bf16, v_bf16,
                enable_gqa=True)
            self.assertIsNotNone(prefill_bf16)
            self.assertIsNotNone(decode_bf16)
            self.assertEqual(str(prefill_bf16.dtype), "bfloat16")
            self.assertEqual(str(decode_bf16.dtype), "bfloat16")
            self.assertEqual(
                jt.nn._acl_scaled_dot_product_attention.backend_name,
                "acl_incre_flash_attention_v4")
            prefill_bf16, decode_bf16 = jt.fetch_sync([
                prefill_bf16.float32(), decode_bf16.float32()])

        np.testing.assert_allclose(
            prefill, expected(q_np, k_np, v_np, True), atol=3e-5, rtol=3e-5)
        np.testing.assert_allclose(
            decode, expected(q_np[:, :, :1, :], k_np, v_np, False),
            atol=3e-5, rtol=3e-5)
        np.testing.assert_allclose(
            additive, expected(q_np, k_np, v_np, False, additive_np),
            atol=3e-5, rtol=3e-5)
        np.testing.assert_allclose(
            prefill_bf16, expected(q_np, k_np, v_np, True),
            atol=2e-3, rtol=2e-2)
        np.testing.assert_allclose(
            decode_bf16,
            expected(q_np[:, :, :1, :], k_np, v_np, False),
            atol=2e-3, rtol=2e-2)

        self.assertIsNone(jt.nn._acl_scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=True))
        with jt.no_grad():
            self.assertIsNone(jt.nn._acl_scaled_dot_product_attention(
                q.float16(), k.float16(), v.float16(),
                is_causal=True, enable_gqa=True))
        messages = [entry["msg"].lower() for entry in logs]
        self.assertFalse(any("compile cpu" in message for message in messages))
        self.assertFalse(any("fallback cpu" in message for message in messages))

    @jt.flag_scope(use_acl=1)
    def test_max(self):
        x = jt.rand(3, 3)
        y = x.max(1).data
        ny = x.data.max(1)
        np.testing.assert_allclose(y, ny)
        print('test_max pass')

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_all_reduction(self):
        values = np.array([[1, -2, 3], [4, 0, -6]], dtype=np.int32)
        bool_values = values != 0
        x = jt.array(values)
        bool_x = jt.array(bool_values)
        with jt.log_capture_scope(
                log_v=0, log_vprefix="acl_op_exec.cc=100") as logs:
            full = x.all()
            full_from_list = jt.all(x, dim=[])
            by_row = jt.all(x, dim=1)
            bool_by_column = jt.all(bool_x, dim=-2)
            full, full_from_list, by_row, bool_by_column = jt.fetch_sync(
                [full, full_from_list, by_row, bool_by_column])

        np.testing.assert_array_equal(full, values.all())
        np.testing.assert_array_equal(full_from_list, values.all())
        np.testing.assert_array_equal(by_row, values.all(axis=1))
        np.testing.assert_array_equal(
            bool_by_column, bool_values.all(axis=-2))
        messages = [entry["msg"].lower() for entry in logs]
        self.assertTrue(any("compile acl op" in message for message in messages))
        self.assertFalse(any("compile cpu" in message for message in messages))
        self.assertFalse(any("fallback cpu" in message for message in messages))

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_any_reduction(self):
        values = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 0.0]],
                          dtype=np.float32)
        bool_values = values != 0
        with jt.log_capture_scope(
                log_v=0, log_vprefix="acl_op_exec.cc=100") as logs:
            full = jt.any(jt.array(values))
            by_row = jt.array(bool_values).any(dim=-1)
            full, by_row = jt.fetch_sync([full, by_row])

        np.testing.assert_array_equal(full, values.any())
        np.testing.assert_array_equal(by_row, bool_values.any(axis=-1))
        messages = [entry["msg"].lower() for entry in logs]
        self.assertTrue(any("compile acl op" in message for message in messages))
        self.assertFalse(any("compile cpu" in message for message in messages))
        self.assertFalse(any("fallback cpu" in message for message in messages))

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
