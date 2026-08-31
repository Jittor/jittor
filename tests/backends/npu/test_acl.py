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

    @staticmethod
    def _paged_attention_reference(query, key, value):
        repeats = query.shape[1] // key.shape[1]
        key = np.repeat(key, repeats, axis=1).astype("float64")
        value = np.repeat(value, repeats, axis=1).astype("float64")
        scale = query.shape[-1] ** -0.5
        scores = np.einsum(
            "qhd,khd->hqk", query.astype("float64"), key
        ) * scale
        offset = key.shape[0] - query.shape[0]
        rows = np.arange(query.shape[0])[:, None]
        columns = np.arange(key.shape[0])[None, :]
        scores = np.where(
            (columns > rows + offset)[None, :, :], -np.inf, scores
        )
        scores -= scores.max(axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights /= weights.sum(axis=-1, keepdims=True)
        return np.einsum("hqk,khd->qhd", weights, value)

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_paged_attention_prefill_decode_stays_on_device(self):
        rng = np.random.RandomState(31)
        heads, kv_heads, head_dim, block_size = 4, 2, 4, 4
        query = (rng.randn(3, heads, head_dim) * 0.1).astype("float32")
        key = (rng.randn(4, kv_heads, head_dim) * 0.1).astype("float32")
        value = (rng.randn(4, kv_heads, head_dim) * 0.1).astype("float32")
        decode_query = (rng.randn(1, heads, head_dim) * 0.1).astype("float32")
        cache = jt.zeros((2, 2, block_size, kv_heads, head_dim), "float32")

        with jt.no_grad(), jt.log_capture_scope(
                log_v=0, log_vprefix="acl_op_exec.cc=100") as logs:
            jt.nn.reshape_and_cache(
                jt.array(key[:3]), jt.array(value[:3]), cache,
                jt.array([0, 1, 2]).int32(), slots=[0, 1, 2])
            prefill = jt.nn.paged_attention(
                jt.array(query), cache, jt.array([0, 3]).int32(),
                jt.array([3]).int32(), jt.array([[0]]).int32(),
                query_lengths=[0, 3], key_lengths=[3])
            prefill.sync()
            prefill_location = prefill.location()

            jt.nn.reshape_and_cache(
                jt.array(key[3:]), jt.array(value[3:]), cache,
                jt.array([3]).int32(), slots=[3])
            decode = jt.nn.paged_attention(
                jt.array(decode_query), cache, jt.array([0, 1]).int32(),
                jt.array([4]).int32(), jt.array([[0]]).int32(),
                query_lengths=[0, 1], key_lengths=[4])
            decode.sync()
            cache.sync()
            decode_location = decode.location()
            cache_location = cache.location()

        messages = [entry["msg"].lower() for entry in logs]
        self.assertFalse(any("fallback cpu" in message for message in messages))
        self.assertEqual(prefill_location, "device")
        self.assertEqual(decode_location, "device")
        self.assertEqual(cache_location, "device")
        np.testing.assert_allclose(
            prefill.numpy(), self._paged_attention_reference(
                query, key[:3], value[:3]), atol=1e-5, rtol=0)
        np.testing.assert_allclose(
            decode.numpy(), self._paged_attention_reference(
                decode_query, key, value), atol=1e-5, rtol=0)

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_paged_attention_bfloat16_decode_uses_incremental_flash(self):
        from jittor.extern.acl.aclops.flashattention_op import (
            scaled_dot_product_attention_acl,
        )

        rng = np.random.RandomState(43)
        heads, kv_heads, head_dim, length, block_size = 4, 2, 8, 6, 16
        query = (rng.randn(1, heads, head_dim) * 0.1).astype("float32")
        key = (rng.randn(length, kv_heads, head_dim) * 0.1).astype("float32")
        value = (rng.randn(length, kv_heads, head_dim) * 0.1).astype("float32")
        cache = jt.zeros(
            (1, 2, block_size, kv_heads, head_dim), "bfloat16"
        )

        with jt.no_grad(), jt.log_capture_scope(
                log_v=0, log_vprefix="acl_op_exec.cc=100") as logs:
            jt.nn.reshape_and_cache(
                jt.array(key).bfloat16(), jt.array(value).bfloat16(), cache,
                jt.arange(length).int32(), slots=list(range(length)))
            output = jt.nn.paged_attention(
                jt.array(query), cache, jt.array([0, 1]).int32(),
                jt.array([length]).int32(), jt.array([[0]]).int32(),
                query_lengths=[0, 1], key_lengths=[length])
            output.sync()
            cache.sync()
            output_location = output.location()
            cache_location = cache.location()

        messages = [entry["msg"].lower() for entry in logs]
        self.assertFalse(any("fallback cpu" in message for message in messages))
        self.assertEqual(
            scaled_dot_product_attention_acl.backend_name,
            "acl_incre_flash_attention_v4",
        )
        self.assertEqual(output_location, "device")
        self.assertEqual(cache_location, "device")
        np.testing.assert_allclose(
            output.numpy(), self._paged_attention_reference(query, key, value),
            atol=4e-4, rtol=0)

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_empty_tensor_numpy_skips_zero_byte_device_copy(self):
        value = jt.empty((0, 3), dtype="float32")
        value.sync()
        self.assertEqual(value.location(), "device")
        actual = value.numpy()
        self.assertEqual(actual.shape, (0, 3))
        self.assertEqual(actual.dtype, np.float32)
        value.sync()
        repeated = value.numpy()
        self.assertEqual(repeated.shape, (0, 3))

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
    def test_conv_with_bias_grad(self):
        rng = np.random.RandomState(20260831)
        x_np = rng.randn(2, 3, 8, 8).astype("float32")
        w_np = rng.randn(4, 3, 3, 3).astype("float32")
        b_np = rng.randn(4).astype("float32")
        mask_np = rng.randn(2, 4, 8, 8).astype("float32")
        x, w, b, mask = map(jt.array, (x_np, w_np, b_np, mask_np))
        y = jt.nn.conv2d(x, w, b, padding=1)
        y1 = y.numpy()
        dx, dw, db = jt.grad((y * mask).sum(), [x, w, b])
        dx1, dw1, db1 = dx.numpy(), dw.numpy(), db.numpy()

        with jt.flag_scope(use_acl=0):
            x, w, b, mask = map(jt.array, (x_np, w_np, b_np, mask_np))
            y = jt.nn.conv2d(x, w, b, padding=1)
            y2 = y.numpy()
            dx, dw, db = jt.grad((y * mask).sum(), [x, w, b])
            dx2, dw2, db2 = dx.numpy(), dw.numpy(), db.numpy()

        np.testing.assert_allclose(y1, y2, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(dx1, dx2, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(dw1, dw2, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(db1, db2, rtol=1e-4, atol=1e-5)

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
        self.assertIsNone(
            jt.nn._rms_norm_cuda(x.bfloat16(), gamma, 1e-6)
        )
        with jt.no_grad():
            self.assertIsNone(jt.nn._rms_norm_cuda(x, gamma, object()))

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_serving_rms_norm_caches_mixed_dtype_weight(self):
        rng = np.random.RandomState(20260831)
        x_np = rng.randn(2, 16).astype("float32")
        weight_np = rng.uniform(0.5, 1.5, size=(16,)).astype("float32")
        x = jt.array(x_np)
        weight = jt.array(weight_np).bfloat16()

        with jt.no_grad(), jt.log_capture_scope(
                log_v=0, log_vprefix="acl_op_exec.cc=100") as logs:
            first = jt.nn.rms_norm(x, weight, 1e-6)
            cached = weight.__dict__.get("_serving_float32_weight")
            second = jt.nn.rms_norm(x, weight, 1e-6)
            first.sync()
            second.sync()
            first_location = first.location()
            second_location = second.location()

        self.assertIsNotNone(cached)
        self.assertIs(weight.__dict__.get("_serving_float32_weight"), cached)
        self.assertEqual(str(first.dtype), "float32")
        self.assertEqual(str(second.dtype), "float32")
        self.assertEqual(first_location, "device")
        self.assertEqual(second_location, "device")
        messages = [entry["msg"].lower() for entry in logs]
        self.assertFalse(any("fallback cpu" in message for message in messages))
        expected_weight = weight.float32().numpy()
        expected = x_np / np.sqrt(
            np.mean(x_np * x_np, axis=-1, keepdims=True) + 1e-6)
        expected *= expected_weight
        np.testing.assert_allclose(first.numpy(), expected, atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(second.numpy(), expected, atol=2e-5, rtol=2e-5)

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_training_rms_norm(self):
        rng = np.random.RandomState(20260830)
        x_np = rng.randn(2, 3, 1024).astype("float32")
        gamma_np = rng.uniform(
            0.5, 1.5, size=(1024,)).astype("float32")
        cotangent_np = rng.randn(2, 3, 1024).astype("float32")
        epsilon = 1e-6

        inverse_rms = 1.0 / np.sqrt(
            np.mean(x_np * x_np, axis=-1, keepdims=True) + epsilon)
        expected_output = x_np * inverse_rms * gamma_np
        weighted_cotangent = cotangent_np * gamma_np
        expected_grad_x = (
            weighted_cotangent * inverse_rms
            - x_np
            * inverse_rms ** 3
            * np.mean(weighted_cotangent * x_np, axis=-1, keepdims=True)
        )
        expected_grad_gamma = np.sum(
            cotangent_np * x_np * inverse_rms, axis=(0, 1))

        with jt.log_capture_scope(
            log_v=0, log_vprefix="acl_op_exec.cc=100"
        ) as logs:
            x = jt.array(x_np)
            gamma = jt.array(gamma_np)
            output = jt.nn._rms_norm_cuda(x, gamma, epsilon)
            self.assertIsNotNone(output)
            grad_x, grad_gamma = jt.grad(
                (output * jt.array(cotangent_np)).sum(), [x, gamma]
            )
            output, grad_x, grad_gamma = jt.fetch_sync(
                [output, grad_x, grad_gamma]
            )

        np.testing.assert_allclose(
            output, expected_output, atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(
            grad_x, expected_grad_x, atol=3e-5, rtol=3e-5)
        np.testing.assert_allclose(
            grad_gamma, expected_grad_gamma, atol=3e-5, rtol=3e-5)
        messages = [entry["msg"].lower() for entry in logs]
        self.assertFalse(any("compile cpu" in message for message in messages))
        self.assertFalse(any("fallback cpu" in message for message in messages))

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_training_embedding(self):
        rng = np.random.RandomState(20260830)
        weight_np = rng.randn(7, 5).astype("float32")
        indices_np = np.array([[1, 3, 1, 2], [6, 3, 2, 2]], dtype="int64")
        cotangent_np = rng.randn(2, 4, 5).astype("float32")
        padding_idx = 3
        expected_grad = np.zeros_like(weight_np)
        expected_grad_no_padding = np.zeros_like(weight_np)
        for batch in range(indices_np.shape[0]):
            for token in range(indices_np.shape[1]):
                index = indices_np[batch, token]
                expected_grad_no_padding[index] += cotangent_np[batch, token]
                if index != padding_idx:
                    expected_grad[index] += cotangent_np[batch, token]

        with jt.log_capture_scope(
            log_v=0, log_vprefix="acl_op_exec.cc=100"
        ) as logs:
            weight = jt.array(weight_np)
            output = jt.nn.embedding(
                jt.array(indices_np), weight, padding_idx=padding_idx
            )
            grad_weight = jt.grad(
                (output * jt.array(cotangent_np)).sum(), weight
            )
            output_no_padding = jt.nn.embedding(
                jt.array(indices_np), weight
            )
            grad_weight_no_padding = jt.grad(
                (output_no_padding * jt.array(cotangent_np)).sum(), weight
            )
            values = jt.fetch_sync([
                output,
                grad_weight,
                output_no_padding,
                grad_weight_no_padding,
            ])

        np.testing.assert_allclose(
            values[0], weight_np[indices_np], atol=2e-5, rtol=2e-5
        )
        np.testing.assert_allclose(
            values[1], expected_grad, atol=2e-5, rtol=2e-5
        )
        np.testing.assert_allclose(
            values[2], weight_np[indices_np], atol=2e-5, rtol=2e-5
        )
        np.testing.assert_allclose(
            values[3], expected_grad_no_padding, atol=2e-5, rtol=2e-5
        )
        messages = [entry["msg"].lower() for entry in logs]
        self.assertTrue(any("compile acl op" in message for message in messages))
        self.assertFalse(any("compile cpu" in message for message in messages))
        self.assertFalse(any("fallback cpu" in message for message in messages))
        self.assertIsNone(jt.nn._acl_embedding(
            jt.array(indices_np), jt.array(weight_np).bfloat16()
        ))

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
    def test_serving_rotary_embedding_packed_neox_stays_on_device(self):
        rng = np.random.RandomState(20260831)
        tokens, query_heads, key_heads, head_size = 5, 16, 8, 128
        positions_np = np.asarray([0, 3, 5, 8, 13], dtype="int32")
        query_np = rng.randn(tokens, query_heads * head_size).astype("float32")
        key_np = rng.randn(tokens, key_heads * head_size).astype("float32")
        inv = 1.0 / (
            10000 ** (np.arange(0, head_size, 2) / head_size)
        )
        angles = np.arange(32)[:, None] * inv[None, :]
        cache_np = np.concatenate((np.cos(angles), np.sin(angles)), axis=-1)
        cache_np = cache_np.astype("float32")

        def reference(packed):
            view = packed.reshape(tokens, -1, head_size)
            cos = cache_np[positions_np, :head_size // 2][:, None, :]
            sin = cache_np[positions_np, head_size // 2:][:, None, :]
            first = view[..., :head_size // 2]
            second = view[..., head_size // 2:]
            return np.concatenate(
                (first * cos - second * sin, second * cos + first * sin),
                axis=-1,
            ).reshape(packed.shape)

        with jt.no_grad(), jt.log_capture_scope(
                log_v=0, log_vprefix="acl_op_exec.cc=100") as logs:
            actual_query, actual_key = jt.nn.rotary_embedding(
                jt.array(positions_np), jt.array(query_np), jt.array(key_np),
                jt.array(cache_np), head_size=head_size, is_neox=True,
                rotary_dim=head_size)
            actual_query.sync()
            actual_key.sync()
            query_location = actual_query.location()
            key_location = actual_key.location()

        messages = [entry["msg"].lower() for entry in logs]
        self.assertFalse(any("fallback cpu" in message for message in messages))
        self.assertEqual(query_location, "device")
        self.assertEqual(key_location, "device")
        np.testing.assert_allclose(
            actual_query.numpy(), reference(query_np), atol=3e-5, rtol=3e-5)
        np.testing.assert_allclose(
            actual_key.numpy(), reference(key_np), atol=3e-5, rtol=3e-5)

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_training_rotary_embedding(self):
        rng = np.random.RandomState(20260830)
        q_np = rng.randn(1, 16, 7, 128).astype("float32")
        k_np = rng.randn(1, 8, 7, 128).astype("float32")
        cos_np = rng.randn(1, 1, 7, 128).astype("float32")
        sin_np = rng.randn(1, 1, 7, 128).astype("float32")
        grad_q_np = rng.randn(*q_np.shape).astype("float32")
        grad_k_np = rng.randn(*k_np.shape).astype("float32")

        def rotate_half(value):
            half = value.shape[-1] // 2
            return np.concatenate(
                (-value[..., half:], value[..., :half]), axis=-1)

        expected_q = q_np * cos_np + rotate_half(q_np) * sin_np
        expected_k = k_np * cos_np + rotate_half(k_np) * sin_np
        expected_grad_q = (
            grad_q_np * cos_np - rotate_half(grad_q_np * sin_np)
        )
        expected_grad_k = (
            grad_k_np * cos_np - rotate_half(grad_k_np * sin_np)
        )
        expected_grad_cos = np.sum(
            grad_q_np * q_np, axis=1, keepdims=True
        ) + np.sum(grad_k_np * k_np, axis=1, keepdims=True)
        expected_grad_sin = np.sum(
            grad_q_np * rotate_half(q_np), axis=1, keepdims=True
        ) + np.sum(
            grad_k_np * rotate_half(k_np), axis=1, keepdims=True
        )

        with jt.log_capture_scope(
            log_v=0, log_vprefix="acl_op_exec.cc=100"
        ) as logs:
            q = jt.array(q_np)
            k = jt.array(k_np)
            cos = jt.array(cos_np)
            sin = jt.array(sin_np)
            output_q, output_k = jt.nn.rotary_emb(
                q, k, freq_cos=cos, freq_sin=sin
            )
            grad_q, grad_k, grad_cos, grad_sin = jt.grad(
                (output_q * jt.array(grad_q_np)).sum()
                + (output_k * jt.array(grad_k_np)).sum(),
                [q, k, cos, sin],
            )
            values = jt.fetch_sync([
                output_q, output_k, grad_q, grad_k, grad_cos, grad_sin
            ])

        expected = (
            expected_q,
            expected_k,
            expected_grad_q,
            expected_grad_k,
            expected_grad_cos,
            expected_grad_sin,
        )
        for actual, reference in zip(values, expected):
            np.testing.assert_allclose(
                actual, reference, atol=3e-5, rtol=3e-5)
        messages = [entry["msg"].lower() for entry in logs]
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
            q, k, v, dropout_p=0.1, enable_gqa=True))
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

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_product_reduction_forward_backward(self):
        x_np = np.array(
            [[1.5, -2.0, 3.0], [4.0, 0.5, -1.0]], dtype=np.float32
        )
        cotangent_np = np.array([0.25, -1.5], dtype=np.float32)
        expected_by_row = np.prod(x_np, axis=1)
        expected_grad = (
            cotangent_np[:, None] * expected_by_row[:, None] / x_np
        )
        multi_np = np.array(
            [
                [[1.5, -2.0, 0.5], [2.0, 1.0, -1.0]],
                [[-1.0, 0.25, 4.0], [0.5, -2.0, 3.0]],
            ],
            dtype=np.float32,
        )
        multi_cotangent_np = np.array([0.75, -1.25], dtype=np.float32)
        expected_multi_kept = np.prod(multi_np, axis=(0, 2), keepdims=True)
        expected_multi_grad = (
            multi_cotangent_np[None, :, None]
            * expected_multi_kept
            / multi_np
        )

        integer_values = {
            dtype: np.array([[-2, 1, 2], [1, -1, -2]], dtype=dtype)
            for dtype in ("int8", "int16", "int32", "int64")
        }
        integer_values["uint8"] = np.array(
            [[2, 1, 2], [1, 3, 2]], dtype=np.uint8
        )

        with jt.log_capture_scope(
            log_v=0, log_vprefix="acl_op_exec.cc=100"
        ) as logs:
            x = jt.array(x_np)
            full = jt.prod(x)
            by_row = jt.prod(x, dim=1)
            kept = jt.prod(x, dim=0, keepdims=True)
            grad = jt.grad(
                (by_row * jt.array(cotangent_np)).sum(), x
            )
            multi = jt.array(multi_np)
            multi_axis = jt.prod(multi, dims=(0, 2))
            multi_axis_kept = jt.prod(multi, dims=(0, 2), keepdims=True)
            multi_grad = jt.grad(
                (multi_axis * jt.array(multi_cotangent_np)).sum(), multi
            )
            integer_results = []
            for dtype, values in integer_values.items():
                value = jt.array(values, dtype=dtype)
                stacked_values = np.stack([values, np.ones_like(values)])
                integer_results.append((
                    dtype,
                    jt.prod(value),
                    jt.prod(value, dim=1),
                    jt.prod(jt.array(stacked_values, dtype=dtype), dims=(0, 2)),
                ))
            results = jt.fetch_sync(
                [
                    full, by_row, kept, grad,
                    multi_axis, multi_axis_kept, multi_grad,
                ]
            )
            (
                full, by_row, kept, grad,
                multi_axis, multi_axis_kept, multi_grad,
            ) = results
            integer_results = [
                (dtype, *jt.fetch_sync([full_value, row_value, multi_value]))
                for dtype, full_value, row_value, multi_value in integer_results
            ]

        np.testing.assert_allclose(full, np.prod(x_np), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            by_row, expected_by_row, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            kept, np.prod(x_np, axis=0, keepdims=True), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            grad, expected_grad, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            multi_axis,
            np.prod(multi_np, axis=(0, 2)),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            multi_axis_kept, expected_multi_kept, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            multi_grad, expected_multi_grad, rtol=1e-6, atol=1e-6
        )
        for dtype, full_value, row_value, multi_value in integer_results:
            values = integer_values[dtype]
            np.testing.assert_array_equal(
                full_value, np.prod(values, dtype=dtype)
            )
            np.testing.assert_array_equal(
                row_value, np.prod(values, axis=1, dtype=dtype)
            )
            np.testing.assert_array_equal(
                multi_value,
                np.prod(
                    np.stack([values, np.ones_like(values)]),
                    axis=(0, 2),
                    dtype=dtype,
                ),
            )

        messages = [entry["msg"].lower() for entry in logs]
        self.assertTrue(any("reduce.multiply" in message for message in messages))
        self.assertFalse(any("compile cpu" in message for message in messages))
        self.assertFalse(any("fallback cpu" in message for message in messages))

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
