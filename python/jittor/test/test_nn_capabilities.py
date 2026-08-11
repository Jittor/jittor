"""Focused contracts for capabilities extracted from downstream runtimes."""

import ast
import inspect
import math
import unittest

import numpy as np

import jittor as jt
import jittor.nn as nn
from jittor.nn import attention, dual_grid, rms_norm_cuda, rope_cuda, sparse


class TestAttentionCapabilities(unittest.TestCase):
    def test_layout_lengths_and_cumulative_cache(self):
        lengths = attention.sequence_lengths([slice(0, 2), (2, 5), slice(5, 5)])
        self.assertEqual(lengths, (2, 3, 0))
        with jt.flag_scope(use_cuda=0):
            first = attention.cumulative_sequence_lengths(lengths, device="cpu")
            second = attention.cumulative_sequence_lengths(lengths, device="cpu")
            self.assertIs(first, second)
            np.testing.assert_array_equal(first.numpy(), [0, 2, 5, 5])

    def test_cumulative_cache_separates_tensor_factories_by_identity(self):
        class Factory:
            __hash__ = None

            def __init__(self):
                self.calls = 0

            def __eq__(self, other):
                return isinstance(other, Factory)

            def __call__(self, values, dtype, device):
                del device
                self.calls += 1
                return jt.array(values, dtype=dtype)

        first_factory = Factory()
        second_factory = Factory()
        with jt.flag_scope(use_cuda=0):
            first = attention.cumulative_sequence_lengths(
                (4, 1), device="cpu", tensor_factory=first_factory
            )
            first_again = attention.cumulative_sequence_lengths(
                (4, 1), device="cpu", tensor_factory=first_factory
            )
            second = attention.cumulative_sequence_lengths(
                (4, 1), device="cpu", tensor_factory=second_factory
            )
        self.assertIs(first, first_again)
        self.assertIsNot(first, second)
        self.assertEqual((first_factory.calls, second_factory.calls), (1, 1))

    def test_varlen_qkv_and_dense_kv_packing(self):
        with jt.flag_scope(use_cuda=0):
            qkv = jt.arange(5 * 3 * 2 * 4).reshape(5, 3, 2, 4).float32()
            calls = []

            def qkvpacked(value, cu, maximum):
                calls.append((tuple(value.shape), tuple(cu.numpy()), maximum))
                return value[:, 0]

            out = attention.varlen_scaled_dot_product_attention(
                qkv, q_lengths=(2, 3), qkvpacked_func=qkvpacked
            )
            self.assertEqual(tuple(out.shape), (5, 2, 4))
            self.assertEqual(calls, [((5, 3, 2, 4), (0, 2, 5), 3)])

            q = jt.arange(2 * 3 * 2 * 4).reshape(2, 3, 2, 4).float32()
            kv = jt.arange(2 * 3 * 2 * 2 * 4).reshape(2, 3, 2, 2, 4).float32()

            def kvpacked(q_value, kv_value, cu_q, cu_kv, max_q, max_kv):
                self.assertEqual(tuple(q_value.shape), (6, 2, 4))
                self.assertEqual(tuple(kv_value.shape), (6, 2, 2, 4))
                self.assertIs(cu_q, cu_kv)
                self.assertEqual((max_q, max_kv), (3, 3))
                return q_value + kv_value[:, 0]

            dense_out = attention.varlen_scaled_dot_product_attention(q, kv, kvpacked_func=kvpacked)
            self.assertEqual(tuple(dense_out.shape), (2, 3, 2, 4))

    def test_varlen_rejects_different_query_and_kv_sequence_counts(self):
        backend_calls = []

        def backend(*args):
            backend_calls.append(args)
            return args[0]

        with jt.flag_scope(use_cuda=0):
            q = jt.arange(3 * 2 * 4).reshape(3, 2, 4).float32()
            kv = jt.arange(3 * 2 * 2 * 4).reshape(3, 2, 2, 4).float32()
            with self.assertRaisesRegex(ValueError, "same number of sequences"):
                attention.varlen_scaled_dot_product_attention(
                    q,
                    kv,
                    q_lengths=(1, 2),
                    kv_lengths=(3,),
                    kvpacked_func=backend,
                )

            k = jt.arange(3 * 2 * 4).reshape(3, 2, 4).float32()
            v = jt.arange(3 * 2 * 4).reshape(3, 2, 4).float32()
            with self.assertRaisesRegex(ValueError, "same number of sequences"):
                attention.varlen_scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    q_lengths=(1, 2),
                    kv_lengths=(3,),
                    varlen_func=backend,
                )
        self.assertEqual(backend_calls, [])


class TestDualGridCapabilities(unittest.TestCase):
    def test_float_index_tensors_are_rejected(self):
        coords = jt.zeros((4, 3), dtype="int32")
        dual_vertices = jt.zeros((4, 3), dtype="float32")
        quad_indices = jt.array([[0, 1, 2, 3]], dtype="int32")
        valid_rows = jt.array([0], dtype="int32")
        split_weight = jt.ones((4,), dtype="float32")
        voxel_size = jt.ones((3,), dtype="float32")
        aabb_min = jt.zeros((3,), dtype="float32")

        with self.assertRaisesRegex(TypeError, "quad_indices"):
            dual_grid.finalize_dual_grid_mesh_cuda(
                coords,
                dual_vertices,
                quad_indices.float32(),
                valid_rows,
                split_weight,
                voxel_size,
                aabb_min,
            )
        with self.assertRaisesRegex(TypeError, "valid_rows"):
            dual_grid.finalize_dual_grid_mesh_cuda(
                coords,
                dual_vertices,
                quad_indices,
                valid_rows.float32(),
                split_weight,
                voxel_size,
                aabb_min,
            )


class TestSparseCapabilities(unittest.TestCase):
    @staticmethod
    def _reference(coords, feats, weight, bias, dilation):
        out_channels, kd, kh, kw, in_channels = weight.shape
        lookup = {tuple(row): index for index, row in enumerate(coords)}
        out = np.zeros((len(coords), out_channels), dtype=np.float32)
        grad_feats = np.zeros_like(feats)
        grad_weight = np.zeros_like(weight)
        centers = (kd // 2, kh // 2, kw // 2)
        for target, coord in enumerate(coords):
            for iz in range(kd):
                for iy in range(kh):
                    for ix in range(kw):
                        query = (
                            coord[0],
                            coord[1] + (iz - centers[0]) * dilation[0],
                            coord[2] + (iy - centers[1]) * dilation[1],
                            coord[3] + (ix - centers[2]) * dilation[2],
                        )
                        source = lookup.get(query)
                        if source is not None:
                            kernel = weight[:, iz, iy, ix, :]
                            out[target] += np.matmul(kernel, feats[source])
                            grad_feats[source] += kernel.sum(0)
                            grad_weight[:, iz, iy, ix, :] += feats[source]
        out += bias
        return out, grad_feats, grad_weight

    def test_submanifold_conv_matches_reference_and_is_differentiable(self):
        coords_np = np.array(
            [
                [0, -2, 4, 8],
                [0, -2, 4, 10],
                [0, -2, 6, 8],
                [0, 0, 0, 0],
                [1, -2, 4, 8],
            ],
            dtype=np.int32,
        )
        feats_np = np.arange(15, dtype=np.float32).reshape(5, 3) / 7
        rng = np.random.RandomState(211)
        weight_np = rng.randn(2, 3, 1, 3, 3).astype("float32")
        bias_np = rng.randn(2).astype("float32")
        dilation = (1, 2, 2)
        expected, grad_feats, grad_weight = self._reference(
            coords_np, feats_np, weight_np, bias_np, dilation
        )

        with jt.flag_scope(use_cuda=0):
            coords = jt.array(coords_np)
            feats = jt.array(feats_np)
            weight = jt.array(weight_np)
            bias = jt.array(bias_np)
            feats.start_grad()
            weight.start_grad()
            neighbors = sparse.build_submanifold_conv3d_neighbors(
                coords, (3, 1, 3), dilation=dilation
            )
            out = sparse.submanifold_conv3d(
                feats, coords, weight, bias, dilation=dilation, neighbors=neighbors
            )
            actual, actual_grad_feats, actual_grad_weight = jt.fetch_sync(
                [
                    out,
                    *jt.grad(out.sum(), [feats, weight]),
                ]
            )
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(actual_grad_feats, grad_feats, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(actual_grad_weight, grad_weight, atol=1e-6, rtol=1e-6)

    def test_sparse_python_path_has_no_per_tap_loop_or_item_sync(self):
        tree = ast.parse(inspect.getsource(sparse.submanifold_conv3d))
        self.assertFalse(any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(tree)))
        calls = [
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        self.assertNotIn("item", calls)
        self.assertEqual(calls.count("matmul"), 1)


@unittest.skipIf(not jt.has_cuda, "No CUDA found")
class TestCudaCapabilities(unittest.TestCase):
    def test_submanifold_conv_cuda_hash_and_backward(self):
        coords_np = np.array(
            [
                [0, -1, 2, 4],
                [0, -1, 2, 5],
                [0, -1, 2, 6],
                [1, -1, 2, 5],
            ],
            dtype=np.int64,
        )
        feats_np = np.arange(8, dtype=np.float32).reshape(4, 2) / 5
        weight_np = np.arange(12, dtype=np.float32).reshape(2, 1, 1, 3, 2) / 11
        bias_np = np.array([0.25, -0.5], dtype=np.float32)
        expected, grad_feats, grad_weight = TestSparseCapabilities._reference(
            coords_np, feats_np, weight_np, bias_np, (1, 1, 1)
        )
        with jt.flag_scope(use_cuda=1):
            coords = jt.array(coords_np)
            feats = jt.array(feats_np)
            weight = jt.array(weight_np)
            feats.start_grad()
            weight.start_grad()
            neighbors = sparse.build_submanifold_conv3d_neighbors(coords, (1, 1, 3))
            out = sparse.submanifold_conv3d(
                feats, coords, weight, jt.array(bias_np), neighbors=neighbors
            )
            actual, actual_grad_feats, actual_grad_weight = jt.fetch_sync(
                [
                    out,
                    *jt.grad(out.sum(), [feats, weight]),
                ]
            )
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(actual_grad_feats, grad_feats, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(actual_grad_weight, grad_weight, atol=1e-6, rtol=1e-6)

    def test_parameterized_multihead_rms_norm(self):
        rng = np.random.RandomState(223)
        x_np = rng.randn(2, 5, 3, 96).astype("float32")
        gamma_np = (1 + 0.1 * rng.randn(3, 96)).astype("float32")
        with jt.flag_scope(use_cuda=1), jt.no_grad():
            x = jt.array(x_np).bfloat16()
            gamma = jt.array(gamma_np)
            actual = rms_norm_cuda.multihead_rms_norm_cuda(x, gamma)
            self.assertIsNotNone(actual)
            actual_np = actual.float32().numpy()
        norm = np.sqrt((x_np * x_np).sum(-1, keepdims=True))
        expected = x_np / np.maximum(norm, 1e-12) * gamma_np * math.sqrt(96)
        np.testing.assert_allclose(actual_np, expected, atol=0.02, rtol=0.01)

    def test_partial_rope_uses_explicit_prefix_and_rotary_dim(self):
        rng = np.random.RandomState(227)
        q_np = rng.randn(2, 5, 7, 40).astype("float32")
        k_np = rng.randn(2, 5, 7, 40).astype("float32")
        cos_np = rng.randn(5, 24).astype("float32")
        sin_np = rng.randn(5, 24).astype("float32")
        with jt.flag_scope(use_cuda=1), jt.no_grad():
            result = rope_cuda.partial_rotary_embedding_cuda(
                jt.array(q_np),
                jt.array(k_np),
                jt.array(cos_np),
                jt.array(sin_np),
                prefix_tokens=2,
                rotary_dim=24,
            )
            self.assertIsNotNone(result)
            actual_q, actual_k = jt.fetch_sync(result)

        def reference(value):
            out = value.copy()
            patch = value[:, :, 2:, :24]
            rotated = np.concatenate((-patch[..., 12:], patch[..., :12]), axis=-1)
            out[:, :, 2:, :24] = patch * cos_np + rotated * sin_np
            return out

        np.testing.assert_allclose(actual_q, reference(q_np), atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(actual_k, reference(k_np), atol=2e-6, rtol=2e-6)

    def test_dual_grid_mesh_finalizer(self):
        coords_np = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
            ],
            dtype=np.int32,
        )
        dual_np = np.arange(12, dtype=np.float32).reshape(4, 3) / 10
        quads_np = np.array([[0, 1, 2, 3]], dtype=np.int32)
        weights_np = np.array([1, 2, 1, 3], dtype=np.float32)
        voxel_np = np.array([0.5, 0.25, 0.125], dtype=np.float32)
        origin_np = np.array([-1, 2, 4], dtype=np.float32)
        with jt.flag_scope(use_cuda=1), jt.no_grad():
            result = dual_grid.finalize_dual_grid_mesh_cuda(
                jt.array(coords_np),
                jt.array(dual_np),
                jt.array(quads_np),
                jt.array(np.array([0], dtype=np.int32)),
                jt.array(weights_np),
                jt.array(voxel_np),
                jt.array(origin_np),
            )
            self.assertIsNotNone(result)
            vertices, faces = jt.fetch_sync(result)
        np.testing.assert_allclose(vertices, (coords_np + dual_np) * voxel_np + origin_np)
        np.testing.assert_array_equal(faces, [[0, 1, 3], [3, 1, 2]])

    def test_layer_norm_backend_accepts_non_trellis_hidden_and_eps(self):
        rng = np.random.RandomState(229)
        x_np = rng.randn(3, 96).astype("float32")
        weight_np = rng.randn(96).astype("float32")
        bias_np = rng.randn(96).astype("float32")
        eps = 3e-4
        with jt.flag_scope(use_cuda=1), jt.no_grad():
            actual = nn._layer_norm_no_grad_cuda(
                jt.array(x_np).bfloat16(),
                (96,),
                jt.array(weight_np),
                jt.array(bias_np),
                eps,
                allow_bfloat16=True,
            )
            self.assertIsNotNone(actual)
            actual_np = actual.float32().numpy()
        mean = x_np.mean(-1, keepdims=True)
        variance = ((x_np - mean) ** 2).mean(-1, keepdims=True)
        expected = (x_np - mean) / np.sqrt(variance + eps)
        expected = expected * weight_np + bias_np
        np.testing.assert_allclose(actual_np, expected, atol=0.025, rtol=0.012)


class TestCapabilityStructure(unittest.TestCase):
    def test_facade_exports_physical_capabilities(self):
        expected = {
            "cumulative_sequence_lengths": attention,
            "varlen_scaled_dot_product_attention": attention,
            "finalize_dual_grid_mesh_cuda": dual_grid,
            "multihead_rms_norm_cuda": rms_norm_cuda,
            "partial_rotary_embedding_cuda": rope_cuda,
            "build_submanifold_conv3d_neighbors": sparse,
            "submanifold_conv3d": sparse,
        }
        for name, module in expected.items():
            value = getattr(nn, name)
            with self.subTest(name=name):
                self.assertIs(value, getattr(module, name))
                self.assertEqual(value.__module__, module.__name__)


if __name__ == "__main__":
    unittest.main()
