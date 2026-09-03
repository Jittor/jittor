"""Focused contracts for capabilities extracted from downstream runtimes."""

import ast
import inspect
import math
import pickle
import unittest

import numpy as np

import jittor as jt
import jittor.nn as nn
from jittor.nn import (
    attention,
    dual_grid,
    packed_qkv_cuda,
    rms_norm_cuda,
    rope_cuda,
    sparse,
)
from jittor.nn.backends.layer_norm_cuda import _layer_norm_no_grad_cuda


class TestAttentionCapabilities(unittest.TestCase):
    def test_scaled_attention_combines_causal_and_explicit_masks(self):
        with jt.flag_scope(use_cuda=0):
            query = jt.zeros((1, 1, 3, 2), dtype="float32")
            value = jt.array([[[[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]]]])
            keep = jt.array(
                [
                    [True, True, True],
                    [False, True, True],
                    [True, False, True],
                ]
            )
            actual = nn.scaled_dot_product_attention(
                query,
                query,
                value,
                attn_mask=keep,
                is_causal=True,
            ).numpy()
        expected = np.array(
            [[[[2.0, 4.0], [6.0, 8.0], [6.0, 8.0]]]],
            dtype="float32",
        )
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)

    def test_scaled_attention_float16_causal_rows_stay_finite(self):
        with jt.flag_scope(use_cuda=0):
            query = jt.array(np.arange(24, dtype="float32").reshape(1, 2, 3, 4) / 10).float16()
            output = nn.scaled_dot_product_attention(query, query, query, is_causal=True)
            actual = output.float32().numpy()
        self.assertTrue(np.isfinite(actual).all())

    def test_attention_masks_reject_integer_dtypes(self):
        with jt.flag_scope(use_cuda=0):
            query = jt.zeros((2, 1, 4), dtype="float32")
            module = nn.MultiheadAttention(4, 2)
            with self.assertRaisesRegex(AssertionError, "bool and floating"):
                module(query, query, query, attn_mask=jt.ones((2, 2), dtype="int32"))
            with self.assertRaisesRegex(AssertionError, "bool and floating"):
                module(
                    query,
                    query,
                    query,
                    key_padding_mask=jt.ones((1, 2), dtype="int32"),
                )
            with self.assertRaisesRegex(AssertionError, "bool and floating"):
                nn.scaled_dot_product_attention(
                    query.transpose(0, 1),
                    query.transpose(0, 1),
                    query.transpose(0, 1),
                    attn_mask=jt.ones((2, 2), dtype="int32"),
                )

    def test_scaled_attention_validates_dtypes_and_dropout_probability(self):
        with jt.flag_scope(use_cuda=0):
            query = jt.zeros((1, 1, 2, 4), dtype="float32")
            with self.assertRaisesRegex(RuntimeError, "same dtype"):
                nn.scaled_dot_product_attention(query, query.float16(), query)
            with self.assertRaisesRegex(RuntimeError, "mask dtype"):
                nn.scaled_dot_product_attention(
                    query,
                    query,
                    query,
                    attn_mask=jt.zeros((2, 2), dtype="float64"),
                )
            for probability in (-0.1, 1.1):
                with self.subTest(probability=probability):
                    with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                        nn.scaled_dot_product_attention(
                            query,
                            query,
                            query,
                            dropout_p=probability,
                        )
            module = nn.MultiheadAttention(4, 2, dropout=-0.1)
            sequence = query.reshape(2, 1, 4)
            with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                module(sequence, sequence, sequence)

    def test_no_weight_attention_matches_weighted_path_for_masks(self):
        rng = np.random.RandomState(317)
        with jt.flag_scope(use_cuda=0):
            module = nn.MultiheadAttention(8, 2)
            query = jt.array(rng.randn(4, 2, 8).astype("float32"))
            key = jt.array(rng.randn(5, 2, 8).astype("float32"))
            value = jt.array(rng.randn(5, 2, 8).astype("float32"))
            boolean_mask = jt.array(np.triu(np.ones((4, 5), dtype=bool), 1))
            float_padding = jt.array([[0.0, 0.0, 0.0, -0.5, -1.0], [0.0, -0.25, 0.0, 0.0, -2.0]])
            weighted, _ = module(
                query,
                key,
                value,
                attn_mask=boolean_mask,
                key_padding_mask=float_padding,
                need_weights=True,
            )
            unweighted, returned_weights = module(
                query,
                key,
                value,
                attn_mask=boolean_mask,
                key_padding_mask=float_padding,
                need_weights=False,
            )
            per_head_mask = boolean_mask.broadcast((4, 4, 5))
            per_head, per_head_weights = module(
                query,
                key,
                value,
                attn_mask=per_head_mask,
                need_weights=False,
            )
            causal_mask = jt.triu(jt.ones((4, 5), dtype="bool"), diagonal=1)
            causal, causal_weights = module(
                query,
                key,
                value,
                attn_mask=causal_mask,
                need_weights=False,
                is_causal=True,
            )
            causal_reference, _ = module(
                query,
                key,
                value,
                attn_mask=causal_mask,
                need_weights=True,
            )
            per_head_reference, _ = module(
                query,
                key,
                value,
                attn_mask=boolean_mask,
                need_weights=True,
            )
            actual = jt.fetch_sync(
                [
                    weighted,
                    unweighted,
                    per_head,
                    per_head_reference,
                    causal,
                    causal_reference,
                ]
            )
        self.assertIsNone(returned_weights)
        self.assertIsNone(per_head_weights)
        self.assertIsNone(causal_weights)
        np.testing.assert_allclose(actual[0], actual[1], atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(actual[2], actual[3], atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(actual[4], actual[5], atol=2e-5, rtol=2e-5)

    def test_fully_masked_rows_preserve_torch_branch_behavior(self):
        with jt.flag_scope(use_cuda=0):
            module = nn.MultiheadAttention(4, 2)
            sequence = jt.ones((2, 1, 4), dtype="float32")
            mask = jt.ones((2, 2), dtype="bool")
            weighted, weights = module(sequence, sequence, sequence, attn_mask=mask)
            unweighted, no_weights = module(
                sequence,
                sequence,
                sequence,
                attn_mask=mask,
                need_weights=False,
            )
            weighted_array, weights_array, unweighted_array = jt.fetch_sync(
                [weighted, weights, unweighted]
            )
        self.assertTrue(np.isnan(weighted_array).all())
        self.assertTrue(np.isnan(weights_array).all())
        np.testing.assert_array_equal(unweighted_array, np.zeros_like(unweighted_array))
        self.assertIsNone(no_weights)

    def test_multihead_attention_dtype_and_static_source_validation(self):
        with jt.flag_scope(use_cuda=0):
            module = nn.MultiheadAttention(4, 2, add_bias_kv=True, dtype=jt.float16)
            parameter_dtypes = {str(parameter.dtype) for parameter in module.parameters()}
            self.assertEqual(parameter_dtypes, {"float16"})
            legacy_positional = nn.MultiheadAttention(
                4, 2, 0.0, True, False, False, None, None, False, jt.float16
            )
            self.assertEqual(
                {str(parameter.dtype) for parameter in legacy_positional.parameters()},
                {"float16"},
            )

            query = jt.zeros((2, 1, 4), dtype="float32")
            weight = jt.concat([jt.init.eye(4), jt.init.eye(4), jt.init.eye(4)], dim=0)
            with self.assertRaisesRegex(AssertionError, "source lengths must match"):
                nn.multi_head_attention_forward(
                    query,
                    query,
                    query,
                    4,
                    2,
                    weight,
                    None,
                    None,
                    None,
                    False,
                    0.0,
                    jt.init.eye(4),
                    None,
                    static_k=jt.zeros((2, 3, 2)),
                    static_v=jt.zeros((2, 4, 2)),
                )

            with self.assertRaisesRegex(AssertionError, "batch sizes must match"):
                module(
                    jt.zeros((3, 2, 4)),
                    jt.zeros((4, 1, 4)),
                    jt.zeros((4, 1, 4)),
                )

            with self.assertRaisesRegex(AssertionError, "3-D tensor"):
                nn.multi_head_attention_forward(
                    query,
                    query,
                    query,
                    4,
                    2,
                    weight,
                    None,
                    None,
                    None,
                    False,
                    0.0,
                    jt.init.eye(4),
                    None,
                    static_k=jt.zeros((2, 2)),
                )

            with self.assertRaisesRegex(AssertionError, "key projection weight"):
                nn.multi_head_attention_forward(
                    query,
                    query,
                    query,
                    4,
                    2,
                    None,
                    None,
                    None,
                    None,
                    False,
                    0.0,
                    jt.init.eye(4),
                    None,
                    use_separate_proj_weight=True,
                    q_proj_weight=jt.init.eye(4),
                    k_proj_weight=jt.zeros((5, 4)),
                    v_proj_weight=jt.init.eye(4),
                )

    def test_multihead_attention_additive_padding_mask_and_pickle(self):
        with jt.flag_scope(use_cuda=0):
            module = nn.MultiheadAttention(2, 1, bias=False)
            identity = np.eye(2, dtype="float32")
            module.in_proj_weight.assign(
                jt.array(np.concatenate([identity, identity, identity], axis=0))
            )
            module.out_proj.weight.assign(jt.array(identity))
            query = jt.zeros((1, 1, 2), dtype="float32")
            value = jt.array([[[2.0, 4.0]], [[10.0, 20.0]]])
            padding = jt.array([[0.0, -1.0]])
            output, weights = module(
                query,
                query.broadcast((2, 1, 2)),
                value,
                key_padding_mask=padding,
            )
            restored = pickle.loads(pickle.dumps(module))
            restored_output, restored_weights = restored(
                query,
                query.broadcast((2, 1, 2)),
                value,
                key_padding_mask=padding,
            )
            actual, actual_weights, roundtrip_output, roundtrip_weights = jt.fetch_sync(
                [output, weights, restored_output, restored_weights]
            )

        expected_weights = np.array(
            [[[1.0 / (1.0 + math.exp(-1.0)), 1.0 / (1.0 + math.exp(1.0))]]],
            dtype="float32",
        )
        expected = np.matmul(expected_weights, np.swapaxes(value.numpy(), 0, 1))
        np.testing.assert_allclose(actual_weights, expected_weights, atol=1e-6)
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        np.testing.assert_allclose(roundtrip_output, actual, atol=1e-6)
        np.testing.assert_allclose(roundtrip_weights, actual_weights, atol=1e-6)

    def test_multihead_attention_causal_values_and_dimension_validation(self):
        with jt.flag_scope(use_cuda=0):
            module = nn.MultiheadAttention(4, 2, bias=False)
            identity = np.eye(4, dtype="float32")
            module.in_proj_weight.assign(
                jt.array(np.concatenate([identity, identity, identity], axis=0))
            )
            module.out_proj.weight.assign(jt.array(identity))

            query = jt.zeros((3, 1, 4), dtype="float32")
            value_np = np.array(
                [
                    [[1.0, 2.0, 3.0, 4.0]],
                    [[5.0, 6.0, 7.0, 8.0]],
                    [[9.0, 10.0, 11.0, 12.0]],
                ],
                dtype="float32",
            )
            output, weights = module(
                query,
                query,
                jt.array(value_np),
                attn_mask=jt.triu(jt.ones((3, 3), dtype="bool"), diagonal=1),
                is_causal=True,
            )
            expected = np.stack([value_np[: index + 1].mean(axis=0) for index in range(3)])
            actual, actual_weights = jt.fetch_sync([output, weights])

        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
        np.testing.assert_array_equal(np.triu(actual_weights, k=1), np.zeros_like(actual_weights))
        with self.assertRaisesRegex(AssertionError, "embedding dimension"):
            module(
                jt.zeros((2, 1, 3)),
                jt.zeros((2, 1, 3)),
                jt.zeros((2, 1, 3)),
            )

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


class TestEmbeddingCapabilities(unittest.TestCase):
    def test_module_honors_functional_max_norm_in_place_without_norm_gradient(self):
        weight = np.array([[3.0, 4.0], [0.0, 2.0], [5.0, 12.0]], dtype="float32")
        with jt.flag_scope(use_cuda=0):
            module = nn.Embedding(3, 2, max_norm=1.0, _weight=jt.array(weight))
            output = module(jt.array([0, 2], dtype="int32"))
            gradient = jt.grad(output.sum(), module.weight)
            actual, actual_weight, actual_gradient = jt.fetch_sync(
                [output, module.weight, gradient]
            )
        expected = np.array([[0.6, 0.8], [5.0 / 13.0, 12.0 / 13.0]])
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(
            actual_weight,
            np.array([[0.6, 0.8], [0.0, 2.0], [5.0 / 13.0, 12.0 / 13.0]]),
            atol=1e-6,
            rtol=1e-6,
        )
        np.testing.assert_array_equal(
            actual_gradient,
            np.array([[1.0, 1.0], [0.0, 0.0], [1.0, 1.0]]),
        )

    def test_max_norm_does_not_modify_rows_at_the_boundary(self):
        weight = np.array([[0.6, 0.8], [0.0, 2.0]], dtype="float32")
        with jt.flag_scope(use_cuda=0):
            module = nn.Embedding(2, 2, max_norm=1.0, _weight=jt.array(weight))
            output = module(jt.array([0], dtype="int32"))
            actual, actual_weight = jt.fetch_sync([output, module.weight])
        np.testing.assert_array_equal(actual, weight[:1])
        np.testing.assert_array_equal(actual_weight[0], weight[0])

    def test_module_normalizes_negative_padding_index(self):
        with jt.flag_scope(use_cuda=0):
            module = nn.Embedding(3, 2, padding_idx=-1)
            self.assertEqual(module.padding_idx, 2)
            np.testing.assert_array_equal(module.weight[2].numpy(), [0.0, 0.0])
        for padding_idx in (3, -4):
            with self.subTest(padding_idx=padding_idx):
                with self.assertRaisesRegex(AssertionError, "within num_embeddings"):
                    nn.Embedding(3, 2, padding_idx=padding_idx)


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
        for num_heads, head_dim in ((3, 96), (12, 128), (4, 256)):
            with self.subTest(num_heads=num_heads, head_dim=head_dim):
                x_np = rng.randn(2, 5, num_heads, head_dim).astype("float32")
                gamma_np = (
                    1 + 0.1 * rng.randn(num_heads, head_dim)
                ).astype("float32")
                with jt.flag_scope(use_cuda=1), jt.no_grad():
                    x = jt.array(x_np).bfloat16()
                    gamma = jt.array(gamma_np)
                    actual = rms_norm_cuda.multihead_rms_norm_cuda(x, gamma)
                    self.assertIsNotNone(actual)
                    quantized_x, actual_np = jt.fetch_sync(
                        [x.float32(), actual.float32()]
                    )
                norm = np.sqrt(
                    (quantized_x * quantized_x).sum(-1, keepdims=True)
                )
                expected = (
                    quantized_x
                    / np.maximum(norm, 1e-12)
                    * gamma_np
                    * math.sqrt(head_dim)
                )
                np.testing.assert_allclose(
                    actual_np, expected, atol=0.02, rtol=0.01
                )

    def test_inference_rms_norm_cuda(self):
        rng = np.random.RandomState(224)
        x_np = rng.randn(3, 128).astype("float32")
        residual_np = rng.randn(3, 128).astype("float32")
        gamma_np = (1 + 0.1 * rng.randn(128)).astype("float32")
        epsilon = 1e-6
        for dtype, atol, rtol in (
            ("float16", 0.004, 0.004),
            ("bfloat16", 0.03, 0.02),
        ):
            with self.subTest(dtype=dtype):
                with jt.flag_scope(use_cuda=1), jt.no_grad():
                    x = jt.array(x_np).cast(dtype)
                    residual = jt.array(residual_np).cast(dtype)
                    gamma = jt.array(gamma_np).cast(dtype)
                    actual = rms_norm_cuda._rms_norm_cuda(x, gamma, epsilon)
                    fused = rms_norm_cuda._fused_add_rms_norm_cuda(
                        x, residual, gamma, epsilon)
                    self.assertIsNotNone(actual)
                    self.assertIsNotNone(fused)
                    actual_np = actual.float32().numpy()
                    fused_np, fused_residual_np = jt.fetch_sync(
                        [fused[0].float32(), fused[1].float32()])

                quantized_x = x.float32().numpy()
                quantized_residual = residual.float32().numpy()
                quantized_gamma = gamma.float32().numpy()
                variance = np.mean(quantized_x * quantized_x, axis=-1, keepdims=True)
                expected = (
                    quantized_x / np.sqrt(variance + epsilon) * quantized_gamma)
                summed = quantized_x + quantized_residual
                fused_variance = np.mean(summed * summed, axis=-1, keepdims=True)
                expected_fused = (
                    summed / np.sqrt(fused_variance + epsilon) * quantized_gamma)
                np.testing.assert_allclose(actual_np, expected, atol=atol, rtol=rtol)
                np.testing.assert_allclose(
                    fused_np, expected_fused, atol=atol, rtol=rtol)
                np.testing.assert_allclose(
                    fused_residual_np, summed, atol=atol, rtol=rtol)

    def test_modulated_layer_norm_preserves_bfloat_rounding(self):
        from jittor.nn.backends.modulated_layer_norm_cuda import (
            _modulated_layer_norm_no_grad_cuda,
        )

        rng = np.random.RandomState(231)
        x_np = rng.randn(7, 96).astype("float32")
        scale_np = (0.1 * rng.randn(1, 96)).astype("float32")
        shift_np = (0.1 * rng.randn(1, 96)).astype("float32")
        eps = 1e-6
        with jt.flag_scope(use_cuda=1), jt.no_grad():
            x = jt.array(x_np).bfloat16()
            scale = jt.array(scale_np).bfloat16()
            shift = jt.array(shift_np).bfloat16()
            actual = _modulated_layer_norm_no_grad_cuda(
                x, scale, shift, eps
            )
            self.assertIsNotNone(actual)
            reference = _layer_norm_no_grad_cuda(
                x, (96,), 1.0, 0.0, eps, allow_bfloat16=True
            )
            reference = reference * (1 + scale) + shift
            actual_np, reference_np = jt.fetch_sync(
                [actual.float32(), reference.float32()]
            )
        np.testing.assert_allclose(
            actual_np, reference_np, atol=0.016, rtol=0.008
        )

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

    def test_packed_qkv_rms_rope_preserves_bfloat_rounding(self):
        rng = np.random.RandomState(230)
        token_count, num_heads, head_dim = 7, 3, 128
        qkv_np = rng.randn(token_count, 3, num_heads, head_dim).astype("float32")
        q_gamma_np = (
            1 + 0.1 * rng.randn(num_heads, head_dim)
        ).astype("float32")
        k_gamma_np = (
            1 + 0.1 * rng.randn(num_heads, head_dim)
        ).astype("float32")
        angles = rng.randn(token_count, head_dim // 2).astype("float32")
        phases_np = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

        with jt.flag_scope(use_cuda=1), jt.no_grad():
            qkv = jt.array(qkv_np).bfloat16()
            result = packed_qkv_cuda.packed_qkv_rms_rope_cuda(
                qkv,
                jt.array(q_gamma_np),
                jt.array(k_gamma_np),
                jt.array(phases_np),
            )
            self.assertIsNotNone(result)
            quantized_qkv, actual = jt.fetch_sync(
                [qkv.float32(), result.float32()]
            )

            def quantize(value):
                return jt.array(value).bfloat16().float32().numpy()

            scale = math.sqrt(head_dim)
            q = quantized_qkv[:, 0]
            k = quantized_qkv[:, 1]
            q_norm = np.sqrt((q * q).sum(-1, keepdims=True))
            k_norm = np.sqrt((k * k).sum(-1, keepdims=True))
            q = quantize(
                q / np.maximum(q_norm, 1e-12) * q_gamma_np * scale
            )
            k = quantize(
                k / np.maximum(k_norm, 1e-12) * k_gamma_np * scale
            )

        def rotate(value):
            pairs = value.reshape(token_count, num_heads, head_dim // 2, 2)
            real = pairs[..., 0]
            imag = pairs[..., 1]
            phase_real = phases_np[:, None, :, 0]
            phase_imag = phases_np[:, None, :, 1]
            return np.stack(
                (
                    real * phase_real - imag * phase_imag,
                    real * phase_imag + imag * phase_real,
                ),
                axis=-1,
            ).reshape(value.shape)

        np.testing.assert_allclose(actual[:, 0], rotate(q), atol=0.03, rtol=0.02)
        np.testing.assert_allclose(actual[:, 1], rotate(k), atol=0.03, rtol=0.02)
        np.testing.assert_array_equal(actual[:, 2], quantized_qkv[:, 2])

    def test_inference_gqa_rotary_embedding_cuda(self):
        rng = np.random.RandomState(228)
        positions_np = np.array([5, 2, 7], dtype="int32")
        q_np = rng.randn(3, 4 * 40).astype("float32")
        k_np = rng.randn(3, 2 * 40).astype("float32")
        cache_np = rng.randn(10, 24).astype("float32")

        def reference(value, cache):
            shaped = value.reshape(3, -1, 40).copy()
            patch = shaped[..., :24].copy()
            selected = cache[positions_np]
            cos, sin = np.split(selected, 2, axis=-1)
            first, second = np.split(patch, 2, axis=-1)
            shaped[..., :24] = np.concatenate(
                (
                    first * cos[:, None, :] - second * sin[:, None, :],
                    second * cos[:, None, :] + first * sin[:, None, :],
                ),
                axis=-1,
            )
            return shaped.reshape(value.shape)

        for dtype, atol, rtol in (
            ("float16", 0.004, 0.004),
            ("bfloat16", 0.04, 0.02),
        ):
            with self.subTest(dtype=dtype):
                with jt.flag_scope(use_cuda=1), jt.no_grad():
                    q = jt.array(q_np).cast(dtype)
                    k = jt.array(k_np).cast(dtype)
                    cache = jt.array(cache_np).cast(dtype)
                    result = rope_cuda._rotary_embedding_cuda(
                        jt.array(positions_np), q, k, cache,
                        head_size=40, rotary_dim=24, is_neox_style=True)
                    self.assertIsNotNone(result)
                    actual_q, actual_k = jt.fetch_sync(
                        [result[0].float32(), result[1].float32()])
                    quantized_q, quantized_k, quantized_cache = jt.fetch_sync(
                        [q.float32(), k.float32(), cache.float32()])
                np.testing.assert_allclose(
                    actual_q, reference(quantized_q, quantized_cache),
                    atol=atol, rtol=rtol)
                np.testing.assert_allclose(
                    actual_k, reference(quantized_k, quantized_cache),
                    atol=atol, rtol=rtol)

    def test_inference_silu_and_mul_cuda(self):
        from jittor.nn.swiglu_cuda import _silu_and_mul_cuda

        rng = np.random.RandomState(229)
        x_np = rng.randn(3, 256).astype("float32")
        for dtype, atol, rtol in (
            ("float16", 0.004, 0.004),
            ("bfloat16", 0.03, 0.02),
        ):
            with self.subTest(dtype=dtype):
                with jt.flag_scope(use_cuda=1), jt.no_grad():
                    x = jt.array(x_np).cast(dtype)
                    result = _silu_and_mul_cuda(x)
                    self.assertIsNotNone(result)
                    actual, quantized = jt.fetch_sync(
                        [result.float32(), x.float32()])
                gate, value = np.split(quantized, 2, axis=-1)
                expected = gate / (1.0 + np.exp(-gate)) * value
                np.testing.assert_allclose(
                    actual, expected, atol=atol, rtol=rtol)

    def test_inference_paged_kv_cache_cuda(self):
        from jittor.nn.kv_cache_cuda import _reshape_and_cache_cuda

        rng = np.random.RandomState(230)
        key_np = rng.randn(3, 2, 3).astype("float32")
        value_np = rng.randn(3, 2, 3).astype("float32")
        slots_np = np.array([0, 5, -1], dtype="int32")
        for dtype, atol in (("float16", 0.0), ("bfloat16", 0.0)):
            with self.subTest(dtype=dtype):
                with jt.flag_scope(use_cuda=1), jt.no_grad():
                    key = jt.array(key_np).cast(dtype)
                    value = jt.array(value_np).cast(dtype)
                    cache = jt.zeros((3, 2, 4, 2, 3), dtype=dtype)
                    result = _reshape_and_cache_cuda(
                        key, value, cache, jt.array(slots_np))
                    self.assertIs(result, cache)
                    actual, quantized_key, quantized_value = jt.fetch_sync(
                        [cache.float32(), key.float32(), value.float32()])
                expected = np.zeros((3, 2, 4, 2, 3), dtype="float32")
                expected[0, 0, 0] = quantized_key[0]
                expected[0, 1, 0] = quantized_value[0]
                expected[1, 0, 1] = quantized_key[1]
                expected[1, 1, 1] = quantized_value[1]
                np.testing.assert_allclose(actual, expected, atol=atol, rtol=0)

    def test_inference_paged_attention_decode_cuda(self):
        from jittor.nn.kv_cache_cuda import _paged_attention_decode_cuda

        rng = np.random.RandomState(231)
        query_np = rng.randn(2, 4, 16).astype("float32")
        cache_np = rng.randn(4, 2, 4, 2, 16).astype("float32")
        seq_lens_np = np.array([5, 3], dtype="int32")
        block_table_np = np.array([[2, 0], [1, 3]], dtype="int32")
        scale = 16 ** -0.5

        def reference(query, cache):
            output = np.empty_like(query)
            for request, seq_len in enumerate(seq_lens_np):
                keys = []
                values = []
                for position in range(int(seq_len)):
                    block = block_table_np[request, position // 4]
                    offset = position % 4
                    keys.append(cache[block, 0, offset])
                    values.append(cache[block, 1, offset])
                keys = np.stack(keys)
                values = np.stack(values)
                for head in range(4):
                    kv_head = head // 2
                    scores = query[request, head] @ keys[:, kv_head].T * scale
                    scores = np.exp(scores - scores.max())
                    scores /= scores.sum()
                    output[request, head] = scores @ values[:, kv_head]
            return output

        for dtype, atol, rtol in (
            ("float16", 0.004, 0.004),
            ("bfloat16", 0.04, 0.02),
        ):
            with self.subTest(dtype=dtype):
                with jt.flag_scope(use_cuda=1), jt.no_grad():
                    query = jt.array(query_np).cast(dtype)
                    cache = jt.array(cache_np).cast(dtype)
                    result = _paged_attention_decode_cuda(
                        query,
                        cache,
                        jt.array(seq_lens_np),
                        jt.array(block_table_np),
                        scale,
                    )
                    self.assertIsNotNone(result)
                    actual, quantized_query, quantized_cache = jt.fetch_sync(
                        [result.float32(), query.float32(), cache.float32()])
                np.testing.assert_allclose(
                    actual, reference(quantized_query, quantized_cache),
                    atol=atol, rtol=rtol)

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
            actual = _layer_norm_no_grad_cuda(
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
    def test_lazy_cuda_device_index_uses_location_fallback(self):
        from jittor.nn._cuda_inference import device_index

        class LazyCudaValue:
            def get_device(self):
                return -1

            def location(self):
                return "device"

        self.assertEqual(device_index(LazyCudaValue()), 0)

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
