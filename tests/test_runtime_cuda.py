"""Focused CUDA regressions migrated with the TRELLIS runtime adapter.

Set ``JITTOR_TRELLIS_TEST_RUNTIME=1`` to run these tests. They intentionally
remain separate from the dependency-free entry-point tests because importing
Jittor initializes its compiler and the assertions require a CUDA device.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType
import unittest
from unittest import mock


sys.path.insert(0, os.fspath(Path(__file__).resolve().parents[1] / "src"))

from jittor_trellis import runtime


_RUN_RUNTIME_TESTS = os.environ.get("JITTOR_TRELLIS_TEST_RUNTIME") == "1"


@unittest.skipUnless(
    _RUN_RUNTIME_TESTS,
    "set JITTOR_TRELLIS_TEST_RUNTIME=1 to run Jittor/CUDA regressions",
)
class TestTrellisRuntimeCuda(unittest.TestCase):
    def test_multihead_rms_norm_fast_path_matches_reference(self):
        import numpy as np
        import jittor as jt

        if not jt.has_cuda:
            self.skipTest("No CUDA found")

        calls = []

        class MultiHeadRMSNorm:
            training = False

            def __init__(self, gamma):
                self.gamma = gamma
                self.scale = 128**0.5

            def forward(self, x, **kwargs):
                calls.append(kwargs)
                value = x.float32()
                norm = (value * value).sum(-1, keepdims=True).sqrt()
                return (
                    value / norm.maximum(1e-12) * self.gamma * self.scale
                ).cast(str(x.dtype))

        module = ModuleType(runtime._ATTENTION_MODULE)
        module.MultiHeadRMSNorm = MultiHeadRMSNorm
        self.assertTrue(runtime._patch_attention_module(module))
        patched = MultiHeadRMSNorm.forward
        self.assertTrue(runtime._patch_attention_module(module))
        self.assertIs(patched, MultiHeadRMSNorm.forward)

        rng = np.random.RandomState(127)
        x_np = rng.randn(2, 7, 12, 128).astype("float32")
        gamma_np = (1.0 + 0.1 * rng.randn(12, 128)).astype("float32")
        with jt.flag_scope(use_cuda=1), jt.no_grad():
            x = jt.array(x_np).bfloat16()
            instance = MultiHeadRMSNorm(jt.array(gamma_np))
            expected = patched._jittor_torch_original(instance, x)
            call_count = len(calls)
            actual = patched(instance, x)
            self.assertEqual(len(calls), call_count)
            expected_np, actual_np = jt.fetch_sync(
                [expected.float32(), actual.float32()]
            )
        np.testing.assert_allclose(
            actual_np, expected_np, atol=0.016, rtol=0.008
        )

        with jt.flag_scope(use_cuda=1), jt.no_grad(), mock.patch.dict(
            os.environ,
            {"JITTOR_TRELLIS_FUSED_RMS_NORM": "0"},
            clear=False,
        ):
            call_count = len(calls)
            patched(instance, x)
        self.assertEqual(len(calls), call_count + 1)

    def test_sparse_packed_self_attention_patch_is_guarded(self):
        calls = []

        class VarLenTensor:
            pass

        class SparseMultiHeadRMSNorm:
            def forward(self, value):
                return value

        class SparseMultiHeadAttention:
            def forward(self, value, context=None, *args, **kwargs):
                calls.append((value, context, args, kwargs))
                return "original"

        module = ModuleType(runtime._SPARSE_ATTENTION_API_MODULE)
        module.VarLenTensor = VarLenTensor
        module.SparseMultiHeadRMSNorm = SparseMultiHeadRMSNorm
        module.SparseMultiHeadAttention = SparseMultiHeadAttention
        self.assertTrue(runtime._patch_sparse_attention_api_module(module))
        patched = SparseMultiHeadAttention.forward
        self.assertTrue(runtime._patch_sparse_attention_api_module(module))
        self.assertIs(patched, SparseMultiHeadAttention.forward)

        attention = SparseMultiHeadAttention()
        sentinel = object()
        with mock.patch.object(
            runtime,
            "_trellis_sparse_packed_self_attention_fast_path",
            return_value=sentinel,
        ) as fast_path:
            self.assertIs(attention.forward("self"), sentinel)
            fast_path.assert_called_once_with(module, attention, "self")
        self.assertEqual(calls, [])

        with mock.patch.object(
            runtime,
            "_trellis_sparse_packed_self_attention_fast_path",
            return_value=None,
        ):
            self.assertEqual(attention.forward("fallback"), "original")
        self.assertEqual(attention.forward("cross", "context"), "original")
        self.assertEqual(
            calls,
            [
                ("fallback", None, (), {}),
                ("cross", "context", (), {}),
            ],
        )

    def test_sparse_modulated_layer_norm_patch_is_guarded(self):
        calls = []

        class SparseTensor:
            pass

        class ModulatedSparseTransformerCrossBlock:
            def _forward(self, value, modulation, context, *args, **kwargs):
                calls.append((value, modulation, context, args, kwargs))
                return "original"

        module = ModuleType(runtime._SPARSE_MODULATED_MODULE)
        module.SparseTensor = SparseTensor
        module.ModulatedSparseTransformerCrossBlock = (
            ModulatedSparseTransformerCrossBlock
        )
        self.assertTrue(runtime._patch_sparse_modulated_module(module))
        patched = ModulatedSparseTransformerCrossBlock._forward
        self.assertTrue(runtime._patch_sparse_modulated_module(module))
        self.assertIs(patched, ModulatedSparseTransformerCrossBlock._forward)

        block = ModulatedSparseTransformerCrossBlock()
        sentinel = object()
        with mock.patch.object(
            runtime,
            "_trellis_sparse_modulated_cross_block_fast_path",
            return_value=sentinel,
        ) as fast_path:
            self.assertIs(block._forward("x", "mod", "context"), sentinel)
            fast_path.assert_called_once_with(
                module, block, "x", "mod", "context"
            )
        self.assertEqual(calls, [])

        with mock.patch.object(
            runtime,
            "_trellis_sparse_modulated_cross_block_fast_path",
            return_value=None,
        ):
            self.assertEqual(
                block._forward("x", "mod", "context"), "original"
            )
        self.assertEqual(
            calls,
            [("x", "mod", "context", (), {})],
        )

    def test_cross_kv_cache_is_opt_in_and_sampler_scoped(self):
        import jittor as jt
        from jittor import nn

        if not jt.has_cuda:
            self.skipTest("No CUDA found")

        projection_calls = []
        received_dtypes = []

        class Projection(nn.Linear):
            def __init__(self):
                super().__init__(1024, 3072)
                self.weight = self.weight.bfloat16()
                self.bias = self.bias.bfloat16()
                self.weight.start_grad()
                self.bias.start_grad()
                self.is_train = False

            def execute(self, context):
                projection_calls.append(context)
                return super().execute(context)

        class Attention:
            _type = "cross"
            channels = 1536
            ctx_channels = 1024
            num_heads = 12
            head_dim = 128
            training = False

            def __init__(self):
                self.to_kv = Projection()

        class Model:
            training = False
            dtype = "bfloat16"

            def modules(self):
                return [attention]

        class FlowEulerSampler:
            def __init__(self):
                self.fail = False

            def sample(self, model, noise, cond=None, *args, **kwargs):
                del model, noise, args
                contexts = [cond, kwargs.get("neg_cond")]
                received_dtypes.append(
                    tuple(str(value.dtype) for value in contexts)
                )
                contexts = [value.bfloat16() for value in contexts]
                outputs = []
                for context in contexts:
                    outputs.append(attention.to_kv(context))
                    outputs.append(attention.to_kv(context))
                if self.fail:
                    raise RuntimeError("expected test failure")
                return contexts, outputs

        sampler_module = ModuleType(runtime._FLOW_EULER_MODULE)
        sampler_module.FlowEulerSampler = FlowEulerSampler
        self.assertTrue(runtime._patch_flow_euler_module(sampler_module))
        patched_sample = FlowEulerSampler.sample
        self.assertTrue(runtime._patch_flow_euler_module(sampler_module))
        self.assertIs(patched_sample, FlowEulerSampler.sample)

        model = Model()
        sampler = FlowEulerSampler()
        with jt.flag_scope(use_cuda=1), jt.no_grad():
            attention = Attention()
            source = jt.randn((1, 1029, 1024)).stop_grad()
            neg_source = jt.randn((1, 1029, 1024)).stop_grad()
            with mock.patch.dict(
                os.environ,
                {
                    "JITTOR_TRELLIS_CROSS_KV_CACHE": "1",
                    "JITTOR_TRELLIS_CROSS_KV_CACHE_MB": "384",
                },
                clear=False,
            ):
                result = sampler.sample(
                    model, None, source, neg_cond=neg_source
                )
                self.assertEqual(received_dtypes[-1], ("bfloat16", "bfloat16"))
                self.assertIs(result[1][0], result[1][1])
                self.assertIs(result[1][2], result[1][3])
                self.assertEqual(len(projection_calls), 2)
                self.assertIsNone(runtime._CROSS_KV_CACHE_SCOPE.get())
                self.assertNotIn("forward", attention.to_kv.__dict__)

                sampler.sample(model, None, source, neg_cond=neg_source)
                self.assertEqual(len(projection_calls), 4)

                sampler.fail = True
                with self.assertRaisesRegex(
                    RuntimeError, "expected test failure"
                ):
                    sampler.sample(model, None, source, neg_cond=neg_source)
                self.assertIsNone(runtime._CROSS_KV_CACHE_SCOPE.get())
                self.assertNotIn("forward", attention.to_kv.__dict__)
                sampler.fail = False
                self.assertEqual(len(projection_calls), 6)

            with mock.patch.dict(
                os.environ,
                {"JITTOR_TRELLIS_CROSS_KV_CACHE": "0"},
                clear=False,
            ):
                sampler.sample(model, None, source, neg_cond=neg_source)
            self.assertEqual(received_dtypes[-1], ("float32", "float32"))
            self.assertEqual(len(projection_calls), 10)

    def test_c2s_topology_cache_is_pair_scoped(self):
        import numpy as np
        import jittor as jt
        import jittor as torch

        if not jt.has_cuda:
            self.skipTest("No CUDA found")

        calls = []

        class SparseTensor:
            def __init__(self, feats, coords, shape=None):
                self.feats = feats
                self.coords = coords
                self._shape = shape
                self._scale = (1, 1, 1)
                self._spatial_cache = {}

            @property
            def device(self):
                return self.coords.device

            def get_spatial_cache(self, key=None):
                bucket = self._spatial_cache.get(str(self._scale), {})
                return bucket if key is None else bucket.get(key)

        class SparseChannel2Spatial:
            factor = 2
            training = False

            def forward(self, x, subdivision=None):
                calls.append((x, subdivision))
                return x

        module = ModuleType(runtime._C2S_MODULE)
        module.torch = torch
        module.SparseTensor = SparseTensor
        module.SparseChannel2Spatial = SparseChannel2Spatial
        self.assertTrue(runtime._patch_c2s_module(module))

        class SparseResBlockC2S3d:
            training = False

            def __init__(self, updown):
                self.updown = updown

            def _forward(self, first, second, subdivision):
                return (
                    self.updown.forward(first, subdivision),
                    self.updown.forward(second, subdivision),
                )

        block_module = ModuleType(runtime._C2S_BLOCK_MODULE)
        block_module.SparseResBlockC2S3d = SparseResBlockC2S3d
        self.assertTrue(runtime._patch_c2s_block_module(block_module))

        coords_np = np.array(
            [[0, 1, 2, 3], [0, 4, 5, 6], [1, 0, 1, 2]],
            dtype=np.int32,
        )
        subdivision_np = np.array(
            [
                [1, 0, 0, 1, 0, 0, 0, 1],
                [0, 1, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 1, 1, 0],
            ],
            dtype=np.bool_,
        )
        feats1_np = np.arange(48, dtype=np.float16).reshape(3, 16)
        feats2_np = (100 + np.arange(24)).astype(np.float16).reshape(3, 8)

        layer = SparseChannel2Spatial()
        with jt.flag_scope(use_cuda=1), jt.no_grad(), mock.patch.dict(
            os.environ,
            {"JITTOR_TRELLIS_C2S_TOPOLOGY_CACHE": "1"},
            clear=False,
        ):
            coords = jt.array(coords_np)
            subdivision = SparseTensor(jt.array(subdivision_np), coords)
            first = SparseTensor(jt.array(feats1_np), coords)
            second = SparseTensor(jt.array(feats2_np), coords)
            shared_cache = {}
            first._spatial_cache = shared_cache
            second._spatial_cache = shared_cache
            out1, out2 = SparseResBlockC2S3d(layer)._forward(
                first, second, subdivision
            )
            self.assertIs(out1.coords, out2.coords)
            self.assertIsNone(runtime._C2S_PAIR_SCOPE.get())
            self.assertEqual(shared_cache, {})
            fetched = jt.fetch_sync(
                [out1.coords, out1.feats, out2.coords, out2.feats]
            )

        self.assertEqual(len(calls), 0)
        rows, subidx = np.nonzero(subdivision_np)
        expected_coords = np.repeat(coords_np, subdivision_np.sum(1), axis=0)
        expected_coords[:, 1:] *= 2
        for index in range(3):
            expected_coords[:, index + 1] += subidx // (2**index) % 2
        expected1 = feats1_np.reshape(3 * 8, -1)[rows * 8 + subidx]
        expected2 = feats2_np.reshape(3 * 8, -1)[rows * 8 + subidx]
        np.testing.assert_array_equal(fetched[0], expected_coords)
        np.testing.assert_array_equal(fetched[2], expected_coords)
        np.testing.assert_array_equal(fetched[1], expected1)
        np.testing.assert_array_equal(fetched[3], expected2)

    def test_flexible_mesh_finalizer_matches_reference(self):
        import numpy as np
        import jittor as jt

        if not jt.has_cuda:
            self.skipTest("No CUDA found")

        coords_np = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [2, 0, 0],
                [3, 0, 0],
                [3, 1, 0],
                [2, 1, 0],
            ],
            dtype=np.int32,
        )
        dual_np = np.arange(24, dtype=np.float32).reshape(8, 3) / 20
        quads_np = np.array(
            [[0, 1, 2, 3], [-1, -1, -1, -1], [4, 5, 6, 7]],
            dtype=np.int32,
        )
        rows_np = np.array([0, 2], dtype=np.int32)
        weights_np = np.array(
            [[1], [1], [1], [1], [2], [1], [3], [1]],
            dtype=np.float32,
        )
        voxel_np = np.array([0.125, 0.25, 0.5], dtype=np.float32)
        aabb_np = np.array(
            [[-0.5, -0.25, 0.125], [0.5, 0.75, 1.125]],
            dtype=np.float32,
        )

        with jt.flag_scope(use_cuda=1), jt.no_grad():
            vertices, faces = runtime._trellis_finalize_flexible_mesh(
                jt.array(coords_np),
                jt.array(dual_np),
                jt.array(quads_np),
                jt.array(rows_np),
                jt.array(weights_np),
                jt.array(voxel_np),
                jt.array(aabb_np),
            )
            vertices_np, faces_np = jt.fetch_sync([vertices, faces])

        expected_vertices = (
            (coords_np.astype(np.float32) + dual_np) * voxel_np + aabb_np[0]
        )
        expected_faces = np.array(
            [[0, 1, 3], [3, 1, 2], [4, 5, 6], [4, 6, 7]],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(vertices_np, expected_vertices)
        np.testing.assert_array_equal(faces_np, expected_faces)


if __name__ == "__main__":
    unittest.main()
