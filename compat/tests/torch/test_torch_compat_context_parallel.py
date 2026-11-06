"""Context-parallel compatibility tests for ``import torch``.

Run:
    python -m pytest compat/tests/torch/test_torch_compat_context_parallel.py

The multi-rank cases (real sharding and cross-rank attention) are in
``test_torch_context_parallel_multirank.py`` and need an NCCL launch.
"""

import os
import unittest
from unittest import mock

import numpy as np
import torch
import jittor as jt

from jittor.compat.fsdp2 import _context_parallel as cp
from jittor.compat.torch.fidelity import Fidelity, fidelity_of


class TestContextParallelSurface(unittest.TestCase):
    def test_public_import_paths_share_one_implementation(self):
        from torch.distributed.tensor.experimental import context_parallel
        from torch.distributed.tensor.experimental._attention import (
            context_parallel as attention_context_parallel,
            set_rotate_method,
            _cp_options,
            _RotateMethod,
        )
        from torch.distributed.tensor.experimental._context_parallel import (
            context_parallel as private_context_parallel,
        )

        self.assertIs(context_parallel, cp.context_parallel)
        self.assertIs(attention_context_parallel, cp.context_parallel)
        self.assertIs(private_context_parallel, cp.context_parallel)
        self.assertIs(set_rotate_method, cp.set_rotate_method)
        self.assertIs(_cp_options, cp._cp_options)
        self.assertIs(_RotateMethod, cp._RotateMethod)
        self.assertEqual(
            context_parallel.__module__,
            "torch.distributed.tensor.experimental")

    def test_fidelity_is_approximate_and_states_the_comm_plan(self):
        record = fidelity_of(
            "torch.distributed.tensor.experimental.context_parallel")
        self.assertEqual(record.level, Fidelity.APPROXIMATE)
        self.assertIn("all-gather", record.detail)
        rotate = fidelity_of(
            "torch.distributed.tensor.experimental._attention.set_rotate_method")
        self.assertEqual(rotate.level, Fidelity.APPROXIMATE)

    def test_set_rotate_method_records_choice_and_rejects_unknown(self):
        previous = cp._cp_options.rotate_method
        try:
            cp.set_rotate_method("alltoall")
            self.assertEqual(cp._cp_options.rotate_method,
                             cp._RotateMethod.ALL_TO_ALL)
            cp.set_rotate_method("allgather")
            self.assertEqual(cp._cp_options.rotate_method,
                             cp._RotateMethod.ALL_GATHER)
            with self.assertRaises(NotImplementedError):
                cp.set_rotate_method("ring")
        finally:
            cp._cp_options.rotate_method = previous

    def test_get_node_local_rank_reads_launcher_environment(self):
        from torch.distributed import get_node_local_rank

        environment = dict(os.environ)
        try:
            os.environ.pop("LOCAL_RANK", None)
            self.assertEqual(get_node_local_rank(), 0)
            self.assertEqual(get_node_local_rank(fallback_rank=5), 5)
            self.assertEqual(get_node_local_rank(fallback_rank=None),
                             torch.distributed.get_rank())
            os.environ["LOCAL_RANK"] = "3"
            self.assertEqual(get_node_local_rank(), 3)
        finally:
            os.environ.clear()
            os.environ.update(environment)

    def test_single_rank_buffers_are_restored_by_default(self):
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor.experimental import context_parallel

        mesh = init_device_mesh("cpu", (1,))
        buffer = torch.arange(12, dtype=torch.float32).reshape(2, 6)
        original = buffer.clone()
        with context_parallel(mesh, buffers=[buffer], buffer_seq_dims=[1]):
            self.assertEqual(tuple(buffer.shape), (2, 6))
        self.assertTrue(bool((buffer - original).abs().max() == 0))
        self.assertEqual(tuple(buffer.shape), (2, 6))

    def test_no_restore_buffers_keep_the_shard(self):
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor.experimental import context_parallel

        mesh = init_device_mesh("cpu", (1,))
        kept = torch.arange(12, dtype=torch.float32).reshape(2, 6)
        restored = torch.arange(12, dtype=torch.float32).reshape(2, 6)
        original = restored.clone()
        with context_parallel(
                mesh, buffers=[kept, restored], buffer_seq_dims=[1, 1],
                no_restore_buffers=[kept]):
            pass
        self.assertTrue(bool((restored - original).abs().max() == 0))
        self.assertEqual(tuple(kept.shape), (2, 6))

    def test_single_rank_attention_matches_full_sequence(self):
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor.experimental import context_parallel

        mesh = init_device_mesh("cpu", (1,))
        buffer = torch.zeros(1, 6)
        q = torch.randn(1, 2, 6, 4, requires_grad=True)
        k = torch.randn(1, 2, 6, 4, requires_grad=True)
        v = torch.randn(1, 2, 6, 4, requires_grad=True)

        q_ref = q.detach().clone()
        k_ref = k.detach().clone()
        v_ref = v.detach().clone()
        q_ref.requires_grad_(True)
        k_ref.requires_grad_(True)
        v_ref.requires_grad_(True)

        with context_parallel(mesh, buffers=[buffer], buffer_seq_dims=[1]):
            inside = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True)
        reference = torch.nn.functional.scaled_dot_product_attention(
            q_ref, k_ref, v_ref, is_causal=True)
        self.assertTrue(bool((inside - reference).abs().max() == 0))

        inside.sum().backward()
        reference.sum().backward()
        for name, actual, expected in (
                ("q", q.grad, q_ref.grad),
                ("k", k.grad, k_ref.grad),
                ("v", v.grad, v_ref.grad)):
            with self.subTest(grad=name):
                self.assertTrue(bool((actual - expected).abs().max() == 0))

    def test_explicit_attention_mask_is_refused(self):
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor.experimental import context_parallel

        mesh = init_device_mesh("cpu", (1,))
        buffer = torch.zeros(1, 6)
        q = torch.randn(1, 2, 6, 4)
        k = torch.randn(1, 2, 6, 4)
        v = torch.randn(1, 2, 6, 4)
        mask = torch.zeros(6, 6, dtype=torch.bool)
        with context_parallel(mesh, buffers=[buffer], buffer_seq_dims=[1]):
            with self.assertRaises(NotImplementedError):
                torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask)

    def test_requires_buffers_and_matching_sequence_dims(self):
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor.experimental import context_parallel

        mesh = init_device_mesh("cpu", (1,))
        with self.assertRaises(ValueError):
            with context_parallel(mesh):
                pass
        with self.assertRaises(ValueError):
            with context_parallel(
                    mesh, buffers=[torch.zeros(2, 6)],
                    buffer_seq_dims=[1, 1]):
                pass
        with self.assertRaises(ValueError):
            with context_parallel(
                    mesh, buffers=[torch.zeros(2, 6)],
                    buffer_seq_dims=[1], no_restore_buffers=[torch.zeros(2)]):
                pass


class TestOffsetCausalMask(unittest.TestCase):
    """The mask that lets a local shard attend the global prefix exactly."""

    @staticmethod
    def _state(local_len, start, world_size):
        # The mesh/group is only used by shard() and _all_gather_sequence();
        # mask construction depends on the three layout numbers alone.
        state = object.__new__(cp._ContextParallelState)
        state.local_len = local_len
        state.start = start
        state.world_size = world_size
        return state

    def test_each_shard_masked_attention_equals_full_causal_attention(self):
        torch.manual_seed(7)
        sequence, heads, dim = 8, 2, 4
        world_size, local_len = 4, 2
        q = torch.randn(1, heads, sequence, dim)
        k = torch.randn(1, heads, sequence, dim)
        v = torch.randn(1, heads, sequence, dim)
        full = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True)
        for rank in range(world_size):
            start = rank * local_len
            state = self._state(local_len, start, world_size)
            mask = state.causal_mask()
            self.assertEqual(tuple(mask.shape), (local_len, sequence))
            local = torch.nn.functional.scaled_dot_product_attention(
                q[:, :, start:start + local_len], k, v, attn_mask=mask)
            with self.subTest(rank=rank):
                np.testing.assert_allclose(
                    local.numpy(),
                    full[:, :, start:start + local_len].numpy(),
                    rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
