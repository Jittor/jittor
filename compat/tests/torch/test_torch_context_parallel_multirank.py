"""Multi-rank context-parallel cases (NCCL launch required).

Run with the maintained multi-rank entry point, for example::

    jtrun --nproc-per-node=2 -m pytest -q \\
        compat/tests/torch/test_torch_context_parallel_multirank.py

Every case executes inside each rank; the reference is the same full-sequence
computation reconstructed locally, so no cross-rank assertion exchange is needed.
"""

import unittest

import numpy as np

import jittor as jt
import torch

from _helpers import capability as _test_capability
from jittor.compat.fsdp2 import common as fsdp2_common
from jittor.compat.fsdp2 import _context_parallel as cp


_HAS_CUDA = _test_capability.check_accelerator("cuda", backend=jt).enabled
_RANKS = 2


def _deterministic_tensor(shape, device, offset=0.0):
    # torch constructors, not jt primitives: under the shim these are the
    # tensor class whose .backward()/grad surface the tests exercise.
    size = 1
    for dim in shape:
        size *= int(dim)
    values = (torch.arange(size, dtype=torch.float32) + 1.0 + offset) / float(size)
    values = values.reshape(shape)
    return values.cuda() if device == "cuda" else values


@unittest.skipUnless(
    _HAS_CUDA and int(jt.world_size) >= _RANKS
    and fsdp2_common._in_true_distributed(),
    "requires a multi-rank NCCL launch",
)
class TestContextParallelMultiRank(unittest.TestCase):
    def setUp(self):
        from torch.distributed.device_mesh import init_device_mesh

        self.rank = int(jt.rank)
        self.world_size = int(jt.world_size)
        self.device = "cuda"
        self.mesh = init_device_mesh(
            "cuda", (self.world_size,), mesh_dim_names=("cp",))

    def test_distribute_tensor_none_source_keeps_each_rank_local_data(self):
        from torch.distributed.tensor import Shard, distribute_tensor

        full = _deterministic_tensor((4, 2), self.device,
                                     offset=100.0 * self.rank)
        local = distribute_tensor(
            full, self.mesh, [Shard(0)], src_data_rank=None).to_local()
        chunk = 4 // self.world_size
        start = self.rank * chunk
        np.testing.assert_allclose(
            local.numpy(), full[start:start + chunk].numpy())

        # The default (src_data_rank=0) still broadcasts rank 0's full tensor
        # before slicing, which is what makes the None behaviour observable.
        broadcast_local = distribute_tensor(
            full, self.mesh, [Shard(0)]).to_local()
        rank_zero = _deterministic_tensor((4, 2), self.device)
        np.testing.assert_allclose(
            broadcast_local.numpy(), rank_zero[start:start + chunk].numpy())

    def test_buffers_shard_by_rank_and_restore(self):
        from torch.distributed.tensor.experimental import context_parallel

        sequence = 8
        buffer = _deterministic_tensor((2, sequence), self.device)
        original = buffer.clone()
        local_len = sequence // self.world_size
        start = self.rank * local_len
        with context_parallel(
                self.mesh, buffers=[buffer], buffer_seq_dims=[1]):
            self.assertEqual(tuple(buffer.shape), (2, local_len))
            np.testing.assert_allclose(
                buffer.numpy(), original[:, start:start + local_len].numpy())
        self.assertEqual(tuple(buffer.shape), (2, sequence))
        np.testing.assert_allclose(buffer.numpy(), original.numpy())

    def test_attention_forward_and_backward_match_full_sequence(self):
        from torch.distributed.tensor.experimental import context_parallel

        sequence, heads, head_dim = 8, 2, 4
        local_len = sequence // self.world_size
        start = self.rank * local_len

        def make(offset):
            return _deterministic_tensor(
                (1, heads, sequence, head_dim), self.device, offset=offset)

        # Parameters, not requires_grad_ on an op result: a leaf is what the
        # shim accumulates .grad into, and it is what a real model hands FSDP.
        query = torch.nn.Parameter(make(0.0))
        key = torch.nn.Parameter(make(3.0))
        value = torch.nn.Parameter(make(7.0))
        reference_query = torch.nn.Parameter(make(0.0))
        reference_key = torch.nn.Parameter(make(3.0))
        reference_value = torch.nn.Parameter(make(7.0))

        reference = torch.nn.functional.scaled_dot_product_attention(
            reference_query, reference_key, reference_value, is_causal=True)

        marker = _deterministic_tensor((1, sequence), self.device)
        local_query = query[:, :, start:start + local_len]
        local_key = key[:, :, start:start + local_len]
        local_value = value[:, :, start:start + local_len]
        with context_parallel(
                self.mesh, buffers=[marker], buffer_seq_dims=[1]):
            output = torch.nn.functional.scaled_dot_product_attention(
                local_query, local_key, local_value, is_causal=True)

        np.testing.assert_allclose(
            output.numpy(),
            reference[:, :, start:start + local_len].numpy(),
            rtol=1e-5, atol=1e-5)

        output.sum().backward()
        reference.sum().backward()
        # Slices are non-leaf Vars; Jittor accumulates .grad on the leaves the
        # graph was built from, so read the same positions through those.
        np.testing.assert_allclose(
            query.grad[:, :, start:start + local_len].numpy(),
            reference_query.grad[:, :, start:start + local_len].numpy(),
            rtol=1e-5, atol=1e-5)
        # The all-gather backward is reduce-scatter: this rank's key/value
        # slice must receive the sum of every rank's contribution, matching the
        # full-sequence gradient for the same positions.
        np.testing.assert_allclose(
            key.grad[:, :, start:start + local_len].numpy(),
            reference_key.grad[:, :, start:start + local_len].numpy(),
            rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            value.grad[:, :, start:start + local_len].numpy(),
            reference_value.grad[:, :, start:start + local_len].numpy(),
            rtol=1e-5, atol=1e-5)

    def test_multidimensional_mesh_slice_and_flatten(self):
        from torch.distributed.device_mesh import DeviceMesh

        if self.world_size != 2:
            self.skipTest("the 2-D mesh case is written for two ranks")
        ranks = torch.arange(self.world_size).reshape(1, self.world_size)
        mesh = DeviceMesh(
            device_type="cuda", mesh=ranks,
            mesh_dim_names=("dp", "tp"))
        tp_mesh = mesh["tp"]
        self.assertEqual(tp_mesh.size(), self.world_size)
        self.assertEqual(tp_mesh.get_local_rank(), self.rank)
        flat = mesh["dp", "tp"]._flatten(mesh_dim_name="dp_tp")
        self.assertEqual(flat.size(), self.world_size)
        self.assertEqual(flat.get_local_rank(), self.rank)

    def test_cp_mesh_rank_matches_axis_coordinate(self):
        from torch.distributed.device_mesh import DeviceMesh

        if self.world_size != 2:
            self.skipTest("the 2-D mesh case is written for two ranks")
        ranks = torch.arange(self.world_size).reshape(1, self.world_size)
        mesh = DeviceMesh(
            device_type="cuda", mesh=ranks,
            mesh_dim_names=("dp", "cp"))
        state = cp._ContextParallelState(mesh["cp"], 8, 1)
        self.assertEqual(state.world_size, self.world_size)
        self.assertEqual(state.rank, int(mesh.get_coordinate()[1]))
        self.assertEqual(state.start, state.rank * state.local_len)


if __name__ == "__main__":
    unittest.main()
