"""FSDP1 constructor arguments must select the compat FSDP2 plan.

The official 3D-parallel example wraps the tensor-parallel model with

    FSDP(model, device_mesh=dp_mesh, sharding_strategy=ShardingStrategy.NO_SHARD)

Both names used to fall through ``fully_shard(**kwargs)`` unread: the mesh was
ignored (so the model sharded over the whole world) and NO_SHARD sharded
anyway.  These cases pin the translation on one CPU rank; the DP>1 behaviour
is in ``test_torch_fsdp2_multirank.py``.
"""

import types
import unittest

import pytest

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy


pytestmark = pytest.mark.xdist_group("fsdp1_legacy_mapping")


def _mesh():
    return init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))


class TestLegacyArguments(unittest.TestCase):
    def test_no_shard_uses_a_one_rank_shard_group(self):
        wrapped = FullyShardedDataParallel(
            torch.nn.Linear(4, 4, bias=False), device_mesh=_mesh(),
            sharding_strategy=ShardingStrategy.NO_SHARD)
        state = wrapped._fsdp_state

        self.assertEqual(state.legacy_sharding_strategy, ShardingStrategy.NO_SHARD)
        self.assertEqual(state.shard_group.size(), 1)
        # A one-rank replicate mesh has nobody to synchronize with.
        self.assertIsNone(state.replicate_group)
        self.assertEqual(tuple(wrapped.module.weight.shape), (4, 4))

    def test_no_shard_without_a_mesh_does_not_shard_over_the_world(self):
        wrapped = FullyShardedDataParallel(
            torch.nn.Linear(4, 4, bias=False),
            sharding_strategy=ShardingStrategy.NO_SHARD)
        state = wrapped._fsdp_state

        self.assertEqual(state.shard_group.size(), 1)
        self.assertEqual(tuple(wrapped.module.weight.shape), (4, 4))

    def test_full_shard_keeps_the_requested_mesh(self):
        mesh = _mesh()
        wrapped = FullyShardedDataParallel(
            torch.nn.Linear(4, 4, bias=False), device_mesh=mesh)
        state = wrapped._fsdp_state

        self.assertEqual(state.legacy_sharding_strategy,
                         ShardingStrategy.FULL_SHARD)
        self.assertIs(state.mesh, mesh)
        self.assertTrue(state.reshard_after_forward)

    def test_shard_grad_op_keeps_parameters_whole_after_forward(self):
        wrapped = FullyShardedDataParallel(
            torch.nn.Linear(4, 4, bias=False), device_mesh=_mesh(),
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP)
        self.assertFalse(wrapped._fsdp_state.reshard_after_forward)

    def test_hybrid_shard_needs_an_explicit_two_axis_mesh(self):
        with self.assertRaisesRegex(NotImplementedError, "HYBRID_SHARD"):
            FullyShardedDataParallel(
                torch.nn.Linear(4, 4, bias=False), device_mesh=_mesh(),
                sharding_strategy=ShardingStrategy.HYBRID_SHARD)

    def test_auto_wrap_policy_is_refused(self):
        with self.assertRaisesRegex(NotImplementedError, "auto_wrap_policy"):
            FullyShardedDataParallel(
                torch.nn.Linear(4, 4, bias=False), device_mesh=_mesh(),
                auto_wrap_policy=lambda *args, **kwargs: False)

    def test_unknown_keyword_is_a_type_error(self):
        with self.assertRaisesRegex(TypeError, "unexpected keyword"):
            FullyShardedDataParallel(
                torch.nn.Linear(4, 4, bias=False), device_mesh=_mesh(),
                not_an_fsdp_option=True)

    def test_device_mesh_and_process_group_are_exclusive(self):
        with self.assertRaisesRegex(ValueError, "not both"):
            FullyShardedDataParallel(
                torch.nn.Linear(4, 4, bias=False), device_mesh=_mesh(),
                process_group=object())

    def test_ignored_states_become_ignored_params(self):
        module = torch.nn.Linear(4, 4, bias=True)
        ignored = module.bias
        wrapped = FullyShardedDataParallel(
            module, device_mesh=_mesh(), ignored_states=[ignored])
        self.assertIn(id(ignored), {id(p) for p in wrapped._fsdp_state.ignored_params})


class TestAttributeForwarding(unittest.TestCase):
    def test_unknown_attributes_reach_the_wrapped_module(self):
        module = torch.nn.Linear(4, 4, bias=False)
        module.config = types.SimpleNamespace(vocab_size=11)
        module.loss_function = lambda *args, **kwargs: "loss"
        wrapped = FullyShardedDataParallel(module, device_mesh=_mesh())

        self.assertEqual(wrapped.config.vocab_size, 11)
        self.assertEqual(wrapped.loss_function(), "loss")
        # nn.Module lookups that do exist are not shadowed by the forwarding.
        self.assertIs(wrapped.training, module.training)

    def test_parameters_and_state_dict_are_the_wrapped_ones(self):
        module = torch.nn.Linear(4, 4, bias=False)
        wrapped = FullyShardedDataParallel(module, device_mesh=_mesh())
        self.assertEqual(len(list(wrapped.parameters())),
                         len(list(module.parameters())))
        self.assertEqual(sorted(wrapped.state_dict()), sorted(module.state_dict()))


if __name__ == "__main__":
    unittest.main()
