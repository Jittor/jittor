"""Multi-rank NCCL regression for true flat FSDP2 sharding and updates.

Run through the maintained gate rather than invoking pytest directly::

    python -m nox -s nccl

Set ``JITTOR_NCCL_WORLD_SIZE`` and expose at least that many devices to run
more than the default two ranks.
"""

from _helpers import capability as _test_capability

import importlib
import math
import os
import unittest

import numpy as np

import jittor as jt
from jittor import nn
from jittor.compat import fsdp2


def _rank_data(rank):
    value = float(rank)
    inputs = np.array(
        [
            [1.0 + 0.25 * value, 2.0 - 0.5 * value, -1.0 + value, 0.5],
            [-2.0 + 0.5 * value, 0.25 + value, 1.5, 3.0 - 0.25 * value],
        ],
        dtype="float32",
    )
    targets = np.array(
        [
            [0.25 - 0.25 * value, -0.5 + value, 1.0 - 0.25 * value],
            [1.5 - 0.25 * value, -0.5 * value, -1.0 + 0.5 * value],
        ],
        dtype="float32",
    )
    return inputs, targets


def _linear_grads(weight, bias, inputs, target):
    output = inputs @ weight.T + bias
    grad_output = 2.0 * (output - target) / output.size
    return grad_output.T @ inputs, grad_output.sum(axis=0)


@unittest.skipUnless(
    _test_capability.check_accelerator('cuda', backend=jt).enabled and int(jt.world_size) >= 2 and fsdp2._common._in_true_distributed(),
    "requires the multi-rank NCCL nox gate",
)
class TestFSDP2Nccl(unittest.TestCase):
    @jt.flag_scope(use_cuda=1, use_parallel_op_compiler=0)
    def test_nested_sharding_and_full_state_reload(self):
        class NestedModel(nn.Module):
            def __init__(self):
                self.inner = nn.Linear(4, 3)
                self.output_bias = jt.ones((3,))

            def execute(self, value):
                return self.inner(value) + self.output_bias

        model = NestedModel()
        full_state = {
            name: value.clone() for name, value in model.state_dict().items()
        }
        original_numel = sum(int(value.numel()) for value in full_state.values())
        fsdp2.fully_shard(model.inner)
        fsdp2.fully_shard(model)
        child_state = model.inner._fsdp_state
        root_state = model._fsdp_state
        managed_numel = sum(
            entry.numel
            for state in (child_state, root_state)
            for entry in state.true_fsdp_params
        )
        self.assertEqual(managed_numel, original_numel)
        self.assertEqual(
            [entry.name for entry in root_state.true_fsdp_params],
            ["output_bias"],
        )

        if int(jt.rank) == 0:
            full_state["output_bias"] = jt.ones((3,)) * 7
        state_dict_api = importlib.import_module(
            "torch.distributed.checkpoint.state_dict")
        state_dict_api.set_model_state_dict(model, full_state)
        np.testing.assert_array_equal(
            model.output_bias.full_tensor().numpy(),
            np.full((3,), 7, dtype="float32"),
        )
        output = model(jt.ones((2, 4)))
        self.assertEqual(tuple(output.shape), (2, 3))
        self.assertTrue(np.isfinite(output.numpy()).all())

    @jt.flag_scope(use_cuda=1, use_parallel_op_compiler=0)
    def test_torch_distributed_world_collectives(self):
        dist = importlib.import_module("torch.distributed")
        dist.init_process_group(
            backend="cpu:gloo,cuda:nccl",
            rank=int(jt.rank),
            world_size=int(jt.world_size),
        )
        self.assertTrue(dist.is_initialized())
        self.assertEqual(dist.get_rank(), int(jt.rank))
        world_size = int(jt.world_size)
        self.assertEqual(dist.get_world_size(), world_size)
        self.assertEqual(dist.group.WORLD.rank(), int(jt.rank))
        self.assertEqual(dist.group.WORLD.size(), world_size)

        reduced = jt.array(np.asarray([int(jt.rank) + 1], dtype="float32"))
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        total = world_size * (world_size + 1) / 2
        np.testing.assert_array_equal(reduced.numpy(), np.asarray([total]))

        for op, expected in (
            (dist.ReduceOp.MAX, float(world_size)),
            (dist.ReduceOp.MIN, 1.0),
            (dist.ReduceOp.PRODUCT, float(math.factorial(world_size))),
        ):
            value = jt.array(np.asarray([int(jt.rank) + 1], dtype="float32"))
            dist.all_reduce(value, op=op)
            np.testing.assert_array_equal(value.numpy(), np.asarray([expected]))

        gathered = [jt.zeros_like(reduced) for _ in range(world_size)]
        dist.all_gather(gathered, reduced)
        for value in gathered:
            np.testing.assert_array_equal(value.numpy(), np.asarray([total]))

        objects = [None] * world_size
        dist.all_gather_object(objects, {"rank": int(jt.rank)})
        expected_objects = [{"rank": rank} for rank in range(world_size)]
        self.assertEqual(objects, expected_objects)

        gathered_objects = [None] * world_size if int(jt.rank) == 0 else None
        dist.gather_object(
            {"rank": int(jt.rank)}, gathered_objects, dst=0)
        if int(jt.rank) == 0:
            self.assertEqual(
                gathered_objects, expected_objects)
        dist.barrier()

    @jt.flag_scope(use_cuda=1, use_parallel_op_compiler=0)
    def test_world_and_subgroup_process_groups_coexist(self):
        dist = importlib.import_module("torch.distributed")
        dist.init_process_group(
            backend="cpu:gloo,cuda:nccl",
            rank=int(jt.rank),
            world_size=int(jt.world_size),
        )
        rank = int(jt.rank)
        world_size = int(jt.world_size)

        # A second full-membership group must own a communicator distinct from
        # WORLD. Reversing rank order also proves that group-local rank is not
        # another spelling of the global rank.
        reversed_group = dist.new_group(list(reversed(range(world_size))))
        self.assertNotEqual(
            reversed_group._backend_handle,
            dist.group.WORLD._backend_handle,
        )
        self.assertEqual(reversed_group.rank(), world_size - rank - 1)

        subgroup_value = jt.array(
            np.asarray([rank + 1], dtype="float32")
        )
        dist.all_reduce(
            subgroup_value, op=dist.ReduceOp.SUM, group=reversed_group
        )
        total = world_size * (world_size + 1) / 2
        np.testing.assert_array_equal(
            subgroup_value.numpy(), np.asarray([total], dtype="float32")
        )

        # Build one singleton communicator per rank. All ranks create the same
        # groups in the same order, then use only the group they belong to.
        singleton_groups = [
            dist.new_group([owner]) for owner in range(world_size)
        ]
        local_group = singleton_groups[rank]
        local_value = jt.array(
            np.asarray([100 + rank], dtype="float32")
        )
        dist.all_reduce(local_value, op=dist.ReduceOp.SUM, group=local_group)
        np.testing.assert_array_equal(
            local_value.numpy(),
            np.asarray([100 + rank], dtype="float32"),
        )

        # WORLD remains usable after subgroup construction and execution. This
        # is the composition DDP + tensor-parallel groups need.
        world_value = jt.array(np.asarray([rank + 1], dtype="float32"))
        dist.all_reduce(world_value, op=dist.ReduceOp.SUM)
        np.testing.assert_array_equal(
            world_value.numpy(), np.asarray([total], dtype="float32")
        )

    @jt.flag_scope(use_cuda=1, use_parallel_op_compiler=0)
    def test_two_dimensional_mesh_hybrid_gradient(self):
        from jittor.compat.fsdp2.dtensor import init_device_mesh

        world = int(jt.world_size)
        if world < 4 or world % 2:
            self.skipTest("requires an even world of at least four ranks")
        mesh = init_device_mesh("cuda", (world // 2, 2), mesh_dim_names=("replicate", "shard"))
        jt.seed(932)
        model = nn.Linear(4, 3)
        initial = {name: np.array(param.numpy(), copy=True)
                   for name, param in model.named_parameters()}
        fsdp2.fully_shard(model, mesh=mesh)
        state = model._fsdp_state
        self.assertEqual(state.true_fsdp_world_size, 2)
        self.assertEqual(state.true_fsdp_rank, int(jt.rank) % 2)
        inputs, target = _rank_data(int(jt.rank))
        output = model(jt.array(inputs))
        loss = ((output - jt.array(target)) ** 2).mean()
        model.sharded_sgd_step(loss, lr=0.03)
        gradients = [_linear_grads(initial["weight"], initial["bias"], *_rank_data(rank))
                     for rank in range(world)]
        expected = {name: initial[name] - 0.03 * np.mean([g[index] for g in gradients], axis=0)
                    for index, name in enumerate(("weight", "bias"))}
        for entry in state.true_fsdp_params:
            np.testing.assert_allclose(entry.shard.full_tensor().numpy(), expected[entry.name],
                                       rtol=2e-5, atol=2e-5)

    @jt.flag_scope(use_cuda=1, use_parallel_op_compiler=0)
    def test_mesh_norm_combines_groups_without_replicating(self):
        from jittor.compat.fsdp2.dtensor import init_device_mesh
        from jittor.compat.torch.grad import _get_total_norm_device

        world = int(jt.world_size)
        mesh = init_device_mesh("cuda", (world, 1), mesh_dim_names=("replicate", "shard"))
        singleton = mesh["shard"].get_group()
        world_group = mesh["replicate"].get_group()
        replicated = jt.array([3.0])
        sharded = jt.array([float(int(jt.rank) + 1)])
        object.__setattr__(replicated, "_fsdp_norm_group", singleton)
        object.__setattr__(sharded, "_fsdp_norm_group", world_group)
        norm = _get_total_norm_device([replicated, sharded], 2, shard_reduce=True)
        expected = math.sqrt(9 + sum(rank * rank for rank in range(1, world + 1)))
        self.assertAlmostEqual(float(norm.item()), expected, places=5)
        zero_norm = _get_total_norm_device([replicated, sharded], 0, shard_reduce=True)
        self.assertEqual(float(zero_norm.item()), 2.0)

    @jt.flag_scope(use_cuda=1, use_parallel_op_compiler=0)
    def test_mesh_reordered_shards_custom_adam_and_lifetime(self):
        import weakref
        import torch
        from jittor.compat.fsdp2.dtensor import DeviceMesh
        from jittor.compat.fsdp2.common import StateRecord

        class CustomAdam(torch.optim.Adam):
            def step(self, *args, **kwargs):
                self.custom_calls = getattr(self, "custom_calls", 0) + 1
                return super().step(*args, **kwargs)

        def lifetime_ref(tensor):
            # Native Var lacks a weakref slot. A marker owned only by its
            # instance dict tracks the same lifetime without retaining it.
            marker = StateRecord()
            object.__setattr__(tensor, "_fsdp_lifetime_probe", marker)
            return weakref.ref(marker)

        jt.seed(1234)
        model = torch.nn.Linear(4, 3)
        initial = {name: np.array(param.numpy(), copy=True)
                   for name, param in model.named_parameters()}
        ranks = list(reversed(range(int(jt.world_size))))
        mesh = DeviceMesh("cuda", ranks)
        fsdp2.fully_shard(model, mesh=mesh)
        state = model._fsdp_state
        self.assertEqual(state.true_fsdp_rank, ranks.index(int(jt.rank)))
        self.assertFalse(state.true_fsdp_flat_shard._is_view(),
                         "a local owned shard must not retain the full allocation as a view")
        opt = CustomAdam(model.parameters(), lr=0.01, eps=1e-6)
        expected = initial
        moment = {name: np.zeros_like(v) for name, v in expected.items()}
        variance = {name: np.zeros_like(v) for name, v in expected.items()}
        for step in range(1, 4):
            opt.zero_grad()
            x, target = _rank_data(int(jt.rank))
            output = model(torch.tensor(x))
            loss = ((output - torch.tensor(target)) ** 2).mean()
            loss.backward()
            old_shards = [lifetime_ref(entry.shard) for entry in state.true_fsdp_params]
            opt.step()
            jt.sync_all(True)
            del output, loss
            gradients = [_linear_grads(expected["weight"], expected["bias"], *_rank_data(rank))
                         for rank in ranks]
            for index, name in enumerate(("weight", "bias")):
                grad = np.mean([g[index] for g in gradients], axis=0)
                moment[name] = 0.9 * moment[name] + 0.1 * grad
                variance[name] = 0.999 * variance[name] + 0.001 * grad * grad
                expected[name] -= (0.01 / (1 - 0.9 ** step)) * moment[name] / (
                    np.sqrt(variance[name]) / np.sqrt(1 - 0.999 ** step) + 1e-6)
            for entry in state.true_fsdp_params:
                np.testing.assert_allclose(entry.shard.full_tensor().numpy(), expected[entry.name],
                                           rtol=2e-5, atol=2e-5)
                self.assertIsNone(entry.full_param)
            # Superseded views must drop promptly, without forcing cyclic GC.
            self.assertTrue(all(ref() is None for ref in old_shards))
        self.assertEqual(opt.custom_calls, 3)

    @jt.flag_scope(use_cuda=1, use_parallel_op_compiler=0)
    def test_nccl_all_gather_autograd(self):
        world_size = int(jt.world_size)
        rank = int(jt.rank)
        local = jt.array(
            np.asarray([rank + 0.25, rank + 0.75], dtype="float32")
        )
        gathered = fsdp2._common._all_gather_shards(local)
        indices = jt.arange(int(gathered.numel()), dtype="float32")
        weights = indices + float((rank + 1) * 100)
        local_grad = jt.grad((gathered * weights).sum(), local)

        local_indices = np.arange(rank * 2, rank * 2 + 2, dtype="float32")
        consumer_sum = world_size * (world_size + 1) / 2
        expected = world_size * local_indices + 100.0 * consumer_sum
        np.testing.assert_allclose(
            local_grad.float32().numpy(), expected, rtol=0, atol=0
        )

    @jt.flag_scope(use_cuda=1, use_parallel_op_compiler=0)
    def test_flat_shard_collectives_and_sgd_update(self):
        rank = int(jt.rank)
        jt.seed(20260825)
        model = nn.Linear(4, 3)
        initial = {
            name: np.asarray(param.float32().numpy()).copy()
            for name, param in model.named_parameters()
        }

        fsdp2.fully_shard(model)
        state = model._fsdp_state
        self.assertTrue(state.true_fsdp_initialized)
        self.assertTrue(state.true_fsdp_flat)
        world_size = int(jt.world_size)
        self.assertEqual(state.true_fsdp_world_size, world_size)
        self.assertEqual(state.true_fsdp_rank, rank)
        self.assertEqual(state.true_fsdp_flat_total_numel, 15)
        self.assertEqual(
            state.true_fsdp_flat_shard_numel,
            (state.true_fsdp_flat_total_numel + world_size - 1) // world_size,
        )

        local_before = np.asarray(state.true_fsdp_flat_shard.float32().numpy()).copy()
        gathered_before = (
            fsdp2._common._all_gather_shards(state.true_fsdp_flat_shard).float32().numpy()
        )
        gathered_before = np.asarray(gathered_before).reshape(-1)[
            : state.true_fsdp_flat_total_numel
        ]
        expected_before = np.concatenate(
            [initial[entry.name].reshape(-1) for entry in state.true_fsdp_params]
        )
        np.testing.assert_allclose(gathered_before, expected_before, rtol=0, atol=0)

        host_inputs, host_target = _rank_data(rank)
        inputs = jt.array(host_inputs)
        target = jt.array(host_target)
        output = model(inputs)
        loss = ((output - target) * (output - target)).mean()
        learning_rate = 0.05
        sharded_grads = model.sharded_sgd_step(loss, lr=learning_rate)

        gathered_after = (
            fsdp2._common._all_gather_shards(state.true_fsdp_flat_shard).float32().numpy()
        )
        gathered_after = np.asarray(gathered_after).reshape(-1)[: state.true_fsdp_flat_total_numel]
        weight_grads = []
        bias_grads = []
        for data_rank in range(world_size):
            host_inputs, host_target = _rank_data(data_rank)
            weight_grad, bias_grad = _linear_grads(
                initial["weight"], initial["bias"], host_inputs, host_target
            )
            weight_grads.append(weight_grad)
            bias_grads.append(bias_grad)
        expected = {
            "weight": initial["weight"] - learning_rate * np.mean(weight_grads, axis=0),
            "bias": initial["bias"] - learning_rate * np.mean(bias_grads, axis=0),
        }
        expected_grads = {
            "weight": np.mean(weight_grads, axis=0),
            "bias": np.mean(bias_grads, axis=0),
        }
        for entry, grad in zip(state.true_fsdp_params, sharded_grads):
            self.assertIs(grad.to_local(), grad)
            np.testing.assert_allclose(
                grad.full_tensor().float32().numpy(), expected_grads[entry.name],
                rtol=2e-5, atol=2e-5)
        expected_after = np.concatenate(
            [expected[entry.name].reshape(-1) for entry in state.true_fsdp_params]
        )
        np.testing.assert_allclose(gathered_after, expected_after, rtol=2e-5, atol=2e-5)

        local_after = np.asarray(state.true_fsdp_flat_shard.float32().numpy()).copy()
        self.assertEqual(local_after.shape, local_before.shape)
        self.assertTrue(np.isfinite(local_after).all())
        self.assertGreater(float(np.max(np.abs(local_after - local_before))), 0.0)
        self.assertIsNotNone(fsdp2._common._nccl_ops())

    @jt.flag_scope(use_cuda=1, use_parallel_op_compiler=0)
    def test_nonflat_gradient_full_tensor(self):
        rank = int(jt.rank)
        jt.seed(20260825)
        model = nn.Linear(4, 3)
        initial = {
            name: np.asarray(param.float32().numpy()).copy()
            for name, param in model.named_parameters()
        }
        previous = os.environ.get("JITTOR_FSDP2_FLAT")
        os.environ["JITTOR_FSDP2_FLAT"] = "0"
        try:
            fsdp2.fully_shard(model)
        finally:
            if previous is None:
                os.environ.pop("JITTOR_FSDP2_FLAT", None)
            else:
                os.environ["JITTOR_FSDP2_FLAT"] = previous
        state = model._fsdp_state
        self.assertFalse(state.true_fsdp_flat)

        host_inputs, host_target = _rank_data(rank)
        inputs = jt.array(host_inputs)
        target = jt.array(host_target)
        output = model(inputs)
        loss = ((output - target) * (output - target)).mean()
        sharded_grads = model.sharded_sgd_step(loss, lr=0.0)

        weight_grads = []
        bias_grads = []
        for data_rank in range(int(jt.world_size)):
            host_inputs, host_target = _rank_data(data_rank)
            weight_grad, bias_grad = _linear_grads(
                initial["weight"], initial["bias"], host_inputs, host_target
            )
            weight_grads.append(weight_grad)
            bias_grads.append(bias_grad)
        expected_grads = {
            "weight": np.mean(weight_grads, axis=0),
            "bias": np.mean(bias_grads, axis=0),
        }
        for entry, grad in zip(state.true_fsdp_params, sharded_grads):
            self.assertIs(grad.to_local(), grad)
            self.assertEqual(tuple(grad.shape), (entry.shard_numel,))
            np.testing.assert_allclose(
                grad.full_tensor().float32().numpy(), expected_grads[entry.name],
                rtol=2e-5, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
