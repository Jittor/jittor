"""Two-rank NCCL regression for true flat FSDP2 sharding and updates.

Run through the maintained gate rather than invoking pytest directly::

    python -m nox -s nccl
"""

import importlib
import unittest

import numpy as np

import jittor as jt
from jittor import nn
from jittor.compat import fsdp2


_INPUTS = (
    np.array([[1.0, 2.0, -1.0, 0.5], [-2.0, 0.25, 1.5, 3.0]], dtype="float32"),
    np.array([[0.5, -1.5, 2.0, 1.0], [3.0, -0.5, -2.0, 0.75]], dtype="float32"),
)
_TARGETS = (
    np.array([[0.25, -0.5, 1.0], [1.5, 0.0, -1.0]], dtype="float32"),
    np.array([[-0.5, 1.0, 0.25], [0.75, -1.25, 0.5]], dtype="float32"),
)


def _linear_grads(weight, bias, inputs, target):
    output = inputs @ weight.T + bias
    grad_output = 2.0 * (output - target) / output.size
    return grad_output.T @ inputs, grad_output.sum(axis=0)


@unittest.skipUnless(
    jt.has_cuda and int(jt.world_size) == 2 and fsdp2._common._in_true_distributed(),
    "requires the two-rank NCCL nox gate",
)
class TestFSDP2Nccl(unittest.TestCase):
    @jt.flag_scope(use_cuda=1, use_parallel_op_compiler=0)
    def test_nested_sharding_and_full_state_reload(self):
        class NestedModel(nn.Module):
            def __init__(self):
                self.inner = nn.Linear(4, 3)
                self.output_bias = jt.ones((3,))

            def forward(self, value):
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
            backend="nccl", rank=int(jt.rank), world_size=int(jt.world_size))
        self.assertTrue(dist.is_initialized())
        self.assertEqual(dist.get_rank(), int(jt.rank))
        self.assertEqual(dist.get_world_size(), 2)
        self.assertEqual(dist.group.WORLD.rank(), int(jt.rank))
        self.assertEqual(dist.group.WORLD.size(), 2)

        reduced = jt.array(np.asarray([int(jt.rank) + 1], dtype="float32"))
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        np.testing.assert_array_equal(reduced.numpy(), np.asarray([3.0]))

        for op, expected in (
            (dist.ReduceOp.MAX, 2.0),
            (dist.ReduceOp.MIN, 1.0),
            (dist.ReduceOp.PRODUCT, 2.0),
        ):
            value = jt.array(np.asarray([int(jt.rank) + 1], dtype="float32"))
            dist.all_reduce(value, op=op)
            np.testing.assert_array_equal(value.numpy(), np.asarray([expected]))

        gathered = [jt.zeros_like(reduced) for _ in range(2)]
        dist.all_gather(gathered, reduced)
        for value in gathered:
            np.testing.assert_array_equal(value.numpy(), np.asarray([3.0]))

        objects = [None, None]
        dist.all_gather_object(objects, {"rank": int(jt.rank)})
        self.assertEqual(objects, [{"rank": 0}, {"rank": 1}])

        gathered_objects = [None, None] if int(jt.rank) == 0 else None
        dist.gather_object(
            {"rank": int(jt.rank)}, gathered_objects, dst=0)
        if int(jt.rank) == 0:
            self.assertEqual(
                gathered_objects, [{"rank": 0}, {"rank": 1}])
        dist.barrier()

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
        self.assertEqual(state.true_fsdp_world_size, 2)
        self.assertEqual(state.true_fsdp_rank, rank)
        self.assertEqual(state.true_fsdp_flat_total_numel, 15)
        self.assertEqual(state.true_fsdp_flat_shard_numel, 8)

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

        inputs = jt.array(_INPUTS[rank])
        target = jt.array(_TARGETS[rank])
        output = model(inputs)
        loss = ((output - target) * (output - target)).mean()
        learning_rate = 0.05
        model.sharded_sgd_step(loss, lr=learning_rate)

        gathered_after = (
            fsdp2._common._all_gather_shards(state.true_fsdp_flat_shard).float32().numpy()
        )
        gathered_after = np.asarray(gathered_after).reshape(-1)[: state.true_fsdp_flat_total_numel]
        weight_grads = []
        bias_grads = []
        for host_inputs, host_target in zip(_INPUTS, _TARGETS):
            weight_grad, bias_grad = _linear_grads(
                initial["weight"], initial["bias"], host_inputs, host_target
            )
            weight_grads.append(weight_grad)
            bias_grads.append(bias_grad)
        expected = {
            "weight": initial["weight"] - learning_rate * np.mean(weight_grads, axis=0),
            "bias": initial["bias"] - learning_rate * np.mean(bias_grads, axis=0),
        }
        expected_after = np.concatenate(
            [expected[entry.name].reshape(-1) for entry in state.true_fsdp_params]
        )
        np.testing.assert_allclose(gathered_after, expected_after, rtol=2e-5, atol=2e-5)

        local_after = np.asarray(state.true_fsdp_flat_shard.float32().numpy()).copy()
        self.assertEqual(local_after.shape, local_before.shape)
        self.assertTrue(np.isfinite(local_after).all())
        self.assertGreater(float(np.max(np.abs(local_after - local_before))), 0.0)
        self.assertIsNotNone(fsdp2._common._nccl_ops())


if __name__ == "__main__":
    unittest.main()
