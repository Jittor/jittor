"""Two-rank NCCL regression for true flat FSDP2 sharding and updates.

Run through the maintained gate rather than invoking pytest directly::

    python -m nox -s nccl
"""

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
