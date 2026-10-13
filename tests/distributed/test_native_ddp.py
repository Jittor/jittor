"""Native DDP and Jittor optimizer lifecycle contracts."""

from contextlib import contextmanager
import unittest
from unittest.mock import patch

import jittor as jt
from jittor import nn
from jittor import distributed
from jittor.nn.parallel import DistributedDataParallel


class _Group:
    def __init__(self, size=2, ranks=None):
        self._size = size
        self.ranks = list(range(size)) if ranks is None else list(ranks)

    def size(self):
        return self._size


@contextmanager
def _native_dist(world_size, all_reduce=None, broadcast=None):
    """Supply the public API contract while exercising native DDP/Optimizer."""
    group = _Group(world_size)
    calls = {"all_reduce": [], "broadcast": []}

    def reduce(tensor, op="sum", group=None):
        calls["all_reduce"].append((tuple(tensor.shape), op))
        if all_reduce is None:
            return tensor
        return all_reduce(tensor, op)

    def send(tensor, src=0, group=None):
        calls["broadcast"].append((tuple(tensor.shape), src))
        if broadcast is None:
            return tensor
        return broadcast(tensor, src)

    with patch.object(distributed, "is_initialized", return_value=world_size > 1,
                      create=True), \
            patch.object(distributed, "get_world_size", return_value=world_size,
                         create=True), \
            patch.object(distributed, "get_default_group", return_value=group,
                         create=True), \
            patch.object(distributed, "all_reduce", side_effect=reduce,
                         create=True), \
            patch.object(distributed, "broadcast", side_effect=send,
                         create=True):
        yield group, calls


class _OneWeight(nn.Module):
    def __init__(self):
        self.weight = jt.array([2.0], dtype="float32")
        self.register_buffer("running", jt.array([5.0], dtype="float32"),
                             persistent=False)

    def execute(self, x):
        return self.weight * x


class _TwoWeights(_OneWeight):
    def __init__(self):
        super().__init__()
        self.second = jt.array([4.0], dtype="float32")

    def execute(self, x):
        return (self.weight + self.second) * x


class _UnusedWeight(_OneWeight):
    def __init__(self):
        super().__init__()
        self.unused = jt.array([7.0], dtype="float32")

    def execute(self, x):
        return self.weight * x


class TestNativeDistributedDataParallel(unittest.TestCase):
    def test_world_one_is_identity_and_optimizer_stays_native(self):
        with _native_dist(1) as (_group, calls):
            model = DistributedDataParallel(_OneWeight())
            optimizer = jt.optim.SGD(model.parameters(), lr=0.1)
            optimizer.step(model(jt.array([3.0])).sum())
            self.assertAlmostEqual(float(model.module.weight.numpy()[0]), 1.7, places=5)
            self.assertEqual(calls["all_reduce"], [])
            self.assertEqual(calls["broadcast"], [])
            self.assertEqual(model.named_parameters()[0][0], "module.weight")
            self.assertNotIn("module.running", model.state_dict())

    def test_constructor_broadcasts_parameters_and_all_buffers(self):
        with _native_dist(2) as (_group, calls):
            model = DistributedDataParallel(_OneWeight())
            self.assertEqual(calls["broadcast"], [((1,), 0), ((1,), 0)])
            self.assertEqual(model.buffers()[0].numpy().tolist(), [5.0])

    def test_no_sync_accumulates_then_means_once_before_step(self):
        # Pretend the remote rank's accumulated gradient is 4. Rank 0 has two
        # local micro-batches whose gradients sum to 2, so the global mean is 3.
        def average_with_remote(tensor, op):
            if tuple(tensor.shape) == (1,):
                return (tensor + 4.0) / 2.0
            return tensor

        with _native_dist(2, all_reduce=average_with_remote) as (_group, calls):
            model = DistributedDataParallel(_OneWeight())
            optimizer = jt.optim.SGD(model.parameters(), lr=0.1)
            with model.no_sync():
                optimizer.backward(model(jt.array([1.0])).sum())

            self.assertEqual(
                len([item for item in calls["all_reduce"] if item[0] == (1,)]), 0
            )
            with self.assertRaisesRegex(RuntimeError, "still local"):
                optimizer.step()

            optimizer.backward(model(jt.array([1.0])).sum())
            grad = optimizer.param_groups[0]["grads"][0]
            self.assertAlmostEqual(float(grad.numpy()[0]), 3.0, places=5)
            self.assertEqual(
                len([item for item in calls["all_reduce"] if item[0] == (1,)]), 1,
                "each DDP parameter must be reduced exactly once per synced backward",
            )
            optimizer.step()
            self.assertAlmostEqual(float(model.module.weight.numpy()[0]), 1.7, places=5)

    def test_optimizer_duplicate_parameter_is_rejected_before_reduction(self):
        with _native_dist(2) as (_group, calls):
            model = DistributedDataParallel(_OneWeight())
            parameter = model.module.weight
            optimizer = jt.optim.SGD([parameter, parameter], lr=0.1)
            with self.assertRaisesRegex(RuntimeError, "more than once"):
                optimizer.backward(model(jt.array([1.0])).sum())
            self.assertEqual(
                len([item for item in calls["all_reduce"] if item[0] == (1,)]), 0
            )

    def test_unused_trainable_parameter_reduces_zero_gradient(self):
        with _native_dist(2) as (_group, calls):
            model = DistributedDataParallel(_UnusedWeight())
            optimizer = jt.optim.SGD(model.parameters(), lr=0.1)
            optimizer.backward(model(jt.array([2.0])).sum())
            gradients = [gradient.numpy().tolist()
                         for gradient in optimizer.param_groups[0]["grads"]]
            self.assertEqual(gradients, [[2.0], [0.0]])
            self.assertEqual(
                len([item for item in calls["all_reduce"] if item[0] == (1,)]), 2,
                "all trainable parameters, including unused leaves, keep the same "
                "collective sequence on every rank",
            )

    def test_zero_grad_discards_unsynced_local_accumulation(self):
        with _native_dist(2) as (_group, calls):
            model = DistributedDataParallel(_OneWeight())
            optimizer = jt.optim.SGD(model.parameters(), lr=0.1)
            with model.no_sync():
                optimizer.backward(model(jt.array([1.0])).sum())
            optimizer.zero_grad()
            before = model.module.weight.numpy().copy()
            optimizer.step()
            self.assertEqual(model.module.weight.numpy(), before)
            self.assertEqual(
                len([item for item in calls["all_reduce"] if item[0] == (1,)]), 0
            )

    def test_missing_optimizer_parameter_is_rejected(self):
        with _native_dist(2):
            model = DistributedDataParallel(_TwoWeights())
            optimizer = jt.optim.SGD([model.module.weight], lr=0.1)
            with self.assertRaisesRegex(RuntimeError, "missing from the optimizer"):
                optimizer.backward(model(jt.array([1.0])).sum())


if __name__ == "__main__":
    unittest.main()
