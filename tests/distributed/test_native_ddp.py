"""Native DDP and Jittor optimizer lifecycle contracts."""

from contextlib import contextmanager
import pickle
import unittest
from unittest.mock import patch

import jittor as jt
from jittor import nn
from jittor import distributed
from jittor.nn.parallel import DistributedDataParallel


class _Group:
    def __init__(self, size=2, ranks=None, backend_name=None):
        self._size = size
        self.ranks = list(range(size)) if ranks is None else list(ranks)
        self._backend_kind = None
        self._backend_name = backend_name

    def size(self):
        return self._size

    def _get_backend_name(self):
        return self._backend_name


@contextmanager
def _native_dist(world_size, all_reduce=None, broadcast=None, backend_name=None):
    """Supply the public API contract while exercising native DDP/Optimizer."""
    group = _Group(world_size, backend_name=backend_name)
    calls = {"all_reduce": [], "broadcast": []}

    def reduce(tensor, op="sum", group=None):
        calls["all_reduce"].append((tuple(tensor.shape), op))
        if all_reduce is None:
            # Identical peers: sum includes every rank; mean stays unchanged.
            return tensor * world_size if op == "sum" else tensor
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
        for world_size in (1, 2):
            with self.subTest(world_size=world_size), \
                    _native_dist(world_size) as (_group, calls):
                model = DistributedDataParallel(_OneWeight())
                parameter = model.module.weight
                optimizer = jt.optim.SGD([parameter, parameter], lr=0.1)
                calls["all_reduce"].clear()
                with self.assertRaisesRegex(RuntimeError, "more than once"):
                    optimizer.backward(model(jt.array([1.0])).sum())
                self.assertEqual(calls["all_reduce"], [])

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
        for world_size in (1, 2):
            with self.subTest(world_size=world_size), \
                    _native_dist(world_size) as (_group, calls):
                model = DistributedDataParallel(_TwoWeights())
                optimizer = jt.optim.SGD([model.module.weight], lr=0.1)
                calls["all_reduce"].clear()
                with self.assertRaisesRegex(RuntimeError, "missing from the optimizer"):
                    optimizer.backward(model(jt.array([1.0])).sum())
                self.assertEqual(calls["all_reduce"], [])

    def test_removing_all_ddp_parameters_after_a_step_is_rejected(self):
        for world_size in (1, 2):
            for keep_unmanaged in (False, True):
                with self.subTest(world_size=world_size,
                                  keep_unmanaged=keep_unmanaged), \
                        _native_dist(world_size) as (_group, calls):
                    model = DistributedDataParallel(_OneWeight())
                    optimizer = jt.optim.SGD(model.parameters(), lr=0.1)
                    optimizer.step(model(jt.array([1.0])).sum())
                    unmanaged = jt.array([9.0], dtype="float32")
                    optimizer.param_groups[0]["params"] = (
                        [unmanaged] if keep_unmanaged else [])
                    calls["all_reduce"].clear()
                    loss = model(jt.array([1.0])).sum() + unmanaged.sum()
                    with self.assertRaisesRegex(RuntimeError, "missing from the optimizer"):
                        optimizer.backward(loss)
                    self.assertEqual(calls["all_reduce"], [])

    def test_structure_changes_after_a_valid_step_are_rejected_locally(self):
        def add_parameter(module):
            module.extra = jt.array([1.0], dtype="float32")

        def delete_parameter(module):
            del module.second

        def replace_parameter(module):
            # Same metadata, different holder: the old optimizer still owns
            # the original leaf and must not silently keep updating it.
            module.second = jt.array([4.0], dtype="float32")

        def change_trainability(module):
            module.second.requires_grad = False

        def add_buffer(module):
            module.register_buffer("extra", jt.array([1.0]).stop_grad())

        def delete_buffer(module):
            del module.running

        changes = (add_parameter, delete_parameter, replace_parameter,
                   change_trainability, add_buffer, delete_buffer)
        for world_size in (1, 2):
            for change in changes:
                for local_only in (False, True):
                    with self.subTest(world_size=world_size,
                                      change=change.__name__, no_sync=local_only), \
                            _native_dist(world_size) as (_group, calls):
                        model = DistributedDataParallel(_TwoWeights())
                        optimizer = jt.optim.SGD(model.parameters(), lr=0.1)
                        optimizer.step(model(jt.array([1.0])).sum())
                        change(model.module)
                        calls["all_reduce"].clear()
                        # Keep one original leaf in the loss so missing/deleted
                        # leaves do not turn this into a forward-error test.
                        loss = model.module.weight.sum()
                        with self.assertRaisesRegex(RuntimeError,
                                                    "changed after construction"):
                            if local_only:
                                with model.no_sync():
                                    optimizer.backward(loss)
                            else:
                                optimizer.backward(loss)
                        self.assertEqual(calls["all_reduce"], [])

    def test_stable_schema_has_only_constructor_signature_collective(self):
        with _native_dist(2) as (_group, calls):
            model = DistributedDataParallel(_OneWeight())
            optimizer = jt.optim.SGD(model.parameters(), lr=0.1)
            self.assertEqual(calls["all_reduce"], [((32,), "mean")])
            calls["all_reduce"].clear()
            for _ in range(3):
                optimizer.step(model(jt.array([3.0])).sum())
            self.assertEqual(calls["all_reduce"], [((1,), "mean")] * 3)
            self.assertAlmostEqual(float(model.module.weight.numpy()[0]), 1.1,
                                   places=5)

    def test_update_and_load_keep_parameter_holder_valid(self):
        for world_size in (1, 2):
            with self.subTest(world_size=world_size), \
                    _native_dist(world_size) as (_group, calls):
                raw = _OneWeight()
                holder = raw.weight
                model = DistributedDataParallel(raw)
                optimizer = jt.optim.SGD(model.parameters(), lr=0.1)
                optimizer.step(model(jt.array([3.0])).sum())
                calls["all_reduce"].clear()
                raw.weight.update(jt.array([8.0], dtype="float32"))
                self.assertIs(raw.weight, holder)
                optimizer.step(model(jt.array([3.0])).sum())
                self.assertAlmostEqual(float(holder.numpy()[0]), 7.7, places=5)
                raw.load_state_dict({"weight": jt.array([6.0], dtype="float32")})
                self.assertIs(raw.weight, holder)
                optimizer.step(model(jt.array([3.0])).sum())
                self.assertAlmostEqual(float(holder.numpy()[0]), 5.7, places=5)
                expected = [] if world_size == 1 else [((1,), "mean")] * 2
                self.assertEqual(calls["all_reduce"], expected)

    def test_optimizer_checkpoint_keeps_runtime_bindings_private_and_local(self):
        for world_size in (1, 2):
            with self.subTest(world_size=world_size), \
                    _native_dist(world_size) as (_group, calls):
                source = DistributedDataParallel(_OneWeight())
                source_optimizer = jt.optim.SGD(source.parameters(), lr=0.1,
                                                momentum=0.9)
                source_optimizer.step(source(jt.array([3.0])).sum())
                snapshot = source_optimizer.state_dict()
                self.assertNotIn("_jittor_ddp_states", snapshot["defaults"])
                serialized = pickle.dumps(snapshot)
                # Var serialization must not follow parameter attributes into
                # the reducer, module, process group, or another optimizer.
                self.assertNotIn(b"_jittor_ddp", serialized)
                parameter_value = source.module.weight.numpy().copy()

                restored = DistributedDataParallel(_OneWeight())
                restored.module.load_state_dict({"weight": parameter_value})
                restored_optimizer = jt.optim.SGD(restored.parameters(), lr=9.0)
                state = restored._jittor_ddp_state
                restored_optimizer.load_state_dict(pickle.loads(serialized))
                self.assertEqual(list(restored_optimizer._jittor_ddp_states),
                                 [state])
                self.assertNotIn(source._jittor_ddp_state,
                                 restored_optimizer._jittor_ddp_states)
                self.assertIs(restored_optimizer.param_groups[0]["params"][0],
                              restored.module.weight)
                calls["all_reduce"].clear()
                source_optimizer.step(source(jt.array([2.0])).sum())
                restored_optimizer.step(restored(jt.array([2.0])).sum())
                self.assertAlmostEqual(float(source.module.weight.numpy()[0]),
                                       1.23, places=5)
                self.assertEqual(restored.module.weight.numpy(),
                                 source.module.weight.numpy())
                expected = [] if world_size == 1 else [((1,), "mean")] * 2
                self.assertEqual(calls["all_reduce"], expected)

    def test_optimizer_created_before_ddp_discovers_and_retains_bindings(self):
        for world_size in (1, 2):
            for discover_with in ("backward", "zero_grad"):
                with self.subTest(world_size=world_size,
                                  discover_with=discover_with), \
                        _native_dist(world_size) as (_group, calls):
                    raw = _OneWeight()
                    optimizer = jt.optim.SGD(raw.parameters(), lr=0.1)
                    self.assertEqual(optimizer._jittor_ddp_states, {})
                    model = DistributedDataParallel(raw)
                    if discover_with == "backward":
                        optimizer.backward(model(jt.array([3.0])).sum())
                        self.assertEqual(
                            optimizer.param_groups[0]["grads"][0].numpy().tolist(),
                            [3.0])
                    else:
                        optimizer.zero_grad()
                    self.assertEqual(list(optimizer._jittor_ddp_states),
                                     [model._jittor_ddp_state])
                    optimizer.param_groups[0]["params"] = []
                    calls["all_reduce"].clear()
                    with self.assertRaisesRegex(RuntimeError,
                                                "missing from the optimizer"):
                        optimizer.backward(raw.weight.sum())
                    self.assertEqual(calls["all_reduce"], [])

    def test_local_device_change_is_rejected_before_any_collective(self):
        # Mock only Python metadata. Native storage and computation stay on
        # CPU, so this checks local plan invalidation without needing a GPU.
        device_id = jt.Var.device_id
        for world_size in (1, 2):
            for move in ("parameter", "buffer"):
                for local_only in (False, True):
                    moved = {}

                    def observed_device(value):
                        return moved.get(id(value), device_id.__get__(value))

                    with self.subTest(world_size=world_size, move=move,
                                      no_sync=local_only), \
                            patch.object(jt.Var, "device_id",
                                         property(observed_device)), \
                            _native_dist(world_size) as (_group, calls):
                        model = DistributedDataParallel(_OneWeight())
                        optimizer = jt.optim.SGD(model.parameters(), lr=0.1)
                        optimizer.step(model(jt.array([1.0])).sum())
                        value = (model.module.weight if move == "parameter"
                                 else model.module.running)
                        moved[id(value)] = int(value.device_id) + 1
                        calls["all_reduce"].clear()
                        loss = model.module.weight.sum()
                        with self.assertRaisesRegex(RuntimeError,
                                                    "changed after construction"):
                            if local_only:
                                with model.no_sync():
                                    optimizer.backward(loss)
                            else:
                                optimizer.backward(loss)
                        self.assertEqual(calls["all_reduce"], [])

    def test_added_ddp_group_remains_known_after_its_parameters_are_removed(self):
        algorithms = (jt.optim.SGD, jt.optim.Adam, jt.optim.AdamW,
                      jt.optim.RMSprop, jt.optim.Adan)
        for algorithm in algorithms:
            for world_size in (1, 2):
                for remove in ("clear_parameters", "remove_group"):
                    with self.subTest(algorithm=algorithm.__name__,
                                      world_size=world_size, remove=remove), \
                            _native_dist(world_size) as (_group, calls):
                        first = DistributedDataParallel(_OneWeight())
                        optimizer = algorithm(first.parameters(), lr=0.1)
                        second = DistributedDataParallel(_OneWeight())
                        group = {"params": second.parameters()}
                        optimizer.add_param_group(group)
                        self.assertEqual(
                            set(optimizer._jittor_ddp_states),
                            {first._jittor_ddp_state, second._jittor_ddp_state})
                        if remove == "clear_parameters":
                            group["params"] = []
                        else:
                            self.assertIs(optimizer.param_groups.pop(), group)
                        calls["all_reduce"].clear()
                        loss = first.module.weight.sum() + second.module.weight.sum()
                        # Isolate coverage from communication: no_sync must
                        # still reject removing the complete newly added model.
                        with first.no_sync(), second.no_sync(), \
                                self.assertRaisesRegex(RuntimeError,
                                                       "missing from the optimizer"):
                            optimizer.backward(loss)
                        self.assertEqual(calls["all_reduce"], [])

    def test_nccl_bucket_syncs_sum_outputs_then_scales_after_exit(self):
        for bucket_inputs in (1, 64):
            with self.subTest(bucket_inputs=bucket_inputs):
                self._check_nccl_bucket(bucket_inputs)

    def _check_nccl_bucket(self, bucket_inputs):
        from jittor.distributed import bucket
        from jittor.backends.cuda.kernels.misc import ddp_bucket
        from jittor.nn.parallel import distributed_data_parallel as reducer

        active = [False]
        events = []
        raw_outputs = []
        remote_gradients = iter((5.0, 10.0))
        sync = jt.sync
        divide = jt.Var.__truediv__

        @contextmanager
        def scope(defer_join=True):
            self.assertFalse(defer_join)
            self.assertFalse(active[0])
            active[0] = True
            events.append("enter")
            try:
                yield
            finally:
                active[0] = False
                events.append("exit")

        def sum_with_remote(tensor, op):
            if tuple(tensor.shape) == (32,):
                self.assertEqual(op, "mean")
                self.assertFalse(active[0])
                return tensor
            self.assertTrue(active[0])
            self.assertEqual(op, "sum")
            remote = jt.array([next(remote_gradients) for _ in
                               range(int(tensor.numel()))], dtype=tensor.dtype)
            output = tensor + remote.reshape(tensor.shape)
            raw_outputs.append(output)
            events.append("sum")
            return output

        def sync_sinks(values, *args, **kwargs):
            if active[0]:
                self.assertEqual([id(value) for value in values],
                                 [id(value) for value in raw_outputs])
                events.append("sync_raw_outputs")
            return sync(values, *args, **kwargs)

        def scale_after_exit(value, divisor):
            if any(value is output for output in raw_outputs):
                self.assertFalse(active[0], "mean consumed an open group's output")
                self.assertEqual(divisor, 2)
                events.append("mean")
            return divide(value, divisor)

        def pack_on_cpu(gradients):
            self.assertFalse(active[0])
            events.append("pack")
            return jt.concat([gradient.reshape((-1,)) for gradient in gradients])

        def unpack_on_cpu(reduced, gradients, world_size):
            self.assertFalse(active[0])
            averaged = reduced / world_size
            events.append("unpack")
            result = []
            offset = 0
            for gradient in gradients:
                count = int(gradient.numel())
                result.append(averaged[offset:offset + count].reshape(gradient.shape))
                offset += count
            self.assertEqual(offset, int(reduced.numel()))
            return result

        with _native_dist(2, all_reduce=sum_with_remote,
                          backend_name="nccl") as (group, calls), \
                patch.object(reducer, "_BUCKET_INPUTS", bucket_inputs):
            # WORLD has no subgroup kind: backend discovery must use the query.
            self.assertIsNone(group._backend_kind)
            model = DistributedDataParallel(_TwoWeights())
            optimizer = jt.optim.SGD(model.parameters(), lr=0.1)
            calls["all_reduce"].clear()
            with patch.object(bucket, "bucket_scope", side_effect=scope), \
                    patch.object(ddp_bucket, "pack_gradients", side_effect=pack_on_cpu), \
                    patch.object(ddp_bucket, "unpack_gradients", side_effect=unpack_on_cpu), \
                    patch.object(jt, "sync", side_effect=sync_sinks), \
                    patch.object(jt.Var, "__truediv__", scale_after_exit):
                loss = (model.module.weight * 3.0
                        + model.module.second * 6.0).sum()
                optimizer.backward(loss)
            if bucket_inputs == 1:
                self.assertEqual(events, ["enter", "sum", "sum", "sync_raw_outputs",
                                          "exit", "mean", "mean"])
                self.assertEqual(calls["all_reduce"], [((1,), "sum")] * 2)
            else:
                self.assertEqual(events, ["pack", "enter", "sum", "sync_raw_outputs",
                                          "exit", "mean", "unpack"])
                self.assertEqual(calls["all_reduce"], [((2,), "sum")])
            self.assertEqual([gradient.numpy().tolist() for gradient in
                              optimizer.param_groups[0]["grads"]], [[4.0], [8.0]])
            optimizer.step()
            self.assertAlmostEqual(float(model.module.weight.numpy()[0]), 1.6,
                                   places=5)
            self.assertAlmostEqual(float(model.module.second.numpy()[0]), 3.2,
                                   places=5)


if __name__ == "__main__":
    unittest.main()
