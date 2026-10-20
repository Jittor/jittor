"""Native DDP and Jittor optimizer lifecycle contracts."""

from contextlib import contextmanager
import pickle
import unittest
from unittest.mock import patch

import jittor as jt
import numpy as np
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
            patch.object(distributed, "get_backend",
                         return_value=backend_name or "mpi", create=True), \
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


class _ExpandedWeight(nn.Module):
    def __init__(self):
        self.weight = jt.array([[2.0], [4.0]], dtype="float32").expand((2, 2))

    def execute(self, x):
        return jt.matmul(x, self.weight)


class _ExpandedBuffer(nn.Module):
    def __init__(self):
        self.weight = jt.array([2.0], dtype="float32")
        self.register_buffer(
            "expanded", jt.array([[3.0]], dtype="float32").expand((1, 2)))

    def execute(self, x):
        return self.weight * x


class TestNativeDistributedDataParallel(unittest.TestCase):
    def test_clip_norm_broadcasts_coefficient_for_synced_nccl_gradients(self):
        with _native_dist(
                2, broadcast=lambda tensor, _src: tensor * 0 + 0.1,
                backend_name="nccl") as (_group, calls):
            wrapped = DistributedDataParallel(_TwoWeights())
            optimizer = jt.optim.SGD(wrapped.parameters(), lr=0.1)
            state = wrapped._jittor_ddp_state
            state.has_unsynced_grads = False
            optimizer.param_groups[0]["grads"] = [
                jt.array([3.0], dtype="float32").stop_grad(),
                jt.array([4.0], dtype="float32").stop_grad(),
            ]
            optimizer._Optimizer__zero_grad = False
            calls["broadcast"].clear()

            optimizer.clip_grad_norm(1.0)
            observed = [
                gradient.numpy().tolist()
                for gradient in optimizer.param_groups[0]["grads"]
            ]

        self.assertAlmostEqual(observed[0][0], 0.3, places=6)
        self.assertAlmostEqual(observed[1][0], 0.4, places=6)
        self.assertEqual(calls["broadcast"], [((), 0)])

    def test_optimizer_readiness_preflight_maps_targets_to_module_slots(self):
        from jittor.optim.base import _native_ddp_readiness_plan

        with _native_dist(2, backend_name="nccl") as (_group, _calls):
            wrapped = DistributedDataParallel(_TwoWeights())
            optimizer = jt.optim.SGD(wrapped.parameters(), lr=0.1)
            targets = [
                parameter for group in optimizer.param_groups
                for parameter in group["params"] if parameter.requires_grad
            ]
            plan = _native_ddp_readiness_plan(optimizer, targets, [])

        self.assertIsNotNone(plan)
        self.assertEqual(
            plan["target_to_parameter"],
            tuple(range(len(targets))),
        )
        self.assertEqual(
            set(plan["parameter_slots"]),
            set(plan["target_to_parameter"]),
        )

    def test_readiness_signature_ignores_local_optimizer_parameter_order(self):
        import jittor.nn.parallel.distributed_data_parallel as reducer
        from jittor.optim.base import _native_ddp_readiness_plan

        with _native_dist(2, backend_name="nccl"):
            wrapped = DistributedDataParallel(_TwoWeights())
            parameters = list(wrapped.parameters())
            signatures = []
            plans = []

            def capture_signature(_distributed, _group, payload,
                                  _description, _reference):
                signatures.append(payload)

            with patch.object(reducer, "_signature_check",
                              side_effect=capture_signature):
                for ordered_parameters in (
                        parameters, list(reversed(parameters))):
                    optimizer = jt.optim.SGD(ordered_parameters, lr=0.1)
                    targets = [
                        parameter for group in optimizer.param_groups
                        for parameter in group["params"]
                        if parameter.requires_grad
                    ]
                    plans.append(_native_ddp_readiness_plan(
                        optimizer, targets, []))

        self.assertEqual(signatures[0], signatures[1])
        self.assertEqual(plans[0]["target_to_parameter"], (0, 1))
        self.assertEqual(plans[1]["target_to_parameter"], (1, 0))
        for plan in plans:
            for parameter_index, (group, slot) in plan["parameter_slots"].items():
                self.assertIs(group["params"][slot], parameters[parameter_index])

    def test_gradient_readiness_drains_out_of_order_targets_in_bucket_order(self):
        from jittor.nn.parallel.distributed_data_parallel import (
            _GradientBucketReadiness,
        )

        launched = []
        readiness = _GradientBucketReadiness(
            buckets=((0, 1), (2,)),
            target_to_parameter=(2, 0, 1),
            on_bucket_ready=lambda indices, values: launched.append(
                (indices, values)),
        )
        readiness(0, "gradient-2")
        readiness(2, "gradient-1")
        self.assertEqual(launched, [])
        readiness(1, "gradient-0")
        readiness.finish()
        self.assertEqual(
            launched,
            [
                ((0, 1), ("gradient-0", "gradient-1")),
                ((2,), ("gradient-2",)),
            ],
        )

    def test_gradient_readiness_rejects_duplicate_and_missing_targets(self):
        from jittor.nn.parallel.distributed_data_parallel import (
            _GradientBucketReadiness,
        )

        readiness = _GradientBucketReadiness(
            buckets=((0, 1),),
            target_to_parameter=(0, 1),
            on_bucket_ready=lambda _indices, _values: None,
        )
        readiness(0, "first")
        with self.assertRaisesRegex(RuntimeError, "reported more than once"):
            readiness(0, "duplicate")
        with self.assertRaisesRegex(RuntimeError, "missing target indices"):
            readiness.finish()

    def test_constructor_rejects_overlapping_storage_before_collectives(self):
        cases = (
            ("parameter", _ExpandedWeight,
             lambda module: module.weight),
            ("buffer", _ExpandedBuffer,
             lambda module: module.expanded),
        )
        for kind, factory, get_view in cases:
            for world_size in (1, 2):
                with self.subTest(kind=kind, world_size=world_size), \
                        _native_dist(world_size) as (_group, calls):
                    module = factory()
                    value = get_view(module)
                    # Some CPU Var implementations materialize expand() before
                    # reporting storage layout. Force the contract predicate in
                    # this process-level test; CUDA evidence exercises the real
                    # stride-zero view directly.
                    with patch.object(value, "_storage_is_contiguous",
                                      return_value=False):
                        with self.assertRaisesRegex(
                                ValueError,
                                "requires contiguous parameter and buffer storage"):
                            DistributedDataParallel(module)
                    self.assertEqual(calls["all_reduce"], [])
                    self.assertEqual(calls["broadcast"], [])

    def test_constructor_signature_check_rejects_corrupted_collective_result(self):
        from jittor.nn.parallel.distributed_data_parallel import _signature_check

        class _CorruptedCollective:
            @staticmethod
            def all_reduce(tensor, op="sum", group=None):
                return jt.zeros(tensor.shape, dtype=tensor.dtype)

        reference = jt.array([1.0], dtype="float32").stop_grad()
        with self.assertRaisesRegex(RuntimeError, "could not verify the reduced"):
            _signature_check(
                _CorruptedCollective(), _Group(),
                (("weight", (1,), "float32", "cpu", True),),
                "module parameter/buffer metadata", reference,
            )

    def test_signature_check_avoids_half_precision_collective_overflow(self):
        from jittor.nn.parallel.distributed_data_parallel import _signature_check

        class _HalfPrecisionCollective:
            @staticmethod
            def all_reduce(tensor, op="sum", group=None):
                values = np.asarray(tensor.numpy(), dtype=np.float16)
                reduced = (values * np.float16(2)).astype(np.float16)
                if op == "mean":
                    reduced = (reduced / np.float16(2)).astype(np.float16)
                return jt.array(reduced.astype(np.float32)).stop_grad()

        reference = jt.array([1.0], dtype="float32").stop_grad()
        _signature_check(
            _HalfPrecisionCollective(), _Group(size=2),
            (("weight", (1,), "float32", "cpu", True),),
            "module parameter/buffer metadata", reference,
        )

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

    def test_constructor_flushes_pending_work_before_signature_collective(self):
        events = []

        def reduce(tensor, op="sum", group=None):
            events.append("all_reduce")
            return tensor

        with _native_dist(2, all_reduce=reduce) as (_group, _calls), \
                patch.object(jt, "sync_all",
                             side_effect=lambda *args, **kwargs: events.append(
                                 "sync_all")):
            DistributedDataParallel(_OneWeight())

        self.assertGreaterEqual(len(events), 2)
        self.assertEqual(events[0:2], ["sync_all", "all_reduce"])

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
            self.assertEqual(calls["all_reduce"], [((16,), "mean")])
            calls["all_reduce"].clear()
            for _ in range(3):
                optimizer.step(model(jt.array([3.0])).sum())
            self.assertEqual(calls["all_reduce"], [((1,), "mean")] * 3)
            self.assertAlmostEqual(float(model.module.weight.numpy()[0]), 1.1,
                                   places=5)

    def test_one_rank_launcher_does_not_use_legacy_mpi_collectives(self):
        # jtrun sets jt.in_mpi for its MPI-free NCCL rendezvous path. A raw
        # single-card job need not initialize a process group and must not call
        # the legacy Var.mpi_all_reduce / Var.mpi_broadcast methods.
        with _native_dist(1):
            weight = jt.array([2.0], dtype="float32")
            optimizer = jt.optim.SGD([weight], lr=0.1)
            with patch.object(jt.compile_extern, "in_mpi", True):
                optimizer.backward((weight * 3.0).sum())
                optimizer.step()

        np.testing.assert_allclose(weight.numpy(), [1.7], rtol=0, atol=1e-6)

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
        pending = [False]
        constructing = [False]
        events = []
        raw_outputs = []
        prepared_inputs = []
        remote_gradients = iter((10.0, 5.0) if bucket_inputs == 1
                                else (5.0, 10.0))
        sync = jt.sync
        grad_with_callback = jt.core.grad_with_callback
        divide = jt.Var.__truediv__

        def build_backward(*args):
            constructing[0] = True
            try:
                return grad_with_callback(*args)
            finally:
                constructing[0] = False

        @contextmanager
        def scope(defer_join=True):
            self.assertTrue(defer_join)
            self.assertFalse(active[0])
            active[0] = True
            events.append("enter")
            try:
                yield
            finally:
                active[0] = False
                events.append("exit")
                pending[0] = True

        def wait_for_communication():
            self.assertFalse(active[0])
            if not pending[0]:
                events.append("wait_empty")
                return False
            pending[0] = False
            events.append("wait")
            return True

        def sum_with_remote(tensor, op):
            if tuple(tensor.shape) == (16,):
                self.assertEqual(op, "mean")
                self.assertFalse(active[0])
                return tensor
            self.assertTrue(active[0])
            self.assertEqual(op, "sum")
            self.assertTrue(any(tensor is value for value in prepared_inputs),
                            "collective input was not prepared before group entry")
            remote = jt.array([next(remote_gradients) for _ in
                               range(int(tensor.numel()))], dtype=tensor.dtype)
            output = tensor + remote.reshape(tensor.shape)
            raw_outputs.append(output)
            events.append("sum")
            return output

        def sync_sinks(values, *args, **kwargs):
            if active[0]:
                self.assertEqual(len(values), 1)
                self.assertIs(values[0], raw_outputs[-1])
                events.append("sync_raw_outputs")
            else:
                # Readiness reduces temporary gradients and publishes the
                # optimizer holders only after the communication join. Check
                # preparation of the actual collective inputs, independently
                # of those holders' identities.
                parameters = model.parameters()
                if (values and all(tuple(value.shape) == (1,) for value in values)
                        and not any(value is parameter for value in values
                                    for parameter in parameters)):
                    self.assertFalse(
                        constructing[0],
                        "backward executed while autograd still owns intermediate roots")
                    self.assertEqual(jt.flags.auto_flush_ops, 0)
                    self.assertFalse(kwargs.get("weak_sync", True))
                    prepared_inputs.extend(values)
                    events.append("gradient_ready")
            return sync(values, *args, **kwargs)

        def scale_after_exit(value, divisor):
            if any(value is output for output in raw_outputs):
                self.assertFalse(active[0], "mean consumed an open group's output")
                self.assertFalse(pending[0], "mean consumed a pending collective")
                self.assertEqual(divisor, 2)
                events.append("mean")
            return divide(value, divisor)

        def pack_on_cpu(gradients):
            self.assertFalse(active[0])
            self.assertTrue(all(any(gradient is value for value in prepared_inputs)
                                for gradient in gradients))
            events.append("pack")
            packed = jt.concat([gradient.reshape((-1,)) for gradient in gradients])
            prepared_inputs.append(packed)
            return packed

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
                    patch.object(bucket, "comm_wait",
                                 side_effect=wait_for_communication), \
                    patch.object(ddp_bucket, "pack_gradients", side_effect=pack_on_cpu), \
                    patch.object(ddp_bucket, "unpack_gradients", side_effect=unpack_on_cpu), \
                    patch.object(jt.core, "grad_with_callback", side_effect=build_backward), \
                    patch.object(jt, "sync", side_effect=sync_sinks), \
                    patch.object(jt.Var, "__truediv__", scale_after_exit):
                loss = (model.module.weight * 3.0
                        + model.module.second * 6.0).sum()
                with jt.flag_scope(auto_flush_ops=37):
                    optimizer.backward(loss, retain_graph=True)
                    self.assertEqual(jt.flags.auto_flush_ops, 37)
            # Retaining a graph must still permit an independent gradient
            # request after readiness has consumed its temporary references.
            retained = jt.grad(loss, model.parameters())
            self.assertEqual([g.numpy().tolist() for g in retained], [[3.0], [6.0]])
            if bucket_inputs == 1:
                self.assertEqual(events, [
                    "gradient_ready", "enter", "sum", "sync_raw_outputs", "exit",
                    "gradient_ready", "enter", "sum", "sync_raw_outputs", "exit",
                    "wait", "mean", "mean",
                ])
                self.assertEqual(calls["all_reduce"],
                                 [((16,), "mean"), ((1,), "sum"), ((1,), "sum")])
            else:
                self.assertEqual(events, [
                    "gradient_ready", "pack", "enter", "sum", "sync_raw_outputs",
                    "exit", "wait", "mean", "unpack",
                ])
                self.assertEqual(calls["all_reduce"],
                                 [((16,), "mean"), ((2,), "sum")])
            self.assertFalse(pending[0])
            self.assertEqual([gradient.numpy().tolist() for gradient in
                              optimizer.param_groups[0]["grads"]], [[4.0], [8.0]])
            optimizer.step()
            self.assertAlmostEqual(float(model.module.weight.numpy()[0]), 1.6,
                                   places=5)
            self.assertAlmostEqual(float(model.module.second.numpy()[0]), 3.2,
                                   places=5)

    def test_nccl_readiness_replay_failure_joins_and_restores_auto_flush(self):
        from jittor.distributed import bucket
        from jittor.nn.parallel import distributed_data_parallel as reducer

        collectives = []

        def fail_second_bucket(tensor, op):
            if tuple(tensor.shape) == (16,):
                return tensor
            self.assertEqual(jt.flags.auto_flush_ops, 0)
            collectives.append(tensor)
            if len(collectives) == 2:
                raise RuntimeError("injected second bucket failure")
            return tensor * 2

        with _native_dist(2, all_reduce=fail_second_bucket,
                          backend_name="nccl"), \
                patch.object(reducer, "_BUCKET_INPUTS", 1):
            model = DistributedDataParallel(_TwoWeights())
            optimizer = jt.optim.SGD(model.parameters(), lr=0.1)
            loss = (model.module.weight * 3.0 + model.module.second * 6.0).sum()
            with patch.object(bucket, "bucket_scope"), \
                    patch.object(bucket, "comm_wait") as wait, \
                    jt.flag_scope(auto_flush_ops=37):
                with self.assertRaisesRegex(RuntimeError, "injected second bucket failure"):
                    optimizer.backward(loss)
                self.assertEqual(jt.flags.auto_flush_ops, 37)
                wait.assert_called_once_with()
            # A failed replay must not publish partially reduced gradients.
            self.assertEqual([g.numpy().tolist() for g in
                              optimizer.param_groups[0]["grads"]], [[0.0], [0.0]])

    def test_nccl_unpack_rejects_checkpoint_gradient_dtype_mismatch(self):
        from jittor.backends.cuda.kernels.misc.ddp_bucket import unpack_gradients

        # Old BF16 checkpoints could restore a gradient's raw bits as uint16.
        # Copying the floating bucket bytes into that holder silently changes
        # their interpretation; reject it before creating a memcpy operator.
        reduced = jt.array([2.0, -4.0], dtype="float32")
        templates = [jt.zeros((1,), dtype="uint16"),
                     jt.zeros((1,), dtype="uint16")]
        with patch.object(jt, "code") as code, \
                self.assertRaisesRegex(ValueError, "bucket dtype"):
            unpack_gradients(reduced, templates, 2)
        code.assert_not_called()


if __name__ == "__main__":
    unittest.main()
