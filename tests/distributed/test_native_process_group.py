"""Host contracts for native group ownership, without importing a JIT runtime."""

import importlib.util
import os
from pathlib import Path
import types
import unittest
from unittest.mock import patch


class TestNativeProcessGroup(unittest.TestCase):
    def setUp(self):
        self.runtime = types.ModuleType("jittor")
        self.runtime.rank = 2
        self.runtime.world_size = 4
        self.runtime.in_mpi = True
        self.runtime.flags = types.SimpleNamespace(use_cuda=0)
        self.runtime.compile_extern = types.SimpleNamespace()
        path = (Path(__file__).resolve().parents[2] / "python/jittor"
                / "distributed/process_group.py")
        spec = importlib.util.spec_from_file_location("native_group_test", path)
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {"jittor": self.runtime}):
            spec.loader.exec_module(self.module)
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_world_queries_live_native_runtime(self):
        group = self.module.ProcessGroup(name="world")
        self.assertEqual((group.rank(), group.size()), (2, 4))
        self.runtime.rank = 1
        self.runtime.world_size = 3
        self.assertEqual((group.rank(), group.size()), (1, 3))
        self.runtime.in_mpi = False
        self.assertEqual((group.rank(), group.size()), (0, 1))

    def test_subgroup_membership_and_singleton(self):
        group = self.module.ProcessGroup([0, 2], "pair")
        self.assertEqual((group.rank(), group.size()), (1, 2))
        self.runtime.rank = 3
        tensor = object()
        self.assertEqual(group.rank(), -1)
        self.assertIs(group._all_reduce(tensor, "sum"), tensor)
        singleton = self.module.ProcessGroup([3])
        singleton._create_backend_communicator()
        self.assertIs(singleton._all_reduce(tensor, "sum"), tensor)

    def test_missing_communicator_is_not_silent(self):
        group = self.module.ProcessGroup([0, 2])
        with self.assertRaisesRegex(NotImplementedError, "require NCCL or HCCL"):
            group._create_backend_communicator()
        with self.assertRaisesRegex(RuntimeError, "no backend communicator"):
            group._all_reduce(4, "sum")

    def test_world_delegates_native_collective(self):
        calls = []
        tensor = types.SimpleNamespace(
            mpi_all_reduce=lambda op: calls.append(op) or 12)
        self.assertEqual(self.module.ProcessGroup()._all_reduce(tensor, "sum"), 12)
        self.assertEqual(calls, ["sum"])

    def test_backend_handle_and_collectives(self):
        from contextlib import nullcontext

        for kind in ("nccl", "hccl"):
            with self.subTest(kind=kind):
                calls = []
                module = types.SimpleNamespace(**{
                    kind + "_create_process_group":
                        lambda ranks: calls.append(tuple(ranks)) or 7})
                ops = types.SimpleNamespace(**{
                    kind + "_all_reduce": lambda *args: calls.append(args) or 12})
                self.runtime.compile_extern = types.SimpleNamespace(**{
                    "nccl" if kind == "nccl" else "hccl_mod": module,
                    kind + "_ops": ops})
                lock = types.SimpleNamespace(unlock_scope=nullcontext)
                fake_utils = types.ModuleType("jittor_utils")
                fake_utils.lock = lock
                group = self.module.ProcessGroup([0, 2])
                with patch.dict("sys.modules", {"jittor_utils": fake_utils}):
                    group._create_backend_communicator()
                self.assertEqual(group._get_backend_name(), kind)
                self.assertEqual(group._backend_handle, 7)
                self.assertEqual(group._all_reduce(6, "mean"), 6)
                self.assertEqual(calls[0], (0, 2))
                self.assertEqual(calls[1], (6, 7) if kind == "nccl" else (6, "sum", 7))
                if kind == "nccl":
                    with self.assertRaisesRegex(NotImplementedError, "sum and mean"):
                        group._all_reduce(6, "max")

    def test_work_retains_completion_value(self):
        value = object()
        work = self.module.Work(value)
        self.assertTrue(work.is_completed())
        self.assertIs(work.wait(), value)

    def test_public_state_queries_include_legacy_mpi_runtime(self):
        self.assertTrue(self.module.is_initialized())
        self.assertEqual(self.module.get_rank(), 2)
        self.assertEqual(self.module.get_world_size(), 4)
        with patch.dict(os.environ, {
            "LOCAL_RANK": "1", "LOCAL_WORLD_SIZE": "2",
        }):
            self.assertEqual(self.module.get_local_rank(), 1)
            self.assertEqual(self.module.get_local_world_size(), 2)
        self.assertIs(self.module.get_default_group(),
                      self.module.get_default_group())
        self.assertEqual(self.module.get_process_group_ranks(), [0, 1, 2, 3])

    def test_native_init_destroy_owns_store_and_backend_state(self):
        self.runtime.rank = 0
        self.runtime.world_size = 1
        self.runtime.in_mpi = False
        self.runtime.current_device = lambda: 3

        class Store:
            closed = 0

            def close(self):
                self.closed += 1

        store = Store()
        self.module.init_process_group(
            backend="mpi", rank=0, world_size=1, store=store)
        self.assertTrue(self.module.is_initialized())
        self.assertEqual(self.module.get_backend(), "mpi")
        self.assertEqual(self.module.get_device(), 3)
        self.assertIs(self.module.get_default_store(), store)
        with self.assertRaisesRegex(RuntimeError, "already initialized"):
            self.module.init_process_group(backend="mpi", rank=0, world_size=1)
        self.module.destroy_process_group()
        self.assertFalse(self.module.is_initialized())
        self.assertEqual(store.closed, 1)
        self.assertIsNone(self.module.get_default_store())

    def test_backend_query_fails_closed_before_process_group_init(self):
        self.runtime.rank = 0
        self.runtime.world_size = 1
        self.runtime.in_mpi = False
        self.runtime.compile_extern.nccl_ops = object()
        with self.assertRaisesRegex(RuntimeError, "not initialized"):
            self.module.get_backend()

    def test_destroy_unregisters_state_without_claiming_backend_teardown(self):
        shutdown_calls = []
        self.runtime.compile_extern = types.SimpleNamespace(
            nccl_ops=types.SimpleNamespace(),
            nccl=types.SimpleNamespace(
                nccl_shutdown=lambda: shutdown_calls.append("shutdown")))
        with patch.dict(os.environ, {
            "JT_NCCL_WORLD_SIZE": "4",
            "JT_NCCL_RANK": "2",
            "JT_NCCL_LOCAL_RANK": "0",
        }):
            self.module.init_process_group(
                backend="nccl", rank=2, world_size=4)
            group = self.module.get_default_group()
            self.module.destroy_process_group()
        self.assertFalse(self.module.is_initialized())
        self.assertEqual(shutdown_calls, [])
        with self.assertRaisesRegex(RuntimeError, "worker until process exit"):
            group.all_reduce(object(), "sum")

    def test_public_collectives_return_autograd_values_and_work(self):
        calls = []

        class Tensor:
            shape = (1,)

            def mpi_all_reduce(self, op):
                calls.append(("all_reduce", op))
                return "reduced"

            def mpi_broadcast(self, root):
                calls.append(("broadcast", root))
                return "broadcasted"

            def mpi_all_gather(self):
                calls.append(("all_gather",))
                return "gathered"

        tensor = Tensor()
        self.assertEqual(self.module.all_reduce(tensor, "sum"), "reduced")
        self.assertEqual(self.module.broadcast(tensor, src=1), "broadcasted")
        self.assertEqual(self.module.all_gather(tensor), "gathered")
        work = self.module.all_reduce(tensor, "mean", async_op=True)
        self.assertIsInstance(work, self.module.Work)
        self.assertEqual(work.wait(), "reduced")
        self.assertEqual(calls, [
            ("all_reduce", "sum"),
            ("broadcast", 1),
            ("all_gather",),
            ("all_reduce", "mean"),
        ])

    def test_public_group_collectives_route_native_backend_handle(self):
        from contextlib import nullcontext

        calls = []
        module = types.SimpleNamespace(
            nccl_create_process_group=lambda ranks: calls.append(
                ("create", tuple(ranks))) or 9)
        ops = types.SimpleNamespace(
            nccl_all_reduce=lambda *args: calls.append(("all_reduce", args)) or 12,
            nccl_broadcast=lambda *args: calls.append(("broadcast", args)) or 13,
            nccl_all_gather=lambda *args: calls.append(("all_gather", args)) or 14,
            nccl_reduce_scatter=lambda *args: calls.append(("reduce_scatter", args)) or 15,
        )
        self.runtime.compile_extern = types.SimpleNamespace(
            nccl=module, nccl_ops=ops)
        lock = types.SimpleNamespace(unlock_scope=nullcontext)
        fake_utils = types.ModuleType("jittor_utils")
        fake_utils.lock = lock
        with patch.dict("sys.modules", {"jittor_utils": fake_utils}):
            group = self.module.new_group([0, 2])
        tensor = types.SimpleNamespace(shape=(4, 2))
        self.assertEqual(group.all_reduce(tensor, "sum"), 12)
        self.assertEqual(group.broadcast(tensor, src=2), 13)
        self.assertEqual(group.all_gather(tensor), 14)
        self.assertEqual(group.reduce_scatter(tensor), 15)
        self.assertEqual(calls[0], ("create", (0, 2)))
        self.assertEqual(calls[1][1][1], 9)


if __name__ == "__main__":
    unittest.main()
