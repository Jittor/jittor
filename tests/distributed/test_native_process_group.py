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


if __name__ == "__main__":
    unittest.main()
