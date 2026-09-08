"""Rank-matrix and native communicator contracts independent of device JIT."""

from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np


class TestDeviceMeshGroups(unittest.TestCase):
    def setUp(self):
        self.calls = []
        runtime = types.ModuleType("jittor")
        runtime.rank, runtime.world_size, runtime.in_mpi = 2, 4, True
        runtime.flags = types.SimpleNamespace(use_cuda=1)
        runtime.has_cuda = True
        def create(ranks):
            self.calls.append(tuple(ranks))
            return len(self.calls)
        runtime.compile_extern = types.SimpleNamespace(
            nccl=types.SimpleNamespace(nccl_create_process_group=create),
            nccl_ops=types.SimpleNamespace())
        common = types.ModuleType("mesh_test.common")
        common._rank = lambda: runtime.rank
        common._world_size = lambda: runtime.world_size
        common._prod = lambda values: int(np.prod(tuple(values)))
        modules = {
            "jittor": runtime,
            "jittor_utils": types.SimpleNamespace(lock=types.SimpleNamespace(unlock_scope=nullcontext)),
            "mesh_test": types.ModuleType("mesh_test"),
            "mesh_test.fsdp2": types.ModuleType("mesh_test.fsdp2"),
            "mesh_test.fsdp2.common": common,
            "mesh_test.diagnostics": types.SimpleNamespace(EXPECTED=(ValueError,), swallowed=None),
        }
        self.patch = patch.dict(sys.modules, modules)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        root = Path(__file__).resolve().parents[2]
        def load(name, path):
            spec = importlib.util.spec_from_file_location(name, root / path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            self.addCleanup(sys.modules.pop, name, None)
            return module
        load("jittor.distributed.process_group", "python/jittor/distributed/process_group.py")
        self.api = load("mesh_test.fsdp2.dtensor", "compat/fsdp2/dtensor.py")
        self.runtime = runtime

    def test_rank_matrix_axes_create_distinct_real_groups(self):
        mesh = self.api.DeviceMesh("cuda", [[0, 1], [2, 3]], mesh_dim_names=("dp", "tp"))
        self.assertEqual(mesh.shape, (2, 2))
        self.assertEqual(mesh.get_coordinate(), [1, 0])
        self.assertEqual(self.calls, [(0, 2), (1, 3), (0, 1), (2, 3)])
        dp, tp = mesh["dp"], mesh["tp"]
        self.assertEqual(dp.mesh.tolist(), [0, 2])
        self.assertEqual(tp.mesh.tolist(), [2, 3])
        self.assertIs(dp.get_group(), mesh.get_group("dp"))
        self.assertEqual(dp.get_local_rank(), 1)
        self.assertEqual(tp.get_local_rank(), 0)
        self.assertNotEqual(dp.get_group()._backend_handle, tp.get_group()._backend_handle)
        self.assertEqual(len(self.calls), 4)

    def test_constructor_ranks_and_factory_shape_are_distinct(self):
        mesh = self.api.DeviceMesh("cuda", [2, 0])
        self.assertEqual(mesh.size(), 2)
        self.assertEqual(mesh.get_local_rank(), 0)
        shaped = self.api.init_device_mesh("cuda", (2, 2), mesh_dim_names=("x", "y"))
        np.testing.assert_array_equal(shaped.mesh, [[0, 1], [2, 3]])
        self.assertEqual(shaped["y", "x"]._flatten().mesh.tolist(), [0, 2, 1, 3])

    def test_invalid_ranks_names_and_nonmember_fail(self):
        for ranks in ([0, 0], [4], [-1], []):
            with self.assertRaises(ValueError):
                self.api.DeviceMesh("cuda", ranks)
        with self.assertRaises(ValueError):
            self.api.DeviceMesh("cuda", [[0, 1], [2, 3]], mesh_dim_names=("x", "x"))
        mesh = self.api.DeviceMesh("cuda", [0, 1])
        self.assertIsNone(mesh.get_coordinate())
        with self.assertRaisesRegex(RuntimeError, "not a member"):
            mesh.get_group()


if __name__ == "__main__":
    unittest.main()
