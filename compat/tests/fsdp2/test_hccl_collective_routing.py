"""Host-only routing regressions; importing this file never loads Jittor's JIT.

Direct execution is supported before accelerator prewarming. Fake device
operations exercise the real collective and ProcessGroup source owners; this
is not evidence of HCCL numerical correctness on a card.
"""

import ast
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[3]


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.shape = self.values.shape

    def __truediv__(self, value):
        return Tensor(self.values / value)

    def __mul__(self, other):
        return Tensor(self.values * other.values)

    def sync(self, device_sync=False):
        self.last_device_sync = device_sync

    def __getitem__(self, value):
        return Tensor(self.values[value])

    def assign(self, other):
        self.values = other.values.copy()
        self.shape = self.values.shape

    def reshape(self, shape):
        return Tensor(self.values.reshape(shape))


class TestHcclWorldRouting(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.runtime = types.ModuleType("jittor")
        self.runtime.rank, self.runtime.world_size = 0, 2
        self.runtime.in_mpi = True
        self.runtime.flags = types.SimpleNamespace(use_cuda=1)
        # ACL also exposes a CUDA-compatible backend; ACL must take precedence.
        self.runtime.has_cuda = True
        self.runtime.compiler = types.SimpleNamespace(has_acl=True, has_cuda=True)
        ops = types.SimpleNamespace(
            hccl_all_gather=self.gather,
            hccl_all_reduce=self.reduce,
            hccl_broadcast=self.broadcast,
        )
        self.runtime.compile_extern = types.SimpleNamespace(
            hccl_ops=ops, nccl_ops=None, has_mpi=False)
        self.modules = {
            "jittor": self.runtime,
            "jittor.distributed": types.ModuleType("jittor.distributed"),
            "jittor_utils": types.SimpleNamespace(
                lock=types.SimpleNamespace(unlock_scope=nullcontext)),
            "routing_test": types.ModuleType("routing_test"),
            "routing_test.diagnostics": types.SimpleNamespace(
                EXPECTED=(ValueError, RuntimeError), swallowed=lambda *a, **k: None),
            "routing_test.fsdp2": types.ModuleType("routing_test.fsdp2"),
        }
        self.context = patch.dict(sys.modules, self.modules)
        self.context.start()
        self.addCleanup(self.context.stop)
        self.environment = patch.dict("os.environ", {"JT_HCCL_WORLD_SIZE": "2"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.process_group = self.load(
            "jittor.distributed.process_group",
            "python/jittor/distributed/process_group.py")
        self.modules["jittor.distributed"].process_group = self.process_group
        self.collectives = self.load("routing_test.collectives", "compat/collectives.py")

    def load(self, name, path):
        spec = importlib.util.spec_from_file_location(name, ROOT / path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        self.addCleanup(sys.modules.pop, name, None)
        return module

    def gather(self, value, handle=0):
        self.calls.append(("gather", handle))
        return Tensor(np.concatenate((value.values, value.values + 10)))

    def reduce(self, value, operation="sum", handle=0):
        self.calls.append(("reduce", operation, handle))
        return Tensor(value.values * 2)

    def broadcast(self, value, root=0, handle=0):
        self.calls.append(("broadcast", root, handle))
        return Tensor(value.values + 7)

    def test_world_hccl_all_gather_needs_no_mpi_tensor_alias(self):
        result = self.collectives._all_gather_shards(Tensor([1., 2.]))
        np.testing.assert_array_equal(result.values, [1., 2., 11., 12.])
        self.assertEqual(self.calls, [("gather", 0)])

    def test_world_hccl_reduce_scatter_uses_native_fallback(self):
        self.runtime.rank = 1
        result = self.collectives._reduce_scatter_padded(Tensor([1., 2., 3., 4.]))
        np.testing.assert_array_equal(result.values, [6., 8.])
        self.assertEqual(self.calls, [("reduce", "sum", 0)])

    def test_mean_and_broadcast_use_same_native_world(self):
        value = Tensor([3., 4.])
        result = self.collectives._all_reduce_mean(value)
        np.testing.assert_array_equal(result.values, [3., 4.])
        self.collectives._broadcast_from_rank0(value)
        np.testing.assert_array_equal(value.values, [10., 11.])
        self.assertEqual(self.calls, [("reduce", "sum", 0), ("broadcast", 0, 0)])

    def test_missing_hccl_operations_raise_without_an_alias_fallback(self):
        self.runtime.compile_extern.hccl_ops = None
        with self.assertRaisesRegex((RuntimeError, NotImplementedError), "HCCL|hccl"):
            self.collectives._all_gather_shards(Tensor([1., 2.]))

    def test_world_identity_mismatch_refuses_instead_of_singleton_fallback(self):
        self.collectives._world_size = lambda: 4
        for function in (self.collectives._all_gather_shards,
                         self.collectives._reduce_scatter_padded,
                         self.collectives._all_reduce_mean,
                         self.collectives._broadcast_from_rank0):
            with self.subTest(function=function.__name__):
                with self.assertRaisesRegex(RuntimeError, "4 ranks"):
                    function(Tensor([1., 2.]))

    def test_single_rank_is_identity_without_any_backend(self):
        self.runtime.world_size = 1
        value = Tensor([2., 3.])
        self.assertIs(self.collectives._all_gather_shards(value), value)
        self.assertIs(self.collectives._reduce_scatter_padded(value), value)
        self.assertIs(self.collectives._all_reduce_mean(value), value)
        self.assertIs(self.collectives._broadcast_from_rank0(value), value)
        self.assertEqual(self.calls, [])

    def test_default_mesh_prefers_npu_on_acl_runtime(self):
        common = types.ModuleType("routing_test.fsdp2.common")
        common._rank = lambda: self.runtime.rank
        common._world_size = lambda: self.runtime.world_size
        common._prod = lambda values: int(np.prod(tuple(values)))
        with patch.dict(sys.modules, {"routing_test.fsdp2.common": common}):
            self.load("jittor._core.dtypes", "python/jittor/_core/dtypes.py")
            module = self.load("routing_test.fsdp2.dtensor", "compat/fsdp2/dtensor.py")
            mesh = module.init_device_mesh(mesh_shape=(2,))
        self.assertEqual(mesh.device_type, "npu")
        self.assertIsNone(mesh.get_group().ranks)

    def public_api(self):
        # Load the unchanged production function bodies with the same native
        # ProcessGroup owner as the other routing contracts. No JIT import.
        source = ast.parse((ROOT / "compat/torch/installers/distributed.py").read_text())
        names = {"_all_reduce", "_broadcast", "_barrier", "_collective_result",
                 "_sync_collective", "_reduce_name", "_ReduceOp", "_copy_tensor",
                 "_require_supported_group", "_group_size", "_native_all_gather_flat"}
        selected = [node for node in source.body if isinstance(
            node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
        self.runtime.array = lambda values: Tensor(values)
        self.runtime.maximum = lambda x, y: Tensor(np.maximum(x.values, y.values))
        self.runtime.minimum = lambda x, y: Tensor(np.minimum(x.values, y.values))
        namespace = {"_process_group": self.process_group, "_collectives": self.collectives,
                     "_distributed_world_size": lambda: self.runtime.world_size,
                     "_distributed_rank": lambda: self.runtime.rank,
                     "_JittorWork": lambda value: types.SimpleNamespace(value=value),
                     "jt": self.runtime, "np": np}
        exec(compile(ast.Module(body=selected, type_ignores=[]),
                     "<distributed WORLD routing>", "exec"), namespace)
        return namespace

    def test_public_world_sum_mean_broadcast_barrier_use_native_owner(self):
        api = self.public_api()
        value = Tensor([1., 2.])
        api["_all_reduce"](value, api["_ReduceOp"].SUM)
        np.testing.assert_array_equal(value.values, [2., 4.])
        api["_all_reduce"](value, api["_ReduceOp"].MEAN)
        np.testing.assert_array_equal(value.values, [2., 4.])
        api["_broadcast"](value)
        np.testing.assert_array_equal(value.values, [9., 11.])
        api["_barrier"]()
        self.assertEqual(self.calls, [("reduce", "sum", 0), ("reduce", "sum", 0),
                                     ("broadcast", 0, 0), ("reduce", "sum", 0)])

    def test_public_world_missing_backend_reports_native_owner_error(self):
        api = self.public_api()
        self.runtime.compile_extern.hccl_ops = None
        for name, args in (("_all_reduce", (Tensor([1.]),)),
                           ("_broadcast", (Tensor([1.]),)), ("_barrier", ())):
            with self.subTest(api=name):
                with self.assertRaisesRegex(RuntimeError, "HCCL.*unavailable"):
                    api[name](*args)

    def test_public_world_singleton_issues_no_collectives(self):
        api = self.public_api()
        self.runtime.world_size = 1
        value = Tensor([1., 2.])
        api["_all_reduce"](value)
        api["_broadcast"](value)
        api["_barrier"]()
        np.testing.assert_array_equal(value.values, [1., 2.])
        self.assertEqual(self.calls, [])

    def test_public_world_extrema_and_product_keep_gather_composition(self):
        api = self.public_api()
        for name, expected in (("MAX", [11., 12.]), ("MIN", [1., 2.]),
                               ("PRODUCT", [11., 24.])):
            with self.subTest(operation=name):
                value = Tensor([1., 2.])
                api["_all_reduce"](value, getattr(api["_ReduceOp"], name))
                np.testing.assert_array_equal(value.values, expected)
        self.assertEqual(self.calls, [("gather", 0)] * 3)

    def test_hccl_backend_capability_is_exposed_from_acl_provider(self):
        source = ast.parse((ROOT / "compat/torch/installers/distributed.py").read_text())
        selected = [node for node in source.body if isinstance(
            node, (ast.FunctionDef, ast.ClassDef)) and node.name in (
                "Backend", "_api_dist_is_backend_available")]
        module = ast.Module(body=selected, type_ignores=[])
        namespace = {"jt": self.runtime}
        exec(compile(module, "<distributed capability>", "exec"), namespace)
        self.assertEqual(namespace["Backend"].HCCL, "hccl")
        available = namespace["_api_dist_is_backend_available"]
        self.assertTrue(available("hccl"))
        self.runtime.compiler.has_acl = False
        self.assertFalse(available("hccl"))


if __name__ == "__main__":
    unittest.main()
