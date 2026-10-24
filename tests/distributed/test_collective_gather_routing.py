"""Host-only routing contracts; these mocks do not establish HCCL correctness."""

import ast
from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import types
import unittest
from unittest.mock import Mock, patch


class TestCollectiveGatherRouting(unittest.TestCase):
    def setUp(self):
        self.runtime = types.ModuleType("jittor")
        self.runtime.world_size = 2
        self.runtime.compile_extern = types.SimpleNamespace(nccl_ops=None, hccl_ops=None)
        self.runtime.core = types.SimpleNamespace(dispatch_context=Mock(return_value=("acl", 0)))
        package = types.ModuleType("collective_routing_test")
        package.__path__ = []
        diagnostics = types.ModuleType("collective_routing_test.diagnostics")
        diagnostics.EXPECTED = (AttributeError, RuntimeError)
        diagnostics.swallowed = Mock()
        path = Path(__file__).resolve().parents[2] / "compat/collectives.py"
        spec = importlib.util.spec_from_file_location("collective_routing_test.collectives", path)
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {"jittor": self.runtime,
                                        "collective_routing_test": package,
                                        "collective_routing_test.diagnostics": diagnostics}):
            spec.loader.exec_module(self.module)
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def test_acl_uses_real_hccl_entry_without_mpi_method(self):
        tensor, result = object(), object()
        gather = Mock(return_value=result)
        self.runtime.compile_extern.hccl_ops = types.SimpleNamespace(hccl_all_gather=gather)
        self.assertIs(self.module._all_gather_shards(tensor), result)
        gather.assert_called_once_with(tensor)

    def test_acl_takes_priority_over_available_nccl_and_mpi(self):
        tensor = types.SimpleNamespace(mpi_all_gather=Mock())
        hccl, nccl = Mock(return_value=object()), Mock()
        self.runtime.compile_extern.hccl_ops = types.SimpleNamespace(hccl_all_gather=hccl)
        self.runtime.compile_extern.nccl_ops = types.SimpleNamespace(nccl_all_gather=nccl)
        self.assertIs(self.module._all_gather_shards(tensor), hccl.return_value)
        hccl.assert_called_once_with(tensor)
        nccl.assert_not_called()
        tensor.mpi_all_gather.assert_not_called()

    def test_missing_hccl_refuses_other_backend(self):
        tensor = types.SimpleNamespace(mpi_all_gather=Mock())
        nccl = Mock()
        self.runtime.compile_extern.nccl_ops = types.SimpleNamespace(nccl_all_gather=nccl)
        with self.assertRaisesRegex(RuntimeError, "HCCL all_gather"):
            self.module._all_gather_shards(tensor)
        nccl.assert_not_called()
        tensor.mpi_all_gather.assert_not_called()

    def test_hccl_execution_error_is_not_swallowed(self):
        self.runtime.compile_extern.hccl_ops = types.SimpleNamespace(
            hccl_all_gather=Mock(side_effect=RuntimeError("communicator failed")))
        with self.assertRaisesRegex(RuntimeError, "communicator failed"):
            self.module._all_gather_shards(object())

    def test_legacy_acl_spelling_routes_hccl(self):
        self.runtime.core.dispatch_context.return_value = ("acl_legacy", 1)
        gather = Mock(return_value=object())
        self.runtime.compile_extern.hccl_ops = types.SimpleNamespace(hccl_all_gather=gather)
        self.assertIs(self.module._all_gather_shards(object()), gather.return_value)
        self.assertEqual(gather.call_count, 1)

    def test_cuda_keeps_nccl_route(self):
        self.runtime.core.dispatch_context.return_value = ("cuda", 0)
        tensor = object()
        nccl, hccl = Mock(return_value=object()), Mock()
        self.runtime.compile_extern.nccl_ops = types.SimpleNamespace(nccl_all_gather=nccl)
        self.runtime.compile_extern.hccl_ops = types.SimpleNamespace(hccl_all_gather=hccl)
        self.assertIs(self.module._all_gather_shards(tensor), nccl.return_value)
        nccl.assert_called_once_with(tensor)
        hccl.assert_not_called()

    def test_cpu_keeps_mpi_route_without_accelerator(self):
        self.runtime.core.dispatch_context.return_value = ("cpu", 0)
        tensor = types.SimpleNamespace(mpi_all_gather=Mock(return_value=object()))
        self.assertIs(self.module._all_gather_shards(tensor), tensor.mpi_all_gather.return_value)
        tensor.mpi_all_gather.assert_called_once_with()

    def test_single_rank_is_identity_without_backend_lookup(self):
        self.runtime.world_size = 1
        tensor = object()
        self.assertIs(self.module._all_gather_shards(tensor), tensor)
        self.runtime.core.dispatch_context.assert_not_called()


class TestDynamicHcclHostContracts(unittest.TestCase):
    """Execute the actual Python control functions against non-JIT doubles."""

    def setUp(self):
        self.events = []
        self.unlocked = False
        self.flags = types.SimpleNamespace(device_id=0, use_cuda=0)
        self.ops = types.SimpleNamespace(hccl_all_reduce=Mock(return_value=12),
                                        hccl_broadcast=Mock(return_value=object()))
        self.hccl = types.SimpleNamespace(hccl_init=Mock(side_effect=self.initialize_hccl))
        self.runtime = types.SimpleNamespace(
            compiler=types.SimpleNamespace(has_acl=True), flags=self.flags,
            compile_extern=types.SimpleNamespace(rank=0, world_size=1, in_mpi=False,
                hccl_mod=self.hccl, hccl_ops=self.ops, nccl_ops=None,
                setup_hccl=Mock(side_effect=lambda **kw: self.events.append(("setup", kw)))),
            core=types.SimpleNamespace(Var=type("FakeVar", (), {})),
        )
        @contextmanager
        def unlock_scope():
            self.unlocked = True
            self.events.append("unlock")
            try:
                yield
            finally:
                self.events.append("relock")
                self.unlocked = False
        fake_utils = types.ModuleType("jittor_utils")
        fake_utils.lock = types.SimpleNamespace(unlock_scope=unlock_scope)
        patched = patch.dict("sys.modules", {"jittor_utils": fake_utils})
        patched.start()
        self.addCleanup(patched.stop)
        environment = patch.dict(os.environ, {
            "JT_HCCL_ROOTINFO_FILE": "/task-owned-unused-rendezvous",
            "LOCAL_RANK": "1", "JITTOR_TORCH_DISTRIBUTED_AUTO_INIT": "1",
        }, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        self.state = {}
        context = types.SimpleNamespace(state={"distributed_api": {
            "state": self.state,
            "world_group": types.SimpleNamespace(_get_backend_name=lambda: "hccl"),
        }})
        self.namespace = {
            "jt": self.runtime, "os": os,
            "set_env": Mock(side_effect=lambda key, value: os.environ.__setitem__(key, str(value))),
            "set_flag": Mock(side_effect=lambda flags, key, value: setattr(flags, key, value)),
            "get_install_context": lambda jt: context,
            "_native_distributed_active": lambda: self.runtime.compile_extern.in_mpi and self.runtime.compile_extern.world_size > 1,
            "_distributed_world_size": lambda: self.runtime.compile_extern.world_size,
            "_distributed_rank": lambda: self.runtime.compile_extern.rank,
        }
        path = Path(__file__).resolve().parents[2] / "compat/torch/installers/distributed.py"
        selected = {"_bootstrap_hccl", "_bootstrap_hccl_all_reduce", "_bootstrap_hccl_broadcast",
                    "_is_truthy", "_bootstrap_native_distributed", "_control_tensor",
                    "_init_process_group", "_backend_matches_active"}
        tree = ast.parse(path.read_text())
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in selected]
        self.assertEqual({node.name for node in tree.body}, selected)
        exec(compile(tree, str(path), "exec"), self.namespace)

    def initialize_hccl(self):
        self.assertTrue(self.unlocked, "communicator init must release the JIT lock")
        self.events.append("init")

    def test_no_acl_fails_before_setup_or_state_mutation(self):
        self.runtime.compiler.has_acl = False
        with self.assertRaisesRegex(RuntimeError, "requires ACL"):
            self.namespace["_bootstrap_hccl"](1, 2)
        self.runtime.compile_extern.setup_hccl.assert_not_called()
        self.namespace["set_env"].assert_not_called()

    def test_missing_rendezvous_fails_before_setup(self):
        del os.environ["JT_HCCL_ROOTINFO_FILE"]
        with self.assertRaisesRegex(RuntimeError, "JT_HCCL_ROOTINFO_FILE"):
            self.namespace["_bootstrap_hccl"](1, 2)
        self.runtime.compile_extern.setup_hccl.assert_not_called()

    def test_initialization_unlocks_and_publishes_native_authority(self):
        self.assertTrue(self.namespace["_bootstrap_hccl"](1, 2))
        self.assertEqual(self.events, [("setup", {"no_mpi": True}), "unlock", "init", "relock"])
        self.assertEqual((self.runtime.compile_extern.rank, self.runtime.compile_extern.world_size,
                          self.runtime.compile_extern.in_mpi), (1, 2, True))
        self.assertEqual(os.environ["JT_HCCL_RANK"], "1")
        self.assertEqual(os.environ["JT_HCCL_WORLD_SIZE"], "2")
        self.assertEqual(self.flags.device_id, 1)
        self.assertIs(self.runtime.core.Var.mpi_all_reduce, self.namespace["_bootstrap_hccl_all_reduce"])
        self.assertIs(self.runtime.core.Var.mpi_broadcast, self.namespace["_bootstrap_hccl_broadcast"])

    def test_single_visible_device_uses_logical_zero(self):
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "5"
        self.namespace["_bootstrap_hccl"](1, 2)
        self.assertEqual(os.environ["JT_HCCL_LOCAL_RANK"], "0")
        self.assertEqual(self.flags.device_id, 0)

    def test_missing_ops_refuses_to_publish_success(self):
        self.runtime.compile_extern.hccl_ops = None
        with self.assertRaisesRegex(RuntimeError, "did not publish"):
            self.namespace["_bootstrap_hccl"](1, 2)
        self.hccl.hccl_init.assert_not_called()
        self.assertFalse(self.runtime.compile_extern.in_mpi)

    def test_init_error_propagates_and_restores_lock(self):
        self.hccl.hccl_init.side_effect = RuntimeError("bad communicator")
        with self.assertRaisesRegex(RuntimeError, "bad communicator"):
            self.namespace["_bootstrap_hccl"](1, 2)
        self.assertFalse(self.unlocked)
        self.assertFalse(self.runtime.compile_extern.in_mpi)
        self.assertEqual(self.runtime.compile_extern.world_size, 1)

    def test_hccl_mean_and_broadcast_use_native_ops(self):
        self.runtime.compile_extern.world_size = 2
        value = object()
        self.assertEqual(self.namespace["_bootstrap_hccl_all_reduce"](value), 6)
        self.ops.hccl_all_reduce.assert_called_once_with(value, "sum")
        self.assertIs(self.namespace["_bootstrap_hccl_broadcast"](value, 1), self.ops.hccl_broadcast.return_value)
        self.ops.hccl_broadcast.assert_called_once_with(value, 1)

    def test_control_tensor_moves_only_protocol_payload(self):
        class Tensor:
            def __init__(self, value):
                self.value, self.device = value, "cpu"
            def cuda(self):
                result = Tensor(self.value)
                result.device = "accelerator"
                return result
        self.runtime.array = Tensor
        result = self.namespace["_control_tensor"]([3])
        self.assertEqual(result.device, "accelerator")
        self.assertEqual(self.runtime.array([9]).device, "cpu")
        self.namespace["set_env"].assert_not_called()
        self.namespace["set_flag"].assert_not_called()
        self.runtime.compile_extern.hccl_ops = None
        self.assertEqual(self.namespace["_control_tensor"]([3]).device, "cpu")

    def test_explicit_single_rank_hccl_cannot_fake_disabled_bootstrap(self):
        os.environ["JITTOR_TORCH_DISTRIBUTED_AUTO_INIT"] = "0"
        with self.assertRaises(RuntimeError):
            self.namespace["_init_process_group"](backend="hccl", rank=0, world_size=1)
        self.assertFalse(self.state.get("initialized", False))


if __name__ == "__main__":
    unittest.main()
