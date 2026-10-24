"""Host-only device contracts; stubs are not accelerator or NCCL evidence."""
from contextlib import ExitStack, contextmanager
import ast
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy  # Load its extension before patching the sys.modules map.


class TestDDPDeviceContract(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.current = None
        self.precision = "high"
        self.torch = ModuleType("torch")
        self.torch._C = object()
        self.torch.device = lambda name, index: SimpleNamespace(type=name, index=index)
        self.torch.backends = SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
            cudnn=SimpleNamespace(allow_tf32=True, benchmark=True),
        )
        self.torch.set_float32_matmul_precision = lambda value: setattr(self, "precision", value)
        self.torch.get_float32_matmul_precision = lambda: self.precision
        self.jt = ModuleType("jittor")
        self.jt.compiler = SimpleNamespace(has_cuda=True, has_acl=False, has_rocm=False, has_corex=False)
        self.jt.flags = SimpleNamespace(use_cuda=0, device_id=0)
        self.jt.rank, self.jt.world_size = 1, 2
        self.jt.core = SimpleNamespace(backend_fallback_count=lambda: 0)
        self.jt.sync_all = lambda device: self.events.append(("jt_sync", device))

        def select(index):
            self.current = index
            self.jt.flags.device_id = index
            self.events.append(("set_device", index))

        def accelerator(name):
            return SimpleNamespace(is_available=lambda: True, set_device=select,
                                   current_device=lambda: self.current,
                                   synchronize=lambda: self.events.append((name, "sync")))
        self.torch.cuda, self.torch.npu = accelerator("cuda"), accelerator("npu")

        @contextmanager
        def scope(**options):
            self.events.append(("scope", options))
            previous = self.jt.flags.use_cuda
            self.jt.flags.use_cuda = options.get("use_cuda", previous)
            try:
                yield
            finally:
                self.jt.flags.use_cuda = previous

        @contextmanager
        def forbid():
            self.events.append("forbid_fallback")
            yield
        self.jt.runtime = SimpleNamespace(scope=scope)
        fallback = ModuleType("jittor._runtime.fallback")
        fallback.forbid_backend_fallbacks = forbid
        activation = ModuleType("jittor.compat.shim.runtime")
        activation.activation_status = lambda jt: SimpleNamespace(active=True)
        publication = ModuleType("jittor.compat.torch.publication")
        publication.namespace_owner = lambda torch: self.jt
        self.modules = {"torch": self.torch, "torch_npu": ModuleType("torch_npu"),
                        "jittor": self.jt, "jittor._runtime.fallback": fallback,
                        "jittor.compat.shim.runtime": activation,
                        "jittor.compat.torch.publication": publication}
        spec = importlib.util.spec_from_file_location("ddp_device_host_contract",
                                                     Path(__file__).with_name("probe_ddp_sync.py"))
        self.probe = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, self.modules):
            spec.loader.exec_module(self.probe)

    def initialize(self, framework, device, guard):
        return self.probe.initialize_runtime(SimpleNamespace(framework=framework, device=device),
                                             guard, rank=1, local_rank=1, world_size=2)

    def test_cuda_cli_is_explicit_and_npu_default_is_preserved(self):
        for options, expected in (([], "npu"), (["--device", "cuda"], "cuda")):
            with patch.object(sys, "argv", ["probe", "--framework", "torch", "--output", "unused"] + options):
                self.assertEqual(self.probe.parse_args().device, expected)

    def test_rank_environment_uses_only_requested_backend(self):
        with patch.dict(self.probe.os.environ, {"JT_HCCL_WORLD_SIZE": "2", "JT_HCCL_RANK": "1",
                                                "JT_HCCL_LOCAL_RANK": "1"}, clear=True):
            self.assertEqual(self.probe.rank_environment("cuda"), (0, 0, 1))
            self.assertEqual(self.probe.rank_environment("npu"), (1, 1, 2))
        with patch.dict(self.probe.os.environ, {"JT_NCCL_WORLD_SIZE": "2", "JT_NCCL_RANK": "1",
                                                "JT_NCCL_LOCAL_RANK": "0"}, clear=True):
            self.assertEqual(self.probe.rank_environment("cuda"), (1, 0, 2))
        with patch.dict(self.probe.os.environ, {"RANK": "1", "WORLD_SIZE": "2", "LOCAL_RANK": "1",
                                                "JT_NCCL_RANK": "0", "JT_NCCL_LOCAL_RANK": "0"}, clear=True):
            self.assertEqual(self.probe.rank_environment("cuda"), (1, 1, 2))

    def test_cuda_collective_backend_is_nccl_not_hccl(self):
        self.assertEqual(self.probe.distributed_backend("cuda"), "nccl")
        self.assertEqual(self.probe.distributed_backend("npu"), "hccl")

    def test_official_cuda_binding_sync_and_precision(self):
        with patch.dict(sys.modules, self.modules), ExitStack() as guard:
            jt, device, sync = self.initialize("torch", "cuda", guard)
            self.assertIsNone(jt)
            self.assertEqual((device.type, device.index), ("cuda", 1))
            sync()
            self.assertIn(("cuda", "sync"), self.events)
            self.assertEqual(self.precision, "highest")
            self.assertIs(self.torch.backends.cuda.matmul.allow_tf32, False)
            self.assertIs(self.torch.backends.cudnn.allow_tf32, False)
            self.assertIs(self.torch.backends.cudnn.benchmark, False)

    def test_official_cuda_unavailable_and_shim_are_rejected(self):
        with patch.dict(sys.modules, self.modules), ExitStack() as guard:
            self.torch.cuda.is_available = lambda: False
            with self.assertRaisesRegex(AssertionError, "CUDA unavailable"):
                self.initialize("torch", "cuda", guard)
            self.torch.cuda.is_available = lambda: True
            self.torch._torch_compat_install_context = object()
            with self.assertRaises(AssertionError):
                self.initialize("torch", "cuda", guard)

    def test_candidate_cuda_build_excludes_other_backends(self):
        for name, value in (("has_cuda", False), ("has_acl", True),
                            ("has_rocm", True), ("has_corex", True)):
            with self.subTest(name=name), patch.dict(sys.modules, self.modules), ExitStack() as guard:
                previous = getattr(self.jt.compiler, name)
                setattr(self.jt.compiler, name, value)
                try:
                    with self.assertRaisesRegex(AssertionError, "CUDA"):
                        self.initialize("jittor", "cuda", guard)
                finally:
                    setattr(self.jt.compiler, name, previous)

    def test_candidate_cuda_context_sync_and_fallback_error(self):
        with patch.dict(sys.modules, self.modules), ExitStack() as guard:
            jt, device, sync = self.initialize("jittor", "cuda", guard)
            self.assertIs(jt, self.jt)
            self.assertEqual((device.type, device.index), ("cuda", 1))
            self.assertEqual(self.jt.flags.use_cuda, 1)
            self.assertIn(("scope", {"backend_fallback": "error"}), self.events)
            self.assertIn("forbid_fallback", self.events)
            sync()
            self.assertIn(("jt_sync", True), self.events)

    def test_cuda_set_device_must_take_effect(self):
        self.torch.cuda.set_device = lambda index: None
        with patch.dict(sys.modules, self.modules), ExitStack() as guard:
            with self.assertRaisesRegex(AssertionError, "CUDA device selection"):
                self.initialize("jittor", "cuda", guard)

    def test_native_rank_identity_cannot_be_published_by_wrong_launcher(self):
        for rank, world in ((0, 2), (1, 1)):
            with self.subTest(rank=rank, world=world), patch.dict(sys.modules, self.modules), ExitStack() as guard:
                self.jt.rank, self.jt.world_size = rank, world
                with self.assertRaises(AssertionError):
                    self.initialize("jittor", "cuda", guard)

    def test_precision_policy_must_be_read_back(self):
        self.torch.set_float32_matmul_precision = lambda value: None
        with patch.dict(sys.modules, self.modules), ExitStack() as guard:
            with self.assertRaises(AssertionError):
                self.initialize("torch", "cuda", guard)

    def test_snapshot_asserts_device_before_any_host_read(self):
        tree = ast.parse(Path(self.probe.__file__).read_text())
        snapshot = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef) and node.name == "snapshot")
        self.assertEqual(snapshot.body[0].value.func.id, "synchronize")
        self.assertEqual(snapshot.body[1].value.func.id, "location")
        self.assertIsInstance(snapshot.body[2], ast.Return)
        self.assertTrue(any(isinstance(node, ast.Attribute) and node.attr == "numpy"
                            for node in ast.walk(snapshot.body[2])))

    def test_native_location_and_ordinal_are_checked_before_host_reads(self):
        with patch.dict(sys.modules, self.modules), ExitStack() as guard:
            self.initialize("jittor", "cuda", guard)
            for location, index in (("cpu", 1), ("device", 0), ("device", 1)):
                native = SimpleNamespace(location=lambda: location, device_id=index)
                if location == "device" and index == 1:
                    self.assertEqual(self.probe.check_location(native, self.jt, "cuda", 1), "device")
                else:
                    with self.assertRaises(AssertionError):
                        self.probe.check_location(native, self.jt, "cuda", 1)
            self.jt.flags.use_cuda = 0
            with self.assertRaisesRegex(AssertionError, "dispatch"):
                self.probe.check_location(SimpleNamespace(location=lambda: "device", device_id=1),
                                          self.jt, "cuda", 1)

    def test_official_tensor_must_be_on_selected_cuda_ordinal(self):
        for name, index in (("cpu", 1), ("cuda", 0), ("cuda", 1)):
            value = SimpleNamespace(device=SimpleNamespace(type=name, index=index))
            if (name, index) == ("cuda", 1):
                self.probe.check_location(value, None, "cuda", 1)
            else:
                with self.assertRaises(AssertionError):
                    self.probe.check_location(value, None, "cuda", 1)

    def test_npu_legacy_binding_is_retained(self):
        for framework in ("torch", "jittor"):
            self.jt.compiler.has_acl = True
            with patch.dict(sys.modules, self.modules), ExitStack() as guard:
                jt, device, sync = self.initialize(framework, "npu", guard)
                self.assertEqual((device.type, device.index), ("npu", 1))
                sync()
                self.assertIn(("npu", "sync") if jt is None else ("jt_sync", True), self.events)


if __name__ == "__main__":
    unittest.main()
