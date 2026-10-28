"""Resident-profile control-plane contracts; no actual accelerator/JIT claim."""
import hashlib
import importlib
import inspect
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest import mock

import pytest

from jittor_adapters._common import UnsupportedAdapterVersion
from jittor_adapters.verl import fsdp2_resident_reference as adapter


class TestFSDP2ResidentReference(unittest.TestCase):
    def config(self, **changes):
        values = dict(strategy="fsdp2", forward_only=True, offload_policy=False,
                      param_offload=False, optimizer_offload=False)
        values.update(changes)
        return types.SimpleNamespace(**values)

    def test_explicit_cuda_resident_scope_registers_existing_engine(self):
        registrations = []
        with mock.patch.dict(os.environ, {
                "VERL_DEVICE": "cuda", "JITTOR_VERL_FSDP2_RESIDENT_REFERENCE": "1",
                "JITTOR_VERL_NATIVE_NPU_KERNELS": "0"}):
            adapter.register(lambda *entry: registrations.append(entry))
        self.assertEqual(registrations, [(adapter.MODULE, adapter.patch_fsdp_engine)])

    def test_explicit_cuda_resident_runtime_does_not_require_native_npu(self):
        frontend = types.ModuleType("torch")
        frontend._C = object()  # Host identity stub, not a binary/hardware claim.
        frontend.version = types.SimpleNamespace(cuda="12.8", hip=None)
        frontend.cuda = types.SimpleNamespace(is_available=lambda: True)
        root = types.ModuleType("verl")
        root.__version__ = "0.9.1"
        with mock.patch.dict(os.environ, {
                "VERL_DEVICE": "cuda", "JITTOR_VERL_FSDP2_RESIDENT_REFERENCE": "1",
                "JITTOR_VERL_NATIVE_NPU_KERNELS": "0", "JITTOR_TORCH_SHIM": "0"}), \
                mock.patch.dict(sys.modules, {"torch": frontend, "verl": root}):
            self.assertEqual(adapter._require_runtime(), adapter.CUDA_POLICY)

    def test_forward_only_builder_changes_only_temporary_placement_policy(self):
        calls = []

        def original(engine, module):
            calls.append((engine.engine_config, engine.engine_config.forward_only, module))
            engine._uses_fsdp2_cpu_offload_policy = False
            return "wrapped"

        config = self.config()
        engine = types.SimpleNamespace(engine_config=config)
        replacement = adapter._make_builder(original)
        with mock.patch.object(adapter, "_require_runtime"):
            self.assertEqual(replacement(engine, "model"), "wrapped")
        self.assertIs(engine.engine_config, config)
        self.assertIsNot(calls[0][0], config)
        self.assertFalse(calls[0][1])
        self.assertEqual(calls[0][2], "model")
        self.assertEqual(engine._jittor_verl_resident_reference, adapter.POLICY)

    def test_trainable_engine_uses_unmodified_builder(self):
        config = self.config(forward_only=False)
        engine = types.SimpleNamespace(engine_config=config)
        seen = []
        replacement = adapter._make_builder(lambda owner, module: seen.append(owner.engine_config) or module)
        self.assertEqual(replacement(engine, "model"), "model")
        self.assertEqual(seen, [config])

    def test_explicit_offload_is_rejected_and_config_is_restored_on_failure(self):
        for name in ("offload_policy", "param_offload", "optimizer_offload"):
            engine = types.SimpleNamespace(engine_config=self.config(**{name: True}))
            replacement = adapter._make_builder(lambda owner, module: self.fail("must fail before build"))
            with self.subTest(name=name), mock.patch.object(adapter, "_require_runtime"), \
                    self.assertRaises(UnsupportedAdapterVersion):
                replacement(engine, "model")

        config = self.config()
        engine = types.SimpleNamespace(engine_config=config)
        replacement = adapter._make_builder(lambda owner, module: (_ for _ in ()).throw(ValueError("build")))
        with mock.patch.object(adapter, "_require_runtime"), self.assertRaisesRegex(ValueError, "build"):
            replacement(engine, "model")
        self.assertIs(engine.engine_config, config)


if __name__ == "__main__":
    unittest.main()


HOST_ENGINE_SOURCE = '''class FSDPEngine:
    def _build_fsdp_module(self, module):
        self.seen_forward_only = self.engine_config.forward_only
        self._uses_fsdp2_cpu_offload_policy = bool(self.engine_config.forward_only or self.engine_config.offload_policy)
        module.offload_policy = "cpu" if self._uses_fsdp2_cpu_offload_policy else None
        return module
'''


@pytest.fixture
def cuda_profile(monkeypatch):
    monkeypatch.setenv("VERL_DEVICE", "cuda")
    monkeypatch.setenv(adapter.OPT_IN, "1")
    monkeypatch.setenv("JITTOR_TORCH_SHIM", "0")
    monkeypatch.setenv("JITTOR_VERL_NATIVE_NPU_KERNELS", "0")
    root = types.ModuleType("verl")
    root.__version__ = "0.9.1"
    frontend = types.ModuleType("torch")
    frontend._C = object()
    frontend.version = types.SimpleNamespace(cuda="12.8", hip=None)
    frontend.cuda = types.SimpleNamespace(is_available=lambda: True)
    frontend.distributed = types.SimpleNamespace(is_initialized=lambda: True, get_world_size=lambda: 1,
                                                get_rank=lambda: 0, get_backend=lambda: "nccl")
    monkeypatch.setitem(sys.modules, "verl", root)
    monkeypatch.setitem(sys.modules, "torch", frontend)
    return frontend


@pytest.fixture
def host_engine(tmp_path, monkeypatch, cuda_profile):
    filename = tmp_path / "host_engine.py"
    filename.write_text(HOST_ENGINE_SOURCE)
    module = types.ModuleType(adapter.MODULE)
    module.__file__ = str(filename)
    exec(compile(HOST_ENGINE_SOURCE, str(filename), "exec"), vars(module))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    # Host ownership/lifecycle tests substitute ONLY fake source digests. The
    # production module/method pins remain fixed and separately checked below.
    monkeypatch.setattr(adapter, "SOURCE_SHA256", hashlib.sha256(filename.read_bytes()).hexdigest())
    monkeypatch.setattr(adapter, "METHOD_SHA256", hashlib.sha256(inspect.getsource(module.FSDPEngine._build_fsdp_module).encode()).hexdigest())
    engine = module.FSDPEngine()
    engine.engine_config = types.SimpleNamespace(strategy="fsdp2", forward_only=True,
                                                  offload_policy=False, param_offload=False, optimizer_offload=False)
    engine.device_mesh = types.SimpleNamespace(device_type="cuda", size=lambda: 1)
    return module, engine


def test_cuda_resident_calls_original_builder_with_no_offload_and_restores_forward_only(host_engine):
    module, engine = host_engine
    config = engine.engine_config
    target = types.SimpleNamespace()
    _, original, replacement = adapter.prepare(module)
    assert replacement(engine, target) is target
    assert engine.engine_config is config and config.forward_only is True
    assert engine.seen_forward_only is False and target.offload_policy is None
    assert engine._uses_fsdp2_cpu_offload_policy is False
    assert engine._jittor_verl_resident_reference == adapter.CUDA_POLICY
    assert engine._jittor_verl_resident_reference_backend == "cuda"
    assert inspect.signature(replacement) == inspect.signature(original)
    assert adapter.POLICY == "single_rank_fsdp2_reference_resident_on_npu_v1"


@pytest.mark.parametrize("option", ("offload_policy", "param_offload", "optimizer_offload"))
def test_cuda_resident_rejects_real_offload_requests(host_engine, option):
    module, engine = host_engine
    setattr(engine.engine_config, option, True)
    _, _, replacement = adapter.prepare(module)
    original = engine.engine_config
    with pytest.raises(UnsupportedAdapterVersion, match="all explicit FSDP2 offload"):
        replacement(engine, types.SimpleNamespace())
    assert engine.engine_config is original
    assert not hasattr(engine, "seen_forward_only")


@pytest.mark.parametrize("failure", ("uninitialized", "world_size", "mesh_type", "mesh_size"))
def test_cuda_resident_rejects_unproven_single_cuda_rank(host_engine, cuda_profile, failure):
    module, engine = host_engine
    if failure == "uninitialized":
        cuda_profile.distributed.is_initialized = lambda: False
    elif failure == "world_size":
        cuda_profile.distributed.get_world_size = lambda: 2
    elif failure == "mesh_type":
        engine.device_mesh.device_type = "cpu"
    else:
        engine.device_mesh.size = lambda: 2
    _, _, replacement = adapter.prepare(module)
    with pytest.raises(UnsupportedAdapterVersion, match="CUDA resident reference requires"):
        replacement(engine, types.SimpleNamespace())
    assert not hasattr(engine, "seen_forward_only")


@pytest.mark.parametrize("failure", ("unavailable", "cpu_build", "hip", "shim", "native_npu", "version"))
def test_cuda_resident_runtime_fails_closed(cuda_profile, monkeypatch, failure):
    if failure == "unavailable":
        cuda_profile.cuda.is_available = lambda: False
    elif failure == "cpu_build":
        cuda_profile.version.cuda = None
    elif failure == "hip":
        cuda_profile.version.hip = "6.0"
    elif failure == "shim":
        cuda_profile._torch_compat_install_context = object()
    elif failure == "native_npu":
        monkeypatch.setenv("JITTOR_VERL_NATIVE_NPU_KERNELS", "1")
    else:
        sys.modules["verl"].__version__ = "0.9.2"
    with pytest.raises(UnsupportedAdapterVersion):
        adapter._require_runtime()


@pytest.mark.parametrize("use_cuda,backend", ((1, "cuda"), (0, "cuda"), (1, "acl")))
def test_cuda_candidate_reads_public_owner_policy(cuda_profile, monkeypatch, use_cuda, backend):
    monkeypatch.setenv("JITTOR_TORCH_SHIM", "1")
    frontend = types.ModuleType("jittor.compat.torch")
    owner = types.SimpleNamespace(capability=types.SimpleNamespace(accelerator=lambda name:
        types.SimpleNamespace(enabled=True, evidence={"build_backend": backend}) if name == "cuda" else None),
        introspection=types.SimpleNamespace(policy=types.SimpleNamespace(runtime=types.SimpleNamespace(use_cuda=use_cuda))))
    frontend.namespace_owner = lambda namespace: owner if namespace is cuda_profile else None
    monkeypatch.setitem(sys.modules, "jittor", types.ModuleType("jittor"))
    monkeypatch.setitem(sys.modules, "jittor.compat", types.ModuleType("jittor.compat"))
    monkeypatch.setitem(sys.modules, frontend.__name__, frontend)
    if use_cuda == 1 and backend == "cuda":
        assert adapter._require_runtime() == adapter.CUDA_POLICY
    else:
        with pytest.raises(UnsupportedAdapterVersion, match="active native CUDA"):
            adapter._require_runtime()


@pytest.mark.parametrize("device,flag", ((None, None), ("cuda", None), ("cuda", "0"),
                                         ("npu", "1"), ("cpu", "1")))
def test_cuda_resident_registration_is_explicit_and_device_scoped(monkeypatch, device, flag):
    monkeypatch.setenv("JITTOR_VERL_NATIVE_NPU_KERNELS", "0")
    for name, value in (("VERL_DEVICE", device), (adapter.OPT_IN, flag)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    registrations = []
    adapter.register(lambda *entry: registrations.append(entry))
    assert not registrations


def test_native_npu_registration_and_runtime_policy_are_preserved(monkeypatch):
    from jittor_adapters.verl import native_npu

    monkeypatch.setenv("VERL_DEVICE", "npu")
    monkeypatch.setenv("JITTOR_VERL_NATIVE_NPU_KERNELS", "1")
    monkeypatch.delenv(adapter.OPT_IN, raising=False)
    with mock.patch.object(native_npu, "_require_runtime") as require:
        assert adapter._require_runtime() == adapter.POLICY
        require.assert_called_once_with()
    registrations = []
    adapter.register(lambda *entry: registrations.append(entry))
    assert registrations == [(adapter.MODULE, adapter.patch_fsdp_engine)]


def test_cuda_source_pins_and_foreign_owner_fail_closed(host_engine, monkeypatch):
    module, _ = host_engine
    with monkeypatch.context() as context:
        context.setattr(adapter, "SOURCE_SHA256", "0" * 64)
        with pytest.raises(UnsupportedAdapterVersion, match="module source changed"):
            adapter.prepare(module)
    with monkeypatch.context() as context:
        context.setattr(adapter, "METHOD_SHA256", "0" * 64)
        with pytest.raises(UnsupportedAdapterVersion, match="_build_fsdp_module source changed"):
            adapter.prepare(module)
    module.FSDPEngine._build_fsdp_module = lambda *args: None
    with pytest.raises(UnsupportedAdapterVersion, match="ownership changed"):
        adapter.prepare(module)


def test_cuda_resident_detects_builder_ignoring_no_offload_and_restores_config(host_engine):
    _, engine = host_engine
    config = engine.engine_config

    def ignores_request(owner, target):
        owner._uses_fsdp2_cpu_offload_policy = True
        return target

    replacement = adapter._make_builder(ignores_request, adapter.CUDA_POLICY)
    with pytest.raises(RuntimeError, match="unexpectedly selected CPUOffloadPolicy"):
        replacement(engine, types.SimpleNamespace())
    assert engine.engine_config is config and config.forward_only
    assert not hasattr(engine, "_jittor_verl_resident_reference")


def test_cuda_resident_keeps_training_engine_unmodified(host_engine):
    module, engine = host_engine
    engine.engine_config.forward_only = False
    config = engine.engine_config
    _, _, replacement = adapter.prepare(module)
    target = types.SimpleNamespace()
    assert replacement(engine, target) is target
    assert engine.engine_config is config and not hasattr(engine, "_jittor_verl_resident_reference")


def test_resident_adapter_import_boundary_and_original_production_pins():
    adapters = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(adapters / "tests"))
    from _adapter_boundary import boundary_errors

    assert not boundary_errors(Path(adapter.__file__).read_text())
    assert adapter.SOURCE_SHA256 == "79d8779fb2a0b8a1ac7b3ec85fba2521ac8cae079925a497c4366fd77f8abfbe"
    assert adapter.METHOD_SHA256 == "35b7b539858443e45894da2d9675fb5a1d7b2bbb7c7c8b35605b0746d3179213"


@pytest.fixture
def resident_patcher(monkeypatch, cuda_profile):
    adapters = Path(__file__).resolve().parents[3]
    package = types.ModuleType("_resident_host_compat")
    package.__path__ = [str(adapters.parent / "compat")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    patcher = importlib.import_module(package.__name__ + ".module_patcher")
    transaction = importlib.import_module(package.__name__ + ".transaction")
    frontend = types.ModuleType("jittor.compat.torch")
    frontend.namespace_owner = lambda api: types.SimpleNamespace(capability=types.SimpleNamespace(accelerator=lambda name:
        types.SimpleNamespace(enabled=True, evidence={"build_backend": "cuda"}) if name == "cuda" else None),
        introspection=types.SimpleNamespace(policy=types.SimpleNamespace(
            runtime=types.SimpleNamespace(use_cuda=1)))) if api is cuda_profile else None
    monkeypatch.setenv("JITTOR_TORCH_SHIM", "1")
    monkeypatch.setitem(sys.modules, "jittor", types.ModuleType("jittor"))
    monkeypatch.setitem(sys.modules, "jittor.compat", package)
    monkeypatch.setitem(sys.modules, "jittor.compat.torch", frontend)
    monkeypatch.setitem(sys.modules, "jittor.compat.module_patcher", patcher)
    yield patcher, transaction
    patcher.uninstall_module_patches()
    patcher._REGISTRY.clear()


def test_cuda_owned_patch_is_idempotent_and_restorable(host_engine, resident_patcher):
    module, _ = host_engine
    patcher, _ = resident_patcher
    original = module.FSDPEngine._build_fsdp_module
    adapter.register(patcher.register_module_patch)
    assert patcher.install_module_patches(load_entry_points=False).ok
    replacement = module.FSDPEngine._build_fsdp_module
    assert replacement is not original
    assert adapter.patch_fsdp_engine(module) is False
    assert module.FSDPEngine._build_fsdp_module is replacement
    patcher.uninstall_module_patches()
    assert module.FSDPEngine._build_fsdp_module is original


def test_cuda_owned_restore_preserves_external_builder(host_engine, resident_patcher):
    module, _ = host_engine
    patcher, transaction = resident_patcher
    adapter.register(patcher.register_module_patch)
    assert patcher.install_module_patches(load_entry_points=False).ok
    external = lambda *args: "external builder"
    module.FSDPEngine._build_fsdp_module = external
    with pytest.raises(transaction.TransactionConflict):
        patcher.uninstall_module_patches()
    assert module.FSDPEngine._build_fsdp_module is external


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[3].parent
         / "agent/skills/verl-torch-compat/runtime/oracle/sitecustomize.py").is_file(),
    reason="agent/skills verl-torch-compat oracle fixture is not ported")
@pytest.mark.parametrize("preload", (False, True))
@pytest.mark.parametrize("hf_async", ("0", "1"))
def test_oracle_cuda_resident_registration_does_not_import_jittor(tmp_path, preload, hf_async):
    adapters = Path(__file__).resolve().parents[3]
    root = adapters.parent
    bootstrap = root / "agent/skills/verl-torch-compat/runtime/oracle/sitecustomize.py"
    filename = tmp_path / "transformer_impl.py"
    filename.write_text(HOST_ENGINE_SOURCE)
    program = (
        "import hashlib, importlib, inspect, runpy, sys, types\n"
        + "sys.path.insert(0, " + repr(str(adapters)) + ")\n"
        "root = types.ModuleType('verl')\nroot.__version__ = '0.9.1'\nroot.__path__ = []\n"
        "sys.modules['verl'] = root\n"
        "for name in ('verl.workers', 'verl.workers.engine', 'verl.workers.engine.fsdp'):\n"
        "    package = types.ModuleType(name)\n"
        + "    package.__path__ = [" + repr(str(tmp_path)) + "]\n"
        "    sys.modules[name] = package\n"
        "torch = types.ModuleType('torch')\ntorch._C = object()\n"
        "torch.version = types.SimpleNamespace(cuda='12.8', hip=None)\n"
        "torch.cuda = types.SimpleNamespace(is_available=lambda: True)\n"
        "torch.distributed = types.SimpleNamespace(is_initialized=lambda: True, get_world_size=lambda: 1)\n"
        "sys.modules['torch'] = torch\n"
        "from jittor_adapters.verl import fsdp2_resident_reference as adapter\n"
        + "adapter.SOURCE_SHA256 = " + repr(hashlib.sha256(HOST_ENGINE_SOURCE.encode()).hexdigest()) + "\n"
        + "adapter.METHOD_SHA256 = " + repr(hashlib.sha256("".join(HOST_ENGINE_SOURCE.splitlines(keepends=True)[1:]).encode()).hexdigest()) + "\n"
    )
    if preload:
        program += "module = importlib.import_module(adapter.MODULE)\n"
    program += "bootstrap = runpy.run_path(" + repr(str(bootstrap)) + ")\n"
    if not preload:
        program += "module = importlib.import_module(adapter.MODULE)\n"
    program += (
        "replacement = module.FSDPEngine._build_fsdp_module\n"
        "assert replacement._jittor_verl_resident_reference_policy == adapter.CUDA_POLICY\n"
        "bootstrap['_register_resident_reference'](module)\n"
        "assert module.FSDPEngine._build_fsdp_module is replacement\n"
        "engine = module.FSDPEngine()\n"
        "engine.engine_config = types.SimpleNamespace(strategy='fsdp2', forward_only=True, "
        "offload_policy=False, param_offload=False, optimizer_offload=False)\n"
        "config = engine.engine_config\n"
        "engine.device_mesh = types.SimpleNamespace(device_type='cuda', size=lambda: 1)\n"
        "target = types.SimpleNamespace()\n"
        "assert engine._build_fsdp_module(target) is target\n"
        "assert target.offload_policy is None\n"
        "assert engine.engine_config is config and config.forward_only\n"
        "assert engine._jittor_verl_resident_reference_backend == 'cuda'\n"
        "assert not any(name == 'jittor' or name.startswith('jittor.') for name in sys.modules)\n"
        "assert 'jittor_adapters.verl.hf_async_rollout' not in sys.modules\n"
        "print('ORACLE_RESIDENT_HOST_CONTRACT_OK')\n"
    )
    environment = {name: value for name, value in os.environ.items() if not name.startswith("JITTOR_")}
    environment.update(VERL_DEVICE="cuda", JITTOR_VERL_FSDP2_RESIDENT_REFERENCE="1",
                       JITTOR_VERL_HF_ASYNC_ROLLOUT=hf_async)
    result = subprocess.run([sys.executable, "-B", "-S", "-c", program], env=environment,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "ORACLE_RESIDENT_HOST_CONTRACT_OK" in result.stdout


class _WorldTwoMesh:
    """Control-plane host stub only; never evidence of real sharding or CUDA."""

    def __init__(self, rank):
        self.device_type = "cuda"
        self.ndim = 1
        self.mesh = self
        self.members = [0, 1]
        self.rank_value = rank
        self.group = types.SimpleNamespace(size=lambda: 2, rank=lambda: rank)

    def size(self):
        return 2

    def get_rank(self):
        return self.rank_value

    def reshape(self, *args):
        return self

    def tolist(self):
        return self.members

    def get_group(self):
        return self.group


def _world_two_control_plane(frontend, engine, rank):
    mesh = _WorldTwoMesh(rank)
    frontend.distributed = types.SimpleNamespace(
        is_initialized=lambda: True,
        get_world_size=lambda group=None: 2,
        get_rank=lambda group=None: rank,
        get_backend=lambda group=None: "nccl",
        get_process_group_ranks=lambda group: [0, 1],
    )
    engine.device_mesh = mesh
    return mesh


@pytest.mark.parametrize("rank", (0, 1))
def test_cuda_resident_world_two_original_builder_and_forward_only_restoration(host_engine, cuda_profile, rank):
    module, engine = host_engine
    _world_two_control_plane(cuda_profile, engine, rank)
    config = engine.engine_config
    target = types.SimpleNamespace()
    _, original, replacement = adapter.prepare(module)
    assert replacement(engine, target) is target
    assert engine.engine_config is config and config.forward_only is True
    assert engine.seen_forward_only is False and target.offload_policy is None
    assert engine._uses_fsdp2_cpu_offload_policy is False
    assert engine._jittor_verl_resident_reference == adapter.CUDA_POLICY
    assert adapter.CUDA_POLICY == "world1_or2_fsdp2_reference_resident_on_cuda_v2"
    assert replacement._jittor_verl_original is original


@pytest.mark.parametrize("failure", (
    "world3", "backend", "mesh_ndim", "mesh_members", "mesh_rank",
    "group_size", "group_rank", "group_members", "group_backend",
))
def test_cuda_resident_world_two_rejects_mismatched_control_plane(host_engine, cuda_profile, failure):
    module, engine = host_engine
    mesh = _world_two_control_plane(cuda_profile, engine, 0)
    if failure == "world3":
        cuda_profile.distributed.get_world_size = lambda group=None: 3
    elif failure == "backend":
        cuda_profile.distributed.get_backend = lambda group=None: "gloo"
    elif failure == "mesh_ndim":
        mesh.ndim = 2
    elif failure == "mesh_members":
        mesh.members = [1, 0]
    elif failure == "mesh_rank":
        mesh.rank_value = 1
    elif failure == "group_size":
        mesh.group.size = lambda: 1
    elif failure == "group_rank":
        mesh.group.rank = lambda: 1
    elif failure == "group_members":
        cuda_profile.distributed.get_process_group_ranks = lambda group: [0, 0]
    else:
        cuda_profile.distributed.get_backend = lambda group=None: "gloo" if group is not None else "nccl"
    _, _, replacement = adapter.prepare(module)
    config = engine.engine_config
    with pytest.raises(UnsupportedAdapterVersion):
        replacement(engine, types.SimpleNamespace())
    assert engine.engine_config is config and config.forward_only is True
    assert not hasattr(engine, "seen_forward_only")
