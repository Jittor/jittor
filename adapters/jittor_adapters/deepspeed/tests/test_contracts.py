"""Offline contracts using actual import finders and public mutation ledgers.

Run directly with Python; no Jittor runtime, Torch installation or device is
loaded. Controlled fake DeepSpeed sources are deliberately hashed as fixtures;
the production accepted hashes remain fixed in source.py.
"""
import hashlib
import importlib
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

ADAPTERS = Path(__file__).resolve().parents[3]
COMPAT = Path(os.environ.get("JITTOR_COMPAT_SOURCE", ADAPTERS.parent / "compat"))

ENGINE = """from deepspeed.compile.backend import register_compile_pass, opt_passes
from deepspeed.compile.passes import zero3_compile, prefetch, selective_gather, offload_adam_states
from deepspeed.compile.init_z1 import init_z1
from deepspeed.compile.init_z3 import init_z3

def is_deepcompile_supported():
    return False

class DeepSpeedEngine:
    def __init__(self, model=None, optimizer=None, config=None):
        self.children = {}
        modules = self.__dict__.get('_modules')
        modules['module'] = model
        if is_deepcompile_supported():
            self.register_compile_pass(zero3_compile.NAME, zero3_compile.add_z3_gather_release)
    def add_module(self, name, model):
        self.children[name] = model
    def compile(self, enable_deepcompile=False):
        if enable_deepcompile:
            return init_z1
    def register_compile_pass(self, pass_name, pass_fn):
        register_compile_pass(pass_name, pass_fn)
"""

ROOT_SOURCE = """from .accelerator import get_accelerator
BOUND = get_accelerator()
from .runtime.engine import DeepSpeedEngine
from .git_version_info_installed import version
__version__ = version

def initialize(model=None, optimizer=None, config=None):
    return DeepSpeedEngine(model=model, optimizer=optimizer, config=config)
"""

REAL_ACCELERATOR = """ds_accelerator = None

def get_accelerator():
    if ds_accelerator is None:
        raise AssertionError('provider was not selected before first use')
    return ds_accelerator

def set_accelerator(value):
    global ds_accelerator
    ds_accelerator = value
"""


ZERO_STAGE = """import torch
from deepspeed.accelerator import get_accelerator
from deepspeed.utils import groups
# Toggle this to true to enable correctness test

def split_half_float_double(tensors):
    device_type = get_accelerator().device_name()
    dtypes = [
        \"torch.{}.HalfTensor\".format(device_type), \"torch.{}.FloatTensor\".format(device_type),
        \"torch.{}.DoubleTensor\".format(device_type), \"torch.{}.BFloat16Tensor\".format(device_type)
    ]
    buckets = []
    for i, dtype in enumerate(dtypes):
        bucket = [t for t in tensors if t.type() == dtype]
        if bucket:
            buckets.append(bucket)
    return buckets

class Fixture:
    def shape_only(self, param):
        return torch.zeros_like(param.cpu_data, device=\"meta\")

    def reduce_ipg_grads(self, bucket, comm_dtype):
        if True:
            if True:
                if False:
                    pass
                else:
                    self.average_tensor(bucket.buffer[bucket.index].narrow(0, 0, bucket.elements), comm_dtype)
            else:
                pass

    def norm_a(self, value):
        return value.data.double()

    def norm_b(self, value):
        return value.data.double()

    def norm_c(self, value):
        return value.data.double()
"""

ZERO_STAGE3 = """def stage3_norms(parts, gradients):
    first = parts[0].data.double().norm(2)
    second = parts[1].data.double().norm(2)
    # upstream diagnostic comment: param.grad.data.double().norm(2)
    third = gradients[0].to("npu").double().norm(2)
    fourth = gradients[1].to("npu").double()
    return first, second, third, fourth
"""

UTILS_TORCH = """def required_torch_version(min_version=None):
    return False

def register_grad_hook(param, hook):
    if required_torch_version(min_version=2.1):
        return param.register_post_accumulate_grad_hook(hook)
    else:
        param_tmp = param.expand_as(param)
        grad_acc = param_tmp.grad_fn.next_functions[0][0]
        return grad_acc.register_hook(hook)
"""


class Contracts(unittest.TestCase):
    def setUp(self):
        self.old_path = list(sys.path)
        self.old_meta = list(sys.meta_path)
        sys.path.insert(0, str(ADAPTERS))
        self.modules = mock.patch.dict(sys.modules)
        self.modules.start()
        self.rank_env = mock.patch.dict(os.environ, {
            "LOCAL_RANK": "0", "OMPI_COMM_WORLD_LOCAL_RANK": "0", "JT_HCCL_LOCAL_RANK": "0"})
        self.rank_env.start()
        for name in list(sys.modules):
            if name == "deepspeed" or name.startswith(("deepspeed.", "jittor_adapters.deepspeed")):
                sys.modules.pop(name)
        native = types.ModuleType("jittor")
        native.__path__ = []
        compat = types.ModuleType("jittor.compat")
        compat.__path__ = [str(COMPAT)]
        torch_compat = types.ModuleType("jittor.compat.torch")
        torch_compat.__path__ = []
        tensor_state = types.ModuleType("jittor.compat.torch.tensor_state")
        tensor_state._clear_resolution_caches = lambda: None
        context = types.ModuleType("jittor.compat.torch.context")
        context._clear_resolution_caches = lambda: None
        for module in (native, compat, torch_compat, tensor_state, context):
            sys.modules[module.__name__] = module
        self.adapter = importlib.import_module("jittor_adapters.deepspeed")
        self.activation = importlib.import_module("jittor_adapters.deepspeed.activation")
        self.source = importlib.import_module("jittor_adapters.deepspeed.source")
        self.scope = importlib.import_module("jittor_adapters.deepspeed.scope")
        self.tx = importlib.import_module("jittor.compat.transaction")
        self.provider = object()
        accelerator = types.ModuleType("jittor_adapters.deepspeed.accelerator")
        accelerator.create_accelerator = lambda device: self.provider
        sys.modules[accelerator.__name__] = accelerator
        self.temp = tempfile.TemporaryDirectory(prefix="ds-adapter-contract-")
        self.package = Path(self.temp.name) / "deepspeed"
        self.make_package()
        sys.path.insert(0, self.temp.name)
        self.production_hashes = dict(self.source.SOURCE_SHA256)
        self.hash_patch = mock.patch.object(self.source, "SOURCE_SHA256", self.hashes())
        self.hash_patch.start()

    def tearDown(self):
        try:
            self.adapter.deactivate()
        finally:
            self.hash_patch.stop()
            self.rank_env.stop()
            self.temp.cleanup()
            sys.path[:] = self.old_path
            sys.meta_path[:] = self.old_meta
            self.modules.stop()

    def make_package(self, version="0.17.6", suffix=""):
        contents = {
            "__init__.py": ROOT_SOURCE + suffix,
            "git_version_info_installed.py": "version=%r\n" % version,
            "accelerator/__init__.py": "from .real_accelerator import get_accelerator, set_accelerator\n",
            "accelerator/real_accelerator.py": REAL_ACCELERATOR,
            "runtime/__init__.py": "",
            "runtime/engine.py": ENGINE,
            "runtime/zero/__init__.py": "",
            "runtime/zero/stage_1_and_2.py": ZERO_STAGE,
            "runtime/zero/stage3.py": ZERO_STAGE3,
            "utils/__init__.py": "",
            "utils/torch.py": UTILS_TORCH,
        }
        for name, text in contents.items():
            path = self.package / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(text.encode("utf-8"))

    def hashes(self):
        return {name: (hashlib.sha256((self.package / name).read_bytes()).hexdigest(),)
                for name in self.source.SOURCE_SHA256}

    def test_only_the_two_pinned_build_manifests_are_admitted(self):
        # Exact generated upstream contents, not recomputed fixture allowlists.
        manifests = {
            "cpu": """version='0.17.6'
git_hash='unknown'
git_branch='unknown'
installed_ops={'deepspeed_not_implemented': False, 'async_io': False, 'deepspeed_ccl_comm': False, 'deepspeed_shm_comm': False, 'cpu_adam': False, 'fused_adam': False}
accelerator_name='cpu'
torch_info={'version': '2.7', 'bf16_support': False, 'cuda_version': '0.0', 'nccl_version': '0.0', 'hip_version': '0.0'}
""",
            "npu": """version='0.17.6'
git_hash='unknown'
git_branch='unknown'
installed_ops={'deepspeed_not_implemented': False, 'async_io': False, 'cpu_adagrad': False, 'cpu_adam': False, 'cpu_lion': False, 'fused_adam': False, 'transformer_inference': False}
accelerator_name='npu'
torch_info={'version': '2.7', 'bf16_support': False, 'cuda_version': '0.0', 'nccl_version': '0.0', 'hip_version': '0.0'}
""",
        }
        filename = "git_version_info_installed.py"
        self.assertEqual(len(self.production_hashes[filename]), 2)
        self.assertEqual({hashlib.sha256(text.encode()).hexdigest() for text in manifests.values()},
                         set(self.production_hashes[filename]))
        self.assertTrue(all(len(hashes) == 1 for name, hashes in self.production_hashes.items()
                            if name != filename))
        self.source.SOURCE_SHA256[filename] = self.production_hashes[filename]
        for device, text in manifests.items():
            with self.subTest(device=device):
                (self.package / filename).write_bytes(text.encode())
                self.assertEqual(self.source.verified_source(self.package, filename), text)
                self.assertEqual(self.source.inspect_package(self.package), "0.17.6")
                # Same version, changed op availability: must still reject.
                changed = text.replace("'async_io': False", "'async_io': True")
                (self.package / filename).write_bytes(changed.encode())
                with self.assertRaisesRegex(self.source.UnsupportedAdapterVersion, "Unvalidated DeepSpeed source"):
                    self.source.inspect_package(self.package)
        self.assertEqual(self.adapter.status()["source_sha256"][filename],
                         list(self.production_hashes[filename]))

    def test_first_accelerator_selection_and_cpu_boundary_and_deactivation(self):
        self.adapter.activate("cpu")
        self.adapter.activate("cpu")
        self.assertEqual(sum(isinstance(item, self.activation._Finder) for item in sys.meta_path), 1)
        ds = importlib.import_module("deepspeed")
        self.assertIs(ds.BOUND, self.provider)
        real = importlib.import_module("deepspeed.accelerator.real_accelerator")
        self.assertTrue(self.adapter.status()["imported"])
        with self.assertRaisesRegex(NotImplementedError, "no real Gloo"):
            ds.initialize(config={"zero_optimization": {"stage": 0}})
        with self.assertRaisesRegex(NotImplementedError, "no real Gloo"):
            ds.DeepSpeedEngine()
        self.adapter.deactivate()
        self.assertIsNone(real.ds_accelerator)
        self.assertFalse(self.adapter.status()["active"])
        self.assertFalse(self.adapter.status()["imported"])
        self.assertFalse(any(name == "deepspeed" or name.startswith("deepspeed.") for name in sys.modules))

    def test_unknown_version_rejected_by_shared_contract(self):
        self.make_package(version="99.0.0")
        self.source.SOURCE_SHA256 = self.hashes()
        self.adapter.activate("cpu")
        with self.assertRaisesRegex(ImportError, "supports.*0.17.6"):
            importlib.import_module("deepspeed")

    def test_changed_source_rejected_before_execution(self):
        path = self.package / "runtime/engine.py"
        path.write_bytes(path.read_bytes() + b"\nraise AssertionError('must not run')\n")
        self.adapter.activate("cpu")
        with self.assertRaisesRegex(ImportError, "Unvalidated DeepSpeed source"):
            importlib.import_module("deepspeed")

    def test_reload_is_rejected_before_provider_can_change(self):
        self.adapter.activate("cpu")
        importlib.import_module("deepspeed")
        real = importlib.import_module("deepspeed.accelerator.real_accelerator")
        provider_module = sys.modules["jittor_adapters.deepspeed.accelerator"]
        provider_module.create_accelerator = lambda device: object()
        with self.assertRaisesRegex(ImportError, "reload"):
            importlib.reload(real)
        self.assertIs(real.ds_accelerator, self.provider)

    def test_already_imported_package_is_rejected(self):
        sys.modules["deepspeed"] = types.ModuleType("deepspeed")
        with self.assertRaisesRegex(RuntimeError, "before importing"):
            self.adapter.activate("cpu")

    def test_import_failure_rolls_back_provider_and_owned_modules(self):
        self.make_package(suffix="raise RuntimeError('controlled import failure')\n")
        self.source.SOURCE_SHA256 = self.hashes()
        self.adapter.activate("cpu")
        with self.assertRaisesRegex(RuntimeError, "controlled import failure"):
            importlib.import_module("deepspeed")
        self.assertFalse(any(name == "deepspeed" or name.startswith("deepspeed.") for name in sys.modules))
        self.assertFalse(self.adapter.status()["imported"])

    def test_foreign_provider_replacement_is_preserved_and_reported(self):
        self.adapter.activate("cpu")
        importlib.import_module("deepspeed")
        real = importlib.import_module("deepspeed.accelerator.real_accelerator")
        foreign = object()
        real.ds_accelerator = foreign
        with self.assertRaises(self.tx.TransactionConflict):
            self.adapter.deactivate()
        self.assertIs(real.ds_accelerator, foreign)

    def test_foreign_finder_replacement_is_preserved_and_reported(self):
        self.adapter.activate("cpu")
        finder = self.activation._finder
        foreign = object()
        sys.meta_path[sys.meta_path.index(finder)] = foreign
        with self.assertRaises(self.tx.TransactionConflict):
            self.adapter.deactivate()
        self.assertIn(foreign, sys.meta_path)

    def test_lazy_engine_transform_executes_public_module_registration(self):
        namespace = {}
        exec(compile(self.source.transform_engine(ENGINE), "engine-fixture.py", "exec"), namespace)
        engine = namespace["DeepSpeedEngine"](model=self.provider)
        self.assertIs(engine.children["module"], self.provider)
        self.assertNotIn("deepspeed.compile.backend", sys.modules)

    def test_zero_stage1_npu_transform_is_fail_closed_and_covers_both_gradient_paths(self):
        transformed = self.source.transform_zero_stage_1_and_2(ZERO_STAGE)
        transformed = self.source.transform_zero_npu_norm(transformed)
        transformed = self.source.transform_zero_npu_stage1(transformed)
        self.assertIn("tensor.dtype == dtype", transformed)
        self.assertNotIn("t.type() == dtype", transformed)
        self.assertIn("grad_reduc.copy_(reduced.narrow", transformed)
        self.assertEqual(transformed.count(".data.float()"), 3)
        with self.assertRaisesRegex(self.source.UnsupportedAdapterVersion, "anchor changed"):
            self.source.transform_zero_npu_stage1(
                ZERO_STAGE.replace("t.type() == dtype", "str(t.type()) == dtype"))

    def test_zero_stage3_npu_norm_transform_is_fail_closed(self):
        transformed = self.source.transform_zero_npu_stage3_norm(ZERO_STAGE3)
        self.assertEqual(transformed.count(".float()"), 5)
        self.assertNotIn(".double()", transformed)
        with self.assertRaisesRegex(self.source.UnsupportedAdapterVersion, "anchors changed"):
            self.source.transform_zero_npu_stage3_norm(
                ZERO_STAGE3.replace(".double()", ".float()", 1))

    def npu_scope_fixture(self, info):
        torch = types.ModuleType("torch")
        torch.float32 = "fp32"
        torch.bool = "bool"
        torch.npu = types.SimpleNamespace(current_device=lambda: 0)

        class Module:
            def named_parameters(self):
                return [("weight", types.SimpleNamespace(dtype="fp32", device="npu:0"))]

            def named_buffers(self):
                return [("mask", types.SimpleNamespace(dtype="bool", device="npu:0")),
                        ("running", types.SimpleNamespace(dtype="fp32", device="npu:0"))]

        class AdamW:
            def __init__(self):
                self.fused = None
                self.param_groups = [{"params": []}]

        torch.nn = types.SimpleNamespace(Module=Module)
        torch.optim = types.SimpleNamespace(AdamW=AdamW)
        public = types.ModuleType("jittor.distributed")
        public.get_hccl_world_info = lambda: info
        sys.modules["torch"] = torch
        sys.modules["jittor.distributed"] = public
        return {"model": Module(), "optimizer": AdamW(), "config": {"zero_optimization": {"stage": 0}}}

    def test_scope_accepts_boolean_buffers_and_initialized_single_rank(self):
        arguments = self.npu_scope_fixture({"initialized": True, "rank": 0, "world_size": 1})
        self.scope.validate_engine("npu", arguments)

    def test_scope_accepts_two_rank_verified_zero_modes(self):
        for zero in ({"stage": 1}, {"stage": 1, "contiguous_gradients": False},
                     {"stage": 2}, {"stage": 2, "contiguous_gradients": False},
                     {"stage": 3}):
            arguments = self.npu_scope_fixture({"initialized": True, "rank": 0, "world_size": 2})
            arguments["config"]["zero_optimization"] = zero
            with self.subTest(zero=zero), mock.patch.dict(
                    os.environ, {"WORLD_SIZE": "2", "RANK": "0"}, clear=False):
                self.scope.validate_engine("npu", arguments)

    def test_scope_rejects_missing_or_multi_rank_real_communicator(self):
        for info in ({"initialized": False, "rank": None, "world_size": None},
                     {"initialized": True, "rank": 0, "world_size": 2}):
            arguments = self.npu_scope_fixture(info)
            with self.subTest(info=info), self.assertRaisesRegex(RuntimeError, "requested initialized HCCL WORLD"):
                self.scope.validate_engine("npu", arguments)

    def test_scope_accepts_normal_optimizer_defaults_repeatedly(self):
        arguments = self.npu_scope_fixture({"initialized": True, "rank": 0, "world_size": 1})
        for flag in ("foreach", "fused", "amsgrad", "capturable", "differentiable", "maximize"):
            arguments["optimizer"].param_groups[0][flag] = None if flag in ("foreach", "fused") else False
        self.scope.validate_engine("npu", arguments)
        self.scope.validate_engine("npu", arguments)

    def test_npu_provider_rejects_cpu_only_runtime(self):
        self.npu_scope_fixture({"initialized": False, "rank": None, "world_size": None})
        sys.modules["torch"].npu.is_available = lambda: False
        ds = types.ModuleType("deepspeed")
        ds.__path__ = []
        ds.__version__ = "0.17.6"
        parent = types.ModuleType("deepspeed.accelerator")
        parent.__path__ = []
        stock = types.ModuleType("deepspeed.accelerator.npu_accelerator")
        stock.NPU_Accelerator = type("NPU_Accelerator", (), {})
        for module in (ds, parent, stock):
            sys.modules[module.__name__] = module
        path = Path(self.activation.__file__).with_name("accelerator.py")
        spec = importlib.util.spec_from_file_location("jittor_adapters.deepspeed.accelerator", path)
        provider = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(provider)
        with self.assertRaisesRegex(RuntimeError, "real available NPU"):
            provider.create_accelerator("npu")

    def test_scope_rejects_unverified_optimizer_modes(self):
        for flag in ("foreach", "fused", "amsgrad", "capturable", "differentiable", "maximize"):
            arguments = self.npu_scope_fixture({"initialized": True, "rank": 0, "world_size": 1})
            arguments["optimizer"].param_groups[0][flag] = True
            with self.subTest(flag=flag), self.assertRaisesRegex(NotImplementedError, flag):
                self.scope.validate_engine("npu", arguments)
        arguments = self.npu_scope_fixture({"initialized": True, "rank": 0, "world_size": 1})
        arguments["optimizer"].fused = True
        with self.assertRaisesRegex(NotImplementedError, "fused"):
            self.scope.validate_engine("npu", arguments)

    def test_scope_rejects_later_deepspeed_device_selection(self):
        arguments = self.npu_scope_fixture({"initialized": True, "rank": 0, "world_size": 1})
        with mock.patch.dict(os.environ, {"LOCAL_RANK": "1"}):
            with self.assertRaisesRegex(NotImplementedError, "LOCAL_RANK"):
                self.scope.validate_engine("npu", arguments)
        arguments["args"] = types.SimpleNamespace(local_rank=1)
        with mock.patch.dict(os.environ, {"LOCAL_RANK": "0"}):
            with self.assertRaisesRegex(NotImplementedError, "local_rank"):
                self.scope.validate_engine("npu", arguments)

    def test_scope_rejects_parameter_on_cpu(self):
        arguments = self.npu_scope_fixture({"initialized": True, "rank": 0, "world_size": 1})
        arguments["model"].named_parameters = lambda: [
            ("weight", types.SimpleNamespace(dtype="fp32", device="cpu"))]
        with self.assertRaisesRegex(NotImplementedError, "parameter dtype/device"):
            self.scope.validate_engine("npu", arguments)

    def test_scope_rejects_every_unverified_configuration(self):
        self.scope.validate_config({"zero_optimization": {"stage": 0}, "fp16": {"enabled": False}})
        self.scope.validate_config({"zero_optimization": {"stage": 3}, "fp16": {"enabled": False}})
        rejected = [
            {"zero_optimization": {"stage": 3, "contiguous_gradients": False}},
            {"zero_optimization": {"stage": 3, "offload_optimizer": {}}},
            {"zero_optimization": {"stage": 1, "offload_optimizer": {}}},
            {"zero_optimization": {"stage": 1, "contiguous_gradients": "yes"}},
            {"fp16": {"enabled": True}},
            {"bf16": {"enabled": True}}, {"activation_checkpointing": {}},
            {"hybrid_engine": {"enabled": True}}, {"flops_profiler": {}},
            {"timers": {}}, {"optimizer": {"type": "AdamW"}},
            {"wall_clock_breakdown": True}, {"memory_breakdown": True},
            {"gradient_accumulation_steps": 2}, {"gradient_clipping": 1},
            {"zero_optimization": {"stage": 0, "offload_optimizer": {}}},
        ]
        for config in rejected:
            with self.subTest(config=config), self.assertRaises(NotImplementedError):
                self.scope.validate_config(config)


if __name__ == "__main__":
    unittest.main()
