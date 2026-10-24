"""Version and import-guard contracts, using only controlled fake packages."""
import builtins
import importlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "adapters"))
from jittor_adapters import transformers, torchmetrics
from jittor_adapters._common import UnsupportedAdapterVersion, require_version

# Load the actual registration mechanism without importing the native runtime.
native = types.ModuleType("jittor")
native.__path__ = []
compat = types.ModuleType("_adapter_tests_compat")
compat.__path__ = [str(ROOT / "compat")]
sys.modules[compat.__name__] = compat
patcher = importlib.import_module(compat.__name__ + ".module_patcher")


class AdapterContracts(unittest.TestCase):
    def setUp(self):
        self.modules = mock.patch.dict(sys.modules, {
            "torch": types.ModuleType("torch"),
            "jittor": native, "jittor.compat": compat,
            "jittor.compat.module_patcher": patcher,
        })
        self.modules.start()
        # These cases build *fake* `transformers`/`torchmetrics` packages, and a
        # real one already in `sys.modules` wins over the fake however the temp
        # directory is put on `sys.path`: `importlib.import_module` returns the
        # cached module. Whether one is there depends on which other file the
        # session collected first -- measured, a module that imports the real
        # package at collection time made `..._npu_probe_rejects_real_pytorch_extension`
        # fail with `module transformers has no attribute probe_result` and
        # `..._unsupported_transformers_version_fails_real_import` fail with
        # `UnsupportedAdapterVersion not raised`, while the same file ran green on
        # its own. Evicted for the duration; `self.modules.stop()` restores them.
        for name in tuple(sys.modules):
            if name in ("transformers", "torchmetrics", "accelerate") or name.startswith(
                    ("transformers.", "torchmetrics.", "accelerate.")):
                sys.modules.pop(name)
        self.import_function = builtins.__import__
        self.saved_registry = dict(patcher._REGISTRY)
        patcher._REGISTRY.clear()

    def tearDown(self):
        patcher.uninstall_module_patches()
        patcher._REGISTRY.clear()
        patcher._REGISTRY.update(self.saved_registry)
        for name in tuple(sys.modules):
            if name in ("transformers", "torchmetrics", "accelerate") or name.startswith(("transformers.", "torchmetrics.", "accelerate.")):
                sys.modules.pop(name)
        self.assertIs(builtins.__import__, self.import_function)
        self.modules.stop()

    def fake_transformers(self, root, version):
        package = Path(root) / "transformers"
        (package / "utils").mkdir(parents=True)
        (package / "__init__.py").write_text(
            "__version__ = %r\nfrom .utils.import_utils import (is_torch_npu_available, "
            "is_torchvision_available, is_torchvision_v2_available)\n"
            "probe_result = is_torch_npu_available(check_device=True)\n" % version)
        (package / "utils/__init__.py").write_text("")
        (package / "utils/import_utils.py").write_text(
            "def is_torch_npu_available(check_device=False):\n"
            "    raise RuntimeError('native torch_npu probe must not execute')\n"
            "def is_torchvision_available():\n"
            "    return False\n"
            "def is_torchvision_v2_available():\n"
            "    raise RuntimeError('unavailable torchvision version must not be parsed')\n"
            "BACKENDS_MAPPING = {'torchvision': (is_torchvision_available, 'torchvision unavailable')}\n")

    #: The shape transformers 5.x ships: the version is assigned while the
    #: package executes, then the module is handed to a ``_LazyModule`` and the
    #: namespace entry is dropped. Once the import is over the value is only
    #: reachable through ``__getattr__`` -- which is the state that was read.
    _LAZY_INIT = '''
import sys

__version__ = %r

from .utils.import_utils import is_torch_npu_available


class _LazyModule(type(sys)):
    _version = %r

    def __getattr__(self, name):
        if name == "__version__":
            return type(self)._version
        raise AttributeError(name)


sys.modules[__name__].__class__ = _LazyModule
del __version__
'''

    def fake_lazy_transformers(self, root, version):
        self.fake_transformers(root, version)
        (Path(root) / "transformers" / "__init__.py").write_text(
            self._LAZY_INIT % (version, version))

    def test_a_version_reachable_only_through_getattr_is_accepted(self):
        """The version lookup used to read the namespace dict, not the module.

        ``require_version`` did ``vars(module).get("__version__")``. Transformers
        5.x ships the package as a ``_LazyModule`` that carries the version in
        ``extra_objects`` and drops the namespace entry, so once the package has
        been imported that lookup reports ``None`` and the adapter rejects
        5.5.3 -- a version ``SUPPORTED_VERSIONS`` lists. The rejection is a
        ``@required_patch``, so it escapes ``install_module_patches`` and took
        five tests in ``compat/tests/torch/test_compat_mechanisms.py`` with it.

        The precondition asserted below is what gives this case teeth: if the
        fake ever keeps ``__version__`` in the dict, the case stops modelling
        the shape that was broken and the check below passes for the wrong
        reason.
        """
        with tempfile.TemporaryDirectory() as root:
            self.fake_lazy_transformers(root, "5.5.3")
            sys.path.insert(0, root)
            try:
                module = importlib.import_module("transformers")
                self.assertNotIn("__version__", vars(module))
                self.assertEqual(module.__version__, "5.5.3")
                self.assertEqual(
                    require_version("transformers", transformers.SUPPORTED_VERSIONS),
                    "5.5.3")
            finally:
                sys.path.remove(root)

    def _fake_trainer_policy_fixture(self, version="4.56.2"):
        package = types.ModuleType("transformers")
        package.__version__ = version
        torch_module = types.ModuleType("torch")
        torch_module.float16 = object()
        torch_module.bfloat16 = object()
        torch_module.float32 = object()

        class FakePolicy:
            def __init__(self, param_dtype=None, reduce_dtype=None, output_dtype=None,
                         buffer_dtype=None, cast_forward_inputs=True):
                self.param_dtype = param_dtype
                self.reduce_dtype = reduce_dtype
                self.output_dtype = output_dtype
                self.buffer_dtype = buffer_dtype
                self.cast_forward_inputs = cast_forward_inputs

        class FakePlugin:
            fsdp_version = 2

            def __init__(self):
                self.mixed_precision_policy = FakePolicy()
                self.validated = False

            def validate_mixed_precision_policy(self):
                if not isinstance(self.mixed_precision_policy, FakePolicy):
                    raise TypeError("unexpected policy type")
                self.validated = True

        class FakeTrainer:
            def __init__(self, args, plugin):
                self.args = args
                self.accelerator = types.SimpleNamespace(
                    state=types.SimpleNamespace(fsdp_plugin=plugin))

        torch_module.distributed = types.SimpleNamespace(fsdp=types.SimpleNamespace(
            MixedPrecisionPolicy=FakePolicy, MixedPrecision=FakePolicy))
        module = types.ModuleType("transformers.trainer")
        module.Trainer = FakeTrainer
        module.torch = torch_module
        return package, module, torch_module, FakePolicy, FakePlugin

    def test_trainer_applies_fsdp2_precision_policy_from_training_args(self):
        package, module, torch_module, policy_type, plugin_type = self._fake_trainer_policy_fixture()
        plugin = plugin_type()
        with mock.patch.dict(sys.modules, {"transformers": package, "torch": torch_module}):
            self.assertTrue(transformers.patch_trainer(module))
            module.Trainer(
                args=types.SimpleNamespace(fsdp_config={"mixed_precision_policy": {
                    "param_dtype": "bf16", "reduce_dtype": "fp32", "output_dtype": "bf16",
                }}),
                plugin=plugin,
            )
        policy = plugin.mixed_precision_policy
        self.assertIsInstance(policy, policy_type)
        self.assertIs(policy.param_dtype, torch_module.bfloat16)
        self.assertIs(policy.reduce_dtype, torch_module.float32)
        self.assertIs(policy.output_dtype, torch_module.bfloat16)
        self.assertTrue(plugin.validated)

    def test_trainer_leaves_accelerate_policy_unchanged_without_fsdp_policy_config(self):
        package, module, torch_module, _, plugin_type = self._fake_trainer_policy_fixture()
        plugin = plugin_type()
        original = plugin.mixed_precision_policy
        with mock.patch.dict(sys.modules, {"transformers": package, "torch": torch_module}):
            self.assertTrue(transformers.patch_trainer(module))
            module.Trainer(args=types.SimpleNamespace(fsdp_config={}), plugin=plugin)
        self.assertIs(plugin.mixed_precision_policy, original)
        self.assertFalse(plugin.validated)

    def test_trainer_rejects_unknown_fsdp_policy_dtype_without_mutating_plugin(self):
        package, module, torch_module, _, plugin_type = self._fake_trainer_policy_fixture()
        plugin = plugin_type()
        original = plugin.mixed_precision_policy
        with mock.patch.dict(sys.modules, {"transformers": package, "torch": torch_module}):
            self.assertTrue(transformers.patch_trainer(module))
            with self.assertRaisesRegex(ValueError, "unsupported FSDP mixed precision dtype"):
                module.Trainer(args=types.SimpleNamespace(fsdp_config={
                    "mixed_precision_policy": {"reduce_dtype": "tf32"},
                }), plugin=plugin)
        self.assertIs(plugin.mixed_precision_policy, original)

    def test_trainer_policy_bridge_leaves_transformers_five_unchanged(self):
        package, module, torch_module, _, _ = self._fake_trainer_policy_fixture("5.5.3")
        original = module.Trainer.__init__
        with mock.patch.dict(sys.modules, {"transformers": package, "torch": torch_module}):
            self.assertFalse(transformers.patch_trainer(module))
        self.assertIs(module.Trainer.__init__, original)

    def test_transformers_npu_probe_rejects_real_pytorch_extension(self):
        transformers.register(patcher.register_module_patch)
        patcher.install_module_patches(load_entry_points=False)
        with tempfile.TemporaryDirectory() as root:
            self.fake_transformers(root, "4.56.2")
            sys.path.insert(0, root)
            try:
                module = importlib.import_module("transformers")
                self.assertFalse(module.probe_result)
                guard = sys.modules["transformers.utils.import_utils"].is_torch_npu_available
                self.assertFalse(guard())
                self.assertIs(module.is_torch_npu_available, guard)
                self.assertTrue(callable(guard.cache_clear))
            finally:
                sys.path.remove(root)

    def test_transformers_npu_probe_reads_public_frontend_capability(self):
        transformers.register(patcher.register_module_patch)
        patcher.install_module_patches(load_entry_points=False)
        frontend = types.ModuleType("torch")
        frontend.npu = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(sys.modules, {"torch": frontend}):
            self.fake_transformers(root, "4.56.2")
            sys.path.insert(0, root)
            try:
                module = importlib.import_module("transformers")
                self.assertTrue(module.probe_result)
                guard = module.is_torch_npu_available
                self.assertTrue(guard())
                frontend.npu.is_available = lambda: False
                guard.cache_clear()
                self.assertFalse(guard(check_device=True))
            finally:
                sys.path.remove(root)

    def fake_flash_attention_utils(self, root):
        package = Path(root) / "transformers"
        (package / "integrations").mkdir()
        (package / "integrations/__init__.py").write_text("")
        (package / "integrations/npu_flash_attention.py").write_text(
            "from torch_npu import npu_fusion_attention\n")
        (package / "modeling_flash_attention_utils.py").write_text(
            "from .utils.import_utils import is_torch_npu_available\n"
            "flags = {'fa2': False, 'fa3': False, 'new_fa2': False}\n"
            "def is_flash_attn_2_available(): return flags['fa2']\n"
            "def is_flash_attn_3_available(): return flags['fa3']\n"
            "def is_flash_attn_greater_or_equal_2_10(): return flags['new_fa2']\n"
            "def flash_attn_supports_top_left_mask():\n"
            "    if is_flash_attn_3_available(): return False\n"
            "    if is_flash_attn_2_available(): return not is_flash_attn_greater_or_equal_2_10()\n"
            "    from .integrations.npu_flash_attention import is_npu_fa2_top_left_aligned_causal_mask\n"
            "    return is_npu_fa2_top_left_aligned_causal_mask()\n"
            "def is_flash_attn_available():\n"
            "    return is_flash_attn_3_available() or is_flash_attn_2_available() or is_torch_npu_available()\n"
            "def _lazy_imports(implementation):\n"
            "    if is_torch_npu_available():\n"
            "        from .integrations.npu_flash_attention import npu_flash_attn_func\n"
            "    raise ValueError('Could not find the currently requested flash attention implementation')\n")

    def test_private_flash_provider_does_not_advertise_native_torch_extension(self):
        transformers.register(patcher.register_module_patch)
        patcher.install_module_patches(load_entry_points=False)
        frontend = types.ModuleType("torch")
        frontend.npu = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(sys.modules, {"torch": frontend}):
            self.fake_transformers(root, "4.56.2")
            self.fake_flash_attention_utils(root)
            sys.path.insert(0, root)
            attempted = []
            original_import = builtins.__import__
            def import_without_extension(name, *args, **kwargs):
                if name == "torch_npu" or name.startswith("torch_npu."):
                    attempted.append(name)
                    raise AssertionError("native extension must not be imported")
                return original_import(name, *args, **kwargs)
            try:
                with mock.patch("builtins.__import__", side_effect=import_without_extension):
                    package = importlib.import_module("transformers")
                    module = importlib.import_module("transformers.modeling_flash_attention_utils")
                    self.assertTrue(package.is_torch_npu_available(check_device=True))
                    self.assertFalse(module.is_torch_npu_available())
                    self.assertFalse(module.is_flash_attn_available())
                    self.assertFalse(module.flash_attn_supports_top_left_mask())
                    for implementation in (None, "flash_attention_2"):
                        with self.assertRaisesRegex(ValueError, "requested flash attention"):
                            module._lazy_imports(implementation)
                    module.flags['fa2'] = True
                    self.assertTrue(module.flash_attn_supports_top_left_mask())
                    self.assertTrue(module.is_flash_attn_available())
                    module.flags['new_fa2'] = True
                    self.assertFalse(module.flash_attn_supports_top_left_mask())
                    module.flags['fa3'] = True
                    module.flags['new_fa2'] = False
                    self.assertFalse(module.flash_attn_supports_top_left_mask())
                    self.assertFalse(transformers.patch_flash_attention_utils(module))
                    self.assertEqual(attempted, [])
                    self.assertNotIn("transformers.integrations.npu_flash_attention", sys.modules)
                    self.assertNotIn("torch_npu", sys.modules)
            finally:
                sys.path.remove(root)

    def test_private_flash_provider_rejects_changed_owned_contract(self):
        package = types.ModuleType("transformers")
        package.__version__ = "4.56.2"
        with mock.patch.dict(sys.modules, {"transformers": package}):
            with self.assertRaisesRegex(UnsupportedAdapterVersion, "FlashAttention"):
                transformers.patch_flash_attention_utils(types.ModuleType("changed"))

    def test_private_flash_provider_leaves_existing_five_x_contract_unchanged(self):
        package = types.ModuleType("transformers")
        package.__version__ = "5.5.3"
        module = types.ModuleType("unverified_flash")
        before = dict(vars(module))
        with mock.patch.dict(sys.modules, {"transformers": package}):
            self.assertFalse(transformers.patch_flash_attention_utils(module))
            self.assertEqual(vars(module), before)

    def fake_accelerate(self, root, version):
        package = Path(root) / "accelerate"
        (package / "utils").mkdir(parents=True)
        (package / "__init__.py").write_text(
            "__version__ = %r\nfrom .utils import imports, modeling\n"
            "from . import state, accelerator\n"
            "from .utils.imports import is_npu_available\n" % version)
        (package / "utils/__init__.py").write_text("")
        (package / "utils/imports.py").write_text(
            "def is_npu_available(check_device=False):\n"
            "    raise RuntimeError('native torch_npu probe must not execute')\n")
        code = ("from {prefix}utils.imports import is_npu_available\n"
                "probe_during_import = is_npu_available(check_device=False)\n"
                "if probe_during_import:\n"
                "    raise RuntimeError('native extension import branch must not execute')\n")
        (package / "utils/modeling.py").write_text(code.format(prefix=".."))
        for name in ("state", "accelerator"):
            (package / (name + ".py")).write_text(code.format(prefix="."))

    def test_accelerate_bootstrap_skips_native_extension_then_uses_frontend_npu(self):
        transformers.register(patcher.register_module_patch)
        patcher.install_module_patches(load_entry_points=False)
        frontend = types.ModuleType("torch")
        frontend.npu = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(sys.modules, {"torch": frontend}):
            self.fake_accelerate(root, "1.10.1")
            sys.path.insert(0, root)
            try:
                module = importlib.import_module("accelerate")
                self.assertTrue(module.is_npu_available(check_device=True))
                for path in ("accelerate.state", "accelerate.accelerator", "accelerate.utils.modeling"):
                    consumer = sys.modules[path]
                    self.assertFalse(consumer.probe_during_import)
                    self.assertIs(consumer.is_npu_available, module.is_npu_available)
                    self.assertTrue(consumer.is_npu_available())
                frontend.npu.is_available = lambda: False
                module.is_npu_available.cache_clear()
                self.assertFalse(module.is_npu_available())
                frontend.npu.is_available = lambda: True
                frontend.npu.device_count = lambda: 0
                module.is_npu_available.cache_clear()
                self.assertFalse(module.is_npu_available(check_device=True))
                self.assertNotIn("torch_npu", sys.modules)
            finally:
                sys.path.remove(root)

    def test_frontend_npu_probe_does_not_hide_backend_failure(self):
        frontend = types.ModuleType("torch")

        def broken():
            raise RuntimeError("broken ACL discovery")

        frontend.npu = types.SimpleNamespace(is_available=broken, device_count=lambda: 1)
        with mock.patch.dict(sys.modules, {"torch": frontend}):
            with self.assertRaisesRegex(RuntimeError, "broken ACL discovery"):
                transformers._frontend_npu_available(check_device=True)

    def test_accelerate_adapter_rejects_unverified_version(self):
        transformers.register(patcher.register_module_patch)
        patcher.install_module_patches(load_entry_points=False)
        with tempfile.TemporaryDirectory() as root:
            self.fake_accelerate(root, "99.0.0")
            sys.path.insert(0, root)
            try:
                with self.assertRaisesRegex(UnsupportedAdapterVersion, "99.0.0"):
                    importlib.import_module("accelerate")
            finally:
                sys.path.remove(root)

    def _fake_fsdp_optimizer_fixture(self):
        package = types.ModuleType("accelerate")
        package.__version__ = "1.10.1"
        module = types.ModuleType("accelerate.utils.fsdp_utils")
        options, model, optimizer = object(), object(), object()
        events = []

        def original(fsdp_plugin, accelerator, optimizer, model, input_dir,
                     optimizer_index=0, adapter_only=False):
            events.append(("original", fsdp_plugin.fsdp_version, input_dir,
                           optimizer_index, adapter_only))
            return "original result"

        module.load_fsdp_optimizer = original
        module.OPTIMIZER_NAME = "optimizer"
        module._prepare_sd_options = lambda fsdp_plugin: options
        fsdp = types.ModuleType("torch.distributed.fsdp")
        fsdp.StateDictType = types.SimpleNamespace(FULL_STATE_DICT=object(), SHARDED_STATE_DICT=object())
        checkpoint = types.ModuleType("torch.distributed.checkpoint")
        state_dict = types.ModuleType("torch.distributed.checkpoint.state_dict")

        def get_optimizer_state_dict(actual_model, actual_optimizer, *, options):
            self.assertIs(actual_model, model)
            self.assertIs(actual_optimizer, optimizer)
            events.append(("get", options))
            return {"state": {"lora.weight": {"step": 0, "exp_avg": 0}}}

        def load(state, *, checkpoint_id, storage_reader):
            events.append(("load", checkpoint_id, storage_reader))
            # DCP requests only template keys. A fresh optimizer.state_dict()
            # has no state keys, so the former implementation omitted these.
            self.assertIn("lora.weight", state["optimizer"]["state"])
            state["optimizer"]["state"]["lora.weight"] = {"step": 7, "exp_avg": 13}

        def set_optimizer_state_dict(actual_model, actual_optimizer, state, *, options):
            self.assertIs(actual_model, model)
            self.assertIs(actual_optimizer, optimizer)
            events.append(("set", state, options))

        state_dict.get_optimizer_state_dict = get_optimizer_state_dict
        state_dict.set_optimizer_state_dict = set_optimizer_state_dict
        checkpoint.load = load
        checkpoint.FileSystemReader = lambda path: ("reader", path)
        distributed = types.ModuleType("torch.distributed")
        distributed.checkpoint = checkpoint
        distributed.fsdp = fsdp
        frontend = types.ModuleType("torch")
        frontend.distributed = distributed
        modules = {"accelerate": package, "accelerate.utils.fsdp_utils": module,
                   "torch": frontend, "torch.distributed": distributed,
                   "torch.distributed.fsdp": fsdp, "torch.distributed.checkpoint": checkpoint,
                   "torch.distributed.checkpoint.state_dict": state_dict}
        plugin = types.SimpleNamespace(fsdp_version=2, state_dict_type=fsdp.StateDictType.SHARDED_STATE_DICT)
        accelerator = types.SimpleNamespace(wait_for_everyone=lambda: events.append(("barrier",)))
        return modules, module, plugin, accelerator, model, optimizer, options, events

    def test_fsdp2_sharded_optimizer_restore_loads_saved_fqn_state(self):
        modules, module, plugin, accelerator, model, optimizer, options, events = self._fake_fsdp_optimizer_fixture()
        alias = types.ModuleType("transformers.trainer")
        original = module.load_fsdp_optimizer
        alias.load_fsdp_optimizer = original
        modules[alias.__name__] = alias
        with mock.patch.dict(sys.modules, modules):
            patcher.register_module_patch("accelerate.utils.fsdp_utils", transformers.patch_accelerate_fsdp_utils)
            patcher.install_module_patches(load_entry_points=False)
            self.assertIsNot(module.load_fsdp_optimizer, original)
            self.assertIs(alias.load_fsdp_optimizer, module.load_fsdp_optimizer)
            self.assertFalse(transformers.patch_accelerate_fsdp_utils(module))
            module.load_fsdp_optimizer(plugin, accelerator, optimizer, model, "/checkpoints/checkpoint-1",
                                       optimizer_index=2, adapter_only=True)
            self.assertEqual([event[0] for event in events], ["barrier", "get", "load", "set"])
            self.assertEqual(events[2], ("load", "/checkpoints/checkpoint-1/optimizer_2",
                                         ("reader", "/checkpoints/checkpoint-1/optimizer_2")))
            self.assertIs(events[1][1], options)
            self.assertEqual(events[-1][1], {"state": {"lora.weight": {"step": 7, "exp_avg": 13}}})
            self.assertIs(events[-1][2], options)
            patcher.uninstall_module_patches()
            self.assertIs(module.load_fsdp_optimizer, original)
            self.assertIs(alias.load_fsdp_optimizer, original)

    def test_fsdp2_optimizer_restore_accepts_an_optimizer_directory(self):
        fixture = self._fake_fsdp_optimizer_fixture()
        modules, module, plugin, accelerator, model, optimizer, _, events = fixture
        with mock.patch.dict(sys.modules, modules):
            transformers.patch_accelerate_fsdp_utils(module)
            directory = Path("/checkpoints/checkpoint-1/optimizer_3")
            module.load_fsdp_optimizer(plugin, accelerator, optimizer, model, directory, optimizer_index=3)
            self.assertEqual(events[2], ("load", directory, ("reader", directory)))

    def test_fsdp_optimizer_restore_preserves_fsdp1_and_full_state_loaders(self):
        fixture = self._fake_fsdp_optimizer_fixture()
        modules, module, plugin, accelerator, model, optimizer, _, events = fixture
        with mock.patch.dict(sys.modules, modules):
            transformers.patch_accelerate_fsdp_utils(module)
            plugin.fsdp_version = 1
            self.assertEqual(module.load_fsdp_optimizer(plugin, accelerator, optimizer, model, "/checkpoint",
                                                        optimizer_index=4, adapter_only=True), "original result")
            plugin.fsdp_version = 2
            plugin.state_dict_type = modules["torch.distributed.fsdp"].StateDictType.FULL_STATE_DICT
            self.assertEqual(module.load_fsdp_optimizer(plugin, accelerator, optimizer, model, "/checkpoint",
                                                        optimizer_index=5), "original result")
            self.assertEqual(events, [("original", 1, "/checkpoint", 4, True),
                                      ("original", 2, "/checkpoint", 5, False)])

    def test_fsdp_optimizer_restore_delegates_local_and_unknown_state_types(self):
        fixture = self._fake_fsdp_optimizer_fixture()
        modules, module, plugin, accelerator, model, optimizer, _, events = fixture
        with mock.patch.dict(sys.modules, modules):
            transformers.patch_accelerate_fsdp_utils(module)
            for state_type in ("LOCAL_STATE_DICT", object()):
                plugin.state_dict_type = state_type
                self.assertEqual(module.load_fsdp_optimizer(plugin, accelerator, optimizer, model, "/checkpoint"),
                                 "original result")
            self.assertEqual(events, [("original", 2, "/checkpoint", 0, False)] * 2)

    def test_fsdp_optimizer_restore_rejects_unverified_contract_and_version(self):
        fixture = self._fake_fsdp_optimizer_fixture()
        modules, module, *_ = fixture
        original = module.load_fsdp_optimizer
        with mock.patch.dict(sys.modules, modules):
            module._prepare_sd_options = lambda: None
            with self.assertRaisesRegex(UnsupportedAdapterVersion, "checkpoint contract changed"):
                transformers.patch_accelerate_fsdp_utils(module)
            self.assertIs(module.load_fsdp_optimizer, original)
            modules["accelerate"].__version__ = "1.11.0"
            with self.assertRaisesRegex(UnsupportedAdapterVersion, "1.11.0"):
                transformers.patch_accelerate_fsdp_utils(module)
            self.assertIs(module.load_fsdp_optimizer, original)

    def test_transformers_recognizes_the_deployed_torchvision_facade(self):
        transformers.register(patcher.register_module_patch)
        patcher.install_module_patches(load_entry_points=False)
        with tempfile.TemporaryDirectory() as root:
            self.fake_transformers(root, "4.56.2")
            facade = Path(root) / "torchvision"
            facade.mkdir()
            (facade / "__init__.py").write_text("__version__ = '2.11.0'\n")
            sys.path.insert(0, root)
            try:
                module = importlib.import_module("transformers")
                import_utils = sys.modules["transformers.utils.import_utils"]
                self.assertTrue(module.is_torchvision_available())
                self.assertTrue(module.is_torchvision_v2_available())
                self.assertIs(module.is_torchvision_available, import_utils.is_torchvision_available)
                self.assertIs(module.is_torchvision_v2_available, import_utils.is_torchvision_v2_available)
                self.assertIs(
                    import_utils.BACKENDS_MAPPING["torchvision"][0],
                    import_utils.is_torchvision_available,
                )
                self.assertTrue(import_utils.BACKENDS_MAPPING["torchvision"][0]())
            finally:
                sys.path.remove(root)

    def test_unsupported_transformers_version_fails_real_import(self):
        transformers.register(patcher.register_module_patch)
        patcher.install_module_patches(load_entry_points=False)
        with tempfile.TemporaryDirectory() as root:
            self.fake_transformers(root, "99.0.0")
            sys.path.insert(0, root)
            try:
                with self.assertRaisesRegex(UnsupportedAdapterVersion, "99.0.0"):
                    importlib.import_module("transformers")
                self.assertIn("99.0.0", patcher.last_module_patch_report().failures[0].detail)
            finally:
                sys.path.remove(root)

    def test_missing_adapter_is_reported_as_unavailable(self):
        with mock.patch.object(patcher, "_entry_points", return_value=()):
            report = patcher.install_module_patches(expected_entry_points=("jittor_transformers",))
        self.assertEqual([(entry.name, entry.status) for entry in report.results],
                         [("jittor_transformers", "unavailable")])

    def test_torchmetrics_version_guard_and_existing_bound_aliases(self):
        root = types.ModuleType("torchmetrics")
        root.__version__ = "1.7.4"
        data = types.ModuleType("torchmetrics.utilities.data")
        alias = types.ModuleType("torchmetrics.functional.classification")
        original = lambda value, minlength=None: ("original", minlength)
        data._bincount = alias._bincount = original
        data.dim_zero_cat = lambda value: value
        sys.modules.update({module.__name__: module for module in (root, data, alias)})
        torchmetrics.register(patcher.register_module_patch)
        patcher.install_module_patches(load_entry_points=False)
        self.assertIs(alias._bincount, data._bincount)
        self.assertIsNot(data._bincount, original)
        self.assertEqual(data._bincount(object()), ("original", None))
        root.__version__ = "2.0.0"
        with self.assertRaisesRegex(UnsupportedAdapterVersion, "2.0.0"):
            torchmetrics.patch_data(data)


if __name__ == "__main__":
    unittest.main()
