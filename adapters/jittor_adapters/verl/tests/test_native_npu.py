"""Opt-in private policy routing/lifecycle only; no tensor or hardware evidence."""
import hashlib
from contextlib import ExitStack
import importlib
import inspect
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import test_adapter as lifecycle
from jittor_adapters import verl
from jittor_adapters.verl import native_npu as adapter
from jittor_adapters.verl import device_probe, optimizer_offload
from jittor_adapters._common import UnsupportedAdapterVersion


FIXTURE = Path(__file__).with_name("_npu_initializer_fixture.py")
HF_FIXTURE = Path(__file__).with_name("_qwen3_source_fixture.py")
patcher, transaction = lifecycle.patcher, lifecycle.transaction


class Qwen3Config:
    model_type, _attn_implementation = "qwen3", "eager"


class HFModelConfig:
    def __init__(self):
        self.hf_config = Qwen3Config()
        self.model_type = "language_model"
        for name in ("use_liger", "use_fused_kernels", "use_remove_padding", "trust_remote_code",
                     "enable_activation_offload", "enable_gradient_checkpointing"):
            setattr(self, name, False)
        self.lora_rank, self.lora_adapter_path = 0, None
        # verl 0.9.1 composes this disabled Megatron block even for FSDP.
        self.lora = {
            "rank": 0,
            "adapter_path": None,
            "merge": False,
            "target_modules": ["linear_qkv", "linear_proj"],
        }
        self.external_lib = None
        self.tiled_mlp = {"enabled": False}
        self.mtp = types.SimpleNamespace(enable=False, enable_train=False, enable_rollout=False)

    def get(self, name, default=None):
        return getattr(self, name, default)


class TestVerlNativeNpuPolicy(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.dict(os.environ, {adapter.OPT_IN: "1"}))
        root = types.ModuleType("verl")
        root.__version__, root.__path__ = "0.9.1", []
        hf_root = types.ModuleType("transformers")
        hf_root.__version__, hf_root.Qwen3Config = "5.9.0", Qwen3Config
        config = types.ModuleType("verl.workers.config")
        config.HFModelConfig = HFModelConfig
        self.module = types.ModuleType(adapter.MODULE)
        self.module.__file__, self.module.is_npu_available = str(FIXTURE), True
        exec(compile(FIXTURE.read_text(), str(FIXTURE), "exec"), vars(self.module))
        self.original = self.module.apply_npu_fsdp_patches
        self.alias = types.ModuleType("verl.workers.alias_npu_fixture")
        self.alias.apply_npu_fsdp_patches = self.original
        self.hf = types.ModuleType(adapter.HF_MODULE)
        self.hf.__file__ = str(HF_FIXTURE)
        exec(compile(HF_FIXTURE.read_text(), str(HF_FIXTURE), "exec"), vars(self.hf))
        self.functions = self.hf.Qwen3RMSNorm.forward, self.hf.Qwen3MLP.forward, self.hf.apply_rotary_pos_emb
        self.frontend = types.ModuleType("torch")
        self.frontend.npu = types.SimpleNamespace(is_available=lambda: True)
        self.frontend.__torch_version__ = "2.11.0"
        self.frontend.compat_report_torch_api_version = lambda enable=True: self.frontend.__torch_version__
        public = types.ModuleType("jittor.compat.torch")
        self.owner = object()
        public.namespace_owner = lambda module: self.owner if module is self.frontend else None
        self.stack.enter_context(mock.patch.dict(sys.modules, {
            "verl": root, "transformers": hf_root, config.__name__: config,
            self.module.__name__: self.module, self.alias.__name__: self.alias,
            self.hf.__name__: self.hf, "torch": self.frontend,
            "jittor": types.ModuleType("jittor"), "jittor.compat": lifecycle.compat,
            "jittor.compat.module_patcher": patcher, "jittor.compat.torch": public,
        }))
        self.saved_registry = dict(patcher._REGISTRY)
        patcher._REGISTRY.clear()
        # Fixtures deliberately omit real tensor/model code. Only file/HF
        # fixture digests differ; the exact upstream initializer function hash
        # is checked unchanged. Real package hashes remain pinned in production.
        self.stack.enter_context(mock.patch.object(adapter, "SOURCE_SHA256", hashlib.sha256(FIXTURE.read_bytes()).hexdigest()))
        self.stack.enter_context(mock.patch.object(adapter, "HF_SOURCE_SHA256", hashlib.sha256(HF_FIXTURE.read_bytes()).hexdigest()))
        self.stack.enter_context(mock.patch.object(adapter, "HF_FUNCTIONS", {
            name: hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
            for name, function in zip(adapter.HF_FUNCTIONS, self.functions)}))
        self.addCleanup(self.cleanup)

    def cleanup(self):
        patcher.uninstall_module_patches()
        patcher._REGISTRY.clear()
        patcher._REGISTRY.update(self.saved_registry)
        self.stack.close()

    def install(self):
        adapter.register(patcher.register_module_patch)
        return patcher.install_module_patches(load_entry_points=False)

    def test_original_requires_vendor_import_policy_keeps_exact_hf_functions(self):
        with self.assertRaises(ModuleNotFoundError):
            self.original(HFModelConfig())
        before = set(sys.modules)
        self.install()
        replacement = self.module.apply_npu_fsdp_patches
        self.assertEqual(inspect.signature(replacement), inspect.signature(self.original))
        self.assertEqual(replacement.__module__, "jittor_adapters.verl.native_npu")
        self.assertEqual(replacement.__name__, "apply_native_npu_fsdp_policy")
        self.assertEqual(replacement._jittor_verl_native_npu_policy, adapter.POLICY)
        self.assertIs(replacement._jittor_verl_original, self.original)
        self.assertIs(self.alias.apply_npu_fsdp_patches, replacement)
        for role in ("language_model", "value_model"):
            config = HFModelConfig()
            config.model_type = role
            self.assertIsNone(replacement(config))
        self.assertEqual(self.functions, (self.hf.Qwen3RMSNorm.forward, self.hf.Qwen3MLP.forward,
                                         self.hf.apply_rotary_pos_emb))
        self.assertNotIn("torch_npu", set(sys.modules) - before)
        self.assertNotIn("verl.models.transformers.npu_patch", sys.modules)

    def test_registration_is_explicit_and_does_not_import_upstream(self):
        for value in ("", "0"):
            for jagged in (None, "0"):
                environment = {adapter.OPT_IN: value}
                if jagged is not None:
                    environment["VERL_CORRECT_JAGGED_POSITIONS"] = jagged
                with self.subTest(native_npu=value, jagged=jagged):
                    with mock.patch.dict(os.environ, environment, clear=True):
                        callbacks = []
                        verl.register(lambda *args: callbacks.append(args))
                        # Optimizer reconciliation is registered whenever verl
                        # loads; its bridge remains an explicit runtime check.
                        self.assertEqual(callbacks, [
                            ("verl.utils.device", device_probe.patch_deferred_device_probe),
                            ("verl", device_probe.patch_root_device_probe),
                            ("verl.utils.fsdp_utils",
                             optimizer_offload.patch_deferred_optimizer_offload),
                            ("verl", optimizer_offload.patch_root_optimizer_offload),
                            ("verl", verl.patch_verl),
                        ])
                        self.assertNotIn(adapter.patch_native_npu_initializer,
                                         [callback for _, callback in callbacks])
        with mock.patch.dict(os.environ, clear=True):
            callbacks = []
            adapter.register(lambda *args: callbacks.append(args))
            self.assertEqual(callbacks, [])

    def test_torch_api_version_is_requested_from_public_owner(self):
        calls = []
        self.frontend.compat_report_torch_api_version = lambda enable=True: (
            calls.append(enable) or self.frontend.__torch_version__
        )
        self.assertTrue(adapter.report_torch_api_version())
        self.assertEqual(calls, [True])
        with mock.patch.dict(os.environ, {adapter.OPT_IN: "true"}):
            with self.assertRaises(UnsupportedAdapterVersion):
                adapter.register(lambda *args: self.fail("invalid option must not register"))
        callbacks = []
        verl.register(lambda *args: callbacks.append(args))
        self.assertIn((adapter.MODULE, adapter.patch_native_npu_initializer), callbacks)

    def test_uninstall_restores_helper_alias_and_reinstall_is_idempotent(self):
        self.install()
        first = self.module.apply_npu_fsdp_patches
        self.install()
        self.assertIs(self.module.apply_npu_fsdp_patches, first)
        patcher.uninstall_module_patches()
        self.assertIs(self.module.apply_npu_fsdp_patches, self.original)
        self.assertIs(self.alias.apply_npu_fsdp_patches, self.original)
        self.install()
        self.assertIsNot(self.module.apply_npu_fsdp_patches, self.original)

    def test_external_replacement_is_preserved_on_uninstall(self):
        self.install()
        external = lambda model_config=None: "external"
        self.module.apply_npu_fsdp_patches = external
        with self.assertRaises(transaction.TransactionConflict):
            patcher.uninstall_module_patches()
        self.assertIs(self.module.apply_npu_fsdp_patches, external)

    def test_pins_source_and_external_function_changes_fail_before_mutation(self):
        mutations = ((sys.modules["verl"], "__version__", "0.9.2"),
                     (sys.modules["transformers"], "__version__", "5.9.1"),
                     (adapter, "SOURCE_SHA256", "0" * 64),
                     (adapter, "FUNCTION_SHA256", "0" * 64),
                     (adapter, "HF_SOURCE_SHA256", "0" * 64),
                     (self.module, "__file__", "/missing/verl-source.py"),
                     (self.module, "apply_npu_fsdp_patches", lambda model_config=None: None),
                     (self.hf.Qwen3MLP, "forward", lambda self, x: x))
        for owner, name, value in mutations:
            with self.subTest(name=name), mock.patch.object(owner, name, value):
                original = self.module.apply_npu_fsdp_patches
                with self.assertRaises(UnsupportedAdapterVersion):
                    self.install()
                self.assertIs(self.module.apply_npu_fsdp_patches, original)
                self.assertIs(self.alias.apply_npu_fsdp_patches, self.original)

    def test_native_oracle_unavailable_backend_and_vendor_modules_rejected(self):
        for change in (mock.patch.dict(sys.modules, {"torch": types.ModuleType("torch")}),
                       mock.patch.object(self.frontend.npu, "is_available", lambda: False),
                       mock.patch.dict(sys.modules, {"torch_npu": types.ModuleType("torch_npu")}),
                       mock.patch.dict(sys.modules, {"verl.models.transformers.npu_patch": types.ModuleType("vendor")})):
            with change, self.assertRaises(UnsupportedAdapterVersion):
                self.install()
            self.assertIs(self.module.apply_npu_fsdp_patches, self.original)

    def test_call_rechecks_opt_in_and_function_ownership(self):
        self.install()
        replacement = self.module.apply_npu_fsdp_patches
        with mock.patch.dict(os.environ, {adapter.OPT_IN: "0"}), self.assertRaises(UnsupportedAdapterVersion):
            replacement(HFModelConfig())
        with mock.patch.object(self.hf.Qwen3RMSNorm, "forward", lambda self, x: x):
            with self.assertRaises(UnsupportedAdapterVersion):
                replacement(HFModelConfig())

    def test_policy_loads_pinned_hf_module_independent_of_import_order(self):
        saved = sys.modules.pop(adapter.HF_MODULE)
        try:
            with mock.patch.object(adapter.importlib, "import_module", return_value=saved) as imported:
                self.install()
            imported.assert_called_once_with(adapter.HF_MODULE)
        finally:
            sys.modules[adapter.HF_MODULE] = saved

    def test_unverified_models_and_options_fail_closed(self):
        self.install()
        replacement = self.module.apply_npu_fsdp_patches
        for invalid in (None, {}, types.SimpleNamespace(hf_config=Qwen3Config())):
            with self.assertRaises(UnsupportedAdapterVersion):
                replacement(invalid)
        mutations = [lambda c: setattr(c, "hf_config", types.SimpleNamespace(model_type="qwen3")),
                     lambda c: setattr(c.hf_config, "model_type", "qwen3_moe"),
                     lambda c: setattr(c.hf_config, "_attn_implementation", "flash_attention_2"),
                     lambda c: setattr(c, "model_type", "embedding"),
                     lambda c: setattr(c.hf_config, "quantization_config", {}),
                     lambda c: setattr(c.hf_config, "num_experts", 2),
                     lambda c: setattr(c, "lora_rank", 4),
                     lambda c: setattr(c, "lora_adapter_path", "adapter"),
                     lambda c: c.lora.update(rank=8),
                     lambda c: c.lora.update(adapter_path="megatron-adapter"),
                     lambda c: c.lora.update(merge=True),
                     lambda c: setattr(c, "external_lib", "other_model"),
                     lambda c: c.tiled_mlp.update(enabled=True),
                     lambda c: setattr(c.mtp, "enable", True),
                     lambda c: setattr(c.mtp, "enable_train", True),
                     lambda c: setattr(c.mtp, "enable_rollout", True)]
        for name in ("use_liger", "use_fused_kernels", "use_remove_padding", "trust_remote_code",
                     "enable_activation_offload", "enable_gradient_checkpointing"):
            mutations.append(lambda c, name=name: setattr(c, name, True))
        for change in mutations:
            config = HFModelConfig()
            change(config)
            with self.assertRaises(UnsupportedAdapterVersion):
                replacement(config)

    def test_real_import_finder_patches_newly_imported_helper(self):
        # Exercise the shared finder/loader rather than calling our callback.
        with tempfile.TemporaryDirectory(prefix="verl-npu-policy-host-") as directory:
            package = Path(directory) / "fsdp"
            package.mkdir()
            source = "is_npu_available = True\n" + FIXTURE.read_text()
            (package / "utils.py").write_text(source)
            parent = types.ModuleType("verl.workers.engine.fsdp")
            parent.__path__ = [str(package)]
            with mock.patch.dict(sys.modules, {parent.__name__: parent}), mock.patch.object(
                    adapter, "SOURCE_SHA256", hashlib.sha256(source.encode()).hexdigest()):
                sys.modules.pop(adapter.MODULE)
                self.install()
                imported = importlib.import_module(adapter.MODULE)
                self.assertEqual(imported.apply_npu_fsdp_patches._jittor_verl_native_npu_policy, adapter.POLICY)
                original = imported.apply_npu_fsdp_patches._jittor_verl_original
                self.assertIsNone(imported.apply_npu_fsdp_patches(HFModelConfig()))
                patcher.uninstall_module_patches()
                self.assertIs(imported.apply_npu_fsdp_patches, original)


if __name__ == "__main__":
    unittest.main()
