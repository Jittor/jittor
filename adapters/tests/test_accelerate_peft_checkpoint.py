"""Host contracts for the pinned private Accelerate/PEFT SHARDED branch."""
import importlib
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "adapters"))
from jittor_adapters import _accelerate_peft as adapter
from jittor_adapters._common import UnsupportedAdapterVersion

compat = types.ModuleType("_peft_adapter_tests_compat")
compat.__path__ = [str(ROOT / "compat")]
sys.modules[compat.__name__] = compat
patcher = importlib.import_module(compat.__name__ + ".module_patcher")


class CheckpointValue:
    def __init__(self, label, offset=0, count=3):
        self.label = label
        self.checkpoint_metadata = {"shape": (3, 4), "offset": offset, "local_count": count}


class AcceleratePeftCheckpointContracts(unittest.TestCase):
    def setUp(self):
        self.saved_registry = dict(patcher._REGISTRY)
        patcher._REGISTRY.clear()
        self.events = []
        self.module = types.ModuleType("accelerate.utils.fsdp_utils")
        package = types.ModuleType("accelerate")
        package.__version__ = "1.10.1"
        peft = types.ModuleType("peft")
        peft.__version__ = "0.17.1"
        peft.PeftType = types.SimpleNamespace(LORA="LORA")
        self.model = types.SimpleNamespace(active_adapter="round2", peft_config={
            "round2": types.SimpleNamespace(peft_type="LORA", bias="none", use_dora=False)})
        self.frozen = CheckpointValue("frozen")
        self.a = CheckpointValue("a", offset=6, count=0)
        self.b = CheckpointValue("b", offset=0, count=6)
        self.canonical = {"base.weight": self.frozen, "q.lora_A.round2.weight": self.a,
                          "q.lora_B.round2.weight": self.b, "q.lora_A.other.weight": CheckpointValue("other")}
        self.options = types.SimpleNamespace(full_state_dict=False, strict=True)
        self.module._prepare_sd_options = lambda fsdp_plugin: self.options
        self.module.is_peft_model = lambda model: model is self.model
        self.module.FSDP_MODEL_NAME = "pytorch_model_fsdp"

        def get_peft_model_state_dict(model, state_dict=None, adapter_name="default",
                                      unwrap_compiled=False, save_embedding_layers="auto"):
            self.assertIs(model, self.model)
            self.assertIsNotNone(state_dict)
            self.assertEqual(adapter_name, "round2")
            return {key.replace("." + adapter_name, ""): value for key, value in state_dict.items()
                    if "lora_" in key and ("." + adapter_name + ".") in key}

        self.getter = peft.get_peft_model_state_dict = get_peft_model_state_dict

        def save(fsdp_plugin, accelerator, model, output_dir, model_index=0, adapter_only=False):
            self.events.append(("original-save", output_dir, model_index, adapter_only))
            return "original-save"

        def load(fsdp_plugin, accelerator, model, input_dir, model_index=0, adapter_only=False):
            self.events.append(("original-load", input_dir, model_index, adapter_only))
            return "original-load"

        self.original_save = self.module.save_fsdp_model = save
        self.original_load = self.module.load_fsdp_model = load
        state = types.ModuleType("torch.distributed.checkpoint.state_dict")

        def get_model_state_dict(model, *, options):
            self.assertIs(model, self.model)
            self.assertIs(options, self.options)
            self.events.append(("get", options))
            return dict(self.canonical)

        def set_model_state_dict(model, state_dict, *, options):
            self.assertIs(model, self.model)
            self.assertIs(options, self.options)
            self.events.append(("set", state_dict))
            return "canonical-result"

        state.get_model_state_dict = get_model_state_dict
        state.set_model_state_dict = set_model_state_dict
        dcp = types.ModuleType("torch.distributed.checkpoint")
        dcp.save = lambda **kwargs: self.events.append(("save", kwargs))
        self.loaded_a, self.loaded_b = CheckpointValue("loaded-a", 6, 0), CheckpointValue("loaded-b", 0, 6)

        def dcp_load(*, state_dict, storage_reader, planner):
            self.events.append(("load", storage_reader, planner))
            self.assertIs(state_dict["model"]["q.lora_A.weight"], self.a)
            state_dict["model"] = {"q.lora_A.weight": self.loaded_a, "q.lora_B.weight": self.loaded_b}

        dcp.load = dcp_load
        dcp.FileSystemWriter = lambda path: ("writer", path)
        dcp.FileSystemReader = lambda path: ("reader", path)
        planner = types.ModuleType("torch.distributed.checkpoint.default_planner")
        planner.DefaultSavePlanner = lambda: "save-default"
        planner.DefaultLoadPlanner = lambda: "load-default"
        fsdp = types.ModuleType("torch.distributed.fsdp")
        fsdp.StateDictType = types.SimpleNamespace(SHARDED_STATE_DICT="sharded", FULL_STATE_DICT="full")
        distributed = types.ModuleType("torch.distributed")
        distributed.checkpoint, distributed.fsdp = dcp, fsdp
        torch = types.ModuleType("torch")
        torch.distributed = distributed
        jittor = types.ModuleType("jittor")
        alias = types.ModuleType("transformers.trainer")
        alias.save_fsdp_model, alias.load_fsdp_model = save, load
        self.alias, self.peft, self.dcp = alias, peft, dcp
        self.modules = {"jittor": jittor, "jittor.compat.module_patcher": patcher,
            "torch": torch, "torch.distributed": distributed, fsdp.__name__: fsdp,
            dcp.__name__: dcp, state.__name__: state, planner.__name__: planner,
            "accelerate": package, self.module.__name__: self.module, "peft": peft,
            alias.__name__: alias}
        self.context = mock.patch.dict(sys.modules, self.modules)
        self.context.start()
        self.plugin = types.SimpleNamespace(fsdp_version=2, state_dict_type="sharded")
        self.accelerator = types.SimpleNamespace(wait_for_everyone=lambda: self.events.append(("barrier",)))

    def tearDown(self):
        patcher.uninstall_module_patches()
        patcher._REGISTRY.clear()
        patcher._REGISTRY.update(self.saved_registry)
        self.context.stop()

    def patch(self):
        self.assertTrue(adapter.patch_accelerate_peft_fsdp_utils(self.module))

    def save(self, **kwargs):
        return self.module.save_fsdp_model(self.plugin, self.accelerator, self.model, "/run",
                                           adapter_only=True, **kwargs)

    def load(self, path="/run", **kwargs):
        return self.module.load_fsdp_model(self.plugin, self.accelerator, self.model, path,
                                           adapter_only=True, **kwargs)

    def test_save_keeps_canonical_metadata_and_peft_names_including_zero_shard(self):
        self.patch()
        self.save(model_index=2)
        self.assertEqual([event[0] for event in self.events], ["get", "save"])
        kwargs = self.events[1][1]
        self.assertEqual(kwargs["storage_writer"], ("writer", "/run/pytorch_model_fsdp_2"))
        self.assertEqual(kwargs["planner"], "save-default")
        self.assertEqual(set(kwargs["state_dict"]["model"]), {"q.lora_A.weight", "q.lora_B.weight"})
        self.assertIs(kwargs["state_dict"]["model"]["q.lora_A.weight"], self.a)
        self.assertEqual(self.a.checkpoint_metadata["local_count"], 0)

    def test_restore_uses_inverse_peft_names_and_preserves_frozen_and_other_adapter(self):
        self.patch()
        self.assertEqual(self.load(model_index=2), "canonical-result")
        self.assertEqual([event[0] for event in self.events], ["barrier", "get", "load", "set"])
        self.assertEqual(self.events[2][1], ("reader", "/run/pytorch_model_fsdp_2"))
        actual = self.events[3][1]
        self.assertEqual(set(actual), set(self.canonical))
        self.assertIs(actual["base.weight"], self.frozen)
        self.assertIs(actual["q.lora_A.other.weight"], self.canonical["q.lora_A.other.weight"])
        self.assertIs(actual["q.lora_A.round2.weight"], self.loaded_a)
        self.assertIs(actual["q.lora_B.round2.weight"], self.loaded_b)
        self.assertEqual(self.model.active_adapter, "round2")

    def test_checkpoint_path_object(self):
        self.patch()
        path = Path("/run/pytorch_model_fsdp_3")
        self.load(path)
        self.assertEqual(self.events[2][1], ("reader", path))

    def test_missing_or_extra_adapter_key_rejected_before_owner_set(self):
        self.patch()
        for keys in ({"q.lora_A.weight": self.loaded_a},
                     {"q.lora_A.weight": self.loaded_a, "q.lora_B.weight": self.loaded_b, "bogus": self.loaded_a}):
            self.events.clear()
            self.dcp.load = lambda *, state_dict, **kwargs: state_dict.update(model=keys)
            with self.assertRaisesRegex(ValueError, "adapter keys"):
                self.load()
            self.assertFalse(any(event[0] == "set" for event in self.events))

    def test_non_target_paths_preserve_upstream_returns_and_arguments(self):
        self.patch()
        for version, state_type, adapter_only, model in ((1, "sharded", True, self.model),
                (2, "full", True, self.model), (2, "local", True, self.model),
                (2, object(), True, self.model), (2, "sharded", False, self.model),
                (2, "sharded", True, object())):
            self.plugin.fsdp_version, self.plugin.state_dict_type = version, state_type
            for name in ("save", "load"):
                result = getattr(self.module, name + "_fsdp_model")(
                    self.plugin, self.accelerator, model, "/original", model_index=7, adapter_only=adapter_only)
                self.assertEqual(result, "original-" + name)
                self.assertEqual(self.events[-1], ("original-" + name, "/original", 7, adapter_only))
        self.assertTrue(all(event[0].startswith("original-") for event in self.events))

    def test_peft_version_and_configuration_are_fail_closed(self):
        self.patch()
        self.peft.__version__ = "0.18.0"
        with self.assertRaisesRegex(UnsupportedAdapterVersion, "0.18.0"):
            self.save()
        self.peft.__version__ = "0.17.1"
        config = self.model.peft_config["round2"]
        for field, value in (("peft_type", "ADALORA"), ("use_dora", True), ("modules_to_save", ["head"]),
                             ("trainable_token_indices", [1])):
            original = getattr(config, field, None)
            setattr(config, field, value)
            with self.assertRaisesRegex(NotImplementedError, "plain LoRA"):
                self.save()
            setattr(config, field, original)
        self.assertFalse(any(event[0] == "save" for event in self.events))

    def test_changed_peft_getter_and_transformed_values_rejected(self):
        self.patch()
        self.peft.get_peft_model_state_dict = lambda model: {}
        with self.assertRaisesRegex(UnsupportedAdapterVersion, "getter contract"):
            self.save()

        def changed(model, state_dict=None, adapter_name="default", unwrap_compiled=False, save_embedding_layers="auto"):
            return {"key": object()}

        self.peft.get_peft_model_state_dict = changed
        with self.assertRaisesRegex(NotImplementedError, "transformed checkpoint values"):
            self.save()
        self.assertFalse(any(event[0] == "save" for event in self.events))

    def test_lifecycle_replaces_and_restores_all_bound_aliases(self):
        patcher.register_module_patch(self.module.__name__, adapter.patch_accelerate_peft_fsdp_utils)
        patcher.install_module_patches(load_entry_points=False)
        self.assertIs(self.alias.save_fsdp_model, self.module.save_fsdp_model)
        self.assertIs(self.alias.load_fsdp_model, self.module.load_fsdp_model)
        self.assertFalse(adapter.patch_accelerate_peft_fsdp_utils(self.module))
        patcher.uninstall_module_patches()
        self.assertIs(self.module.save_fsdp_model, self.original_save)
        self.assertIs(self.module.load_fsdp_model, self.original_load)
        self.assertIs(self.alias.save_fsdp_model, self.original_save)
        self.assertIs(self.alias.load_fsdp_model, self.original_load)

    def test_changed_upstream_contract_does_not_partially_patch(self):
        self.module.FSDP_MODEL_NAME = "changed"
        with self.assertRaisesRegex(UnsupportedAdapterVersion, "checkpoint contract changed"):
            self.patch()
        self.assertIs(self.module.save_fsdp_model, self.original_save)
        self.assertIs(self.module.load_fsdp_model, self.original_load)

    def test_entire_adapter_package_remains_relocatable(self):
        spec = importlib.util.spec_from_file_location("_peft_relocatable_contract",
            ROOT / "adapters/tests/relocatable_contract.py")
        contract = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(contract)
        contract.audit_adapter(ROOT / "adapters/jittor_adapters")


if __name__ == "__main__":
    unittest.main()
