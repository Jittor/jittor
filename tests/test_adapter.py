from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock


sys.path.insert(0, os.fspath(Path(__file__).resolve().parents[1] / "src"))

from jittor_hf_compat import register_patches
from jittor_hf_compat.patches import (
    _patch_transformers_default_rope_scaling,
    _patch_transformers_legacy_symbols,
    _patch_transformers_legacy_tied_weights_keys,
)


class TestTransformersAdapter(unittest.TestCase):
    def test_entry_point_registers_exactly_three_transformers_patches(self):
        registrations = []

        register_patches(lambda path, callback: registrations.append((path, callback)))

        self.assertEqual(
            [path for path, _ in registrations],
            [
                "transformers.utils.import_utils",
                "transformers.modeling_utils",
                "transformers.configuration_utils",
            ],
        )
        self.assertTrue(all(callable(callback) for _, callback in registrations))

    def test_legacy_tied_weight_list_is_expanded_to_a_mapping(self):
        weight = object()

        class Model:
            _tied_weights_keys = [
                "model.embed_tokens.weight",
                "lm_head.weight",
            ]

            def get_input_embeddings(self):
                return SimpleNamespace(weight=weight)

            def named_parameters(self, remove_duplicate=True):
                del remove_duplicate
                return [("model.embed_tokens.weight", weight)]

            def get_expanded_tied_weights_keys(self):
                return self._tied_weights_keys

        module = SimpleNamespace(PreTrainedModel=Model)
        self.assertTrue(_patch_transformers_legacy_tied_weights_keys(module))
        self.assertFalse(_patch_transformers_legacy_tied_weights_keys(module))

        model = Model()
        self.assertEqual(
            model.get_expanded_tied_weights_keys(),
            {"lm_head.weight": "model.embed_tokens.weight"},
        )

    def test_default_rope_placeholder_changes_only_for_minicpm(self):
        class Config:
            def __init__(self, model_type="", architectures=()):
                self.model_type = model_type
                self.architectures = architectures
                self.rope_scaling = {"rope_type": "default"}

        module = SimpleNamespace(PretrainedConfig=Config)
        self.assertTrue(_patch_transformers_default_rope_scaling(module))
        self.assertFalse(_patch_transformers_default_rope_scaling(module))

        self.assertIsNone(Config(model_type="minicpm3").rope_scaling)
        self.assertIsNone(
            Config(architectures=("MiniCPMForCausalLM",)).rope_scaling
        )
        self.assertEqual(
            Config(model_type="llama").rope_scaling,
            {"rope_type": "default"},
        )

    def test_legacy_import_symbols_refresh_lazy_backend_state(self):
        class LazyModule(ModuleType):
            pass

        import_utils = ModuleType("transformers.utils.import_utils")
        import_utils._LazyModule = LazyModule
        import_utils.PYTORCH_IMPORT_ERROR = "missing torch"
        import_utils.BACKENDS_MAPPING = {
            "torch": (lambda: False, "missing torch")
        }
        utilities = ModuleType("transformers.utils")
        generic = ModuleType("transformers.utils.generic")
        lazy = LazyModule("transformers.lazy_model")
        lazy._object_missing_backend = {
            "TorchOnly": ["torch"],
            "Mixed": ["torch", "vision"],
        }
        torch_alias = ModuleType("jittor")
        modules = {
            "torch": torch_alias,
            "transformers.utils": utilities,
            "transformers.utils.generic": generic,
            "transformers.lazy_model": lazy,
        }

        with mock.patch.dict(sys.modules, modules, clear=False):
            self.assertTrue(_patch_transformers_legacy_symbols(import_utils))
            self.assertFalse(_patch_transformers_legacy_symbols(import_utils))

        self.assertTrue(import_utils.is_torch_available())
        self.assertEqual(import_utils.get_torch_version(), "2.4.0")
        self.assertTrue(import_utils.BACKENDS_MAPPING["torch"][0]())
        self.assertIs(utilities.is_torch_available, import_utils.is_torch_available)
        self.assertTrue(generic._is_torch_available)
        self.assertNotIn("TorchOnly", lazy._object_missing_backend)
        self.assertEqual(lazy._object_missing_backend["Mixed"], ["vision"])
        self.assertTrue(callable(import_utils.is_torch_fx_available))

    def test_source_contains_no_unrelated_project_adapters(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "jittor_hf_compat"
            / "patches.py"
        ).read_text(encoding="utf-8")
        for marker in (
            "jittor_fsdp2",
            "trellis2",
            "flex_gemm",
            "ms.swift",
            "swift.llm",
            "peft.",
            "trl.",
        ):
            self.assertNotIn(marker, source)


if __name__ == "__main__":
    unittest.main()
