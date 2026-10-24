"""Pure-host negative controls for observed, not installed, loader provenance."""
import copy
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

from worker_loading_provenance import ENVIRONMENT_KEY, observe_loading, validate_loading


class TestLoadingProvenance(unittest.TestCase):
    def observe(self, framework="jittor", requested="0", active=False, source=True, versions=("5.9.0",)):
        def entry(cls):
            raise AssertionError("observation must never execute or replace the loader")
        if active:
            entry._jittor_transformers_materialized_init = True
        owner = type("PreTrainedModel", (), {"get_init_context": classmethod(entry)})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.py"
            path.write_text("# observed adapter source\n")
            adapter = types.SimpleNamespace(__file__=str(path) if source else None,
                                            SUPPORTED_VERSIONS=frozenset(versions))
            with mock.patch.dict(os.environ, {ENVIRONMENT_KEY: requested}), mock.patch.dict(sys.modules, {
                "transformers.modeling_utils": types.SimpleNamespace(PreTrainedModel=owner),
                "jittor_adapters.transformers_loading": adapter,
            }):
                descriptor = vars(owner)["get_init_context"]
                result = observe_loading(framework)
                self.assertIs(vars(owner)["get_init_context"], descriptor)
                validate_loading(result, framework)
                return result

    def test_native_and_inactive_candidate_make_no_adapter_claim(self):
        for framework in ("torch", "jittor"):
            value = self.observe(framework)
            self.assertEqual(value["mode"], "upstream_default")
            self.assertIsNone(value["adapter_source_sha256"])

    def test_active_candidate_records_real_source_without_calling_entry(self):
        value = self.observe(requested="1", active=True)
        self.assertEqual(value["mode"], "materialized_cpu_constructor")
        self.assertEqual(len(value["adapter_source_sha256"]), 64)
        self.assertFalse(value["meta_support_claimed"])

    def test_requested_active_mismatch_and_unsupported_configuration_rejected(self):
        for options in ({"requested": "1"}, {"active": True}, {"requested": "yes"},
                        {"requested": "1", "active": True, "source": False},
                        {"requested": "1", "active": True, "versions": ("99",)},
                        {"framework": "torch", "requested": "1", "active": True},
                        {"framework": "unknown"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.observe(**options)

    def test_missing_loaded_entry_is_not_imported_or_invented(self):
        with mock.patch.dict(sys.modules, {"transformers.modeling_utils": None}), self.assertRaises(ValueError):
            observe_loading("jittor")

    def test_every_provenance_field_is_required(self):
        good = self.observe(requested="1", active=True)
        for field in good:
            corrupt = copy.deepcopy(good)
            del corrupt[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_loading(corrupt, "jittor")

    def test_false_meta_native_and_corrupt_source_claims_rejected(self):
        good = self.observe(requested="1", active=True)
        for field, value in (("meta_support_claimed", True), ("active_materialized_cpu", False),
                             ("requested_materialized_cpu", 1), ("scope", "all adapters"),
                             ("adapter_source_sha256", "z" * 64), ("supported_versions", ["99"]),
                             ("mode", "native_meta")):
            corrupt = {**good, field: value}
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_loading(corrupt, "jittor")
        with self.assertRaises(ValueError):
            validate_loading(good, "torch")
        inactive = self.observe()
        inactive["adapter_source_sha256"] = "a" * 64
        with self.assertRaises(ValueError):
            validate_loading(inactive, "jittor")


if __name__ == "__main__":
    unittest.main()
