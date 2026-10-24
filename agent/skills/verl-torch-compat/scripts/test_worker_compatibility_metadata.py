"""Host-only provenance tests; fake loaded modules are not device evidence."""
from copy import deepcopy
import hashlib
import importlib.metadata
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

from probe_training_worker import compatibility_metadata
from compare_training_worker import validate_compatibility
from test_compare_training_worker import compatibility_fixture


class Nested:
    pass


def marked(marker):
    function = lambda: None
    setattr(function, marker, Nested)
    return function


class TestWorkerCompatibilityMetadata(unittest.TestCase):
    def setUp(self):
        self.expected = compatibility_fixture("jittor")
        self.frontend = types.ModuleType("jittor.compat.torch")
        self.frontend.nested_tensor_type = lambda: Nested
        self.verl_adapter = types.ModuleType("jittor_adapters.verl")
        self.verl_adapter.SUPPORTED_VERSIONS = frozenset(("0.9.1",))
        self.verl_adapter.SOURCE_SHA256 = self.expected["adapters"]["verl"]["upstream_source_sha256"]
        self.verl_adapter.__file__ = __file__
        self.td_adapter = types.ModuleType("jittor_adapters.tensordict")
        self.td_adapter.SUPPORTED_VERSIONS = frozenset(("0.10.0",))
        self.td_adapter.__file__ = __file__
        self.helper = types.ModuleType("verl.utils.tensordict_utils")
        for name in self.expected["adapters"]["verl"]["active_targets"]:
            function = marked("_jittor_verl_nested_type")
            function._jittor_verl_original = lambda: None
            setattr(self.helper, name, function)
        self.base = types.ModuleType("tensordict.base")
        self.base._ACCEPTED_CLASSES = (Nested,)
        self.base.TensorDictBase = types.SimpleNamespace(**{
            name: marked("_jittor_tensordict_nested")
            for name in self.expected["adapters"]["tensordict"]["active_targets"]
        })
        self.modules = {module.__name__: module for module in (
            self.frontend, self.verl_adapter, self.td_adapter, self.helper, self.base)}
        self.stack = mock.patch.dict(sys.modules, self.modules)
        self.stack.start()
        self.addCleanup(self.stack.stop)

    def test_native_torch_never_queries_or_claims_candidate_adapters(self):
        with mock.patch.object(importlib.metadata, "version", side_effect=AssertionError("must not query")):
            observed = compatibility_metadata("torch")
        self.assertEqual(observed, compatibility_fixture("torch"))
        validate_compatibility(observed, "torch")
        with self.assertRaises(ValueError):
            validate_compatibility(observed, "jittor")

    def test_observation_uses_loaded_source_and_real_activation_markers(self):
        before = dict(vars(self.helper))
        with mock.patch.object(importlib.metadata, "version", return_value="1.3.11.0") as version:
            observed = compatibility_metadata("jittor")
        version.assert_called_once_with("jittor-torch-adapters")
        digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        for item in self.expected["adapters"].values():
            item["source_sha256"] = digest
        self.assertEqual(observed, self.expected)
        self.assertEqual(vars(self.helper), before)
        validate_compatibility(observed, "jittor")

    def test_source_checkout_does_not_fabricate_distribution_version(self):
        with mock.patch.object(importlib.metadata, "version", side_effect=importlib.metadata.PackageNotFoundError):
            observed = compatibility_metadata("jittor")
        self.assertIsNone(observed["distribution_version"])
        self.assertEqual(observed["distribution_version_status"], "metadata_unavailable")
        validate_compatibility(observed, "jittor")

    def test_missing_or_inactive_adapter_is_observed_not_silently_called_active(self):
        with mock.patch.object(importlib.metadata, "version", return_value="1.3.11.0"):
            for target, field, bad in (
                (self.helper.get, "_jittor_verl_nested_type", object),
                (self.base, "_ACCEPTED_CLASSES", ()),
                (self.verl_adapter, "__file__", "/missing/adapter-source.py"),
                (self.td_adapter, "SUPPORTED_VERSIONS", frozenset(("99.0.0",))),
            ):
                with self.subTest(field=field), mock.patch.object(target, field, bad):
                    observed = compatibility_metadata("jittor")
                    with self.assertRaises(ValueError):
                        validate_compatibility(observed, "jittor")
            with mock.patch.dict(sys.modules):
                sys.modules.pop("jittor_adapters.verl")
                observed = compatibility_metadata("jittor")
                self.assertFalse(observed["adapters"]["verl"]["loaded"])
                with self.assertRaises(ValueError):
                    validate_compatibility(observed, "jittor")

    def test_incomplete_provenance_and_false_native_claim_rejected(self):
        for framework in ("torch", "jittor"):
            with self.assertRaises(ValueError):
                validate_compatibility(None, framework)
            bad = compatibility_fixture(framework)
            bad["harness_private_patches"] = True
            with self.assertRaises(ValueError):
                validate_compatibility(bad, framework)
        with self.assertRaises(ValueError):
            validate_compatibility(compatibility_fixture("jittor"), "torch")
        for key, value in (("distribution_version_status", "installed_metadata"),
                           ("adapters", {}), ("status", "not_applicable")):
            bad = deepcopy(self.expected)
            if key == "distribution_version_status":
                bad["distribution_version"] = None
            bad[key] = value
            with self.assertRaises(ValueError):
                validate_compatibility(bad, "jittor")


if __name__ == "__main__":
    unittest.main()
