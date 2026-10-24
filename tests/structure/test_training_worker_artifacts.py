"""Maintain the real critic-worker comparator's pure-host negative controls."""
import importlib.util
import importlib
from pathlib import Path
import sys
import unittest


SCRIPTS = Path(__file__).resolve().parents[2] / "agent/skills/verl-torch-compat/scripts"

class TestTrainingWorkerArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "_critic_worker_host_contracts", SCRIPTS / "test_compare_training_worker.py")
        module = importlib.util.module_from_spec(spec)
        original_path = list(sys.path)
        try:
            sys.path.insert(0, str(SCRIPTS))
            spec.loader.exec_module(module)
            metadata_spec = importlib.util.spec_from_file_location(
                "_worker_compatibility_contracts", SCRIPTS / "test_worker_compatibility_metadata.py")
            metadata_module = importlib.util.module_from_spec(metadata_spec)
            metadata_spec.loader.exec_module(metadata_module)
            device_spec = importlib.util.spec_from_file_location(
                "_worker_device_assert_contracts", SCRIPTS / "test_worker_device_assert.py")
            device_module = importlib.util.module_from_spec(device_spec)
            device_spec.loader.exec_module(device_module)
            buffer_module = importlib.import_module("test_worker_buffer_contract")
            loading_module = importlib.import_module("test_worker_loading_provenance")
            # Buffer negative controls exercise both role-specific comparators.
            # Resolve fixtures while the temporary scripts path is active.
            importlib.import_module("test_compare_training_worker")
            importlib.import_module("test_compare_actor_worker")
        finally:
            sys.path[:] = original_path
        cls.contract_types = (
            module.TestWorkerComparison,
            metadata_module.TestWorkerCompatibilityMetadata,
            device_module.TestWorkerDeviceAssert,
            buffer_module.TestWorkerBufferContract,
            loading_module.TestLoadingProvenance,
        )

    def test_artifact_contracts(self):
        self.assertEqual(len(self.contract_types), 5)
        for contract_type in self.contract_types:
            names = unittest.defaultTestLoader.getTestCaseNames(contract_type)
            self.assertTrue(names, "the maintained critic contract suite must not be empty")
            for name in names:
                with self.subTest(suite=contract_type.__name__, contract=name):
                    contract_type(name).debug()


if __name__ == "__main__":
    unittest.main()
