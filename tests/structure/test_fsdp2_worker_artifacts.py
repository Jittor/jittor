"""Maintained pure-host gate for the real two-rank worker artifact protocol."""
import importlib.util
from pathlib import Path
import sys
import unittest

SCRIPTS = Path(__file__).resolve().parents[2] / "agent/skills/verl-torch-compat/scripts"


class TestFSDP2WorkerComparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("_fsdp2_worker_contracts", SCRIPTS / "test_compare_fsdp2_worker.py")
        module = importlib.util.module_from_spec(spec)
        original_path = list(sys.path)
        try:
            sys.path.insert(0, str(SCRIPTS))
            spec.loader.exec_module(module)
        finally:
            sys.path[:] = original_path
        cls.contract_type = module.TestFSDP2WorkerComparison

    def test_artifact_contracts(self):
        names = unittest.defaultTestLoader.getTestCaseNames(self.contract_type)
        self.assertTrue(names, "the maintained worker contract suite must not be empty")
        for name in names:
            with self.subTest(contract=name):
                self.contract_type(name).debug()


if __name__ == "__main__":
    unittest.main()
