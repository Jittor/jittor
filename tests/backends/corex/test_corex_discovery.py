import os
import tempfile
import unittest
import importlib.util
from pathlib import Path


def load_corex_module():
    source = Path(__file__).parents[3] / "python/jittor/extern/corex/corex_compiler.py"
    spec = importlib.util.spec_from_file_location("corex_compiler_probe", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCorexDiscovery(unittest.TestCase):
    def test_discovery_is_read_only_and_path_configurable(self):
        corex_compiler = load_corex_module()

        with tempfile.TemporaryDirectory() as root:
            home = os.path.join(root, "corex")
            os.makedirs(os.path.join(home, "bin"))
            compiler_path = os.path.join(home, "bin", "clang++")
            with open(compiler_path, "w") as stream:
                stream.write("fake clang++\n")
            before = set(os.listdir(root))
            result = corex_compiler.discover(home)
            self.assertTrue(result.available)
            self.assertEqual(result.home, os.path.abspath(home))
            self.assertEqual(result.compiler_path, compiler_path)
            self.assertEqual(set(os.listdir(root)), before)

    def test_missing_compiler_is_reported_without_global_setup(self):
        corex_compiler = load_corex_module()

        with tempfile.TemporaryDirectory() as root:
            result = corex_compiler.discover(root)
            self.assertFalse(result.available)
            self.assertIn("compiler", result.reason)
            self.assertEqual(corex_compiler.has_corex, 0)


if __name__ == "__main__":
    unittest.main()
