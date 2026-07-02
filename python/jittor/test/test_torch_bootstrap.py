import os
import tempfile
import textwrap
import unittest


class TestTorchBootstrap(unittest.TestCase):
    def test_scan_torch_extension_setup(self):
        from jittor.torch_shim import scan_extension_dirs

        with tempfile.TemporaryDirectory(dir=os.getcwd()) as d:
            root = os.path.join(d, "pkg")
            os.makedirs(root)
            with open(os.path.join(root, "setup.py"), "w") as f:
                f.write(textwrap.dedent("""
                    from setuptools import setup
                    from torch.utils.cpp_extension import CUDAExtension, BuildExtension
                    setup(
                        name="pkg",
                        ext_modules=[
                            CUDAExtension(name="pkg._C", sources=["kernel.cu", "ext.cpp"])
                        ],
                        cmdclass={"build_ext": BuildExtension},
                    )
                """))
            for name in ("kernel.cu", "ext.cpp"):
                open(os.path.join(root, name), "w").close()

            exts = scan_extension_dirs(project_root=d)
            self.assertEqual([e.root for e in exts], [root])
            self.assertEqual(len(exts[0].sources), 2)
            self.assertTrue(exts[0].setup_py.endswith("setup.py"))

    def test_torch_utils_unknown_submodule_fails_fast(self):
        import importlib
        import time
        import jittor as torch

        start = time.time()
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("torch.utils.definitely_missing_submodule")
        self.assertLess(time.time() - start, 1.0)
        self.assertTrue(hasattr(torch, "utils"))


if __name__ == "__main__":
    unittest.main()
