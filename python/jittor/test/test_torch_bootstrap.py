import os
import pathlib
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

    def test_entry_script_runtime_defaults_to_project_dir(self):
        import jittor as jt

        with tempfile.TemporaryDirectory(dir=os.getcwd()) as d:
            entry = os.path.join(d, "train.py")
            with open(entry, "w") as f:
                f.write("from jittor.torch_shim import enable\n")
            old_argv0 = jt._sys.argv[0]
            try:
                jt._sys.argv[0] = entry
                self.assertEqual(
                    jt._jt_torch_entry_runtime_root(),
                    os.path.join(d, ".jittor_torch_runtime"),
                )
            finally:
                jt._sys.argv[0] = old_argv0

    def test_pythonpath_extension_roots_skip_conda_prefix(self):
        from jittor.torch_shim.bootstrap import _pythonpath_extension_roots

        project = pathlib.Path(os.getcwd()).resolve()
        runtime = project / ".jittor_torch_runtime"
        roots = _pythonpath_extension_roots(project, runtime)
        for root in roots:
            self.assertNotIn("site-packages", set(root.parts))
            self.assertFalse(str(root).startswith(os.path.realpath(os.sys.prefix)))

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
