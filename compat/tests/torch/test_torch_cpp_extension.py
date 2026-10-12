"""Smoke tests for the Jittor-backed ``torch.utils.cpp_extension`` ABI shim.

3D Gaussian Splatting builds PyTorch-style C++/CUDA extensions
(``#include <torch/extension.h>``) for packages such as
diff-gaussian-rasterization, simple-knn, and fused-ssim. This test locks the
minimal contract those packages need: the deployed ``import torch`` shim exposes
``torch.utils.cpp_extension``, and a compiled pybind extension can receive/return
jittor Vars through ``torch::Tensor``.

Run:
    python -m pytest compat/tests/torch/test_torch_cpp_extension.py
"""
import os
import sys
import tempfile
import unittest
from types import ModuleType, SimpleNamespace
from unittest import mock
from _helpers import capability as _test_capability

import jittor as jt


class TestTorchCppExtensionArchFlags(unittest.TestCase):
    def test_reports_the_builder_cxx11_abi(self):
        import torch
        from jittor.compat.shim import cpp_extension

        expected = bool(cpp_extension.CXX11_ABI)
        self.assertEqual(torch.compiled_with_cxx11_abi(), expected)
        self.assertEqual(torch._C._GLIBCXX_USE_CXX11_ABI, expected)

    def test_uses_detected_jittor_archs(self):
        from jittor.compat.shim.cpp_extension import _cuda_arch_flags

        fake_jittor = type("Jittor", (), {
            "flags": type("Flags", (), {"cuda_archs": [89, 80]})(),
        })()
        fake_compiler = type("Compiler", (), {"nvcc_flags": ""})()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            self.assertEqual(
                _cuda_arch_flags(fake_jittor, fake_compiler),
                ["-arch=compute_80", "-code=sm_80", "-code=sm_89"],
            )

    def test_honors_torch_cuda_arch_list(self):
        from jittor.compat.shim.cpp_extension import _cuda_arch_flags

        fake_jittor = type("Jittor", (), {
            "flags": type("Flags", (), {"cuda_archs": [89]})(),
        })()
        fake_compiler = type("Compiler", (), {"nvcc_flags": ""})()
        with mock.patch.dict(os.environ, {"TORCH_CUDA_ARCH_LIST": "8.0;8.6+PTX"}):
            self.assertEqual(
                _cuda_arch_flags(fake_jittor, fake_compiler),
                [
                    "-gencode=arch=compute_80,code=sm_80",
                    "-gencode=arch=compute_86,code=sm_86",
                    "-gencode=arch=compute_86,code=compute_86",
                ],
            )

    def test_expands_named_torch_cuda_arches(self):
        from jittor.compat.shim.cpp_extension import _torch_cuda_arch_flags

        self.assertEqual(
            _torch_cuda_arch_flags("Ampere;Ada;Hopper"),
            [
                "-gencode=arch=compute_80,code=sm_80",
                "-gencode=arch=compute_86,code=sm_86",
                "-gencode=arch=compute_86,code=compute_86",
                "-gencode=arch=compute_89,code=sm_89",
                "-gencode=arch=compute_89,code=compute_89",
                "-gencode=arch=compute_90,code=sm_90",
                "-gencode=arch=compute_90,code=compute_90",
            ],
        )

    def test_falls_back_to_compiler_arch_flags(self):
        from jittor.compat.shim.cpp_extension import _cuda_arch_flags

        fake_jittor = type("Jittor", (), {
            "flags": type("Flags", (), {"cuda_archs": []})(),
        })()
        fake_compiler = type("Compiler", (), {
            "nvcc_flags": "--fmad=false -arch=compute_75 -code=sm_75",
        })()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            self.assertEqual(
                _cuda_arch_flags(fake_jittor, fake_compiler),
                ["-arch=compute_75", "-code=sm_75"],
            )


class TestTorchCppExtensionImportIdentity(unittest.TestCase):
    def test_invalid_identity_fails_before_build(self):
        from jittor.compat.shim import cpp_extension
        from jittor.compat.shim.cpp_extension import torch_utils

        with mock.patch.object(cpp_extension, "build") as build:
            with self.assertRaisesRegex(ValueError, "must be a dotted"):
                torch_utils.load(
                    name="same_build_name",
                    sources=["unused.cpp"],
                    import_identity="invalid_identity",
                )
        build.assert_not_called()

    def test_distinct_identities_load_distinct_modules_with_same_build_name(self):
        from jittor.compat.shim.backends import flash_attention as flashattn_jittor
        from jittor.compat.shim import cpp_extension
        from jittor.compat.shim.cpp_extension import torch_utils

        build_name = "flash_attn_2_cuda_jittor"
        capabilities = (
            ("h32-fp16", "a1"),
            ("h32-h64-fp16", "b2"),
            ("h32-h64-fp16-bf16", "c3"),
        )
        imported_names = []
        created_modules = []

        class Loader:
            def exec_module(self, module):
                return None

        def fake_spec(name, path):
            imported_names.append(name)
            return SimpleNamespace(name=name, path=path, loader=Loader())

        def fake_module_from_spec(spec):
            module = ModuleType(spec.name)
            module.__file__ = spec.path
            created_modules.append(module)
            return module

        module_keys = []
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(cpp_extension, "cfg", return_value={"ext_suffix": ".so"}), \
                mock.patch.object(cpp_extension, "build") as build, \
                mock.patch.object(torch_utils.importlib.util, "spec_from_file_location",
                                  side_effect=fake_spec), \
                mock.patch.object(torch_utils.importlib.util, "module_from_spec",
                                  side_effect=fake_module_from_spec):
            try:
                modules = []
                for generation, (capability, build_digest) in enumerate(
                        capabilities, start=1):
                    build_dir = os.path.join(tmp, build_digest)
                    identity = flashattn_jittor._official_import_identity(
                        capability, build_dir, build_name, generation=generation)
                    import_name = torch_utils._extension_import_name(
                        build_name, identity)
                    module_keys.append(import_name)
                    modules.append(torch_utils.load(
                        name=build_name,
                        sources=[os.path.join(tmp, "probe.cpp")],
                        build_directory=build_dir,
                        import_identity=identity,
                    ))
            finally:
                for key in module_keys:
                    sys.modules.pop(key, None)

        self.assertEqual(len({id(module) for module in modules}), 3)
        self.assertEqual(len({module.__file__ for module in modules}), 3)
        self.assertEqual(imported_names, module_keys)
        self.assertTrue(all(name.endswith("." + build_name)
                            for name in imported_names))
        self.assertEqual([call.kwargs["name"] for call in build.call_args_list],
                         [build_name] * 3)


def _torch_cpp_extension_available():
    try:
        import torch
        from torch.utils.cpp_extension import load_inline  # noqa: F401
        # This test is meaningful only when bare import torch is the deployed
        # jittor torch-shim, not a real PyTorch install.
        if not issubclass(torch.Tensor, jt.Var):
            return False
    except ImportError:
        return False
    return (_test_capability.check_accelerator("cuda", backend=jt).enabled
            and bool(jt.introspection.policy.startup.nvcc_path))


class TestTorchCppExtension(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not _torch_cpp_extension_available():
            raise unittest.SkipTest("needs deployed torch-shim + nvcc")

    def _build_probe_extension(self):
        import torch
        from torch.utils.cpp_extension import load_inline
        from jittor.compat.shim.cpp_extension import _find_pybind_include

        # ``torch/extension.h`` includes <pybind11/pybind11.h>, so building any
        # extension needs those headers. They are not part of jittor, and this
        # machine has no pybind11 at all -- report that rather than reporting a
        # compile failure whose cause is a missing build dependency.
        if _find_pybind_include() is None:
            self.skipTest("building a C++ extension needs the pybind11 headers; "
                          "they are not installed here")

        src = r"""
#include <torch/extension.h>

torch::Tensor identity(torch::Tensor x) {
    return x;
}

int64_t first_dim(torch::Tensor x) {
    return x.size(0);
}

bool is_cpu_tensor(torch::Tensor x) {
    return x.is_cpu();
}

bool empty_byte_data_ptr_is_null() {
    auto b = torch::empty(0, torch::kByte);
    return b.data_ptr<unsigned char>() == nullptr;
}

torch::Tensor resize_byte_buffer() {
    auto b = torch::empty(0, torch::kByte);
    b.resize_({16});
    TORCH_CHECK(b.data_ptr<unsigned char>() != nullptr, "resized byte buffer");
    return b;
}

torch::Tensor sort_values(torch::Tensor x) {
    auto r = torch::sort(x, 0, false);
    return std::get<0>(r);
}

torch::Tensor select_dim0(torch::Tensor x, torch::Tensor idx) {
    return torch::index_select(x, 0, idx);
}

torch::Tensor mask_ne(torch::Tensor x) {
    auto m = x != 2.0;
    return x.masked_select(m);
}

torch::Tensor view_as_bytes(torch::Tensor x) {
    return x.view(torch::kByte);
}

torch::Tensor zeros_like_with_options(torch::Tensor x) {
    return torch::zeros_like(x, x.options());
}
"""
        build_dir = os.path.join(tempfile.gettempdir(), "jt_cpp_extension_test")
        mod = load_inline(
            name="jt_cpp_extension_test",
            cpp_sources=src,
            functions=[
                "identity", "first_dim", "is_cpu_tensor",
                "empty_byte_data_ptr_is_null", "resize_byte_buffer",
                "sort_values", "select_dim0", "mask_ne", "view_as_bytes",
                "zeros_like_with_options",
            ],
            build_directory=build_dir,
            verbose=False,
        )
        return mod

    def test_load_inline_tensor_roundtrip_and_cpu_residency(self):
        import torch

        mod = self._build_probe_extension()

        x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, device="cpu")
        y = mod.identity(x)
        self.assertIsInstance(y, jt.Var)
        self.assertEqual(tuple(y.shape), (2, 2))
        self.assertAlmostEqual(float(y.sum().item()), 10.0)
        self.assertEqual(mod.first_dim(x), 2)
        self.assertTrue(mod.is_cpu_tensor(x))

    def test_cuda_extension_preserves_setup_metadata(self):
        from torch.utils.cpp_extension import CUDAExtension

        ext = CUDAExtension(
            "pkg._C",
            ["rasterize_points.cu", "ext.cpp"],
            include_dirs=["third_party/glm"],
            define_macros=[("WITH_CUDA", None)],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "--use_fast_math"]},
            extra_link_args=["-Wl,--as-needed"],
        )
        self.assertEqual(ext.name, "pkg._C")
        self.assertEqual(ext.include_dirs, ["third_party/glm"])
        self.assertEqual(ext.extra_link_args, ["-Wl,--as-needed"])
        self.assertEqual(ext.extra_compile_args["nvcc"], ["-O3", "--use_fast_math"])

    def test_import_jittor_as_torch_exposes_cpp_extension(self):
        import torch
        from torch.utils.checkpoint import checkpoint

        self.assertTrue(hasattr(torch, "utils"))
        self.assertTrue(hasattr(torch.utils, "cpp_extension"))
        self.assertTrue(hasattr(torch.utils.cpp_extension, "CUDAExtension"))
        self.assertTrue(hasattr(torch.utils.cpp_extension, "load_inline"))
        y = checkpoint(lambda x: x * 2, torch.tensor([3.0]))
        self.assertEqual(float(y.item()), 6.0)

    def test_3dgs_style_tensor_ops(self):
        import torch

        mod = self._build_probe_extension()

        self.assertTrue(mod.empty_byte_data_ptr_is_null())
        byte_buf = mod.resize_byte_buffer()
        self.assertEqual(tuple(byte_buf.shape), (16,))
        self.assertEqual(str(byte_buf.dtype), "uint8")

        x = torch.tensor([3.0, 1.0, 2.0])
        idx = torch.tensor([2, 0], dtype=torch.int64)
        np = mod.sort_values(x).numpy()
        self.assertEqual(np.tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(mod.select_dim0(x, idx).numpy().tolist(), [2.0, 3.0])
        self.assertEqual(mod.mask_ne(x).numpy().tolist(), [3.0, 1.0])

        b = mod.view_as_bytes(torch.tensor([1.0], dtype=torch.float32))
        self.assertEqual(tuple(b.shape), (4,))
        self.assertEqual(str(b.dtype), "uint8")

        z = mod.zeros_like_with_options(x)
        self.assertEqual(tuple(z.shape), (3,))
        self.assertEqual(str(z.dtype), "float32")
        self.assertEqual(float(z.sum().item()), 0.0)


class TestShimHeadersInvalidateTheBuildCache(unittest.TestCase):
    """A shim header edit has to recompile, not report "up-to-date".

    The up-to-date checks compare an object's source mtime and its compile
    command, so they never saw a header change. Editing ``torch/extension.h``
    therefore reused objects compiled against the old text: the extension kept a
    call to a symbol the new header no longer defined, every object was reported
    "up-to-date", and the failure only appeared later as ``undefined symbol`` at
    import. The shim's header content is carried in the command as
    ``-DJTORCH_SHIM_ABI=<digest>`` so that cannot happen silently again.
    """

    # build() only needs these keys for a single C++ source; the compiler itself
    # is faked below, so the paths do not have to exist.
    _CFG = {
        "cc_path": "/bin/true", "nvcc_path": "/bin/true", "ext_suffix": ".so",
        "src_inc": "", "extern_inc": "", "extern_cuda_inc": "",
        # Every other include is an empty placeholder because the compiler is
        # faked and the values never reach one. `pybind_inc` is not: a build
        # without pybind11 headers cannot succeed, so `_common_includes`
        # refuses instead of emitting a command g++ would reject later.
        "py_inc": "", "pybind_inc": "/stub/pybind11/include",
        "cuda_includes": [],
        "core_dirs": [], "arch_flags": [], "cores": {}, "cuda_libs": [],
        # The real cfg() always carries these two (None when there is no CUDA
        # runtime / no wheel stack); the build path reads them unconditionally.
        "cudart_lib": None, "cuda_wheel_fingerprint": None,
    }

    @staticmethod
    def _abi_digest_in(commands):
        for cmd in commands:
            for arg in cmd:
                if arg.startswith("-DJTORCH_SHIM_ABI="):
                    return arg.split("=", 1)[1]
        return None

    def test_editing_a_shim_header_recompiles(self):
        from jittor.compat.shim import cpp_extension

        with tempfile.TemporaryDirectory() as tmp:
            include = os.path.join(tmp, "include")
            os.makedirs(os.path.join(include, "torch"))
            header = os.path.join(include, "torch", "extension.h")
            build_dir = os.path.join(tmp, "build")
            os.makedirs(build_dir)
            src = os.path.join(tmp, "probe.cpp")
            commands = []

            def write(path, text):
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(text)

            def fake_run(cmd, **kwargs):
                commands.append(list(cmd))
                # The compiler is faked, so create the object it was asked for --
                # otherwise _object_matches_command sees a missing object and the
                # "unchanged tree" case would compile for the wrong reason.
                if "-o" in cmd:
                    write(cmd[cmd.index("-o") + 1], "")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            write(header, "// v1\n")
            write(src, "int probe() { return 0; }\n")

            with mock.patch.object(cpp_extension, "SHIM_INCLUDE", include), \
                    mock.patch.object(cpp_extension, "SHIM_SOURCES", []), \
                    mock.patch.object(cpp_extension, "cfg",
                                      return_value=dict(self._CFG)), \
                    mock.patch.object(cpp_extension.subprocess, "run",
                                      side_effect=fake_run):
                cpp_extension.build("probe", [src], build_dir, verbose=False)
                self.assertEqual(len(commands), 2,
                                 "expected one compile and one link, got %r" % commands)
                first = self._abi_digest_in(commands)
                self.assertIsNotNone(
                    first, "the shim header digest never reached the compiler")

                commands.clear()
                cpp_extension.build("probe", [src], build_dir, verbose=False)
                self.assertEqual(commands, [],
                                 "an unchanged tree recompiled: %r" % commands)

                commands.clear()
                write(header, "// v2: the ABI changed\n")
                cpp_extension.build("probe", [src], build_dir, verbose=False)
                self.assertNotEqual(commands, [],
                                    "a shim header edit was ignored by the cache")
                self.assertNotEqual(first, self._abi_digest_in(commands))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestMissingPybind11IsNamed(unittest.TestCase):
    """A missing prerequisite has to name itself.

    ``include/torch/extension.h`` includes <pybind11/pybind11.h> and
    <pybind11/stl.h> unconditionally, so an extension cannot build without the
    headers. Before this, ``_common_includes`` simply dropped the ``-I`` and
    let g++ fail, and the flash-attention bridge reported the result as
    ``import flash_attn_jittor_cuda failed: No module named
    'flash_attn_jittor_cuda'`` -- a symptom that sends the reader looking for a
    missing Python module rather than a missing header.
    """

    def _config(self, pybind_inc):
        return {
            "pybind_inc": pybind_inc,
            "src_inc": "/src", "extern_inc": "/extern",
            "extern_cuda_inc": "/extern/cuda", "py_inc": "/py",
            "cuda_includes": (), "core_dirs": (),
        }

    def test_the_error_names_pybind11_and_how_to_supply_it(self):
        from jittor.compat.shim.cpp_extension import _common_includes
        with self.assertRaises(RuntimeError) as caught:
            _common_includes(self._config(None), [])
        message = str(caught.exception)
        self.assertIn("pybind11", message)
        self.assertIn("PYTHONPATH", message)

    def test_a_found_include_still_reaches_the_command_line(self):
        # The guard must not cost the ordinary path its include.
        from jittor.compat.shim.cpp_extension import _common_includes
        flags = _common_includes(self._config("/somewhere/pybind11/include"), [])
        self.assertIn("-I/somewhere/pybind11/include", flags)


class TestCoreHeadersAreOnTheIncludePath(unittest.TestCase):
    """`src_inc` has to find `core/common.h` in a checkout, not just a wheel.

    4.15 moved the C++ core out of the Python package to the repo top level.
    The config used to join `jittor_path/src`, which is the installed-wheel
    layout; from a checkout it named a directory that does not exist, so every
    extension failed on `#include "core/common.h"`. The flash-attention bridge
    reported that as `No module named 'flash_attn_jittor_cuda'`, so the cause
    never reached the caller. `core_root()` exists for exactly this and names
    `core/common.h` as its marker.
    """

    def test_src_inc_contains_the_marker_header(self):
        import os
        from jittor.compat.shim.cpp_extension import cfg
        src_inc = cfg()["src_inc"]
        self.assertTrue(
            os.path.isfile(os.path.join(src_inc, "core", "common.h")),
            "src_inc=%r has no core/common.h, so every extension built "
            "through the shim will fail to compile" % (src_inc,))
