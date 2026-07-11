"""Smoke tests for the Jittor-backed ``torch.utils.cpp_extension`` ABI shim.

3D Gaussian Splatting builds PyTorch-style C++/CUDA extensions
(``#include <torch/extension.h>``) for packages such as
diff-gaussian-rasterization, simple-knn, and fused-ssim. This test locks the
minimal contract those packages need: the deployed ``import torch`` shim exposes
``torch.utils.cpp_extension``, and a compiled pybind extension can receive/return
jittor Vars through ``torch::Tensor``.

Run:
    python -m jittor.test.test_torch_cpp_extension
"""
import os
import tempfile
import unittest
from unittest import mock

import jittor as jt


class TestTorchCppExtensionArchFlags(unittest.TestCase):
    def test_uses_detected_jittor_archs(self):
        from jittor.torch_shim.cpp_extension import _cuda_arch_flags

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
        from jittor.torch_shim.cpp_extension import _cuda_arch_flags

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
        from jittor.torch_shim.cpp_extension import _torch_cuda_arch_flags

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
        from jittor.torch_shim.cpp_extension import _cuda_arch_flags

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


def _torch_cpp_extension_available():
    try:
        import torch
        from torch.utils.cpp_extension import load_inline  # noqa: F401
        # This test is meaningful only when bare import torch is the deployed
        # jittor torch-shim, not a real PyTorch install.
        if not isinstance(torch.tensor([1]), jt.Var):
            return False
        from jittor import compiler
        return bool(getattr(compiler, "nvcc_path", ""))
    except Exception:
        return False


@unittest.skipUnless(_torch_cpp_extension_available(),
                     "needs deployed torch-shim + nvcc")
class TestTorchCppExtension(unittest.TestCase):
    def _build_probe_extension(self):
        import torch
        from torch.utils.cpp_extension import load_inline

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
        import jittor as torch
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
