"""Native Torch-extension shim must expose Jittor's physical view strides."""
import os
import tempfile
import unittest

import jittor as jt

from _helpers import capability as _test_capability


_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
std::vector<int64_t> physical_strides(torch::Tensor x) {
    std::vector<int64_t> result;
    for (int64_t d = 0; d < x.dim(); ++d) result.push_back(x.stride(d));
    return result;
}
void write_second_block_first(torch::Tensor x, int32_t value) {
    auto *target = x.data_ptr<int32_t>() + x.stride(0);
    if (cudaMemcpy(target, &value, sizeof(value), cudaMemcpyHostToDevice)
        != cudaSuccess) throw std::runtime_error("cudaMemcpy failed");
}
"""
_MODULE = None


def _module():
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    try:
        import torch
        from torch.utils.cpp_extension import load_inline
        from jittor.compat.shim.cpp_extension import _find_pybind_include
        if not issubclass(torch.Tensor, jt.Var) or _find_pybind_include() is None:
            raise unittest.SkipTest("requires the Jittor Torch shim and pybind11")
        if not (_test_capability.check_accelerator("cuda", backend=jt).enabled
                and jt.introspection.policy.startup.nvcc_path):
            raise unittest.SkipTest("requires CUDA")
    except ImportError as exc:
        raise unittest.SkipTest("requires the Jittor Torch shim") from exc
    name = "jt_cpp_extension_storage_stride_probe"
    _MODULE = load_inline(
        name=name,
        cpp_sources=_SOURCE,
        functions=["physical_strides", "write_second_block_first"],
        build_directory=os.path.join(tempfile.gettempdir(), name),
        verbose=False,
    )
    return _MODULE


class TestCppExtensionStorageStride(unittest.TestCase):
    def test_interleaved_kv_page_views(self):
        module = _module()
        with jt.flag_scope(use_cuda=1):
            cache = jt.zeros((4, 2, 256, 2, 64), dtype=jt.float16)
            cache.sync()
            key = cache[:, 0]
            value = cache[:, 1]
            key.sync()
            value.sync()
            expected = [2 * 256 * 2 * 64, 2 * 64, 64, 1]
            for view in (key, value):
                view._jittor_torch_ext_readonly_borrow = True
                try:
                    self.assertEqual(module.physical_strides(view), expected)
                finally:
                    del view._jittor_torch_ext_readonly_borrow
            dense = jt.zeros((4, 256, 2, 64), dtype=jt.float16)
            dense.sync()
            dense._jittor_torch_ext_readonly_borrow = True
            try:
                self.assertEqual(module.physical_strides(dense),
                                 [256 * 2 * 64, 2 * 64, 64, 1])
            finally:
                del dense._jittor_torch_ext_readonly_borrow

    def test_mutable_borrow_writes_original_interleaved_cache(self):
        module = _module()
        with jt.flag_scope(use_cuda=1):
            cache = jt.zeros((4, 2, 256, 2, 64), dtype=jt.int32)
            cache.sync()
            key = cache[:, 0]
            key.sync()
            key._jittor_torch_ext_mutable_borrow = True
            try:
                module.write_second_block_first(key, 123)
            finally:
                del key._jittor_torch_ext_mutable_borrow
            actual = cache.numpy()
            self.assertEqual(int(actual[1, 0, 0, 0, 0]), 123)
            self.assertEqual(int(actual[1, 1, 0, 0, 0]), 0)


if __name__ == "__main__":
    unittest.main()
