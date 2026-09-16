"""A C++ extension must see real CUDA indices, and the guard must move jittor too.

Two defects met here, and each hid the other:

1. ``Tensor::device()``/``get_device()`` in the ABI header hardcoded index 0 for
   every CUDA tensor, so ``at::cuda::CUDAGuard{q.device()}`` -- how flash-attn
   and other extensions bind their launch device -- bound device 0 no matter
   which device the tensor was on.
2. The shim's tensor factories (``torch::empty`` and friends, which an extension
   uses for its ``out``/``rng_state``/``softmax_lse_accum``/``out_accum``
   buffers) build their Var from *jittor's* current device, and ``cudaSetDevice``
   does not move that one.

Fixing either alone left the fault identical: a device-1 extension still handed
its kernels a mixture of device-1 inputs and device-0 buffers, and the kernel
died with ``cudaErrorIllegalAddress`` -- while index 0, where both notions of
"current device" already agree, was fine. The probe below reports both, so a
regression in either half fails here rather than as an illegal address inside an
extension.

Run:
    python -m pytest compat/tests/torch/test_cpp_extension_device_index.py
"""
import os
import tempfile
import unittest

from _helpers import capability as _test_capability

import jittor as jt

_PROBE_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

int64_t tensor_device_index(torch::Tensor x) {
    return (int64_t)x.device().index();
}

int64_t tensor_get_device(torch::Tensor x) {
    return x.get_device();
}

// The device a kernel launched here would actually run on: `<<<...>>>` follows
// the CUDA current device, which is what CUDAGuard binds.
int64_t cuda_device_inside_guard(torch::Tensor x) {
    at::cuda::CUDAGuard guard{x.device()};
    int dev = -1;
    if (cudaGetDevice(&dev) != cudaSuccess) { cudaGetLastError(); return -2; }
    return (int64_t)dev;
}

// Where a fresh tensor lands while the guard for `x` is alive. The shim's
// factories build their Var from jittor's current device, not from
// cudaSetDevice's, so before the fix this was jittor's device (0), not x's.
int64_t allocation_device_inside_guard(torch::Tensor x) {
    at::cuda::CUDAGuard guard{x.device()};
    auto fresh = torch::empty(
        {4}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    return (int64_t)fresh.device().index();
}

// The guard is scoped: the current device has to be back where it started.
int64_t cuda_device_after_guard(torch::Tensor x) {
    {
        at::cuda::CUDAGuard guard{x.device()};
    }
    int dev = -1;
    if (cudaGetDevice(&dev) != cudaSuccess) { cudaGetLastError(); return -2; }
    return (int64_t)dev;
}

// Same, for the optional guard a few extensions use instead.
int64_t allocation_device_inside_optional_guard(torch::Tensor x) {
    at::cuda::OptionalCUDAGuard guard{x.device()};
    auto fresh = torch::empty(
        {4}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    return (int64_t)fresh.device().index();
}
"""


def _device_count():
    try:
        return int(jt.get_device_count())
    except Exception:
        return 0


def _shim_cpp_extension_available():
    try:
        import torch
        from torch.utils.cpp_extension import load_inline  # noqa: F401
        if not issubclass(torch.Tensor, jt.Var):
            return False
    except ImportError:
        return False
    return (_test_capability.check_accelerator("cuda", backend=jt).enabled
            and bool(jt.introspection.policy.startup.nvcc_path))


_PROBE_NAME = "jt_cpp_extension_device_probe"
_probe_module = None


def _probe():
    """The probe extension, built once per process.

    Built lazily rather than in ``setUpClass`` so the file also runs under the
    repo's path-loading runner, which instantiates test cases without calling
    class-level setup.
    """
    global _probe_module
    if _probe_module is not None:
        return _probe_module
    if not _shim_cpp_extension_available():
        raise unittest.SkipTest("needs deployed torch-shim + nvcc")
    if _device_count() < 2:
        raise unittest.SkipTest(
            "this machine has %d visible CUDA device(s), the test needs 2"
            % _device_count())

    from torch.utils.cpp_extension import load_inline

    _probe_module = load_inline(
        name=_PROBE_NAME,
        cpp_sources=_PROBE_SRC,
        functions=[
            "tensor_device_index", "tensor_get_device",
            "cuda_device_inside_guard", "allocation_device_inside_guard",
            "cuda_device_after_guard",
            "allocation_device_inside_optional_guard",
        ],
        build_directory=os.path.join(tempfile.gettempdir(), _PROBE_NAME),
        verbose=False,
    )
    return _probe_module


class TestCppExtensionDeviceIndex(unittest.TestCase):
    def _prepare(self):
        """Pin the ambient device to 0 and return the probe module.

        Done here rather than in ``setUp``: the repo's path-loading runner
        instantiates a case and calls the test method directly, without running
        ``setUp``/``setUpClass``. The failing shape in every report was
        "jittor's ambient device is 0, the tensors are on 1", so the ambient
        device is pinned and each tensor moved explicitly.
        """
        import torch
        module = _probe()
        torch.cuda.set_device(0)
        jt.set_device(0)
        return module

    def _tensor_on(self, index):
        import torch
        return torch.zeros(4, dtype=torch.float32).to("cuda:%d" % index)

    def test_device_index_is_reported_for_every_index(self):
        module = self._prepare()
        for index in range(_device_count()):
            x = self._tensor_on(index)
            self.assertEqual(module.tensor_device_index(x), index,
                             "device().index() lied about cuda:%d" % index)
            self.assertEqual(module.tensor_get_device(x), index,
                             "get_device() lied about cuda:%d" % index)

    def test_cpu_tensor_reports_no_device(self):
        import torch
        module = self._prepare()
        x = torch.zeros(4, dtype=torch.float32).cpu()
        self.assertEqual(module.tensor_get_device(x), -1)

    def test_guard_binds_the_running_device(self):
        module = self._prepare()
        for index in range(_device_count()):
            x = self._tensor_on(index)
            self.assertEqual(
                module.cuda_device_inside_guard(x), index,
                "an extension on cuda:%d would have launched on the wrong "
                "device" % index)

    def test_guard_moves_jittor_to_the_tensors_device(self):
        module = self._prepare()
        for index in range(_device_count()):
            x = self._tensor_on(index)
            self.assertEqual(
                module.allocation_device_inside_guard(x), index,
                "an extension on cuda:%d allocated its buffer elsewhere" % index)

    def test_optional_guard_moves_jittor_to_the_tensors_device(self):
        module = self._prepare()
        for index in range(_device_count()):
            x = self._tensor_on(index)
            self.assertEqual(
                module.allocation_device_inside_optional_guard(x), index,
                "OptionalCUDAGuard did not move jittor's device for cuda:%d" % index)

    def test_guard_restores_the_running_device(self):
        module = self._prepare()
        x = self._tensor_on(_device_count() - 1)
        self.assertEqual(module.cuda_device_after_guard(x), 0,
                         "the guard leaked the CUDA current device")


if __name__ == "__main__":
    unittest.main(verbosity=2)
