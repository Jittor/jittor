# ***************************************************************
# Copyright (c) 2026 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************

from __future__ import print_function

import hashlib
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


# Load the leaf module directly: importing ``jittor`` would initialize the
# compiler and turn these filesystem-only tests into CUDA integration tests.
_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "python" / "jittor_utils" / "cuda_wheel.py"
)
_CACHE_ROOT = Path(
    os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
).expanduser()
_TEMP_ROOT = Path(
    os.environ.get(
        "JITTOR_TEST_STATE_ROOT",
        _CACHE_ROOT / "jittor" / "tests",
    )
).expanduser() / "test_cuda_wheel"
cuda_wheel = None


def setUpModule():
    global cuda_wheel
    _TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("jittor_cuda_wheel_test", _MODULE_PATH)
    cuda_wheel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cuda_wheel)


_REQUIRED_HEADERS = {
    "cuda_runtime": "cuda_runtime.h",
    "cublas": "cublas.h",
    "cudnn": "cudnn.h",
    "cufft": "cufft.h",
    "curand": "curand.h",
    "cusparse": "cusparse.h",
    "nccl": "nccl.h",
}

_LIBRARY_ABIS = {
    "cudart": "12",
    "cublas": "12",
    "cublasLt": "12",
    "nvrtc": "12",
    "nvrtc-builtins": "12.2",
    "cudnn": "8",
    "cudnn_ops_infer": "8",
    "cudnn_ops_train": "8",
    "cudnn_cnn_infer": "8",
    "cudnn_cnn_train": "8",
    "cudnn_adv_infer": "8",
    "cudnn_adv_train": "8",
    "cufft": "11",
    "curand": "10",
    "cusparse": "12",
    "nvJitLink": "12",
    "nvToolsExt": "1",
    "nccl": "2",
}


class _FakeDistribution:
    def __init__(self, version, site_packages):
        self.version = version
        self.site_packages = Path(site_packages)

    def locate_file(self, relative_path):
        return self.site_packages / relative_path


class _DistributionRegistry:
    def __init__(self, distributions):
        self.distributions = distributions
        self.calls = []

    def __call__(self, name):
        self.calls.append(name)
        try:
            return self.distributions[name]
        except KeyError:
            raise cuda_wheel.importlib_metadata.PackageNotFoundError(name)


class _WheelStackFixture:
    """Build a complete wheel stack with every package under its own prefix."""

    def __init__(self, base):
        self.base = Path(base) / "wheel roots"
        self.components = {}
        self.distributions = {}
        self.library_paths = {}

        for component, dist_name, version, relative_path in cuda_wheel.CUDA12_COMPONENTS:
            site_packages = self.base / dist_name / "site-packages"
            component_root = site_packages / relative_path
            (component_root / "include").mkdir(parents=True)
            (component_root / "lib").mkdir()
            self.components[component] = component_root.resolve()
            self.distributions[dist_name] = _FakeDistribution(version, site_packages)

        for component, header in _REQUIRED_HEADERS.items():
            (self.components[component] / "include" / header).touch()

        for name, abi in _LIBRARY_ABIS.items():
            component = cuda_wheel.LIBRARY_COMPONENTS[name]
            path = self.components[component] / "lib" / ("lib%s.so.%s" % (name, abi))
            path.touch()
            self.library_paths[name] = path.resolve()

        self.registry = _DistributionRegistry(self.distributions)

    def discover(self, nvcc_version="12.2.140", strict=True):
        return cuda_wheel.discover_cuda_wheel_stack(
            nvcc_version=nvcc_version,
            distribution=self.registry,
            strict=strict,
        )


@unittest.skipUnless(os.name == "posix" and os.uname().sysname == "Linux",
                     "CUDA component wheels use Linux shared-library names")
class TestCudaWheel(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory(dir=str(_TEMP_ROOT))
        self.addCleanup(self.temporary_directory.cleanup)
        self.fixture = _WheelStackFixture(self.temporary_directory.name)
        disable = mock.patch.dict(
            os.environ, {"JITTOR_CUDA_WHEEL_DISABLE": ""}, clear=False
        )
        disable.start()
        self.addCleanup(disable.stop)

    def test_discovers_exact_versions_across_distribution_roots(self):
        stack = self.fixture.discover()

        expected_packages = [entry[1] for entry in cuda_wheel.CUDA12_COMPONENTS]
        self.assertEqual(self.fixture.registry.calls, expected_packages)
        self.assertEqual(stack.components, {
            component: str(self.fixture.components[component])
            for component, _, _, _ in cuda_wheel.CUDA12_COMPONENTS
        })
        self.assertEqual(stack.versions, {
            component: version
            for component, _, version, _ in cuda_wheel.CUDA12_COMPONENTS
        })
        self.assertEqual(len(stack.include_dirs()), len(cuda_wheel.CUDA12_COMPONENTS))
        self.assertEqual(len(stack.lib_dirs()), len(cuda_wheel.CUDA12_COMPONENTS))

    def test_versioned_only_shared_object_gets_exact_linker_flag(self):
        stack = self.fixture.discover()
        library = self.fixture.library_paths["cudnn"]

        self.assertFalse((library.parent / "libcudnn.so").exists())
        self.assertEqual(stack.find_library("cudnn"), str(library))
        self.assertEqual(
            stack.linker_flags("cudnn"),
            '-L"%s" -l:libcudnn.so.8' % library.parent,
        )

    def test_preload_paths_follow_dependency_order(self):
        stack = self.fixture.discover()

        expected = [
            str(self.fixture.library_paths[name])
            for name in cuda_wheel.PRELOAD_ORDER["cudnn"]
        ]
        self.assertEqual(stack.preload_paths("cudnn"), expected)
        self.assertEqual(
            stack.preload_paths("cusparse"),
            [
                str(self.fixture.library_paths[name])
                for name in cuda_wheel.PRELOAD_ORDER["cusparse"]
            ],
        )

    def test_wrong_component_version_is_rejected(self):
        package = "nvidia-cudnn-cu12"
        self.fixture.distributions[package].version = "8.9.6.50"

        self.assertIsNone(self.fixture.discover(strict=False))
        with self.assertRaisesRegex(
                cuda_wheel.CudaWheelError,
                r"nvidia-cudnn-cu12==8\.9\.7\.29 is required, found 8\.9\.6\.50"):
            self.fixture.discover(strict=True)

    def test_wrong_nvcc_version_is_rejected_before_metadata_lookup(self):
        def unexpected_distribution(_):
            self.fail("distribution metadata should not be read for incompatible nvcc")

        self.assertIsNone(cuda_wheel.discover_cuda_wheel_stack(
            nvcc_version="12.6.85",
            distribution=unexpected_distribution,
            strict=False,
        ))
        with self.assertRaisesRegex(
                cuda_wheel.CudaWheelError,
                r"requires nvcc 12\.2, found 11\.8\.89"):
            cuda_wheel.discover_cuda_wheel_stack(
                nvcc_version="11.8.89",
                distribution=unexpected_distribution,
                strict=True,
            )

    def test_fingerprint_is_order_independent_and_version_sensitive(self):
        stack = self.fixture.discover()
        version_text = ";".join(
            "%s=%s" % (component, version)
            for component, _, version, _ in cuda_wheel.CUDA12_COMPONENTS
        )
        digest = hashlib.sha256(version_text.encode("ascii")).hexdigest()[:12]
        self.assertEqual(stack.fingerprint, "pipcu122_" + digest)

        reversed_stack = cuda_wheel.CudaWheelStack(
            dict(reversed(list(stack.components.items()))),
            dict(reversed(list(stack.versions.items()))),
        )
        self.assertEqual(reversed_stack.fingerprint, stack.fingerprint)

        changed_versions = dict(stack.versions)
        changed_versions["cudnn"] = "8.9.7.30"
        changed_stack = cuda_wheel.CudaWheelStack(stack.components, changed_versions)
        self.assertNotEqual(changed_stack.fingerprint, stack.fingerprint)


if __name__ == "__main__":
    unittest.main(verbosity=2)
