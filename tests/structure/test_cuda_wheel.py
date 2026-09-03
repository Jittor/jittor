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

# The cuDNN libraries are added per fixture: which ones exist, and at which
# ABI, is what the cuDNN major version decides.
_LIBRARY_ABIS = {
    "cudart": "12",
    "cublas": "12",
    "cublasLt": "12",
    "nvrtc": "12",
    "nvrtc-builtins": "12.2",
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
    """Build a complete wheel stack with every package under its own prefix.

    ``cudnn_version`` overrides the cuDNN wheel's version, which is the one
    component jittor accepts a range of. The split libraries laid down follow
    that version's major, so a cuDNN 9 fixture contains cuDNN 9's libraries
    and none of cuDNN 8's.
    """

    def __init__(self, base, cudnn_version=None):
        self.base = Path(base) / "wheel roots"
        self.components = {}
        self.distributions = {}
        self.library_paths = {}
        self.versions = {}

        for component, dist_name, specifier, relative_path in cuda_wheel.CUDA12_COMPONENTS:
            version = cuda_wheel.reference_version(specifier)
            if component == "cudnn" and cudnn_version is not None:
                version = cudnn_version
            self.versions[component] = version
            site_packages = self.base / dist_name / "site-packages"
            component_root = site_packages / relative_path
            (component_root / "include").mkdir(parents=True)
            (component_root / "lib").mkdir()
            self.components[component] = Path(os.path.abspath(component_root))
            self.distributions[dist_name] = _FakeDistribution(version, site_packages)

        for component, header in _REQUIRED_HEADERS.items():
            (self.components[component] / "include" / header).touch()

        self.cudnn_major = int(self.versions["cudnn"].split(".")[0])
        abis = dict(_LIBRARY_ABIS)
        abis["cudnn"] = str(self.cudnn_major)
        for name in cuda_wheel.CUDNN_SPLIT_LIBRARIES.get(self.cudnn_major, ()):
            abis[name] = str(self.cudnn_major)

        for name, abi in abis.items():
            component = cuda_wheel.LIBRARY_COMPONENTS[name]
            path = self.components[component] / "lib" / ("lib%s.so.%s" % (name, abi))
            path.touch()
            self.library_paths[name] = Path(os.path.abspath(path))

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
        self.assertEqual(stack.versions, self.fixture.versions)
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
            for name in cuda_wheel.cudnn_preload_order(8)
        ]
        self.assertEqual(stack.preload_order("cudnn"),
                         cuda_wheel.cudnn_preload_order(8))
        self.assertEqual(stack.preload_paths("cudnn"), expected)
        self.assertEqual(
            stack.preload_paths("cusparse"),
            [
                str(self.fixture.library_paths[name])
                for name in cuda_wheel.PRELOAD_ORDER["cusparse"]
            ],
        )

    def test_wrong_component_version_is_rejected(self):
        package = "nvidia-cuda-runtime-cu12"
        self.fixture.distributions[package].version = "12.2.139"

        self.assertIsNone(self.fixture.discover(strict=False))
        with self.assertRaisesRegex(
                cuda_wheel.CudaWheelError,
                r"nvidia-cuda-runtime-cu12==12\.2\.140 is required, "
                r"found 12\.2\.139"):
            self.fixture.discover(strict=True)

    def test_cudnn_is_a_range_because_torch_pins_its_own(self):
        """An exact cuDNN pin made jittor[cuda12] and torch uninstallable.

        pip resolves two different ``==`` pins on one distribution as a
        conflict, and every modern torch pins its own cuDNN 9. So this one
        component is a range -- and the range is what is checked.
        """
        self.fixture.distributions["nvidia-cudnn-cu12"].version = "8.9.6.50"
        self.assertIsNone(self.fixture.discover(strict=False))
        with self.assertRaisesRegex(
                cuda_wheel.CudaWheelError,
                r"nvidia-cudnn-cu12>=8\.9\.7,<10 is required, found 8\.9\.6\.50"):
            self.fixture.discover(strict=True)

        # Inside the range, any patch level is accepted rather than one exact
        # string, which is the whole point.
        self.fixture.distributions["nvidia-cudnn-cu12"].version = "8.9.7.29"
        self.assertIsNotNone(self.fixture.discover())

    def test_a_declined_stack_says_why(self):
        """The diagnostic used to be built and then dropped by `return None`.

        The user who installed jittor[cuda12] and then let something upgrade
        one wheel got the system CUDA in silence, and met the consequence
        several minutes later as an unrelated-looking error with no visible
        cause.
        """
        self.fixture.distributions["nvidia-cudnn-cu12"].version = "8.9.6.50"
        report = cuda_wheel.inspect_cuda_wheel_stack(
            nvcc_version="12.2.140", distribution=self.fixture.registry)

        self.assertIsNone(report.stack)
        self.assertIn("nvidia-cudnn-cu12>=8.9.7,<10 is required, found 8.9.6.50",
                      report.reason)
        # Some of the stack is installed, so the caller should say so out loud
        # rather than at log level v.
        self.assertEqual(report.present, len(cuda_wheel.CUDA12_COMPONENTS))
        self.assertFalse(report.broken)

    def test_a_machine_with_none_of_the_stack_is_not_a_broken_install(self):
        registry = _DistributionRegistry({})
        report = cuda_wheel.inspect_cuda_wheel_stack(
            nvcc_version="12.2.140", distribution=registry)

        self.assertIsNone(report.stack)
        self.assertEqual(report.present, 0)
        self.assertFalse(report.broken)
        # And it must not raise by default: running against the system CUDA is
        # a supported configuration, and torch pulls in some of these wheels
        # at its own versions on plenty of machines.
        self.assertIsNone(cuda_wheel.discover_cuda_wheel_stack(
            nvcc_version="12.2.140", distribution=registry))

    def test_a_complete_but_unusable_stack_is_fatal_by_default(self):
        """Nothing but a broken jittor[cuda12] can produce this.

        Every pinned wheel is installed at its pinned version and the set is
        still missing a library, so there is no third party whose dependency
        resolution could have caused it and nothing to fall back to quietly.
        """
        library = self.fixture.library_paths["cudnn"]
        library.unlink()

        report = cuda_wheel.inspect_cuda_wheel_stack(
            nvcc_version="12.2.140", distribution=self.fixture.registry)
        self.assertIsNone(report.stack)
        self.assertTrue(report.broken)
        self.assertIn("libcudnn", report.reason)

        with self.assertRaises(cuda_wheel.CudaWheelError):
            cuda_wheel.discover_cuda_wheel_stack(
                nvcc_version="12.2.140", distribution=self.fixture.registry)

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
            "%s=%s" % (component, self.fixture.versions[component])
            for component, _, _, _ in cuda_wheel.CUDA12_COMPONENTS
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

        # cuDNN is a range now, so two accepted cuDNNs coexist in the wild.
        # They must not share a compilation cache: the headers differ.
        cudnn9 = dict(stack.versions, cudnn="9.5.1.17")
        self.assertNotEqual(
            cuda_wheel.CudaWheelStack(stack.components, cudnn9).fingerprint,
            stack.fingerprint)


@unittest.skipUnless(os.name == "posix" and os.uname().sysname == "Linux",
                     "CUDA component wheels use Linux shared-library names")
class TestCudnnMajorDecidesTheLayout(unittest.TestCase):
    """cuDNN 9 renamed every split library, so the major picks the set.

    Before cuDNN 9 was supported at all, the six cuDNN 8 names were a
    constant. Validating and preloading them on a cuDNN 9 install fails at
    "libcudnn_ops_infer.so is missing" -- a name that no longer exists in any
    cuDNN, which says nothing about the version being the reason.
    """

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory(dir=str(_TEMP_ROOT))
        self.addCleanup(self.temporary_directory.cleanup)
        disable = mock.patch.dict(
            os.environ, {"JITTOR_CUDA_WHEEL_DISABLE": ""}, clear=False
        )
        disable.start()
        self.addCleanup(disable.stop)

    def _fixture(self, cudnn_version):
        return _WheelStackFixture(self.temporary_directory.name, cudnn_version)

    def test_the_split_library_names_are_disjoint_between_majors(self):
        self.assertFalse(
            set(cuda_wheel.CUDNN_SPLIT_LIBRARIES[8])
            & set(cuda_wheel.CUDNN_SPLIT_LIBRARIES[9]),
            "if any name were shared, testing one major would not prove the "
            "other is handled")

    def test_a_cudnn_9_stack_resolves_and_preloads_cudnn_9s_libraries(self):
        fixture = self._fixture("9.5.1.17")
        stack = fixture.discover()

        self.assertEqual(stack.cudnn_major, 9)
        self.assertEqual(stack.versions["cudnn"], "9.5.1.17")
        self.assertEqual(
            stack.linker_flags("cudnn"),
            '-L"%s" -l:libcudnn.so.9' % fixture.library_paths["cudnn"].parent)
        self.assertEqual(stack.preload_order("cudnn"),
                         cuda_wheel.cudnn_preload_order(9))
        self.assertEqual(
            stack.preload_paths("cudnn"),
            [str(fixture.library_paths[name])
             for name in cuda_wheel.cudnn_preload_order(9)])

    def test_a_cudnn_9_stack_is_not_asked_for_cudnn_8s_libraries(self):
        """The fixture lays down no cudnn_ops_infer, and that is not a defect."""
        fixture = self._fixture("9.5.1.17")
        self.assertNotIn("cudnn_ops_infer", fixture.library_paths)
        self.assertIsNotNone(fixture.discover())

    def test_a_cudnn_8_stack_still_needs_cudnn_8s_libraries(self):
        fixture = self._fixture("8.9.7.29")
        (fixture.library_paths["cudnn_adv_train"]).unlink()

        report = cuda_wheel.inspect_cuda_wheel_stack(
            nvcc_version="12.2.140", distribution=fixture.registry)
        self.assertIsNone(report.stack)
        self.assertIn("libcudnn_adv_train", report.reason)

    def test_an_unsupported_cudnn_major_is_named_as_the_reason(self):
        """A cuDNN 10 wheel is out of range, and the message says so.

        Without this the failure would be "libcudnn_graph is missing", which
        blames a file for the version that has no such file.
        """
        fixture = self._fixture("10.0.0.1")
        report = cuda_wheel.inspect_cuda_wheel_stack(
            nvcc_version="12.2.140", distribution=fixture.registry)

        self.assertIsNone(report.stack)
        self.assertIn("nvidia-cudnn-cu12>=8.9.7,<10 is required, found 10.0.0.1",
                      report.reason)


if __name__ == "__main__":
    unittest.main(verbosity=2)
