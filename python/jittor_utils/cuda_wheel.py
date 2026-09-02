# ***************************************************************
# Copyright (c) 2026 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************

"""Discover a coherent NVIDIA CUDA component-wheel installation.

CUDA 11/12 pip packages install each component below ``site-packages/nvidia``
instead of a single CUDA toolkit root.  Jittor still needs a real ``nvcc``
from the system or JTCUDA; this module only resolves the headers and shared
libraries supplied by pip.
"""

from __future__ import print_function

import collections
import hashlib
import os
import re

try:
    from importlib import metadata as importlib_metadata
except ImportError:  # Python 3.7
    import importlib_metadata


class CudaWheelError(RuntimeError):
    pass


# Keep this matrix in sync with setup.py's ``cuda12`` extra.  Exact versions
# prevent pip from combining a cuDNN wheel with an incompatible CUDA family.
CUDA12_COMPONENTS = (
    ("cuda_runtime", "nvidia-cuda-runtime-cu12", "12.2.140", "nvidia/cuda_runtime"),
    ("cublas", "nvidia-cublas-cu12", "12.2.5.6", "nvidia/cublas"),
    ("cuda_nvrtc", "nvidia-cuda-nvrtc-cu12", "12.2.140", "nvidia/cuda_nvrtc"),
    ("cudnn", "nvidia-cudnn-cu12", "8.9.7.29", "nvidia/cudnn"),
    ("cufft", "nvidia-cufft-cu12", "11.0.8.103", "nvidia/cufft"),
    ("curand", "nvidia-curand-cu12", "10.3.3.141", "nvidia/curand"),
    ("cusparse", "nvidia-cusparse-cu12", "12.1.2.141", "nvidia/cusparse"),
    ("nvjitlink", "nvidia-nvjitlink-cu12", "12.2.140", "nvidia/nvjitlink"),
    ("nvtx", "nvidia-nvtx-cu12", "12.2.140", "nvidia/nvtx"),
    ("nccl", "nvidia-nccl-cu12", "2.18.3", "nvidia/nccl"),
)


LIBRARY_COMPONENTS = {
    "cudart": "cuda_runtime",
    "cublas": "cublas",
    "cublasLt": "cublas",
    "nvrtc": "cuda_nvrtc",
    "nvrtc-builtins": "cuda_nvrtc",
    "cudnn": "cudnn",
    "cudnn_ops_infer": "cudnn",
    "cudnn_ops_train": "cudnn",
    "cudnn_cnn_infer": "cudnn",
    "cudnn_cnn_train": "cudnn",
    "cudnn_adv_infer": "cudnn",
    "cudnn_adv_train": "cudnn",
    "cufft": "cufft",
    "curand": "curand",
    "cusparse": "cusparse",
    "nvJitLink": "nvjitlink",
    "nvToolsExt": "nvtx",
    "nccl": "nccl",
}


PRELOAD_ORDER = {
    "cudart": ("cudart",),
    "cublas": ("cublasLt", "cublas"),
    "cudnn": (
        "cudart",
        "cublasLt",
        "cublas",
        "nvrtc",
        "cudnn_ops_infer",
        "cudnn_ops_train",
        "cudnn_cnn_infer",
        "cudnn_cnn_train",
        "cudnn_adv_infer",
        "cudnn_adv_train",
        "cudnn",
    ),
    "cufft": ("cudart", "cufft"),
    "curand": ("cudart", "curand"),
    "cusparse": ("cudart", "nvJitLink", "cusparse"),
    "nvToolsExt": ("nvToolsExt",),
    "nccl": ("cudart", "nccl"),
}


def _natural_version_key(path):
    name = os.path.basename(path)
    return tuple(int(x) for x in re.findall(r"\d+", name))


def _unique_existing(paths):
    output = []
    for path in paths:
        path = os.path.abspath(os.fspath(path))
        if os.path.isdir(path) and path not in output:
            output.append(path)
    return output


def _version_tuple(version):
    return tuple(int(x) for x in re.findall(r"\d+", str(version))[:3])


def _truthy(value):
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


class CudaWheelStack:
    """Resolved paths and versions for one supported CUDA wheel stack."""

    cuda_version = "12.2"

    def __init__(self, components, versions):
        self.components = dict(components)
        self.versions = dict(versions)
        version_text = ";".join(
            "%s=%s" % (name, self.versions[name])
            for name, _, _, _ in CUDA12_COMPONENTS
        )
        digest = hashlib.sha256(version_text.encode("ascii")).hexdigest()[:12]
        self.fingerprint = "pipcu122_" + digest

    def component_dir(self, component):
        return self.components.get(component)

    def include_dirs(self, component=None):
        if component is not None:
            roots = [self.component_dir(component)]
        else:
            roots = [self.components[name] for name, _, _, _ in CUDA12_COMPONENTS]
        return _unique_existing(
            os.path.join(root, "include") for root in roots if root
        )

    def lib_dirs(self, component=None):
        if component is not None:
            roots = [self.component_dir(component)]
        else:
            roots = [self.components[name] for name, _, _, _ in CUDA12_COMPONENTS]
        return _unique_existing(
            os.path.join(root, "lib") for root in roots if root
        )

    def find_library(self, name, component=None):
        component = component or LIBRARY_COMPONENTS.get(name)
        dirs = self.lib_dirs(component) if component else self.lib_dirs()
        # No macOS branch: NVIDIA publishes no macOS CUDA wheels, so a stack
        # object never exists there to look a .dylib up in.
        if os.name == "nt":
            patterns = (name + "64*.dll", name + "*.dll")
        else:
            patterns = ("lib" + name + ".so", "lib" + name + ".so.*")
        matches = []
        import glob
        for directory in dirs:
            for pattern in patterns:
                matches.extend(glob.glob(os.path.join(directory, pattern)))
        matches = [os.path.abspath(path) for path in matches if os.path.isfile(path)]
        if not matches:
            return None
        # Prefer the ABI SONAME (for example .so.12) over a patch-version file.
        matches.sort(key=lambda path: (
            os.path.basename(path).count("."),
            len(os.path.basename(path)),
            _natural_version_key(path),
        ))
        return matches[0]

    def linker_flags(self, name):
        path = self.find_library(name)
        if not path:
            raise CudaWheelError("CUDA wheel library lib%s was not found" % name)
        directory = os.path.dirname(path)
        if os.name == "nt":
            return '-L"%s" -l%s' % (directory, name)
        return '-L"%s" -l:%s' % (directory, os.path.basename(path))

    def preload_paths(self, name):
        names = PRELOAD_ORDER.get(name, (name,))
        paths = []
        for dependency in names:
            path = self.find_library(dependency)
            if path and path not in paths:
                paths.append(path)
        return paths

    def owns_path(self, path):
        path = os.path.realpath(os.path.abspath(os.path.expanduser(path)))
        for root in self.components.values():
            try:
                if os.path.commonpath((path, os.path.realpath(root))) == os.path.realpath(root):
                    return True
            except ValueError:
                pass
        return False


def _validate_stack(stack):
    required_headers = {
        "cuda_runtime": "cuda_runtime.h",
        "cublas": "cublas.h",
        "cudnn": "cudnn.h",
        "cufft": "cufft.h",
        "curand": "curand.h",
        "cusparse": "cusparse.h",
        "nccl": "nccl.h",
    }
    for component, header in required_headers.items():
        if not any(os.path.isfile(os.path.join(path, header))
                   for path in stack.include_dirs(component)):
            raise CudaWheelError(
                "%s %s is missing from the NVIDIA wheel" % (component, header)
            )
    required_libraries = (
        "cudart", "cublas", "cublasLt", "nvrtc", "nvrtc-builtins", "cudnn",
        "cudnn_ops_infer", "cudnn_ops_train", "cudnn_cnn_infer",
        "cudnn_cnn_train", "cudnn_adv_infer", "cudnn_adv_train",
        "cufft", "curand", "cusparse", "nvJitLink", "nvToolsExt", "nccl",
    )
    for name in required_libraries:
        if not stack.find_library(name):
            raise CudaWheelError("lib%s is missing from the NVIDIA wheel stack" % name)


#: What ``inspect_cuda_wheel_stack`` returns.
#:
#: ``stack``   the resolved stack, or None
#: ``reason``  why it is None, in one sentence, or None on success
#: ``present`` how many of the pinned component distributions are installed at
#:             all, at any version -- the signal for whether the user asked
#:             for this stack or merely has some ``nvidia-*`` wheels because
#:             something else (torch, most often) depends on them
#: ``broken``  the stack resolved completely at the pinned versions and then
#:             failed validation, which no third party's dependencies can
#:             cause: this is a broken ``jittor[cuda12]`` and nothing else
CudaWheelReport = collections.namedtuple(
    "CudaWheelReport", "stack reason present broken")


def inspect_cuda_wheel_stack(nvcc_version=None, distribution=None):
    """Resolve the CUDA 12.2 wheel stack and say why if it cannot be.

    Every one of these failures used to be swallowed -- the diagnostic strings
    below were constructed and then dropped on the floor by a bare
    ``return None``. The user who installed ``jittor[cuda12]`` and then, say,
    let something upgrade one wheel, silently got the *system* CUDA instead,
    and found out several minutes later through an unrelated-looking cuDNN 9
    error. The two events had no visible connection.

    ``distribution`` is injectable for unit tests and follows
    ``importlib.metadata.distribution``'s interface.
    """

    if os.name != "posix":
        return CudaWheelReport(None, None, 0, False)
    if _truthy(os.environ.get("JITTOR_CUDA_WHEEL_DISABLE")):
        return CudaWheelReport(
            None, "JITTOR_CUDA_WHEEL_DISABLE is set", 0, False)
    if nvcc_version and _version_tuple(nvcc_version)[:2] != (12, 2):
        return CudaWheelReport(
            None, "jittor[cuda12] requires nvcc 12.2, found %s" % nvcc_version,
            0, False)

    distribution = distribution or importlib_metadata.distribution
    components = {}
    versions = {}
    reason = None
    for component, dist_name, expected, relative_path in CUDA12_COMPONENTS:
        try:
            dist = distribution(dist_name)
        except importlib_metadata.PackageNotFoundError:
            reason = reason or "%s is not installed" % dist_name
            continue
        actual = str(dist.version)
        if actual != expected:
            reason = reason or (
                "%s==%s is required, found %s" % (dist_name, expected, actual))
            continue
        root = os.path.abspath(os.fspath(dist.locate_file(relative_path)))
        if not os.path.isdir(root):
            reason = reason or (
                "%s does not contain %s" % (dist_name, relative_path))
            continue
        components[component] = root
        versions[component] = actual

    if reason is not None:
        # Only now, and only on the failure path: the success path must not
        # read any distribution's metadata twice.
        return CudaWheelReport(
            None, reason, _count_installed(distribution), False)
    stack = CudaWheelStack(components, versions)
    try:
        _validate_stack(stack)
    except CudaWheelError as error:
        # Every pinned wheel is installed at its pinned version and the set is
        # still unusable. Nothing outside jittor[cuda12] pins this exact
        # combination, so this is that installation and it is broken.
        return CudaWheelReport(
            None, str(error), len(CUDA12_COMPONENTS), True)
    return CudaWheelReport(stack, None, len(CUDA12_COMPONENTS), False)


def _count_installed(distribution):
    """How many of the pinned component distributions exist at any version."""
    present = 0
    for _, dist_name, _, _ in CUDA12_COMPONENTS:
        try:
            distribution(dist_name)
        except Exception:
            continue
        present += 1
    return present


def discover_cuda_wheel_stack(nvcc_version=None, distribution=None, strict=None):
    """Return the supported CUDA 12.2 wheel stack, or ``None``.

    ``strict`` unset means "raise for a failure that can only be a broken
    ``jittor[cuda12]``". That is strict by default for the case where strict
    is safe, and it deliberately stops short of the whole set: a machine that
    has ``nvidia-cublas-cu12`` at some other version because torch depends on
    it is the *normal* case, and raising there would refuse to run on every
    system-CUDA install. ``strict=True`` raises for any failure to resolve.
    """
    report = inspect_cuda_wheel_stack(nvcc_version, distribution)
    if report.stack is not None:
        return report.stack
    if strict is None:
        strict = report.broken
    if strict and report.reason:
        raise CudaWheelError(report.reason)
    return None


def is_nvidia_wheel_path(path):
    """Whether a path has the standard ``site-packages/nvidia`` layout."""

    if not path:
        return False
    normalized = os.path.abspath(os.path.expanduser(path)).replace("\\", "/")
    return "/site-packages/nvidia/" in normalized + "/"
