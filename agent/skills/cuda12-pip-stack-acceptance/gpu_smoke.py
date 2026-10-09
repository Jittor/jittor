"""GPU smoke for a jittor[cuda12] environment, with provenance checks.

Runs an add, a matmul forward/backward and a cuDNN conv2d on the visible GPU,
then asserts that nvcc, libcudart and libcudnn all come from this
interpreter's ``site-packages/nvidia``. Decoy roots that must not be used are
passed in ``CUDA12_PIP_FORBIDDEN`` (os.pathsep separated). Prints one JSON
line prefixed ``CUDA12_PIP_SMOKE=`` and exits non-zero on the first failure.
"""

import ctypes
import json
import os
import subprocess
import sys
import sysconfig


def fail(message):
    print("CUDA12_PIP_SMOKE_FAIL: " + message, flush=True)
    sys.exit(1)


purelib = os.path.realpath(sysconfig.get_paths()["purelib"])
prefix = os.path.realpath(sys.prefix)
nvidia_root = os.path.join(purelib, "nvidia") + os.sep
forbidden = [p for p in os.environ.get("CUDA12_PIP_FORBIDDEN", "").split(os.pathsep) if p]
if not purelib.startswith(prefix + os.sep):
    fail("site-packages %s is outside the environment %s" % (purelib, prefix))
if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    fail("CUDA_VISIBLE_DEVICES must select the card explicitly")
for name in ("nvcc_path", "JT_BUILD_NVCC_PATH"):
    if name in os.environ:
        fail("%s is set; the pip compiler must be found without it" % name)
for name in ("PATH", "LD_LIBRARY_PATH", "CUDA_HOME", "CUDA_PATH"):
    if nvidia_root.rstrip(os.sep) in os.environ.get(name, ""):
        fail("%s already points at the pip CUDA wheels" % name)


def is_forbidden(path):
    candidates = {path, os.path.realpath(path)}
    return any(c.startswith(root) for c in candidates for root in forbidden)


def from_env(path):
    real = os.path.realpath(path)
    return real.startswith(nvidia_root) and not is_forbidden(path)


import numpy as np
import jittor as jt

if not jt.has_cuda:
    fail("jittor reports no CUDA")
jt.flags.use_cuda = 1

_libcuda = ctypes.CDLL("libcuda.so.1")
CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9


def device_of(var):
    value = ctypes.c_int(-1)
    status = _libcuda.cuPointerGetAttribute(
        ctypes.byref(value), CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
        ctypes.c_void_p(var.device_raw_ptr))
    return value.value if status == 0 else None


report = {"python": sys.version.split()[0], "prefix": prefix}

a = jt.array(np.arange(8, dtype="float32"))
b = a + a
if device_of(b) != 0:
    fail("add result is not in GPU memory")
if not np.array_equal(b.numpy(), np.arange(8, dtype="float32") * 2):
    fail("add returned %s" % b.numpy())
report["add_sum"] = float(b.sum().item())

rng = np.random.default_rng(0)
x_np = rng.standard_normal((16, 32)).astype("float32")
w_np = rng.standard_normal((32, 8)).astype("float32")
x, w = jt.array(x_np), jt.array(w_np)
y = jt.matmul(x, w)
gx, gw = jt.grad(y.sum(), [x, w])
for name, var in (("matmul", y), ("grad_x", gx), ("grad_w", gw)):
    if device_of(var) != 0:
        fail("%s is not in GPU memory" % name)
ones = np.ones((16, 8), dtype="float32")
for name, got, want in (("matmul", y.numpy(), x_np @ w_np),
                        ("grad_x", gx.numpy(), ones @ w_np.T),
                        ("grad_w", gw.numpy(), x_np.T @ ones)):
    if not np.allclose(got, want, rtol=1e-3, atol=1e-3):
        fail("%s differs from numpy by %g" % (name, np.abs(got - want).max()))
report["matmul"] = "forward+backward ok"

image = rng.standard_normal((1, 3, 16, 16)).astype("float32")
kernel = rng.standard_normal((4, 3, 3, 3)).astype("float32")
with jt.log_capture_scope(log_silent=1, log_v=0,
                          log_vprefix="cudnn_conv=100") as logs:
    conv = jt.nn.conv2d(jt.array(image), jt.array(kernel), padding=1)
    conv.sync()
if not any("cudnn_conv precision select" in entry["msg"] for entry in logs):
    fail("conv2d did not run through cudnn_conv")
if device_of(conv) != 0:
    fail("conv2d result is not in GPU memory")
padded = np.pad(image, ((0, 0), (0, 0), (1, 1), (1, 1)))
want = np.zeros((1, 4, 16, 16), dtype="float32")
for i in range(16):
    for j in range(16):
        patch = padded[0, :, i:i + 3, j:j + 3]
        want[0, :, i, j] = np.tensordot(kernel, patch, axes=([1, 2, 3], [0, 1, 2]))
if not np.allclose(conv.numpy(), want, rtol=1e-3, atol=1e-3):
    fail("conv2d differs from numpy by %g" % np.abs(conv.numpy() - want).max())
report["conv2d"] = list(conv.shape)

nvcc = jt.flags.nvcc_path
if not from_env(nvcc):
    fail("nvcc %s is not this environment's pip nvcc" % nvcc)
report["nvcc"] = os.path.realpath(nvcc)
report["nvcc_version"] = [
    line for line in subprocess.run(
        [nvcc, "--version"], capture_output=True, text=True).stdout.splitlines()
    if "release" in line]
stack = jt.compiler.cuda_wheel_stack
if stack is None:
    fail("jittor did not activate the CUDA pip wheel stack")
report["stack"] = stack.fingerprint
report["versions"] = dict(stack.versions)

watched = ("libcudart", "libcudnn", "libcublas", "libnvrtc", "libcufft",
           "libcurand", "libcusparse", "libnvJitLink", "libnccl")
mapped = set()
with open("/proc/self/maps") as maps:
    for line in maps:
        fields = line.split(None, 5)
        if len(fields) == 6 and os.path.basename(fields[5].strip()).startswith(watched):
            mapped.add(fields[5].strip())
bad = sorted(p for p in mapped if not from_env(p))
if bad:
    fail("CUDA libraries mapped from outside site-packages/nvidia: %s" % bad)
cudart = sorted(p for p in mapped if os.path.basename(p).startswith("libcudart.so."))
cudnn = sorted(p for p in mapped if os.path.basename(p).startswith("libcudnn.so."))
if not cudart or not cudnn:
    fail("libcudart %s / libcudnn %s not mapped" % (cudart, cudnn))
if any(".so.13" in p for p in cudart):
    fail("the CUDA 13 runtime from the compiler wheel was loaded: %s" % cudart)
report["libcudart"] = cudart
report["libcudnn"] = cudnn
report["mapped_cuda_libraries"] = len(mapped)
print("CUDA12_PIP_SMOKE=" + json.dumps(report, sort_keys=True), flush=True)
