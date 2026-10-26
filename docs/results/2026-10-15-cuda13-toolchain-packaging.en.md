# CUDA 13.0 toolchain wheels and uv validation

- Status: verified on Linux x86-64 / RTX 4090 (SM 89); toolchain and the
  explicitly listed library combination below. This is not a full CUDA 13
  math-library stack certification.
- Baseline: `dd1013dbb422c70147efe8f1c8ab2720403a2c92`, after integrating
  `origin/2.0-refactor` at `ad47b5814` and retaining the AMP/SDPA commits.
- Implementation: the commit containing this report; parent integration
  should record its final merge SHA when registering this report.
- Maintainer: Jittor maintainers. Recheck on compiler/runtime/CCCL upgrades,
  Python-marker changes, or another driver, architecture or library mixture.

## Reproduction and fixes

Official PyPI wheels were downloaded, checked against their published SHA-256
digests and installed into independent state directories. The relevant
[nvcc release](https://pypi.org/project/nvidia-cuda-nvcc/13.0.88/),
[CCCL wheel release](https://pypi.org/project/nvidia-cuda-cccl/13.0.85/) and
[NVIDIA CCCL installation guide](https://nvidia.github.io/cccl/unstable/python/setup.html)
are primary references. The CCCL Python API package is distinct from the
`nvidia-cuda-cccl` C++ header wheel used by Jittor; no Python CCCL API dependency
was added.

The real installed layout is `nvidia/cu13`, with the CUB/Thrust/libcu++ headers
under `include/cccl`. The old provider immediately rejected nvcc 13.0.88 with
“requires nvcc 12.2”. An actual compiler invocation against the installed CUB
headers failed because CUB requires C++17. The initial framework build also
failed on `cudaDeviceProp.computeMode` and `clockRate`, removed in CUDA 13.

The new `cuda13` extra pins the compiler, runtime, CRT, NVVM and CCCL together.
The provider verifies their shared prefix, executable compiler, device
compiler/libdevice, runtime, device-runtime archive and CCCL headers. An
installed but partial or mismatched CUDA 13 wheel toolchain raises instead of
silently selecting another installation. With no explicit/system compiler,
Jittor discovers the wheel's nvcc. CUDA 13 builds use C++17 on both host and
device; CUDA 12.2 keeps its existing dialect and fingerprint.

Two further actual linker failures were resolved: the wheel supplies `lib/`
and `libcudart.so.13`, whereas nvcc's default shared-runtime arguments request
`lib64/` and unversioned `libcudart.so`. CUDA 13 wheel compilation now supplies
the library directory and exact runtime SONAME explicitly, preserving the
default device-runtime archive. Device properties use the supported attribute
queries on both CUDA versions. Library setup distinguishes a toolchain-only
provider from the complete CUDA 12.2 library stack.

Actual universal uv resolution exposed an existing Python 3.9 incompatibility:
`nvidia-cudnn-frontend>=1.30` requires Python 3.10. Its extra marker now matches
that requirement, and the regenerated lock resolves successfully.

## Actual platform and libraries

Python 3.11.16, g++ 11.5.0, glibc 2.34, NVIDIA driver 580.178.04 and SM 89.
No system toolchain or shared Python environment was modified. Separate raw
logs, wheel archives, venvs and JIT caches remain outside the checkout.

| Component | Wheel / actual version | Observed library/header source |
| --- | --- | --- |
| nvcc, CRT, NVVM | 13.0.88 | Private `nvidia/cu13` compiler, CRT, NVVM/libdevice |
| Runtime | 13.0.96; API reports 13000 | Private `nvidia/cu13/lib/libcudart.so.13` |
| CCCL | 13.0.85 | Private `nvidia/cu13/include/cccl` |
| cuBLAS / cuBLASLt | `nvidia-cublas==13.0.0.19`; API 13.0.0 | Real private `libcublas.so.13` / `libcublasLt.so.13` |
| cuDNN | Existing cu12 wheel; API 92400 | External `nvidia/cudnn/lib/libcudnn.so.9` |
| cuRAND | Existing cu12 wheel 10.3.3.141; API 10303 | External `nvidia/curand/lib/libcurand.so.10` |
| cuFFT | Existing cu12 wheel 11.0.8.103; API 11008 | External `nvidia/cufft/lib/libcufft.so.11` |
| cuSPARSE | Existing cu12 wheel 12.1.2.141; API 12.1.2 | External `nvidia/cusparse/lib/libcusparse.so.12` |
| NVRTC / nvJitLink | Existing cu12 wheels 12.2.140; NVRTC API 12.2 | External `nvidia/cuda_nvrtc` / `nvidia/nvjitlink` libraries |

Requested nvcc 13.0.88, observed nvcc `[13, 0, 88]`, runtime API 13000, provider
fingerprint `pipcu130_9bded6158c61`. `/proc/self/maps` confirmed loaded paths;
private links expose the external libraries in the toolkit include/library
directories. Loading those libraries is not a numerical validation of FFT,
sparse, random, NVRTC or nvJitLink operations.

The initial mixed cu12 cuBLAS + cuDNN configuration passed matrix multiplication
but aborted on convolution. `LD_DEBUG=libs` showed cuDNN trying to load missing
`libcublasLt.so.13`. Installing the real, hash-verified cuBLAS 13.0.0.19 wheel
resolved this for the measured convolution. No fake `.so.13` aliases to CUDA 12
binaries were used. Other mixtures remain unverified.

## Execution evidence

- Real CUDA 13 core compilation and elementwise output `[3, 3, 3]` passed.
- Three new CUB tests passed with zero skips: 6007-element float32 scan,
  `(5, 1031)` float64 scan, and `(5, 6, 7)` non-last-axis scan. Each compares
  forward values and reverse-scan gradients to independent float64 NumPy,
  checks device residency and forbids/counts backend fallback. They require
  neither CuPy nor another Jittor scan as a reference.
- Independent real PyTorch `2.11.0+cu128`, binary `_C` extension, CUDA 12.8 and
  the same physical GPU produced matrix/conv outputs and both input gradients
  in a separate process with Jittor import/loader paths unset. Matrix shape
  `(17, 23) @ (23, 13)` had maximum absolute error zero for all three outputs.
  Convolution `(2, 3, 9, 11)` with `(4, 3, 3, 3)` weights and padding one had
  output error zero, input-gradient error `2.98e-8`, and weight-gradient error
  `8.94e-8`. All Jittor outputs/gradients were resident on CUDA, fallback delta
  zero, and TF32 disabled.
- Existing real pip-cu122 stack retained fingerprint `pipcu122_a89fa4fed08c`.
  CUB, cuBLAS gradient and cuDNN convolution-plan regression: **23 passed,
  zero skips**, 290.52 s including stale-cache compilation.
- uv 0.13.1 performed universal lock resolution, actual PEP 517 wheel build,
  dependency resolution/install of `jittor[cuda13]`, and execution via `uv run`.
  It installed all five exact 13.0 toolchain packages from verified official
  PyPI wheel artifacts; small remaining runtime dependencies came from PyPI.
  In the installed-wheel run outside the checkout, with `PYTHONPATH`,
  `JITTOR_SOURCE_ROOT` and explicit nvcc unset, Jittor found the wheel compiler,
  freshly compiled the complete core and performed CUB forward/backward on
  CUDA with device residency, fallback delta zero and runtime API 13000.
- Provider filesystem/negative regression: **22 passed**, including missing
  CCCL/libdevice, drift to NVVM 13.4, separate prefixes, missing CRT,
  unexecutable compiler and propagation of damaged metadata.
- Final operand-residency/dtype assertions in the three CUB regressions passed
  on both CUDA 13 (2.74 s) and pip-cu122 (2.04 s).
- CPU `core`, both native and Torch modes: **322 passed, 42 skipped, one
  existing xfail**. The two CPU cores and oneDNN were built in cold external
  runner caches; these timings are not warm-cache performance measurements.
- Related build/provider/preflight consumers: **25 passed, two skipped**.
- Repository layout and whitespace checks passed. Complete structure run:
  **1434 passed, four skipped, three failed**, 362.15 s. The actual failures
  are the frozen baseline's stale ACL `DropoutACL` gradient-manifest entry,
  the AMP/SDPA/this report awaiting their results-index entries, and generated
  `MANIFEST.in` needing those report entries. The parent integration owns these
  shared registrations and the already repaired ACL manifest; they were not
  overwritten from this older branch. There were no additional structure
  failures from the CUDA 13 provider.

For reproduction, install `jittor[cuda13]` with pip or uv into an external
environment and supply the math-library headers/libraries in that toolchain's
include/library paths. The extra intentionally installs the compiler/runtime/
CCCL toolchain only. Use separate CUDA 13/CUDA 12.2 test caches, select an actual
device, and run:

```bash
python -m pytest -q tests/backends/cuda/test_cub_cumsum.py::TestCubToolchain
python -m pytest -q tests/backends/cuda/test_cublas_matmul_grad.py \
    tests/backends/cuda/test_cudnn_conv_plan.py
JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q \
    tests/structure/build/test_cuda_wheel.py
```

CUDA 13.4, Windows, ARM64, other GPU architectures/drivers, full CUDA 13 vendor
libraries, NCCL, and all CUDA kernel families are not certified by these runs.
The uv run uses installed official wheel artifacts and a separate installed
Jittor wheel, not merely a changed lockfile or a source-only import.
