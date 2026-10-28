# Platform Support

This page answers one question: **where has a given operating system, Python version, or
accelerator backend actually been verified?** Every row gives the location of the evidence --
a CI workflow, a report in `docs/results/`, or "unverified".

**Unverified does not mean unusable**; it means no one can guarantee it for you. For a
combination without a gate, run `python -m jittor.selftest` and the relevant tests yourself
before treating it as usable.

Installation steps are not here; see the
[README installation section](https://github.com/Jittor/jittor#install).

## Operating systems

| Platform | Verification location |
| --- | --- |
| Linux x86_64 | Gate. `.github/workflows/cpu.yml` runs `nox -s smoke` and `nox -s full` in an `ubuntu:24.04` container; `structure.yml` runs structure, lint, typing, packaging, and multiple Python versions |
| Linux aarch64 | Verified only as part of the Ascend gate (self-hosted ARM64 machines in `.github/workflows/npu.yml`). **There is no independent aarch64 CPU job** |
| macOS (Intel) | Verified at release time. `release.yml`'s `platform-validation` installs the canonical wheel on `macos-15-intel` and runs `python -m jittor.selftest`; **it does not run the test tree** |
| Windows x86_64 | Verified at release time, as above, with the `windows-2022` runner |
| macOS (Apple Silicon) | Unverified. No runner and no report |

The platform matrix is defined in exactly one place, `.github/ci-baseline.env`
(`CI_RELEASE_PLATFORM_MATRIX`), and workflows consume it rather than rewriting it. Jittor ships
a `py3-none-any` pure-Python wheel plus JIT source resources, so these three platforms install
**the same artifact**; the release process also specifically asserts that no platform-specific
wheel is produced.

## Python versions

`pyproject.toml` declares `requires-python = ">=3.7"`, and the classifiers cover 3.7--3.13.

| Version | Verification location |
| --- | --- |
| 3.11 | Gate. All maintained tooling and test sessions are pinned to 3.11 (`CI_PYTHON_VERSION`), and the CPU/CUDA/NPU/structure/docs gates all run on it |
| 3.7 | Gate. `nox -s py37` compiles the whole tree with a real 3.7 interpreter (a lower-bound check, not the full tests) |
| 3.12 / 3.13 | Gate. `nox -s py312`, `nox -s py313` build and run the installed wheel's self-test; the 3.13 entry runs against NumPy 2.x |
| 3.8 / 3.9 / 3.10 | Declared supported, **with no dedicated CI job**. 3.9 has one real-use record: the machine in the Ascend verification report ran Python 3.9.25 ([Ascend 910B3 validation](../results/2026-08-28-ascend-910b-validation.en.md)) |

Running `transformers` requires Python 3.10 or newer (it uses `types.UnionType`); that is an
upstream requirement, not Jittor's, see [Torch compatibility](../compatibility/torch.en.md).

## Backends

| Backend | Version range | Verification location |
| --- | --- | --- |
| CPU | g++ ≥ 5.4 (Linux), clang ≥ 8 + `libomp` (macOS), OpenMP required | Gate, `cpu.yml` |
| CPU oneDNN relay | float32 only | With the CPU gate; the dtype range and per-call overhead are open item `KI-BACKEND-011` |
| CUDA | building needs `nvcc`; CUDA < 11 additionally fetches cub, CUDA 11 prefers cuDNN 8. The `jittor[cuda12]` extra pins the CUDA 12.2.2 runtime and cuDNN ≥ 8.9.7 < 10 (Linux x86_64) | Gate, `cuda.yml`: self-hosted RTX 4090 (sm_89) + CUDA 12.2 (`CI_CUDA_VERSION`). CUDA 11 and 13 wheel families are **not provided** |
| Ascend NPU (ACL/HCCL), single card | CANN 9 (the gate asserts `atc --version` is 9.x), device must report Ascend 910B | Gate, `npu.yml` (Linux ARM64, `/dev/davinci0`, fail-closed device probe). A complete cold-cache verification is recorded in [Ascend 910B3 validation](../results/2026-08-28-ascend-910b-validation.en.md): CANN 9.0.0, driver 25.5.1, `397 passed, 9 skipped` |
| Ascend NPU (ACL), Ascend 950PR / x86-64 | CANN 9.1 | **Partially verified, no gate**. `e5886b6f6` (2026-09-10) ran matmul, convolution forward/backward, random distributions, CNN and transformer block training with `backend_fallback=error` and zero fallback on real hardware; on the same commit `tests/backends/acl/test_acl.py` was 32 passed / 11 failed. One synchronization-switch measurement in `docs/guides/ascend-910b.md` also comes from this machine |
| Ascend NPU, multi-card HCCL | — | **Unverified**: needs ≥ 2 910B3 cards, see `8.02` in [`agent/manuals/deferred-hardware.md`](https://github.com/Jittor/jittor/blob/master/agent/manuals/deferred-hardware.md) |
| ROCm | — | **Unverified**. `nox -s rocm` exists but there is no CI workflow and no device; the HIP provider's `runtime/driver.cc` has never been read by a compiler (`4.12` in `deferred-hardware.md`). MIOpen and rccl are **intentionally unimplemented** and raise `NotImplementedError` |
| Iluvatar Corex | — | **Unverified**. Only a read-only discovery contract exists (`tests/backends/corex/test_corex_discovery.py`, offline fake compiler); no nox session and no device. For hardware steps see [Iluvatar Corex](corex.en.md) and `8.14` in `deferred-hardware.md` |
| MPI / NCCL single-machine multi-GPU | — | Distributed cases in `nox -s mpi` and `nox -s cuda`, **with no dedicated CI workflow**; the machine behind the CUDA gate has a single card |
| Cross-machine (≥2 nodes) | — | **Unverified**: rendezvous, launcher, checkpoint, and disconnect gates are all waiting for two machines (`8.15`--`8.18`, `10.22` in `deferred-hardware.md`) |

Backend `grad()` coverage is also incomplete: of 60 backend gradients, 36 need hardware to
test, 7 have no tests yet, and 1 is unimplemented (open item `KI-BACKEND-013`).

## PyTorch versions targeted by the Torch compatibility layer

| Item | Value | Verification location |
| --- | --- | --- |
| Declared Torch API level | `2.11.0` (`torch.__torch_version__`, `torch.version.__version__`) | `compat/torch/installers/cuda/bindings.py`; `torch.__version__` still reports Jittor's own version |
| Reference torch for nightly ecosystem cross-checks | CPU build `2.7.1`, with `transformers 4.46.3`, `diffusers 0.31.0`, `peft 0.13.2` | Gate, `.github/workflows/ecosystem.yml` (`nox -s ecosystem`, seven cases run twice, comparing outputs and every gradient) |
| Reference torch for the real-model gap table | `2.11.0+cu128` (cuDNN 9.19), Transformers 4.56.2, Diffusers 0.35.1 | [Torch compatibility layer vs PyTorch on real models](../results/2026-09-24-torch-compat-real-models.en.md) |
| Reference torch for recent SDPA / end-to-end measurements | `2.13.0+cu129` | [Downstream library performance ledger](../performance/library-standing.en.md) |
| Library versions supported by adapters | Transformers 4.56.2 / 5.5.3; TorchMetrics 1.7.4 | `adapters/pyproject.toml` and [`adapters/README.md`](https://github.com/Jittor/jittor/blob/master/adapters/README.md); unrecognized versions raise `UnsupportedAdapterVersion` |

**Compiled PyTorch extensions are not supported at all**: packages such as `mmcv.ops` and
`torch_npu` that ship `.so` files compiled against the PyTorch C++ ABI cannot be installed by
a Python-level compatibility layer. This is an architectural boundary, not a to-do item; the
reasoning is in [Torch compatibility](../compatibility/torch.en.md).

## Container images

| Image | Base image | Verification location |
| --- | --- | --- |
| `jittor` | `ubuntu:24.04` | Gate, `containers.yml`: builds and smoke-starts the default notebook service |
| `jittor-cuda` | `nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04` | Same as above |

Both images share the single `Dockerfile` at the repository root; the matrix is defined in
`CI_CONTAINER_MATRIX` in `.github/ci-baseline.env`.

User-visible limitations and their identifiers are in
[Known limitations](known-limitations.en.md).
