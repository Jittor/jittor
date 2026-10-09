# CUDA 13.4 nvcc with CUDA 12 runtime on a real GPU

- Status: Superseded by [the decoy acceptance](2026-10-09-cuda12-pip-stack-decoy-acceptance.md);
  this run had no system CUDA visible and missed a libcudart leak
- Date: 2026-09-22 (UTC; raw log timestamps use 2026-09-23 Asia/Shanghai)
- Baseline commit: `632a7cfb9e003bb65494bb13bc8a765e10d2e4f2`
- Implementation commit: `c546a81e66867c653945c4464690a573bc56deb7`
- Owner: Jittor maintainers
- Review when: pinned CUDA wheels, compiler discovery, or CUDA JIT link flags change

## Summary

The tested Linux x86_64 `jittor[cuda12]` extra installs CUDA 12.2 runtime
components and a separate CUDA 13.4.92 nvcc front end with CCCL headers.
`nvidia-cuda-nvcc-cu12` supplies compiler support such as ptxas/NVVM, but not
the `nvcc` driver used by Jittor; the extra therefore pins `nvidia-cuda-nvcc`.
That compiler wheel also installs CUDA 13 CRT/runtime/NVVM support distributions.
Jittor deliberately excludes their runtime from its CUDA wheel stack: its core
links and loads CUDA 12 `libcudart.so.12`, while generated kernels use
`--cudart=none`. Process-map checks after the GPU cases showed CUDA 12 cudart
and cuDNN 9 from the Python environment, with no CUDA 13 cudart loaded.

CUDA 12 component headers precede the CUDA 13 nvcc root in include order. This
keeps CUDA 12 public API fields used by Jittor available while the CUDA 13 root
supplies compiler CRT and CCCL headers. An explicit `nvcc_path` remains
authoritative; otherwise the package-provided compiler is selected before
system/JTCUDA discovery.

## Real-GPU verification

The two smoke runs used an RTX 4090 (SM 8.9) with NVIDIA driver 610.57.04.
Both resolved CUDA runtime 12.2.140, cuDNN 9.26.0.51, NPP 12.2.1.4,
nvcc 13.4.92, and CCCL 13.3.4.3.1; Jittor reported the system g++ 14.2.0.

| Environment | Python | Selected compiler |
| --- | --- | --- |
| uv | 3.13.12 | environment `site-packages/nvidia/cu13/bin/nvcc` |
| conda | 3.11.16 | environment `site-packages/nvidia/cu13/bin/nvcc` |

Each environment ran a cold-cache GPU smoke serially with its own `JITTOR_HOME`:

- Elementwise add/reduction returned `56.0` for `(arange(8) * 2).sum()`.
- Matrix multiplication forward and backward completed with finite outputs and gradients.
- cuDNN convolution passed and returned shape `(1, 4, 16, 16)`.
- Runtime inspection found only the environment's CUDA 12 `libcudart.so.12`
  and cuDNN 9 split libraries mapped.
- No `CUDA_HOME`, `nvcc_path`, or `LD_LIBRARY_PATH` was set. No host `nvcc`
  was available on the tested `PATH`.

## Commands and repository checks

The dependency setup was `uv sync --locked --no-default-groups --extra cuda12`
for uv and `uv pip install --python <conda-python> -e '.[cuda12]'` for conda.

- `uv lock --check`: passed.
- `tests/structure/build/test_cuda_wheel.py`: 16 passed.
- `bash tools/check_repo_layout.sh`: passed.
- `JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q tests/structure`:
  1375 passed, 4 skipped (324.90 s).
- No nox session and no core/smoke/full maintained test-suite tier was run.

