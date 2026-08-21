# MMCV/MMEngine CUDA typed tensor 导入兼容

- Status: Verified
- Date: 2026-08-21
- Baseline: `7cc35238`
- Owner: Torch compatibility maintainers
- Review when: the typed-tensor implementation, tensor residency reporting, or
  maintained MMCV/MMEngine versions change

## Problem and fix

MMEngine 0.10.7 builds `InstanceData` type annotations from
`torch.cuda.BoolTensor` and `torch.cuda.LongTensor`. The Torch shim did not
publish the legacy dtype-specific classes under `torch.cuda`, so importing the
real `mmengine.model` package raised `AttributeError`.

The CUDA installer now publishes all ten PyTorch CUDA typed tensor classes.
They are device-aware classes rather than aliases of the top-level typed tensor
classes: `isinstance` checks both dtype and actual accelerator residency, and a
constructor migrates its result to the accelerator. This prevents a host bool
or int64 tensor from being silently classified as a CUDA tensor.

The maintained boundary remains unchanged: pure-Python `mmcv.cnn` and
`mmengine.model` are supported; `mmcv.ops` is a binary extension built against
PyTorch's C++ ABI and is outside the Python shim contract.

## Verification

Environment: Python 3.11.15, Jittor 1.3.11.0, mmcv-lite 2.1.0, mmengine 0.10.7,
CUDA 12.2, NVIDIA driver 595.84, RTX 4090. CPU and CUDA used separate isolated
`HOME`, `JITTOR_HOME`, `TMPDIR`, XDG cache, and `cache_name` directories under
the unversioned lab state root. The first JIT build was serialized.

Reproduction before the fix: after removing the ten proposed `torch.cuda`
typed names from a fresh Torch-shim process, `import mmengine.model` failed at
`mmengine/structures/instance_data.py` when reading
`torch.cuda.BoolTensor`.

Results:

| Gate | Result |
| --- | --- |
| Focused CPU import/type regression | `2 passed, 1 skipped` |
| Focused real-device CUDA regression | `3 passed` |
| dtype-promotion plus repeated-install regression, CPU | `24 passed` |
| dtype-promotion plus repeated-install regression, CUDA | `24 passed` |
| Real PyTorch parity: MMCV ConvModule + MMEngine BaseModule, CPU/CUDA | `4 passed` |
| `tests/structure` | `210 passed, 2 skipped` |
| `bash agent/scripts/check_repo_layout.sh` | passed |
| `git diff --check` | passed |

The CUDA regression constructs `torch.cuda.FloatTensor`, asserts actual device
residency, executes a multiplication on that tensor, synchronizes by reading
the full result, and checks the numerical output. The downstream parity gate
loads identical weights into real PyTorch and Jittor and compares forward
outputs, input gradients, and all parameter gradients on both CPU and CUDA.

The new test file passes ruff. Running ruff on the whole existing `cuda.py`
still reports two pre-existing style findings at lines 85 and 176; this focused
compatibility change does not claim a whole-file lint cleanup.
