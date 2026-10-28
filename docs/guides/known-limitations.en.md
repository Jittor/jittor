# Known Limitations

This page summarizes **user-visible open items**: things that affect the numbers you get,
the speed, or whether an API is available at all. Each entry has an identifier pointing to
the [issue ledger](https://github.com/Jittor/jittor/blob/master/agent/manuals/known-issues.md)
-- **the ledger is the single source of truth**, with reproduction commands, measurement
conditions, disproven hypotheses, and current progress; this page gives only one sentence
and its impact.

Only **open** items are listed. Fixed entries are removed from the ledger, and the fix
evidence stays in result reports and Git history. Entries that are purely test
infrastructure, gate runtime, or static-check coverage are not listed here.

## Numerics and semantics

| ID | Symptom | Impact |
| --- | --- | --- |
| `KI-DTYPE-002` | implicit construction in `jt.array` narrows 64-bit NumPy values to 32 bits | current default behavior, with an explicit escape hatch; specify the dtype explicitly when mixing float64 data |
| `KI-SEMANTICS-003` | backend verification for floating-point comparisons is incomplete | CPU/CUDA and NPU float32 verified; other NPU dtypes and ROCm unverified |
| `KI-OPS-002` | backend verification for integer floor-division is incomplete | CPU/CUDA/NPU verified, ROCm pending |
| `KI-COMPLEX-001` | native complex capabilities have gaps | some linear algebra and FFT go through a real/imaginary bridge, `complex128` unsupported; see [complex dtype](../notes/complex-dtype.en.md) |
| `KI-EXEC-015` | an external graph created between warmed optimizer replays and awaiting backward may use updated parameter values | reproduced on CPU; complete backward for that graph before replay writes, or use eager |

## Performance and memory

| ID | Symptom | Impact |
| --- | --- | --- |
| `KI-CODEGEN-001` | broadcasting against real inputs is a strided view, and fused elementwise kernels pay a division and modulo per element for it | memory-bound broadcast addition is about 4.22x the dense form |
| `KI-CODEGEN-003` | CUDA code-generated reductions lag PyTorch on UNet shapes | diffusers UNet2D training step |
| `KI-CODEGEN-004` | fused elementwise kernels in one UNet2D step are about 10% slower | same as above |
| `KI-TUNER-001` | with hand-written meta-operator product forms, matmul/conv relay does not trigger and falls back to a generic kernel | the common user path is fixed; hand-written forms are still much slower |
| `KI-OPS-006` | NaN-correct CPU max/min reductions run at only half speed | results correct, throughput halved (already improved by half) |
| `KI-EXEC-004` | pipelined execution costs 43% more peak memory | measured and accepted trade-off; see [pipelined execution](../notes/pipelined-execution.en.md) |
| `KI-EXEC-010` | backward graph construction does not participate in pipeline submission | the device is idle while the backward graph is built |
| `KI-COMPAT-009` | dividing a float32 tensor by a Python float computes in float64 | results match PyTorch, at the cost of one extra wide-type operation |
| `KI-DIST-001` | FSDP2 flat sharding peaks higher than the unsharded model | numerically correct; no memory savings, see `compat/tests/fsdp2/test_fsdp_memory.py` |
| `KI-BACKEND-011` | the oneDNN CPU path only handles float32, and per-call overhead is unmeasured | non-float32 CPU convolution/matmul goes through the generic kernel |

## Torch compatibility gaps

| ID | Symptom | Impact |
| --- | --- | --- |
| `KI-COMPAT-001` | the Torch namespace mixes native-only helpers with imported symbols | `dir(torch)` has more entries than real torch; registered in the Torch API inventory |
| `KI-COMPAT-002` | `torch.ops.aten` has no native operator surface | `import vllm_omni` does not get past the import |
| `KI-COMPAT-003` | nested tensors are not `Tensor` | `isinstance(torch.nested.as_nested_tensor(...), torch.Tensor)` is false |
| `KI-COMPAT-004` | factory function results are not backward leaves | `torch.ones(3, requires_grad=True).is_leaf` is false; gradients themselves are correct |
| — | no Lightning compatibility layer | `pytorch_lightning` / `lightning` are unsupported; the in-house `jittor.lightning` was removed in 2026-07 |
| — | compiled PyTorch extensions cannot be installed | packages compiled against the PyTorch C++ ABI such as `mmcv.ops` and `torch_npu`; an architectural boundary, see [Torch compatibility](../compatibility/torch.en.md) |

## Backends and hardware

| ID | Symptom | Impact |
| --- | --- | --- |
| `KI-BACKEND-001` | narrow-integer sum/max/min lack atomics on NPU | those combinations are skipped on NPU |
| `KI-BACKEND-002` | the composite `atan2` may crash on NPU | skipped on NPU |
| `KI-BACKEND-003` | complex `irfft` may hang on NPU | skipped on NPU |
| `KI-BACKEND-006` | `cublas_test` / `cudnn_test` are dispatched to the host, which has no kernel | the two library self-check operators cannot run; the libraries themselves work |
| `KI-BACKEND-012` | the ACL descriptor cache is not wired into any runner, and the launcher and attribute migration have not run on Ascend devices | these ACL paths have only host-side evidence |
| `KI-BACKEND-013` | 7 backend gradients have no gradient tests, 1 is unimplemented | HCCL / ROCm / ACL |
| `KI-BACKEND-016` | `sd15_unet_train` occasionally crashes at 507035 on ACL (nondeterministic) | this training workload sometimes passes and sometimes fails on Ascend; `jt.flags.auto_graph_replay=0` works around it |
| `KI-BACKEND-017` | device-bound single-core-dense workloads are slower than torch_npu on Ascend | training/launch-bound graphs (`qwen3_train` about 11-13% faster) match or beat it, while `qwen3_prefill`/`vit` are slower |
| `KI-COMPILER-001` | the parallel compiler may corrupt process state | the file-level compile-pool deadlock was fixed on 2026-09-16, operator-level state remains open; see [parallel compiler segfault](../development/known-issues/parallel-compiler-segfault.en.md) |
| `KI-COMPILER-007` | CPU-only import without CUDA reports heap corruption at exit | currently reproduced only inside the gate |
| `KI-EXEC-003` | cuDNN autotuning is not isolated from the execution scheduler | training numbers silently change for the same model and input |
| `KI-EXEC-005` | a var released by another thread mid-batch makes that batch fail | rare under multiple threads and currently unverifiable |
| `KI-EXEC-008` | the CUDA convolution test file occasionally aborts when the live forward count underflows | the whole process aborts |

Backends awaiting hardware acceptance (ROCm, Iluvatar Corex, multi-card HCCL, cross-machine)
are listed in [Platform support](platform-support.en.md) and
[`agent/manuals/deferred-hardware.md`](https://github.com/Jittor/jittor/blob/master/agent/manuals/deferred-hardware.md).
