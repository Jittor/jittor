# Downstream P1: allocator coexistence limit, numeric protocol, and transpose's runtime permutation

- Status: Fixed and verified (CPU; transpose/protocol additionally verified on real CUDA RTX 4090)
- Date: 2026-10-09
- Baseline: `6a850ea5`, then merged with upstream `2.0-refactor`
- Owner: Jittor core maintainers
- Review when: the id/key design of `mem/allocator/sfrl_allocator.*`, `Var`'s numeric protocol bindings, the JIT key and templates of transpose/fuse_transpose, or `CpuParallelPass`'s work estimate change

## Conclusion

The downstream quantum circuit library listed three P1 items for Jittor core, all handled:

| Problem | Root cause | Handling |
| --- | --- | --- |
| Allocator `block id limit extended`, after which every operation in the same process fails | Not an id leak: ids were being reclaimed all along. The lazily executed graph of a 30-bit tensor-network sampling held about 8 million small buffers live at peak (about 250k at batch=32, 920k at 128, proportional to batch), exceeding the hard limit `1<<21`; that limit came from packing `origin_size * ID_LIMIT + id` into one `size_t` pool key | The pool key became a `(size, origin_size, id)` triple, `ID_LIMIT` raised to `1<<40`, only guarding against counter corruption |
| `complex(v)`, `np.asarray([complex 0-dim Var, ...])` fail | `Var` has no `__complex__`, falling back to `__float__` | Added `Var.__complex__` |
| `Var @ external object` throws an internal `AttributeError` | `__matmul__` is a Python-level binding that hands any object to `matmul` | Match torch: `Var @ ndarray` converts to Var; other non-Var returns `NotImplemented` (the other side's `__rmatmul__` runs, otherwise `TypeError`); `ndarray @ Var` is `TypeError`; `@=` handles likewise |
| transpose's permutation is written into the JIT key, compiling on the spot for each new permutation | The `transpose`/`fuse_transpose` key (including the CUDA backend implementation) had per-axis `AXES` | The key keeps only dtype, dimension count and form; the permutation is converted on the host into per-axis strides passed as runtime parameters (strides were already runtime values) |
| After the previous fix, a 6-bit DMCircuit was still 300–390 ms after warmup | Unrelated to compilation: one [4,4,64] complex multiply-add (about 1K times) spawned thread groups on 128 cores. `CpuParallelPass` multiplied all inner loop upper bounds in the nesting, so the two sibling reduction loops (zeroing rows, accumulating rows) were counted as a product, crossing the 65536 threshold | Work estimate changed to multiply along the path and add sibling loops |

## Verification

Environment: 128-core CPU host (load about 23), RTX 4090, conda `jittor` and downstream `tensorcircuit` environments (using this worktree as editable), independent cache per line. Logs are in `$JITTOR_LAB_ROOT/_state/tc-p1/` (unversioned).

| Check | Result |
| --- | --- |
| 30-bit GHZ, `allow_state=False`, 1024 tensor-network samples (same parameters as the downstream official case) | completed, 924 s, peak RSS 26.6 GB, `0…0`/`1…1` = 507/517; before the fix it failed at `tot_block_id < ID_LIMIT - 1` |
| C++ `sfrl_holds_more_than_two_million_live_allocations` (2^21+1000 coexisting allocations, fake allocator for addresses, no memory used) | passed |
| Downstream `jitcount.py`: new kernels per iteration / elapsed (default thread count) | dense 6: +0 / 3–4 ms; DM 4: +0 / 7–8 ms; DM 6: +0 / 11–12 ms (before the fix +1 kernel and 0.7–4.6 s each; PyTorch DM 6 on the same machine 6–7 ms) |
| Random 12-dim permutation (float32/complex64; pure permutation, both fuse forms, gradients) | +0 compiles after warmup, about 1 ms |
| `complex(v)`, `np.asarray`/`np.real` (complex64/128), `Var @ ndarray`/`ndarray @ Var`/custom object/`@=` | consistent with the torch contract |
| New `tests/ops/test_transpose_runtime_permutation.py`, extended `tests/ops/test_binary_reflected_and_complex_order.py` | CPU and CUDA passed |
| Targeted regression (cpu_parallel_pass, transpose, reduce, high-rank tensors, JIT C++ tests) | passed; see also the pre-existing failures below |

## Known boundaries and follow-ups

- **Peak memory**: the 26.6 GB in the sampling case comes from the lazily executed graph accumulating all samples' intermediates before a single `numpy()`; this change only removed the incorrect limit and did not change the execution order.
- **Failures unrelated to this change and not separately re-verified on the original baseline** (they appear once oneDNN is loaded): `tests/codegen/test_jit_tests.py::test_fused_op_relay_matmul` (the relay reads only `#include <cmath>` from `mkl_matmul_op.cc`, and `jit_run` is inside `#ifdef JIT_cpu`), `tests/ops/test_matmul.py`'s `test_matmul`/`test_matmul_type` (the same relay misses), `tests/ops/test_misc_issue.py`'s `test_mkl_conflict1/2` (`jt.mkl_ops` is None in the subprocess). These paths are outside this change's scope.
- `TempAllocator` still has the same `1<<21` key design; it only holds a few workspaces and was not changed.
- ROCm/NPU were not verified.
