# Tensor rank limit 10 → 32, and four core gaps exposed by downstream evaluation

- Status: Fixed and verified (CPU + real CUDA RTX 4090), committed at `6a850ea5`
- Date: 2026-10-05
- Baseline: `316e1fe7`
- Owner: Jittor core maintainers
- Review when: the packing format of `type/nano_vector.h`, the dimension encoding of the JIT key, the reduce/broadcast mask width, the `for` loop output of `KernelIR::to_string`, the numeric protocol slots of `pyjt_compiler.py`, or the `numpy_code` forward of `linalg` change

## Conclusion

One real evaluation of a downstream quantum circuit library produced five problems attributed to Jittor core, all handled by priority:

| Priority | Problem | Handling |
| --- | --- | --- |
| P0 | Tensors max out at 10 dimensions (the 11th dimension directly reports `NanoVector exceeds its ten-entry ...`) | Raised the limit to 32 dimensions; exceeding it gives an explicit error |
| P0 | Complex `jt.norm` unusable (`maximum` has no complex kernel; a compile-time error) | `norm` changed to `sqrt(sum(abs(x)**2))` with a real result; complex `maximum`/`minimum` changed to an explicit construction-time error (matching torch) |
| P1 | `Var * obj` throws RuntimeError directly for an unrecognized object, so `obj.__rmul__` never gets a chance to run | The generated binary numeric protocol slots return `NotImplemented` when no overload matches |
| P1 | Gradients of rank-deficient QR: CPU throws `LinAlgError`, CUDA silently yields NaN | Root cause is CuPy's `solve`/`inv`/`cholesky` returning NaN for singular matrices; changed so both raise `LinAlgError` |
| P1 | Under the default `use_mkl`, on a machine without cmake the first CPU matmul fails while building oneDNN on the spot | When oneDNN is unavailable (no cmake, no existing build, no explicit path), the mkl library reports unavailable with a one-time warning, and CPU multiplication uses the generic kernel |

## P0: dimension limit

The limit is not only in `NanoVector`. After raising it, the following silent assumptions were exposed in turn; each has a regression case:

1. **`NanoVector` packing format**: two 64-bit words, a 4-bit entry count plus 10×6-bit end offsets, 64 bits of values in total. Now overflow goes to a reference-counted, copy-on-write heap block (entry-count field marked as 15), while the packing fast path is unchanged; shrinking back to a packable size repacks, so vectors built by appending still have exactly one representation per content. The Python-side held `NanoVector` gained `__dealloc__`.
2. **Old bug in the packing format itself**: an entry whose end bit is exactly 64 is stored as 0 in its 6-bit field and carries into the next field; a later push then overwrites data with `pre_offset=0` (measured: stride `1024` read back as `1092`). The old code never pushed after that point, so it never surfaced; now 0 is uniformly interpreted as 64.
3. **JIT key encoding**: each dimension's key name was written in hex via `JK::hex1`/`<< int`, while the template `NAME@i` expands in decimal, so from dimension 10 on they disagreed; `hex1` also took the value mod 16. Now subscripts in names use `JK::dec`, `=` values use the full `JK::hex`, and `:` values (`IV`/`IO`, code_op's `_dim`) use decimal; the fused op's `var_info` gained a separator.
4. **Mask width**: the `ReduceOp`/`BroadcastToOp` mask was `uint16`, so a 17-dimension full reduction returned shape `[2]` (**silent error**). Changed to `uint32`; `1<<i` changed to an unsigned shift, and `reduce_tuner`'s `(1<<j)<=mask` loop, which does not terminate at bit 31, was fixed as well. The 32-bit mask is the source of the 32-dimension limit, checked uniformly at `Var` construction and `set_shape`.
5. **Fixed-capacity arrays**: `StackVector<>` defaulted to 10 slots with no bounds check (out-of-bounds getitem caused a stack overflow / `double free`); changed to 32 slots with an assertion on overflow; the cuTT plan key went 10 → 32; reindex's `@e{i}(i0,...)` no longer builds characters with `'0'+i`.
6. **nvcc compile time**: cudafe++ is exponential in the nesting depth of "declare a variable in for-init". Independent reproduction: depth 14 = 2.1 s, 16 = 11 s, 17 = 30 s (about ×2.5 per level); a rank-20 reduction kernel would not finish compiling within 10 minutes; moving the declaration into an outer block `{ T i = 0; for (; ...) }` makes 20 levels compile in 0.75 s, and g++ has no such problem (20 levels in 0.08 s). `KernelIR::to_string` uses this form for loops that already have ≥8 outer nested loops, and source generated for shallow loops is unchanged byte for byte.

## Verification

Environment: conda `jittor`, g++ 11.5, CUDA 12.2, RTX 4090. Each line used an independent `JITTOR_HOME`/`cache_name`. Logs are in `$JITTOR_LAB_ROOT/_state/tc-gaps/` (unversioned).

| Check | CPU | CUDA |
| --- | --- | --- |
| Dimension probe: n = 10/11/12/16/17/20 × 21 operations (construction, elementwise, full/partial reduction, keepdims, transpose, reshape, getitem/setitem, Var indexing, broadcast, where, argmax, argsort, concat, matmul, grad, complex) + rank 32 | 127/127 | 127/127 |
| rank 33 | explicit error | explicit error |
| New `tests/core/test_high_rank_tensors.py`, `tests/ops/test_binary_reflected_and_complex_order.py`, `tests/linalg/test_singular_input_raises.py` | 16 passed | 16 passed |
| New `tests/build/test_onednn_availability.py` | 4 passed | — |
| C++ JIT tests (including the new `nano_vector` overflow/copy-on-write/64-bit-boundary cases) | 97 passed / 2 skipped / 4 xfailed | — |
| Targeted regression of affected ops and bindings (binary/reduce/broadcast/transpose/reindex/getitem/setitem/argsort/code/where, dlpack, pyjt protocol, error messages, all linalg) | 409 passed / 3 skipped | — |
| `tools/run_test_suite.py --tier core` (default configuration) | native 98 + torch 212 passed, 0 failed | native 128 + torch 212 passed; the only failure was a 600 s subprocess timeout under a cold cache, and it passed when rerun separately with a warm cache |
| `tests/structure` (Torch mode) | after correcting two count ledgers, only 1 pre-existing failure remains (see below) | — |

With a warm cache `test_high_rank_tensors.py` takes about 9 s and stays in the default tier.

## Known boundaries and follow-ups

- **Pre-existing failures, not introduced here**: `tests/structure/test_pytest_contract.py::test_test_modules_avoid_collection_time_backend_side_effects` points at `tests/linalg/test_linalg_higher_order.py` from 2026-09-30 (it calls a local helper at collection time); one item of `test_process_mode_contract` fails only in a full structure run and passes when run alone.
- Run with `use_mkl=0`, `--tier core` fails three memory/scheduling assertions in `test_core.py` — they assume the oneDNN matmul path and pass under the default configuration; unrelated to this change.
- `Var @ obj` still does not fall back to `obj.__rmatmul__`: `__matmul__` is a Python-level binding, and a safe fallback criterion (downstream objects often carry `.shape`) needs separate design. (Handled on 2026-10-09; see the [downstream P1 report](2026-10-09-downstream-p1-allocator-protocol-transpose.en.md).)
- QR rank-deficient gradients now raise consistently on both ends rather than giving finite values. torch's `linalg.qr` backward likewise does not give finite gradients for singular R; downstream needing finite gradients should supply its own regularized QR backward.
- The 32-dimension limit is determined by the 32-bit mask; raising it further requires widening the mask together with the Python-side parameters.
- ROCm/NPU were not verified (no hardware).
