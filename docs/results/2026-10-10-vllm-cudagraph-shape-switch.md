# vLLM CUDA Graph shape-switch repair

- Status: Verified for TP1 Qwen3-0.6B FP16, stable decode shapes 1 and 4.
- Date: 2026-10-10.
- Baseline: `d7436b0c863dc9f2ceba381fbad67129596bec0c` plus the already dirty CUDA Graph prototype and adapter worktree. The Torch Tensor fix in this report is a separate scoped change.
- Owner: Jittor Torch compatibility maintainers.
- Review when: Torch in-place assignment, native Var storage ownership, CUDA Graph capture, or vLLM attention metadata changes.

## Failure and cause

The same vLLM engine captured shapes 4 then 1. The first shape-4 decode could fail with `cudaErrorIllegalAddress`; capturing only shape 4 worked. The existing graph allocator deferred frees **during** capture, but vLLM changed persistent metadata **between** captures. Jittor's Torch `fill_`, `zero_`, `copy_`, basic `__setitem__`, and scan `out=` routes reassigned a native Var, which could replace and free the address embedded in an older CUDA Graph.

A standalone CUDA reproducer captured a view of a Torch tensor, called `fill_` on its root, and forced allocator reuse. The captured graph returned `202` from the unrelated new occupant; the updated root required `46`. The original input address was found in the new occupant. Keeping a Python reference to the view did not prevent the storage change.

## Change

- Dense, no-grad CUDA `fill_`, `zero_`, `copy_`, basic slice writes, and scan `out=` now copy values into the existing allocation, preserving graph input/output addresses.
- Trainable tensors, detached `.data` aliases, strided/advanced indexing, and shape-mismatched writes retain their existing graph-aware assignment behavior.
- Large `int64` scalar slice writes use exact host bytes because the current native `full` factory narrows some large values.

## Verification

- Real RTX 4090, vLLM 0.24.0, Transformers 5.5.3, Qwen3-0.6B FP16, tensor parallel size 1, greedy, 16 prompt tokens and 64 output tokens, `FULL_DECODE_ONLY` graphs for sizes 1 and 4.
- Alternating request sizes `1,4,1,4`: one warmup and one measured pass per size, 8 rounds and 20 sequences total. Both Jittor and independent PyTorch runs reported `status=completed`, all workers exited with code 0, and no tracked shared-memory segment remained. All 20 token sequences (1,280 output tokens) matched exactly.
- Reverse request order `4,1,4,1`: a second Jittor process also completed 8 rounds and 20 sequences with zero token differences against the same PyTorch batch-matched reference; worker exit code 0 and no tracked shared memory remained.
- `compat/tests/torch/test_cuda_graph_storage.py` exercises replay after `fill_`, `zero_`, `copy_`, partial basic writes, scalar updates, scan `out=`, and exact large `int64` writes on real CUDA.
- Targeted Torch compatibility tests: 10 passed, 7 accelerator skips. Repository layout and Python syntax checks passed. The full repository structure gate ran to completion (1,366 passed, 4 skipped, 7 failed); the remaining failures point to earlier worktree files outside this fix, including three other unindexed vLLM reports, and are not a green structure-gate result.
- The bug is fixed within this bounded decode configuration. Other models, TP>1, changing KV-cache allocation, and production soak remain unverified.

Diagnostic throughput in the 64-token protocol was 212–213 tok/s (Jittor batch 1) versus 399–409 (PyTorch), and 747–749 (Jittor batch 4) versus 1516 (PyTorch). This is one warmup and one timed pass per size, not a formal performance gate; the remaining speed gap is separate from the shape-switch correctness defect.

Unversioned JSON/log artifacts: `$JITTOR_LAB_ROOT/_state/vllm-tp/20261009/1010-{repro,final64,final-b4first,oracle64}.{json,log}`. The standalone input-rebinding reproducer is in the same external state directory.
