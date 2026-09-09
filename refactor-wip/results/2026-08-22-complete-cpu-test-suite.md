# 完整 CPU native/Torch 双会话验证

- Status: verified for the complete CPU repository gate
- Date: 2026-08-22
- Baseline: `137f9dd1`
- Owner: test infrastructure and Jittor core maintainers
- Review when: process-mode ownership, CPU defaults, Torch shim activation,
  compiler scheduling, or the repository test inventory changes

## Scope

The repository cannot be validated by one `pytest tests` process because
Jittor's Torch compatibility mode changes process-global execution, reduction,
dtype, and gradient semantics. `tools/run_test_suite.py` now owns two isolated
CPU sessions: native Jittor and the deployed Jittor-backed Torch mode. Each has
its own `JITTOR_HOME` and `TMPDIR`; the runner clears inherited CUDA, device,
independent-Torch, and mode variables, disables the unresolved parallel
operator compiler path, and executes a real CPU warmup before pytest.

This report covers the full maintained CPU inventory. The Torch session is the
Jittor compatibility surface, not an independent binary PyTorch oracle.

## Findings and changes

The end-to-end runs found failures that focused modules did not expose:

- native test collection could import Torch and activate a process-global shim;
- Torch collection could overwrite `torch.utils` through a later native import;
- an active unrelated optimizer pruned standalone Torch input leaves;
- generated parallel-pass thread ranges used per-dimension bit counts where
  cumulative boundaries were required, causing wrong results or hangs;
- random dtype, unsigned inference, native complex scalar arithmetic,
  `Parameter` identity, optimizer state loading, MPI buffers, CHW transform
  chaining, and several test lifecycle assumptions diverged from their stated
  contracts;
- the complete notebook smoke was unstable with eight compile workers even
  after the independent SIGCHLD fix.

The runner and collection contracts now prevent cross-mode activation.
Generated parallel-pass source is corrected before compilation and locked by
the existing high-thread regression. The notebook smoke remains a real
nbclient execution but uses serial compilation. Tests that measure performance
are opt-in, while correctness assertions remain in the default suite.

## Verification

The complete runs used Python 3.11.15, GCC 12.3, and Jittor 1.3.11.0. Runtime
state was isolated below `$JITTOR_LAB_ROOT/_state/test-suite/`; the CPU runner
asserted that CUDA and ACL were unavailable inside each session.

| Gate | Result |
| --- | --- |
| Complete native CPU session | `721 passed, 891 skipped, 2 xfailed` in 28m47s |
| Complete Torch-mode CPU session | `1502 passed, 236 skipped` in 1h11m06s |
| Full-collection merge lifecycle regression | `1 passed, 1613 deselected` |
| Transform module after CHW/HWC fix | `41 passed` |
| Repository layout | passed |
| Isolated `tests/structure` | `216 passed, 2 skipped` in 68.90s |
| Diff whitespace check | passed |

The direct structure command was first invoked against a shared default cache.
After a normal `jit_utils updated` refresh, one fresh-process import timed out
while another long-running compiler suite on the host used that cache. Repeating
the same command with the maintained isolated Torch-session cache produced the
result above. This is environment evidence, not a framework failure.

## Maintained commands

```bash
python tools/run_test_suite.py --session native -- -x
python tools/run_test_suite.py --session torch -- -x
bash agent/scripts/check_repo_layout.sh
python -m pytest -q tests/structure
```

Configure isolated state as described in `agent/manuals/environment.md` before
running the structure command concurrently with any other Jittor process.

## Limits

This does not close the todo item requiring every test on every backend. CUDA,
NPU, and ROCm tests skipped by the CPU runner still require execution on their
declared real devices. Optional external packages, independent binary PyTorch
oracle runs, network tests, and performance benchmarks are also separate gates.
No accelerator correctness or performance conclusion is inferred from these
CPU results.
