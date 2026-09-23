# vLLM CUDA adaptation acceptance

- Status: focused regressions and single-GPU eager Qwen cases verified; advanced modes have explicit remaining blockers.
- Date: 2026-09-23.
- Baseline: `61294cd14673ba60f4072542a2e73ea2c8f8c509` (`chk`), with the existing uncommitted vLLM compatibility changes and this fix.
- Owner: vLLM adaptation task.
- Review when: vLLM buffer pool contracts, Torch tensor transfers, adapter module patches, or compatibility transaction APIs change.

> Follow-up: [single-card correctness](2026-09-23-vllm-singlecard-correctness.md)
> supersedes the generator/OPT and sampling-diagnosis status below. Performance
> numbers here belong to this earlier source version and have not been remeasured
> after those changes.

## Failure and repair

vLLM 0.24.0 maintains CPU/NumPy sampling state and publishes it through
`UvaBufferPool.copy_to_uva`. The existing adapter returned a one-time CUDA copy
from `get_cuda_view_from_cpu_tensor`; later CPU writes never refreshed that
copy. A complete Qwen warmup captured `k=[0]`, `k.long()=[0]`, and the invalid
gather index `151936 - 0 = 151936` for a vocabulary of size 151936.

Before changing production code, five tests of pool updates and SamplingStates
failed: the host held the requested k but CUDA still held zero. The repair is
an adapter patch on `UvaBufferPool.copy_to_uva`, making each submission an
explicit CUDA snapshot. Tensor, NumPy, and list sources are supported. Transfers
finish before host metadata can change again; independent allocations preserve
outstanding consumers across pool rotation. This is an explicit-copy fallback,
not zero-copy UVA. No gather-index clamping was introduced.

## Validation

RTX 4090, Python 3.10.21, vLLM 0.24.0, transformers 5.5.3. The independent
oracle uses PyTorch 2.11.0+cu130; the shim uses Jittor with CUDA toolkit 12.2.

- Initial UVA/real SamplingStates regression: **5 failed** before repair.
- Final UVA/top-k tests, including pool rotation, list/NumPy/Tensor snapshots,
  k=1, k=vocab and disabled top-k: **11 passed** on shim and **11 passed** on oracle.
- Those tests plus the real vLLM last-token logprob kernel and adapter structure:
  **24 passed, 0 skipped** on shim.
- Lifecycle/arming/ownership contracts: **11 passed**.
- Full Qwen warmup after repair captured k=50 and legal gather index 151886.
- Qwen3-0.6B completed both an instrumented run and a clean run without the
  diagnostic wrapper. Clean process exit code: 0.

Qwen configuration: FP16, TP=1, eager, max_model_len=512, max_num_seqs=1,
prefix cache disabled, FLASH_ATTN, temperature=0, logprobs=5, seed=0.

| Prompt | Generated tokens | Token IDs equal to oracle | Maximum generated-token logprob absolute difference |
| --- | ---: | --- | ---: |
| The capital of France is | 4 | Yes | 0.006989 |
| The capital of France is | 16 | Yes | 0.006989 |
| 1 + 1 = | 4 | Yes | 0.007987 |
| 1 + 1 = | 16 | Yes | 0.007987 |

All 40 generated token IDs match. Across common top-5 candidates, the maximum
logprob absolute difference is 0.029878; each 16-token case has one step with a
different top-5 candidate set. This is restricted greedy generation agreement,
not exact logprob parity, stochastic sampling validation, multi-device serving
validation, or a performance claim. This was the initial, restricted acceptance. The broader follow-up below supersedes
its sampling status; the initial workaround was not a valid stochastic sampler.

## Reproduction and artifacts

After activating the existing isolated shim environment, from the checkout:

```bash
PYTHONPATH=adapters python -m pytest -q \
  adapters/tests/vllm/test_uva_updates.py \
  adapters/tests/vllm/test_logprob_boundary.py \
  adapters/tests/vllm/test_structure.py
```

Unversioned raw logs on the GPU test host are under
`$JITTOR_LAB_ROOT/_state/vllm-gpu/20260921/`: `topk-capture-red.log`,
`uva-updates-red.log`, `uva-updates-green.log`, `uva-final-tests.log`,
`uva-lifecycle.log`, `qwen-uva-fixed.log`, `qwen-uva-clean.log`, and
`qwen-uva-clean.json`. The local comparison artifact is
`$JITTOR_LAB_ROOT/_state/vllm-gpu/20260923/comparison.json`.


## Expanded acceptance: random sampling and mode matrix

The same baseline and GPU environment were used with the new
`agent/skills/vllm-torch-compat/acceptance.py` runner. Raw artifacts are unversioned
under `$JITTOR_LAB_ROOT/_state/vllm-acceptance/20260923/`. Each JSON records model,
options, prompt/output lengths, token IDs, measured wall time, and exceptions.
Tests and timings refer to the installed vLLM 0.24.0 only.

### New bugs demonstrated before fixes

1. The temporary gumbel override treated `apply_temperature=False` as greedy.
   vLLM uses this flag after scaling logits even for nonzero temperature. The
   new stochastic tests produced **8 failed, 1 passed** (`random-red.log`). The
   override was removed entirely, restoring the upstream Triton sampler.
2. With the original sampler restored, real readback exposed CUDA illegal
   address (**9 failed**, `random-native-kernel.log`). The bridge translated
   Python `None` to a runtime `*i8` NULL pointer. Triton consequently considered
   `optional_ptr is not None` true and wrote to absent processed-logit outputs.
   A safe branch-only reproducer failed without poisoning the CUDA context.
   The bridge now specializes `None` at compile time and omits its runtime
   argument slot. The optional-pointer and stochastic tests then yielded
   **11 passed**, no skips (`random-green.log`). No argmax replacement remains.
3. Fourteen unavailable quantization/fusion calculation symbols silently
   returned `False`. Their direct CUDA-tensor calls failed the new explicit
   rejection tests (**14 failed**, real capability test passed). They now
   retain importable names but raise `NotImplementedError` on execution.
   Real capability probes remain false. Backend, lifecycle, ownership and
   structural regression: **49 passed**, no skips (`fp8-green.log`). This is
   explicit unsupported behavior, not an FP8 implementation.
4. OPT-125m's legacy runner warmup exposed missing `Tensor.exponential_`.
   Targeted tests first yielded **11 failed** (`exponential-red.log`). The
   default-generator implementation now draws on the tensor's native device
   and applies the inverse CDF with in-place/view writeback. **11 CPU and
   11 CUDA tests passed** on each of shim and oracle. Explicit generators
   remain a stated unsupported boundary; the shim rejects them rather than
   ignoring the seed or silently drawing on the host. OPT's next warmup
   reached that boundary, so OPT end-to-end acceptance still fails.

Fixed-logit gumbel parity: four temperatures (0, .8, 1, 1.5), two scaling flags,
two positions, and 64 seeds per combination give **1024/1024 identical sampled
IDs** between independent shim/oracle processes (`*-sampler-parity.json`).
These are finite test inputs, not a proof of distributional equivalence for
all logits or devices.

### Functional results

| Case | Jittor result | Oracle result and limits |
| --- | --- | --- |
| Qwen3-0.6B random, temperature=.8/top-k=40/top-p=.9 | Completed four 32-token generations; same seed repeats, three different seeds give three outputs | Two of three distinct seeds match all 32 tokens; seed 10 diverges. Exact end-to-end random parity is not claimed |
| Four simultaneous engine requests | Four single requests and two batches of four completed; batch equals single | All 384 compared output token IDs match; this is offline engine batching, not HTTP load testing |
| Long inputs | 1024/3072 input tokens, 128 new tokens each | All 256 output token IDs match. max_model_len=4096; larger contexts untested |
| TP=2 | NCCL rank 0/1 initialize successfully after configuring an installed NCCL library; engine then fails | Shim reports the requested CPU group as NCCL; vLLM rejects `in_the_same_node_as` on a NCCL group. Oracle TP=2 completes 32 tokens |
| FP8 online quantization | Fails importing `torch.library.wrap_triton` | Oracle FP8 completes 32 tokens. AWQ/GPTQ/other quantizers untested; import-only calculation symbols explicitly reject execution |
| CUDA Graph | Fails because `graph_pool_handle` is not callable | Oracle captures decode graphs and completes 32 tokens; shim capture/replay not accepted |
| Compilation mode 3, graphs disabled | Missing `torch.fx._lazy_graph_module` | Oracle completes compilation and 32-token generation. No claim based on a no-op `torch.compile` |
| OPT-125m, FP16/eager | Fails on explicit-generator `exponential_` in legacy sampler warmup after the default API repair | Oracle completes 32 tokens; no other-model support claim |

OPT weights came from `facebook/opt-125m`, revision
`27dcfa74d334bc871f3234de431e71c6eeba5dd6`.

Multi-GPU diagnostics preserved separate failures: long IPC socket path,
missing spawn main guard in the initial runner, NCCL source/toolchain conflict,
and final host-group backend mismatch. Only the last one describes the current
adaptation boundary. The harness main guard is fixed; short `VLLM_RPC_BASE_PATH`
and prebuilt NCCL paths resolve the environment failures. Two-rank NCCL was
precompiled serially before the final engine attempt.

Random-generation logprob follow-up (`*-random-logprobs.json`): seed 10 first
differs at zero-based generated-token index 6. Before/at that divergence, common
reported candidate logprobs differ by at most 0.017160. Seeds 11/12 retain exact
32-token agreement with maxima 0.031215/0.028276. Together with fixed-logit sampler
parity, this points to small model-numerical differences affecting a stochastic
choice, rather than the tested RNG contract being broken. This is an inference,
not a proof of end-to-end distributional equivalence or a universal tolerance.


### Repository checks and limits

- Final optional-pointer, stochastic sampler, and packed-mask boundary regression:
  **14 passed**, no skips (`final-sampling-regression.log`). The small-vocabulary
  case now actually uses vocab=34 against block=8192, and compares GPU values
  rather than only inspecting lazy tensor shapes.
- `tools/check_repo_layout.sh` and `git diff --check`: passed.
- `tests/structure`: **1368 passed, 4 failed, 4 skipped**. The initially detected
  top-level Torch import in the new packed-mask test was corrected. A focused
  rerun confirms that the four remaining failures refer only to unchanged
  baseline files: `tests/core/test_executor_python_threads.py` (interpreter
  launch policy) and `tests/integration/test_h3_decode_thread_race.py` (collection
  side effects). Both files match the baseline commit; they were not changed
  to make this task's checks green. See `structure-remaining.log`.
- CUDA Torch core tier was attempted in its own cache with a 420-second bound;
  it timed out during initial warmup/compilation without returning a pytest
  result. This is **not a core-tier pass** (`core-torch-cuda.log`, exit 124).
  A 240-second retry with the populated cache passed CUDA warmup and launched
  pytest, but again timed out without a test summary
  (`core-torch-cuda-retry.log`, exit 124). Its surviving pytest process group
  was explicitly terminated; no core pass/fail count can be inferred.
  The focused tests above do not establish that the entire repository is green.


### Exact integer index-fill repair during final review

The earlier compatibility repair preserved `index_fill_`'s dtype by casting the
result back after native execution. New value-level tests showed this was
insufficient: native `index_fill` blended through float32, losing untouched
int32/int64 low bits, and multiplication by zero did not overwrite NaN/Inf.
The new cases initially gave **6 failed, 3 passed** on CUDA; native PyTorch
passed all nine (`index-fill-values-red.log`, `index-fill-oracle-cuda.log`).

The repair is now in `python/jittor/ops/advanced_indexing.py`: a boolean selector
chooses between the original values and a fill constructed explicitly in the
input dtype/device. The compatibility after-cast wrapper was removed.
The final shim CPU/CUDA tests each give **9 passed**; direct native Jittor CPU
and CUDA checks each pass five value cases, including int64 values above 2**60.
The native checks specify dtype explicitly, respecting Jittor's native default
64-bit conversion policy. Artifacts: `index-fill-values-*-green*.log` and
`index-fill-native-*.log`.

An intermediate attempt using `full_like` exposed a separate large-int64 scalar
factory limitation (the fill value became 17). Explicit-dtype array construction
fixes the index-fill path; the general factory was not changed or declared
correct for that domain. The interrupted benchmark had collected **zero** Jittor
measurement samples and was archived as `jittor-pre-indexfill-warmup.*`.
Final performance numbers must come from the subsequent, uniform final version.

After the final index-fill repair, optional-pointer/packed-mask/stochastic/UVA/
logprob regression yielded **26 passed**, with no skips. Qwen random generation
and both long-context cases were rerun (`shim-random-final.json`,
`shim-long-final.json`): the seed reproducibility/diversity results and oracle
agreement stated above remain unchanged. The direct generic `full_like`
reproducer is recorded as KI-SCALAR-001; it remains outside the repaired
index-fill path.

### Final steady-state performance

Both environments used the same physical RTX 4090 sequentially, Qwen3-0.6B,
FP16, TP=1, eager, greedy sampling, prefix caching disabled, and 32 output tokens
per request. For each backend, three fresh engine processes each ran three
warmups and 21 measured calls at each batch size. Unit tests and benchmarks used
separate JIT caches. Results below are medians across the three process-level
statistics; the P95 column is a median of process P95s, not a pooled percentile.
Latency measures the entire synchronous generation call (all requests in a batch).

| Backend | Batch size | Median latency (s) | P95 latency (s) | Aggregate output tokens/s |
| --- | ---: | ---: | ---: | ---: |
| Jittor | 1 | 1.602 | 1.646 | 19.89 |
| PyTorch | 1 | 0.523 | 0.535 | 60.53 |
| Jittor | 4 | 1.761 | 1.779 | 72.35 |
| PyTorch | 4 | 0.545 | 0.550 | 234.94 |

All six final benchmark processes completed. Every measured generated token
sequence matched the independent oracle single-request reference: 10,080 output
tokens per backend across 315 request outputs. Jittor achieves approximately
32.9%/30.8% of oracle throughput for batches 1/4; median generation-call latency
is approximately 3.06x/3.23x the oracle. These are this workload's measured
results, not general speed claims. The machine was shared, although the two
backends used the same test GPU; host activity can affect timings.

This does not measure HTTP serving concurrency, time to first token, inter-token
latency, cold compilation, sustained production load, or maximum batch/context
capacity. Those remain unaccepted. Random sampling is functionally exercised
with seed reproducibility and fixed-logit parity, but complete model-level
stochastic numerical equivalence remains unproven. Multi-GPU, quantization,
graphs, compilation, and OPT remain blocked as detailed above.

Artifacts: `jittor-benchmark-{1,2,3}.json`,
`oracle-benchmark-{1,2,3}.json`, their logs, `comparison.json`,
`benchmark-host-state.txt`, and `code-manifest.json`. The manifest records the
baseline and SHA256 of all 27 changed/new Python source and test files; local
sources were checked against this final remote benchmark manifest.
