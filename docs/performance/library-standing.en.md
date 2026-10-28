# Downstream library performance ledger

- Status: maintained
- Last reviewed: 2026-09-09
- Baseline commit: `5932d7c9b`
- Owner: performance and compatibility maintainers
- Review trigger: after any library-level performance measurement, or when a row's evidence goes stale

[Benchmarking methodology](benchmarking.en.md) explains **how to measure**; this file records **where we stand now**.

The two are separate because they fail differently: the methodology is essentially stable, while the numbers change with every measurement. These numbers used to be scattered across 25 dated reports with different formats, baselines, dates, and hardware -- answering "how good is library X now" meant reading all 25.

## Reporting basis

**Ratio = Jittor time / reference time. Less than 1 means Jittor is faster.** This applies throughout, including the cited reports.

The reference differs by path:

| Path | Reference |
|---|---|
| Torch compatibility layer | Independently installed binary PyTorch (**not** Jittor's own shim) |
| Ascend | Same-machine, same-card `torch_npu` |
| vLLM Ascend | Native `vllm-ascend` |
| Native Jittor API | See "The category without a reference" below |

**A number without a date, commit, and hardware is not a number.** Every row in this table must carry all three; otherwise delete it rather than keep it -- a ratio of unknown provenance is worse than no ratio, because it will be cited.

## Two kinds of rows, do not mix them

**Gate rows** are re-measured by the maintained nightly, and staleness is discovered automatically. **One-off rows** come from a single investigation and are never re-measured, **so they silently go stale**.

Currently only a small part of the first kind is gated.

### Gate rows: the speed half of `nox -s ecosystem`

Seven cases, Torch compatibility layer against independent PyTorch, threshold `ECOSYSTEM_SPEED_RATIO = 1.07`. The ratio is **always reported**, and asserted only in the nightly -- a wall-clock upper bound is the only assertion that machine load can turn red unilaterally, and a false red on the speed gate costs more than a late true red: it teaches people to ignore a gate that is also checking numerics.

| Case | Source |
|---|---|
| `large_convnet` / `large_transformers_bert` / `large_transformers_gpt2` | `compat/tests/torch/test_ecosystem_speed.py` |
| `large_transformers_llama` / `large_transformers_qwen3` / `large_transformers_vit` | same as above |
| `large_diffusers_unet2d` | same as above |

### One-off rows: current standing from reports

The following numbers are **not in any gate**; they come from their respective investigation reports. The date is the measurement date; nobody has re-measured since.

| Library / case | Device | Ratio | Status | Evidence |
|---|---|---|---|---|
| ViT | CUDA | `1.33x` | **open**, but the attribution is no longer valid, see below | `2026-08-26-common-network-training-trajectories.md` |
| ConvNet | CUDA | `1.08x` | improved | same as above |
| UNet | CUDA | `0.79x` | accepted | same as above |
| GPT-2 (math attention) | CUDA | `1.13x` | configuration-dependent, see below | `2026-08-23-ecosystem-parity-performance.md` |
| GPT-2 (explicit FlashAttention + fp16 conversion) | CUDA | `0.90–0.94x` | accepted | same as above |
| Llama (math attention) | CUDA | `1.22x` | configuration-dependent | same as above |
| Llama (explicit FlashAttention) | CUDA | about `1.03–1.04x` | conservative gap | same as above |
| TRELLIS.2 end-to-end | CUDA | `1.093x` | **open** (was `1.20x`) | `2026-08-23-verl-vllm-trellis-current-baseline.md` |
| vLLM Qwen3-0.6B | CUDA | about `0.795x` (about 20.5% faster) | accepted, 4-token protocol | same as above |
| Diffusers | Ascend 910B3 | `0.964x` | accepted | `2026-08-30-diffusers-ascend-parity-performance.md` |
| MMCV / MMEngine (tiny) | Ascend 910B3 | `0.927x` / `0.796x` | accepted | `2026-08-30-mmcv-mmengine-ascend-parity.md` |
| ms-swift LoRA (tiny, fused fp32 causal SDPA) | Ascend 910B3 | `0.969x` | accepted | `2026-08-31-ms-swift-ascend-parity-performance.md` |
| Qwen3-0.6B decoding | Ascend 910B3 | 15.90 vs 16.19 token/s | accepted | `2026-08-28-qwen3-ascend-performance.md` |
| Qwen3-0.6B BF16 SDPA generation | Ascend 910B3 | 14.92 vs 15.31 token/s | accepted, zero CPU fallback | [`2026-08-30-qwen3-ascend-training.en.md`](../results/2026-08-30-qwen3-ascend-training.en.md) |
| Qwen3-0.6B FP32 forward/loss/backward | Ascend 910B3 | `1.07–1.12x` | accepted | same as above |
| Qwen3 same-device training protocol | Ascend 910B3 | `1.063x` (was `1.195x`) | **open**, no performance gate for the exact path yet | same as above |
| vLLM Qwen3-0.6B warm-state request median | Ascend 910B3 | `0.36330s` vs native `vllm-ascend` `0.38998s` | accepted, single request, short context, TP=1 | [`2026-08-31-vllm-ascend-jittor-bootstrap.en.md`](../results/2026-08-31-vllm-ascend-jittor-bootstrap.en.md) |

The two linked reports have been condensed and migrated into [validation conclusions](../results/index.en.md); the other evidence entries are the original report names from the 2.0 remediation period, and those reports were retired when the remediation closed, remaining only in Git history (`e3c369acb`). **Nobody has re-measured these rows since**; to cite a row, first re-measure and then write the conclusion as a new `docs/results/` report.

### `fuse_op_limit` is already in the good range (measured 2026-09-22, negative result)

The video VAE profile is flat -- the top operator accounts for only `11.8%`, and one decode has `499GB` of memory traffic while the output is only `41MB`. The intuition for this shape is "fuse harder, fewer passes", so `fuse_op_limit` was swept (H3 video VAE `decode_base`, 4 frames):

| `fuse_op_limit` | decode | Relative to default |
| --- | --- | --- |
| `4` | `1.748s` | 37% slower |
| `8` | `1.236s` / `1.240s` | — |
| `12` | `1.266s` | — |
| `16` (default) | `1.275s` | baseline |
| `32` | `1.369s` | 7% slower |
| `64` | `1.529s` | 20% slower |
| `0` (no limit) | `2.694s` | **111% slower** |

**The direction is opposite to intuition: wider fusion is slower**, and no limit literally doubles the time. The flag's own description already gave the reason -- very wide fusion must keep every live intermediate in registers. The default `16` is already in the good part of the curve; **"raise the fusion limit" is a dead end**, recorded here so nobody walks it again.

**`8` looks 3% faster than `16`, but this machine cannot measure 3%.** In the serial A/B/A/B, four runs: `fuse=8` gave `1.236`/`1.240` (0.3% apart), and `fuse=16` gave `1.275`/`2.402` -- the latter nearly doubles. Machine load drifted during the measurement, and **run-to-run variance can reach 90%**, so differences below about 10% are not measurable here. The `4`, `32`, `64`, and `0` rows in the table above are far above the noise and trustworthy; the difference between `8` and `16` is not trustworthy, and the default was not changed based on it.

### Two speedup measurements from 2026-09-22

**Grouped `conv_transpose` now asks for an accelerated kernel.** The `groups == 1` branch of `conv_transpose` always called `select_kernel`, but the grouped branch never did, so every depthwise transposed convolution went through the eight-dimensional `reindex * reindex -> reindex_reduce` fallback -- even though the cuDNN adapter itself accepts `groups` and passes it to `cudnn_conv_backward_x`. The capability was always there; nobody asked for it.

| Case | Device | Before | After | Ratio |
| --- | --- | --- | --- | --- |
| H3 audio VAE `decode` | CUDA | `2.962s` | `1.404s` | `0.47x` |

That single fused operator previously accounted for `79.8%` of the whole decode, at `344MB/s` bandwidth. The two paths are **bit-identical** under depthwise, `groups < C`, padding, output_padding, and dilation (`rel = 0`), and the case asserts equality rather than a tolerance.

**`transpose_storage_view` stays off by default because measurements in the two directions contradict each other.** It makes a permutation a view of the input allocation instead of a copy. Which side is faster depends on who consumes the transpose:

| Workload | Off | On | Change |
| --- | --- | --- | --- |
| H3 video VAE `decode_base` (4 frames) | `1.581s` | `1.272s` | **24% faster** |
| `1576x768 @ 768x3072` (transposed operand feeding cuBLAS) | `594us` | `702us` | **18% slower** |
| `4096³` | `9964us` | `10260us` | 3% slower |
| `8192³` | `76485us` | `77812us` | 1.7% slower |

When the transpose feeds an elementwise/convolution chain, the view saves a copy; when it feeds cuBLAS, GEMM wants dense operands. The numerical difference in the video VAE row (`0.00151`) is **smaller** than that row's own run-to-run jitter (`0.00166`), so this is not precision traded for speed. **There is no single correct default**; measure per model.

### The ViT row's attribution has expired (measured 2026-09-22)

That row said "the dominant CUDA GEMM trails the reference", based on `27.97ms` vs `16.11ms` for the per-step plain GEMM, and concluded "a stronger CUDA GEMM backend/algorithm is needed". **That attribution does not hold today.** Four shapes measured back and forth, back-to-back, on the same card with the same process pacing:

| Shape | Jittor | Independent PyTorch | Ratio |
| --- | --- | --- | --- |
| `1576x768 @ 768x3072` (the ViT one) | `593.0us` | `583.4us` | `1.02x` |
| `1576x768 @ 768x768` | `230.2us` | `229.6us` | `1.00x` |
| `4096³` | `9963us` | `12798us` | `0.78x` |
| `8192³` | `76497us` | `78463us` | `0.97x` |

Both sides use `float32_matmul_precision = highest` (the default for both frameworks, so the gap is not a precision tier either). **Plain GEMM is already on par** -- the first two rows differ by less than two percent, and another section of this file records that this machine's run-to-run variance can reach 90%, so "on par" is all these two rows can support; they cannot be read as either side being faster. The 22% in the `4096³` row is above the noise floor, but it was **measured only once**, which is not enough for the assertion "Jittor is faster"; it only shows there is no reverse gap here that exceeds the noise.

**What was not re-measured is the end-to-end `1.33x` itself**: it needs the `nox -s ecosystem` basis, meaning an environment with independent binary PyTorch installed, and the environment for this measurement did not have one, so the case skipped outright. So this only refutes the attribution; it does not announce a ratio. Hand-rolling a ViT to stand in for that basis does not work -- the top of this file already says why: that compares two implementations.

This row is also an instance of "one-off rows silently go stale": the number sat for a month, and the direction it gives (change the GEMM backend) would waste someone's time today.

**Absolute throughput on this machine is not citable**: a `gpu_occupy` job was spinning all eight cards to 100% utilization, and every number in the table above was depressed by it. Ratios measured back and forth on the same machine are unaffected; peak percentages are.

One recorded counterexample is worth keeping: using CANN RoPE directly on Ascend reached `0.988x` and was **rejected**, because its logits and gradient trajectory differed from the reference. **A faster but inconsistent path is not a performance result.**

## The category without a reference

Native Jittor's downstream libraries -- [JSeg](https://github.com/Jittor/JSeg),
[JDet](https://github.com/Jittor/JDet),
[JittorGeometric](https://github.com/AlgRUC/JittorGeometric) -- **currently have no performance cases at all**, and this repository's `benchmarks/asv/` is framework-level (operators, optimizer steps, reductions, host↔device transfer, tiny_llama), not library-level.

There is also a methodological problem these libraries must solve first, or the measured numbers are meaningless: **they have no direct PyTorch counterpart**. JDet is not mmdetection, JSeg is not mmsegmentation; model definitions, data augmentation, and training schedules all differ. So there are only three possible bases, each with a cost:

1. **Against an equivalent PyTorch implementation** (e.g. JDet against the same-named model in mmdetection) -- the number is meaningful, but each case must be confirmed to actually compute the same thing; otherwise it compares two implementations rather than two frameworks;
2. **Against its own history** -- effective for regression detection, but cannot answer "how much slower than PyTorch";
3. **Only framework-level operators** -- already covered, but the real bottleneck of a downstream library is often in the data pipeline and scheduling.

**Do not produce a ratio before the basis is chosen.** A "JSeg is 1.4x slower than PyTorch" with an unclear basis will be cited and then cannot be refuted.

## Maintenance rules

- A new row must carry a date, commit, hardware, and evidence link;
- Gate rows are updated by the nightly; one-off rows are synchronized **when their evidence report is updated**, otherwise they are considered stale;
- A row's status is only "accepted" or "open". "Accepted" means someone judged the gap acceptable to carry, not "done measuring";
- **Record rejected faster paths** (such as the CANN RoPE above), or later maintainers will rediscover it, re-adopt it, and rediscover that it is wrong.

### Native SDPA has no registered kernel on CUDA, 2.72x behind on long sequences (measured 2026-09-22)

`python/jittor/nn/functional/attention.py` asks
`try_dispatch("nn.scaled_dot_product_attention", ...)`. Nothing on CUDA registers under
that name -- `backends/acl/kernels/install.py:53` registers it, **for Ascend only** -- so
every native call falls back to math, materializing an N×N score matrix. A flash path
exists, but it lives in `compat/torch/installers/nn/attention.py` and is reachable only
through the torch shim.

Same batch of q/k/v, interleaved A/B/A/B, medians, fp16, `hits=6 misses=0` (flash hit every time):

| Shape | Native (math) | shim (flash) | Ratio |
| --- | --- | --- | --- |
| H3 video VAE attention `2x32x1797x1797x64` | `0.0081s` | `0.0030s` | **`2.72x`** |
| GPT-2 medium `4x16x1024x1024x64` | `0.0030s` | `0.0026s` | `1.14x` |
| Llama prefill `1x32x2048x2048x128` | `0.0038s` | `0.0031s` | `1.23x` |

**The gap grows with sequence length**, consistent with O(N²) memory versus O(N). Dispersion across five repetitions is below 1%, far above this machine's noise (occupying programs were running, so absolute values are not comparable; ratios are).

The `no_grad` and with-gradient variants give the same results: as long as the backward kernel is compiled, the training capability check at `attention.py:212` passes, so the worry that "gradients block flash" is unfounded.

**That gap has been closed.** `backends/cuda/kernels/nn/flash_attention_cuda.py` registers the bridge under that name, and `nn/functional/attention.py` lazily imports it on first call (the same shape as `softmax.py` does for `softmax_cuda`). The before/after of the native entry itself, with two independent processes each running one case, to avoid in-process kernel unregistration perturbing dispatch state:

| Shape | Before (math) | After (flash) | Ratio |
| --- | --- | --- | --- |
| H3 video VAE `2x32x1797x1797x64` | `0.0081s` | `0.0029s` | **`2.79x`** |
| GPT-2 medium | `0.0030s` | `0.0026s` | `1.15x` |
| Llama prefill | `0.0037s` | `0.0030s` | `1.23x` |

Comparing the two paths **on the same inputs, within one process**: across four shapes,
`max|flash-math|` is `0.000488` (i.e. `2**-11`, relative `4.9e-4`), which is just the
rounding in float16 from flash changing the softmax accumulation order, and does not vary
with shape. A cross-process checksum only reflects aggregate sums; it is not evidence, so
it was not used to draw conclusions.

**Nobody without flash installed changes by a single bit.** If `load_backend_for` cannot find a source checkout, there is no backend, the kernel returns None, `try_dispatch` passes through unchanged, and the caller still takes the math path. The rejected cases are listed in the kernel's comments: masks, dropout, non-4-D, unequal q/k/v head counts, a head_dim for which flash has no template, and `is_causal` with unequal q/k lengths -- the last because flash's causal mask aligns bottom-right when lengths differ, while the fallback here uses `triu(diagonal=1)` aligning top-left, so the shapes match but the answers differ.

One pitfall that must be recorded: the kernel originally returned the extension output's `reshape().permute()` lazy view directly, and a later math attention in the same process then crashed inside `matmul` with a corrupted NanoVector shape. The output, like the inputs, must be `clone()`d into place.

Reproduce: `$JITTOR_LAB_ROOT/sdpa-dispatch/bench_sdpa_paths.py`. Running it requires the pybind11 headers and `JITTOR_FLASH_ATTN_JITTOR_SRC`; see `examples/flash-attention/README.md`.

### Adding the column against PyTorch: still 6–14% slower after wiring in flash, because the wrong kernel was chosen

The `2.79x` in the previous section is **Jittor before vs. Jittor after**, not against PyTorch. Reporting it as a "speedup" conclusion is misleading -- PyTorch's SDPA already uses fused attention, so that change was **catching up**, not surpassing. This section fills in the missing column.

Real PyTorch `2.13.0+cu129`, the same H20, the same shapes/dtype/`is_causal`, and the same 10 warmups + 20 repetitions taking the median (5 repetitions is not enough for torch: raw drops from `0.0024` all the way to `0.0001`, which is not warmed up, not a result):

| Shape | jittor + flash-attn 2 | torch default | Ratio |
| --- | --- | --- | --- |
| H3 video VAE `2x32x1797x1797x64` | `0.00292s` | `0.00262s` | `11%` slower |
| GPT-2 medium | `0.00258s` | `0.00244s` | `6%` slower |
| Llama prefill | `0.00303s` | `0.00266s` | `14%` slower |

**The source of the gap is not Jittor's calling path but the algorithm choice.** Pinning each torch backend in turn:

| Shape | torch cudnn (= default) | torch flash | torch mem_efficient | jittor flash |
| --- | --- | --- | --- | --- |
| H3 video VAE | `0.00262s` | `0.00280s` | `0.00296s` | `0.00292s` |
| GPT-2 medium | `0.00244s` | `0.00250s` | `0.00256s` | `0.00258s` |
| Llama prefill | `0.00266s` | `0.00289s` | `0.00303s` | `0.00302s` |

Two conclusions:

1. **Jittor's flash path and torch's flash path differ by only `3–5%`**. Layout conversion, dispatch, and crossing extension boundaries add up to that much; the pipeline is essentially caught up.
2. **On H20, cuDNN's fused attention is `6–9%` faster than flash-attn 2**, and torch picks it by default. Jittor runs the slower algorithm.

So the next step toward "make Jittor's attention no slower than torch" is **to write a cuDNN fused attention kernel for the CUDA backend**, registered under the same `nn.scaled_dot_product_attention` name with higher priority than the flash bridge. The benefit is already measured: another `6–9%` over flash-attn 2, and cuDNN ships with the CUDA stack, so no flash source checkout or pybind11 is needed.

**A refuted hypothesis, recorded so nobody walks it again.** I thought the gap came from this kernel's `permute().reshape().clone()` on each of q/k/v plus the output `clone()` -- four whole-tensor copies, about 117MB of extra memory traffic for the H3 row, enough in magnitude to explain it. Removing the three input clones measured `0.00292 / 0.00258 / 0.00302`, matching the `0.00292 / 0.00258 / 0.00303` with them kept, row by row, and correctness was unchanged. Jittor's graph eliminates the redundant copies, so those clones cost nothing and this path is dead. The clones are kept (they guard against dangling views across the extension boundary, see the previous section).

### Correction: flash has always been compilable on this machine; the earlier "never compiled" claim was wrong

The two items above (the commit message of `469d8445` and the SDPA section) both said "on this machine `torch_extensions` is completely empty; not one extension has ever been compiled", and inferred from that that the H3 run used math attention and a `2.72x` was still on the table. **That is wrong and is hereby retracted.**

The reality: the H3 deployment has its own `XDG_CACHE_HOME` (pointing at the deployment's own `_state` directory), and under it, in
`torch-shim/minimax-h3-*/torch_extensions/flashattn_jittor/official_flash_attn/`, there are **225 object files** and multiple `flash_attn_2_cuda_jittor.cpython-312-*.so`. The bridge has always compiled, and the server has always been running with `JITTOR_FLASH_ATTN_JITTOR_REQUIRED=1` -- if it truly failed to compile, startup would fail.

The cause was an **incomplete search**: only the default `~/.cache/jittor/torch-shim/` was checked, not where `XDG_CACHE_HOME` points, and "not here" was generalized to "not on the whole machine". Also, the H3 deployment and the benchmarking lab are two different `JITTOR_LAB_ROOT`s, which were treated as the same at the time.

**The `src_inc` bug itself still stands, but the scope narrows**: it only affects **source checkouts**. The H3 deployment uses an installed wheel (`jittor-1.3.11.0.dist-info`), where `jittor/src/core/common.h` exists, so the `jittor_path/src` join is correct. `469d8445` fixes the checkout side and keeps its value; but generalizations like "those 3DGS packages have never compiled" do not hold -- under the wheel install they were always fine.

**Knock-on consequence**: there is no `2.72x` waiting to be picked up in the H3 pipeline. That pipeline was already using flash.

### End to end: after wiring in flash, Llama is `1.94x` faster and GPT-2 is `3%` faster (measured 2026-09-23)

Everything above was a bare SDPA call. This section is a real-model training step, using the project's own ecosystem comparison harness (two interpreters, `REAL_TORCH_PYTHON` provides the oracle, realistic sizes, "at least ten repetitions, take the fastest"). Real torch `2.13.0+cu129`, transformers `5.5.3` on both sides (the Jittor-side adapter supports only `4.56.2` / `5.5.3`), `OMP_NUM_THREADS=1` aligned on both sides.

Same machine, same session; the only variable is `JITTOR_FLASH_ATTN_JITTOR=0`:

| Case | flash off | flash on | Contribution of this change |
| --- | --- | --- | --- |
| `large_transformers_gpt2` | `0.0851s` (`0.97x`) | `0.0825s` (`0.94x`) | `3%` faster |
| `large_transformers_llama` | `0.0865s` (`0.94x`) | `0.0445s` (`0.48x`) | **`1.94x`** faster |

The ratio is jittor/torch; less than 1 means Jittor is faster. Gradients also pass comparison one by one (100 / 75).

**To be clear: Jittor was already faster than torch on both cases** (`0.97x` / `0.94x`), and that part is not from this change. This change cut Llama in half again on top of that.

Two fixes forced by this measurement, neither aimed at passing a gate:

- `torch.get_num_threads()` honors neither `OMP_NUM_THREADS` nor `set_num_threads` (it unconditionally returns `os.cpu_count()`), so the harness refused to compare because the two sides had different thread counts -- it was right to refuse; the shim was what needed fixing. See `f9481ddf`.
- `_ecosystem_harness.py:274` finds the runner's JSON with `line.startswith("ECOSYSTEM_RESULT ")`. This machine has 384 cores, and numexpr prints a warning without a trailing newline, so the marker lands mid-line and is missed, so **a successful measurement is reported as `runner failed`**. That is the worst failure direction. Not yet fixed; recorded here.

The GPT-2 `1.13x` / Llama `1.22x` rows in the ledger were measured on 2026-08-23 on an RTX 4090 + torch 2.12.1 + python 3.11; **do not subtract them from the table above**: the hardware, torch version, and python version all changed, with months of other work in between. The table's attribution comes only from the on/off comparison on the same machine and session.
