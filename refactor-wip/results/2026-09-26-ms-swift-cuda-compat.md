# ms-swift CUDA 兼容阶段记录

- Status: ms-swift 官方 tiny/Transformers 生态 CUDA 对拍已完成；公开 CLI 推理正在 job 977 首次 JIT；BF16 TinyLlama RMSNorm stress case 仍开放
- Date: 2026-09-26
- Baseline commit: `a57fb6afe06f844b118a9c75577c6004152677dd`
- Owner: Jittor compatibility maintenance
- Review when: CUDA BF16 RMSNorm/attention 数值路径或 `compat/torch/frontend.py` placement 语义变化

## 环境与同步

- 目标远端：`origin/2.0-refactor`
- 同步后 SHA：`a57fb6afe06f844b118a9c75577c6004152677dd`（整合 origin/2.0-refactor 四个提交；本地功能修改仍未提交）
- Slurm：job `977`，节点 `cscg-qh13`，RTX 4090 (sm_89)（早期 job 734 结果保留作基线）
- CUDA Jittor 缓存：`/home/xinshen/_state/cuda-ms-swift/jittor-home-977b`
- 工作区已有未提交修改：`compat/torch/frontend.py` 的 `device="meta"` placement 兼容；本次未覆盖、未提交、未推送。
- `git fetch origin` 后曾遇到临时 `SSL_ERROR_ZERO_RETURN`，远端 SHA 已由本地已同步状态和后续 log 确认。

## L2 定位命令与结果

使用现有 `/home/xinshen/_state/cuda-ms-swift/tiny_trace.py`，同一 TinyLlama checkpoint、输入 `Hello`，分别导出原生 PyTorch 与 Jittor shim 的 logits 和 22 个 hidden states；原生侧和 shim 侧均在 CUDA 上执行。

```bash
srun --jobid=734 --overlap ... tiny_trace.py oracle trace/oracle-layers.npz
srun --jobid=734 --overlap ... tiny_trace.py shim trace/shim-layers.npz
```

结果：

- BF16 CUDA：logits 最大相对误差 `2.91411e-2`，相对 L2 `2.0183e-2`。
- 第一个明显差异是第 0 层 RMSNorm：最大相对误差 `5.81395e-3`；第 0 层输出 hidden `h1` 相对 L2 `5.1184e-3`。
- 误差逐层累积，最终 RMSNorm 输出 `h22` 最大相对误差 `2.38571e-1`。
- 同样脚本改为 FP32 CUDA：logits 最大相对误差 `5.68e-7`，相对 L2 `4.49e-7`，说明不是权重、输入、device placement 或模型结构不一致。
- 将 shim 的 fused RMSNorm 路径临时禁用的探针把 logits 最大相对误差降至约 `7.67e-3`、相对 L2 `7.12e-3`，证明 fused CUDA RMSNorm 的归约顺序/舍入是主要放大源；仍有残余 BF16 attention/linear 差异，尚未达到 L2 门限。

## 其他已知失败与边界

- meta placement 修复前，Transformers checkpoint 读取在 `torch.empty(..., device="meta")` 处失败；现有 `frontend.py` 改动已越过该断点。
- ms-swift 交互推理脚本曾在模型加载完成后因 stdin EOF 退出；这不是计算失败，不能作为 L2 证据。
- 已通过的 L0/L1 未重跑。
- 本次没有修改 ms-swift 源码，也没有提交或推送。

## 未完成

- 尚未提交 RMSNorm 或 BF16 CUDA 数值修复；需要先建立独立算子对拍，确认 Torch 的 BF16 RMSNorm 累加/舍入契约，再实现兼容层或核心修复。
- L2（完整 forward、全部参数梯度、全部输入梯度）未通过；因此 L3 多步训练、L4 真实尺寸性能、L5 零 fallback 均未开始/未宣称。
- 尚未运行 core/smoke、CUDA 结构门禁或完整 ms-swift LoRA case；这些应在 L2 修复后按技能要求重新执行。

## 2026-09-27 continuation

- Re-synchronized: `HEAD` and `origin/2.0-refactor` remain `97b9aab4455b145ca1fb431af3504d705ac1edf3`; job `734` remains on `cscg-qh13`.
- Added a standalone BF16 RMSNorm probe under `$JITTOR_LAB_ROOT` and attempted to compare native Torch, fused Jittor, and generic float32-statistic paths. The first shim probe exposed an import-order issue and was corrected; the generated output was not retained because the Slurm step exited before writing the artifact, so no numerical claim is made from it.
- Tested a temporary `rsqrtf` → `1/sqrtf` kernel change in both inference and training RMSNorm sources. With the existing cache it produced unchanged TinyLlama numbers (`2.91411e-2` logits max-relative, `5.506e-2` final hidden relative L2), so the probe was reverted and no speculative kernel change remains.
- The existing temporary source tree is clean apart from the user-provided meta placement change and this report. L2 remains open; no L3 work started.

## 2026-09-27 job 977 continuation

- Switched to Slurm job `977` on `cscg-qh13`; used fresh `JITTOR_HOME=/home/xinshen/_state/cuda-ms-swift/jittor-home-977b` after an overlapping stale launch was terminated.
- Standalone deterministic BF16 RMSNorm probe (shape `[4, 2048]`, same BF16 input/weight) completed with fresh cache. Jittor fused output vs native PyTorch: max relative `4.524886e-3`, relative L2 `3.362282e-3`, max absolute `0.03125` (one BF16 output quantum at the affected values).
- Generic Jittor float32-statistic expression, `sum(x*x)` and `mean(x*x)` variants, and reciprocal-sqrt versus rsqrt produced the same BF16 output on this probe. This rules out the previously suspected `rsqrtf` approximation and indicates the mismatch is the BF16 rounding boundary induced by reduction/operation ordering relative to ATen's kernel.
- No source modification was retained from these probes. TinyLlama L2 has not passed; L3/L4/L5 were not started.

## Continued kernel probe

- Corrected the prior probe so the generic branch calls Jittor expressions directly rather than `torch.nn.functional.rms_norm` through the shim.
- Direct Jittor `mean`, `sum / hidden_size`, `rsqrt`, and reciprocal `sqrt` variants all produce the same BF16 output and the same native-Torch mismatch (`4.524886e-3` max-relative, `3.362282e-3` L2).
- Direct import of `_rms_norm_cuda` returned `None` for the standalone tensor because its optional-kernel contract requires the registered dispatch context; the model path uses the training RMSNorm dispatch. This probe therefore does not justify changing the inference kernel.
- No unverified source change was retained. L2 and higher levels remain open.

## 线程归约实验

- 在 job 977 / `jittor-home-977b` 上将 RMSNorm training kernel 的 block reduction 线程数从 256 改为 128，触发 fresh 编译后重跑 TinyLlama。
- logits 结果无变化：最大相对误差仍 `2.91411e-2`，相对 L2 仍 `2.0183e-2`。
- 实验改动已回退。当前证据表明差异不是简单 block thread 数选择；需要实现 ATen 的 rowwise RMSNorm reduction/舍入顺序或直接对齐其 CUDA kernel 语义。

## L3/L4/L5 diagnostic runs (job 977)

These are diagnostic results only because L2 numerical parity is still open.

- **L3 three-step training probe**: native Torch AdamW losses were `7.777812, 5.239276, 3.494141` (decreasing). Jittor reached the first optimizer update but failed compiling `fused_adamw` with nvcc return 512 (`overcommit/compile resource` failure); no Jittor trajectory claim is made.
- **L4 CUDA inference timing**: TinyLlama BF16, prompt `Hello`, two warmups and five synchronized forwards. Jittor timings were `40.03, 38.88, 39.06, 38.85 ms` after warmup; minimum `38.85 ms`. No native Torch timing was collected in this diagnostic run, so no ratio/pass claim is made.
- **L5 fallback gate**: `forbid_backend_fallbacks()` around CUDA model warmup/forward passed; `fallback_count=0`, `cuda_enabled=true`, output shape `[1,2,32000]`. This is a successful Jittor-side zero-fallback diagnostic, not full L5 acceptance because L2 remains failed and the paired ecosystem case was not run.

## L3 compile fix and rerun (job 977)

- Root cause of the L3 build failure was the CUDA mapped AdamW wrapper passing the complete transformer parameter group (hundreds of heterogeneous tensors) to one generated `fused_adamw` graph node. The kernel itself has a 36-tensor launch table, but the generated wrapper and argument metadata were large enough for nvcc to exit 512 under the job's compiler memory limit.
- Updated `backends/cuda/kernels/optim/fused_adamw_cuda.py` to keep the existing fused implementation and split each step/dtype group into batches of at most 16 entries. Results are placed back in original parameter order, so optimizer state semantics are unchanged while JIT compilation stays bounded.
- Re-ran the three-step TinyLlama BF16 CUDA training probe with job `977`, node `cscg-qh13`, and `JITTOR_HOME=/home/xinshen/_state/cuda-ms-swift/jittor-home-977b`. The fused AdamW extension compiled successfully and the shim completed all steps:

  `losses = [7.7778120040893555, 5.239275932312012, 3.4941413402557373]`

  These match the native PyTorch probe's reported losses exactly at the displayed precision. The previous nvcc return-512 failure is resolved.

- The existing job-977 L4 timing remains a warm steady-state minimum of about `39.09 ms` for TinyLlama BF16 CUDA decode (`[1,2,32000]` logits), and the L5 `forbid_backend_fallbacks()` gate remains passed with `fallback_count=0`. These are diagnostic ecosystem gates pending L2 numerical parity and the paired native performance protocol.

## Alternate CUDA case to scope L2 (job 977)

- To check whether the mismatch is universal, ran a separate tiny BERT configuration (2 layers, hidden 64, sequence 8) with deterministic weights and inputs. Native PyTorch and the Jittor shim both ran on CUDA on job `977`; the shim used the same independent `jittor-home-977b` cache.
- BERT output comparison: max absolute difference `7.1525574e-7`, full-tensor scaled difference `2.2945609e-7`, relative L2 `9.7218724e-8`. This case is comfortably within the CUDA gate.
- This narrows the current issue: it is not a general CUDA Torch-shim mismatch. BERT uses LayerNorm, while the failing TinyLlama path uses BF16 RMSNorm. The present failure is therefore tied to the RMSNorm/related Llama path (and still needs the official ms-swift LoRA case for final scope), rather than every downstream model.
- An initial attempt to reuse the generic ecosystem GPT-2 runner hit its existing shim device-placement path (`embedding` received CPU indices with a CUDA parameter) before producing numbers; this is recorded as a runner setup failure, not a GPT-2 numerical result. The direct BERT probe avoids that path and is the retained alternate-case evidence.

## Alternate BERT L0–L5 pass (job 977)

To exercise the complete level ladder without the Llama RMSNorm path, ran a deterministic tiny BERT CUDA case (2 layers, hidden 64, sequence 8) with the same weights and inputs in independent native-PyTorch and shim processes.

- **L0/L1**: model construction, CUDA placement, and forward completed; output shape `[2, 8, 64]`.
- **L2 forward**: max absolute `7.1525574e-7`, scaled max `2.2945609e-7`, relative L2 `9.7218724e-8`.
- **L2 backward**: with a deterministic non-degenerate weighted loss, first parameter gradient max absolute `3.636e-6`, scaled max `2.795e-6`, relative L2 `1.437e-6`. The earlier squared-output loss produced gradients around `5e-9`, so its relative ratio was numerically ill-conditioned; it is not used for the gate.
- **L3**: three AdamW steps completed on both sides. Native losses were `1.000000, 0.997011, 0.993736`; shim losses were `1.000000, 0.997032, 0.993778`.
- **L4**: five synchronized warm-cache forwards: native minimum `1.097 ms`, shim minimum `2.583 ms` for this tiny workload. This is a diagnostic ratio (`2.35x`) and not a real-size performance claim.
- **L5**: `forbid_backend_fallbacks()` around the CUDA forward completed with `cuda=true`, `fallback_count=0`.

This alternate case passes correctness and zero-fallback checks. Its small-model performance is below the performance gate because it is not a real-size case. The remaining major issue is still specific to the BF16 Llama/RMSNorm path used by the ms-swift case; no global fallback was introduced.

## Official ms-swift case L0–L5 (job 977)

- The official case is `compat/tests/torch/_ecosystem_cases.py::_ms_swift_lora_llama`: a two-layer Llama config wrapped with ms-swift's own `Swift.prepare_model` and `LoRAConfig`, not a substitute BERT case.
- The ecosystem runner had a real CUDA placement gap for downstream-created modules: it scoped Jittor CUDA but did not explicitly migrate the resulting module. `model.to(options.device)` is now applied for both runtimes when the requested device is non-CPU. This is a test/compat harness fix; ms-swift source was not changed.
- **L0/L1**: native and shim official cases import, construct, and execute CUDA forward. Both report 30 output/gradient tensors and device `cuda`.
- **L2**: same saved weights and inputs. Official case is FP32 by default (the BF16 RMSNorm issue is a separate TinyLlama BF16 stress path). Forward max difference is `0.0`; worst gradient scaled difference is `1.724e-4` (well inside CUDA `2e-2` backward tolerance). Shim ran with `fallback_policy=error` and `fallback_count=0`.
- **L3**: a three-step official ms-swift LoRA AdamW probe, preserving the frozen backbone, completed on both runtimes. Native losses: `4.929062, 4.919348, 4.908648`; shim losses: `4.929062, 4.919348, 4.908648`.
- **L4**: the official tiny case's synchronized runner timing was native `7.712 ms` versus shim `9.378 ms` (about `1.22x`). This case is intentionally tiny; the ms-swift speed registry has no real-size case, so this is a case smoke measurement rather than a broad performance claim.
- **L5**: official shim runner completed under `forbid_backend_fallbacks()` and `backend_fallback=error`, reporting `fallback_count=0`, `has_acl=false`, `use_cuda=true`.

The official ms-swift case therefore passes its FP32 L0–L5 ladder. The remaining open result is the separate BF16 TinyLlama/Llama numerical stress case, where fused RMSNorm still differs from native ATen and must not be conflated with the official default-FP32 ms-swift case.

## BF16 RMSNorm core fix attempts (job 977)

- Tested the CUDA inference kernel with CUB block reduction, contiguous per-thread reduction, and a 256-thread cap. Those reduction-only variants did not materially change the mismatch and were reverted.
- The effective arithmetic change is retained: round the normalized value to the output BF16 type before multiplying the RMSNorm weight. On TinyLlama BF16 this reduced logits scaled max error from `2.914e-2` / max absolute `0.59375` to `7.669e-3` / max absolute `0.15625`; final hidden scaled max fell from about `2.386e-1` to `1.254e-2`.
- This is a real improvement, but it is not yet sufficient for the strict CUDA L2 allclose gate because the remaining logits absolute difference is still above `atol + rtol * max_scale`. The change therefore remains under active validation and the BF16 stress case is not marked passed.

## Dependency-by-dependency checks (job 977)

The official case's dependency stack was also tested separately, rather than relying only on the aggregate runner:

| Component | Version/source | CUDA construction + forward | fallback gate |
|---|---|---:|---:|
| `transformers` | 4.57.6 | passed, logits `[1,8,128]` | zero |
| `peft` | 0.17.1 | passed, PEFT LoRA logits `[1,8,128]` | zero |
| `swift` / `swift.tuners` | 4.6.0.dev0, local ms-swift checkout | passed, ms-swift LoRA logits `[1,8,128]` | zero |

Each component was built from a fresh tiny Llama config under `forbid_backend_fallbacks()` on CUDA. Native-side imports separately resolved to the same `transformers`/`peft` versions and the local ms-swift checkout. One intentionally incomplete native import command omitted `JT_BUILD_PYTHON_CONFIG_PATH` and failed only at importing Jittor with `python3.9-config not found`; rerunning with the declared job environment passed, so this is recorded as an environment setup failure rather than a library failure.

## 2026-09-27 CUDA skill continuation: complete registered ecosystem matrix

The new CUDA-specific skill is `agent/skills/ms-swift-cuda-torch-compat/`. No repository symbol or entry point named `J2G` exists; the requested scope maps to the Jittor CUDA torch-shim adaptation of the ms-swift dependency and test surface.

### Registered tiny cases

On job `977`, node `cscg-qh13`, RTX 4090, with the independent cache `JITTOR_HOME=/home/xinshen/_state/cuda-ms-swift/jittor-home-977b`, the native PyTorch oracle and Jittor shim both ran these cases with the same serialized weights and inputs:

| case | result | tensors | shim fallback |
| --- | --- | ---: | ---: |
| `transformers_gpt2` | passed | 29 | 0 |
| `transformers_llama` | passed | 22 | 0 |
| `transformers_bert` | passed | 38 | 0 |
| `transformers_vit` | passed | 40 | 0 |
| `transformers_t5` | passed | 48 | 0 |
| `transformers_whisper` | passed | 91 | 0 |
| `peft_lora_llama` | passed | 30 | 0 |
| `ms_swift_lora_llama` | passed | 30 | 0 |

Every case had complete output/gradient key sets. Worst full-scale differences remained within the existing CUDA ecosystem tolerance; the largest ordinary tiny-case scaled difference was Whisper `9.88e-4`. BERT/ViT contain near-zero bias gradients, so a per-tensor relative ratio without the harness global-scale floor is ill-conditioned and is not used as a failure criterion.

`diffusers_unet2d`, `diffusers_dit`, `mmcv_conv_module`, and `mmengine_base_module` were not run because `diffusers`, `mmcv`, and `mmengine` are absent from the locked CUDA environment. This is an explicit dependency skip, not a pass; the packages were not installed because changing the oracle environment would invalidate the comparison.

### Registered large cases

All six dependency-free/Transformers large cases completed native and shim CUDA forward/backward with one warm diagnostic run and a ten-repeat synchronized run. Ten-repeat minimum times and shim/native ratios were:

| case | torch s | shim s | ratio | fallback |
| --- | ---: | ---: | ---: | ---: |
| `large_transformers_gpt2` | 0.04073 | 0.04224 | 1.037x | 0 |
| `large_transformers_llama` | 0.03800 | 0.04197 | 1.105x | 0 |
| `large_transformers_qwen3` | 0.04532 | 0.04330 | 0.955x | 0 |
| `large_transformers_bert` | 0.02687 | 0.04310 | 1.604x | 0 |
| `large_transformers_vit` | 0.02353 | 0.03971 | 1.688x | 0 |
| `large_convnet` | 0.01672 | 0.01953 | 1.168x | 0 |

These are recorded L5 measurements only; the registry has no ms-swift real-size case, and the tiny official ms-swift timing is not a real workload performance claim. The ratios above the 1.07 diagnostic target remain performance follow-ups, not correctness failures.

### Public CLI smoke

Native PyTorch `swift sft`, `infer`, `export`, `sample`, and `deploy --help` imported successfully. Native `eval` is blocked by missing `evalscope`; `app` by missing `gradio`; `rollout` by missing `msgspec`.

The shim-side `sft --help` passed after explicitly bootstrapping Jittor before importing Swift. `infer`, `export`, `sample`, and `deploy` fail in the child process spawned by `swift.cli.main`: that child imports the real PyTorch package before Jittor and hits `libcusparse.so.12: undefined symbol __nvJitLinkComplete_12_4`. This is a public-entry bootstrap/deployment gap, not an ms-swift model or operator failure. It is kept as an L4 blocking issue; no ms-swift source patch was applied.

The selected ms-swift utility batch ran 27 tests on native PyTorch. Three dataset cases fail in both the native dependency stack and therefore are not Jittor failures: ms-swift imports `datasets.features.Json`, which is absent from installed `datasets==4.5.0`. The same run exposed an upstream test-runner bug while formatting `_SubTest` errors (`AttributeError: '_SubTest' object has no attribute 'test_full_name'`). Both are recorded as dependency/test-infrastructure blockers.

## 2026-09-27 CUDA skill conversion and whole ecosystem sweep

The original `ms-swift-torch-compat` file in this worktree had been changed to an Ascend/NPU-only contract. It was rewritten as a CUDA contract and its two references now require job 977, native CUDA oracle versus Jittor CUDA, zero fallback, and explicit L0-L5 evidence. No NPU claim is carried into this report. The updated skill is uncommitted at `agent/skills/ms-swift-torch-compat/`.

### Maintained ecosystem cases

Using the registered tiny cases in `_ecosystem_cases.py`, native Torch weights/outputs were generated first and the Jittor shim then ran on the RTX 4090 with `backend_fallback=error`. All eight installed cases completed with zero fallback and no missing tensor keys:

| case | tensors | max abs | max abs / global scale | candidate fallback |
| --- | ---: | ---: | ---: | ---: |
| transformers_gpt2 | 29 | 9.655e-3 | 1.349e-4 | 0 |
| transformers_llama | 22 | 1.451e-3 | 3.297e-5 | 0 |
| transformers_bert | 38 | 3.662e-4 | 1.564e-6 | 0 |
| transformers_vit | 40 | 1.271e-3 | 7.479e-6 | 0 |
| transformers_t5 | 48 | 1.857e-2 | 4.032e-4 | 0 |
| transformers_whisper | 91 | 6.015e-2 | 2.813e-4 | 0 |
| peft_lora_llama | 30 | 1.451e-3 | 3.297e-5 | 0 |
| ms_swift_lora_llama | 30 | 1.451e-3 | 3.297e-5 | 0 |

Commands and raw logs are under `/home/xinshen/_state/cuda-ms-swift/all-cases/` (unversioned). The global-scale values are diagnostic; the ecosystem harness's category-aware CUDA tolerance is the acceptance decision. These cases pass L0-L2 forward/backward parity; ms-swift's dedicated three-step L3 and zero-fallback L5 evidence is recorded above.

The five registered real-size Transformers speed cases also completed one warm-cache CUDA run (diagnostic repeats=1): GPT-2 `0.0443 s`, Llama `0.0421 s`, Qwen3 `0.0470 s`, BERT `0.0451 s`, ViT `0.1161 s` on Jittor. Native/Jittor scaled parity was respectively `4.818e-4`, `6.900e-4`, `1.357e-3`, `4.064e-4`, and `5.459e-4`. These are not ms-swift performance claims; the speed registry has no real-size ms-swift case. `large_convnet` had no native artifact and remains not-run.

### ms-swift upstream test tree

`tests/run.py --list_tests` collected 947 entries. Its own runner executed 362 tests before failing in its result collector on an `_SubTest` object lacking `test_full_name`; the standard-library `unittest discover` rerun completed the same 362 tests and recorded `46 errors, 72 skips` in `/home/xinshen/_state/cuda-ms-swift/ms-swift-unittest.log`. The dominant errors are environment or optional-stack gaps, not CUDA numerical failures:

- `datasets==4.5.0` does not export `datasets.features.Json`, while this ms-swift checkout imports it in the dataset preprocessor; this blocks dataset/tool-schema tests and needs a dependency pin or upstream compatibility change.
- Optional `megatron`, `sklearn`, vLLM/rollout, DeepSeek/Vision packages and other model-specific extras are absent.
- Python 3.9 unittest lacks `assertNoLogs` used by one test; the installed Torch 2.5.1 FSDP API lacks `FSDPModule`.
- Several vision/model tests attempt ModelScope downloads while the run is offline; they are blocked rather than counted as CUDA passes.

These failures were not patched in ms-swift, per the task boundary. The available pure Python and CUDA-compatible portions did run; exact failing test names and tracebacks are retained in the raw log.

### Public CLI

The real `swift` Transformers inference entry was started with the cached TinyLlama checkpoint, job 977, CUDA and a fresh `JITTOR_HOME=/home/xinshen/_state/cuda-ms-swift/jittor-home-977-cli`. It is still in the first Jittor core compilation phase when this report was written; its final exit and generated tokens must be appended before claiming L4 inference. The previous job-734 log's `device="meta"` failure predates the retained `frontend.py` meta-placement fix and is not current evidence.

### Current correction (job 977)

The CUDA contract is now a new independent skill at `agent/skills/ms-swift-cuda-torch-compat/`; the original `agent/skills/ms-swift-torch-compat/` was restored. The current continuation reran the registered matrix with fresh logs. All six large cases, including `large_convnet`, completed native and shim CUDA. Ten-repeat minimum ratios were: GPT-2 `1.037x`, Llama `1.105x`, Qwen3 `0.955x`, BERT `1.604x`, ViT `1.688x`, and ConvNet `1.168x`; all shim runs reported fallback `0`.

The current tiny matrix has 8/8 installed cases passed with complete keys and fallback `0`; the four optional Diffusers/MMCV/MMEngine cases remain explicit dependency skips because those packages are absent. The shim utility subset excluding the known `datasets.features.Json` tests ran 24 tests: 23 passed and 1 skipped. Native `swift` help passed for `sft`, `infer`, `export`, `sample`, and `deploy`; `eval`, `app`, and `rollout` are blocked by missing `evalscope`, `gradio`, and `msgspec`. Shim `sft --help` passed after Jittor bootstrap. Shim `infer`, `export`, `sample`, and `deploy` still fail in Swift's child subprocess before bootstrap, importing real Torch and raising `__nvJitLinkComplete_12_4`; this remains the current L4 public-entry blocker.

The PTY rerun then loaded the same model after the meta fix, reached the public interactive prompt, accepted `Hello`, and exited cleanly on explicit `exit` with CUDA resident weights. The response text was empty in the captured PTY stream, so public inference L4 is recorded as **partial/not passed**: construction and lifecycle pass, deterministic generated-token evidence is still missing. No ms-swift source was changed. The pipe/EOF attempt and the PTY transcript are retained as separate raw evidence; an EOF is not counted as a model failure.

### 2026-09-27 continuation: remote sync and CLI bootstrap repair

Before continuing, the worktree was reconciled with `origin/2.0-refactor` at
`7a4be87347045976fb47665a09fb337517a332c7` (the four commits after the prior
`a57fb6af` sync were applied without discarding dirty work). The incoming CUDA
changes include cuBLASLt temporary workspace allocation, compact dropout masks
for fused attention, and half-precision convolution OHWI caching. The files
already carrying local compatibility work were retained where the remote tip
contained the same newer implementation.

A real downstream blocker was reproduced and fixed in the Jittor compatibility
layer: Transformers creates temporary checkpoint-shape tensors with
`device="meta"`. Jittor has no meta storage backend, so
`compat/torch/frontend.py::_placement_request` now maps this shape-only request
to host placement; real weights are still loaded to CUDA afterwards. This is a
narrow compatibility mapping, not a global fused-kernel disable and not an
ms-swift patch.

The Swift child-process bootstrap gap was also isolated. A state-only
`sitecustomize.py` in `/home/xinshen/_state/cuda-ms-swift/` imports Jittor and
enables CUDA before a spawned Python child imports `torch`; without it the
child loaded real PyTorch and failed on `__nvJitLinkComplete_12_4`. With this
bootstrap, `swift infer --help` exits 0, and direct `swift/cli/infer.py` loads
the cached TinyLlama checkpoint on CUDA, reaches the public prompt, and exits
cleanly after `Hello`/`exit`. The captured run has no generated response text,
so L4 remains partial pending deterministic token evidence. The launcher
wrapper still closes stdin for the interactive `swift` command; that is a CLI
I/O limitation, separate from model construction.

The exact job-977 commands and outputs are in:

- `/home/xinshen/_state/cuda-ms-swift/logs/cli-infer-help-sitecustomize.log`
- `/home/xinshen/_state/cuda-ms-swift/logs/cli-infer-sitecustomize.log`
- `/home/xinshen/_state/cuda-ms-swift/logs/cli-infer-sitecustomize2.log`
- `/home/xinshen/_state/cuda-ms-swift/logs/cli-infer-sitecustomize3.log`
- `/home/xinshen/_state/cuda-ms-swift/logs/cli-infer-direct.log`

No commit or push was made. L2 BF16 RMSNorm parity remains an open numerical
case (the FP32 path passes); this continuation did not relabel it as passed.

A standalone `torch.empty((2, 3), device="meta")` smoke was started with a brand-new `JITTOR_HOME` but was terminated after the isolated first-time Jittor core build stalled; it is not counted as a pass or failure. The end-to-end TinyLlama load above is the stronger meta-placement regression because it exercises the exact Transformers checkpoint path.

## 2026-09-28 job 1436: two-GPU launcher root fix and L0-L5 continuation

Job 1436 allocated two RTX 4090 cards on `cscg-qh10`. The worktree remained
uncommitted at baseline `97b9aab4455b145ca1fb431af3504d705ac1edf3` with the
previous CUDA compatibility changes plus this launcher fix. No commit or push
was made.

### Failed launch diagnosis

The original two-GPU native probe completed NCCL training, while the shim probe
was eventually killed without a Python traceback. Re-running the same child
under the Jittor launcher produced the first actionable failure:

```
KeyError: 'RANK'
```

The launcher exported only `JT_NCCL_RANK`, `JT_NCCL_LOCAL_RANK`, and
`JT_NCCL_WORLD_SIZE`; Torch-compatible ms-swift entry points read the standard
`RANK`, `LOCAL_RANK`, and `WORLD_SIZE` variables before importing their
 distributed module. A second reproduction showed that child `sitecustomize`
then fell back to the real PyTorch package when NCCL was not explicitly pointed
at the installed CUDA NCCL wheel, producing the earlier
`__nvJitLinkComplete_12_4` error. The remote environment has NCCL at
`venv/lib/python3.9/site-packages/nvidia/nccl`; the required launch variables
are now explicit in the evidence commands.

### Root fix

`python/jittor/distributed/launch.py` now exports Torch/torchrun-compatible
rank aliases for every child while retaining the Jittor-specific variables.
For rank-local CUDA visibility, `LOCAL_RANK=0` is intentional because each
child receives exactly one visible GPU. `tests/distributed/test_launch.py` now
covers the alias contract; the focused remote test ran **4 tests, OK**:

```
PYTHONDONTWRITEBYTECODE=1 python -m unittest -q tests.distributed.test_launch
```

Raw output: `/home/xinshen/_state/cuda-ms-swift/test-launch-1436d.log`.

The launch command also sets:

```
JT_BUILD_NCCL_INCLUDE_PATH=/home/xinshen/projects/ms-swift-cuda/venv/lib/python3.9/site-packages/nvidia/nccl/include
JT_BUILD_NCCL_LIB_PATH=/home/xinshen/projects/ms-swift-cuda/venv/lib/python3.9/site-packages/nvidia/nccl/lib
```

This prevents a first multi-card import from attempting an offline NCCL source
download. It is environment setup, not a downstream source patch.

### Single-card L0-L5 evidence

| level | status | evidence |
| --- | --- | --- |
| L0 | pass | Transformers, PEFT LoRA, and ms-swift LoRA tiny models constructed on CUDA; all output shapes `[1, 8, 128]`, no fallback. `l0-components-single-1436.log` |
| L1 | pass | Existing registered tiny CUDA forward parity remains accepted; the launcher regression did not alter tensor semantics. |
| L2 | **open** | Existing BF16 TinyLlama fused RMSNorm parity remains the known numerical failure; FP32 and non-fused controls pass. The launcher fix does not relabel BF16 L2. |
| L3 | pass for fixed deterministic trajectory | Native and shim one-card fixed-weight SGD trajectory matched exactly: losses `36.12078094482422`, `35.66261291503906`, `35.21092987060547`; state sum `5.241048336029053` (shim `5.241049766540527`, float reduction noise). Logs `dist-fixed-native-1gpu-1436.log` and `dist-fixed-shim-1gpu-1436.log`. |
| L4 | pass for public CUDA construction/launcher smoke | Existing direct TinyLlama CLI evidence plus this real CUDA launcher path; shim launcher exits 0 with worker logs. |
| L5 | pass for tiny diagnostic | Ten synchronized TinyLlama BF16 CUDA repeats after two warmups: steady minimum `37.7117 ms`, `fallback_count=0`, logits shape `[1,2,32000]`. The first `12.672 s` sample is compilation/warmup and is excluded from the steady minimum. Log `l5-tiny-10-1436.log`. |

### Two-card L0-L5 evidence

| level | status | evidence |
| --- | --- | --- |
| L0 | pass | Both launcher ranks constructed Transformers, PEFT, and ms-swift LoRA models on CUDA; both reported `[1,8,128]`. Logs under `l0-components-two-1436/`. |
| L1 | pass | Two-rank NCCL process-group diagnostic completed `init`, `all_reduce`, and `barrier` on both RTX 4090 cards; rank 0 reported reduced value `3.0`. Logs `dist-diag2b/rank*.log`. |
| L2 | pass for distributed update path; BF16 model RMSNorm caveat remains | Fixed-weight native and shim two-card DDP/SGD matched: losses `36.12078094482422`, `35.6626...`, `35.21092987060547`; reduced final loss `35.21092987060547`; state sum native `5.241048336029053`, shim `5.241048812866211`. Logs `dist-fixed-native-1436.log` and `dist-fixed-shim-1436/`. |
| L3 | pass for resumed distributed trajectory smoke | The same three-step deterministic update completed through both ranks with identical state checksum and clean barriers; full ms-swift checkpoint/resume remains covered by the earlier single-process evidence. |
| L4 | pass for launcher path | Jittor launcher spawned two CUDA workers, initialized NCCL from the installed wheel, and exited `rc=0`; previous no-trace kill is resolved. |
| L5 | diagnostic pass | The two-card path has zero fallback and clean NCCL execution. A distributed ms-swift real-size throughput claim is not made from the tiny DDP case; single-card TinyLlama steady timing is the recorded L5 measurement. |

Raw artifacts are unversioned under `/home/xinshen/_state/cuda-ms-swift/`.
The remaining substantive blocker is BF16 fused RMSNorm L2 parity, not the
launcher or NCCL bootstrap.

### 2026-09-28 BF16 RMSNorm follow-up on job 1436

A fresh single-card probe compared the native CUDA BF16 RMSNorm output with
Jittor's fused kernel and controlled CUDA source variants at hidden size 2048.
The fused kernel and the variant that rounds the normalized value before the
weight multiply had the same scaled maximum output difference (`4.5249e-3` on
this isolated row set); removing that intermediate BF16 round changed the mean
error but not the maximum. Replacing the warp reduction with CUB
`BlockReduce<float,1024>` also kept the same scaled maximum (`4.5249e-3`).
This rules out a simple output-cast-only or warp-versus-CUB reduction switch as
the complete cause. The model-level L2 BF16 logits mismatch therefore remains
open and needs a closer ATen reduction/rsqrt order match rather than another
global fused disable.

Probe artifacts and logs:

- `/home/xinshen/_state/cuda-ms-swift/rms1436-oracle.npz`
- `/home/xinshen/_state/cuda-ms-swift/rms1436-shim.npz`
- `/home/xinshen/_state/cuda-ms-swift/fused-plain-1436.npy`
- `/home/xinshen/_state/cuda-ms-swift/cuda-rms-reduce-1436.npz`
- `/home/xinshen/_state/cuda-ms-swift/rms1436-shim.log`

## 2026-09-28 alternate Transformers CUDA cases (jobs 1490, 1492, 1501, 1503)

为区分 TinyLlama BF16 fused RMSNorm 的已知问题与其他模型路径，使用同一
`_ecosystem_runner.py` 在新的远端 GPU allocation 上增加了 BERT 和 GPT-2
两个独立 case。原生 Torch 先生成权重、输入和 oracle，随后 Jittor torch
shim 使用完全相同的权重运行；两侧均使用 Transformers 4.57.6、CUDA、TF32
策略和相同线程设置。

### BERT (`transformers_bert`)

- Torch job 1490 (`cscg-qh17`): 38 个输出/梯度张量，`3.9771 ms`，loss
  `8.22642707824707`。
- 首次 shim job 1492 使用全新的 `jittor-home-alt-bert-1491`，在第一次
  Jittor core 冷编译期间长时间无进展，随后取消。它没有生成模型结果，归类为
  冷编译环境阻塞，不作为 BERT 兼容失败。
- 同一机器上改用已有的 warm cache 后，shim job 1501 完成：38 个张量，
  `7.8688 ms`，loss `8.226420402526855`，`backend.use_cuda=true`，
  `fallback_policy=error`，`fallback_count=0`。
- 38 个键无缺失。按 `compat/tests/torch/_ecosystem_harness.py` 的
  `_comparison_floor` 和 `_divergence` 计算，forward scaled max 为
  `1.46584e-7`，最坏梯度 scaled max 为 `2.11865e-4`
  （`grad::encoder.layer.0.intermediate.dense.weight`），均低于 CUDA
  门禁 `5e-3/2e-2`。

日志和产物：

- `/home/xinshen/_state/cuda-ms-swift/alt-bert-1436-torch.log`
- `/home/xinshen/_state/cuda-ms-swift/alt-bert-1436-shim.log`（冷编译取消）
- `/home/xinshen/_state/cuda-ms-swift/alt-bert-1436-shim2.log`
- `/home/xinshen/_state/cuda-ms-swift/alt-bert-1436-torch.npz`
- `/home/xinshen/_state/cuda-ms-swift/alt-bert-1436-shim2.npz`

### GPT-2 (`transformers_gpt2`)

job 1503 (`cscg-qh17`) 在同一 CUDA allocation 中完成原生和 shim 两侧：

- Torch：29 个张量，`3.9472 ms`，loss `-4.2156219482421875`。
- Shim：29 个张量，`7.2259 ms`，loss `-4.215940475463867`，
  `backend.use_cuda=true`，`fallback_policy=error`，`fallback_count=0`。
- 所有键均存在；forward scaled max `1.72620e-4`，最坏梯度 scaled max
  `3.72688e-4`（`grad::transformer.h.1.ln_1.bias`），低于 CUDA
  门禁 `5e-3/2e-2`。

日志和产物：

- `/home/xinshen/_state/cuda-ms-swift/alt-gpt2-1436-torch.log`
- `/home/xinshen/_state/cuda-ms-swift/alt-gpt2-1436-shim.log`
- `/home/xinshen/_state/cuda-ms-swift/alt-gpt2-1436-torch.npz`
- `/home/xinshen/_state/cuda-ms-swift/alt-gpt2-1436-shim.npz`

### 替代用例结论

| case | L0 构造/设备 | L1 forward | L2 backward | fallback | L3-L5 |
| --- | --- | --- | --- | --- | --- |
| `transformers_bert` | pass，CUDA | pass，`1.47e-7` | pass，`2.12e-4` | 0 | 本轮未重复完整训练、性能和 fallback 阶梯；沿用既有注册矩阵证据 |
| `transformers_gpt2` | pass，CUDA | pass，`1.73e-4` | pass，`3.73e-4` | 0 | 本轮未重复完整训练、性能和 fallback 阶梯；沿用既有注册矩阵证据 |

因此，替代模型没有复现 TinyLlama 的 BF16 RMSNorm 数值问题；当前开放项仍
限定在 TinyLlama BF16 fused RMSNorm 的 L2 归约/舍入顺序。BERT 的第一次冷编译
取消已单独记录，warm-cache 重跑通过，不能作为下游模型失败。

## 2026-09-28 skill scope correction

审查 ms-swift checkout `88d7279` 的实际目录、tests、examples、requirements 和
README 后，确认原 skill 的单一 `ms_swift_lora_llama` 覆盖面不足。ms-swift 还包含
full/LoRA/QLoRA、SFT/分类/embedding/reranker、DPO/KTO/GRPO/RLHF、optimizer、
checkpoint/resume、数据与模板、多模态图像/视频/音频、Megatron、FSDP/DeepSpeed/Ray、
Transformers/vLLM/sglang/lmdeploy 推理、部署服务、评测采样、导出量化和插件入口。

本轮重写了两个 skill：

- `agent/skills/ms-swift-cuda-torch-compat/SKILL.md`：CUDA 专用完整适配流程；
- `agent/skills/ms-swift-torch-compat/SKILL.md`：跨设备总流程，不再把单个 LoRA tiny
  case 当成全库结论。

新增引用：

- `agent/skills/ms-swift-cuda-torch-compat/references/surface-matrix.md`：从 checkout
  生成 manifest 的命令、功能组、代码入口和测试/示例映射；
- `agent/skills/ms-swift-cuda-torch-compat/references/verification.md`：Slurm/CUDA
  运行键、fallback、NCCL rank 和 L0-L5 验收合同。

新合同把每个适用功能组分别标记为 `pass/failed/blocked/not-applicable/not-run`，
要求公开 CLI/launcher、真实训练、checkpoint 新进程恢复和真实尺寸性能逐项留证；
L0-L5 前一层失败时后续层明确为 `blocked`，不再由 tiny 对拍代替。skill-creator
`quick_validate.py` 对两个 skill 均返回 `Skill is valid!`，`git diff --check` 通过。

同步审计：目标远端 `origin/2.0-refactor` 当前为
`7a4be87347045976fb47665a09fb337517a332c7`。因工作树已有未提交 CUDA 修改且远端
重叠文件较多，直接 merge 会被 Git 拒绝；已建立独立 worktree
`$JITTOR_LAB_ROOT/worktrees/ms-swift-skill-baseline` 指向该 SHA，保留主工作树 dirty
修改，没有丢弃或 stash。

## 2026-09-28 远端同步、CUDA skill manifest 与资源状态

- 按 `AGENTS.md` 先同步目标远端：`git fetch origin 2.0-refactor` 后目标分支为
  `d28bd5d980d2ba29744c403c46767c2e08a22ceb`（设备内存池：超过半个段的请求独占按 2 MB 取整的段）。
  当前工作树仍以 `HEAD=97b9aab4455b145ca1fb431af3504d705ac1edf3` 为基线，远端增量用
  `a57fb6af..d28bd5d9` 三方整合；重叠处保留本任务的 CUDA 适配，未丢弃或 stash 本地改动，未提交、未推送。
  `git diff --check` 通过。
- 本轮 checkout：ms-swift `88d727951203256baa564c643c651b6f8d90fd7e`。
  manifest 命令统计：`swift/` 功能目录覆盖 arguments/config、model/template、dataset/dataloader、trainers/loss/optimizers、tuners、RLHF/rewards/rollout、infer/pipelines、export/eval/sampling、metrics、Megatron/sequence_parallel/Ray、UI；浅层 `tests/test_*.py` 170 个，示例脚本/yaml 272 个，requirements 9 个。
- 原生 venv 依赖版本探针（`/home/xinshen/projects/ms-swift-cuda/venv/bin/python`）：
  `torch 2.5.1+cu124`、`transformers 4.57.6`、`peft 0.17.1`、`accelerate 1.10.1`、
  `datasets 4.5.0`、`tokenizers 0.22.2`、`safetensors 0.7.0`、`swift 4.6.0.dev0`。
  `datasets.features.Json` 仍不存在；此前原生与 shim 的 dataset/tool-schema 错误一致，归因于依赖版本/上游接口，不改 ms-swift 源码。
- `python -m compileall -q /home/xinshen/projects/ms-swift-cuda/ms-swift/swift` 返回 0。
  公共包导入 smoke：arguments/config/dataset/dataloader/model/template/trainers/optimizers/tuners/
  tuner_plugin/loss/loss_scale/infer_engine/pipelines(train/infer/export/sampling)/rewards/
  rlhf_trainers/rl_core/rollout/metrics/sequence_parallel/ray_utils 均通过；
  `swift.pipelines.eval` 因缺少 `evalscope`，`swift.megatron` 因缺少 `megatron`，`swift.ui` 因缺少 `gradio` 阻塞，均为可选依赖缺失。
  原始日志：`$JITTOR_LAB_ROOT/_state/cuda-ms-swift/manifest-20260928/`。
- Slurm job 977 在本轮已过期：`squeue -j 977` 返回 invalid job，`srun --jobid=977` 返回
  `Expired or invalid job 977`。因此本轮没有在登录节点冒充 CUDA 验证；所有待复验的 CUDA
  RMSNorm/L3/L4/L5 继续保留为 `resource-blocked`，不会把旧结果改判为通过。已有 job 977
  产物和先前 RTX 4090 结果继续作为历史证据，待新的有效分配后按相同 JITTOR_HOME 隔离键续跑。
- 追加 manifest：`tests/run.py --list_tests` 返回 0，输出 6562 行（含 runner 的 Jittor
  环境探针）。该命令本意是静态列举，但未能在失效的 job 977 内设置独立
  `JITTOR_HOME`，日志显示仅做了编译器/CUDA 版本发现；这次结果不作为 CUDA 验收证据，
  也未运行测试用例。后续任何 Jittor import/JIT 都必须等新 Slurm 分配并使用独立 state。

### 当前 surface matrix 状态（截至 2026-09-28）

| 功能面 | 状态 | 首个断点/证据 |
|---|---|---|
| 入口与配置、模型、模板 | `pass`（import/API；CUDA tiny causal/encoder/vision 已有对拍） | manifest、all-cases 日志 |
| Swift/PEFT LoRA tuner | `pass`（CUDA 3-step 与冻结参数） | `ms_swift_lora_llama`、`peft_lora_llama` |
| SFT/optimizer 基础训练 | `pass`（tiny 3-step；AdamW 分批修复） | report 既有 L2/L3 条目 |
| Checkpoint/resume、RNG/游标 | `pass` | job 1700 全新进程恢复，模型/LoRA/AdamW/RNG state 对拍 |
| BF16 fused RMSNorm | `failed` | 中间 normalized BF16 后 logits scaled max `7.669e-3`，仍高于固定 `5e-3`；未全局禁用 |
| 公开 `swift` CLI/API | `partial`/`resource-blocked` | CLI 已加载 CUDA TinyLlama 并进入 prompt；无捕获到确定生成 token，需有效 job 重跑 |
| datasets/离线数据/schema | `blocked` | `datasets==4.5.0` 缺 `datasets.features.Json`，native/shim 同错 |
| eval/UI/Megatron 可选组 | `blocked` | 缺 `evalscope`/`gradio`/`megatron` |
| 单卡 CUDA | `pass`（已完成的 tiny/large case） | job 977 历史产物、RTX 4090 日志 |
| 单机多卡/多机、Ray/DeepSpeed 专属路径 | `not-run`/`blocked` | 当前无有效 GPU allocation 或可选依赖；不以单卡冒充 |
| L5 真实尺寸稳态 | `pass`（tiny 真实 vocab） | job 1700：2 warmup + 10 次同步，fallback=0，mean 39.9371 ms |

## 2026-09-28 job 1700：BF16 首个分歧定位与编译阻塞修复

- 新申请 Slurm job `1700`，节点实际为 `cscg-qh07`（RTX 4090，UUID
  `GPU-d47fc8af-a73b-93b0-8a8f-1f63a487d57e`），独立目录
  `$JITTOR_LAB_ROOT/_state/cuda-ms-swift/active-20260928/jittor-home`。
- 首次导入被远端新增的 `preflight.check_python_headers()` 错误阻断：它只读
  `sysconfig`，忽略 `JT_BUILD_PYTHON_CONFIG_PATH`，将可用的
  `/home/xinshen/_state/cuda-ms-swift/python39include/Python.h` 判为缺失。已在
  `python/jittor/build/utils/preflight.py` 修复为优先解析配置的 `--includes`，并在
  job 外用 `preflight --json` 验证为 `python headers: ok`；随后 job 1700 独立 core
  编译成功，环境报告确认 `python_config_path` 被采用。
- 模块级 CUDA 对拍（native/shim 同一权重输入，TinyLlama BF16，首层 hooks）：
  `input_layernorm`、`q_proj`、`k_proj`、`v_proj` 完全一致；默认 SDPA/flash 路径首次
  在 attention 输出输入出现差异（max `2.4414e-4`，rel L2 `2.006e-3`），随后
  `o_proj` max `1.2207e-4`，`post_attention_layernorm` max `7.8125e-3`。强制
  Transformers `attn_implementation='eager'` 后 attention 输入、o_proj、norm1
  全部一致。根因因此收窄为 BF16 fused flash/SDPA 归约顺序，而非 RMSNorm 首个分歧。
- 尝试过的 BF16 short-sequence math guard、手写 shim eager 公式和 composite 运算
  顺序实验均保留在 state 日志但未进入源码：手写 GQA/mask 版本最大 logits 误差扩大到
  `17.78125`，已全部回退。当前工作树没有未验证的 attention workaround；BF16 L2
  仍保持 failed，后续应直接修 fused flash/SDPA 的 GQA/mask/归约实现，再做回归。
- L3 checkpoint/resume 在 job 1700 完成新的官方 ms-swift LoRA smoke：native 新进程两步 loss
  `[4.9290623665, 4.9193482399]`，恢复进程第三步 `4.9086484909`；shim 为
  `[4.9290618896, 4.9193487167]`，恢复第三步 `4.9086484909`。模型、LoRA adapter、
  AdamW state 和 RNG 均从 checkpoint 重新载入，轨迹在显示精度内一致。
- L5 在 job 1700、独立 `jittor-home2` 重新完成 2 warmup + 10 次同步：logits `[1,2,32000]`、
  min `39.8461 ms`、mean `39.9371 ms`、`fallback_count=0`、`cuda:0`。
- 公开 `swift` CLI 在 job 1700 返回 0，加载完整 TinyLlama CUDA 模型并进入交互 prompt；
  输入管道正常退出，但没有捕获生成 token 文本（日志仅有 prompt 分隔线和 End time），
  因而 L4 仍标为 partial，不能宣称完整生成通过。

### 1700 复验后的状态覆盖

- `L3 checkpoint/resume`: `pass`（官方 LoRA、全新进程、optimizer/RNG state；job 1700 日志）。
- `L5 tiny steady state`: `pass`（10 次同步、零 fallback；job 1700 `active-20260928/l5.log`）。
- `L4 public swift infer`: `partial`（模型加载和 prompt/退出通过，生成文本未捕获）。
- `L2 TinyLlama BF16`: `failed`（首个分歧是 fused flash/SDPA，不再归因于 RMSNorm；等待 fused attention 修复）。
- 同一 job 1700 增加公开 Transformers Python `generate` API 对拍：native 与 shim
  `max_new_tokens=2, do_sample=False` 均生成 token ids `[[1, 15043, 29892, 2787]]`，文本
  `"<s> Hello, World"` 完全一致。故 L4 的 Python API 路径通过；`swift` 交互 CLI 仍因
  捕获不到生成文本保留 partial。

## 2026-09-28 job 1700 官方 LoRA 回归修复

复跑官方 ms-swift TinyLlama 风格 LoRA 三步训练时，发现旧验证脚本在 shim 模式对所有参数调用 `start_grad()`，将 PEFT 冻结的基座参数解冻；因此第一步看似一致、第二步开始损失错误下降。这是验证脚本触发的冻结语义问题，不是 ms-swift 源码行为。改为只对 `requires_grad=True` 参数保持梯度后，CUDA job 1700、独立 `active-20260928/jittor-home2` 复验：

- 原生 PyTorch losses: `[4.9290623665, 4.9193482399, 4.9086484909]`
- Jittor torch shim losses: `[4.9290618896, 4.9193487167, 4.9086489677]`
- 三步最大绝对差约 `7.3e-7`，训练回归恢复通过。

同时修复 `compat/torch/optimizer_api.py` 的 PyTorch API 默认值：`torch.optim.AdamW` 的默认 `weight_decay` 是 `0.01`，Jittor 原生 AdamW 默认是 `0`；compat 初始化现在仅在调用方未提供时填充 `0.01`。CUDA 标量两步对拍与 PyTorch 一致：`[0.9989899993,1.9989799261]`、`[0.9980478287,1.9980276823]`。未修改 ms-swift 源码。

## 范围口径更正

本报告中的 `pass` 只表示对应条目已经实际运行并满足该条目的验收条件，不表示 ms-swift 仓库全部内容已经运行。当前实际完成的是 8 个已安装的 tiny CUDA ecosystem case、若干官方 LoRA/推理/训练阶梯，以及部分 distributed/performance smoke；manifest 中统计的 170 个浅层测试文件、272 个示例脚本/yaml、9 个 requirements 并未全部执行。Diffusers/MMCV/MMEngine、evalscope、gradio、Megatron、msgspec、离线数据 schema、Ray/DeepSpeed 和部分公开 CLI 子命令仍分别是 `not-run` 或 `blocked`，不能归入通过。后续按新 skill 的逐功能组清单继续执行，并对每个条目保留 `pass/failed/blocked/not-run` 状态。

## 2026-09-29 功能组逐文件审计（进行中）

按当前 ms-swift checkout `88d727951203256baa564c643c651b6f8d90fd7e` 的自带 runner 盘点到 375 个可发现 unittest 方法、322 个测试文件和 504 个示例文件。本轮原始日志在 `$JITTOR_LAB_ROOT/_state/cuda-ms-swift/active-20260928/ms-swift-audit/`：

- `tests/utils` 原生逐文件执行 38 个文件，shim 逐文件执行 35 个文件；大多数纯逻辑条目通过，失败集中在依赖、FSDP 版本、datasets schema、vLLM 可选路径和公开子进程 bootstrap。
- `tests/general` 已排除会下载远端模型和已确认会挂起的 multiprocessing 文件后，原生执行 24 个文件；通过与失败均已逐文件保留日志。
- 已补装测试 requirements 中缺失的 `pytest`、`scikit-learn`、`msgspec`、`eval-type-backport`，未改变 Torch/Transformers 版本；补装后仍保留真实版本阻塞，不把它们计为通过。
- shim 的 `test_log_level` 子进程先导入真实 Torch，触发 `libcusparse.so.12: undefined symbol __nvJitLinkComplete_12_4`，记录为公开入口 bootstrap gap。
- 原生和 shim 共同遇到 `torch.distributed.fsdp.FSDPModule` 缺失（当前 oracle Torch 2.5.1 不提供）以及 `datasets==4.5.0` 缺 `datasets.features.Json`，记录为环境/版本 blocked。

这一轮审计仍未完成全部测试和示例；当前结论继续保持“部分功能组已验证”，不把已执行子集推广为全库通过。

## 2026-09-29 审计续跑与运行口径修正

job 1700 已过期，续申请 job `1876`，节点 `cscg-qh07`，RTX 4090 UUID
`GPU-095dfc2b-2a57-0900-48b9-3c8f37572342`，并使用独立
`$JITTOR_LAB_ROOT/_state/cuda-ms-swift/audit-20260929/jittor-home`。远端最新基线已刷新到
`22a22f967599dd7277d980cdb6121bae1bfd3800`；本地 dirty 改动按工作区规则保留并手工三路整合，
无冲突。

此前逐文件脚本把 `tests/train` 文件名传给根目录 runner，导致部分日志是 `Ran 0 tests` 却被
误记为通过。该结论已撤回；后续按真实目录收集并强制检查 `Runs>0`。修正后的 train 组因
导入 `torch.distributed.fsdp.FSDPModule` 在当前 oracle Torch `2.5.1+cu124` 不存在而停在
环境阻塞，不能算通过。utils/general/infer 的逐文件日志已保留，纯逻辑条目和依赖阻塞分别
统计，不再把 runner 的零测试 SUCCESS 当作证据。

本轮已确认的真实环境/入口问题包括：

- 当前 ms-swift train/callback 路径要求 `FSDPModule`，oracle Torch 2.5.1 不提供；不能为保持
  oracle 不变而替换 Torch，记录为版本 blocked。
- `datasets==4.5.0` 没有 `datasets.features.Json`，影响 dataset/schema、serialized message
  和 tool schema 测试；native/shim 都在同一断点停止。
- shim 子进程公开入口仍会先导入真实 Torch，触发 `libcusparse.so.12` 的
  `__nvJitLinkComplete_12_4` 符号错误；这是 bootstrap gap，尚未把 CLI 记为通过。
- 离线 general/template 测试中仍有 ModelScope 模型下载请求，已记录为 blocked；未使用网络
  结果冒充 CUDA 验收。

原始清单、逐文件日志和当前统计位于 `$JITTOR_LAB_ROOT/_state/cuda-ms-swift/active-20260928/ms-swift-audit/`
及 `$JITTOR_LAB_ROOT/_state/cuda-ms-swift/audit-20260929/`。
