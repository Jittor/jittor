# MiniMax-H3 through jittor + torch-compat + vLLM-Omni: the shim gaps

- 状态：接受。单卡与两卡 TP2 的 vLLM-Omni 请求端到端正确：单卡首个完整 `fl2va`
  请求 `GENERATE` 由 710.7 s 降到 39.6 s，OpenAI 服务与离线路径持平；TP2 的噪声画面
  （§34）、粉色标题卡（§35）与部署侧视频噪声（§48–§49）都已修复，去掉临时绕过后
  4/4 段视频正常。**仍开放**：参考 VAE 解码比真 PyTorch 慢 1.33x（§46–§47），以及
  文末「未结事项」
- 日期：2026-09-14 至 2026-09-22（速度 2026-09-15，TP2 噪声画面 2026-09-17，部署噪声
  2026-09-22）；2026-10 压缩归档
- 基线提交：最终结论对应 `2198d438`（§49 的修复为 `a60d12f9`）；各节修复的提交见正文
- 验证范围：NVIDIA H20 96 GiB，单卡与两卡 TP2；vLLM-Omni MiniMax-H3 recipe（层级
  offload），`FLASH_ATTN` 走经 shim 构建的官方 flash-attention 扩展；对照为同机真
  PyTorch（§33 的 diffusers 路径、§37–§47 的 VAE 解码类）。同机有其他租户长期占满
  全部 GPU，绝对时间只能当上限、比值才可引用（§40）。未覆盖：ROCm、NPU、TP>2、量化
- 维护者：Torch 兼容层维护者
- 复查条件：offload 路径、`torch.device` 放置、conv/pow op-type 表、`.data`/视图处理、
  `torch.as_strided`/`empty_strided`、matmul dtype 路由、triton bridge 的 guard 与同步
  策略、`fuse_op_limit` 默认值或执行器的跨线程计算流交接（`backend_compute_stream_*`）
  变化

Condensed from a ~6,700-line lab notebook; the diary, raw logs and retracted
hypotheses are in git history (`git log -p -- <this file>`). Section numbers are
kept because code comments, tests and other documents cite them. Lab harnesses
(`serve-vllmomni.sh`, `run_tp2_probe.sh`, `probe_*.py`, `env-jittor.sh`) live in
`$JITTOR_LAB_ROOT/minimax-h3/`, run state in `$JITTOR_LAB_ROOT/_state/h3/`.

## Question

Can `jittor` + `JITTOR_TORCH_SHIM=1` + `vllm-omni` run MiniMax-H3 inference
without patching vLLM-Omni? The reference run is a single H20 with the
layerwise-offload profile from `recipes/MiniMaxAI/MiniMax-H3.md` (the DiT and
the Qwen3-VL encoder cannot be resident together: ~62 GiB + ~63 GiB of BF16
weights). Each row below names the jittor-side defect, its fix and, where one
exists, the regression test.

## Findings by section

### Engine construction and the first request (§1–§11)

| § | defect | fix / evidence |
| --- | --- | --- |
| 1 | `TCPStore` bound `127.0.0.1` while clients dialled `::1` on a multi-address `localhost` | bind the first address the client dials; `tests/distributed/test_process_store.py::TestHostnameRendezvous` |
| 2 | `with torch.device("cpu"):` did not move the allocation default | thread-local device-context stack consulted by factory placement |
| 3 | op-type tables picked by the runtime `use_cuda` flag, so host units got CUDA-only `pow`, half `abs` and half comparisons | `OpByType::expand_op(args, is_cuda)` from the unit's own `JIT_cuda`/`JIT_cpu` |
| 4 | `nn.Conv3d` rejected `padding_mode` | shim pads with the requested mode, then convolves unpadded |
| 5 | `torch.get_device_module` missing | added |
| 6 | `tensor.data = other` copied elements | rebind with `_update`, as torch replaces the data |
| 7 | size-changing rebind aborted on lazily recorded views | a recorded view is re-derived only while it still fits; `tests/core/test_transpose_view_staleness.py` |
| 8 | `torch.as_strided` / `torch.empty_strided` missing | added (gather-based window; contiguous allocation) |
| 9 | cross-device `copy_` moved the destination to the source's device | materialise the source on the destination's device first |
| 10 | core helpers allocated on the ambient placement | `placement_scope_like`; `*_sdp_enabled` getters; `dtype.is_complex` callable bool |
| 11 | a mixed fp32/fp16 `linear` matched no cuBLAS row and fell back to a materialised outer product (53.6 ms against 0.49 ms fp16 at `512x2048x6144`) | resolve the pair to the autocast compute dtype and retry the relay (`c34b1240`) |

With §1–§11 one `fl2va` request completes (124 frames, 165,600 audio samples per
channel, 24.3 GiB peak on a 96 GiB card); §11 is what made it fast.

### Serving, correctness and single-GPU speed (§12–§18)

| § | defect | fix / evidence |
| --- | --- | --- |
| 12 | `from torchaudio.functional import melscale_fbanks` failed in the stub | `_AnyFinder` for `torchaudio.*`; `vllm-omni serve ... --omni --num-gpus 1` returns the clip in 39.5 s against 39.6 s offline |
| 13 | TP2 bring-up: `JITTOR_TORCH_DISTRIBUTED_AUTO_INIT` must be exported; NCCL without mpirun needs `JT_BUILD_NCCL_INCLUDE_PATH`/`_LIB_PATH`; a `make_cache_dir` race; `JT_NCCL_ROOTINFO_FILE` not derived when a store is present; sub-group communicators created on the wrong device; `record_event` on a foreign context; `share_with` not range-checked; extension `device()`/`CUDAGuard` pinned to index 0 | the first two are environment settings, the rest are fixed in jittor or the shim; `compat/tests/torch/test_cpp_extension_device_index.py` (6/6) |
| 14 | `linspace` did not land exactly on `end`, so every request with >= 4 steps failed `sigma_next >= 0` | endpoint pinned; `tests/ops/test_random_op.py`. The lab had also used 2 steps where the recipe says 50. The shim's VAE decode agrees with real torch to 0.30/255 (256x256) and 0.31/255 (512x512) |
| 15 | speed at 256x256, 50 steps: 3.35 s/step with the DiT offloaded, 1.68 s/step with it resident | first lever is residency, second the attention backend (§16) |
| 16 | `int(cu[1])` parked `cu_seqlens` on the host and the shim's "already on cuda" shortcuts trusted metadata, so `FLASH_ATTN` refused it; `TORCH_SDPA` loses packed multi-document boundaries above 256x256 | shortcuts require real residence; `compat/tests/torch/test_native_tensor_placement.py::test_a_device_var_parked_on_the_host_is_moved_back_by_to_device`. 832x480: 10.09 s/it against `TORCH_SDPA`'s 25.2 s/it |
| 17 | the extension build cache ignored header edits and served a stale `.so` | shim ABI header digest in the command (`-DJTORCH_SHIM_ABI=`); `TestShimHeadersInvalidateTheBuildCache` |
| 18 | orphaned workers and leftover servers produced wrong-looking results | operational: stop by process tree, check `ps` before blaming jittor |

### TP2 device and threading defects (§19–§32)

| § | defect | fix / evidence |
| --- | --- | --- |
| 19 | intermittent TP2 startup hang: rank 0 holds the GIL inside `ncclCommInitRank` while its store server must answer rank 1 | two-phase barrier before the collective; `tests/distributed/test_nccl_store_rendezvous.py` |
| 20 | the six generated direct flash-attn entries emitted `CUDAGuard device_guard{0}` | guard from the input tensor; `compat/tests/torch/test_flash_attn_compat.py::TestPackedEntryDeviceGuard` |
| 21 | an unreadable index buffer on device 1 | diagnosis only: the context was already dead, the read is a victim |
| 22 | `ArrayOp::run` freed through a null allocator | skip the free; `tests/core/test_array.py::TestArrayOpDoesNotAssumeAnOutputAllocator` |
| 23–24 | four loader threads corrupt the fused batch (`fused_op.cc:89`); the request phase itself is single-threaded | confirmed by `SERIALISE=1`; fixed in §29 |
| 25 | suspected lying `transpose` strides | withdrawn: CUDA transpose is eager; the attempted `set_storage_strides` broke values and was reverted |
| 26 | `accelerator_current()` was process-global while CUDA's current device is per thread, so loader threads ran on device 0 | bind the calling thread on first use; `tests/runtime/test_runtime_device_state.py::test_current_device_binds_the_calling_thread` |
| 27 | triton bridge launched on the legacy stream; `jt.unique` and the `*_like` family used the ambient device; `device_raw_ptr` migrated to the ambient device | `_launch_stream()`; `device_scope_like`; `tests/ops/test_unique_paths.py::TestUniqueOffTheAmbientDevice`, `tests/core/test_array.py::TestLikeConstructorsKeepTheDevice` |
| 28 | flash-attn build identity did not cover the shim headers | `shim_hdrs=` digest in the identity |
| 29 | a batch node was freed under the planner by another thread | the executor holds a `VarPtr` for every planned var, plus an edge snapshot: 20/20 loader-race runs clean (keep-alive, lock and snapshot-only variants each failed) |
| 30 | the deployed core lacked the per-thread-stream set; `_as_strided` gathered out of bounds | lab redeploy; `as_strided` now validates the view against the storage |
| 31 | `Tensor.to(1)` silently dropped the device index | a bare int (not `bool`) is a device index; `compat/tests/torch/test_multi_device.py::TestMultiDeviceFacade::test_to_and_cuda_with_an_index` |
| 32 | `_Storage.nbytes()`/`set_` described the tensor, not its allocation; the triton driver was pinned to device 0 and switched context before materialising | byte-addressed storage (`74bfe44c`); per-device driver and context order (`95ad1ea2`); `compat/tests/triton/test_triton_backend.py::TestDriverIsPerDevice` |

After §32, TP2 at 256x256, 2 steps, `FLASH_ATTN` completes: 495.3 s without
offload, 50.1 s with the default text-encoder offload, zero illegal addresses.

### Picture correctness (§33–§35)

- **§34 TP2 frames were noise.** The shim's `torch.Generator` kept only a seed and
  drew from jittor's global stream, so the two ranks denoised different latents.
  Each generator now owns a stream (`0ea3448c`);
  `compat/tests/torch/test_generator_streams.py`. TP1 against TP2 at 2 steps:
  block SSIM 0.972–0.985 (TP1 against itself: 0.942–0.985).
- **§35 the pink title card.** The triton bridge's over-read guard bounced
  strided `chunk` views as if contiguous, silently corrupting the AdaLN modulation
  kernels on both ranks. The bounce is taken only for contiguous operands;
  `TestGuardedBounceRequiresContiguous`. Block output cos against diffusers went
  from 0.18744 to 0.99978, and all four end-to-end clips follow the prompt.

## Speed

### Against real torch on the diffusers path (§33)

`infer_h3.py`, 512x512, 124 frames, 6 steps, `--vae-dtype float16`, one GPU each:

| phase | jittor, cold | jittor, warm | torch | warm / torch |
| --- | --- | --- | --- | --- |
| dit | 61.41 | 25.92 | 23.35 | 1.11x |
| text_encoder | 17.25 | 3.77 | 10.18 | 0.37x |
| vae.video | 66.50 | 15.69 | 6.79 | 2.31x |
| vae.audio | 36.90 | 5.64 | 0.28 | 20.1x |
| **generate_seconds** | **253.69** | **92.34** | **75.94** | **1.22x** |

The cold run pays JIT compilation and is not a speed number. Server path,
8 steps at 832x480: TP1 309.2 s, TP2 180.2 s.

### The server request (§36–§40)

The server's video VAE was the gap, and under `autocast(fp16)` it was the Triton
bridge: a device drain per operand (`device_ptr_ready` missing from the deployed
core, blocking bounce copies, a waiting barrier) and then one post-launch
`synchronize` per launch.

| 512x512, 8 steps, `t2va` through `/v1/videos` | TP1 | TP2 |
| --- | --- | --- |
| after §35 | 118.9 s | 130.1 s |
| after §38 (no per-operand drain) | 85.7 s | 85.5 s |
| after §39 (no per-launch wait in the shim) | **78.7 s** | **77.0 s** |

Per launch 16.65 → 2.72 ms (§38) and 2.65 → 0.515 ms (§39); tests
`tests/backends/cuda/test_device_ptr_ready.py` and `TestLaunchFollowsItsProducers`.
§40 shows the remaining request time is device work; the cheap routes
(128-bit vectorisation 1.10x, `auto_flush_ops`, `--enforce-eager`, autocast
weight-cast caching ~2%) were each measured and closed.

### The VAE decode probes (§41–§47)

Autocast VAE decode, fp32 weights under `autocast(fp16)`, latent
`(1, 24, 37, 32, 32)`. §41–§43 use the server-side decode probes
(`probe_vae_checksum.py`, `probe_decode_cprofile.py`):

- §41 cuBLASLt linear made amp-aware (scale type `CUDA_R_32F`): the GEMMs halve,
  the decode does not move (17.51 → 17.57 s);
- §42 memoised per-op shim bookkeeping (`get_install_context`, owner lookup):
  17.51 → 12.41 s;
- §43 a backend cache hit no longer re-walks the project tree: 11.50 → 9.93 s.

From §44 on, the reference decode runs under both runtimes
(`probe_decode_base_repeat.py`, repeated minimum):

| step | shim | torch | ratio |
| --- | --- | --- | --- |
| §44 bridge barrier names its operands (`TestTheLaunchBarrierNamesItsOperands`) | 10.14 s | 6.67 s | 1.52x |
| §45 SDPA reaches the fused backend once `JITTOR_FLASH_ATTN_JITTOR_SRC` is set | 8.30 s | 6.67 s | 1.248x |
| §47 bias cast to the compute dtype (15.97 s), then `fuse_op_limit=16` (`1345192d`) | 8.86 s | 6.64 s | 1.33x |

- **§45 numerics.** The fused path differs from torch by rms 1.543e-03 against the
  composite's 1.567e-03; the switch is numerically neutral.
- **§46 the elision baseline was invalid.** `return value` let jittor prune `q`/`k`.
  With a `consume` arm the 1.61 s gap is 1.09 s of non-attention device work plus
  0.53 s of attention (kernel 0.29, wrapper 0.23); a size sweep shows the gap
  scales with tensor size, i.e. it is device-side.
- **§47 the bias fix cost 2x because the fuser recomputed without a bound.** Under
  uniform half precision `count_fuse` built kernels of up to 361 operators
  (5.5x the operator-executions). `fuse_op_limit` bounds half-precision groups
  only: limit 0 → 15.97 s, 16 → 8.87 s, 8 → 8.62 s; fp32 is untouched (25.94 s
  against torch's 28.69 s). The bound is below the decode's own repeat noise
  (3.0e-04); `tests/codegen/test_fuse_op_limit.py`, `tests/core/test_fuser.py`.

## The deployment's noise (§48–§49)

- **§48 symptom and traps.** The ComfyUI/vLLM-Omni deployment returned uniform
  random `uint8` frames in ~70% of requests. Any probe that read the tensor forced
  evaluation and repaired the run, so four "fixes" were scored clean with a tracer
  attached; the checker also passed black, empty and degraded clips. The working
  bandage was `H3_PREP_MODE=varsync` (`VarHolder::sync(true, false)` on the
  quantiser's input). The paired three-arm test made the thread the variable:
  syncing on the producing thread 0/6 failures, on the consuming thread 3/6,
  no sync 4/6. On the way two crash sites under two Python threads were fixed:
  the compile failure handlers no longer dereference dead `Op*` (`841cca38`) and
  the batch hold covers the ops it is about to compile (`a5b6a29c`).
  `tests/core/test_nonsink_holder_evaluation.py` pins the invariant the
  investigation pointed at (`sync_all` sweeps only sink holders, and the decode's
  held Var was read back unfinished); it does not reproduce the failure.
- **§49 root cause.** `compute_stream()` returned `cudaStreamPerThread`, a
  different stream per calling thread, while the graph and its buffers are
  process-global; a second thread's kernels could run while the first thread's
  were still writing their inputs. `backend_compute_stream_acquire`/`_release`
  (`src/runtime/backend_streams.cc`) now hand the last issuer's event to the next
  thread at the ends of `run_exec_plan` (`a60d12f9`); a single-threaded process
  pays nothing. Cross-thread decode: 12/12 all-NaN → 0/12 (worst rel 0.0023,
  same-thread control 0.00166). Deployment with the workaround removed: 4/4 clips
  inside the clean band. An audit of every `thread_local` in `src/` and
  `backends/` found no second instance.

## Gates and reproducing

Gate failures seen during this work were checked against a pristine worktree at
the same commit: `tests/structure/backends/comm/test_comm_resource_layout.py::test_legacy_runtime_resource_trees_are_absent`
failed only because of an untracked local `python/jittor/extern` (a fresh
checkout passes), and the `test_compat_write_entry_points.py` `unclassified` list
was the gitignored `compat/build/lib/**` tree.

```bash
cd $JITTOR_LAB_ROOT/minimax-h3 && source ./env-jittor.sh
# shim tests must be collected through the installed package
PYTHONPATH=$REPO/python "$VENV/bin/python" -m pytest --pyargs jittor.compat.tests.torch.test_multi_device
GPU=0 PORT=8100 ATTN=FLASH_ATTN ./serve-vllmomni.sh      # shipped single-GPU server
./run_tp2_probe.sh                                       # TP2 loop, retries known startup flakes
"$VENV/bin/python" probe_decode_base_repeat.py 5         # reference VAE decode, repeated minimum
"$VENV/bin/python" probe_decode_sdpa_decompose.py        # section 46
H3_FUSE_DUMP=1 "$VENV/bin/python" probe_loader_race.py threads 4 6   # section 29
```

The lab venv (`$JITTOR_LAB_ROOT/_state/h3/venv-jittor`) holds a **copy** of
`python/jittor` plus `jittor-torch`: a repository edit does nothing until it is
redeployed, and a `src/` edit also needs a core rebuild and a JIT-cache clear
(the JIT key covers neither compile flags nor codegen prefixes).

## Open items

- Reference VAE decode 1.33x behind torch: 1.09 s non-attention device work and
  0.53 s attention (§46–§47). Needs kernel work, not configuration.
- Masked SDPA calls never reach the fused backend (50 per server generation,
  §45); needs a mask-aware fused kernel.
- `FusedOp::update_ops` use-after-free under two *concurrent* Python threads is
  a separate defect from §49; `tests/core/test_executor_python_threads.py`
  reproduces it with `JT_TEST_THREAD_RACE=1`. Related ledger entries:
  KI-EXEC-005, KI-EXEC-007.
- `torch.backends.cudnn.version()` returns `None` under the shim (§45), so
  vLLM-Omni's `cudnn_version >= 90500` routing would differ from torch on
  Blackwell.
- The server aborts (`SIGABRT`) during shutdown after a successful generation
  (§45); not attributed.
- vLLM-Omni imports the top-level `flash_attn_interface`, which the shim only
  ships as `flash_attn.flash_attn_interface` (§16); the lab uses an alias.
- `Tensor.to(memory_format=...)` is accepted and ignored (§48); `lt_linear_cuda`
  and the portable linear disagree on the result dtype (§47); `check_graph=1`
  flags one orphan audio Var (§48).
- ROCm's driver caches `hipGetDevice` in the same process-global way §26 fixed
  for CUDA; left alone for lack of hardware.

## Lessons worth keeping

- Reading a lazy tensor evaluates it: an attached probe can repair the run it
  measures, and a checker that only knows past failures certifies new ones.
- An ablation arm that stops consuming an operand measures pruning too (§46);
  identical inputs collapse the graph and an unread result is pruned.
- Interleave arms in one process and quote repeated minima; never read
  `nvidia-smi` utilisation on a shared box. A correct fallback can hide a broken
  fast path indefinitely (§41).
