# verl、vLLM 与 TRELLIS 当前基线复验

- Status: vLLM CUDA functionality/performance accepted; TRELLIS performance not accepted
- Last reviewed: 2026-08-26
- Baseline: `566d8087`
- Owner: Torch compatibility and downstream integration maintainers
- Review when: Jittor Torch version identity, FSDP2, verl algorithms,
  vLLM adapter/engine, TRELLIS adapter, FlashAttention, FlexGEMM, or CUDA
  inference performance changes

## 结论

当前 Jittor 基线没有暴露新的 FSDP2/DTensor 核心注册缺陷。verl 的 import、protocol、
FSDP2、PPO 核心算法和真实 CPU/CUDA PyTorch 对拍门禁全部通过。vLLM V1 可在外置
adapter 下实际加载 Qwen3-0.6B、建立 KV cache 并完成真实 CUDA greedy decode；四个
token 与真实 PyTorch/Transformers 完全一致。`f6980c42` 又将 Qwen3-0.6B 的 4-token
热态从历史 `0.6410s` 降至 3 进程中位数 `0.10924s`；同机真实
PyTorch/Transformers 为 `0.13745s`，Jittor-vLLM 快约 `20.5%`，性能门禁已接受。
TRELLIS.2 4B 也在当前 Jittor、外置 adapter 和四个真实 CUDA 扩展上完成 aligned
端到端 pipeline；本轮将默认热态从 `8.2843s` 降到三进程中位数
`7.5149s`，但仍比真实 PyTorch 慢约 `9.3%`，性能未接受。

总 todo 仍不能勾选：

- TRELLIS Jittor 三个独立进程的 pipeline 中位数为
  `7.6291 / 7.5149 / 7.4246s`，median-of-medians `7.5149s`；真实 PyTorch
  中位数 `6.8778s`，约 `1.093x`；
- vLLM/TRELLIS 是 inference runtime，没有本轮反向对拍结论。

## 身份与环境

- Jittor: `566d8087`; Python 3.11.15; JIT operator compiler serial
- GPU: NVIDIA GeForce RTX 4090, compute capability 8.9
- CUDA: 12.2.140 for Jittor/TRELLIS; real references used their recorded binary
  CUDA runtimes
- verl: `3d66a3d7ca1cf783df949816ec6862d5a7af9406`
- vLLM checkout: `51a99565c398c8320de8131e07731c75c52eb87c`
- TRELLIS.2: `75fbf0183001ed9876c8dbb35de6b68552ee08bd`
- jittor-trellis: `f2c23acdf2402abcf04222a4866fc87451efe959`
- Real PyTorch: 2.12.1; Transformers references: 5.12.1 for Qwen3 and 4.56.2
  for TRELLIS

The vLLM checkout contains a local untracked `vllm-0.11.0.dist-info`, so
`vllm.__version__` reports 0.11.0. The exact checkout commit above, not that local
metadata string, is the authoritative source identity.

All caches, generated extensions, model links, raw JSON and meshes are outside
the repository under `$JITTOR_LAB_ROOT`. CPU/CUDA tests and benchmarks do not
share a compilation cache.

## verl

### Root cause and adapter boundary

`torch.distributed.tensor._dtensor_spec.DTensorSpec` imports correctly from the
Jittor shim. The import failure came from verl branching on `torch.__version__`:
Jittor 2.0 deliberately preserves its backend version there and exposes the
emulated API level as `torch.__torch_version__`. verl then evaluated an
unimported `DTensorSpec` annotation.

The downstream adapter now parses
`getattr(torch, "__torch_version__", torch.__version__)` in
`verl/utils/fsdp_utils.py`. Real PyTorch behavior is unchanged; Jittor's version
identity contract is not weakened.

### Complete maintained gate

`verl_jittor/run_all.sh` passed all 14 requested stages:

```text
py_compile import_scan ppo_config_smoke protocol_smoke
protocol_extended_smoke protocol_v2_nested_smoke
torch_functional_extended_smoke core_algos_smoke
core_algos_extended_smoke core_algos_parity_cpu fsdp2_smoke
jittor_torch_compat_ops core_algos_parity_cuda cuda_sanity
```

The harness was updated to the canonical
`jittor/compat/torch/__init__.py` and root
`tests/compat/torch/test_torch_compat_ops.py` entries; the selected three tests
passed. Import scan accepted only explicit missing optional environments for
Megatron, `transfer_queue`, and TorchTitan.

The real-PyTorch algorithm matrix covered CISPO, geometric-mean, GPG, GRPO,
GSPO, SAPO and vanilla policy paths. Maximum errors were:

| Device | Forward | Gradient |
| --- | ---: | ---: |
| CPU | `2.384e-7` | `2.794e-9` |
| CUDA | `2.384e-7` | `1.863e-9` |

The summary artifact SHA-256 is
`37ee701c4ead019a49f599d3f3578f11de1f9da982a2b40c69f13040f3e10f4b`.

## vLLM

### Current import compatibility

The external `vllm_jittor_ops` adapter needed current-import maintenance:

- expose the Torch API version to vLLM/Transformers 5 for this adapter process,
  while retaining `__jittor_version__`;
- skip unavailable lazy AutoProcessor classes without loading real torchvision;
- publish distributed `Store` types through `torch._C._distributed_c10d`;
- provide/re-export eager-only fake-tensor context types required at import;
- inspect already-bound Transformers module attributes without triggering lazy
  import of every model family.

The resulting adapter files are unversioned external state. Their hashes are:

| File | SHA-256 |
| --- | --- |
| `vllm_jittor_ops/bootstrap.py` | `c6dd413bba75d8ea923ab703ad0b61484017845edd79a8857da99869bdbb7f3b` |
| `vllm_jittor_ops/ops.py` | `60029f8f2d73cdc5005d1fd9ebd8036860016138914f7144521b2c317c445612` |
| `vllm_jittor_ops/tvfunc.py` | `366952a34208e8c98e834372daaa8aed35d86feaef0b6f93b17620b08576744a` |

After the fixes, vLLM, `model_executor.custom_op`, distributed Store and fake
tensor imports passed with Jittor owning `torch`. A real CUDA tensor produced
`[3.0, 6.0]`; missing native `_qutlass_C`/`_moe_C` warnings are expected because
the external adapter replaces those compiled PyTorch extensions.

### MoE oracle and Qwen3 V1 engine

Five routed-experts cases passed on real CUDA against an independent float64
NumPy implementation. Relative errors ranged from `1.875e-7` to `2.217e-7`,
including softmax/sigmoid routing, renormalization, top-k 1/4 and a gated shared
expert.

Qwen3-0.6B V1 loaded a 1.40 GiB safetensors checkpoint, selected the external
FlashAttention path, allocated 6.63 GiB KV cache (62,096 tokens), and decoded:

```text
prompt:    The capital of France is
token ids: [12095, 13, 576, 6722]
text:       Paris. The capital
```

The isolated real PyTorch/Transformers run produced the same IDs and text. A
16-token extension also matched exactly:

```text
[12095, 13, 576, 6722, 315, 15344, 374, 21718,
 13, 576, 6722, 315, 17689, 374, 24081, 13]
```

| Runtime | Three process medians for 4-token generate | Median |
| --- | --- | ---: |
| Jittor-vLLM warm | `0.10924 / 0.10888 / 0.10995s` | `0.10924s` |
| PyTorch/Transformers warm | `0.14824 / 0.13586 / 0.13745s` | `0.13745s` |

Each process performed 3 warmups followed by 21 measured generations, with prefix
cache disabled. The ratio is `0.7947x`, so Jittor-vLLM is about `20.5%` faster and
the Qwen3-0.6B gate is accepted. Cold values include serial JIT and remain excluded.

The improvement comes from generic, inference-only CUDA capabilities in Jittor:

- fp16/bf16 RMSNorm and fused residual-add RMSNorm;
- GQA RoPE with different query/key head counts;
- SwiGLU `silu_and_mul`;
- in-place V1 paged KV scatter;
- a guarded one-token causal paged-attention path with online softmax.

All kernels return `None` outside their CUDA/no-grad/dtype/shape contract, so the
adapter retains its existing fallback. Decode attention additionally requires no
sliding window, ALiBi, logits soft-cap, or sinks. Profiler GPU-op time fell from
`85.2ms` to `31.5ms` per 4-token generation and reported memory access from
`246GB` to `86.4GB`; the default graph break interval moved from 1 to 16 layers,
with `VJ_SYNC_EVERY=1` retained as a conservative override.

The run also found a late-activation compatibility bug: importing Jittor before
the deployed Torch placeholder left `Parameter.data` as a NumPy view, breaking
vLLM's default safetensors loader at `param.data.copy_`. Torch mode now also checks
the completed install state, while plain Jittor continues to expose its shared
NumPy data view. The default 1.40 GiB checkpoint loader, tiny-model ZMQ weight
apply (`delta=0.125`) and restore/logprob smoke all pass.

## TRELLIS.2

The current run loaded CuMesh, FlexGEMM, nvdiffrast, o-voxel, official
FlashAttention and its packed forward adapter from isolated external state. The
runtime selected `flash_attn`, `flex_gemm`, sm_89 CUDA, no managed allocator,
no low-vram mode, and tensorcore level 2. The shared aligned tape contained 3,540
coordinates and three random tensors. Cold builds and each process's first
warmup were excluded.

The same-machine starting point was Jittor `8.2843s` versus PyTorch `6.8778s`.
The retained optimizations are:

- one warp per row for BF16 multi-head RMSNorm at head dimensions up to 256;
- one packed self-attention kernel that preserves the BF16 rounding point while
  combining Q/K RMSNorm, pairwise-complex RoPE and V passthrough;
- fused non-affine LayerNorm plus scale/shift for exact sparse modulated blocks;
- lazy-CUDA device fallback and stable `Var.id` cache identity in the maintained
  `jittor-trellis` adapter.

Rejected A/B paths were removed: mixed-length Q/KV RMS, residual-plus-LayerNorm,
default cross-KV cache and C2S topology cache all regressed or remained within
noise. Historical cublasLt/GELU/RoPE experiments were not repeated after their
recorded negative end-to-end results.

| Runtime | Independent process medians | Median of medians |
| --- | --- | ---: |
| Jittor | `7.6291 / 7.5149 / 7.4246s` | `7.5149s` |
| PyTorch | `6.7972 / 6.8778 / 6.8979s` measured runs | `6.8778s` |

Each Jittor process performed one warmup and three measured runs. The final
ratio is `1.0926x`; Jittor improved `9.3%` from the same-round starting point,
but performance remains unaccepted because it is not yet at most the reference.

Both runs emitted valid PLY meshes. The comparison used 20,000 sampled vertices
per direction:

- vertex count difference: `+3,368` (`0.226515%`);
- face count difference: `+14,700` (`0.456760%`);
- bounding-box extent L2: `9.9765e-5` (`7.0852e-5` of bbox diagonal);
- centroid L2: `6.3595e-4` (`4.5164e-4` of bbox diagonal);
- nearest Jittor-to-Torch mean: `0.0058707` (`0.0041693` of bbox diagonal);
- nearest p95: `0.0104870`.

Artifact hashes:

| Artifact | SHA-256 |
| --- | --- |
| Jittor process 1 JSON | `2f4e3360f4898a7272ceb7377b7ebfde76722a4fb06d12f7f80cbce812db3147` |
| Jittor process 2 JSON | `984bac6cc21f6dd2033bb55c9958c2591ab64dc4ed65578b5fb8f5501e34d712` |
| Jittor process 3 JSON | `845bfc9ee26ca623670aa3b096b26752053057e4731040f033c5c5802aa86a88` |
| PyTorch result JSON | `a9e9a5d8d8f575d3e0c36e21e9d98957ed5011a4994dc83ec7d6f02adecd6154` |
| Mesh comparison | `847502a0ed0da81c3851c3979fca872a1794d4e272debf70dc19741059d78e91` |

## Reproduction

Raw commands and reusable scripts remain in:

```text
$JITTOR_LAB_ROOT/verl_jittor/
$JITTOR_LAB_ROOT/_state/vllm-current/
$JITTOR_LAB_ROOT/_state/trellis-current/
```

The vLLM scripts lock Jittor `torch`/`torchvision` before adding the
Transformers 5 dependency site. The TRELLIS commands use the current source
checkout, adapter `src/`, local checkpoints, offline model assets, isolated
extension/JIT caches, and one visible GPU.

The machine-readable vLLM result is
`$JITTOR_LAB_ROOT/_state/vllm-current/20260826-performance-summary.json`
(SHA-256
`0d2f8c4e4fae614bd1d029d670dbee532c29d06d2c4df8ca1cffdd91ed1d4731`).
The final profiler SHA-256 is
`f94eee3762be84ef4302ae49e10ea45666c9e8ec0d930fb65b0d21fd3fd315da`.
The Jittor and PyTorch benchmark script SHA-256 values are
`5b17cdced5e0d3b356bdcca63cb8f7b73caff3c53063973219772ca1097660f5` and
`45055408854618eaf395279d8758771b2541a72ceb91417f2ad5e51b44d37178`.

The updated TRELLIS benchmark and launcher hashes are
`0dbbdf0f206ee80ff946f7d804f294521a93411ead16c58efcd03bcc2de3a77c` and
`2ac0260f502961d26b36aaba764741b2af303df9f1913c16bf2839d572366d76`.

Regression scope: TRELLIS CUDA capability `33 passed`, retained capability subset
`4 passed`, `jittor-trellis` adapter `14 passed`, structure `218 passed`, Torch
optimizer/data alias `22 passed`, native array semantics `17 passed`, and real
vLLM weight apply/restore smoke passed. The unrelated legacy
`test_memcopy_overlap` 10ms timing assertion remained noisy (`33-75ms` delta)
when run alone and is not counted as a functional regression result.

## Remaining work

- Bring TRELLIS steady pipeline time from current `1.093x` to at most the real
  PyTorch baseline; the remaining gap is concentrated in BF16 linear
  epilogues/fused elementwise around the `4.2 it/s` shape and `7.4 it/s`
  texture stages, not FlashAttention or the GEMM kernels themselves.
- Re-run the accepted vLLM performance protocol for larger dense, MoE and TP
  configurations before generalizing the 0.6B single-GPU conclusion.
- Turn the unversioned vLLM adapter into a maintained external distribution with
  import, MoE oracle and Qwen engine gates.
- Do not claim vLLM/TRELLIS backward parity or NPU/ROCm support from these CUDA
  inference results.
