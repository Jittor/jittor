# verl、vLLM 与 TRELLIS 当前基线复验

- Status: CUDA functionality accepted; performance not accepted
- Last reviewed: 2026-08-23
- Baseline: `7e7c23cf`
- Owner: Torch compatibility and downstream integration maintainers
- Review when: Jittor Torch version identity, FSDP2, verl algorithms,
  vLLM adapter/engine, TRELLIS adapter, FlashAttention, FlexGEMM, or CUDA
  inference performance changes

## 结论

当前 Jittor 基线没有暴露新的 FSDP2/DTensor 核心注册缺陷。verl 的 import、protocol、
FSDP2、PPO 核心算法和真实 CPU/CUDA PyTorch 对拍门禁全部通过。vLLM V1 可在外置
adapter 下实际加载 Qwen3-0.6B、建立 KV cache 并完成真实 CUDA greedy decode；四个
token 与真实 PyTorch/Transformers 完全一致。TRELLIS.2 4B 也在当前 Jittor、外置
adapter 和四个真实 CUDA 扩展上完成 aligned 端到端 pipeline。

总 todo 仍不能勾选：

- vLLM 热态四 token decode 为 `0.6410s`，真实 PyTorch eager 为 `0.1604s`，约
  `4.00x`；
- TRELLIS Jittor 三次 pipeline 中位数 `8.1712s`，同轮 PyTorch 中位数
  `6.8007s`，约 `1.201x`；
- verl 只验证了核心算法与 FSDP2/protocol 门禁，没有在本轮重新执行完整 PPO actor、
  rollout、weight transfer 和 optimizer 训练闭环；
- vLLM/TRELLIS 是 inference runtime，没有本轮反向对拍结论。

## 身份与环境

- Jittor: `7e7c23cf`; Python 3.11.15; JIT compiler serial
- GPU: NVIDIA GeForce RTX 4090, compute capability 8.9
- CUDA: 12.2.140 for Jittor/TRELLIS; real references used their recorded binary
  CUDA runtimes
- verl: `3d66a3d7ca1cf783df949816ec6862d5a7af9406`
- vLLM checkout: `51a99565c398c8320de8131e07731c75c52eb87c`
- TRELLIS.2: `75fbf0183001ed9876c8dbb35de6b68552ee08bd`
- jittor-trellis: `d10ed75e8183f456dc2107d7cf6ca5cb26f7686a`
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
| `vllm_jittor_ops/bootstrap.py` | `79f680798fd8debb6e5ed86fb44696b91faef01ea9af9ca0a277ff448b8da76d` |
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

The isolated real PyTorch/Transformers run produced the same IDs and text.

| Runtime | Load/init | 4-token generate |
| --- | ---: | ---: |
| Jittor-vLLM cold | `630.58s` | `325.23s` |
| Jittor-vLLM warm cache | `15.77s` | `0.6410s` |
| PyTorch/Transformers warm | not compared | `0.1604s` |

Cold values include serial JIT and are not performance results. The hot generate
ratio is about `4.00x`, so performance is explicitly rejected.

## TRELLIS.2

The current run compiled and loaded CuMesh, FlexGEMM, nvdiffrast, o-voxel,
official FlashAttention and its packed forward adapter from external state. The
runtime selected `flash_attn`, `flex_gemm`, sm_89 CUDA, no managed allocator,
no low-vram mode, and tensorcore level 2. The shared aligned tape contained 3,540
coordinates and three random tensors.

One cold warmup took `2377.24s` because it built all external and JIT kernels.
This value is excluded from steady-state timing.

| Runtime | Pipeline runs | Mean |
| --- | --- | ---: |
| Jittor | `8.2401 / 8.1327 / 8.1712s` | `8.1813s` |
| PyTorch | `6.7481 / 6.8007 / 6.8336s` | `6.7941s` |

The mean ratio is `1.204x`; the median ratio is `1.201x`. Performance is not
accepted.

Both runs emitted valid PLY meshes. The comparison used 20,000 sampled vertices
per direction:

- vertex count difference: `+3,687` (`0.247969%`);
- face count difference: `+20,154` (`0.626227%`);
- bounding-box extent L2: `1.7997e-4` (`1.2781e-4` of bbox diagonal);
- centroid L2: `1.2446e-3` (`8.8386e-4` of bbox diagonal);
- nearest Jittor-to-Torch mean: `0.0058584` (`0.0041605` of bbox diagonal);
- nearest p95: `0.0104673`.

Artifact hashes:

| Artifact | SHA-256 |
| --- | --- |
| Jittor result JSON | `0100e44095fe0876f1d4fabd53c766c2c52bbe03d0ea19c511115d44c14e8ae5` |
| PyTorch result JSON | `d729c72d936c1bb72563d5cb99d022897322431ca60e46e020be3579ef4ae188` |
| Mesh comparison | `d6677c59c0e8aa719637db5c2b166db6e4df143be6089cfaa9106b7fea14d209` |

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

## Remaining work

- Bring TRELLIS steady pipeline time back to at most the real PyTorch baseline;
  profile the current `3.65 it/s` shape and `6.5 it/s` texture stages first.
- Close vLLM eager decode overhead; the external plan identifies graph capture
  and Python lazy-graph construction as the main remaining tier.
- Run a current full verl PPO training/rollout/weight-transfer step after the
  vLLM adapter is packaged and versioned.
- Turn the unversioned vLLM adapter into a maintained external distribution with
  import, MoE oracle and Qwen engine gates.
- Do not claim vLLM/TRELLIS backward parity or NPU/ROCm support from these CUDA
  inference results.
