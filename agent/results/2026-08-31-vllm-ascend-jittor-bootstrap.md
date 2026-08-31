# vLLM Ascend 与 Jittor-NPU bootstrap 验证

- Status: native vLLM-Ascend inference accepted; Jittor Qwen3 manual prefill/decode token parity accepted; engine integration and performance open
- Date: 2026-08-31
- Baseline: `0df29eed` plus the compatibility changes in this report's commit
- Owner: Jittor compatibility and external vLLM adapter maintainers
- Review when: vLLM, Transformers, CANN, ATB, or the external adapter version changes

## Scope

本轮区分两个执行路径：

1. 原生 PyTorch-NPU 的 `vLLM 0.20.2 + vllm-ascend 0.20.2rc1` 基线；
2. `vLLM 0.20.2` 在 Jittor Torch shim 和外置 `vllm-jittor-npu` platform
   plugin 下的 bootstrap、custom-op 注册和真实 ACL tensor 执行。

第二条路径已经完成 Qwen3-0.6B 模型构造、严格 safetensors 权重加载、attention
独立对拍、完整 prefill 和复用 KV cache 的手工 greedy decode，四个 token 与原生
基线一致。尚未完成 worker 和 engine 采样/调度，因此本报告不声明完整
Jittor-vLLM NPU engine 推理或性能通过。

## Native vLLM-Ascend baseline

环境为 Python 3.10.20、PyTorch 2.10.0、torch-npu 2.10.0、CANN 9.0.0、
ATB 和一张可见 Ascend 910B3。为避开当前 custom transformer op 在进程退出时的
double-free，稳定路径使用 `VLLM_BATCH_INVARIANT=1`，不启用该 custom-op library。

Qwen3-0.6B 对提示词 `The capital of France is` 的 4-token greedy 输出为：

```text
token ids: [12095, 13, 576, 6722]
text:       Paris. The capital
```

复用 engine 后，另一条提示词的 4-token generation 为 `0.364996s`，约
`10.96 token/s`。该数据只作为原生 vLLM-Ascend 参考，不是 Jittor 性能结果。

## Versioned Jittor compatibility changes

- `torch.library.infer_schema`：支持 Tensor、Optional、Sequence/List、tuple return、
  dtype/device/default value、mutation alias 和 fail-closed error contract；四组 schema
  与独立 PyTorch 2.10.0 输出逐字符一致。
- `torch.library.Library` 和 `torch.ops`：`define`、`impl`、`_register_fake`、
  `.default` overload 和实际 callable 注册不再是空 stub。
- `torch.Tag`、`torch.types.Device/Number/FileLike/Storage`、
  `torch._C._distributed_c10d.Store` identity、FakeTensor eager context 和抽象
  `CustomGraphPass` import boundary。
- 单物理 backend stream 下的一致逻辑 `set_stream/current_stream/default_stream/stream`
  状态；不声明多流并发。
- shim 部署同时发布 `flash-attn` dist-info，避免可导入 stub 与
  `importlib.metadata.packages_distributions()` 不一致。
- Torch module 注册不再读取 tensor truthiness；分布式 backend 查询、fail-closed
  LazyBatchNorm、FlashAttention rotary import surface 和空 tensor 跨 allocator 迁移补齐
  vLLM Qwen3 构造、权重加载所需边界。
- NVTX 的惰性原生导入会恢复共享 Jittor/Torch 根模块的 `torch.utils` 绑定，避免一次
  range 调用破坏已完成的 shim 安装图。
- 公有 `jittor.nn.paged_attention/reshape_and_cache` 在 ACL 下用 Gather/Scatter 完成
  block-table 读取、KV 更新、GQA 扩展和 packed slice；prefill、decode 和 KV cache
  不再触发 `reindex` CPU fallback。

## External adapter state

外置 adapter 位于 `$JITTOR_LAB_ROOT/npu-vllm/vllm-jittor-npu`，未版本化，不进入
Jittor 主仓库。当前文件 SHA-256：

| File | SHA-256 |
| --- | --- |
| `pyproject.toml` | `77872fe7d972071a55852f17103b7abb2ec3b55e04a8500eeaae7843eb9b92d2` |
| `vllm_jittor_npu/__init__.py` | `fb67de68f407a898a47c44b9c96bdbcde2955c0e219fe2bf13e0fe7b29836582` |
| `vllm_jittor_npu/bootstrap.py` | `25961da3ad6de1cceb59e4bdf6e0938f1cc183166726e351f8da992385a41459` |
| `vllm_jittor_npu/platform.py` | `e3d7d2b9abcbf45d07df5cf0d168392d2a67d56f7dba28b9056a3e3bcca8939a` |
| `vllm_jittor_npu/attention.py` | `7b1c3ba5c2aeabcca9c8e4a6cfd817f622c35a07c270ab3e6a1be4c3d0e45771` |
| `probes/qwen3_forward.py` | `ed8ca5891f4df4482ecdb87140008a64a916007384f55f060a430f7d60fed85d` |

Bootstrap 只在检测到 `torch.__jittor_version__` 后激活，拒绝已加载的
`torch_npu`，并让 Transformers 的可选 torch-npu 探测返回 false。platform 通过
`vllm.platform_plugins` entry point 注册为 `PlatformEnum.OOT`，禁用 TorchInductor
和 CUDA graph。CUSTOM attention 使用公有 Jittor paged-attention，metadata 的 host
materialization 每个 batch 只执行一次。adapter 还修复了 vLLM embedding loader 对
PyTorch slice-view 写回的依赖；当前只支持未量化 TP=1，其他 TP 形态 fail-closed。
worker 仍待实现。

## Evidence

- 本轮受影响的 library、stream/NVTX、compiler context、module hook、安装幂等、
  FSDP2 和 shim alias 回归：`76 passed`。
- shim deploy 与资源结构：`21 passed`。
- 完整 structure：`217 passed, 2 skipped, 2 failed`；两项失败均来自仓库内既有
  `.claude/worktrees`，不是本次变更。布局检查还报告既有 `TODO.md`、旧 egg-info
  和相同 worktree 污染。
- ratcheted Ruff lint：通过；新增文件的 Ruff Python 3.7 target 检查：通过。
- Python 3.9 `py_compile` 和 `git diff --check`：通过。本机无真实 Python 3.7
  解释器，本轮未运行 `nox -s py37`。
- 真 NPU probe：`vLLM 0.20.2`、`CustomOp` 和唯一
  `vllm_jittor_npu.platform.JittorNPUPlatform` 导入成功；
  `acl_compiler=true`、`use_cuda=1`。通过新 `torch.library` 注册的乘法在 ACL
  执行，结果 tensor 在同步后报告 `location=device`，值为 `[6.0, -8.0]`；
  `torch_npu` 与 `vllm_ascend` 均未加载。
- Qwen3 registry probe：lazy entry 成功加载
  `vllm.model_executor.models.qwen3.Qwen3ForCausalLM`；当前 platform、ACL
  compiler 与零 PyTorch-NPU/vllm-ascend 条件保持不变。为到达该层，本轮补齐
  `torch.cuda.nvtx`、fail-closed CUDA pluggable allocator、`torch.func`、
  `torch._ops`、symbolic scalar aliases、native module `reset_parameters` 和全局
  module-registration hook。相关 Torch 兼容回归 `42 passed`，native reset 回归
  `3 passed`。
- Qwen3-0.6B model construction：在单 rank MPI process/model parallel group、
  `set_current_vllm_config` 和 CUSTOM attention backend 下构造成功，共 226 个参数
  tensor、596,049,920 个参数；进程未加载 `torch_npu` 或 `vllm_ascend`。
- 严格 safetensors load：没有 missing-weight error，热缓存加载为 `5.58s`。首层 QKV
  与末层 MLP 样本均在 `device`，为有限且非零值；ACL compiler 保持启用，
  `torch_npu` 与 `vllm_ascend` 仍未加载。该证据只验收到权重加载，不代表模型输出
  正确。
- 空 tensor 真实 NPU 回归：`empty((0, 3))` 在 `device` 驻留后，两次 `.numpy()` 和
  重复 `.sync()` 通过；零字节迁移不调用 vendor memcpy，也未复现 double-free。
- 公有 paged-attention：CPU 独立 reference 与 negative-slot 契约 `6 passed`；真实
  NPU 上同一 cache 的 float32 prefill、追加 token 和 decode 均匹配 NumPy，GQA、
  cache 更新和输出全在 `device`，fallback 为 0。外置 backend 的 float32
  prefill/decode 最大误差分别为 `6.25e-9/5.11e-9`；BF16 cache 最大误差
  `1.64e-4`，均为零 fallback。
- Qwen3-0.6B 完整 prefill：提示词 `The capital of France is` 编码为
  `[785, 6722, 315, 9625, 374]`，首个 greedy token 为 `12095`（` Paris`），与原生
  vLLM-Ascend 基线的第一个 token 一致。28 层 KV cache、hidden states 和 logits
  均在 `device`，fallback 为 0；embedding/hidden/logits 非零且有限，进程未加载
  `torch_npu` 或 `vllm_ascend`。热缓存严格加载为 `3.95s`，完整 prefill+logits 为
  `3.03s`；该时延尚无等价原生 prefill 协议，不作为性能验收。
- 同一进程、模型和 28 层 KV cache 上继续手工 greedy decode，四个 token 为
  `[12095, 13, 576, 6722]`，文本为 ` Paris. The capital`，逐 token 匹配原生基线。
  四步 hidden/logits/cache 均在 `device`，fallback 为 0，且未加载 PyTorch-NPU。
  首次 decode 含单 token shape 冷编译为 `95.08s`；随后两步热态为
  `0.259s/0.256s`，约 `3.9 token/s`，明显慢于原生约 `10.96 token/s`。因此
  correctness 接受，performance 不接受。

## Open work

- 实现、测试外置 Jittor NPU worker 和 model runner，将已验证的 prefill/decode
  backend 接入 engine 调度和 KV cache 分配。
- 优化单 token decode 的 KV gather、GQA 和 attention launch，达到不慢于原生的
  等价暖态生成协议，再完成吞吐、首 token、decode latency 和显存对比。
- 在当前 vLLM 版本重新验证历史 CUDA adapter 基线；旧 adapter 当时未版本化，
  不能作为当前可复现产物。
