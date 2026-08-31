# vLLM Ascend 与 Jittor-NPU bootstrap 验证

- Status: native vLLM-Ascend inference accepted; Jittor platform bootstrap, Qwen3 construction and strict weight load accepted; Jittor-vLLM model inference open
- Date: 2026-08-31
- Baseline: `ccafc52e` plus the compatibility changes in this report's commit
- Owner: Jittor compatibility and external vLLM adapter maintainers
- Review when: vLLM, Transformers, CANN, ATB, or the external adapter version changes

## Scope

本轮区分两个执行路径：

1. 原生 PyTorch-NPU 的 `vLLM 0.20.2 + vllm-ascend 0.20.2rc1` 基线；
2. `vLLM 0.20.2` 在 Jittor Torch shim 和外置 `vllm-jittor-npu` platform
   plugin 下的 bootstrap、custom-op 注册和真实 ACL tensor 执行。

第二条路径已经完成 Qwen3-0.6B 模型构造和严格 safetensors 权重加载，但尚未完成
attention 独立对拍、完整 forward、KV cache、采样和解码。因此本报告不声明
Jittor-vLLM NPU 推理、token 对齐或性能通过。

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

## External adapter state

外置 adapter 位于 `$JITTOR_LAB_ROOT/npu-vllm/vllm-jittor-npu`，未版本化，不进入
Jittor 主仓库。当前文件 SHA-256：

| File | SHA-256 |
| --- | --- |
| `pyproject.toml` | `77872fe7d972071a55852f17103b7abb2ec3b55e04a8500eeaae7843eb9b92d2` |
| `vllm_jittor_npu/__init__.py` | `fb67de68f407a898a47c44b9c96bdbcde2955c0e219fe2bf13e0fe7b29836582` |
| `vllm_jittor_npu/bootstrap.py` | `6faa4d55f4cdb47666f8c29ec4ba41818987fa519a935cd262fa86d95df8bdfe` |
| `vllm_jittor_npu/platform.py` | `2db812921edde5727d8d835228681559b255e493e3b6c269a0f5eea08ebe2405` |
| `vllm_jittor_npu/attention.py` | `e88de4df9c9c66dd1257c3e4cc64d8d17aded71d072f99cc58d5442540d0ed39` |

Bootstrap 只在检测到 `torch.__jittor_version__` 后激活，拒绝已加载的
`torch_npu`，并让 Transformers 的可选 torch-npu 探测返回 false。platform 通过
`vllm.platform_plugins` entry point 注册为 `PlatformEnum.OOT`，禁用 TorchInductor
和 CUDA graph。外置 adapter 已有标准 KV shape 的 CUSTOM attention 结构实现；它的
数值正确性和性能尚未独立验证，worker 仍待实现。

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

## Open work

- 用独立 reference 对拍外置 attention 的 prefill、decode、GQA 和 KV cache 更新，
  并消除 metadata host synchronization 路径。
- 实现、测试外置 Jittor NPU worker、model runner 和 paged KV cache，完成
  Qwen3-0.6B 完整 forward 与 greedy decode。
- 完成 greedy token 对齐、零 CPU/PyTorch fallback 审计、暖态性能和显存对比。
- 在当前 vLLM 版本重新验证历史 CUDA adapter 基线；旧 adapter 当时未版本化，
  不能作为当前可复现产物。
