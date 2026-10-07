# vLLM 在 Ascend 910B3 上经 Jittor 运行：bootstrap、正确性与单请求性能

- 状态：原生 vLLM-Ascend 基线接受；Jittor + Torch 兼容层 + 外置 NPU platform 插件
  下的 Qwen3-0.6B 公开 `vllm.LLM.generate` 在受限协议内正确性与性能均接受；覆盖面扩展
  与插件入仓仍开放
- 日期：2026-08-31（至 2026-09-02）
- 基线提交：`ed6ede29`
- 验证范围：一张 Ascend 910B3，CANN 9.0.0 + ATB；原生侧 Python 3.10.20、PyTorch
  2.10.0、torch-npu 2.10.0、vLLM 0.20.2 + vllm-ascend 0.20.2rc1；Jittor 侧 vLLM 0.20.2；
  Qwen3-0.6B BF16、未量化、TP=1、单请求、`max_model_len=32`、block size 128、eager、
  关闭 async scheduling 与 prefix cache。未覆盖：多请求、长上下文、chunked prefill、
  量化、TP>1、重复 engine 生命周期
- 维护者：Torch 兼容层与 vLLM adapter 维护者
- 复查条件：vLLM、Transformers、CANN、ATB 或外置插件版本变化；serving 原语
  （paged attention、RMSNorm、RoPE、SwiGLU、split）的 ACL 路由变化

## 结论

提示词 `The capital of France is` 的 4-token greedy 输出两条路径一致：

```text
token ids: [12095, 13, 576, 6722]
text:       " Paris. The capital"
```

Jittor 路径完成了模型构造（226 个参数张量）、严格 safetensors 加载、attention 独立
对拍、完整 prefill、复用 KV cache 的手工 decode，以及公开 `vllm.LLM.generate` 的
worker/scheduler/KV 分配/采样闭环。每个请求 `fallback_count=0`、
`cpu_compile_count=0`，进程没有加载 `torch_npu` 或 `vllm_ascend`。

性能（同一提示词与 engine 配置，每进程 10 次请求、排除首个图热身请求）：

| 路径 | 暖态请求中位数 | 样本 |
| --- | ---: | --- |
| 原生 vLLM-Ascend | 0.389976 s | 两轮合并 18 个，0.3708–0.4127 s |
| Jittor，最初通用路径（block16） | 0.926 s | — |
| Jittor，基线提交 | 0.363298 s | 三轮合并 27 个，0.3468–0.3760 s |

即在该受限协议下 Jittor 快约 6.8%。原生侧为避开 custom transformer op 在进程退出时的
double-free，使用 `VLLM_BATCH_INVARIANT=1` 且不启用该 custom-op library。

## 让它跑通、跑快的改动

兼容层（主仓）：

- `torch.library.infer_schema`（四组 schema 与 PyTorch 2.10.0 逐字符一致）、
  `Library.define/impl/_register_fake`、`torch.ops` 的 `.default` overload 不再是空
  stub；`torch.Tag`、`torch.types.*`、FakeTensor eager context 等导入边界补齐；
- 单物理流下一致的逻辑 `set_stream/current_stream/default_stream` 状态（不声称多流
  并发）；shim 部署同时发布 `flash-attn` dist-info；
- `torch.add(..., out=view)` 与 `torch.index_select(..., out=view)` 返回原 `out` 并写回
  父张量（vLLM 的 input ID staging 依赖它）。

Jittor serving 原语（`python/jittor/nn/paged_attention.py`、`serving_ops.py` 与
`backends/acl/kernels/`），均只在 ACL、no-grad、受支持 dtype/形状下接管，其它情况走
原路径：

- paged attention / `reshape_and_cache` 用 Gather/Scatter 完成 block-table 读取与 KV
  更新，不再触发 reindex 的 CPU 回退；block size 128 的单 token decode 直接把合并 cache
  的 strided descriptor 与 block table 交给 `aclnnIncreFlashAttentionV4`，cache 更新
  用同流小块 D2D 复制；新请求 prefill 直接对当前 Q/K/V 做 causal CANN SDPA；
- RMSNorm → `aclnnRmsNorm`；NeoX full rotary → `aclnnApplyRotaryPosEmb`；
  `silu_and_mul` → `aclnnSwiGlu`；静态 split → `aclnnSplitWithSize`；RoPE 位置表查找
  → Embedding/Gather；
- 按 batch-invariant 舍入顺序把每层 residual Add+RMSNorm、Q/K 的精确 BF16
  RMSNorm+weight multiply 与两次 RoPE 各自合入一个 CodeOp。decode profile 的设备
  算子合计从约 118 ms 降到约 20 ms。

外置插件 `vllm-jittor-npu`（未进仓库）：仅在检测到 `torch.__jittor_version__` 后激活、
拒绝已加载的 `torch_npu`，经 `vllm.platform_plugins` 注册为 OOT 平台，禁用
TorchInductor 与 CUDA graph；CUSTOM attention 用上述公开 paged attention；worker 复用
V1 `GPUModelRunner`，显式同步 `CpuGpuBuffer`/`InputBatch` 的 NumPy 与张量镜像（Jittor
的 `.numpy()` 不共享内存），冲突时 fail-closed；只支持未量化 TP=1，其它 fail-closed。

被撤回的实验（数值都对，但更慢或不安全）：让 CANN 直接消费 block16 strided cache
（约 6.95 s/token）、用 host block ID 改走 SliceV2（7.996 s/请求）、静态 RoPE cache
（中位数反而变慢）、对 CANN executor 的 TensorList 做函数级 RAII（后续异步 Cast 中
use-after-free，其生命周期必须晚于流完成）。

## 复现

```bash
source "$CANN_SET_ENV"; export ASCEND_RT_VISIBLE_DEVICES=<allocated-device>
python -m jittor.compat.shim.deploy --target "$RUN_ROOT/shim-site"
export PYTHONPATH="$RUN_ROOT/shim-site:$JITTOR_SOURCE/python"
pip install -e <vllm-jittor-npu checkout>      # 外置插件，提供 platform entry point
JITTOR_TORCH_SHIM=1 python <vllm-jittor-npu>/probes/qwen3_engine.py     # Jittor 路径
python <vllm-jittor-npu>/probes/qwen3_native_engine.py   # 在原生 vLLM-Ascend 环境里
```

探针随外置插件，不在本仓库。两侧用相同提示词、engine 参数与请求次数，丢弃第一个
请求；冷缓存首请求因串行生成 ACL JIT 图要数分钟（基线复验 332 s），不计入性能。主仓侧的维护回归：
`tests/backends/acl/test_acl.py`、`test_acl_torch_compat.py`（NPU），以及 CPU 上的
serving 原语与 vLLM layer patch 用例。

## 仍开放

- 外置 NPU platform/worker 插件未版本化。主仓 `adapters/jittor_adapters/vllm` 后来
  收进了 CUDA/通用 adapter，但**不包含**昇腾 platform 与 worker（`e3c369acb` 上核对），
  因此本报告的 engine 结论目前没有仓库内可复现入口；
- 多请求吞吐、首 token 与逐 token 延迟、长上下文、prefix cache、chunked prefill、
  异常退出与重复 engine 生命周期均未测；量化与 TP>1 保持 fail-closed；
- `CpuGpuBuffer`/`InputBatch` 的整块 host 镜像同步与 metadata materialization 仍可减少。

问题总账中没有单独条目；本报告是 NPU vLLM 路径的唯一证据。
