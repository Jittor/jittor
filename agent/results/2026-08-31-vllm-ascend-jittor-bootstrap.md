# vLLM Ascend 与 Jittor-NPU bootstrap 验证

- Status: native vLLM-Ascend inference accepted; Jittor Qwen3 manual and public engine token parity accepted; performance open
- Date: 2026-09-01
- Baseline: `c7834d16`
- Owner: Jittor compatibility and external vLLM adapter maintainers
- Review when: vLLM, Transformers, CANN, ATB, or the external adapter version changes

## Scope

本轮区分两个执行路径：

1. 原生 PyTorch-NPU 的 `vLLM 0.20.2 + vllm-ascend 0.20.2rc1` 基线；
2. `vLLM 0.20.2` 在 Jittor Torch shim 和外置 `vllm-jittor-npu` platform
   plugin 下的 bootstrap、custom-op 注册和真实 ACL tensor 执行。

第二条路径已经完成 Qwen3-0.6B 模型构造、严格 safetensors 权重加载、attention
独立对拍、完整 prefill、复用 KV cache 的手工 greedy decode，以及公开
`vllm.LLM.generate` 的 worker、scheduler、KV 分配和采样闭环。两条 Jittor 路径的
四个 token 均与原生基线一致。本报告接受当前单请求短上下文 engine correctness，
不接受性能，也不外推到多请求、长上下文、量化或 TP>1。

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
  不再触发 `reindex` CPU fallback。BF16 单 token decode 在 gather 后直接调用
  `aclnnIncreFlashAttentionV4`，不再展开 GQA 或执行通用 matmul-softmax-matmul。
  对 CANN 原生 page size 128、单请求单 token decode，runner 从合并 cache storage
  构造 K/V strided descriptor，并将 block table 和实际长度直接交给
  `aclnnIncreFlashAttentionV4`；最多 16 个已知有效 slot 的 cache 更新用同 stream
  D2D 小块复制，不再转置和 Scatter 整块 cache。其他组合保留通用路径。
- 公有 serving `rms_norm` 在调用时解析当前 backend hook；ACL/no-grad 的
  float32 activation + BF16/FP16 weight 会缓存一份 float32 weight 并执行
  `aclnnRmsNorm`，保持 float32 输出语义。
- 公有 serving `rotary_embedding` 在 ACL/no-grad、NeoX、full rotary、64 对齐
  head size 和 Q/K/cache 同 dtype 条件下，将 packed Q/K 转为 BNSD 并调用
  `aclnnApplyRotaryPosEmb`；其他 RoPE 形态保留原路径。
- 公有 serving `silu_and_mul` 在 ACL/no-grad 和受支持浮点 dtype 下调用
  `aclnnSwiGlu`；未命中 ACL 时仍继续尝试既有 CUDA fused path。单 token decode
  的 packed RoPE 直接 reshape 为 BNSD 并恢复原形状，不再执行四个恒等 transpose；
  多 token 路径保持原有重排。
- `torch.add(..., out=view)` 和 `torch.index_select(..., out=view)` 返回原 `out`
  对象并写回切片的父 tensor；vLLM 的 optimistic sequence length 和 input ID
  staging 不再保留旧值。

## External adapter state

外置 adapter 位于 `$JITTOR_LAB_ROOT/npu-vllm/vllm-jittor-npu`，未版本化，不进入
Jittor 主仓库。当前文件 SHA-256：

| File | SHA-256 |
| --- | --- |
| `pyproject.toml` | `77872fe7d972071a55852f17103b7abb2ec3b55e04a8500eeaae7843eb9b92d2` |
| `vllm_jittor_npu/__init__.py` | `fb67de68f407a898a47c44b9c96bdbcde2955c0e219fe2bf13e0fe7b29836582` |
| `vllm_jittor_npu/bootstrap.py` | `9e8b9f9c02a9f5d9f56bba3d8affedaca42ad27ca29201d85d6c9f8267fa7819` |
| `vllm_jittor_npu/platform.py` | `87e6e555483f7df944038585f02ae56475fe218d8bb8a9806e09552cca95d663` |
| `vllm_jittor_npu/attention.py` | `41fc53d2b5245c5bac9bb7bc6c6d84aa610ea79952910e2efdf9ad609c76e8de` |
| `vllm_jittor_npu/worker.py` | `b383042d63711f538ed14ae748e2d589c28d9b0c37e2278409840b7b34d46239` |
| `probes/qwen3_forward.py` | `fecdec74e781d1a21100c634a098a26670ae32d99ccc7a7d520f86691b50f2f5` |
| `probes/qwen3_engine.py` | `8edec2385347bcfdf5a6ad6dc89491bbe482f4d8370b47b9ee26af322504c566` |
| `_state/npu-vllm/current-a84614f4/logs/qwen3-engine-swiglu-rope-10.log` | `308aac72b243342478d63e0246402ad0713ca3b344522b622eae4de3cb00efdc` |
| `_state/npu-vllm/profiles/qwen3_decode_fused.json` | `44a44104122c4956d98b946910403a3792e414f49a89ed10bd1b1c3a01e3b343` |
| `_state/npu-vllm/profiles/qwen3_decode_paged128.json` | `c163d5669e718cfc3c4872900e0f4385d627d6b29f5dd81412eebf3c4604e745` |

Bootstrap 只在检测到 `torch.__jittor_version__` 后激活，拒绝已加载的
`torch_npu`，并让 Transformers 的可选 torch-npu 探测返回 false。platform 通过
`vllm.platform_plugins` entry point 注册为 `PlatformEnum.OOT`，禁用 TorchInductor
和 CUDA graph。CUSTOM attention 使用公有 Jittor paged-attention，metadata 的 host
materialization 每个 batch 只执行一次。adapter 还修复了 vLLM embedding loader 对
PyTorch slice-view 写回的依赖，并让自定义 vocab loader 用 `copy_` 写入既有参数，
避免 BF16 embedding 被 checkpoint 的 FP32 tensor 替换；当前只支持未量化 TP=1，
其他 TP 形态 fail-closed。
adapter 通过 vLLM OOT CustomOp registry 将标准 `RMSNorm` 和 `RotaryEmbedding`
路由到公有 Jittor serving primitive；其他 CustomOp 不变。worker 使用原 V1
`GPUModelRunner`，在 Jittor NPU 上初始化单 rank MPI，并替换 Triton slot-mapping。
它还显式同步 vLLM `CpuGpuBuffer` 和 `InputBatch` 的 NumPy/tensor 镜像，因为 Jittor
`.numpy()` 不提供 PyTorch 的共享内存语义；双方同一元素发生冲突时 fail-closed。
完整新请求 prefill 在写 cache 后直接对当前 Q/K/V 执行 causal CANN SDPA，不再从
128-token page 读回；cache-hit/chunked/ragged 继续使用公有 paged-attention。

## Evidence

- 本轮受影响的 library、stream/NVTX、compiler context、module hook、安装幂等、
  FSDP2 和 shim alias 回归：`76 passed`。
- Torch indexing/out-view 定向回归：`26 passed`；其中 `add` 验证 `alpha`、返回对象
  identity 和父 tensor 写回，`index_select` 验证切片父 tensor 未选区域保持不变。
- shim deploy 与资源结构：`21 passed`。
- 完整 structure：`226 passed, 2 skipped, 2 failed`；两项失败均来自仓库内既有
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
  prefill/decode 最大误差分别为 `6.25e-9/5.11e-9`；BF16 incremental flash
  最大误差 `2.59e-4`，实际 backend 为 `acl_incre_flash_attention_v4`，均为零
  fallback。
- block size 128 paged decode：合并 cache 的 K/V strided descriptor、int32 block
  table 和 host key length 直接调用 CANN V4，最大误差 `5.19e-4`，热内核约
  `0.34ms`。block16 通用路径、contiguous BF16 IncreFA 和 block128 paged FIA 三项
  在同一进程连续通过，均驻留 `device`、fallback 为 0，未复现资源退出错误。
- 公有 serving RMSNorm：CPU serving 回归 `7 passed`；真实 NPU 上 float32 activation
  + BF16 weight 连续执行两次，weight cast cache 保持同一对象，输出匹配 NumPy、
  dtype 为 float32、驻留 `device` 且 fallback 为 0。
- 公有 serving RoPE：真实 NPU packed Q/K、非连续 positions、head size 128 的 NeoX
  full rotary 匹配 NumPy，Q/K 均驻留 `device` 且 fallback 为 0。
- Qwen3-0.6B 完整 prefill：提示词 `The capital of France is` 编码为
  `[785, 6722, 315, 9625, 374]`，首个 greedy token 为 `12095`（` Paris`），与原生
  vLLM-Ascend 基线的第一个 token 一致。28 层 KV cache、hidden states 和 logits
  均在 `device`，fallback 为 0；embedding/hidden/logits 非零且有限，进程未加载
  `torch_npu` 或 `vllm_ascend`。热缓存严格加载为 `3.95s`，完整 prefill+logits 为
  `3.03s`；该时延尚无等价原生 prefill 协议，不作为性能验收。
- 同一进程、模型和 28 层 KV cache 上继续手工 greedy decode，四个 token 为
  `[12095, 13, 576, 6722]`，文本为 ` Paris. The capital`，逐 token 匹配原生基线。
  四步 hidden/logits/cache 均在 `device`，fallback 为 0，且未加载 PyTorch-NPU。
  block128 paged FIA 接入后的两步热态为 `0.222s/0.183s`；相比最初通用 attention
  路径继续改善，但仍慢于原生约 `10.96 token/s`。因此 correctness 接受，
  performance 不接受。
- OOT fused RMSNorm/RoPE 接入前的单 token profiler 设备算子合计约 `118ms`：
  incremental flash 仅 `2.36ms`；
  KV Scatter 为 `8.03ms`，多组 block/KV Gather 合计超过 `15ms`，RoPE 约
  `8.10ms`，四类线性 matmul 合计约 `6.31ms`。未被设备算子覆盖的墙钟主要在
  Python 构图、调度与 profiler 开销。一个让 CANN 直接消费 block-size 16 strided
  paged cache 的实验虽保持 token 正确，却退化到约 `6.95s/token`，已撤回且不进入
  主仓库；CANN 文档推荐的 paged block size 为 128/256，后续不复用该实验结论到
  其他 block size。
- block128 paged FIA profile 的设备算子合计从通用路径的 `98.4ms` 降到 `60.1ms`；
  cache Gather 已消失，paged IncreFA 为 `2.44ms/28层`。该 profile 尚未包含随后接入
  的 D2D cache update；后者继续移除了整块 cache Scatter/transpose/index 图。
  一个用 host block ID 改走 SliceV2 的实验虽数值正确，却将稳态请求退化到
  `7.996s`，已完全撤回。对 CANN executor 的 TensorList 做函数级 RAII 会在后续
  异步 Cast 中触发 use-after-free，也已撤回；其生命周期必须晚于 stream 完成。
- 公开 `vllm.LLM.generate`：单请求、`max_model_len=32`、
  `max_num_batched_tokens=32`、`max_num_seqs=1`、block size 128、32 MiB KV budget、
  eager 且关闭 async scheduling。提示词四个 greedy token 为
  `[12095, 13, 576, 6722]`，文本为 ` Paris. The capital`，逐 token 匹配手工路径和
  原生基线；进程未加载 `torch_npu` 或 `vllm_ascend`。关闭 prefix cache、同一 engine
  重复完整请求的稳态时延从 block16 通用路径的 `0.926s`，降到 block128 paged FIA
  和 direct prefill 的 `0.687s`，再降到 D2D cache update 后的 `0.610s/0.633s`，
  累计改善约 `33%`。原生同类复用 engine 结果为 `0.365s`，性能仍未验收。
- 当前 `2.0@d3c58fc0` 从新的隔离 Jittor/ACL 缓存重新启动公开 engine。冷缓存
  首请求因串行生成 ACL JIT 图耗时 `332.235s`；复用同一缓存再次启动后，三个请求
  分别为 `3.035s/0.61484s/0.61563s`，后两个稳态样本中心约 `0.615s`。所有请求和
  额外日志审计请求均输出 `[12095, 13, 576, 6722]`；审计得到
  `fallback_count=0`、`cpu_compile_count=0`，且 `torch_npu`、`vllm_ascend` 均未加载。
  这次复验确认当前 HEAD 的受限 engine correctness，未改变性能未验收结论。
- `2.0@c7834d16` 加上外置 adapter 的 BF16 参数保持修复后，完整参数审计确认
  embedding、QKV 和 MLP 权重均为 BF16 且驻留 device。随后接入 CANN SwiGLU 和
  单 token RoPE reshape 快路；公开 engine 的 10 次请求中，首个图热身为
  `2.990s`，其余 9 次为 `0.53619-0.55433s`，中位数 `0.53875s`。四个 greedy
  token 始终为 `[12095, 13, 576, 6722]`，审计仍为零 fallback、零 CPU compile，
  且未加载 `torch_npu`/`vllm_ascend`。相对同一 Jittor 协议此前约 `0.615s` 的
  基线改善约 `12.4%`，但相对原生 `0.364996s` 仍慢约 `47.6%`，故只接受这轮
  优化，不接受总体性能目标。
- 新增 SwiGLU 和单 token RoPE 的真 NPU 定向回归 `2 passed`；完整
  `tests/backends/npu/test_acl.py` 为 `42 passed`。CPU serving 回归为 `8 passed`，
  包含 ACL miss 后 CUDA backend 仍可达的调度契约。

## Open work

- 将外置 adapter 纳入可版本化、可安装的维护边界，补充多请求、prefix cache、
  chunked prefill、长上下文、异常退出和重复 engine 生命周期集成测试；继续保持
  量化和 TP>1 fail-closed，直到分别验证。
- 优化单 token decode 的 KV gather、GQA 和 attention launch，达到不慢于原生的
  等价暖态生成协议；减少 `CpuGpuBuffer`/`InputBatch` 的整块 host 镜像同步和
  metadata materialization，再完成吞吐、首 token、decode latency 和显存对比。
- 在当前 vLLM 版本重新验证历史 CUDA adapter 基线；旧 adapter 当时未版本化，
  不能作为当前可复现产物。
