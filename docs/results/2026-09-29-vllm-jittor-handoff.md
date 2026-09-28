# vLLM + Jittor 适配交接记录

- 分支：`chk`
- 本次交接基线：`c0bdcc543138a62a9221a926c3448f65e3a7481d`
- 远程仓库：`git@github.com:Jittor/jittor.git`
- 更新日期：2026-09-29

## 已完成内容

### 单卡 CUDA

Qwen3-0.6B 和 OPT-125m 的 FP16、eager、TP=1 验收已经完成。已覆盖贪心生成、固定种子的随机采样、logprob、重复惩罚、长输入、连续请求、取消后恢复、KV cache 紧张时的抢占恢复、HTTP 服务和 WikiText-2 语言模型质量对照。

在固定的有界验收矩阵中，Jittor 与 PyTorch 的贪心 token、缓存冷算/命中/清空后的结果一致；Qwen 和 OPT 的 WikiText-2 PPL 也在预先设定的误差范围内。随机生成在固定样例中已通过，但这不等于所有种子和所有调度下的浮点结果都逐位一致。

单卡热态性能基线仍明显低于 PyTorch：此前 Qwen 基准中，Jittor 吞吐约为 PyTorch 的 33.2%（batch 1）和 35.5%（batch 4）。已经完成热点统计，主要候选包括重复 dtype/精度策略查询、安装上下文查找、同步、UVA 快照和 Triton 设备拷贝；尚未宣称这些优化已经完成。

### 双卡 CUDA（TP=2）

Qwen3-0.6B 在单机两张 RTX 4090 上完成了 FP16、eager、TP=2 的正确性收口。最终无 hook 矩阵两侧均完成 70 个请求、1208 个输出 token，长文本矩阵完成 10 个请求、224 个输出 token；贪心和固定随机样例、连续批次、混合长度、反序请求、KV cache 复用及清空后的重新计算均通过。两 rank 的参数、KV cache 和 forward 输出均确认位于对应 CUDA 设备，worker 正常退出且共享内存清理完成。

本轮修复和验证覆盖了 CPU 控制通信、Store 生命周期、`get_group_rank`、PyNccl 字节通信、跨卡 staged write、自动图重放遇到外部 Triton/NCCL 操作时的边界、事件同步竞态，以及高词表重复惩罚的整数类型截断问题。自动图重放不会再错误复用缺少外部操作的旧结果。

双卡热态性能仍明显落后 PyTorch。已有口径下，batch 1 Jittor/PyTorch 吞吐约为 36.55%，batch 4 约为 34.62%；这只是 TP=2 的有界离线 `engine.step` 测量，不是 HTTP 或所有负载的性能结论。

## 尚未完成的工作

1. **四卡/TP=4 验收**：当前没有完成四卡端到端正确性验收。已有脚本正在从 TP=2 写成可配置 TP 大小，但相关改动还没有完成真实 GPU 验证。
2. **性能优化**：需要在空闲 GPU 上用同一模型、输入和依赖版本重测当前基线，再按热点逐项修改。每项都应先写最小失败/回归测例，修改后先跑测例，再跑真实 vLLM 和性能对照。
3. **多卡更广范围**：多卡 HTTP 服务、长时间 soak、请求取消/抢占的更大矩阵、多机、PP，以及不同 GPU/驱动组合还没有验收。
4. **高级执行模式**：CUDA Graph、compile 模式、量化（AWQ/GPTQ/FP8）和更多模型还没有完成兼容性验收。
5. **数值差异定位**：固定样例已经一致，但不能据此声明任意随机种子、任意批处理形状下模型浮点值或采样分布完全一致；若要做更强承诺，需要继续定位批处理导致的首个算子差异。

## 下一步建议

### 第一阶段：TP=4 正确性

- 先确认四张卡空闲，并为每个 rank 使用独立的 Jittor/Triton/NCCL 缓存目录。
- 使用现有双卡验收矩阵扩展到 `VLLM_TP_SIZE=4`，至少覆盖 placement、NCCL/CPU 控制通信、Qwen 贪心、固定随机采样、混合长度、cache 复用和显式关闭。
- 每个失败先保留最小失败测例，再修改生产代码；通过后保存两套环境的 JSON、日志、worker 退出和设备指针证据。

### 第二阶段：性能优化

- 在同一空闲卡上分别测 PyTorch、当前 Jittor 和每次单项修改后的 Jittor。
- 记录 TTFT、ITL、吞吐及 p50/p95，避免把初始化、编译和诊断 hook 纳入热态计时。
- 优先检查重复 dtype/精度策略查询、安装上下文查找、无效类型转换、UVA 快照、同步和临时张量生命周期；每次只改一个热点并做正确性回归。

### 第三阶段：扩展能力

- 在 TP=4 正确性稳定后，再验收多卡 HTTP、长时间运行、量化、CUDA Graph/compile 和更多模型。
- 每新增一种执行模式，都要分别记录“能加载”“能生成”“与 PyTorch 对照”“资源是否正常释放”四类结果，避免只凭一次生成宣称完成。

## 复现入口

主要脚本位于 `agent/skills/vllm-torch-compat/`，双卡入口包括：

- `multigpu_acceptance.py`
- `multigpu_acceptance_compare.py`
- `multigpu_engine.py`
- `multigpu_collectives.py`
- `multigpu_latency.py`

详细环境约束和验证规则见 `agent/skills/vllm-torch-compat/SKILL.md`；双卡结果见 `docs/results/2026-09-27-vllm-multigpu.md`，单卡结果见 `docs/results/2026-09-23-vllm-singlecard-extended-acceptance.md`。

## 工作区注意事项

本次交接提交只包含已经提交并形成基线的生产代码、测试和文档。以下两个文件在本地还有未提交的 TP 通用化改动，尚未经过 TP=4 实机验证，因此没有随本次推送发布：

- `agent/skills/vllm-torch-compat/multigpu_acceptance.py`
- `agent/skills/vllm-torch-compat/multigpu_engine.py`

后续代理接手时应先查看 `git status`，确认是否继续这些改动；验证通过后再单独提交。
