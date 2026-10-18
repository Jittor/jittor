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

双卡热态性能仍明显落后 PyTorch。当前同配置正式测量中，优化后的 batch 1
Jittor/PyTorch 吞吐约为 45.3%，batch 4 约为 43.1%；这只是 TP=2 的有界离线
`engine.step` 测量，不是 HTTP 或所有负载的性能结论。

## 尚未完成的工作

1. **四卡/TP=4 扩展验收**：TP=4 性能和 placement、worker 清理已经通过；混合长度、取消/抢占和 HTTP 等更广矩阵仍待扩展。
2. **性能优化**：需要在空闲 GPU 上用同一模型、输入和依赖版本重测当前基线，再按热点逐项修改。每项都应先写最小失败/回归测例，修改后先跑测例，再跑真实 vLLM 和性能对照。
3. **多卡更广范围**：多卡 HTTP 服务、长时间 soak、请求取消/抢占的更大矩阵、多机、PP，以及不同 GPU/驱动组合还没有验收。
4. **高级执行模式**：CUDA Graph、compile 模式、量化（AWQ/GPTQ/FP8）和更多模型还没有完成兼容性验收。
5. **数值差异定位**：固定样例已经一致，但不能据此声明任意随机种子、任意批处理形状下模型浮点值或采样分布完全一致；若要做更强承诺，需要继续定位批处理导致的首个算子差异。

## 下一步建议

### 第一阶段：TP=4 扩展正确性

- 先确认四张卡空闲，并为每个 rank 使用独立的 Jittor/Triton/NCCL 缓存目录。
- 在已通过的 TP=4 性能和 placement 基础上，继续把双卡验收矩阵扩展到
  `VLLM_TP_SIZE=4`，覆盖 NCCL/CPU 控制通信、Qwen 贪心、固定随机采样、混合长度、cache 复用和显式关闭。
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

本地以下两个文件包含未提交的 TP 通用化改动；本轮已经完成 TP=4 实机验证，后续可按需要单独提交：

- `agent/skills/vllm-torch-compat/multigpu_acceptance.py`
- `agent/skills/vllm-torch-compat/multigpu_engine.py`

后续代理接手时应先查看 `git status`，确认是否继续这些改动；验证通过后再单独提交。

## 2026-09-29 参数生命周期与 TP 性能复测

本轮按 PyTorch 的 `engine.step` 边界加入了默认关闭的阶段探针，测量
`prepare_inputs`、`prepare_attn`、模型 forward 和采样，并在每轮比较参数的 Python
对象、`data_ptr`、dtype、device 和 shape。TP=2 短测中每个 rank 均为 226 个参数，
重复 forward 后没有参数新增、替换、迁移或 dtype/device 变化。因此性能差距不是
“每个 token 疯狂造新参数”。

定位到两处实际重复分配并逐项验证：

1. `layers.py` 的 RMSNorm 每次取 `self.weight.data`；Torch shim 的 `.data` 会执行
   `detach().stop_grad()`，每层每 token 都会构造新的 Var。改为只读的
   `self.weight` 后，TP=2 短测 steady model forward 中位数约从 25.0 ms 降至
   23.6--24.5 ms，生成 token 未变化。
2. `buffers.py` 的 CPU→CUDA UVA 快照在 `.to()` 后无条件 `.clone()`；迁移已经产生
   独立目标存储，clone 是第二次 GPU 拷贝。保留“若 `.to()` 没有迁移才 clone”后，
   TP=2 短测吞吐从 27.73 增至 28.50 tok/s，生成 token 未变化。

正式结果（同一 Qwen3-0.6B、FP16、eager、128 输入/32 输出、21 次重复）：

- TP=2：`tp2-jittor-optimized-20260929/result.json`，batch 1 为 34.404 tok/s，
  batch 4 为 129.673 tok/s；原基线为 32.696/127.312 tok/s。
- TP=4：`tp4-jittor-optimized-20260929/result.json`，batch 1 为 34.009 tok/s，
  batch 4 为 131.073 tok/s；原基线为 32.098/126.444 tok/s。

两次正式测量均完成 worker 正常退出、共享内存清理和 CUDA placement 检查。TP=4
为了避免 rank 2/3 重建相同 Jittor 核心，复用了 rank 0 的只读热缓存；四个 rank
仍分别确认在 CUDA 0/1/2/3，参数数均为 226。PyTorch 对照结果仍见
`tp2-oracle-latency.json` 和 `tp4-oracle-final-20260929/result.json`。

## 2026-09-29 当前 checkout 四组重测与 NCCL 生命周期优化

为避免旧缓存/旧 checkout 的结果混入结论，本轮在当前 `chk` checkout 上重新跑了
PyTorch、Jittor 的 TP=2/TP=4 四组基线。统一模型为 Qwen3-0.6B，FP16、eager、
128 输入/32 输出、batch 1/4、3 次 warmup、21 次正式重复；全部原始文件位于
`/home/chk/vllm-runs/compare-current-20260929/`。

修改前的当前 checkout 结果：

- PyTorch TP=2：78.215/305.624 tok/s，ITL 12.771/13.044 ms。
- Jittor TP=2：27.517/105.687 tok/s，ITL 32.830/35.937 ms。
- PyTorch TP=4：76.002/293.866 tok/s，ITL 13.122/13.468 ms。
- Jittor TP=4：23.828/84.002 tok/s，ITL 36.628/42.120 ms。

阶段探针确认两个 rank 始终各有 226 个参数，跨 token 的对象、data pointer、dtype、
device 和 shape 均未变化。主要差距在 model forward：PyTorch 约 12.0 ms，Jittor
约 28.0 ms；模块探针继续定位到 attention、MLP 和带 TP all-reduce 的 row-parallel
projection，而不是注册参数反复创建或迁移。

vLLM 在没有 PyNccl/custom all-reduce 时会先 clone 输入，再调用原地
`torch.distributed.all_reduce`。Jittor 的 NCCL op 本身已经是 out-of-place；兼容层随后
又把结果更新回 clone holder。`backend.py` 现在直接对原输入发起 Jittor NCCL，随后只
创建一个临时 holder 并用 `update` 发布 NCCL 结果。该 holder 的 clone 在执行前被替换，
保留 `_out_hint`/lazy graph 生命周期边界，但不再把输入先做一次实际设备拷贝。

直接裸返回 NCCL Var 的短测更快，但约 50 秒后在 `count_fuse/build_exec_plan` 崩溃；这
证明输出发布边界不能删除。当前 holder 方案已通过 TP=2、TP=4 各 21 次的长测，所有
worker 正常退出且共享内存清理通过。最终结果：

- Jittor TP=2：30.874/110.001 tok/s，ITL 31.527/34.649 ms；相对本轮修改前吞吐
  +12.2%/+4.1%。结果为 `final-jittor-tp2-holder/result.json`。
- Jittor TP=4：25.268/86.480 tok/s，ITL 36.141/41.408 ms；相对本轮修改前吞吐
  +6.0%/+3.0%。结果为 `final-jittor-tp4-holder/result.json`。

另测试了单 warp paged-decode attention 核和把同步间隔从 16 层增至 28 层：前者端到
端约 -0.5%，后者 model forward 从约 25.5 ms 退化至约 30.7 ms，均已撤回。最终代码
只保留有稳定收益的 NCCL holder 优化、对应回归测试，以及默认关闭的阶段/模块探针。

## 2026-09-30 临时 Var 生命周期与小算子拆分

在阶段探针中继续记录 Jittor 全局 liveness，而不只检查注册参数。TP=2 的 26 个执行
步里，两个 rank 的 `hold_vars`/`lived_vars` 都只在 720--734 之间波动，最后回到
720；`lived_ops` 始终为 12。RoPE 的 672 次 cache 匹配也全部返回原对象，没有
attribute 替换或 dtype/device 迁移。证据在
`liveness-jittor-tp2/result.json`。因此当前 decode 没有持续增长的临时 Var、参数
泄漏或 RoPE cache 反复转换。

用同步的子阶段探针与 PyTorch 对照后，Jittor/PyTorch 的中位耗时如下（探针具有侵入
性，只用于归因）：

- `prepare_pos_seq_lens`：0.311/0.037 ms；
- `combine_sampled_and_draft_tokens`：0.443/0.043 ms；
- `gather_block_tables`：0.332/0.033 ms；
- `compute_slot_mappings`：0.367/0.032 ms；
- `compute_logits`：0.955/0.278 ms；
- sampling 状态回写：0.438/0.041 ms。

对应证据为 `subphase-jittor-tp2-v2/result.json` 和
`subphase-oracle-tp2-v2/result.json`。差距主要是多个小 GPU kernel/提交的固定成本，
而不是 Python 端反复编译 Triton；`triton-stats-jittor-tp2/run.log` 显示初始化后
compile/bind 准备时间基本不再增长。

两项后续实验均已撤回：

1. 按 request-row 映射和 block 写入代次缓存 `gather_block_tables`，使该子阶段从
   0.332 ms 降至 0.011 ms，但 21 次正式 TP=2 吞吐反而从 30.874/110.001 降至
   30.710/108.732 tok/s，ITL 从 31.527/34.649 增至 31.655/35.049 ms。Python key
   和缓存管理抵消了收益；证据为 `final2-jittor-tp2-blockcache/result.json`。
2. 在 rotary fast path 直接使用已经对齐的 cache，短测吞吐为
   30.751/110.093 tok/s，对比 30.874/110.001 没有稳定收益，且会缩小动态 dtype/device
   兼容范围；证据为 `opt-rope-direct-tp2/result.json`。

因此生产代码仍只保留上一节的 NCCL holder 优化。下一步若继续压缩
`prepare_inputs`/sampling，应把相关小 kernel 合并成一个后端原语，或复用 vLLM 的
持久输入 buffer；不要再增加 Python 层逐 token cache key。

## 2026-10-01 paged decode 与 greedy sampling 微基准及优化

本轮开始前已 fetch `origin/chk`，本地与远端同步于 `a8b255db`。新增
`agent/skills/vllm-torch-compat/hotpath_microbench.py`，在同一张空闲 RTX 4090（物理
GPU 3）上分别调用原生 PyTorch vLLM 的 FlashAttention paged-cache 实现、Jittor 的
`jt.nn.paged_attention`，以及两端相同 vLLM 0.24 的 `gumbel_sample`。每项 20 次热身、
100 次测量，调用前后同步；预先规定“中位数至少 1.5 倍且绝对差至少 0.05 ms”为明显
差距。该协议是侵入式 kernel 诊断，不代表端到端吞吐。

修改前的中位数：

- paged decode，batch 1、128 KV token：Jittor 0.114 ms，PyTorch 0.033 ms（3.46 倍）；
- paged decode，batch 1、512 KV token：Jittor 0.325 ms，PyTorch 0.039 ms（8.44 倍）；
- greedy gumbel sampling，batch 1：Jittor 0.322 ms，PyTorch 0.041 ms（7.78 倍）；
- Jittor 直接 rowwise argmax 为 0.061 ms，且与 temperature=0 的 gumbel token 完全一致。

据此作两项有界修改：paged-decode kernel 由整个 block 串行遍历 KV 位置，改为 8 个
warp 各自遍历一部分位置、warp 内做 QK 点积和在线 softmax，最后合并各 warp 的
softmax 状态；sampling 仅在整个 batch 均为 temperature=0 且没有 top-k/top-p 时，
保留全部 logits processor 后用等价 rowwise argmax，混合/随机/过滤路径仍调用 vLLM
原实现。

修改后 paged decode 为 0.049/0.068 ms，相对原实现快 2.34/4.75 倍；相对 PyTorch
差距缩至 1.48/1.78 倍。跨框架输出仅有 FP16 舍入级差异，greedy token 完全一致。
CUDA 独立参考的 FP16/BF16 kernel 测试、6 项 paged-attention 测试和 2 项 greedy
路由测试均通过。

真实 Qwen3-0.6B engine smoke（FP16 eager、128 输入/32 输出、batch 1/4、3 次热身、
5 次测量）为 42.50/156.55 tok/s，逐轮 greedy token 稳定；相对上一轮正式
42.77/157.09 tok/s 没有可声明的端到端吞吐提升。阶段探针的 decode 中位数为：
batch 1 的 execute/sample 20.171/2.065 ms，batch 4 为 22.625/1.801 ms。sampling
相对上一轮约 2.519 ms 有明确下降，而 paged kernel 的同步微基准收益没有等比例反映到
懒执行的完整 model forward。因此本轮只声明 kernel 与 sampling-stage 改善，不声明
端到端加速。未版本化原始证据位于
`$JITTOR_LAB_ROOT/_state/vllm-hotpath/20261001/`。
