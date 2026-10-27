# DeepSpeed 双 rank BF16 Engine 验证

- Status: 有界 CUDA 正确性验证；不作性能主张。
- Reviewed: 2026-10-15；owner: Jittor 核心维护者。
- Synchronized numerical baseline: `5ddb1f1399dfb9976ba73500e2764e4c27638049`。
- Storage/scalar implementation: `15ec2a32e0bae713cb79723df9b865583e71a5cf`。
- Subsequent synchronization: origin `344dba6cd259259050c64b41541fdff281788b22`，
  merge 为 `d4a4f1100a36d1e6b34bfda49db1e37c85f5242e`。
- Recheck when: native view/alias 存储、标量除法、autograd 发布、
  集合输出拷贝、optimizer 状态或固定版本的 DeepSpeed 源码变化。

## 范围与独立参考

较晚的 [维护 gate 验收](2026-10-15-deepspeed-maintained-gate.md)
增加了双输入扰动矩阵、独占 FileStore 引导与 CPU Adam 严格续训修正。这份较早的报告
保留其测量时的源码 pin。

真实的上游 `DeepSpeedEngine` 在两块 CUDA 设备上执行了反向、optimizer 步骤、保存、
新 Engine 加载与继续训练。两个 rank 都使用 BF16 model/输入张量、FP32 通信与
optimizer 主分区，GAS 2，六个 microstep 与三次更新。checkpoint 发生在第四个
microstep 之后；新 Engine 重复最后两个 microstep。确定性的残差
Linear/GELU/Linear/LayerNorm/Linear 模型有八个参数、共 332 个元素。
每个 optimizer rank 实际拥有 166 个 FP32 主元素。各 rank 的输入相差一个固定偏移。
该证据覆盖 AdamW、输入扰动零以及 ZeRO 第 1、2 阶段；输入偏移在各 rank 之间仍有差异。

两个 runtime 都使用官方 DeepSpeed 源码
`f5af15c2b4da10411e340850c00ba0cb5b14fd87`（`0.19.7+f5af15c`）。构建出的包
SHA256 为 `467622416c9819162ad853ea517805b5eb28e59ba3d43a3641add97668286c48`。
继续使用此前准入的绑定源码 adapter；本批次不新增 adapter 或替代训练算法。

目标环境是 Linux x86-64、RTX 4090/SM89、驱动 580.178.04、nvcc 12.2.140、
g++ 11.5 与 `pipcu122_a89fa4fed08c`。独立参考加载二进制
PyTorch `2.11.0+cu128`、其真实 `_C` 扩展，且没有 Jittor 安装上下文标记。
这是两套不同的 CUDA 库栈。两个 runtime 记录了匹配的线程/精度设置。
BF16 操作数、参数、输出与所请求的梯度都在 CUDA 上物化并检查。两个 rank 都在
禁止后端 fallback 的条件下运行，观察到计数增量为零。

## 共享缺陷与修正

在 `copy_`、`div_`、`mul_` 或 `zero_` 期间重绑定稠密切片，可能让已分离的别名继续
读取先前的分配。既有的 native view 记录现在区分物理存储视图与物化的逻辑转置，
包括弱 base holder 过期之后。证明使用实际的 pending share link 或活跃分配身份，
绝不用两个未分配指针的相等性。无分离/no-grad 的稠密写入在解析 pending reader 之后
复用 native `_copy_into`。物化转置的回写保留其既有 owner。公开的 copy 广播经过
校验，部分物理重叠在写入前被拒绝。连续的物理视图在既有有界的 `_foreach_zero_`
实现中也可用。

另外，被 Python float 除的冻结张量会错误地获得 `requires_grad=True`。Python 的
精度提升路径创建了一个可训练的内部控制数组常量。C++ 快速路径也在既有 Torch
frontend 作用域之外构造其内部转换。两条路径现在都保持公开梯度域：只有内部标量
常量停止梯度，而真实的张量输入梯度保持连接。Dtype/opmath 规则保留。
该修正覆盖前向/反射除法、整数/Python/NumPy 标量除数、FP32/FP64/FP16/BF16 以及
NaN/Inf 行为。非叶 `zero_` 使用 native 选择产生精确的零值与零导数，包括非有限输入。

原先未被观测的 ZeRO2 运行复现了约 31.6% 的完整梯度误差。仅仅物化诊断观测就可能
把它掩盖。因此验收使用不带这些诊断包装的普通仓库 runner。

## 结果

比较复用公共生态 harness 的 `_comparison_floor` 与 `_divergence`：类别下限 `1e-3`
乘以参考最大值，forward/state 容差 `2e-3`，梯度容差 `1e-2`。元数据、shape、有限值
与实际 optimizer 类别单独检查。没有放宽任何阈值。

| 用例，每 rank | 数值字段 | 梯度字段 | 前向/状态结果 | 严格 fresh 续训 |
| --- | --- | --- | --- | --- |
| ZeRO1，真实 `BF16_Optimizer` | 240 | 全部 144 个与 Torch 逐位相等 | 84/96 字段逐位相等；最大归一化误差 `1.56e-7` | 64 字段逐位相等 |
| ZeRO2，真实 `DeepSpeedZeroOptimizer` | 208 | 全部 112 个与 Torch 逐位相等 | 84/96 字段逐位相等；最大归一化误差 `1.56e-7` | 56 字段逐位相等 |

字段计数包含续训的 microstep。每个传入参数梯度都通过真实 Tensor hook 观测。
ZeRO2 的公开完整梯度 getter 在 GAS 边界处调用，即上游实际暴露它的位置；省略的
非边界完整梯度不计为成功观测。

严格续训比较保存/恢复与最终/续训的 optimizer moments 与 FP32 master，以及两个续训
microstep 的输出、loss、model 参数、输入梯度、传入参数梯度与可用的完整梯度。
两个 rank 都在第三步更新了其 FP32 分区的全部 166 个元素。对 ZeRO2，更新 L2 范数为
`0.00906536` 与 `0.02911891`。

| 数值源码上的共享回归 | 结果 |
| --- | --- |
| CPU core，native 与 Torch 模式 | 322 passed, 42 skipped, 1 existing xfail |
| CPU inplace/标量/dtype/除法 | 60 passed；九项仅 CUDA 跳过 |
| CUDA inplace/标量/dtype/除法 | 69 passed，零跳过，fallback 增量为零 |
| Autograd/optimizer/存储/detach/load/to owner，CPU 与 CUDA | 各 108 passed，1 项仅 ACL skip；CUDA fallback 增量为零 |
| 独立二进制 Torch 公开契约，CPU 与 CUDA | 各 12 passed |
| 仓库布局 | 通过 |
| 同步前的完整结构 | 1459 passed，六项跳过；先前 Linear 报告的两项中心登记失败 |

结构失败涉及先前报告的 results 索引与生成的 MANIFEST 登记。上游 `344dba6cd` 包含
这些整合更新；表中保留其测量时的原始源码标签。

## 主树整合验证

主 checkout 把八个存储/标量文件逐字节整合到同步后的
`344dba6cd259259050c64b41541fdff281788b22`。CPU core 通过 322 项测试，
42 项已声明跳过与 1 项既有 xfail。CPU 整合 owner 通过 49 项测试，跳过 1 个仅 CUDA
的 BF16 类。实际 CUDA 的 inplace、foreach-zero 与 autograd owner 通过 51 项测试
且无跳过。整合审计重新哈希双 rank v2 证据 manifest 中的全部 31 个文件，并直接检查
全部四个 rank 归档的有限值与逐位相等的梯度。这些 owner 检查不重标历史双 rank
数值基线。

整合日志与哈希审计未版本化，位于
`$JITTOR_LAB_ROOT/_state/inplace-scalar-integration/`。

## 复现与产物

使用专用 runner `compat/tests/torch/_ecosystem_deepspeed_runner.py`。
先串行预热每个 rank 各自的缓存，包括真实 NCCL 初始化。
以独立的设备可见性、HOME、JITTOR_HOME、TMPDIR 与缓存名并发启动两个 rank，
使用公共 rendezvous 与共享 checkpoint 目录。实际未观测的 runner 参数为：

```bash
<jittor-python> compat/tests/torch/_ecosystem_deepspeed_runner.py \
  --runtime jittor --device cuda --dtype bfloat16 --world-size 2 \
  --rank <rank> --rendezvous tcp://<address>:<port> \
  --checkpoint <shared-checkpoint> --optimizer AdamW --zero-stage 2 \
  --output <fresh-case-directory>/rank-<rank>.npz
```

设置匹配的标准 RANK/WORLD_SIZE/LOCAL_RANK 与 MASTER_ADDR/MASTER_PORT
变量、`JITTOR_TORCH_DISTRIBUTED_AUTO_INIT=1`、`JT_BUILD_USE_MPI=0`、
`JT_BUILD_USE_NCCL=1` 以及公共的 `JITTOR_DIST_RENDEZVOUS_DIR`。参考侧在其独立环境中
使用相同参数并加 `--runtime torch`。stage 1 只更改特定阶段的
case/checkpoint/output 位置与 `--zero-stage 1`。每个用例必须使用全新的输出目录；
stage、optimizer 与精度应包含在其路径中。仅靠手动输出标签不足以防止冲突。

原始产物未版本化，位于证据 lab 的
`_state/refactor-goal/deepspeed/`。完整的源码/产物 manifest 是
`multirank/deepspeed-bf16-stage1-stage2-v2-evidence.json`：1208 个源码哈希与
31 个原始/结果/比较日志哈希。该 manifest 明确记录了一次历史 Stage2 输出名冲突；
按阶段区分的新 Stage2-v2 产物取代了被覆盖的原始文件。其当前 native 产物为
`native/rank*/formal-stage1-bf16-scalar-scope-v1.npz` 与
`native/rank*/formal-stage2-bf16-scalar-scope-v2.npz`。比较日志为
`formal-stage{1,2}-scalar-scope-comparison-v2.log`；严格续训日志使用同目录下的
`formal-stage1-scalar-scope-strict-resume-v1.log` 与
`formal-stage2-scalar-scope-strict-resume-v2.log`。

## Leaf/view 后续

本工作期间观察到的 leaf-view 缺口在
`489417b368640e724372d9b22b509d2ef42b00f3` 单独修正，与 origin
`3ba5bb7714ad93a7a1cef6a67d2f85a234b412a6` 同步。这不重标较早矩阵的源码。
通过可训练叶子的活跃或孤儿视图进行非法的 grad 启用写入，在修改之前即失败。
合法的非叶写入保留真实的输入与 RHS 梯度，包括切片拷贝的零输入边。
冻结视图可以获得可训练的 RHS 边，对最初冻结的叶子与可逆的 `requires_grad_(False)`
都是如此。no-grad 的 leaf-view 写入与冻结物理别名保持其受支持的存储行为。

该修正使用既有的 native view provenance、writeback 与图刷新 owner。native 赋值
保持其默认策略；只有既有的 Torch `_update` 路径在获得 RHS 边时绕过对冻结目标旧
梯度标志的继承。没有替换导数算法，也不对可微视图做一刀切拒绝。

| 检查 | 结果 |
| --- | --- |
| 新 view 契约加 inplace/foreach 邻近项，CPU / CUDA 选择的运行 | 26 / 26 通过 |
| 同样的六项公开契约，独立二进制 Torch CPU / CUDA | 6 / 6 通过 |
| Native view/存储/setitem/step 捕获，CPU | 31 passed，27 项仅 CUDA 跳过 |
| 同样的 native 文件，启用 CUDA 并按测试显式设置设备作用域的运行 | 58 passed，无跳过，fallback 增量 0；未设作用域的节点仍使用 CPU |
| Autograd/optimizer/存储/dtype owner 邻近项，CPU / CUDA 选择的运行 | 108 / 108 通过，各 1 项仅 ACL skip |
| CPU core | 322 passed, 42 skipped, 1 existing xfail |
| 结构与布局 | 1478 passed，六项跳过；布局通过 |

CUDA 选择的 26 项包含一项私有的 host 来源检查；生成的设备用例在 CUDA 上执行。
CUDA owner 检查同样要求 fallback 增量为 0。历史的 native 58 节点运行没有全局启用
CUDA：带显式 CUDA 作用域的节点在 CUDA 上运行，而未设作用域的节点（包括被消耗的
expand 分配检查）在 CPU 上运行。它不构成该分配检查在 CUDA 上的证明。在整合到
`4a4a480e6` 期间，全局启用 CUDA 得到 57 通过与该既有分配检查的 1 项失败：数值和
正确，但分配的字节增长 68,157,440，超过 16,777,216。该节点在单独运行与严格 kernel
math 下同样失败。这些失败保留在 `_state/leaf-view-integration/` 下；没有放宽任何
分配阈值。整合后的公开 view/inplace/foreach 契约在 CPU 与 CUDA 选择的运行中都通过
全部 26 个节点；native CPU 给出 31 通过与 27 个仅 CUDA 跳过，CPU core 给出 322 通过、
42 跳过与 1 项既有 xfail。一个独立的干净
`4a4a480e628dae634a9d660f320c3aa957341e99` worktree 配以单独的 CUDA 编译缓存，
复现了相同的数值结果与 68,157,440 字节分配失败。这是既有的 CUDA 全归约融合边界，
记为 KI-CODEGEN-005，而不是新的 leaf/view 回归。Profile 显示默认的两阶段 CUDA
full sum 物化了乘法的结果；显式命名两个归约轴则保留 native 融合消费者并分配
1,048,576 字节。既有失败测试与预算不变。根整合的结构/工具验证也通过 1661 个节点，
含六项环境跳过。下文中绑定源码的 ZeRO2 oracle 测量仍固定在 `489417b36`。
报告的两项关于已安装包解析的结构 skip 是环境检查，而非执行的张量契约。

在串行的逐 rank 预热之后，真实的 ZeRO2 BF16 用例针对干净的 `489417b36` 源码
重跑，使用唯一的 `stage2-leaf-owner-v1` 输出与新的 checkpoint 目录。两个 rank 再次
通过针对独立二进制 Torch oracle 的全部 208 个数值字段，全部 112 个梯度字段逐位相等，
最大归一化前向/状态差异仍为 `1.5522042910257974e-7`。每个 rank 的 56 个严格
保存/加载/续训字段在 Jittor 内逐位相等；全部 166 个主分区元素在第三次更新时改变，
L2 增量为 `0.009065359830856323` 与 `0.02911890670657158`。两个 rank 的 fallback
增量均为 0。

未版本化证据仍位于同一任务状态根下：
`multirank/leaf-owner-evidence-v1.json` 绑定 1531 个源码哈希与 28 个证据哈希，
包括干净源码 pin、原始 rank NPZ/日志、独立 oracle、比较、严格续训以及全部
unit/core/structure 日志。它还保留修复前的 leaf-view 与 frozen-RHS 反例。较早的 v2
矩阵 manifest 保持不变。这是一项绑定源码的专用 CUDA 结果；没有等价的 world-size-two
BF16 CPU 运行，因此它无法满足公共记分卡匹配的 CPU 阶梯，也不能计为加速器广度。

## 边界

这个极小的双 rank 用例不确立 ZeRO3、CPU/NVMe offload、compiled autograd、overlap
模式、BF16 下的其他 optimizer、大模型、吞吐、ROCm 或 NPU 行为。既有的可微/非连续
foreach 限制仍明确保留。leaf/view 修正及其独立的验证范围记录在上文。维护中的双 GPU
专用 gate 与第二个显式输入扰动仍是后续工作；原有的单 rank FP32 gate 不能替代该协议。
