# 保存输出 tanh 梯度与原生 FSDP BF16 验证

- 状态：在源码
  `cc45f12a216af639faf1e7bd3fc2ff916ce16653` 上于 CPU 与两个真实 CUDA/NCCL rank 验证通过。
- 维护者：Jittor 核心维护者。
- 远端基线：`344dba6cd259259050c64b41541fdff281788b22`，最终验证前整合为
  `1d5fc1480401cf6e22220ff803f058f4013a0231`。
- 复查条件：一元微分、Linear 累加、混合精度、
  collective 归约、编译器或独立 Torch 参考变化。

原生 tanh backward 现在对其保存的输出求导。这消除了固定原生 FSDP fixture 中
剩余的 BF16 梯度差异，并使饱和 FP16 输入处的一阶与二阶导数保持有限。生产改动
仅限 `src/ops/unary_op.cc`；梯度引擎、AMP 策略与优化器数学未变。

## 问题与修复

此前的 backward 重算 `cosh(x)` 再除以其平方。对于舍入后的 BF16 或 FP16 tanh 输出，
这与独立 Torch 实现的导数不同。在量级为 20 的 FP16 输入下，该重算表达式
还在 CPU 与 CUDA 的十三输入二阶导数 probe 中产生两个 NaN。

受控层 probe 在 step 0 与 step 4 使用原 oracle 权重及两个 rank 的固定输入。
Linear 输出、tanh 输出、预测值以及进入 tanh 的梯度逐位一致。首个差异是
离开 tanh、进入第一个 Linear 的梯度。另一次 AMP level-three probe 使
全部 340 个 BF16 轨迹数组保持不变，排除了该拟议修复。

替换实现通过保存的输出构建普通可微图
`dout * (1 - out * out)`。CPU FP16/BF16 使用 FP32 opmath 并将结果舍入一次；
CUDA 保留在实际 Torch CUDA kernel 中观测到的低精度标量舍入边界。这些边界
可观测地不同：冻结的 BF16 grid case 在输入 -1 处产生
Torch CPU 上的 `-0.040283203125` 与 Torch CUDA 上的 `-0.0400390625`。该回归
为各设备保留各自的 gold 值。FP32 与 FP64 保留自身 dtype。

[新回归](../../tests/ops/test_tanh_backward_precision.py)在 AMP level 0 与 3 下检查
FP32、FP16 与 BF16 的一阶和二阶导数，包括舍入饱和情形，并以独立二进制 Torch 值为参考。另有一个
FP64 测试验证解析二阶导数。低精度 grid 结果与对应真实设备 oracle 逐位一致；每个被检查值都必须
具有精确 shape 且有限，禁止后端 fallback。

## 独立分布式标定与固定协议

独立 runtime 为二进制 Torch `2.11.0+cu128`，git
`70d99e998b4955e0049d13a98d77ae1b14db1f45`。其真实 FSDP2 使用两个 CUDA rank、
BF16 参数、FP32 reduce dtype、FP32 输出与实际 DTensor 收集。
全部 **396** 个数组与其 dense 独立 Torch oracle 逐位相等，包括
输出、loss、输入/参数梯度、参数与优化器状态。
因此此前的残差是该 fixture 的实现差异。

可复用的
[标定工具](../../agent/skills/jittor-distributed-verification/native_fsdp_torch_calibration.py)
单独提交为 `eca1d57cb`。它导入同一个仅依赖 NumPy 的 fixture
与真实 Torch FSDP2；不引入第二套分片实现或新的数值预算。使用独立二进制 Torch 解释器运行：

```bash
<binary-torch-python> -m torch.distributed.run --standalone --nproc_per_node=2 \
  agent/skills/jittor-distributed-verification/native_fsdp_torch_calibration.py \
  --source-root "$PWD" \
  --output-dir "$JITTOR_LAB_ROOT/_state/native-fsdp/calibration"
```

原生 fixture 与阈值和原
[已接受源码](2026-10-15-native-fsdp.md)具有相同字节。其 fixture SHA-256 仍为
`5b675b8c250fde9bbddc42d9ffc1f7c5c68caa342bdd80631a196eee75bcdd7a`。
它使用 Linear(5, 7)、tanh、Dropout 与 Linear(7, 3)，四个参数与 66
个逻辑元素。每个 rank 拥有 35 个填充元素。五步分别用带动量的 SGD
与 AdamW，各含两个参数组，在 FP32 与 BF16 下执行。

逐元素预算保持 FP32 `atol=2e-6, rtol=2e-5` 与 BF16
`atol=4e-4, rtol=3e-2`。逐参数更新 L2 预算分别保持
`2e-6 + 2e-5 * reference_norm` 与 `1e-6 + 0.03 * reference_norm`。
每个参考更新必须超过十个绝对下限。每个实际参数
与步都拒绝注入的零梯度与零更新。

## 三个保留源码上的同输入结果

下表为 AdamW、rank 0、step 5 的 `fc1.bias` 梯度。其
参考 L2 范数始终为 **0.001057407702319324**。

| 源码 | 梯度误差 L2 | 相对 L2 误差 | 精确匹配的 BF16 梯度数组 |
| --- | --- | --- | --- |
| 原始 `ee24fac2b` | 0.00011868552246596664 | 11.2241969% | 11 / 80 |
| 共享 Linear 修复 `251fc6bc7` | 0.00007667446334380656 | 7.2511731% | 35 / 80 |
| 保存输出 tanh `cc45f12a2` | 0 | 0% | 80 / 80 |

原始逐元素绝对容差接纳了该小梯度的绝对误差。其报告与原始数组保留原始的 11.224% 结论；
中间的 7.251% 结果也保留。最终结果证明该固定 fixture 上全部
80 个被测 BF16 参数梯度数组与两个独立 Torch 参考都逐位相等。

完整参数轨迹保留少量 FP32 更新差异：
BF16 AdamW 参数相对 L2 误差最大为 `1.0407533e-7`。BF16
逐步更新相对 L2 误差最大为 `5.3062839e-7`。被测 BF16
更新范数最小为 `0.00014121785352472216`，超过 141 个绝对下限；绝对
项最多贡献更新误差预算的 19.1%。两种精度都通过全部
**160 项零梯度与 160 项零更新负向检查**。逐参数、
逐步范数、误差、下限与预算保留在原始审计中。

## 验收与范围

| 检查 | 结果 |
| --- | --- |
| CPU 一元、FP64、原生 FSDP 与 checkpoint 相邻用例 | 65 passed, 20 accelerator skips |
| CUDA 一元与 FP64 相邻用例 | 37 passed, zero skipped |
| 既有 CUDA Linear 二阶梯度节点 | 1 passed, 5 other nodes deselected |
| CPU core，native 与 Torch 模式 | 322 passed, 42 skipped, 1 expected failure |
| 完整结构门禁 | 1472 passed, 4 skipped, zero failures |
| 原生 CUDA 对拍 | 7 passed per rank, zero skipped |
| 三步后保存与在线续训 | 7 passed per rank, zero skipped |
| fresh launcher 恢复与再两步 | 7 passed per rank, zero skipped |
| 布局、Ruff 与空白 | Passed |

unit/core/structure 门禁使用的正是随后提交的 runtime
字节。三个最终 CUDA 阶段都在干净、固定的 `cc45f12a2` HEAD 下运行。
初次一次可选 SciPy 依赖失败与一次缺失的显式 pytest
plugin 配置在成功重跑之前已在外部修正；
被拒的尝试保留在原始日志中，不计为通过。共享
依赖安装未改动。

最终 24 个候选 NPZ 文件含 **1,968 个数组**。fresh launcher 的
全部 **368 个数组**与不中断的原生训练精确相等，包括
Dropout 输出、梯度、参数、动量以及每个 rank 的 Python、NumPy、
native CPU 与 CUDA RNG 流。保存/恢复中 Dropout 概率为 0.25；
独立 Torch 对拍使用概率零。两个 rank 的 RNG 流
被独立检查且互不相同。

原生运行使用 CUDA 12.2.140、NCCL 2.18.3 与严格 CUDA kernel math。两个
原生精度层级均为 `highest`；独立 Torch 关闭 TF32 并使用
`highest` matmul 精度。插桩在指针查询前验证实际 BF16 GEMM 操作数
与 FP32 累加器的 CUDA 分配，以及 BF16
all-gather、FP32 reduce-scatter 与 FP32 master/动量。三个阶段
在两个 rank 上共执行 **384 次 all-gather 与 384 次 reduce-scatter 本地调用**。
每个阶段零 fallback，且既不导入 Torch 也不导入其兼容
前端。rank 本地 checkpoint 标注仅覆盖真正分片的状态
与 RNG；复制状态引起的 gain/control 漂移会被拒绝。

这确立了上述固定的两 rank CUDA 训练与 fresh-restore 范围。
其他后端、改变 world size、任意模型族、峰值显存与
吞吐量不在本次测量范围内。当时存在其他 GPU 进程，因此
这些结果只支持正确性结论。

## 主树整合验证

runtime 修复在 `3ba5bb771` 上逐字节整合。CPU core 通过
322 项测试，含 42 项 declared skip 与一个既有 xfail。原生 CPU tanh/TP
owner 通过 17 项测试，含四项 accelerator skip；七个 tanh 测试在
将其精确 gold 值显式限定到严格 kernel math 后也通过。
真实 CUDA 一元、低精度与 FP64 相邻用例无 skip 通过全部 37 项测试。
首次 default-math 尝试暴露两处 FP16 精确 gold 不匹配
与四处缺少真实 SciPy `erfinv` 的失败；这些日志保持未版本化。
测试现在声明严格 math，私有既有 SciPy 安装
提供相邻用例的真实依赖。生产算术未因该测试/环境修正而改变。
其他一元相邻用例在 default math 下运行。

真实 CUDA/NCCL TP 复验在独立 rank cache 串行预热后，每 rank 无 skip
通过 13 项测试。这包括 FP32/BF16 训练、
优化器状态、完整梯度与 fresh 原生 checkpoint 恢复。全部 431
个记录的 TP 源码哈希在整合运行期间保持不变。
这些是整合检查；历史 FSDP 测量仍绑定 `cc45f12a2`。

根审计独立重新哈希全部 801 个后续文件、检查全部
1,968 个候选数组的有限性、将全部 80 个填充 BF16 梯度
分片与 dense 独立 Torch 比较，并逐位比较全部 368 个 fresh-resume 数组。
原始整合状态保存在
`$JITTOR_LAB_ROOT/_state/tanh-integration/` 与
`$JITTOR_LAB_ROOT/_state/native-tp-tanh-revalidation/` 下。

## 保留证据

原始数据保存在
`$JITTOR_LAB_ROOT/_state/native-fsdp-linear-recheck/20261015/`；最终测量
在 `tanh-final/` 中，与仅 Linear 的数组及所有更早的
probe 分开。`SOURCE.json` 绑定 34 个源码文件。`STAGES_COMPARISON.json` 为每个历史源码保留全部
80 个梯度行。最终审计针对
`tanh-final/artifacts/SHA256SUMS.json` 独立验证
324 个产物、21 份已完成日志与五个 runner，该文件的 SHA-256 为
`2b405f496e679f131d20ab493cf83dc8e830a3c5b24a44f6ba593bb4417d1556`。

原始 `ee24fac2b` 封存仍为
`9901d1ed7dd834dae4d11b4afdb501a37d2cdc3bc692019daad2d6a0c27c9a27`；
其 236 个产物、13 份日志与五个 runner 经独立重新哈希全部匹配。
中间仅 Linear 封存中的 312 个产物、五个 runner
与另外十二份日志也匹配。该历史封存对其取哈希时，它自己的 stdout 日志仍在追加；
这一处日志不匹配已被记录，两个原始文件都保留。最终封存只对已完成日志取哈希。

`FOLLOWUP_SHA256SUMS.json` 记录该历史封存注意事项，并将全部
801 个后续产物/日志/runner 文件绑定到其完整字节。其 SHA-256 为
`0589274e5982de851438c7134ecdd326ca2aa9ab7a184f66e4e95f38529b2195`。
新审计不覆盖原始张量证据或阈值。
