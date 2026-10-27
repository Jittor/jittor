# CPU BF16 累加器与 LayerNorm

Status: 下文限定的算子与 Engine 检查已验证；维护中的 CPU/CUDA 双 rank 协议记分卡仍待
完成。Owner: Jittor 维护者。Reviewed: 2026-10-15。CPU 收缩、oneDNN matmul、
LayerNorm、autograd 快照或固定版本下游包变化后复查。

Code: `a2ca1b364c79c853623c4f65bf58b9232d7f4171`。同步后的干净测量：
`5f5ab7ecfee0a896c3565392cc9282559c9f8953`，已合入远端
`0751e425b0f197b8e7b388f218613378c71f701a`。既有的 MPI 前置条件保留其独立的
[报告与源码基线](2026-10-15-mpi-bfloat16-process-groups.md)。

## 变更与参照

CPU BF16 Linear 此前在加 bias 之前就对矩阵乘积做了舍入。
对 `1.0078125 * 1.0078125 + 0.00390625`，该实现产生 `1.015625`；独立二进制
Torch 产生 `1.0234375`。新路径保留 BF16 操作数，把 bias 加到 FP32 累加器上，
然后再做唯一一次 BF16 输出转换。

既有的 oneDNN matmul owner 现在为 BF16 操作数提供可选的 FP32 输出。其默认输出
与 FP32 路径保持不变。Descriptor 与 JIT 缓存区分输入/输出存储位宽，包括空的收缩
维度。反向保留 BF16 梯度与 native 高阶导数。在 `JT_BUILD_USE_MKL=0` 下，同一既有
收缩使用 native FP32 opmath。其自定义梯度递归复用该收缩，对完成的 GEMM 梯度做
舍入，而不是对每个广播贡献分别舍入。两条路径都不以 FP16 或使用 FP32 输入张量的
GEMM 作为替代。

通用 Function 在一阶 tape 之前保存原始 autograd 图的独立 raw-reshape holder。
实际的输入/权重 holder 重绑定、通过单个 Function 实例的重复调用以及全部三个二阶
导数，都与 28 个独立 Torch 观测逐位一致。标量二阶导数契约仍为 `4.0625`；产生
`2.03125` 的实验实现已被拒绝，不是生产代码。该快照契约覆盖图重绑定，而不是针对
任意物理存储写入的深拷贝。

对于 CPU 非 mixed BF16 LayerNorm，反向现在消费 BF16 保存的 mean/rstd 与 BF16
affine 部分和。前向保留未舍入的 FP32 统计量。混合 FP32 affine 参数与其他
dtype/设备路径保持既有行为。这遵循在
[Torch 的 CPU LayerNorm owner](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cpu/layer_norm_kernel.cpp)
中单独观察到的统计量与累加边界。选择依据是 native 操作数布局，而不是全局 CUDA 标志。

## 实际执行

平台：Linux x86_64、AMD EPYC 9554、GCC 11.5，每个 rank 四个 CPU 线程；
oneDNN 3.9.1，观察到 `brg_matmul:avx512_core_bf16` 的 BF16-to-FP32 primitive。
独立的 CPU 参考是二进制 Torch `2.11.0+cu128`，带有实际 `_C` 扩展且无 shim 标记。
CUDA 回归在 RTX4090/SM89 上执行，NVIDIA 驱动 580.178.04，使用真实的 pip-cu122/nvcc
12.2.140，并处于禁止 fallback 的策略下。这些检查不作计时或性能主张。

在干净的 `5f5ab7ec` 上，CPU ZeRO1/2 × oneDNN/native-opmath 执行真实 DeepSpeed
`0.19.7+f5af15c`，两个 rank，BF16 model 参数、AdamW、六个 microstep 与三次更新。
每个 rank 都更新全部 166 个 FP32 主分区元素；fallback 计数为零。保存/加载到新
Engine 并继续训练，在每个 runtime 内逐位相等：ZeRO1 每 rank 64 个严格字段，
ZeRO2 为 56 个。候选进程组是实际 MPI；参考进程组是 Gloo。
这不构成 native Gloo 支持。

在这四个诊断用例中，全部 1,792 个数值字段满足未变化的 state/forward `0.002` 与
gradient `0.01` 预算。全部 1,024 个请求的梯度字段都被检查；其中 1,012 个逐位相等。
ZeRO1 的梯度全部逐位相等；ZeRO2 的最大梯度误差为 `1.2822511817226889e-4`。
最大状态误差为 `2.1931527140484152e-7`。在 affine 部分和修复之前，ZeRO2 有两个
moment 字段为 `0.0025165948`，超出原预算；该失败保留在证据中，容差未变。

这些是 API 诊断，不是维护 gate 的准入。其独立 CPU 参考保留原始的
`f945b223cd4316feaa733556ef1413344f98e11f` 测量身份。完整配对协议仍需新的、匹配的
CPU/CUDA 维护测量、源码/安装绑定以及两个输入偏移。这里不授予任何公共记分卡分子。

同步后的干净门禁：

| 检查 | 结果 |
| --- | --- |
| CPU 新累加器/LayerNorm/公开 Linear 契约 | 12 passed，零跳过 |
| 实际 CUDA Linear/LayerNorm 契约 | 14 passed，3 项既有 xfail，零 fallback |
| Core，native 与 Torch 模式 | 322 passed, 42 skipped, 1 existing xfail |
| runtime 设备选择结构契约 | 7 passed |
| 仓库布局 | 通过 |

最终同步前的其他 owner 检查通过 native CUDA 53 例与 compat CUDA 91 例，零 fallback；
后者保留三项既有 xfail。五个独立 CPU LayerNorm shape 的前向逐位一致；逐字段缩放
梯度的最大误差为 `0.00502513`。BF16 affine 归约依赖行/线程分组，对任意 shape 不作
逐位主张。已测试的 LayerNorm 高阶导数保留一个 ULP 的差异，处于既有梯度预算内。
这些用例不推出整库 DeepSpeed、ZeRO3、offload、其他 optimizer、硬件或性能覆盖。

较早的完整结构运行有 1,502 通过、六项跳过与三项失败：MPI 报告索引/manifest 登记，
以及新引入的直接设备标志读取。后者已修复并完整重跑其七例契约通过；中心文档登记
仍属整合责任。

## 证据与复现

未版本化证据位于
`$JITTOR_LAB_ROOT/_state/refactor-goal/deepspeed/cpu-bf16/`。
`cpu-bfloat16-math-v1-evidence.json` 绑定 1,574 个 runtime 源码哈希与 105 个
原始/脚本/checkpoint 哈希，包括干净的用例目录、两个独立参考包、源码清单、被拒绝的
数值结果与门禁日志。其 SHA256 为
`80aa5e3b0a77699f954dba1f75b749a025c94ae62e3b97ea80b3244b6f75f81a`。
`seal-math-evidence-v1.py` 重算每个诊断比较与严格/F0 元数据检查。它不会把这些诊断
提升为协议准入。

在配置好独立的 native/参考解释器与显式可选 oneDNN 路径后，可复用的算子回归为：

```bash
JITTOR_TORCH_SHIM=1 PYTHONPATH=python <jittor-python> -m pytest -q \
  compat/tests/torch/test_bfloat16_cpu_accumulators.py \
  compat/tests/torch/test_bfloat16_layer_norm_cpu.py \
  compat/tests/torch/test_bfloat16_linear_rounding.py

JITTOR_TEST_DEVICES=cuda CUDA_VISIBLE_DEVICES=<allocated-device> \
  JITTOR_TORCH_SHIM=1 PYTHONPATH=python <jittor-python> -m pytest -q \
  compat/tests/torch/test_bfloat16_linear_rounding.py \
  compat/tests/torch/test_torch_compat_norm.py::TestLayerNorm
```

保持 CPU/CUDA 缓存分离，串行化首次构建，并保留独立的 rank home/临时目录。归档的
`run-math-clean-proof-v1.py` 捕获精确的任务本地启动环境与不可变源码检查。其机器
路径是证据，不是仓库默认值，也不替代维护中的 Engine launcher。

## 主树整合验证

整合到同步后的 `01351fe6174a44b688bb17ad3f5f7f0fd1cf340d`。
全部十个代码/测试文件与审查过的 `a2ca1b364` 源码逐字节一致；runtime 源码快照在
整个验证过程中保持不变。一项独立整合审计重新哈希原始 1,574 个源码与 105 个产物
绑定，并重算全部 1,792 个数值字段，包括 1,024 个梯度。原始容差、失败结果与测量
身份保持完整。

整合后的树在实际启用 oneDNN 的 CPU 算子契约（12 passed，零跳过）、禁用 oneDNN 的
Linear 路径（6 passed，零跳过）、实际 CUDA Linear/LayerNorm 回归（14 passed，三项
既有 xfail）、core（322 passed，42 skipped，一项既有 xfail）以及完整结构
（1,505 passed，六 skipped）上通过。全部改动 Python 文件的 Ruff 与仓库布局通过。
报告与源码 manifest 已在中心登记。

整合日志与源码检查未版本化，位于
`$JITTOR_LAB_ROOT/_state/cpu-bf16-math-integration-013/`；独立重算的诊断审计位于
`$JITTOR_LAB_ROOT/_state/cpu-bf16-math-integration-075/independent-audit.json`。
这些结果仍只确立算子正确性与有界的 Engine 诊断，不授予维护中的 DeepSpeed 协议准入。
