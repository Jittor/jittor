# CUDA BF16 Linear bias 累加

- Status: 已在 CUDA SM89 上验证；CPU 行为不变。
- Reviewed: 2026-10-15。
- Owner: Jittor 核心维护者。
- Synchronized baseline: `3890a4ae8c79767fe5251a1afae818308db4d447`。
- Implementation: `e64d4a27f282740ad9e496bbf0d1330ed4717a27` 与
  `e07a2e0469594b56680716516277f27a1f984635`。
- Recheck when: Linear dispatch、cuBLAS dtype 推断、AMP 输入转换或
  native matmul 求导变化。

## 问题与实现

可移植的 BF16 Linear 路径在加 bias 之前就对它的 GEMM 输出做了舍入。
对 BF16 输入与权重 `1.0078125`、bias `0.00390625`，该路径返回
`1.015625`；独立 PyTorch Linear 返回 `1.0234375`。后者在唯一一次 BF16
输出舍入之前把 bias 加到 FP32 累加器上。三种 CUDA shape 用相同的 BF16
操作数复现了这一差异。

注册的 CUDA `linear_bias_accumulator` provider 复用 native
`CublasMatmulOp`，以 BF16 为操作数、FP32 为累加器输出，在 FP32 中加上取整后的
BF16 bias，然后只做一次转换。新的 `fp32_output` 选项默认为 false，并显式拒绝
非 BF16 的计算 dtype。普通 FP32/FP16 matmul 行为仍由其现有测试覆盖。
native matmul 的反向保持可微，包括 `create_graph`；没有自定义反向公式或额外
adapter。Autocast 在运算前把输入与 bias 舍入为 BF16，并把梯度转换回每个原始
参数的 dtype。

Torch 模式下无 bias Linear 的 native Module 快速调用显式把新的构造函数参数传为
false。其 FP32、FP16 与 BF16 的推理和训练用例保持原有的结果与梯度 dtype。

## 验证

目标环境使用 RTX 4090、SM89、驱动 580.178.04，以及已安装的
`pipcu122_a89fa4fed08c` provider 与 nvcc 12.2.140。独立 oracle 是二进制
PyTorch `2.11.0+cu128`；其 `_C` 扩展已加载，Jittor 安装上下文标记不存在。
这是两套不同的 CUDA 库栈，而非对每个 CUDA 版本或架构的主张。

| 检查 | 结果 |
| --- | --- |
| BF16 CUDA `(rows, in, out)` = `(2,8,16)`、`(2,16,8)`、`(2,8,4)` | 输出与全部输入/权重/bias 梯度与独立 oracle 逐位相等 |
| 舍入中点、rank 1/2/3、module/autocast、畸形 shape、二阶导数、无 bias Module 推理/训练 | 6 项 CUDA 测试通过；独立二进制 Torch 的 CPU/CUDA 各自通过同样的六项数值契约 |
| Native 累加器选项、FP32/FP16 默认值、既有 Lt 路径 | 20 通过；provider 观测验证实际 BF16 操作数、FP32 输出请求、CUDA 驻留与零 fallback |
| AMP CPU / CUDA，包括六项 CUDA 契约 | 76 / 82 通过 |
| CPU core | 322 passed, 42 skipped, 1 existing xfail |
| 报告插入前的完整结构 / 仓库布局 | 1460 passed, 6 skipped / 通过 |

最终的构造函数后续修改重新成功运行了 CPU core。在新报告存在的情况下，完整结构报告
1458 passed、6 skipped 且恰好两项文档整合失败：该报告需要中心 results 索引条目与
生成的 MANIFEST 登记。这些共享文件由整合 owner 维护；计算与其他结构检查均通过。

六项结构 skip 包括两项依赖加速器的检查、两项已声明的用例，以及两项源码安装解析
检查。源码 checkout 解释器对后者没有已安装的 Jittor 发行元数据。

在仓库根目录、使用隔离缓存与已配置的 CUDA 工具链复现：

```bash
JITTOR_TORCH_SHIM=0 PYTHONPATH=python <target-python> -m pytest -q \
  tests/backends/cuda/test_cublas_matmul_amp_dtype.py tests/nn/test_lt_linear_cuda.py
JITTOR_TORCH_SHIM=1 JITTOR_TEST_DEVICES=cuda PYTHONPATH=python \
  <target-python> -m pytest -q compat/tests/torch/test_bfloat16_linear_rounding.py \
  compat/tests/torch/test_torch_amp_fidelity.py \
  compat/tests/torch/test_torch_amp_boundaries.py \
  compat/tests/torch/test_torch_amp_training_loop.py
```

原始日志与观测未版本化，位于
`$JITTOR_LAB_ROOT/_state/refactor-goal/deepspeed/`：`bf16-linear-*-final-v2.log`、
`bf16-linear-ctor-*.log`、
`bf16-linear-oracle.npz`，以及按 rank 区分的
`multirank/native/rank*/bf16-linear-accumulator-v1.npz`。

## DeepSpeed 与其余边界

`_ecosystem_deepspeed_cases.py` 中未改动的残差模型、真实 DeepSpeed
`0.19.7+f5af15c` 与两个独立的 CUDA rank 也执行了带 FP32 主分区的 ZeRO1 BF16
训练。六个 microstep、梯度累积为二、三次更新、保存、新 Engine 加载与续训，
每个 rank 产生 241 个观测。每个输入/目标梯度、传入参数梯度与重建的完整参数梯度
都与二进制 Torch 逐位一致。输出与 BF16 参数也逐位一致；
FP32 master/moment 差异至多 `1.49e-8`。native 严格的同 runtime 续训与零 fallback
检查通过。既有的 `2e-3` 前向/状态与 `1e-2` 梯度门限未变。最终注册的 provider
重复了该结果；日志使用 `bf16-registered-accumulator-v2`。

这项补充性的 ZeRO1 结果并不完成 DeepSpeed R14。同一双 rank ZeRO2 协议在零
fallback 下执行并续训，但完整梯度在归约后发散，而最初的本地传入梯度逐位一致。
这些失败的观测仍保留在 `bf16-accumulator-v2` 下；共享缓冲区改写是另一项独立调查。

CPU 保留原有的乘积/bias 分开舍入。一次 FP32 CPU 组合试验改善了中点前向与
第一梯度，但把二阶导数从 `4.0625` 变为 `4.09375`，因此被排除。独立 Torch 的显式
FP32 组合确认了后一个量化边界；它不等价于 native BF16 Linear 的反向。本报告
不主张任何 CPU BF16 精度修复或 ROCm/NPU 验证。

## 父级整合验证

父级在 `71c5be74ef6991b75fa79d73fe52eb04c0319d59` 上整合了全部四个 Linear 提交，
包括 native module 构造函数修正，并同时合入交付的 TP AdamW 与兼容 checkpoint
变更。全部 36 个交付文件在验证前与其最终源码引用逐字节一致。结果与生成的
source-manifest 登记已加入。

在该冻结的合并源码上，CPU core 通过 **322 项测试**，42 项已声明跳过与 1 项既有
xfail。native CPU owner 通过 **56** 项，15 项 CUDA 跳过；兼容 CPU owner 通过
**18** 项，排除了仅 CUDA 的 BF16 类。实际 CUDA 的 native Linear/cuBLAS owner
通过 **20** 项，实际 CUDA 兼容 owner 通过 **24** 项，均零跳过。完整结构通过
**1470** 项，六项跳过。这是正确性验证，不作速度结论。较早的独立双 rank 数值证据
保留其原始源码标签。原始整合日志、源码等价性与产物审计未版本化，位于
`$JITTOR_LAB_ROOT/_state/distributed-bf16-integration/`。
