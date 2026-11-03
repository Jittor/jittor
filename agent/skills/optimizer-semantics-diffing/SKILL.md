---
name: optimizer-semantics-diffing
description: 定位梯度累积、裁剪、参数组或优化器状态差异时，用固定训练步逐项比较更新。
---

# 对拍优化器的逐步语义

## 固定输入与逐步证据

用相同初值、数据、loss、参数顺序及 dtype，在独立真实 Torch 解释器生成参考。
每步保存 loss、梯度、参数和优化器状态，至少连续四步以暴露动量、偏差修正和 step 计数错误。
最终 loss 相似不能替代逐参数更新正确。

- 用线性 loss 构造 microbatch 累积，显式比较 sum/mean 缩放与 zero_grad 时机。
- 区分 `None` 梯度和零张量；检查未参与 loss 参数、稀疏/空参数及冻结后的状态是否更新。
- 梯度裁剪记录调用次数与范数，检查是在累积前还是累积后执行。
- 参数组有效学习率按 `pg.get("lr", opt.lr)` 等实际继承语义计算，覆盖新增组、组内覆盖与默认值变化。
- Adam/AdamW 检查 epsilon 位置、偏差修正、解耦权重衰减与状态 dtype；SGD 检查 momentum 与 dampening。
- 更新量必须足够大以超过容差地板；注入零更新的阴性对照应被拒绝，避免大参数上的宽 atol 掩盖不更新。

CPU 先定位合同，实际目标设备再验数值和驻留。差异属于优化器、autograd 还是 compat API
先用最小算子分流，不能直接在下游 adapter 补偿核心错误。
