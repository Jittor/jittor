---
name: torch-checkpoint-stride-oracle
description: 修改 Torch checkpoint 读取或张量转换时，用真实 Torch 和独立 storage 解释验证布局、别名和 dtype。
---

# 验证 checkpoint 的 storage 与 stride

## 操作

- 在真实 Torch 环境生成 contiguous、transpose、带 offset 的切片、共享 storage 和非连续张量。
  将期望 dtype、shape、stride、offset、值及别名关系一同保存，不只保存最终 contiguous 数组。
- 分别通过真实 Torch 与被测加载器读取；再依据原始 storage 和元数据独立解释地址映射。
  两条路径若共用同一错误转换函数，不能构成独立参考。
- 比较每个值及必要的共享写入语义，覆盖零大小、标量、复数与支持的低精度类型。
- dtype 窄化必须符合明确接口合同，不能默认“数值差不多”就允许 float64 或整数精度损失。
- 验证损坏元数据和不支持格式能给出明确错误，不以 blanket exception 继续加载部分模型。

fixture 和原始大权重留在 lab；回归用最小可重建样本。模型完整加载成功不等于 stride 与别名正确。
