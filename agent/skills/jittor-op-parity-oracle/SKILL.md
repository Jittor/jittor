---
name: jittor-op-parity-oracle
description: 定位原生或 compat 算子、梯度及公开 API 的数值差异，构造独立参考并拒绝缺失、非有限或形状错误的结果。
---

# 建立算子与梯度的独立参考

## 固定比较合同

- 将输入、权重、dtype、设备、随机种子、loss 和梯度目标保存为固定 fixture。
  原生 Jittor、compat 与真实 PyTorch 使用独立解释器，并检查每个进程的实际导入来源。
- 逐项核对输出与所有请求的梯度：缺失、形状不符、非有限值分别失败，不能只打印后返回成功。
  明确允许断开的输入，区分“无梯度”与数学上的零。使用逐元素绝对/相对误差；全局归一化误差仅作诊断。
- 先验证前向，再验证一阶/需要时的二阶导数。loss 使用非均匀种子，避免全一求和掩盖置换错误。
- 独立 NumPy float64 参考和少量坐标中心差分可定位反向错误。`tests/_helpers/gradcheck.py`
  仍调用 Jittor 前向，前向未证实时不能当独立 oracle；参考 `tests/ops/test_gradcheck_oracle.py`。

## 易漏语义

- 输入污染：先复制输入，强制输出执行，再重新读取输入；交换读回顺序，并加入基本切片 view、
  高级索引 copy 和非 view 对照。延迟执行可能把一次原地写错误藏到下一次读取。
- `reindex` 与 `reindex_reduce` 的输入/输出形状角色相反；检查映射的方向和边界。
- `numpy_code` 在设备模式下可能使用 CuPy，不凭接口名宣称 CPU；QR/SVD 等分解用重构、
  正交性及符号/相位不变量比较，避免把等价分解误判为错误。
- float64 fixture 要显式指定 dtype；确认未被默认窄化。fast-math 下 NaN/Inf 的传播需单独覆盖。
- 公开 API 扫描区分“名字可达”“签名可用”“数值正确”，空集合和任意异常跳过都不构成验收。

## 梯度静默与图查询

检查真正计算、对外暴露和图断开三种来源。无 warning 不是梯度完整的证据；比较重建前向、
重复 backward、retain_graph、共享中间量与断开输入。改变缺失梯度默认语义前跑受影响用例，
不能将全部未使用参数统一变为错误。

图查询应与 `src/core/grad.h` 的反向可达性保持一致，同时检查：

1. Var 的 stop_grad 与 requires_grad_disabled；
2. 生产者算子的 stop_grad（detach 可能停在算子上）；
3. 负索引控制依赖边不传播梯度；
4. 构图时冻结的 requires-grad-disabled 边。

同时断言查询结果和实际梯度；native 与 Torch 的 requires_grad 默认值差异应单列，不能用
伪造 is_leaf 去补偿。现有入口为 `tests/core/test_backward_leaf_query.py`、
`src/tests/test_backward_leaf.cc`。用遍历计数器增量与嵌套 epoch 的标记完整性检查属性查询成本。

崩溃/挂死复现使用 `tests/_helpers/child_process.py` 的受控子进程。
CPU 结果不能外推 CUDA/NPU；目标设备需验证真实驻留与无回退。
