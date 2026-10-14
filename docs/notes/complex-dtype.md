# 复数 dtype

- 状态：已接受，限制已登记
- 上次复查：2026-09-22
- 本轮基线：`refactor-2.0` 的 `c7d9650a`（同步 origin 后）
- 复查触发：二阶复数自动微分、原生复数线性代数 kernel 落地、或 FFT complex128
  支持落地时

Jittor 的 `complex64` 与 `complex128` 都是**原生 dtype**，不是用两个实数张量模拟
出来的。这一页说明支持范围到哪里、哪些明确不支持。

## 支持的范围

CPU 与 CUDA 上有维护测试覆盖的部分（`complex64`、`complex128` 均适用，除非注明）：

- **构造与转换**：从 NumPy complex64/complex128 构造、零初始化、标量赋值、NumPy
  往返、实数与复数互转、`complex64`↔`complex128` 精度转换（含梯度）；
- **算术**：加减乘除、取负、与标量混合、相等/不等、三元选择；
- **提升**：`complex64` 与需要双精度的操作数（`complex128`，或 `float64`/
  `int64`/`uint64` 宽度的实数）混合时提升为 `complex128`；与更窄操作数混合时
  保持 `complex64`；
- **结构操作**：reshape、转置（含 CUDA）、索引、切片、广播、拼接、堆叠；
- **归约与线代**：`sum`、`mean`、`abs`、`conj`、二维与批量矩阵乘、
  `inv`/`svd`/`svdvals`/`qr`/`eigh`/`pinv`（经 `nn.ComplexNumber` 桥接，两种精度
  均验证）；
- **超越函数**：`exp`、`log`、`sin`、`cos`、`sqrt`；
- **实部虚部视图**：`.real`、`.imag`、`.angle()`、`view_as_real`、
  `view_as_complex`、`polar`（按输入分量宽度选择 complex64/complex128）；
- **标量提取**：`.item()` 对两种宽度都返回 Python `complex`；
- **一阶梯度**：上述算术、模长、矩阵乘、桥接操作、超越函数、精度转换 cast 的
  Wirtinger 型梯度；`svd`/`svdvals` 的联合 (U,S,Vh) 反向梯度（含非方阵、batch）；
- VJP 以及维护中的线性代数接口都接受并返回原生复数。

二阶梯度与 JVP 的支持范围**由参与的算子决定，不是整个 complex64 dtype 都不支持**。
上述基线的本轮 CPU 定向验证已覆盖：

- 0D 实/复桥接保留标量形状、模方实 loss 的两阶梯度，以及 `.item()` 返回 Python complex；
- complex64 的平方与 `exp` 基础 JVP、实输入产生复输出的 JVP；平方用例还覆盖标量/向量
  输出及 `create_graph=False/True`；
- 既有原生 complex64 VJP 回归。

证据为新增的 [标量与 JVP 合同测试](../../tests/autograd/test_complex64_scalar_contract.py)
及 [函数式 AD 测试](../../tests/autograd/test_complex64_gradfunctional.py)。新增标量合同的
6 项 CUDA 用例本轮也已通过，覆盖 0D 桥接/二阶梯度、`.item()` 和平方 JVP；上述既有
函数式 AD 文件本轮仅执行 CPU，不能将其 exp/VJP 覆盖直接外推为本轮 CUDA 已测。

## dtype 与梯度的不变量

- `complex64` 元素占 8 字节，`complex128` 占 16 字节；均被归类为**复数**，既不是
  浮点也不是整数。
- 复数转实数取实部；复数转 bool 时，任一分量非零即为真。
- `abs(complex64)` 返回 `float32`，`abs(complex128)` 返回 `float64`；算术与复数
  归约保持输入的复数宽度，混合宽度按上面的提升规则处理。
- 裸 Python/NumPy 复数**标量**（不依附于任何 Var）构造 Var 时窄化为
  `complex64`，与裸 Python `float` 窄化为 `float32` 是同一约定，不代表复数支持
  仅限 complex64——见 `tests/core/test_complex_scalar_operand.py`。
- 可导的复数值可以携带梯度，但**反向模式的标量 loss 仍然必须是实数**。
- 复数二元算子的梯度按 torch 的实 loss 约定对另一个操作数取共轭；全纯一元函数对
  局部导数取共轭。
- 未实现的梯度应显式报错，不得静默返回零；下面另列已确认的 SVD 回调数值缺口。

## 明确不支持

- **FFT 的 complex128 kernel 支持尚未接入。** `tests/opinfo/definitions/fft.py`
  的 `_NO_COMPLEX128` 跳过标记仍保留——dtype 注册本身不足以让 cuFFT/kernel 侧
  支持这个宽度，是单独的后续工作。
- **CUDA 上的复数 `prod`** 缺少所需的原子乘实现。CPU 已覆盖；CUDA 必须报错而不是
  返回部分结果。
- **原生复数 JVP** 依赖尚未实现的二阶自动微分，`jvp` 抛 `NotImplementedError`；
  原生复数 VJP 是支持的。二阶复数自动微分（含 `svd` 反向的反向）不在本次范围内。
- **复数线代回调的高阶导数**仍未实现：complex `inv`、`QR`、`SVD`、`eigh` 的不透明
  `numpy_code` backward 不能据基础算术的两阶梯度/JVP 通过而认定支持高阶。JVP 使用双重
  backward，组合到这些算子时仍受相同限制。
- **CUDA 上的一般复数特征分解**依赖可用的 CuPy 线代路径，在某些本来正常的 CUDA
  环境下可能不可用。
- 部分超越函数尚未支持。
- 复数 SVD 在奇异值重根/退化处的梯度是数学上的不可微边界，不是实现缺口。

限制只有在**配套一个覆盖此前不支持的后端或导数阶的定向回归测试**时才会被移除。

## 内部桥接

`view_as_real` 与其逆在设备上完成 `complex64[...]` 与 `float32[..., 2]` 的互转，
并保持梯度；0D 桥接与基础实 loss 的二阶 CPU/CUDA 回归见上面的定向证据，不能据此扩展到所有
复数算子的任意阶导数。这是**实现桥接，不承诺零拷贝别名**。

`jittor.linalg` 里仍有部分函数把原生复数转成内部的 `ComplexNumber` 实部/虚部表示、
跑既有的实数算法、再转回原生复数。`jt.nn.ComplexNumber` 只作为这类尚未重写的线代
算法的内部桥接保留，**新的公开 API 不得再引入第二种模拟复数表示**。该桥接对用户
可见结果已废弃，但要等所有这类 kernel 都有原生实现和等价测试后才能删除。

## 新增复数算子的检查清单

1. 独立于 kernel 代码指明输入提升规则与输出 dtype；
2. CPU 前向与 NumPy 或精确数学参考对拍；
3. 声称支持的设备上补 CUDA/NPU 执行与设备一致性覆盖；
4. 按实 loss 约定推导并测试一阶梯度；
5. 覆盖零值、分支切割、空张量、批量与非连续等情形；
6. 对不支持的导数阶或后端**显式报错**；
7. 更新本文与问题总账，不要把实验过程复制进任何一份。

主要回归文件（master 分支路径；refactor-2.0 上对应
[`tests/linalg/test_complex64_linalg.py`](../../tests/linalg/test_complex64_linalg.py)、
[`tests/core/test_complex128_native.py`](../../tests/core/test_complex128_native.py)、
[`tests/linalg/test_complex_spectral_gradients.py`](../../tests/linalg/test_complex_spectral_gradients.py) 等）：
[`test_complex64_native.py`](https://github.com/Jittor/jittor/blob/master/tests/type/test_complex64_native.py)、
[`test_complex64_linalg.py`](https://github.com/Jittor/jittor/blob/master/tests/linalg/test_complex64_linalg.py)、
[`test_complex64_gradfunctional.py`](https://github.com/Jittor/jittor/blob/master/tests/autograd/test_complex64_gradfunctional.py)、
以及覆盖剩余内部桥接的
[`test_complex.py`](https://github.com/Jittor/jittor/blob/master/tests/type/test_complex.py)。
