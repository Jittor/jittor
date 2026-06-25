# 设计：原生 complex Var dtype（complex64/complex128）

> Task #5（用户优先级）。现状靠 `ComplexNumber`(real/imag 两个实 Var) 在 python 层仿真（`nn.py:3905-4119`）。目标：让 complex 成为 jittor_core 一等 dtype。多日深核心工程，守 G1（不破坏元算子/统一计算图）。
> 状态：🟢 **Phase 1 已实现 + 提交**（2026-06-26）：complex64 成为 jittor_core 一等注册 dtype（`NanoString("complex64")` dsize=8 / is_complex=True / 非 float 非 int），`test_torch_compat` 171/0 无回归。**关键发现：枚举移位安全**——dtype-vs-op 按名字判定非索引（`init_ns` 查 dsize_map/unary_ops/binary_ops），所以给 `FOR_ALL_NS` 加 dtype 不破坏 op 索引。complex128(16B) 需先扩 2-bit `_dsize` 字段（推迟）。**Phase 2+（codegen/ops/grad）待续**。

## dtype 系统落点（精确 file:line）

| # | 文件 | 行 | 改动 |
|---|---|---|---|
| 1 | `src/misc/nano_string.h` | 15-30 | `FOR_ALL_NS` 宏加 `m(complex64) m(complex128)` |
| 2 | `src/misc/nano_string.h` | 147-148 | `is_complex()` 现恒返回 false → 按 index 区间正确实现 |
| 3 | `src/misc/nano_string.h` | 205-234 | `dtype_infer()` 支持 complex 混合：complex OP float → complex |
| 4 | `src/misc/nano_string.h` | 237-265 | `binary/unary_dtype_infer()` complex 逻辑（比较→bool，abs→实数） |
| 5 | `src/misc/nano_string.cc` | 12-23 | `FOR_ALL_TYPES` 宏加 complex64/128 |
| 6 | `src/misc/nano_string.cc` | 181-237 | `init_ns()` 配置 complex 的 dsize(64→8B/128→16B)、is_float=0、is_complex 标志 |
| 7 | `src/var.h` | 30-31 | 加 `is_complex()` inline |
| 8 | `src/type/complex_op_type.cc` | 新建 | `ComplexOpType : OpByType`，`expand_op()` 生成 complex kernel（CPU `std::complex<T>`；CUDA 自定义 struct/cuComplex），`post_pass` 注入 `#include <complex>` |
| 9 | `src/type/complex_compute.h` | 新建 | complex kernel 辅助函数（conj/abs/...） |

参照现有：`src/type/common_op_type.cc`（实数 OpByType + cpu_map/cuda_map kernel 模板）、`src/type/fp16_op_type.cc`（按 dtype 定制 kernel 的范例，如 `::__habs`）。binary kernel codegen 在 `ops/binary_op.cc:555-571`（`@expand_op(@OP,@Tz,xp[i],@Tx,yp[i],@Ty)`）。

## MVP 算子集（Phase 1-2 优先）
创建（zeros/from real+imag/view_as_complex）、real/imag、conj、加、减、乘、abs(→实数)。多数可复用实数 kernel 分量计算。**梯度先禁用**（grad() 里 is_complex → 响亮 NotImplementedError），Phase 5 再补 Wirtinger 导数。

## C++/Python 分工
- **C++ 必须**：dtype 注册（nano_string）、codegen（complex_op_type）、Var.is_complex。
- **Python**：`__init__.py` 的 complex64/128 常量 + 创建函数 dtype 支持；`nn.py` 便利函数；ComplexNumber 保留、逐步迁移。
- **双卡 kernel**：CPU 用 `std::complex<T>`（运算符重载齐全）；CUDA 用自定义 struct 或 thrust::complex（cuComplex 只有基本四则）。

## 分阶段
- **Phase 1**（1-2d）dtype 核心注册 → 编译通过、`jt.complex64.dsize()==8`、`is_complex()==True`。
- **Phase 2**（2-3d）codegen → complex 张量四则+conj 能跑（先 CPU 后 CUDA）。
- **Phase 3**（1d）Python API（常量/创建/便利函数）。
- **Phase 4**（1-2d）G1 防御：grad 禁用、dtype 推导规则、融合排除 complex。
- **Phase 5**（迭代）超越函数（exp/log/sqrt/sin）+ 完整梯度。

## G1 风险与防御
- dtype 推导链：complex+float→complex（显式规则，回退原逻辑）。
- 梯度：先禁用（响亮报错）而非静默错。
- 融合：complex 操作排除出 FusedOp，保持显式。
- reduce：abs(complex)→float、sum(complex)→complex，在 `reduce_dtype_infer` 分类处理。

## Phase 2 实测铺路（2026-06-26，已撞墙定位）
Phase 1 后实测：`jt.zeros((3,),"complex64")` 能创建（dsize 生效），但**任何 op（连 .numpy() 的 broadcast_to copy）生成的 kernel 用 `complex64*` 作 C++ 类型 → nvcc/g++ 报 "identifier 'complex64' is undefined"**——jittor 没有 complex64 的 **C++ 类型**。Phase 2 三块（按依赖序）：
1. **C++ 类型**：新建 `src/type/complex_compute.h` 定义 `struct complex64 { float real, imag; }` + 运算符（`operator+ - *`，其中 `*` 是复数乘 `(ar*br-ai*bi, ar*bi+ai*br)`、`==`、`abs`→float）；CUDA 版用 `cuComplex`/`float2` 或同 struct（`__host__ __device__`）。参照 `src/type/fp16_compute.h`（float16 同样非原生、定义了 struct + typedef）。
2. **codegen 包含它**：让生成 complex64 kernel 时 include `complex_compute.h`——参照 fp16 的注入机制（`fp16_op_type.cc` 的 post_pass / op_compiler 按用到的 dtype 注入 header）。先让 **copy 类 op（broadcast_to/reshape/getitem）编译过** → 解锁 `.numpy()` 往返。
3. **OpByType 算术**：新建 `src/type/complex_op_type.cc`（`types={"complex64"}`，`expand_op` 用 `(($2)+($4))` 等靠 struct 运算符；或显式复数公式），注册进 op 派发。binary/unary 才能跑。
4. **numpy 桥**（已知确切改动）：`numpy.cc` 的 `npy2ns[14]=ns_complex64`（NPY_CFLOAT）+ `ns2npy` 末尾加 `NPY_CFLOAT`（complex64 的 ns.index()=14）。**注意：单独加这个会让 complex64 可创建但 op 崩**（比清晰报错更糟），必须配合上面 1-3 一起上，别单独提交。
5. grad：先在 binary/unary grad 里 `is_complex()→NotImplementedError`（响亮、非静默）。
**复用**：现有 `ComplexNumber`(real/imag python 仿真) 仍可用、可作对拍 oracle 验证原生路径。

## 验收
```python
jt.complex64.dsize()==8 and jt.complex64.is_complex()   # ✅ Phase 1 已达
z = jt.zeros((3,), dtype="complex64"); (z+z).dtype=='complex64'  # Phase 2 目标
# complex+float→complex；backward 暂报 NotImplementedError（不静默）
```
