# 设计：原生 complex Var dtype（complex64/complex128）

> Task #5（用户优先级）。现状靠 `ComplexNumber`(real/imag 两个实 Var) 在 python 层仿真（`nn.py:3905-4119`）。目标：让 complex 成为 jittor_core 一等 dtype。多日深核心工程，守 G1（不破坏元算子/统一计算图）。
> 状态：🟢 **Phase 1 已实现 + 提交**（2026-06-26）：complex64 成为 jittor_core 一等注册 dtype（`NanoString("complex64")` dsize=8 / is_complex=True / 非 float 非 int），`test_torch_compat` 171/0 无回归。**关键发现：枚举移位安全**——dtype-vs-op 按名字判定非索引（`init_ns` 查 dsize_map/unary_ops/binary_ops），所以给 `FOR_ALL_NS` 加 dtype 不破坏 op 索引。complex128(16B) 需先扩 2-bit `_dsize` 字段（推迟）。
> 🟢🟢 **Phase 2 也已实现 + 提交**（2026-06-26）：**原生 complex64 算术可用**——`jt.array(复数numpy)` 创建 + numpy 往返 + `add/sub/mul/div/neg` 对 numpy 复数 **maxdiff=0.0（精确）**，CPU+CUDA，`test_torch_compat` **171/0 无回归**。实现：`complex_compute.h`(struct+运算符) + `complex_op_type.cc`(OpByType+post_pass 注入) + `dtype_infer` complex 规则 + numpy 映射(npy2ns/ns2npy) + **关键 bug 修复**：`py_array_op.cc` 的 64→32 自动窄化路径误伤 complex64(dsize 也是 8)，加 `!is_complex()` 排除。回归锁在 `test_complex64_native.py`。**Phase 2b 进展**（precise 边界，实测）：✅ **可用** = 创建/numpy 往返 / add·sub·mul·div·neg / **复数⊕标量** / **`sum`(reduce)**（加了 CUDA `atomicAdd(complex64*)` 重载，拆 real/imag）/ **`abs`→float32**（unary_dtype_infer abs(complex)→float + OpByType `jittor::jt_cabs`，限定名避开 codegen 改名）/ **`conj`**（新增原生 unary op `conj`：FOR_ALL_NS + nano_string/unary_op 的 unary_ops + 三个 OpByType——real/fp16=identity 符合 torch、complex=`jittor::jt_conj`；grad real 路径=identity·ones 已验，complex grad 走 `!is_float` 守卫返回 nullptr 延后）。/ **`matmul`（2D + batched，双卡全可用）**——*免费得来*：matmul 下沉到 elementwise multiply + sum-reduce，二者复数已实现，所以 `jt.matmul(复数,复数)` 2D/bmm CPU+CUDA 对 numpy `A@B` 精确（maxdiff~1e-6）。原 batched-CUDA 缺口已修：`nn.py` matmul 的 bmm 分支加 `"complex" not in str(a.dtype)`，复数绕过 `cublas_batched_matmul`(float-only) 走 reindex 路径。/ **mean(reduce)** 双卡（`reduce_dtype_infer` 加复数守卫，否则 mean∈float_ops 把输出推成 float64 编译崩）/ **prod** 仅 CPU（CUDA multiply-reduce 需 `atomicCAS(complex64)`，未实现，响亮失败）/ 结构算子(reshape/transpose/slice/getitem/concat/stack/broadcast)+比较(==/!=)+ternary 全双卡。/ **超越函数 `exp`/`log`/`sin`/`cos`/`sqrt` 双卡**（complex_compute.h 加 jt_cexp/clog/csin/ccos/csqrt 主支，complex_op_type map 限定名；这些 op 本就注册无需动枚举，只补 kernel）对 numpy 精确。/ **✅ autograd（Wirtinger，对真 torch 2.12 精确，双卡）**——`add/sub/mul/div/neg/conj/abs/matmul/exp/log/sin/cos/sqrt` 的 grad 全部实现并对拍真 torch maxdiff~1e-7。关键 3 处核心改：`var.cc` 复数不再自动 stop_grad（`!is_float && !is_complex`）、`var_holder.cc` start_grad 允许复数、`grad.cc` target 允许复数（loss 仍须 real-float，同 torch）；binary mul/div 用 torch 约定 `conj(other)`（real 路径保持字节不变），unary holomorphic 用 `dout*conj(f'(z))`，abs 用 `dout*z/|z|`。其余复数 unary grad 返 nullptr=响亮无梯度。❌ **仍缺** = `complex128`(需扩 `_dsize` 字段 2→3 bit，invasive)、prod-CUDA(atomicCAS(complex64))、tan/asin/sinh 等其余超越(未补 kernel→响亮 op-not-supported)。complex64 已**功能完整**(算术/归约/matmul/超越/结构/比较/autograd 双卡 torch 级)。

> 🔻 **ComplexNumber 废弃立项（2026-06-26，用户决策）**：native complex64 已功能完整，但 **FFT / 复数 linalg(inv·eig·eigh·qr·svd) / angle·polar·view_as_real·view_as_complex / gradfunctional / `torch.fft.*` shim 仍唯一依赖 `ComplexNumber`**。统一到 native、废弃 `ComplexNumber` 的有门槛迁移路线见本文末 **§Phase 6**。

**新增原生 unary op 的安全配方（conj 已验，171/0 无回归）**：加一个作用于 complex 的 unary 需同步改 6 处——(1) `nano_string.h` FOR_ALL_NS 在 dtype 块之后的 op 块加 `m(name)`（放 dtype 之后才不动 complex64=14 的 npy 映射）；(2) `nano_string.cc` unary_ops set 加 `"name"`（不进 float_ops/int_ops，则 real→real、complex→complex）；(3) `unary_op.cc` 的 unary_ops doc-list 加条目（自动暴露 `jt.name`/`.name()`）+ grad() 加分支（real 路径，complex 被 `!is_float` 守卫挡掉）；(4)(5) `common_op_type.cc` + `fp16_op_type.cc` 的 cuda_map & cpu_map 各加 real 语义；(6) `complex_op_type.cc` expand_op 加 complex kernel（限定名 `jittor::`）。漏 (4)(5) → real 输入该 op 直接编译崩（响亮，不会 silent）。

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

## Phase 2 原型已端到端实测（2026-06-26，跑通到只剩 1 个 bug）
**完整原型写过并实测**（后回退保 Phase 1 干净，因最后一个 bug 是 silent-wrong）：写了 `complex_compute.h`(struct+运算符)+`complex_op_type.cc`(OpByType+post_pass)+dtype_infer complex 规则+numpy 映射 → **全部编译通过（codegen 墙已破）**，且 **`jt.zeros((3,),"complex64").numpy()` 完全正确（[0j,0j,0j]）** → 证明 C++ 类型/OpByType/dsize/alloc/numpy-读取/init 全对。**唯一剩的 bug**：`jt.array(复数numpy)` 的 **ArrayOp memcpy（`array_op.cc:80` `memcpy(allocation.ptr, args.ptr, output->size)`）出垃圾**（第一个元素乱码、其余 0，像复制了错的字节数/源指针）——zeros 对说明 size/alloc 对，所以是 numpy 源 `args.ptr`（`py_array_op.cc` 设置处）或 complex 源数据解释的问题，需字节级调试（打印 args.ptr/output->size/numpy data）。**修掉这一个 memcpy → 原生 complex64 算术(add/sub/mul/div/neg)即全通**（算术 codegen 已写好待验）。下次直接接这里。

原始撞墙记录：Phase 1 后 `complex64*` 无 C++ 类型 → "identifier 'complex64' is undefined"。Phase 2 三块（按依赖序，已全部原型化）：
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

---

## Phase 6：ComplexNumber 废弃 / 统一到 native complex64（立项 2026-06-26，用户决策）

> **决策（用户）**：*"在已经有复数类型的情况下，ComplexNumber 应当废弃。"* native complex64 是 jittor_core 一等 dtype、天然在统一计算图内（守 G1/G5、对齐 torch 的单一 complex tensor）；`ComplexNumber`(real/imag 浮点对 `jt.stack([r,i],dim=-1)`，`nn.py:4095`) 是 **python 层漏抽象**——游离于图外，fusion / autodiff / 设备派发都要特判它。历史代价就是 `stack_acl` 把 `jt.stack` 覆写后**一次性拖垮所有 FFT/复数**，正因 ComplexNumber 底层是 stack（见 §历史 `7988cead`）。**终点：native complex64 成为唯一复数表示，ComplexNumber 删除。**

### ⚠️ 但这是有门槛的迁移，不能立即删 —— 现仍唯一依赖 ComplexNumber 的能力
（本会话 grep 全仓核对，native complex64 这些**全无**，与 native 已覆盖的算术/归约/matmul/超越/grad **互补不重叠** → 现在删 = 丢掉以下整块）

| 依赖方（file:line） | ComplexNumber 提供的能力 | native complex64 现状 |
|---|---|---|
| `linalg.py:13–1079` | 复数 **inv / eig / eigh / qr / svd**（`isinstance(x,ComplexNumber)` 派发：`complex_inv`/`complex_eig`/`linalg_qr`/`complex_svd`） | ❌ 无 |
| `nn.py:4086` + cufft | **fft2 / ifft2**（CUDA-only；⚠️ 见下 cufft 编译 bug） | ❌ 无 |
| `torch_compat.py:3729–3863` | `torch.fft.*`（DFT-matrix，双卡可微）返回 ComplexNumber + `fftshift`/`ifftshift` | ❌ 无 |
| `torch_shim/torch__init__.py:1019–1055` | `torch.fft.*`（numpy-wrap）返回 ComplexNumber | ❌ 无 |
| `gradfunctional/functional.py:27–395` | jacobian / vjp / jvp 的复数支持（特判 ComplexNumber） | ❌ 未接 |
| `ComplexNumber` 方法/自由函数 | **angle / polar / view_as_real / view_as_complex / tensordot / norm** | ❌ 无 |

> native complex64 已覆盖的互补另一半：创建/numpy 往返 · add·sub·mul·div·neg · 复⊕标量 · sum·mean·abs·conj · matmul(2D+bmm) · exp·log·sin·cos·sqrt · 结构(reshape/transpose/slice/getitem/concat/stack/broadcast) · 比较(==/!=) · ternary · 以及这些的 Wirtinger grad（均双卡 torch 级，见 line 5）。

### 迁移路线（桥接优先，杠杆全在第 1 步）
1. **view 桥接层（keystone，最便宜）**：native complex64 实现 `view_as_real`(complex64→float32 `[...,2]`) / `view_as_complex`(逆)，**零拷贝 reinterpret**（dsize 都是 8，按字节 reinterpret + reshape；参照 reinterpret/ArrayOp 路径）。**有它之后 native 即可内部复用现有 FFT/linalg/gradfunctional —— 后续 P3–P5 多为"重新接线"而非"重写算法"**，复数 svd/eig 这种硬骨头可继续复用 ComplexNumber 实现、只换入口。
2. **访问器**：`real` / `imag` / `angle` / `polar`（基于桥接，各几行）。
3. **FFT 迁移**：让 `torch.fft.*`（torch_compat 的 DFT-matrix 路径，已双卡可微）吃/吐 **native complex64** 而非 ComplexNumber。⚠️ **blocker（本会话实测发现，待确认/立账）**：原生 cufft 路径当前**编译失败**——生成的 `cufft_fft__*_op.cc` 报 `std::array<int,2> fft = {n1,n2}; incomplete type is not allowed`（codegen 缺 `#include <array>`），CUDA-only、与本次复数改动无关、疑预存 bug。修它，或 FFT 干脆只走 DFT-matrix 路径绕开 cufft。
4. **复数 linalg 迁移**：`linalg.py` 的 `isinstance(...,ComplexNumber)` 派发改吃 native complex64（经桥接复用 `complex_inv/eig/qr/svd` 内核）。
5. **gradfunctional + shim 迁移**：`functional.py` 与 `torch_compat`/`torch_shim` 改识别/返回 native complex64。
6. **废弃 + 删除**：`ComplexNumber.__init__` 加 `DeprecationWarning`，退化成 native 之上的薄 shim 保后向兼容；待无内部依赖后删类，测试从 `test_complex.py` 迁并入 `test_complex64_native.py`。

### 验收（每阶段双卡 + 不退化 `test_torch_compat` 171/0）
- **P1**：`jt.view_as_real(jt.array(复数)).shape==(...,2)`、`view_as_complex(view_as_real(z))` 往返 maxdiff=0（零拷贝）。
- **P3**：`torch.fft.fft(native complex64)` 返回 **native complex64**（非 ComplexNumber），对 `np.fft.fft` 精确。
- **P4**：`jt.linalg.svd(native complex64)` 对 numpy 复数 svd 对齐（前向 + FD 反向）。
- **P6**：构造 `ComplexNumber(...)` 触发 `DeprecationWarning`，旧 `test_complex.py` 仍全绿。

### 风险 / 规模
复数 linalg 反向（eig/svd）本身是硬专项；FFT 的 cufft `<array>` bug 需先清或绕过。整体属**多日深核心**，建议 P1→P6 串行、每步独立 commit + 双卡验证。**P1（view 桥接）是性价比最高的起点**：几十行解锁 P3–P5 的复用。
