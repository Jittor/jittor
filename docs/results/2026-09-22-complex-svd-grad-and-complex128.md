# complex64 SVD 联合梯度（§3.2）与原生 complex128 dtype（§3.1）

- 状态：两项均已关闭；`tests/linalg/test_complex_spectral_gradients.py` 与
  `tests/core/test_complex128_native.py` 在 CPU 和真实 CUDA 上通过；
  已运行 `tools/run_test_suite.py --tier core` 与 `--tier smoke`；
  在更新一处已迁移边界计数后，`tests/structure` 通过。
- 日期：2026-09-22
- 负责人：本次会话
- 基线：`refactor-2.0` 于 `f34817660f239487d327c189fd4503da5e1c1f5e`
  （== `origin/refactor-2.0`，无需同步）
- 复查条件：FFT complex128 kernel 支持落地、二阶复数 AD 落地，或
  `src/type/nano_string.h` 的 `Flags` 位布局再次变化

## 问题

`local/problems-2026-9-22.md`（一份外部的 TensorCircuit 适配报告，不在本仓库
跟踪）列出了 9 个未解决问题。用户把本次会话范围限定为仅两个 P0 项：§3.2
（complex64 SVD 梯度在联合 U/S/Vh 损失下出错）与 §3.1（complex128 未注册为
原生 dtype）。其余 7 个 P1/P2 项明确不在本次会话范围内。

## 环境

Podman 镜像 `jittor-cuda:local`，由本仓库自己的 `Containerfile.cuda` 构建
（CUDA 11.8，Ubuntu 22.04）。容器 `jittor-p0-dev`，仓库绑定挂载在
`/usr/src/jittor`（宿主机上的编辑会即时反映到容器），
`JITTOR_HOME=/root/.cache/jittor-p0dev`（与任何其他并发工作隔离）。GPU：真实
NVIDIA GeForce RTX 4090（驱动 610.57.04，通过 CDI 可见的 3 张之一），计算能力
sm_89。容器内安装了 CuPy 13.6.0 以支持 CUDA 上的 `numpy_code`。`/opt/real-torch-venv`
处另有一个独立 venv（仅容器内，不属于仓库），装有真实上游 PyTorch 2.14.0+cpu，
用作真正独立的 oracle——本仓库自己的 `torch` 导入解析到 Jittor 的兼容 shim
（`jittor-torch`），而非真实 PyTorch，因此该 shim 不能作为独立参考。

## §3.2 — complex64 SVD 联合梯度

### 实际存在的实现 vs. 问题文档的描述

文档把一段微妙错误的 `(1-eye)` 掩码反向公式描述为“当前”实现。该公式只存在于
一个重构前的提交（`32001665`，可从 `master` 到达，从 `refactor-2.0` 不可达）。
在本分支上，`python/jittor/linalg/complex.py` 中 `complex_svd` 的
`backward_code` 直接是 `raise NotImplementedError`。因此这里要做的是“正确实现
反向”，而不是“修复已有公式”——在动手之前已用最小复现确认（对 complex64 输入
经 `jt.linalg.svd` 求 `jt.grad` 会抛出 `NotImplementedError`）。

### 修复

解析推导了复数 SVD 的联合伴随（不是猜的）：求解
`dC := U^H dA V = dP@S + diag(dS) - S@dQ`（`dP=U^H dU`、`dQ=V^H dV`，二者均为
反 Hermitian）的对角与非对角部分，可知实 SVD 反向中的非对角反对称化项
（`F ⊙ (utgu - H(utgu))`）原样适用，但对角项需要一个实 SVD 没有对应物的额外
项：复酉 `U` 只使 `U^H dU` 成为反*Hermitian*（对角线被迫为纯虚，而非零——旧的
错误提交直接删掉了这一项，而没有在 U/Vh 两分支之间正确组合它）。正确的对角项是
`i*(Im(diag(U^H gU)) - Im(diag(V^H gV))) / (2*S)`。

在编写任何 Jittor 代码*之前*，先用独立的 NumPy 有限差分 oracle 做了计算验证
（脚本用完即弃，不作为仓库产物）：方阵/高矩阵/宽矩阵、批处理、联合重建损失
（`loss = Re<(U*S)@Vh, P>`，期望 `dL/dA == P` 精确成立）以及仅 dS/dU/dV 的隔离
损失——全部匹配到约 1e-15，即有限差分自身的约 1e-9/1e-10 噪声下限。实现在
`python/jittor/linalg/complex.py::complex_svd`，保留现有的
`numpy_code`/`out_index` 结构（由 VJP 线性性证明其正确——每个输出的余切贡献
恰好可加，因此按分支分别以隐式零余切调用 `out_index`，与联合调用的总和相同；
无需 `jt.Function` 重构）。

### SVD 梯度验证

新增 `tests/linalg/test_complex_spectral_gradients.py`：联合重建梯度精确匹配
`P`（方阵/高矩阵/宽矩阵/批处理），匹配独立的 NumPy 有限差分 oracle，并匹配独立
的真实 PyTorch 进程（`torch.linalg.svd` autograd，经上述隔离 venv）——另加仅
dS/dU/dV 的隔离回归用例。CPU 与 CUDA 上 18/18 通过（最大误差约 4e-7 至 1e-6，
符合 float32 精度）。完整 `tests/linalg/`（46 个测试）仍通过，无回归。

## §3.1 — 原生 complex128 dtype

### 已确认的范围

硬性阻碍与范围所述完全一致：`NanoString::Flags::_dsize_nbits = 2`
（`src/type/nano_string.h`）把每个 dtype 上限压到 8 字节；complex128 需要
16。编辑前已验证：`jt.array(np.array(..., dtype=complex128))` 抛出
`RuntimeError: numpy.h:67: Numpy type not support, type_num: 15`。

### 改动（按依赖顺序）

1. `src/type/nano_string.h` — 将 `_dsize_nbits` 从 2 位扩宽到 3 位，相应地把
   `_white_list`/`_no_need_back_in`/`_no_need_back_out`/`_complex` 各上移一位；
   将 `complex128` 加入 `FOR_ALL_NS`；将四处硬编码的 `return ns_complex64`
   提升点替换为宽度感知的 `complex_promote`/`complex_needs_double` 辅助函数
   （若任一操作数为 complex128，或为 float64/int64/uint64 宽度的实数，则返回
   complex128，否则 complex64）。
2. `src/type/nano_string.cc` — 注册 `complex128`（`dsize_map=4`，即 16 字节）。
3. `src/type/complex_compute.h` — 新增 `struct complex128`（一对 `double`），
   镜像 `complex64` 的完整运算符/`jt_c*` 集合，包括 CUDA `atomicAdd` 分解，并新增
   显式的 `jt_c64_to_c128`/`jt_c128_to_c64` 精度转换辅助函数。
4. `src/type/complex_op_type.cc` — 注册 `"complex128"`；扩展 include guard；把
   cast 特例替换为区分复到实（丢弃虚部，不变）、complex64<->complex128（显式
   精度转换）与实到复（不变，使用目标结构体的转换构造函数）的实现。
5. `src/bindings/pyjt/numpy.cc`/`.h` — 将 `NPY_CDOUBLE` 映射到 `ns_complex128`
   而非 `ns_void`；新增对应的 `ns2npy` 条目。
6. `src/bindings/pyjt/py_converter.h` — 新增 `ns_complex64` 的 `.item()`
   `to_py_object` 分支（此前**完全缺失**——会撞上通用断言）以及 `ns_complex128`
   的新分支，两者都返回 Python `complex`。**没有**改动 Python 复数标量的窄化
   路径：`tests/core/test_complex_scalar_operand.py` 已经固定了裸 Python/NumPy
   复标量窄化为 complex64，这与裸 Python `float` 窄化为 `float32` 是同一约定——
   这是有意的设计，不是 complex128 的缺口，动手前阅读现有测试已确认。
7. `src/core/var_holder.h` — 将 `ItemData::data`（一个扁平的 8 字节 `int64`）
   扩宽为 16 字节 union，使 `.item()` 不会因 16 字节载荷溢出到相邻的 `dtype`
   字段。
8. `src/ops/composite/reinterpret_view_op.cc` — 为 complex128<->float64 镜像了
   complex64<->float32 的成对视图检查（`grad`、`infer_shape`）。
9. `python/jittor/nn/functional/complex.py` — 新增宽度通用的
   `_complex_to_real2`/`_real2_to_complex`（按分量宽度分派），供
   `view_as_real`/`view_as_complex`/`polar`/`.real`/`.imag` 使用；为遗留的
   `nn.ComplexNumber` 桥保留现有的固定宽度
   `_complex64_to_real2`/`_real2_to_complex64`（按
   `python/jittor/nn/legacy_complex.py` 的构造，始终为 float32）。
10. `python/jittor/linalg/complex.py` — 将五处硬编码的
    `assert dtype == "float32"` 检查（`complex_inv`/`eig`/`eigh`/`qr`/
    `pinv`）放宽为接受 float32 或 float64 分量对。
11. `python/jittor/linalg/_helpers.py::_cn_to_native` — **发现并修复一个真实的
    静默降精度 bug**：该处不管 `ComplexNumber` 的实际分量宽度都硬编码
    `_real2_to_complex64`，因此对原生 complex128 输入调用 `svd()`/`qr()` 等会
    静默返回 complex64。改为使用新的宽度通用 `_real2_to_complex`。
12. `src/ops/unary_op.cc::UnaryOp::grad` — **发现并修复第二个真实缺口**：复数
    分支没有处理复到复 `cast` 的用例（只有复到实），因此经
    `complex64.cast("complex128")`（或反向）求 `jt.grad` 会静默返回零。补上缺失
    分支（线性精度转换的伴随就是把余切转回原宽度）。
13. `src/ops/composite/transpose_op.cc` — **发现并修复第三个真实缺口，且在真实
    硬件上**：CUDA 的 cuTT 转置库根本无法规划 16 字节元素的置换
    （`cuttPlan failed ... dsize 16`），因此*任何* complex128 CUDA Var 的第一次
    `.transpose()`/`.T` 都会硬崩溃（这很普遍——linalg 桥中每次
    `_conj_transpose` 调用都会命中）。将 CUDA 加速转置能力查找限制在
    `dsize() <= 8`，16 字节 dtype 落回现有的通用 `Tx` 模板 kernel。
14. `backends/cuda/kernels/nn/complex_views.py` — 新增
    `COMPLEX128_TO_REAL2_CUDA_SOURCE`/`REAL2_TO_COMPLEX128_CUDA_SOURCE`
    （镜像现有 complex64 kernel；仅在 `reinterpret_view` 不可用的回退路径上
    执行）。
15. `docs/notes/complex-dtype.md` — 更新：complex128 移出“明确不支持”；记录新的
    提升规则与标量窄化约定；FFT complex128、CUDA 复数 `prod`、二阶复数 AD，以及
    退化奇异值 SVD 梯度记为剩余且明确划出范围的限制。

第 1–14 项没有一项是照抄历史提交 `7609dab5` 的精确 diff（文件布局不同——
`python/jittor/src/...` vs `src/...`——且 `refactor-2.0` 自己的 `_index_nbits`
修复后 `NanoString` 位布局也不同）；它只作为模式参考，每个改动点在动手时都对照
本分支当前实际代码重新核实。

### complex128 验证

新增 `tests/core/test_complex128_native.py`：16 字节 `dsize()`、无损 NumPy
往返、complex64/complex128 混合提升（以及较窄操作数*不会*强制扩宽）、算术 +
超越函数（确认 `abs`->`float64`）、一阶梯度、两种宽度下的 `.item()`、
complex64<->complex128 cast（值与梯度，双向），以及 complex128 精度下的 linalg
桥（`inv`/`svd`/`qr`/`eigh`/`pinv`）及重建恒等检查。CPU 与 CUDA 上 20/20
通过。`tests/core/test_complex_scalar_operand.py`（现有，docstring 更新以反映
complex128 落地，断言未改）仍 5/5。

### 门禁结果

- `tools/run_test_suite.py --tier core`：native 98 passed/41 skipped/1
  xfailed，torch 212 passed/3 skipped。0 失败。
- `tools/run_test_suite.py --tier smoke`：native 2162 passed/47 failed/228
  errors，torch 2782 passed/23 failed。失败/错误**不在**本次会话改动的文件中
  （MPI 多 rank 入口、ACL/Ascend 后端路由、RNN/cuDNN、TF32 精度域接线、DDP
  启动、`arg_pool`/`cumprod`/`einsum`/`unique` 的 `ModuleNotFoundError`——这个
  极简容器缺可选依赖，不是本次会话的代码）。抽查了唯一一个*确实*触及本次会话
  所读文件的簇（`tests/type/test_complex.py`，不在 `core` 层白名单中）：它的
  失败是该文件自身独立-Torch 检测中既有的 xdist 工作线程顺序问题
  （`tests/_helpers/torch_runtime.py` 的
  `modules_available`/`import_torch_modules` 在 `run_test_suite.py` 自身调用
  之外的 `-n 4` 下，收集时与 `setUpModule` 时结论不一致）——用本次会话的独立
  oracle venv 在场与缺席分别复现，每次都是两条依赖顺序的不同错误信息，确认不是
  本次会话的源码改动。
- 完整 `tests/linalg/`（46）与 `tests/core/`（446，一个既有、无关的
  `test_namespace_exports.py` 失败，关于 `dlpack`/`vmap`/`Generator` 的 stub
  声明——在问题文档中另行按 §3.4/§3.5 跟踪，此处未改动）通过。
- `tools/check_repo_layout.sh`：一处发现，`local/problems-2026-9-22.md`
  （用户自己的外部报告文件，本次会话之前就存在，引用了位于另一个
  repo/harness 的 `development-plan.md`）——不是本次会话创建的文件；留给用户
  处理。
- `JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest tests/structure`：
  收集 1378，修复前 3 失败，修复后 1 失败。已修复：本次会话在
  `reinterpret_view_op.cc` 新增的两个 `USER_CHECK`（complex128<->float64 形状
  配对检查）把一处受跟踪的已迁移边界计数从 8 变成 10
  （`tests/structure/core/test_error_categories.py`），按该文件已有的说明注释
  约定更新。未修复（既有、无关）：与上面相同的 `local/` 文档治理发现，以及
  `test_a_timeout_ends_the_grandchildren_too`（孙进程在超时后存活，本容器内的
  进程/信号处理问题，不是本次会话触及的文件）。

## 仍开放的内容

原始 9 项列表中的其余 7 项（SGD dampening、原生 vmap 批处理规则、DLPack
stride/空缓冲区处理、CUDA RNG checkpoint 重放、高阶复数 AD、函数级编译/缓存
契约、稀疏设备侧 coalesce）**没有**在本次会话处理——用户明确把本次会话限定为仅
这两个 P0 项。在 §3.1 自身范围内，明确仍开放（记录于
`docs/notes/complex-dtype.md`）：FFT complex128 kernel 支持（独立的
cuFFT/kernel 级工作）、CUDA 复数 `prod`、二阶复数自动微分（包括 SVD 自身的反向
之反向），以及 CUDA 上一般复特征分解（取决于 CuPy 可用性）。
