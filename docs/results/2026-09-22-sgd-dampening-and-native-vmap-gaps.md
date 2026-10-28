# SGD dampening（§3.3）与原生 vmap 批处理规则缺口（§3.4）

- 状态：§3.3 在 CPU portable、CUDA fused 与 FSDP2 上均已关闭（全部对照真实
  CUDA 上的独立真实 PyTorch 进程验证）；ACL 仅凭代码推理修复——本次会话没有
  昇腾硬件，未验证。§3.4：`cumsum` 与 `solve` 获得真实批处理规则并做了完整组合
  测试；`numpy_code` 获得清晰的快速失败；稀疏已能干净失败（已加锁，未改动）；
  “惰性 patch 安装”一项做了调查，实现了修复，随后因它破坏了更强的、有显式测试
  的项目不变量而回退——见下文。
- 日期：2026-09-22
- 负责人：本次会话
- 基线：`refactor-2.0` 于 `f34817660f239487d327c189fd4503da5e1c1f5e`
  （自今早更早的 P0 会话以来未变）
- 复查条件：ACL/Ascend 硬件可用时（§3.3 需要在那边做真实验证），或
  `python/jittor/linalg/solving.py` 的输出形状处理变化时（见下文新发现的、
  超出范围的 `solve()` 缺口）

## 环境

与今早更早的 P0 会话相同的 `jittor-p0-dev` podman 容器（`Containerfile.cuda`，
CUDA 11.8，3x RTX 4090，`JITTOR_HOME=/root/.cache/jittor-p0dev`），外加同一个
`/opt/real-torch-venv` 独立 PyTorch 2.14.0+cpu venv，复用于 SGD oracle 测试。
此环境无 Ascend/NPU 硬件——ACL 侧 SGD 修复仅据代码/系数数学推理实现，明确未在
真实硬件上运行。

## §3.3 — SGD momentum/dampening

### 根因，编辑前用复现确认

`python/jittor/optim/algorithms/sgd.py:16` 把无 momentum 快路径的门槛设为
`momentum==0 and dampening==0`，因此任何其他组合——包括 `momentum==0` 且
`dampening!=0`——都会对构造时始终零初始化的 buffer 执行
`velocity = momentum*velocity + dp*(1-dampening)`，而任何地方都没有“这是否是
第一次更新”的信号。动手前复现了文档中的精确数字：`p=1`、固定 `g=2`、`lr=0.1`、
`dampening=0.2`，3 步在 `momentum=0` 下得到 `[0.84, 0.68, 0.52]`，在
`momentum=0.8` 下得到 `[0.84, 0.552, 0.1616]`——两者都与文档报告的错值完全
一致。

### 修复

与 PyTorch 的实际规则一致：momentum buffer 在首次触碰时由原始（未 dampen 的）
梯度播种，dampened 递推只从第二次更新起适用。向此前只认识
`(momentum, dampening, nesterov)` 的每一层传入 `step`（从 1 开始）参数：

- `sgd_update(..., step=1)`（portable/CPU，FSDP2 也直接使用）：快路径现在是
  `momentum==0 and not nesterov`（没有 momentum 时 dampening 无关——这正是
  bug，而不是缺了某个分支）；`step<=1` 用原始 `dp` 播种 velocity，否则走既有
  递推。
- `SGD.step()`：每个 param group 调用一次 `n = self._advance_step_count(pg)`，
  复用 `Optimizer._advance_step_count`——Adam/AdamW 已用于偏置校正的同一机制
  （`python/jittor/optim/base.py:143`），它通过
  `state_dict`/`load_state_dict` 持久化，因此恢复的 checkpoint 不会“免费”再次
  被当作第 1 步。
- `backends/cuda/kernels/optim/fused_sgd_cuda.py`：同样的 `plain` 条件修正，
  另在生成的 CUDA 源码中加入一行以 `first_step` 为条件的 velocity 更新行
  （系数仍按该文件既有设计保持编译期字面量——代价是每个参数组在 step-1→step-2
  转换时多一次 JIT 编译，而不是每步重编译）。
- ACL：`FusedSgdAttr::isFirstStep` **早已存在**为字段
  （`backends/acl/include/acl_jittor.h:55`），但在 op 分派处被硬编码为 `false`
  （`backends/acl/src/acl_op_exec.cc:731`，其注释恰好断言了这个 bug 所依赖的
  错误假设），且 `fused_sgd_op_acl.cc` 的系数计算从不按它分支。修复：把来自
  `SGD.step()` 计数器的真实 `first_step` bool 经 `FusedSgdOp` 构造函数传到
  ACL 分派处，并拆分系数计算，使只有 velocity buffer 自身的递推系数在第一步
  归零（Nesterov 校正自身的 `-lr*momentum` 项不变——PyTorch 在那里始终使用
  真实 momentum，无论是否第一步，因为到那时 buffer 已持有正确值）。
- `compat/fsdp2/optimizer.py::_sgd_update_for_param`：接受 `step`，由调用方
  已有的 `param_steps[i]` 计数器传入——正是已经传给并行 Adam 路径 `n_step` 的
  同一个计数器，无需新增状态跟踪。

### 验证

- portable 与 CUDA fused 路径：直接复现文档的 3 步配方，现在在真实 CUDA 上
  精确匹配文档期望的 `[0.8, 0.6, 0.4]` / `[0.8, 0.48, 0.064]`。
- 新增 `tests/optim/test_sgd_dampening_pytorch_oracle.py`：同一 3 步配方对照
  真实、独立的 `torch.optim.SGD` 进程做交叉检查（不是本仓库自己的 torch-compat
  shim，那只是 jittor 自己实现换了个名字）——portable CPU、portable CUDA 与
  CUDA fused 全部在 `atol=1e-5` 内匹配。
- `tests/optim/test_optim_core.py::TestSGDUpdate`：修正参考公式（`_ref_sgd`）
  以接受 `first_step` 标志——它此前断言的正是错误的第一步 dampened 值；新增
  `test_sgd_dampening_two_steps`（dampening 从第 2 步起生效）与
  `test_sgd_momentum_zero_ignores_dampening`（文档提到的另一症状）。
- `compat/tests/fsdp2/test_fsdp_optimizer_math.py`：现有的
  `test_sgd_momentum_and_nesterov` 有意从非零 velocity buffer 出发（测试递推
  路径）——显式加上 `step=2`；新增
  `test_sgd_momentum_first_touch_seeds_from_raw_gradient` 覆盖配套的首次触碰
  场景。
- ACL：据代码推理修复（系数数学在 diff 本身中推导并记录）；**未在真实昇腾
  硬件上运行**——本次会话无 NPU 可用。这是真实、开放的验证缺口，不是悄悄宣称
  的通过。
- 完整 `tests/optim/` + `compat/tests/fsdp2/test_fsdp_optimizer_math.py`：
  67 passed（取消选择 1 个既有失败，见下文；8 个 error 是既有的“独立 Torch
  未预加载” oracle 会话上下文要求，这些特定文件需要
  `tools/run_test_suite.py` 自身的调用方式，已通过在更早 P0 会话中在无关联
  文件上复现同一错误模式确认无关）。

### 发现的既有、无关 bug，已排除

`tests/optim/test_optim_core.py::TestOptimizerPlumbing::
test_low_precision_parameter_and_state_dtypes_are_preserved` 触发了一个
Jittor 执行器内部不变量崩溃（`exec_plan.cc:271: queue.size()(14) ==
roots.size()(16)`）。通过 `git stash` + 完整 core 重建确认，它在真正未修改的
基线上完全相同地复现——既有、与本次会话无关，未继续追查。上述回归运行中保持
取消选择；未改动。

## §3.4 — 原生 vmap 批处理规则缺口

### 架构回顾

`python/jittor/vmap.py`：一个 `BatchedVar(value, level)` 包装器，加一张手工
维护的 monkeypatch `jt`/`jt.Var`/`jt.linalg` 入口表，由幂等的
`install_batching_patches()` 安装。经直接探查确认：`solve`、`cumsum`、稀疏
算子与 `numpy_code` 的批处理规则覆盖为零（各自目前以不同、大多令人困惑的方式
失败）；compat 的 `real`/`imag` 别名不认识 `BatchedVar`。

### 已修复

1. **`cumsum`** — 新增 `_apply_cumsum`，与现有
   `_apply_reduce`/`_apply_transpose` 规则同为轴敏感的单输入形状（通过现有
   `jt.misc._cumsum_dim` 按*逻辑* rank 解析 `dim=None`，按嵌套深度偏移，再
   委托）。注册为自由函数 `jt.cumsum` 与 `BatchedVar.__getattr__` 方法用例
   （`.cumsum(dim)`）。
2. **`solve`**（双输入）— 新增 `_apply_solve`，与 `_apply_binary` 同为最大
   level 分派形状，不带 `_align_logical_rank` 的 rank 填充（不需要：
   `np.linalg.solve` 自身的前导维广播已能处理一个操作数共享/未批处理的情形，
   `_apply_linalg` 的单输入规则同样依赖这一性质）。注册到 `jt.linalg.solve`。
3. **`numpy_code`** — 不是批处理规则（任意不透明回调无法在不逐 batch 元素
   调用的情况下被 vmap，这违背 vmap 的全部意义），而是当 `BatchedVar` 到达时
   给出清晰、显式的 `NotImplementedError`，而非令人困惑的低层 pybind 类型
   错误。
4. **稀疏算子** — 已能干净失败（`SparseVar.__init__`/`spmm` 硬性
   `isinstance(x, jt.Var)` 检查并在任何 op 运行前抛出 `TypeError`）。无代码
   改动；新增一个锁定测试，因为此前没有。
5. **`compat/torch/installers/numerical/complex.py` 的 `real`/`imag`** — 一个
   真实的静默错答 bug：`isinstance(input, (jt.nn.ComplexNumber, jt.Var))` 不
   认识 `BatchedVar`，因此 `torch.real(bv)` 静默原样返回 `bv`（被误标为“实
   部”），`torch.imag(bv)` 即使对真正为复数的 `bv` 也静默返回全零。修复方式是
   把 `BatchedVar` 加入检查——原生 `BatchedVar.real`/`.imag` 本就正确，这纯粹
   是 compat 别名的识别缺口。

### 调查、实现后又回退：惰性 patch 安装的“陈旧引用”缺口

文档的担忧：`install_batching_patches()` 只在 `vmap()` 自身函数体内运行，因此
在一个进程首次调用 `vmap()` 之前保存的函数引用（`my_qr = jt.linalg.qr`）会
永远绑定到 patch 前的函数，即便 `vmap()` 后来在其他地方 patch 了
`jt.linalg.qr`。

实现了“显而易见”的修复——在 `python/jittor/__init__.py` 末尾无条件调用一次
`install_batching_patches()`——结果它**破坏了两个结构契约测试**：
`tests/structure/nn/test_nn_structure.py::TestPublicBindings::
test_tensor_method_bindings_stay_on_public_functions`（断言
`jittor.Var.matmul is nn.matmul`）与
`tests/structure/runtime/test_runtime_composition_structure.py::
test_compat_composition_keeps_native_core_implementations_available`（断言
`jittor.grad is core_api.grad`）。本项目自己的别名组合系统
（`_runtime/composition.py` 的 `make_inplace_alias`/`publish`）让这些函数中的
每一个在所有入口点（`jittor.matmul`/`jittor.Var.matmul`/`nn.matmul`/
`__matmul__` 等）都拥有*单一共享的对象身份*。急切安装只会 patch 这些别名位置
中的*一部分*（`jt.Var.matmul`、`jt.matmul`），而不 patch 另一些
（`nn.matmul`），从而对**每一次** jittor 导入都破坏该身份，而不只是 vmap
用户——一个更宏观、有显式测试的不变量，不应为一个狭窄的 patch 时机修复而
牺牲。若也要 patch 每个别名位置，就意味着要把修复贯穿整个组合系统，这远超
vmap 范围。

**已回退。** 残余风险恰好就是 `vmap()` 自己的 docstring 已记录并绕开的内容
（优先用 `vmap(lambda x: jt.linalg.qr(x))` 而非裸函数引用）——现在由
`tests/core/test_vmap.py::TestVmapPatchTiming::
test_reference_captured_before_first_vmap_call_stays_unpatched` 锁定，它同时
断言已记录的失败与已记录的绕法，而不是断言一个试过、发现代价大于收益的修复。
此项**未关闭**——在此记录为已调查并有意保持原样，而非悄悄宣称已修复。

### 途中发现的第二个无关真实 bug，已修复

在测试（后来回退的）急切安装改动时，`vmap.py` 自身 patch 注册中另外两个真正
既有、潜伏的 bug 浮现，并独立于急切安装回退被修复（无论安装时机如何它们都是
真的，只是此前除非进程恰好在触及受影响代码前调用过 `jt.vmap()`，否则不可达）：

- **`_apply_unary` 的原始函数偏好**：注册的是 `orig =
  getattr(jt.Var, name, None) or getattr(jt, name, None)`（Var 方法优先），
  而不是 `_BINARY_NAMES` 注册已在用的自由函数优先顺序（且已记录其理由）。裸
  Python float 参数——例如 `adam_update` 的 `jt.sqrt(1 - b1**step)` 偏置
  校正——在 Var 方法描述符形式下失败（`TypeError: descriptor 'sqrt' ...
  doesn't apply to a 'float' object`）。改为优先自由函数，与二进制注册的既有
  约定一致。
- **`view`/`reshape` 与 `broadcast`/`broadcast_var` 被静默别名到同一个原始
  实现**：`jt.Var.view is jt.Var.reshape` 与 `jt.Var.broadcast is
  jt.Var.broadcast_var` 均为 `False`（已确认），但注册把*同一个*包装闭包
  （只捕获 `jt.Var.reshape` / `jt.Var.broadcast` 作为
  `_ORIG_RESHAPE`/`_ORIG_BROADCAST`）安装到了每对中的两个名字上——静默把
  `.view()`/`.broadcast_var()` 调用改走*另一个*方法的原始实现。具体破坏了
  `expand().view()` 的存储别名不变量
  （`tests/core/test_storage_strides.py::
  test_expanded_view_and_explicit_dense_library_boundary`，`expand()` 把结果
  标记为输入的 `_set_storage_view_of`，只有真正的 `.view()` 调用——而非顶替
  它的 `.reshape()`——才能检测并保留该关系）。是在本次会话已在运行的同一个完整
  `tests/core/` 回归扫荡中发现的；确认为既有（经 `git stash` 在真正未修改的
  基线上复现），但直到 `install_batching_patches()` 在（后来回退的）急切安装
  实验中无条件运行才变得可触发。修复方式是给
  `_apply_reshape`/`_apply_broadcast` 显式 `orig` 参数，并分别捕获/使用
  `_ORIG_VIEW`/`_ORIG_BROADCAST_VAR`。**保留此修复**（它正确且无论安装时机都
  必需——任何在进程内任意 `vmap()` 调用之后恰好调用
  `.view()`/`.broadcast_var()` 的代码路径本就已暴露）；确认本文件中所有其他
  同形多名注册（`transpose`/`permute`、`matmul`/`__matmul__`、
  `jt.X`/`jt.Var.X` 自由函数对）都是安全的——每对确实就是同一个底层对象。

### 测试（`tests/core/test_vmap.py`）

`test_cumsum_batching_rule`：对照 `np.cumsum`，非默认与负 `dim`，2 层嵌套
vmap，grad 组合。`test_solve_batching_rule`：两操作数均批处理、共享/未批处理
`a` + 批处理 `b`（文档的“共享参数”担忧）、grad 组合——使用规范的
`(...,M,K)` 矩阵 RHS（原因见下文新发现的 `solve()` 缺口，为何不用裸
`(...,M)` 形式）。`test_numpy_code_raises_clear_error_under_vmap`、
`test_sparse_raises_clear_error_under_vmap`。`test_unsupported_op_raises`
从 `cumsum`（现已支持）改为 `cumprod`（同一算子家族，确实仍不支持），以如实
继续测试快速失败契约。新增 `python/jittor/compat/tests/torch/
test_real_imag_under_vmap.py`（torch 模式）用于 compat 修复。

### 第三个既有无关 bug，发现但明确不修（超出范围）

在构造 `solve` 批处理规则测试时发现，`jt.linalg.solve` 本身——完全独立于
vmap，用普通非 vmap 调用即可复现——不能正确支持：(a) 裸 `(...,M)` 向量栈
`b` 对批处理 `a`（此 numpy 版本的 `np.linalg.solve` gufunc 只在
`b.ndim==1` 恰好成立时才走“b 是向量”路径；批处理的 `(...,M)` 被误解析为 2-D
`(m,n)` 核心矩阵，抛出核心维不匹配——`solve()` 自己的 docstring 暗示
`(...,M)` 是受支持的向量形式，但对真正批处理的情形在此 numpy 版本下并非
如此）；(b) 未批处理 `b` 对批处理 `a`——`numpy_code` 的输出形状按
`bt.shape` 原样声明（`python/jittor/linalg/solving.py:316`），没有对 `a` 的
额外 batch 维做广播，因此正确的、更大的 `np.linalg.solve` 结果无法拷入过小的
声明输出缓冲（`ValueError: could not broadcast input array from shape
(4,3,1) into shape (3,1)`）。两者都是真实的 `solve()` 正确性缺口，不是 vmap
缺口——vmap 批处理规则只需（也确实）正确扩展 `solve()` 已正确处理的情形，即
两个操作数都批处理（或都不）且 `b` 为规范的 `(...,M,K)` 矩阵。此处未修——
标记为需要对 `solving.py` 自身的广播/形状推断逻辑单独专门处理。

### 门禁结果

- `tools/run_test_suite.py --tier core`：native 98 passed/41 skipped/1
  xfailed，torch 212 passed/3 skipped。0 失败。
- `tools/run_test_suite.py --tier smoke`：与更早 P0 会话的 smoke 运行同一
  数量级的既有环境噪声（这个极简容器中的 MPI/ACL/DDP/TF32/RNN/pooling/
  einsum 缺口，以及相同的“独立 Torch 未预加载” oracle 会话上下文错误）——
  定向搜索除已知的 oracle 上下文模式外，没有发现
  SGD/vmap/cumsum/solve/numpy_code 特有的新问题；
  `tests/codegen/test_broadcast_tuner.py` 的一个失败是 P0 会话 smoke 运行中
  已见（且未修）的同一个。
- 完整 `tests/core/` + `tests/linalg/`（两次——一次捕获 view/broadcast_var
  回归，修复后一次干净）：463 passed，只剩 2 个既有的
  `test_namespace_exports.py` 失败（dlpack/vmap/Generator stub 声明，作为本文
  档自身的 §3.4/§3.5 另行跟踪，本次会话未改动）。
- `tools/check_repo_layout.sh`：一处发现，`local/problems-2026-9-22.md`
  （用户自己的外部报告文件，两个会话之前就存在，引用了不同 repo/harness 中的
  `development-plan.md`）——留给用户处理，与 P0 会话的报告相同。
- `JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest tests/structure`：
  收集 1378；途中修复两处真实发现（为本次及更早 P0 会话新增文件重新生成
  `MANIFEST.in`；两处急切安装身份破坏，通过回退急切安装而非追查别名系统
  解决）——降到与 P0 会话相同的 2 个既有失败
  （`test_a_timeout_ends_the_grandchildren_too`、
  `test_documentation_governance_checker`，后者也是上面的 `local/` 发现）。

## 仍开放的内容

§3.3 内：ACL 未在真实硬件上验证（本次会话无 NPU 可用）。§3.4 内：惰性 patch
安装的“陈旧引用”缺口已调查且有意不修（与更强的既有不变量冲突；已记录的绕法
已存在且现在有测试）；途中发现的 `jt.linalg.solve` 广播/输出形状缺口是另一个
真实、未修的问题。原始 9 项列表中剩余的 5 项（原生 vmap 自身超出此处测试范围
的 JVP/VJP/pytree/共享参数组合、DLPack、CUDA RNG checkpoint、高阶复数 AD、
函数级编译/缓存契约、稀疏设备侧 coalesce）本次会话未处理，按用户自己对
§3.3/§3.4 的限定。
