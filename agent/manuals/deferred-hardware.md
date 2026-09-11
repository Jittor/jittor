# 缺硬件时怎么把代码改完，以及硬件到手那天跑什么

看板上有两个归并桶（「并入 硬件验收」「并入 多机硬件验收」）和几条混合任务，共同的意思是
「代码可以现在写，验收要等机器」。桶只说了归并，没说**到时候跑什么**、**绿了之后哪几条算被证明**、
**哪些还根本没有可跑的测试**。这一页只回答这三个问题。

各硬件的操作步骤不在这里：Ascend 看 [`docs/guides/ascend-910b.md`](../../docs/guides/ascend-910b.md)，
HCCL 多卡看 [`hccl-on-device-verification.md`](hccl-on-device-verification.md)，Corex 看
[`docs/guides/corex.md`](../../docs/guides/corex.md)。

## 没有硬件时，什么算「代码改完了」

按有效性从强到弱，**能用强的就不要用弱的**：

1. **拿桩 SDK 编真实翻译单元。** `agent/skills/acl-host-syntax-check` 生成一棵 CANN 桩树，对真实
   `.cc` 跑 `g++ -fsyntax-only`。它抓得到签名不匹配、缺头、类型错——这些占了缺硬件时会犯的大多数
   错。**局限写在 skill 里**：`aclnnXxxGetWorkspaceSize` 的实参无法校验（要真 SDK 头），而且
   `-fsyntax-only` 单独不足以抓「把 workspace query 传给 `launch()`」这类错（可变参桩会愉快地转成
   `std::function`），要另加一条比较裸函数指针类型的检查。**ROCm 与 Corex 目前没有等价物**，谁做那两
   条时值得照着建一个。
2. **行为形状的静态合同。** 断言「每个 family 都不自己发 execute 调用」这类**不变量**，而不是
   「共有 65 处调用 `checkRet`」这类**计数**。计数式合同挡不住合法重构：`test_acl_runner_failure_contract`
   曾断言 65 处，8.06 的第一个提交把样板收进共享 launcher 后它就作废，然后红了约 40 个提交没人看见。
   不变量式合同还要**自证扫到了东西**：ACL 正处在 `python/jittor/extern/acl` 与 `backends/acl`
   两处都有内容的半途搬迁状态，按「总数 > N」断言会在只扫到一侧时仍然发绿，所以要求**每个根各自非空**。
2b. **去样板前后的静态等价性。** 把每个 owner 归约成 (workspace 查询, execute 入口, 同步策略)
   的有序 token 流，逐 owner 对比改前改后。`tools/build/acl_launch_program.py` 做这件事，两个树
   当参数、退出码非 0 即有差异。它把「样板删对了」和「顺手改了行为」分开：前者 token 流不动，
   后者会以具体 owner 的具体 token 出现，必须逐条解释。**这比字符串合同强**，因为它比较的是
   设备会观察到的东西，而不是源码长什么样。
3. **桩后端跑通编译与注册。** `setup_fake_cuda_lib` / `tests/backends/corex/test_corex_discovery.py`
   的离线 fake compiler 属这一类：证明发现与注册路径不依赖真设备。
4. **只写上机文档。** 最弱的一档。可以，但**必须同时给出精确的 nodeid 或命令**，否则硬件到手那天
   要重新读一遍代码才知道该跑什么——这正是本页存在的原因。

**不允许的做法**：把缺硬件的用例写成自我 skip 然后当通过。`nox -s cuda`/`npu` 已经用
`JITTOR_TEST_REQUIRE_CUDA=1`／`JITTOR_TEST_REQUIRE_ACL=1` 加 `JITTOR_TEST_ACCELERATOR_MIN_EXECUTED=1`
把这条堵成 fail-closed（合同在 `tests/structure/test_hardware_gate_matrix.py`）。新开硬件 session
照这个形状建。

## 硬件到手那天跑什么

### Ascend 910B3，单卡（CANN）

| | |
| --- | --- |
| 命令 | `nox -s npu`（需 `CANN_SET_ENV` 指向 `set_env.sh`，否则 session 直接 error） |
| fail-closed | `JITTOR_TEST_REQUIRE_ACL=1`、`JITTOR_TEST_ACCELERATOR_MIN_EXECUTED=1`；session 先跑 `npu-smi info`，再跑一个带精确期望值的 matmul 探针 |
| 覆盖 | `tests/backends/acl/test_acl.py`、`tests/backends/acl/test_acl_torch_compat.py`、`tests/backends/acl/test_aclop.py`、`tests/backends/acl/test_acl_indexing.py`、`tests/ops/test_ops.py`、`tests/ops/test_floor_divide.py::TestFloorDivideNPU` 等 |
| 步骤 | [`docs/guides/ascend-910b.md`](../../docs/guides/ascend-910b.md) |

绿了之后可以判定的看板项：**`6.B02`**（executeOp 失败都抛；静态侧已由
`test_acl_runner_failure_contract` 与 `test_acl_tensor_workspace_contract` 钉住）、**`6.B16`**
（`sync_run=1/0` 两条路径；精确 nodeid 在 Ascend 指南里）、**`8.06`** 的已迁 family（静态合同
102 passed，逐 family 的设备行为要在这里过一遍）、**`4.11`/`4.12`** 的 ACL 注册路径。

#### 8.06 的 launch 尾部归零，还差什么

本机做到的是三档里的第 1、2 档：桩 SDK 过 TU（44 个源文件 `-fsyntax-only` 全过、70 个
launcher ABI 断言全过，反向对照见下）、以及不变量式静态合同。**一条设备指令都没执行过**，
`tests/backends/acl` 在本机是 `164 skipped, 0 executed -- explained: skipped: no acl found`。

前置：Ascend 910B3 + CANN，`CANN_SET_ENV` 指向 `set_env.sh`。四条按顺序跑，前一条不过不要往下走。

1. **能编。** 这一层桩 SDK 挡不住（`aclnnXxxGetWorkspaceSize` 的实参无从校验），必须真 SDK：

   ```bash
   source "$CANN_SET_ENV" && npu-smi info
   JITTOR_TEST_REQUIRE_ACL=1 python -c "import jittor; jittor.flags.use_acl = 1"
   ```

   判据：`aclops/*_acl.cc` 全部编过。特别看本波改的 6 个文件——`reduce_op_acl.cc`、
   `arg_reduce_op_acl.cc`、`truth_reduce_op_acl.cc`、`norms_op_acl.cc`、`upsample_op_acl.cc`。
   `launch(ret, aclnnProd, true)` / `launch(ret, aclnnProdDim, false)` 是本波新增的两个
   launcher 站点，`aclnnProd` 与 `aclnnProdDim` 的真实 execute ABI 只有这里能确认。

2. **算得对，且与迁移前逐算子一致。** 本波声称的等价性是**源码级**的：
   `python tools/build/acl_launch_program.py <迁移前树> <当前树>` 证明 69/71 个 owner 的
   (workspace 查询, execute 入口, 同步策略) token 流逐字相同。设备侧要把它变成数值对拍：

   ```bash
   nox -s npu   # JITTOR_TEST_REQUIRE_ACL=1 JITTOR_TEST_ACCELERATOR_MIN_EXECUTED=1
   ```

   判据：`tests/backends/acl/test_aclop.py`（114 条）、`test_acl.py`（43 条）、
   `test_acl_indexing.py`（7 条）全部**执行**而非 skip——本机这 164 条一条都没执行。
   重点算子：`prod`（三条路径都要覆盖：整张量归约、单轴、多轴分步）、`argmax`/`argmin`、
   `all`/`any`、`GroupNorm` 前反向、`UpsampleNearest2d` 前反向。

3. **失败真的抛，而不是静默给出未初始化输出。** 本波把三处「打印后 return / 根本不检查」
   改成 `launch()` 的 `LOGf`。设备侧的判据是**负向**的：构造一个会让 workspace 查询失败的
   调用（例如给 `prod` 一个 ACL 不接受的 dtype 组合），确认得到带算子名与解码后 ACL 状态的
   异常，而**不是**一个数值错误的结果。`docs/guides/ascend-910b.md` 的
   `forbid_backend_fallbacks()` / `backend_fallback=error` 一节是配套开关。
   在此之前不得声称这三处已验证：静态合同只能证明源码里没有那条 return。

4. **`sync_run` 两条路径。** `launch()` 的第三个参数就是这个策略。
   `sync_run=1` 与 `sync_run=0` 各跑一遍第 2 步的清单，判据是数值一致且
   `AdamWList` 与 reduce prod 分步路径不因少了逐步同步而出错——这两个 owner 是唯一
   自己放 `syncRun()` 的，见 `test_the_two_owners_that_keep_their_own_sync_are_the_documented_ones`。

### Ascend 910B3，≥2 卡（HCCL）

| | |
| --- | --- |
| 命令 | **尚无 nox session** ← 硬件日之前要建 |
| 现有材料 | [`hccl-on-device-verification.md`](hccl-on-device-verification.md) 的四步：第 0 步先证明没静默回落 CPU、A/B 数值对拍、200 轮压竞态、证明时序真的变了 |
| 主机侧已可跑 | `tests/backends/comm/hccl/test_hccl_check_macros.py`、`test_hccl_collective_sync_switch.py`（对桩编译，任何机器都能跑），但**它们不在任何 nox session 里** |

等它的看板项：**`8.02`** 的 HCCL 那半（4 次全设备同步收进了 `JT_HCCL_COLLECTIVE_SYNC` 开关，
默认 `full` 与改前逐字等价，删除要等实机 A/B）、**`10.19`** 的 HCCL 四项（`HcclAllGatherOp::grad()`
目前直接 `LOGf << "not implemented"`，另三个要在多卡上补真实梯度与 CPU 对照）。

### ROCm

| | |
| --- | --- |
| 命令 | `nox -s rocm` |
| 覆盖 | `tests/backends/rocm/test_rocm.py` 与 `tests/distributed/` 的 MPI 组 |
| 静态侧 | `tests/structure/test_no_unexplained_binaries.py`、`tests/structure/backends/rocm/test_rocm_library_provider.py`、`tests/structure/backends/rocm/test_rocm_native_provider.py` |

等它的看板项：**`4.12`** 的 ROCm 那半。**代码半已闭合并标已合并**，这里只剩真卡确认。

4.12 把 ROCm 从「吃 CUDA 源码文本替换的产物」换成了 `backends/rocm/` 下自有的 HIP provider。
主机侧能证明的只有形状：`configure()` 返回的 `BuildConfig` 只带 `runtime/driver.cc` 一个
`BuildSource`（`language="hip"`）、`resources` 里没有 `rocm_converter`、入口点是
`rocm = "jittor.backends.rocm"`。**这些都没碰过 hipcc，更没跑过 kernel。**
按本页开头的分档，ROCm 现在只有第 2、3 档（静态合同＋桩发现），**没有第 1 档**——
`agent/skills/acl-host-syntax-check` 那样的桩 SDK 语法检查 ROCm 还没有等价物，
`runtime/driver.cc` 至今没有被任何编译器读过。真卡日的第一件事应当是先补它。

真卡到手那天按顺序确认四条，前一条不过不要往下走：

1. **能编。** `runtime/driver.cc` 过 hipcc。这是唯一完全没被验证过的一层。
2. **能装能选。** `JT_BACKEND=rocm` 走 entry point 选到原生 provider，`import jittor` 通过，
   `jt.core.registered_backends()` 含 `rocm`。失败要显式报错而不是静默回落 CPU。
3. **算得对。** `nox -s rocm`（`tests/backends/rocm/test_rocm.py`：`TestROCm`、
   `TestROCmUnaryOp`/`BinaryOp`/`ReduceOp`/`ReindexOp`/`WhereOp`/`CodeOp`、`TestBMM`、
   `TestExample`）。**照 `npu` session 的形状加 fail-closed**：`JITTOR_TEST_REQUIRE_ROCM=1`
   加 `JITTOR_TEST_ACCELERATOR_MIN_EXECUTED=1`，否则缺卡时整组自我 skip 读着是绿的。
4. **库族的边界是真的。** `backends/rocm/build.py` 目前对 MIOpen 与 rccl 抛
   `NotImplementedError("no native ROCm library provider")`——这是**有意的未实现**，不是 bug。
   真卡上确认依赖它们的用例是明确失败而不是算出错误结果；要补实现的话是新任务，不在 4.12 内。

绿了之后可以判定的：4.12 的 ROCm 那半。**在此之前不得声称 ROCm 硬件验证完成**；
本机是 8 张 RTX 4090，无 ROCm 设备，上述四条一条都没跑过。

### Corex / 天数

| | |
| --- | --- |
| 命令 | **尚无 nox session** ← 硬件日之前要建 |
| 现有材料 | [`docs/guides/corex.md`](../../docs/guides/corex.md)（48 行，最薄的一份）；`tests/backends/corex/test_corex_discovery.py` 用离线 fake compiler 验发现路径 |

等它的看板项：**`8.14`**（`check()` 只读、路径可配置）。**代码半已闭合**（2026-09-07 复核）：
前置 `4.12` 已合并，`process_acl` 全树 0 处；`corex_compiler.py` 现在没有 `check()`，只有只读的
`discover()`，路径经 `corex_home` 实参或 `COREX_HOME` 解析、默认 `/usr/local/corex`。

**「探测无副作用」这条验收现在真的被证明了。** 原来那条断言只比较一个临时目录**顶层**的
`os.listdir`，对四类真副作用全盲——而 `check()` 当年的问题恰恰是**去跑编译器**。现在
`side_effect_recorder` 同时看:进程派生（`subprocess.run`/`Popen`/`check_output`/`os.system`/
`os.popen` 全部换成会记录并抛出的守卫）、写模式的 `open`、`os.environ` 前后比较、监视目录的
**递归**文件树快照（大小 + mtime）、以及 cwd。

`TestTheGuardCanNoticeSideEffects` 给上面几条装牙齿:把三类副作用分别注入 `discover()` 的一份
**副本**，断言守卫报得出来。牙齿本身也验过——把守卫里的 `os.environ` 比较去掉，那条立刻报
`env side effect went unnoticed`（1 failed / 4 passed）。没有这一层，前几条会在守卫悄悄失效后
继续全绿，正是本轮那七例的形状（见交接文档 §6bis）。

**硬件日要跑的**（本机无 Corex/Iluvatar 卡）：

1. `COREX_HOME=<真实安装路径> python -c "from jittor.backends import corex as c; print(c.discover())"`
   ——判据：`available=True`、`reason == "ready"`、`compiler_path` 指向真实 `bin/clang++`。
2. 带 `COREX_HOME` 的 `import jittor`，判据：`jt.flags.backend == "corex"`、`has_corex=True`、
   `has_cuda=False`（`configure()` 明确把 CUDA 标记置假而只在设备编译上保留 CUDA ABI 标记）。
3. 一个逐元素算子加一个归约算子的数值对拍（对 numpy），判据：与 CPU 参考一致；
   注意 `configure()` 把 `use_cutt=0` 写进 environment，所以 transpose 走内建 kernel。
4. `kernel_source_roots` 只声明了 CUDA 的 `kernels/core`——判据：真机上确认内建加速器覆盖生效，
   即那批算子没有退回通用路径（源码注释里点明空元组会让它静默退回，这一条要在真机复核）。

### 两台机器

| | |
| --- | --- |
| 命令 | **尚无 nox session** ← 这本身就是 `10.22` 的内容 |
| 现有材料 | `8.15` 已合入 TCP/SQLite Store 与 NCCL WORLD bootstrap，可在单机多进程下验 rendezvous、超时与失败传播 |

等它的看板项：**`8.15`**（真实两机 collective 对拍）、**`8.16`**（`torchrun --nnodes=2` 跑通 transformers
训练、两机 loss 轨迹一致）、**`8.17`**（跨机带宽微基准、掉线在超时内退出）、**`8.18`**（2 机保存、
4 机加载的续训）、**`10.22`**（两节点 smoke 进 nightly）。

### 后端 `grad()` 的 CPU 参考对拍

全树共 **60 个后端梯度实现**（C++ `::grad()` 28 个，Python `jt.Function.grad` 32 个），清单在
`tests/structure/test_backend_grad_contract.py` 的 `BACKEND_GRAD_COVERAGE`，与源码树逐条相等，
新增或删除任何一条都会报红。其中 **24 条在本机能真跑**（CUDA 22 条 + oneDNN 1 条 + …），
**36 条要等硬件**。清单里的 `kind` 字段就是下面这张表：

| kind | 含义 | 硬件到手那天跑什么 | 通过判据 |
| --- | --- | --- | --- |
| `mpi_hardware` | MPI collective 的反向，要 `mpirun` 起多进程 | `tests/backends/comm/mpi/test_mpi_op.py` | 各 rank 的梯度与单进程 CPU 参考一致 |
| `nccl_hardware` | NCCL collective 的反向，要多卡 launcher | `tests/backends/comm/nccl/test_nccl_ops.py`、`compat/tests/fsdp2/test_fsdp2_nccl.py` | 同上；`all_gather` 的反向要真的走 reduce-scatter |
| `npu_hardware` | ACL/CANN 的反向，已有梯度用例，缺卡 | `nox -s npu` | 用例内已带 CPU 对照，全绿即判定 |
| `npu_hardware_no_grad_test` | **有 `grad()` 但只有前向用例**——即使有卡也测不到反向 | 见下面逐条 | 要先补用例，不是跑一遍就行 |
| `rocm_hardware` | ROCm 的反向，已有 CPU 对拍用例，缺卡 | `nox -s rocm` | `TestBMM::test_bmm_rocm` 自带 `calc(0)`/`calc(1)` 双跑对拍 |
| `rocm_hardware_no_grad_test` | 同 `npu_hardware_no_grad_test`，ROCm 侧 | 见下面逐条 | 同上 |
| `unsupported_hardware` | 反向根本没实现，当前直接报错 | — | 先实现，再谈验收 |

**缺卡不等于缺用例。** 上表后四行是真正的窟窿，逐条列在这里，硬件日之前就可以先把用例写出来：

| 符号 | 源码 | 现状 |
| --- | --- | --- |
| `HcclAllGatherOp` | `python/jittor/extern/acl/hccl/ops/hccl_all_gather_op.cc` | `grad()` 直接 `LOGf << "not implemented"`；要在 Ascend 910B3 多卡上补实现与 CPU 对照 |
| `RocprimCumsumOp` | `backends/rocm/libraries/rocprim/rocprim_cumsum_op.cc` | 全树**没有任何用例**碰过它，前向反向都没有；`tests/backends/rocm/test_rocm.py` 里要补一条照 `TestBMM` 形状的 CPU 对拍 |
| `FloorIntACL` | `backends/acl/kernels/ops/floor_op.py` | 只有 `test_aclop.py::TestACL::test_floor_int` 前向 |
| `IndexACL` | `backends/acl/kernels/ops/index_op.py` | 只有 `test_aclop.py::TestACL::test_index` 前向 |
| `NonzeroACL` | `backends/acl/kernels/ops/where_op.py` | 只有 `test_aclop.py::TestACL::test_nonzero_1` 前向 |
| `StackACL` | `backends/acl/kernels/ops/stack_op.py` | 只有 `test_aclop.py::TestACL::test_stack` 前向 |
| `TriuACL` | `backends/acl/kernels/ops/triu_op.py` | 只有 `test_aclop.py::TestACL::test_triu` 前向 |

这七条**不得用 skip 冒充通过**。`test_backend_grad_contract.py` 里
`test_every_gradient_without_a_gradient_test_is_listed_in_the_manual` 会强制它们逐个出现在本节：
补上用例并改掉 `kind` 之前，谁都删不掉这段文字。

## 硬件日之前还要补的三件事

上面三处「尚无 nox session」是这一页最该被人接手的部分。它们不是文档问题：**没有 session 就意味着
硬件到手那天没有一条命令可跑**，只能靠人照着散文重新拼命令，而那正是最容易漏掉 fail-closed 检查
的时候。三个都照 `npu` session 的形状建即可——强制环境变量、先跑设备探针、`REQUIRE_*` 加
`MIN_EXECUTED` 堵住自我 skip、然后跑清单。

## 这一页怎么保持不过期

`tests/structure/test_deferred_hardware_manifest.py` 钉住三件事：看板上归入硬件验收桶的每条任务都
在本页出现；本页提到的每个测试路径都真实存在；本页声称有 nox session 的每种硬件，`noxfile.py` 里
确实有那个 session（而声称「尚无 session」的，确实没有——这一条是为了让上面那三件事被补上之后，
本页必须跟着改，而不是留着一句过期的「尚无」）。
