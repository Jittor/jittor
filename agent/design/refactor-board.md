# 整改看板

> 第102波增量：`cb0a4e77` 完成 cuBLAS batched matmul rank 用户错误 cohort（结构 3 passed）；`dbe72f0f` 完成 torch.polar numerical owner cohort（CPU 2 passed）。2.19/7.03 聚合任务仍按剩余范围保持「待领」；`6.B16` 仍待 Ascend 910B3 实机。
>
> 第103波增量：`e38cce97` 完成 cuBLAS acc matmul rank 用户错误 cohort（结构 3 passed）；`6fdb6120` 完成 complex accessor numerical owner cohort（CPU 2 passed）；`448aa10a` 删除 ACL 无消费者 `op_idx_map`（静态合同 68 passed）。2.19/7.03 聚合任务仍待完整审计，ACL 仍待硬件验证。
>
> 第104波增量：`0e275f14` 完成 cuBLAS batched matmul inner-dim 用户错误 cohort（结构 4 passed）；`84f29d9b` 完成 `hann_window`/`stft` signal numerical owner cohort（CPU 3 passed）；ACL `aclnn.h` 头文件契约由 `1e8e90c6` 已完成。聚合任务和硬件验收状态不变。
>
> 第105波增量：`7d366087` 完成 cuBLAS batched matmul batch-shape 用户错误 cohort（结构 5 passed）；`f848a6a7` 完成 `torch.equal` numerical owner cohort（CPU 3 passed）。ACL `AclOpFunctions` 类型擦除需跨注册表/构造器/owner 协同，未做不安全半改。
>
> 第106波增量：`cc9ed2bc` 完成 cuSPARSE CSR shape 用户错误 cohort（结构 3 passed）；`258be343` 完成 `kron`/`logsumexp` numerical owner cohort（CPU 2 passed）；`15c86886` 记录 ACL 类型擦除/属性 data/descriptor cache 的原子迁移边界（静态合同 1 passed）。2.19/7.03/8.06 聚合任务仍按剩余范围保持「待领」。
>
> 第107波增量：`a831fbd6` 完成 cuSPARSE COO shape 用户错误 cohort（结构 3 passed）；`f20bb4ff` 完成 `all`/`any` reduction numerical owner cohort（CPU 2 passed）；`7a35e8e7` 记录 ACL 属性 data 通道候选的协同边界（静态合同 1 passed）。聚合任务和硬件验收状态不变。
>
> 第108波增量：`55a1f481` 完成 cuBLAS matmul 输入 b rank 用户错误 cohort（结构 4 passed）；`9954b4ed` 完成 `tensor_split`/`take` numerical owner cohort（CPU 2 passed）；`296e0837` 补充 ACL 属性 data/descriptor/type-erasure 迁移顺序与 910B3 验收合同（静态合同 1 passed）。聚合任务和硬件验收状态不变。
>
> 第109波增量：`354ec6f1` 完成 cuTT transpose 标量输入用户错误 cohort（结构 3 passed）；`aa3ca532` 完成 `index_copy` numerical owner cohort（CPU 2 passed）。ACL `softmax.dim`/`triu.diagonal` data-channel 迁移仍需统一 schema、C++ 解码和 cache-key 协同，未做半改。
>
> 第110波增量：`26ee6bee` 完成 cuDNN RNN descriptor mode 用户错误 cohort（结构 3 passed）；`d3e33b0e` 完成 `index_put` numerical owner cohort（CPU 2 passed）。ACL data-channel C++ 解码入口仍缺统一实现，未做半改；0.15/10.19 验收状态不变。
>
> 第111波增量：`038819e6` 完成 cuBLAS acc matmul inner-dim 用户错误 cohort（结构 4 passed）；`64f4ecfa` 建立 `torch.autocast` numerical owner（CPU 2 passed）；`3cb0da4e` 形成 ACL data-channel schema 草案与 910B3 验收边界（静态合同 1 passed）。聚合任务和硬件验收状态不变。
>
> 第112波增量：`d2532ac2` 完成 cuDNN convolution format 用户错误 cohort（结构 3 passed）；`dbdfb6d7` 完成 `index_copy_` 原地 numerical owner（CPU 2 passed）；`a768cc9b` 锁定 ACL data schema 静态合同（2 passed）。聚合任务和硬件验收状态不变。
>
> 第113波增量：`de5188ab` 完成 cuDNN backward-x format 用户错误 cohort（结构 3 passed）；`11e9b456` 完成 `index_put_` 原地 numerical owner（CPU 2 passed）；`4b22f6d9` 增加 ACL data schema Python validator（静态合同 6 passed）。聚合任务和硬件验收状态不变。
>
> 第114波关闭 `10.02`：`151c5856` 将 `cpu` 加入 `nox.options.sessions` 默认列表，新增 AST 合同确认默认数值门禁；定向 1 passed。看板已合并 200、待领 74。
>
> 第115波关闭 `10.01`：`5501d0b6` 增加稳定 `nox -s full` 完整 CPU/nightly 入口并切换 CPU workflow；调度 AST 合同 2 passed。看板已合并 201、待领 73。
>
> 第116波整卡审计：未发现可在当前环境完整关闭的新任务。`0.15` 仍约 390 s，`2.19`/`7.03`/`8.06` 仍有聚合剩余，ACL/HCCL 仍待 910B3 实机；本波无代码提交，待领保持 73。
>
> 第117波增量：`c329a822` 完成 cuDNN backward-w format 用户错误 cohort（结构 3 passed）；`24637c96` 完成 vmap numerical owner wrapper（CPU 2 passed）；`25c5ffed` 明确 ACL C++ decoder 接口、错误映射和 canonical cache key 合同（静态合同 2 passed）。聚合任务和硬件验收状态不变。
>
> 第118波整卡审计：8.12、10.17、10.18、10.20、10.21 均缺完整实现/门禁；0.20、0.22、9.01、9.07、9.19 受布局、CUDA、导入或前置依赖阻塞。本波无代码提交，待领保持 73。
>
> 第119波增量：`bb5f72f1` 完成 tensor installer `corrcoef` numerical owner（CPU 2 passed）；`80e0f1b4` 形成 10.17 异步错误诊断契约和静态合同（1 passed），但低开销 ring/runtime 与 CUDA 实机仍待。待领保持 73。
>
> 第120波增量：`a0d3be31` 完成 `broadcast_shapes` numerical owner（CPU 2 passed）；10.17 仍缺 per-thread ring/stream 关联和 CUDA 异步注入，8.12 仍缺 cuDNN POD key 全套。待领保持 73。
>
> 第121波整卡审计：8.12 的 cuDNN 2D/3D forward/backward 六条 legacy cache 路径仍需共享 `LegacyConvAlgoKey`、per-device cache 与 CUDA 验收，当前无可安全半改；10.17 同样缺 runtime ring/stream 真实链路。本波无代码提交，待领保持 73。
>
> 第122波整卡审计：2.22/4.06/8.12/9.07 仍缺统一配置、fallback 三态、cuDNN POD key 或 import 环境隔离；7.05/7.07/10.18/10.20/10.21 仍缺事务锁、patch 边界、核心属性测试、内省 API 或 import lint；布局/多机/ACL 任务也有明确前置。本波无代码提交，待领保持 73。
>
> 第123波整卡审计：7.05 现有 namespace 事务只覆盖部分模块树，仍缺全局安装锁、失败可重试/可查询和 os.environ/flags 全量回滚；本波无代码提交，待领保持 73。
>
> 第124波整卡审计：7.05 复核确认上述缺口仍未改变，不能用现有 namespace snapshot 代替安装锁、失败状态 API 或环境/flags 回滚；本波无代码提交，待领保持 73。
>
> 第125波 7.05 前置：`3c8b46f3` 使 optional install 失败 warn-once、可查询、可重试；`44272e89` 新增 reversible `InstallTransaction`（RLock、逆序 rollback/commit/retry）及 module/env/flags/meta_path 合成测试 2 passed。尚未接入全部 installers，7.05 仍待领。
>
> 第126波 7.05 前置：`fa2027e4` 将 InstallTransaction 锁和 namespace undo 接入 `compat.torch.install`；`4f66701e` 明确 flags/env/import/meta_path/module-patcher 尚未纳入回滚，并补状态边界合同 1 passed。7.05 仍待领。
>
> 第127波 7.05 安全修正：有风险的整表 global snapshot 提交 `ec720cd8` 已由 `9a674001` 撤回；`1aa640cf` 补充 flags/env 写入清单与显式 allowlist/owner-aware restore 要求。7.05 仍待领。
>
> 第128波 7.05 子项：`9949fa10` 修复 completed-install namespace 冲突异常路径的 RLock 泄漏，新增回归测试；`test_install_context.py -k 'completed_install_conflict or optional_failure'` 2 passed。完整 installer mutation ledger 仍待接入，7.05 保持待领。
>
> 第129波 7.05 子项：`c8b993b4` 让 transaction rollback 做 owner-aware 值校验，外部改写时抛 `TransactionConflict`；事务合成测试 3 passed。仍需将所有 installer 写入口纳入 ledger，7.05 保持待领。
>
> 第130波 7.05 子项：`75793c04` 将 distributed installer 的 JT_NCCL_*、`use_nccl/use_mpi` 和 `jt.flags.use_cuda` 写入纳入 mutation ledger；`783699cd` 补 child-env 隔离合同 2 passed。仍有其他 installer 写入口未迁移，7.05 保持待领。
>
> 第131波 7.05 测试修正：`f8f838b2` 将 distributed 环境结构合同改为断言 `tx.mutate_env/mutate_flag`，避免旧 direct-write 断言误报；结构 2 passed、事务相关 5 passed。7.05 仍待领。
>
> 第132波 7.05 修正：`1eb7ec07` 让 `mutate_env` 先规范化字符串再记录 owner 值，修复整数环境变量回滚误报 `TransactionConflict`；事务定向 6 passed。7.05 仍需覆盖全部 installer 写入口。
>
> 第133波 7.05 测试修正：`c49efb80` 删除两个已被显式 activation API 取代的 `wrap_flags` 旧 patch，完整 `test_install_context.py` 从 2 个夹具错误恢复为 20 passed；事务/状态结构合同保持通过。7.05 仍待全量 mutation 接入。
>
> 第134波 7.05 子项：`60197b81` 将 factories/tensor installer 的 `jt.flags.use_cuda` 写入接入 transaction helper；`d9d063a5` 将 core installer 的安装期 `use_cuda` 写入接入 ledger 并补失败回滚测试。nn 的 `.to()` 写入确认属于运行时用户语义，不纳入安装事务；7.05 仍待全部 installer 写入口。
>
> 第135波 7.05 子项：`87ca0a82` 新增 owner-aware `mutate_attr`，将 transformers runtime guard 与 torchmetrics fastpath 的 `builtins.__import__` hook 接入 ledger；`sys.meta_path` permissive finder 因 allowlist/身份耦合暂不迁移。7.05 仍待领。
>
> 第135波 companion：`7af13605` 新增 utilities import-hook 回滚/外部替换冲突测试，定向 6 passed；不改变 7.05 整卡待领状态。
>
> 第136波 7.05 子项：`88795374` 让 permissive finder 新增和既有 allowlist 增量进入 transaction ledger，compiler installer 三处调用已接线；尚未覆盖 module_patcher finder 和并发外部替换，7.05 仍待领。
>
> 第137波测试修正：`9c6a7e92` 将 compile refusal 测试从过时的 `never checked` 文案更新为当前 `unchecked` 合同；transaction/permissive 定向共 27 passed。7.05 仍待完整 mutation 接入。
>
> 第138波 7.05 子项：`1a37b895` 将 module_patcher finder、registry、entry-point 状态接入 transaction undo，并由 integrations 传递 active transaction；外部 finder/allowlist owner 冲突测试仍待，7.05 保持待领。
>
> 第139波 7.05 安全修正：`ee1317c2` 让 permissive finder 新增/索引和既有 allowlist 增量回滚做 owner-aware 校验；外部 allowlist/finder 重排测试纳入事务合同，9 passed。module_patcher/其他 installer 全流程仍待，7.05 保持待领。
>
> 第140波 7.05 子项：`02b1733b` 将已加载 module 属性差异纳入 transaction ledger，`6e0f838a` 增加 module-patcher finder/registry 冲突保护；`test_compat_mechanisms.py` 20 passed，事务测试 9 passed。可变对象内部 mutation 和全部 installer 汇总仍待，7.05 保持待领。
>
> 第141波 7.05 子项：`9f154035` 将 vLLM `_ArmOnFirstImport` finder 插入和 active transaction 传递接入 ledger；external backend registry、vLLM callbacks/extension modules 和 shim runtime 全局写入仍未覆盖，7.05 保持待领。
>
> 第142波 7.05 子项：`367716a7` 将 external backend 的 `_BACKENDS`、`_BACKEND_HINTS`、`_ENTRY_POINTS_LOADED` 注册状态接入 transaction，并由 integrations 传递 active tx；source import 的 sys.path/sys.modules 仍需 owner-aware 或子进程隔离，7.05 保持待领。
>
> 第143波整卡边界：shim runtime activation 的 sys.path、sys.modules、flags、递归安装和扩展构建跨越现有 InstallTransaction 生命周期，不能半接入；需先建立独立 ActivationTransaction（path/module/flag owner token、冲突硬失败、child/retry 测试）。7.05 保持待领。
>
> 第144波 7.05 前置：`c204f4e9` 扩展 InstallTransaction 为 ActivationTransaction，新增 owner-aware `mutate_path`/`publish_module` 和 activation path/module 回滚合同；事务定向 11 passed。尚未接入 shim.runtime.activate，7.05 保持待领。
>
> 第145波 7.05 子项：`4d5b8e61` 将 shim runtime 的 `_composition=True` 分支接入 ActivationTransaction，包住 torch install 与 module publish；普通 activation 的 path/build/no_grad 仍未覆盖，7.05 保持待领。
>
> 第146波 7.05 子项：`e0885bf7` 将普通 shim activation 的父进程 sys.path 增量接入 `ActivationTransaction.mutate_path`，异常回滚、成功提交；child `PYTHONPATH` 保持独立。sys.modules/no_grad/build 和 child/path 专项测试仍待，7.05 保持待领。

一行一个任务，与 [refactor-plan.md](refactor-plan.md) 的编号对应。领任务把状态改成「进行中」并写名字，
完成改成「已合并」并填提交号；推送冲突说明别人先领了。状态只有四种：待领 / 进行中 / 已合并 / 并入 X。

> **2026-09-03**：上一轮中断留下的五个 `wip/*` 与工作树残留已经验证、合入或明确退回待领；任务表已无
> 「进行中」或「部分完成」。下一波从 [交接说明](refactor-handoff.md) 第 6 节开始。

## 起点已知失败清单（归责之前先减掉这些）

**任何失败在算成回归之前，先确认它在分支起点是否也失败。** 分支起点是
`9eb696d9`（`origin/2.0`，即 `merge-base origin/2.0 origin/2.0-refactor`）。
没有这份清单，会把继承来的失败当成新回归、把责任安到无辜的提交上。

### A. 分支起点就存在的失败——不是任何 agent 引入的

| 用例 | 症状 |
| --- | --- |
| `tests/compat/torch/test_torch_compat.py` | `RandomOp` 子进程段错误 |
| `tests/data/test_dataset.py::TestDatasetSeed::test_children_died` | **worker 被杀之后 Dataset 不快速退出。** 子脚本 `dataset.workers[0].p.kill()` 之后，父进程应当收到 SIGCHLD 并 quick exit（用例断言 stderr 里有 `SIGCHLD` 与 `quick exit`），实测父进程一直阻塞到子进程超时。单独跑、空闲机器上稳定复现，起点 `9eb696d9` 同样失败（前任两步回退确认）。现已 `xfail(strict=True)`，仍然每轮跑、仍然可见，但不再让门禁整体变红；修好的那天 strict 会把它变红提示删标记。**2026-09-03 补充（bindings，做 507a0f1f 时查清）**：这一条与「worker 抛异常后父进程挂死」是**同一个洞的两半**。抛异常那半已修（worker 写好错误后 `buffer.stop()` 再推 id，父进程两处阻塞都能被叫醒）。这一条是 `p.kill()`：SIGKILL 的 worker 跑不到任何 Python 收尾代码，不写共享槽、不 stop、不推队列，而父进程阻塞在 `RingBuffer::pop` 里**握着 GIL**（`py_ring_buffer.cc` 全文没有 `Py_BEGIN_ALLOW_THREADS`），所以 Python 看门狗线程也跑不了。真正的修法是让这两个阻塞调用释放 GIL 并带超时（核心 C++ 改动，需单独立项）；**不要**退回旧的「SIGCHLD → 父进程 quick exit」，那是 6.C31 拆掉的无声消失 |
| `tests/core/test_array.py::TestArray::test_memcopy_overlap` | **墙钟阈值型 flake，非回归。** 断言是 `t2-t1 < 0.010`——「重叠版比纯计算版慢不超过 10 毫秒」，一条**绝对**墙钟阈值。机器常驻十几个 agent、负载 24 时它必然超。两个分区各自独立确认：内存分区在**未打补丁的树**上跑两次失败，绑定分区独立得出同一结论。**归责方向和真回归相反**：真回归查代码，这条查负载 |
| `tests/compiler/test_atomic_tuner.py::TestAtomicTunerClass::test_atomic_tuner` | 第 4 项 `x.sum()+x.sqr().mean()` 期望两条 `atomictuner: move atomicAdd to loop -1`，实得 0 条。根因是 `032ecfe1`（2026-08-28，起点前 202 个提交）把 CUDA 全量归约改走 `nn/backends/full_reduce_cuda.py` 的 cub 两级折叠 code op，整条全归约不再进融合算子 JIT，AtomicTunerPass 根本看不到 atomic 语句。前三项 add/max/min（reindex_reduce）在起点与起点父提交上都通过 |

（这份表正在用一棵钉在 `9eb696d9` 的只读 worktree 实测补全，跑完会把失败 nodeid 逐条列全。）

**`test_atomic_tuner` 已定论，`9eb696d9` 洗清。** 两棵只读 worktree、两份独立 `JITTOR_HOME`、同一条用例、**串行**跑（并行会串号，见 skill）：`9eb696d9^`(`a88ae02a`) 与 `9eb696d9` 的失败**逐字一致**——同为第 69 行第 4 项 `AssertionError: (0, 2)`。WarpReducePass 挂在 `pass_manager.cc` 的 `AtomicTunerPass` **之后**，原子调优早已打完日志才轮到它改写，它不可能吃掉这些日志。真正的原因见上表那一行：`032ecfe1` 的全归约快路径绕开了整个 JIT。**该用例现在断言的是一条已经不存在的代码路径，属于过期断言，不是回归。**

### B. 已归责，按表中状态处理——不要重复归因

| 用例 | 引入提交 | 责任 |
| --- | --- | --- |
| `tests/compat/torch/test_torch_compat_interpolate.py::TestInterpolateBicubic::test_bicubic_constant_stays_constant` | `13ac1d14` [6.C05] | coreops，正改成变长编码（原属 3.02） |
| `tests/compat/torch/test_torch_compat_autograd.py::TestCustomFunctionCompatibility::test_torch_style_function_keeps_context_and_broadcast_grad`、`test_torch_compat_autograd_semantics.py::TestSavedTensorVersions` 两条 | `5c4e624b` [5.07] | autograd。**已实测确认**：把工作区还原到 `origin/2.0-refactor` 干净树（`git diff > .patch` + `git checkout --`，不用 stash）后三条同样失败，与其他分区改动无关。一次性上下文使 `execute` 里写在 `self` 上的属性（`self.seen_needs_input_grad`）不再留在用户持有的实例上 |
| `tests/structure/test_pytest_contract.py::test_test_modules_avoid_collection_time_backend_side_effects` | `93b48a8e` [4.02 3/3] | 4.02。`tests/backends/cuda/test_device_copy.py:44`、`tests/backends/cuda/test_multi_device.py:34,35`、`tests/compat/torch/test_multi_device.py:37,38` 在**模块体**里调本地 helper `_device_count()`，收集期就去问设备数。改法：挪进 fixture，或用可调用形式的 `skipIf` |

（`test_runtime_composition_structure.py::test_moved_scope_state_stays_synchronized_with_the_root`
已由 `90fc2f0b` 修复并移出本表：2026-09-03 复测 `JITTOR_TORCH_SHIM=1` 下该文件 13 passed。）
| 并发编译读到写了一半的 `.s`：原生 `tests/data/test_dataset.py::TestDataset2::test_dataset_use_jittor`（`.so` 里 `undefined symbol: SetitemOp::jit_run`）+ CPU torch 536 条散布失败（`op.s` 报 `unknown pseudo-op .lasf10` / `invalid operands (*UND* and *UND*) for -` / `junk at end of line`） | `70d97137..a12d81c0` 之间的构建改动 | **已由 `1919b035` 修复并复验**：原子替换专用回归与原四 worker Dataset 用例均通过 |
| `tests/structure/test_nn_structure.py::TestModuleBoundaries::test_first_import_paths_are_cycle_free_in_fresh_processes` | `46dbe946` [0.21] | gates，**已修**：收编子进程调用时把调用点原来的 `env.pop` 丢了。旧写法自己 pop 掉四个 `JITTOR_TORCH_*` / `REAL_TORCH_SITE`；改走 helper 后 `env=` 是**叠加**在 `os.environ` 上的，叠加无法删除，于是四个变量原样回来了——而 `tests/structure` 自己就在 `TORCH_MODE_PATHS` 里，父进程带着 `JITTOR_TORCH_SHIM=1`。子进程先塞了个假 `torch` 模块再 import jittor，于是报 `cannot install Jittor Torch compatibility over an existing Torch module graph`。修法不止是加 `inherit=False`：helper 现在**拒绝**「完整环境 + inherit=True」这种有歧义的调用（认 `PATH` 在不在里面），并提供 `without_torch_mode=True` 把这四个变量的清理写进 helper，不再让每个调用方自己记得 |
| `tests/structure/test_stage2_delivery.py::TestStage2Delivery::test_nox_keeps_fast_structure_and_packaging_separate` | `6adbf488` [0.04] | gates，**已修**：断言 noxfile **源码文本**里出现 `tests/optim/test_optimizer.py` 等具体路径，而清单已搬进 `gate_scope.py`。这两条改成从 `gate_scope` 求门禁选择集再判断；仍是清单的那六条 oracle 路径继续按文本断言（"要跟真 PyTorch 对拍"是测试的属性，不是树的属性） |

### C. 先看这条：把 `tests/compat/torch` 和原生目录写在同一条 pytest 命令里，会让原生用例整片变红

这是**跑法造成的假失败**，不是回归，也不在上面 A 表里。`tests/conftest.py`
的 `_select_torch_mode_for_test_process()` 按 `sys.argv` 选模式：

- 选择「宽」（只写 `tests` 或 `.`）→ 原生模式；
- 选择「窄」且其中**任何一个**路径命中 `TORCH_MODE_PATHS`（含 `tests/compat/torch`）
  → 给**整个进程**设上 `JITTOR_TORCH_SHIM=1`。

所以

```bash
pytest tests/core tests/nn tests/optim tests/structure tests/distributed tests/compat/torch
```

会把 `tests/core`、`tests/nn`、`tests/optim` 全部拖进 torch shim 模式跑。conftest 自己的注释
写得很清楚：torch 模式是进程全局的，会改掉惰性求值、归约默认值和梯度语义，
**「switching the whole tree into it made ordinary native tests fail」**。

**正确跑法是拆成两条命令**，和门禁的划分一致：

```bash
JITTOR_TORCH_SHIM=0 pytest tests/core tests/nn tests/optim tests/distributed   # 原生
JITTOR_TORCH_SHIM=1 pytest tests/structure tests/compat/torch                  # shim
```

拿着一把 F 又不确定来源时，**先按这条拆开重跑**，再来对 A 表。

## 门禁 agent 的最新结果

三套门禁的判据不是 `failed == 0`，而是：`failed` 不超过上面 A 表，且 `passed` **不下降**。
整改期间每个 agent 都在加测试，所以 passed 一路上涨是正常的，起点那行的数是下界不是等号。

| 提交 | 原生 | CPU torch | CUDA | 失败用例 / 责任任务 |
| --- | --- | --- | --- | --- |
| 9eb696d9（分支起点） | 775 passed / 765 skipped | 1595 passed / 285 skipped | 574 passed / 9 skipped / 0 failed | — |
| `70d97137` | **822 passed / 816 skipped / 0 failed**（50 分） | 未完成（运行中断） | 未开始 | 原生绿；收集总数 1540 → 1638，是有人加了测试 |
| `a12d81c0` | 1039 passed / 866 skipped / **1 failed**（51 分） | 1207 passed / 536 skipped / **536 failed**（59 分） | 未跑 | 两套的失败**同一个根因**：并发编译读到写了一半的 `.s`（详见下方一行）。`passed` 两套都在涨，判据里下降的那一半没触发 |

## 热点文件占有（历史）

第一波（2026-09-02 21:xx 派出）：九个 agent 并行，各自独占一张卡/一段核/一个 worktree
（`/home/zy/jittor-lab/refactor/<分区>`，分支 `wk/<分区>`，推送到 `2.0-refactor`）。

| 分区 | 第一波占有者（历史） | 任务 |
| --- | --- | --- |
| 核心节点 | coreops (2.01/2.02/2.03) | GPU3 c24-35 |
| 执行器 | — |  |
| 代码生成 | — |  |
| 类型与日志 | coreops (6.C01/05/06/07/09/30) | GPU3 c24-35 |
| 内存 | — |  |
| 绑定 | bindings (6.C02/22/23/24/25/27/28/29) | GPU0 c0-11 |
| Python 核心 | — |  |
| Python 算子 | — |  |
| Python 其他 | pyother (5.18、5.19、6.P25) | GPU6 c64-75 |
| 兼容层 | compat (7.01) | CPU c96-103 |
| CUDA 后端 | cudabk (6.B07/08/17、8.01、8.03) | GPU7 c76-87 |
| ACL/ROCm/Corex | — |  |
| 分布式 | dist (6.B01/03/04/06/10/11/15) | CPU c88-95 |
| 构建 | build (0.07–0.11/0.17, 9.02–9.06/9.08/9.09/9.11/9.15/9.17, 9.04 部分) | GPU4 c48-63 |
| 门禁 | gates (0.01–0.04/06/12/13/18/19) | CPU c104-111 |

## 执行中出现的、需要认领的杂项

| 事项 | 现象 | 建议归属 |
| --- | --- | --- |
| 全树只有两条「性能断言当正确性门禁」 | 全树扫了一遍时间断言（`time.time()` / `perf_counter` 的 16 个文件逐个看过）。**上界型**（负载会让它变红）只有两条：`tests/core/test_array.py:81` 的 `t2-t1 < 0.010`（绝对毫秒阈值，就是 A 表那条）、`tests/core/test_nano_string.py:28` 的 `nano_time < builtin_time * 1.25`（**相对**比值 + 5 次取最小，抗噪好得多，但仍是上界）。其余全是**下界型**——`test_nccl_rendezvous_timeout.py:188/249` 的 `assertGreater(elapsed, TIMEOUT)`、`test_torch_compat_unimplemented.py:635` 的 `assertGreater(elapsed, 5.0)`、`test_tracer.py:69` 的 `elapsed < 30`（2 秒超时的 15 倍余量，判的是"有界 vs 无界"）——负载只会让下界更成立，不会误伤。**结论：这类假红的来源只有一条，不是一类。** 修法应是把绝对阈值改成相对比值或改成有界性断言，不是放宽数字 | 门禁 gates，随 0.15 |
| wheel 内容基线过期 | **已处理**：基线本身没错，错的是那条断言写死了条目数——任何人加一个模块都会改变它。用当前源码树真构建一个 wheel 核对，45 个新增全是各分区加的合法源码、1 个删除是 9.17 删的 `flags.cc`，基线整个重新生成（861 条）；条目数那条断言换成「基线头部的 `# entries:` 必须与自身条目数一致」这条规则。注意基线是对**当时的源码内容**取的哈希，发版前需要再刷一次（办法写在提交说明里） | 构建，`f869cab8` |
| 结构测试子进程超时 flaky | 已处理，并已与 0.21 合并成一份实现：超时预算搬进 `tests/_helpers/child_process.DEFAULT_TIMEOUT`（600s，`JITTOR_TEST_CHILD_TIMEOUT` 与旧名 `JITTOR_TEST_SUBPROCESS_TIMEOUT` 都认，仍在门禁 `--timeout=900` 之内），`process_modes.SUBPROCESS_TIMEOUT` 随之删除——同一件事只留一处 | 门禁 gates，`46dbe946` |
| `(void)x;` 的识别曾按「语句含 void 一词」 | 已修（3.09，`66e5a153`）。原判据会把 `memset((void*)p,0,n);` 整条删掉：编译通过、缓冲区没清零、静默算错。今天没被咬到只是因为树里没有算子往融合 kernel 里写带 void 转型的语句 | — |
| `UnrollPass` 与 `ExpandEmptyBlockPass` 同名 | 已修（3.14，`6c899325`）。`exclude_pass` 与 `pass_map` 都按名字索引，`emplace` 不覆盖，后跑的那个根本没进表；`get_pass` 还会把它 C 风格强转成另一个类型。g++ 构建上 UnrollPass 根本不跑，所以运行期校验抓不到，用例改成实例化 30 个 pass 比名字 | — |
| 3.13 前置核实：`range10` 今天真的会出现，且 `loop_id` 在合并后已经与循环变量对不上 | 实测（CPU，8 维加 3 次 split、9 维加 2 次 split 起）生成源码里确实出现 `range10`；我试过的 20 组（元素级/归约/广播 × 7–10 维 × 0–4 次 split）**数值都还是对的**，所以这条今天是潜伏的，不是已经在算错。两条具体机制：(1) `merge_loop_var_pass.cc` 用 `size()==6` 判断「是单个 range」、用逐字符拆分把 `range23` 展开成 `range2*range3`，对基础的 `range10` 会拆成 `range1*range0`；更危险的是合并出的新 id 是 `aid+bid` 字符串拼接，合并 1 和 0 得到的 `range10` 与基础 `range10` **同名**，而代码里 `if (!find_define(new_range)) push_back(定义)` 恰好会跳过定义、直接复用那个基础 range —— 循环上界完全错误且能编译。(2) 合并后 `loop_id` 变成拼接串，但循环的归纳变量仍是 `id{bid}`，而 `restride_pass.cc:50` 用 `"id"+fa->attrs["loop_id"]` 去找它 —— 合并后这两者已经对不上了。所以 3.13 不是改一个函数，确实需要计划里说的「loop id 用整数向量、名字只在输出时生成」 | 代码生成分区，3.13 |
| 3.10 前置核实：改名的默认动作是「白名单之外一律加 `op{i}_` 前缀」 | `op_compiler.cc` 的 `unchanged` 只有 24 项（`for/const/auto/int/float/bool/void/if/true/false/...`），`return`、`else`、`while`、`static`、`unsigned`、`char`、`long`、`double`、`size_t`、`int64`、`uint`、`nullptr` 全都不在里面，会变成 `op0_return` 这种。因为这些写法今天写了就编不过，**现有算子里一个都没有**，所以把它们加进「不改名」集合对现存代码是行为不变的（只会让新写法能用）。建议的形状：完整的 C++ 关键字集合 + 运行期已知类型（`op_type->types` 已有）作为不改名集合，再加一条改名后的合法性校验 —— 两个相邻的被改名标识符（`op0_A op0_B`）几乎只能是「把类型名改了名」，直接报一条指名道姓的错误，而不是把 `op0_size_t 不是类型` 丢给 C++ 编译器 | 代码生成分区，3.10 |
| 3.13 结论：合并 range 的命名冲突今天**不可达**，两条别处的巧合各兜着一半 | 3.13 已合并（`bd5b5a67`）。前置核实那条说的命名冲突（合并 1 和 0 得到 `range10`，撞上基础/split 的 `range10`）**构造不出来**：我用 10 维 + `split9` + `order1=1`/`order{i}=2` 把循环 1、0 排到最内层，合并 id 恰为 "10"、`range10` 也确实存在——但 reorder 之后 1 与 0 在**内存序**上不再相邻，`expr::match` 直接失配，合并根本不发生。也就是说合并 id 永远是按嵌套序递增拼接，而嵌套序就是内存序。另外两条兜底：`NanoVector::push_back_check_overflow` 断言 `s<10`（张量最多 10 维，基础 range 只到 `range9`），以及 **split 出来的循环永远进不了合并**（父循环 `inner` 里多一条 range 定义，过不了 `inner.size()==3`），所以所有合并 id 的每一段都是一位数、逐字符拆分恰好是对的。这三条都是别处代码的巧合，不是这里声明的前提——已改成 `parse_loop_id`/`format_loop_id`，段间用 `_` 隔开 | 代码生成分区，3.13 |
| 3.15 结论：pass pipeline 跑两遍是真的，但只占首次执行的 0.04%–0.22%，不值得改 | 3.15 已合并（`97cac22f`）。加 `LOGvvv` 计时后实测：元素级 CPU parse 72us / passes 627us / to_string 38us，而首次执行 1012ms；matmul CPU 两遍合计 4.6ms，首次执行 2080ms；CUDA 元素级 pipeline 0.95ms、首次执行 3871ms。**其余全是 g++/nvcc**。计划提的「一次解析后 clone IR」只省得到第二次的 parse（~170us），而 clone 与 parse 是同一量级、没有 tuner 自信时那一份 clone 纯属白做——最好情况省 0.008%，代价是给 KernelIR 引入一条 father/scope 指针容易出错的复原路径。**真正做了的两件**：ReorderTuner 的候选从 N!（10 维 + 3 次 split 实测 order0..order12、乘积 **3,628,800,000**）改成按 `jit_search_max_candidates`（默认 1024）截断；`Searcher::timeout` 从「声明了没人读」改成由 `jit_search_timeout` 设置并真正生效 | 代码生成分区，3.15 |
| 3.11 结论：relay kernel 里的字节偏移，主路径上被 `cache_compile` 的内容哈希兜着，但 `rewrite_op=0` 时真可达 | 3.11 已合并（`f32d3c83`）。`get_relay_src` 生成 `GET_VAR_MEMBER(rop_0_0, 120) = vars[2].var;`，120/128/136 是算子结构体的字节偏移，写进 JIT 缓存复用而 jit key 里没有布局信息。主路径上 `cache_compile` 把生成源码的内容哈希算进缓存键，偏移一变就重编；但 `rewrite_op` 是个 flag，设成 0 时已存在的 `.cc` 不重写，旧偏移与自己的哈希一致、不重编——这条路真可达。已改成按名字 `set_var_member("a", ...)`。另外审计说「偏移由 compiler.py 用正则扫头文件得出」**不准确**：正则只扫**名字**，偏移是 `offsetof` 在 C++ 编译期算的；但那个正则确实有一种哑失败（`jittor::Var* x;` / `const Var* x;` 扫不到，成员不进注册表、relay 时永不绑定），已抽成 `compiler.parse_var_members` 并让读不懂的写法**构建失败** | 代码生成分区，3.11 |
| 3.10 结论：扩充「不改名」集合对现存代码逐字节不变 | 3.10 已合并（`864fa52c`）。58 组生成源码（元素级 1..8 维 / 五种 dtype / 一元链 / 六种归约 / broadcast / ternary / index / transpose 融合 / 六组 loop option，CPU 与 CUDA 各一遍）改前改后逐字节相同。新增的合法性校验（两个相邻的被改名标识符只可能是「类型名被改名了」）要放过算子自己 `#define` 出来的宏——`index_t`、`Tx`、`T` 都是宏，`op0_index_t op0_i` 是**正常**生成结果 | 代码生成分区，3.10 |
| 冷缓存下跑整套 `tests/compiler` CUDA：会多出几条只在这种条件下红的用例，而且跑到 `test_probe_cache` 附近**无声退出** | 2026-09-03 实测：`rm -rf $JITTOR_HOME/.cache` 之后跑 `tests/compiler` CUDA 全套，`test_cache_path_precedence::test_the_imported_core_is_the_cuda_one`、`test_console`、`test_custom_op` 两条、`test_fused_identifier_rename::test_reserved_identifiers_are_not_renamed` 共 5 条红；把这 5 条**单独**拿出来在同一棵树、同一张卡上跑是 **7 passed**。同一棵树的热缓存轮此前也只有一条失败（且是用例自身问题，已修）。之后进程在第 302 条（`test_probe_cache`）附近**没有 summary 就退出**（`EXIT=1`，pytest 零输出）——这正是简报 §7 与任务 6.C31 记的那个模式：子进程被信号杀死，jittor 的进程级 SIGCHLD 处理器让父进程无声退出。**判据**：整套红了一片而单独跑全绿、且日志没有 summary，先看是不是冷缓存 + 子进程信号，不要当成代码回归 | 代码生成分区，验证方法 |
| `split{i}` 与 `parallel` 不兼容 | 同时设这两个 loop option，`ParallelPass` 在 `ASSERT(def)` 上失败（`Check failed: def`）。`SplitLoopPass` 给内层循环的 range 是 `::min(range{i}-id{i}, stride{i})`，定义在外层循环里且随它变化，`ParallelPass` 在调用点 `func->find_define` 找不到、也无法在调用点求值。CUDA 恒走 `ParallelPass`，所以 CUDA 上任何 split 候选都必然编译失败。用例已钉住：`tests/compiler/test_reduce_tuner.py::test_a_split_candidate_would_not_compile_under_parallel` | 代码生成分区，1.04 的前置 |
| CUDA 归约需要的是线程分解候选，不是 CPU 那套 | `orderN` 候选实测五种形状全部不优于默认（最差 2.1 倍，破坏访存合并），`split{i}` 被上一条挡着，L1 分块尺寸对 GPU 无意义。真正有用的候选是 `ParallelPass` 里的线程分解，属于新工作 | 代码生成分区，待 1.04 前置解决后 |
| `para_opt_level=4` 的块内共享内存归约比默认慢 1.6–2.0 倍 | 实测四种 UNet 形状：默认（warp shuffle）15.7/14.0/15.0/18.1us，lvl 4（`SharedReducePass`）25.3/31.3/25.3/34.8us，不优化 157/92/159/171us。默认值保持 3。要提升需要「warp shuffle → 每 warp 一个值 → 共享内存 → 每输出一次原子」的混合实现，并且要有生态 harness 的端到端数据；数据与方法在 `agent/skills/cuda-reduction-strategy-comparison/` | 代码生成分区，新任务待派 |
| `tests/structure/test_source_root.py` 3 条 | `AttributeError: module 'conftest' has no attribute 'source_python_dir'`——有人从 `tests/conftest.py` 删/改名了 `source_python_dir`，没同步用例。与并发编译无关，是独立真回归 | 改 conftest 的那个提交（0.13/0.17 一带），门禁已记录 |
| `tests/compat/vllm/test_flash_attn.py::TestTheBundleItPublishes::test_a_submodule_it_does_not_carry_still_imports` | `ModuleNotFoundError: No module named 'vllm.vllm_flash_attn.layers'` | 兼容层分区（7.14 一带） |
| `tests/structure` 在一次 session 里被收集两遍 | `-rf` 摘要里 `test_source_root.py` 的三条各出现两次 | 门禁分区，随 0.04 查收集规则 |
| `tests/core/test_type_system.py` 一套门禁都不跑 | **已修**：0.04 之后 CPU 门禁的 torch 会话就是 `TORCH_MODE_PATHS` 本身，这个文件自然进来了。同一批还有 233 个此前一套 workflow 都碰不到的文件 | 门禁 gates，`6adbf488` |
| `test_atomic_tuner` 抓不到日志 | **已修**：根因确认为 `032ecfe1` 的 `full_reduce_cuda.py` 快路径猴补 `Var.sum`/`Var.mean`，全归约不再进 JIT；第 4 条用例改走 `jt.reduce` | codegen，`72f020b3` |
| `asm_tuner.py` 非原子写 `.s`，并发编译读到截断的汇编 | **已修**：`pass_asm()` 改成写 `<路径>.tmp.<pid>` 再 `os.replace`。判据是 inode——改名换 inode，原地重写不换，也就不会消掉那个窗口；用例 `test_asm_tuner.py::TestAsmTunerWritesAtomically` 钉住。缓存里已经存在的坏 `.s` 不会自动修复，删掉再跑 | 构建，`1919b035` |
| `tests/backends/cuda/test_backend_teardown.py` 过不了 0.21 的静态门禁 | **已修**：gates 在 `a5ce7310` 里已改成 `run_python_child(..., crash_isolated=True)`。cudabk 复核了改后的文件保留全部断言（无 terminate / 退出码 0 / 有 teardown 记录 / 真错误 `cudaErrorIllegalAddress` 仍可见）加那条干净退出的对照，并把 `cuda-backend-choice-proof` 里「子进程 abort 会带走 pytest」那段从只描述现象改成指向 helper 的 `crash_isolated`（`1b117a91`） | 门禁 gates，`a5ce7310` |
| `tests/compat/torch` 的 17 条失败与核心分区无关 | 2026-09-03 在 `77641cc8` 上跑 `test_torch_compat_optim/rnn/unimplemented/linalg` 四个文件：17 failed / 104 passed。逐条看过失败原因，**没有一条落在 flag、节点、遍历或执行器上**：AdamW 八条全是 `fused_adamw is only available through a mapped backend`（`ops/fused_adamw_op.cc` 在 CUDA 上拒绝执行）；RNN 五条全是 `cudnn_rnn_descriptor.cc Check failed: is_type<string>(_slots[7])`；`set_default_device` 两条是设备选择；`test_det_slogdet` 是 cupy 的 `NVRTC_ERROR_COMPILATION`；`test_autocast_actually_lowers_op_dtype` 是 shim 的 autocast 没接上 amp——**这条专门核过**：同一个 amp 探针（level 5 下 `a*b`、`matmul` 的 dtype，以及 AmpGradGuard 把六位 amp 字段读回来后的梯度）在 2.01 前后逐字一致（`float16/float16`、`GRAD float32 128.0`），2.01 的重新编号没有动 amp。同一批失败在 rebase 前的树上也是同一份名单 | 依次为 pyops/compat、cudabk（8.01 一带）、device（4.02 一带）、环境、compat |
| 下一次 rebase 会全量重编一次 | 9.04（`2569fe3b`）同时改了缓存路径与缓存键格式，**这是预期的，不是缓存坏了，不要删自己的 `JITTOR_HOME`**（本机冷构建约 63s）。另外 `cache_name` 的语义从「不设 = 当前 git 分支」变成「不设 = `default`」——靠分支自动分开缓存来隔离并行任务的，改成显式设 `cache_name` 或不同 `JITTOR_HOME`；反过来切分支不再触发全量重编 | 全体，已由协调者广播 |
| 8.03 的前期分析（未实现，交接用） | **已落地**，见 8.03 与 `agent/design/float32-precision-policy.md`（三档映射表、默认值为何不变、两条实质行为变化各自的证据）。 | 已完成 |
| 7.08 的 tf32 映射可以再进一步（8.03 之后） | 9aaedba9 把 high/medium 的细分记在 Python 侧（`_torch_float32_matmul_refinement`），理由是「Jittor 表达不了」。8.03 之后 `jt.flags.float32_matmul_precision` 是真的三档 C++ 状态，`medium` 会真的走 bf16 累加，**表达得了了**。但接上去之前要先决定：Jittor 这个策略是 matmul 与卷积**共用**的，torch 的 `set_float32_matmul_precision` 不动 cuDNN；直接接到共用策略上会让下游一句 `set_float32_matmul_precision("high")` 把卷积也降到 tf32。要么 shim 只写 per-domain 覆盖（现状，medium 仍然只到 tf32），要么核心再分出 matmul-only 一档。cudabk 没有替 7.08 做这个决定 | 7.08 接手人 |
| cuDNN 卷积计划缓存缺一个观测点 | `cudnn_conv_plan.h` 的 plan 缓存没有任何 Python 可见的读数，所以「某个字段确实进了缓存键」只能靠规则测试（源码里不许手写 `req.` 赋值）间接钉，没法直接断言。cuFFT 与 cuTT 都有 `*_plan_cache_size()` / `*_set_plan_cache_size()`（6.C 那批加的），照抄一份 `cudnn_conv_plan_cache_size()` 就能直接断言「同形状二次调用只 +1」「只翻 `cuda_allow_cudnn_tf32` 会多一条」。**数值路线试过、不成立**：允许 tensor-op 只是让那些 engine 可选，cuDNN 仍可能挑 FMA——实测 32×32×24×24 的 fp32 卷积，tf32 关 4.58e-05、开 5.91e-05，差 1.3 倍，做不了判据。注意缓存是 header 里的 inline 函数静态量，而 `jit_run` 编在另一个 .so 里，加访问器前要先确认是同一份实例（`fwd_algo_cache` 那种 EXTERN_LIB + 非 JIT 段定义是现成的写法） | 后端，随 8.07 后续 |
| 8.05 的前期核实（未实现，交接用） | **四条前提今天全部成立，逐条核对过**：(1) **版本钉死**：`jittor_utils/manifest.py:41` 钉 `dnnl_lnx_2.2.0_cpu_gomp.tgz`（2021 年的 oneDNN v2.2，带 sha256/md5）；`compile_extern.py:59` 靠 `lib/libmkldnn.so` 判断解包是否成功，而解出来的 v2.2 目录里 `libdnnl.so` 与 `libmkldnn.so` 两套名字都在——v3 只有 `libdnnl.so`，**换库不先改这一处，表现是「下载了但认为没装上」**。`_asset()` 支持显式 `url=`（CUB 就是指向 github codeload），所以不必往镜像上传资产。(2) **用了 v3 已移除的 API**：六个算子里全是 `convolution_forward::desc` / `convolution_backward_data::desc` / `convolution_backward_weights::desc`，v3 删了 `*::desc`，primitive_desc 直接从参数构造。(3) **每调用重建**：`mkl_conv_op.cc:119-120` 每次 `jit_run` 都 `engine eng(cpu,0)` + `stream s(eng)`，后面 memory desc / primitive desc / reorder 目标 memory 全部重建；三个 conv 与两个 matmul 都是这个形状。(4) **matmul 只支持 fp32**：`mkl_matmul_op.cc:28` 一句 `ASSERT(dsize()==4)`，而 CUDA 侧 fp16/bf16/fp32/fp64 全支持，这种后端能力差异今天无处声明。 | **另有一条计划里没写的**：前向 `mkl_conv_op.cc:153` 用 `prop_kind::forward_inference`，两个反向算子（`mkl_conv_backward_x_op.cc:133`、`mkl_conv_backward_w_op.cc:135`）构造 backward pd 所需的 hint 时却用 `prop_kind::forward`（即 `forward_training`）。**前向与反向 hint 对 prop_kind 的说法不一致**；oneDNN 要求 backward pd 的 hint 来自 forward_training 的 forward pd，真实前向用 inference 时 oneDNN 给前向挑的 layout 可能与反向 hint 假定的不同，代价是多一次 reorder。这就是 8.05 里「训练用 forward_training」那条的落点。**做的顺序建议**：先补测试再动代码——`tests/backends/cpu` 今天只有 **5 条**（conv 4 + test_op 1），撑不住一次换库；先加 MKL 卷积与矩阵乘对 reindex 参考路径的对拍（形状/groups/stride/dilation/dtype 矩阵），再 (2) 迁 v3 与 (1) 放开版本，最后 (3) 缓存与 (4) 能力表——**缓存写在 v2 API 上会被 v3 迁移全部重写**。缓存要注意 memory 对象包着 `x->mem_ptr`，每次调用指针会变，所以缓存的是 pd 与 primitive，执行前用 `set_data_handle()` 重新绑指针。 | 下一位接手 8.05 的人 |
| `jt.flags.nvcc_flags` 的拼法变了 | 9.08 之后架构 flag 是 `--generate-code=arch=...,code=...`，不再是 `-arch=compute_N -code=sm_N`。按后者做字符串匹配的地方要改 | 各分区自查，`2d71f792` |
| 全树跑时 `test_notebooks.py` 没有被当成 manual 跳过 | **已修**：`pytest_collection_modifyitems` 里 `test_notebooks.py` 的 `pytest.mark.manual` 加在跳过判断**之后**，所以全树跑时它照跑不误——2026-09-03 的全树原生一遍里实测 537 秒，是全树最慢的一项（第二名 289 秒）。现在所有标记先挂完再统一判断，manual 探针改由 `JITTOR_TEST_MANUAL=1` 或 `-m manual` 显式打开。**这是「筛选逻辑的顺序决定筛选结果」的第三例**（另两例：按 `sys.argv` 选 shim 模式、`@onlyCPU` 被设备过滤全部跳过） | 门禁 gates，`5c0f2364`（0.13） |

## 跨用例状态泄漏清单（0.15 的前置，2026-09-03 全树实测）

`tests/conftest.py` 在**每个测试文件**前后拍一次快照（三个存活计数、六个关键 flag、
`sys.modules` 里换了对象的条目），只报告不失败。全树原生一遍的结论比预期干净得多：

| 文件 | 留下什么 | 处置 |
| --- | --- | --- |
| `tests/nn/test_nn_capabilities.py` | `number_of_hold_vars 0 → 7` | **不是测试留下的**，见下 |
| `tests/ops/test_fft_op.py` | `number_of_hold_vars 0 → 26` | **不是测试留下的**，见下 |

**这两条已定论，而且原来的解释（「模块级留着 Var」）是错的。** 实测手法：跑完文件后
依次丢掉测试模块、`_pytest`、`_helpers.common`，每步 `gc.collect()` 再读计数——
**三步之后计数一个都没掉**（26 → 26 → 26）。再用 `jt.dump_all_graphs().hold_vars`
把它们逐个打出来，形状说明了一切：

- `test_fft_op.py` 的 26 个是 **13 对** float32 `[n,n]`（n=1..12），正是
  `python/jittor/fft/__init__.py` 的 `_dft_mat_cache`——按尺寸缓存的 DFT cos/sin 矩阵对，
  `OrderedDict` LRU，`_dft_mat_cache_limit = 16` 对。
- `test_nn_capabilities.py` 的 7 个是 int32 一维小向量（`[2] [3] [3] [3] [4] [3] [3]`），
  正是 `python/jittor/nn/attention.py` 的 `_CU_SEQLENS_CACHE`（cu_seqlens 前缀和），
  同样是 LRU，`_CU_SEQLENS_CACHE_LIMIT = 128`。

**两个都是 jittor 自己的、有上限的进程级 memoization，不是泄漏，测试这边没有东西可改。**
真正的结论是那条一般规律有了具体机理：**`number_of_hold_vars` 有一个下界，取决于
这个进程曾经跑过哪些算子，而不取决于当前这条用例**——所以对它做绝对断言按构造就是错的。
`tests/_helpers/state_leaks.py` 现在把这两个缓存的条目数一起快照，报告会直接写
「26 个里有 26 个是 `jittor.fft._dft_mat_cache` 0 → 13 条（上限 16）」，剩下的差值
才是值得去查的东西。

**六个 flag（`use_cuda`/`no_grad`/`amp_reg`/`use_parallel_op_compiler`/`exclude_pass`/`th_mode`）
在原生这一遍一个都没泄漏**，`sys.modules` 也没有未还原的替换——0.12 那一批修到位了。

### 受害者一侧才是要改的地方

已知的五个"单独跑绿、合跑红"样本里，机理清楚的三个都不是污染源的错，是**受害者对全局
计数器做了绝对断言**：

| 样本 | 状态 |
| --- | --- |
| `test_fused_op.py::TestFusedOp::test_add` | **已修**（`bffe0bf4`）：断言 `(hv,lv,lo) == (0,0,0)` 改成比用例开头的基线增量。它真正想断的是"这张图创建了几个节点、融合后活下来几个"，那是一个差 |
| `tests/ops/test_linalg.py::TestBUG4_2Op` | **已修**：`use_cuda=1` 改 `@jt.flag_scope`（0.12 / 6.P23） |
| `tests/compiler/test_jit_tests.py` 的两条 sfrl | **已标记**：墙钟阈值，改 `@pytest.mark.load_sensitive` |
| `test_torch_compat_fsdp2::test_single_rank_fully_shard_preserves_math_and_state` | 待查（torch 会话） |
| `tests/compat/torch/test_torch_compat.py::test_torch_compat` | 待查（torch 会话；单独跑 549s 通过，整套里失败，且在兼容层那批改动之前的基线上就这样） |
| `tests/data/test_dataset.py::TestDatasetSeed::test_children_died` | **已定论，进 A 表**：单独跑也失败，恒在子进程超时上（300s）。不是泄漏，是真缺陷——worker 被 `p.kill()` 之后父进程不再靠 SIGCHLD 快速退出，而是一直阻塞等那个死掉的 worker 的数据。已改 `xfail(strict=True)` 加 `slow`，子进程超时从 300s 收到 90s（「快速退出」本来就该用更短的界来断言），门禁每轮从 302s 降到 95s。strict 意味着谁修好了它门禁会红，提示删掉这个标记 |

**一般规律**：对进程级全局量（存活计数、墙钟、flag）做**绝对**断言，断的不是这条用例的
性质。能写成增量就写增量，写不成就说明这条断言依赖一个它管不着的前提。

还有一类**绝对上界**留着没改，风险低但同形状：`test_inception.py:125`
（`lived_vars < 50000`）、`test_resnet.py:136/138`（`< 8100` / `< 7000`）。
余量是实测污染量（33）的两个数量级，暂不动；真要动就同样改成比循环前的基线。

> 8.06 补充证据：`ba8e2621` TruthReduce all/any 统一共享 launcher，静态合同 36 passed；本机无 CANN/NPU，仍待实机验证。

### 2026-09-04 第六十五波补充证据

- `8.06`：`3f0b8c7d` 将 GroupNorm forward 接入共享 launcher，保留 group/eps/三输出 query；静态合同 53 passed，本机无 CANN/NPU，仍待实机。
- `8.06`：`016fc62d` 将 GroupNorm backward 接入共享 launcher，保留 output-mask、group 属性、三输出 query 与 cleanup；静态合同 54 passed，本机无 CANN/NPU，仍待实机。

## 任务

| 编号 | 任务 | 状态 | 负责 | 提交 |
| --- | --- | --- | --- | --- |
| 0.01 | `TestGradients` 改用 `only_for=("cpu",)` 显式实例化 | 已合并 | gates | aee8ecaa（+355deb6e） |
| 0.02 | 设备过滤后 bases 为空或方法数为 0 时生成器直接 raise | 已合并 | gates | e5eb0d05 |
| 0.03 | `tests/compiler/test_jit_tests.py` 进 CPU 门禁，并断言 … | 已合并 | gates | a5e7f654 |
| 0.04 | 门禁改为「整个 `tests/` 减显式排除清单」，排除项必须写理由 | 已合并 | gates | 6adbf488、689e206b |
| 0.05 | 生态对拍进 nightly | 已合并 | gates | 97125c6e。`nox -s ecosystem` + `.github/workflows/ecosystem.yml`（每天 02:00）。**fail-open 是这里的真问题**：这些用例在 `REAL_TORCH_PYTHON` 没设时自我 skip（对的，拿 shim 和自己比证明不了什么），于是丢了 oracle 的 nightly 会**为它唯一存在的理由报成功**。三道闸：`JITTOR_REQUIRE_REAL_TORCH=1` 时把「缺 torch」从 0.18 的环境解释里撤走并让这类 skip 退出非零、逐条列出哪些对拍没发生；session 层缺 oracle 直接 abort；起手在 oracle 解释器里断言它的 `torch` **不是** shim（防的是最坏情况：两边其实是同一棵树，对拍全绿而什么都没证明——本机 jt311 的 `torch` 正是 shim）。本机验证（jt311 对 jt312b 的 torch 2.12.1）：12 个 CPU 对拍用例真跑、逐参数与逐输入梯度全过、8 分 23 秒 |
| 0.06 | `make_tensor` 种子改为 `hash(nodeid, shape, dtype)` … | 已合并 | gates | a4d041e6。稳定种子包含 nodeid/shape/dtype 等输入，失败信息报告 seed；6 条契约测试覆盖单跑/全量一致性，固定 seed 值不受进程哈希盐影响 |
| 0.07 | 缓存路径追加构建配置指纹 | 已合并 | 构建 | 82dfce6e、6379b2b5、6fdb3807、b25fcdfa（复验） |
| 0.08 | 锁统一为一种类型、一个 fd | 已合并 | 构建 | 460bead0 |
| 0.09 | 探测结果落盘 `cache_path/probe.json` | 已合并 | 构建 | 240a92a3 |
| 0.10 | 写缓存前检查可用磁盘空间，不足时给明确错误 | 已合并 | 构建 | 73eceeaf |
| 0.11 | 「jit_utils 已更新请重跑」改非零退出码 | 已合并 | 构建 | 7e8c7c74 |
| 0.12 | 14 处在用例里裸赋值 `jt.flags.*` 且无 tearDown 的测试改 `flag_… | 已合并 | gates | 26a20905 |
| 0.13 | conftest 的模式由显式环境变量决定，删除 `sys.argv` 嗅探 | 已合并 | gates | 5c0f2364、a4ebb31a。**日常影响**：手跑 `tests/structure`、`tests/compat/torch`、`tests/ops/test_ops.py` 等 `TORCH_MODE_PATHS` 下的路径要带 `JITTOR_TORCH_SHIM=1`，不带会得到一条指名变量的报错（而不是一次语义不对的绿）。`nox -s structure` 已经自己设了 |
| 0.14 | `_session_env` 不再 `os.environ.copy()` | 已合并 | gates | 6b8fb594。未声明宿主变量显式屏蔽，工具链/下载入口按白名单透传；OMP/MKL/OpenBLAS 等线程池固定并随 worker 缩放，子进程 probe 断言线程数与 CPU affinity。聚焦结构 18 passed，真实 nox probe 在受限 affinity 下通过 |
| 0.15 | 门禁分两层 | 待领 | | d957e4aa、9329c4f9、9f6a80c7、2fd26522 已合入：按实测慢文件拆出 smoke/full、并行度单点声明、PR smoke job 与 JIT cache 已接入。`876ec09c` 修正 RingBuffer worker-death 等待；独立 Dataset 两个 worker 监管 nodeid 在临时缓存下 2 passed/65.68 s，但完整 smoke 仍约 390 s、预算模型约 446 s，尚未达到原验收的 5 分钟；还需减少或降低有效测试工作量，不能靠扩大排除清单假达标 |
| 0.16 | `test_device_parity.py` 按算子分片并行，不再在 `setUpClass`… | 已合并 | gates | 120b004b。实测结论与原方案相反：4-worker 只快 6% 且 26 项丢 3 个结论，因此保留单进程；只移除错误的串行编译器强制关闭。后续真正压缩时长另见 0.22 |
| 0.17 | `pyproject.toml` 的 `pythonpath` 改由 conftest 按环境变… | 已合并 | 构建 | b19d098f |
| 0.18 | 门禁每条目断言至少执行 1 个非 skip 用例 | 已合并 | gates | ee29bee3、2f3f1aaf。恒 skip 的判据**从路径清单改成规则**：读测试自己写的 skip 理由，全都在说「这台机器缺某样东西」才算解释得通。清单版在这台机器上会是 73 条、换台机器又是另外 73 条，而且每加一个设备测试都要记得报到。规则一上线就抓出四个说不清自己缺什么的文件（`Not use cub, Skip`、`skip_this_test`），都改成说明缺什么，而不是给它们开豁免 |
| 0.19 | 结构测试从「精确清单」改成「规则」 | 已合并 | gates | c3bcd277 |
| 0.20 | 布局收尾 | 待领 | | ef31a0d6 已合入 1/N：删除 `tools/services/legacy` 的 converter launcher 与说明，清除 tools 活跃导航和 compat converter 对旧部署脚本的引用；converter 模块保留，HTTP 服务部署由应用负责。不存在结构节点 1 passed，仓库布局通过。`agent/design`/`agent/results` 权威树迁移、`tests/system` 删除及 AWESOME/ASV 归位均未做，保持待领 |

### 第101波增量证据（2026-09-04）

- `2.19`：`ba2f4077` 将 cuBLAS matmul 内维不匹配从 `ASSERTop` 改为可捕获的 `USER_CHECKop`，补负向结构合同；定向 3 passed。2.19 仍是聚合任务，未改为已合并。
- `7.03`：`94df46f7` 将 `complex`、`view_as_complex`、`view_as_real` 提升为 numerical 模块级稳定对象，登记 approximate fidelity；CPU identity/metadata/value 定向 2 passed。其余 tensor/nn/module family 仍待领。
- `0.15`：RingBuffer 修复后，独立 Dataset worker 监管两个 nodeid 在临时缓存下 2 passed/65.68 s；完整 smoke 仍约 390 s，任务保持待领。
| 0.21 | 测试起的子进程不带 PYTHONPATH，门禁机器上是假绿 | 已合并 | gates | 46dbe946、a5ce7310 |
| 0.22 | 压缩设备对拍时长（保留与单进程相同的 nodeid 集合） | 待领 | | |
| 1.01 | 把 `utils/data.gz` 解出的 `data.cc` 还原为可读的五个翻译单元 | 已合并 | codegen | ecb6a112（+72f020b3 用例） |
| 1.02 | `op_compiler.cc:30-69` 用正则给 `ParallelPass` 输出打补丁… | 已合并 | codegen | 3eb34e6a |
| 1.03 | 查明 `SharedReducePass` 在约 4900 个归约 kernel 里零命中的触发… | 已合并 | codegen | 3eb34e6a |
| 1.04 | `ReduceTuner::run` 不再对 CUDA 直接返回 | 已合并 | codegen | aebb1d73 |
| 1.05 | 布局收尾 | 待领 | | 5ac222bb 已合入 1/N：`python/jittor/src/test` 的 20 个 C++ 单元测试原样迁至 `src/tests`，构建递归发现、两项结构引用和 Python bridge/JIT_TEST 继续工作；活跃注释、allocator skill 与目标布局同步。结构引用 2 passed，bridge 发现与 `test_expr` 2 passed。其余 39 个根文件、三个 pass 与 `data.gz`/`vdp` 未做，保持待领 |
| 2.01 | Var 与 Op 各持自己的 flag 类型 | 已合并 | coreops | 5b197cae |
| 2.02 | 删除 `Node::custom_data` | 已合并 | coreops | 505e9b37（上半：拓扑排序自带入度，内存分析器的手工备份删掉）、77641cc8（下半：grad/dump 各持局部表，执行器与 fuser 的批下标搬到 `Node::batch_index`+`batch_stamp`，写用 `set_batch_index`、读一律 `batch_index_at(stamp)`并当场校验）。**字段本身仍在**：第六个用法是 FusedOp 的跨阶段映射，见 2.24（排在 3.11 之后）。审计描述的危害「任意两个遍历交错就互相破坏」到此消除 |
| 2.03 | `tflag` 全局计数器加魔数改为 epoch 对象或局部集合 | 已合并 | coreops | 6833f96d。嵌套 TraversalEpoch 恢复外层标记，grad/graph/memory profiler 改局部索引或集合；CPU 33 项、CUDA 3 项及结构聚焦通过 |
| 2.04 | `Var::allocator` 去类型双关 | 已合并 | | 9b3841b7 |
| 2.05 | 真正的 0 维张量 | 已合并 | coreops | 2cfc5a0d。空 shape 保留，Python/NumPy/C++ 标量来源及 reduce/arg-reduce/getitem/reshape/JIT/CUB 全归约统一 0-D；新 CPU 3 passed、GPU2 4 passed，相关 CPU 52 项与 GPU autograd 8 项通过 |
| 2.06 | 边表由 list 加反向迭代器改 SmallVector，按下标 O(1) | 已合并 | mem | ae2a1b70。输入/输出边使用内联容量 2 的 SmallVector 与反向下标，随机访问 O(1)；保序删除同步修正移位边下标，保留 fuser 依赖的消费者创建顺序。C++ 容器/边契约 2 项、CPU 生命周期 1 项、GPU1 CUDA 节点 1 项通过 |
| 2.07 | `hold_vars`/`sync_ptr` 析构里 `std::next(end())` 的 … | 已合并 | coreops | 1101f3f5 |
| 2.08 | `Node` 不再 include `pybind/py_var_tracer.h` | 已合并 | coreops | 6221d4c4。NodeLifecycleObserver 接口由 pybind tracer 注册；无 Python include 的语法编译、CPU/CUDA lifecycle/tracer 聚焦通过 |
| 2.09 | `th_mode` 从 C++ 核心上移为 autograd 策略对象 | 已合并 | coreops | b55f1acb。核心改为通用 `AutogradPolicyState`，Python autograd 层提供不可变 native/explicit-requires-grad 策略与可恢复 scope；核心 Torch 专属状态名归零。新 CPU 4 passed/1 skipped、GPU2 1 passed，相关结构 7 passed，布局与静态检查通过 |
| 2.10 | 三套 liveness 计数 | 已合并 | coreops | 8bd07e51。f/b/p 收进无额外存储的 NodeLiveness；own 防溢出，release 对无匹配 owner 的下溢立即报错，跨零返回值统一传播边界；need_free 与 graph expected-count 由封装提供，release 构建常开。C++ liveness/check_graph 契约与 CPU 生命周期 2 项通过；状态逻辑后端无关，未追加 GPU 编译 |
| 2.11 | `VarHolder` 不再是执行触发点 | 已合并 | coreops | 0f709cff。VarHolder 构造只登记持有关系；lazy/eager/auto-flush 策略迁入 Executor::submit_pending，Var 完成 Python 对象转换后才提交，显式 core.submit_pending 可无设备同步启动目标子图；删除 flush_suspended 与构造期吞错。构造/边界结构 2 项、CPU 显式提交/错误边界 2 项、GPU1 auto-flush 等价 1 项通过 |
| 2.12 | 打破 `Executor ⇄ VarHolder` include 环 | 已合并 | coreops | 318a688e。依赖 exe.allocator 的 migrate_to_cpu/data/raw_ptr/set_data 四个 inline 实现移到 var_holder.cc，var_holder.h 不再包含 executor.h 或引用全局 exe；executor.cc -> var_holder.h 保持单向，方法签名与行为不变。无 Python include 的独立头语法编译、依赖方向结构节点、CPU submit_pending 节点通过 |
| 2.13 | 执行相关全局状态 | 待领 | | |
| 2.14 | `src/misc/` 拆散 | 待领 | | |
| 2.15 | NanoString | 已合并 | bindings | 9d5ed413（索引位宽 7→8、static_assert 把表与字段绑住、`ns_check_registration` 在注册期查索引与名字长度；"dtype 表改运行期注册"那半未做，见提交说明） |
| 2.16 | 类型提升表 | 已合并 | bindings | d821c34a（int_dtype_promote 提升格；标量按 `_is_scalar` 标志认，不再按形状；float 标量把整数张量提到默认 float dtype）、a39a2f1c（补：双标量走提升格，交换左右操作数不再改变 dtype 与结果） |
| 2.17 | 算子身份用注册期整型 id | 已合并 | coreops | 1d792e16。OpInfo 注册分配 OpId，核心/tuner/pass 名字比较归零，fast_strcmp 删除，Tape 用显式 pending flag；CPU 80 项、CUDA 5 项及结构契约通过 |
| 2.18 | 算子注册表惰性初始化 | 已合并 | coreops | bca71d1f。注册表函数内惰性构造，typed polymorphic constructor 取代 `type_info + void*` 手工分派，ACL API/op_types 同步惰性；结构 3、C++ 注册 3、custom-op 2、GPU2 跨 so 1 项通过 |
| 2.19 | 错误分两档 | 待领 | | ed12fe21 已合入析构半项；c119f3bf 迁 7 处公开维度边界；83754995 迁 code/numpy/reindex 共 10 处 shape/数量边界；7c018c86 迁 transpose/fuse_transpose/reshape 共 9 处视图形状边界；b32cd6df 迁 ternary 两处 shape/dim；32758304 迁 broadcast_to 三处用户 shape 边界；37a626bc 迁 reinterpret_view 六处 dtype/shape 用户边界；8a2aebab 迁 binary 一处 shape 用户边界；8e427d2c 迁 setitem 两处 data dim/shape 用户边界；97cf5e0e 迁 getitem 三处索引/shape 用户边界；c7e5306b 迁 py_converter bool slice 一处用户输入边界；7bab54f6 迁 device_copy 一处非法设备号用户边界；83c46ffc 迁 NumPy object dtype 一处用户输入边界；a6ad6585 迁 fused_adamw 四处 TensorList cardinality 用户边界；be7ef67a 迁 var_slices 一处字符串切片长度用户边界；02795d51 迁 set_data 两处 dtype/size 用户边界；a0dd9c44 迁 reuse_np_array 两处类型/C-contiguous 用户边界；58df26d0 迁 random 一处 type 用户边界；28cba3b9 迁 py_caller 一处返回字符串用户边界；6d816bcc 迁 unary 一处 op 语义用户边界；0e2a8483 迁 CUDA curand 两处 dtype/type 用户边界（静态前置）；3ee7669e 迁 CUDNN RNN descriptor 一处 dtype 用户边界（静态前置）；3d943240 迁 CUDNN RNN x/weight dtype 一处用户边界（静态前置）；6375a852 迁 Cutt transpose axes 两处用户边界（静态前置）；0bee930e 迁 CUBLAS matmul 两处 dtype 用户边界（静态前置）；1a4f0b27 迁 CUBLAS batched matmul 两处 dtype 用户边界（静态前置）；c3c437b5 迁 CUBLAS acc matmul 两处 dtype 用户边界（静态前置）；595dac8d 迁 cuSPARSE CSR 两处 dtype 用户边界（静态前置）；c1176841 迁 cuSPARSE COO 两处 dtype 用户边界（静态前置）；f98e5b80 迁 NCCL reduce-scatter 两处 shape 用户边界（静态前置）；ec2ee53c 迁 CUB cumsum 一处 rank 用户边界（静态前置）；45f77257 迁 CUB argsort/arg_reduce 两处 offsets dtype 用户边界（静态前置）；a3890dd9 迁 CUDNN conv forward 一处 format 用户边界（静态前置）；67e710b7 迁 CUDNN conv backward-x 一处 format 用户边界（静态前置）；0a0e820e 迁 CUDNN conv backward-w 一处 format 用户边界（静态前置）；37004fe0 迁 CUDNN conv3d 输入 rank 一处用户边界（静态前置）；85ae0688 迁 CUDNN conv3d 权重 rank 一处用户边界（静态前置）；9d77a5a7 迁 CUDNN conv3d backward-x 权重 rank 一处用户边界（静态前置）；496dd510 迁 CUDNN conv3d backward-x dy rank 一处用户边界（静态前置）；e81ef514 迁 CUDNN conv3d backward-w 输入 rank 一处用户边界（静态前置）；f07cb966 迁 CUDNN conv3d backward-w dy rank 一处用户边界（静态前置）；1910f343 迁 CUB argsort x/indexes rank 一处用户边界（静态前置）；4fe6f687 迁 CUB argsort indexes 维度 shape 一处用户边界（静态前置）；166010a8 迁 CUB argsort offsets rank 一处用户边界（静态前置）；36502b8e 迁 CUB argsort offsets 长度一处用户边界（静态前置）；193d5171 迁 CUB arg_reduce offsets rank 一处用户边界（静态前置）；4c64067e 迁 CUB arg_reduce offsets 长度一处用户边界（静态前置）；fbc69232 迁 CUDNN RNN LSTM mode 一处用户边界（静态前置）；b1c604af 迁 CUDNN RNN 非 LSTM mode 一处用户边界（静态前置）；6afd44df 迁 CUDNN RNN proj_size 一处用户边界（静态前置）；57c6cd92 迁 CUDNN RNN 第二处 proj_size 一处用户边界（静态前置）；aae2f5bc 迁 CUDNN conv3d 分组通道一处用户边界（静态前置）；935bb1a9 迁 CUDNN RNN backward-x LSTM mode 一处用户边界（静态前置）；b3826005 迁 CUDNN RNN backward-x proj_size 一处用户边界（静态前置）；35664df5 迁 CUDNN RNN backward-x 非 LSTM mode 一处用户边界（静态前置）；76dc9dc3 迁 CUDNN RNN backward-x 第二处 proj_size 一处用户边界（静态前置）；408b4832 迁 CUDNN conv 输入 rank 一处用户边界（静态前置）；ceabd84c 迁 CUDNN conv 权重 rank 一处用户边界（静态前置）；44a80c8a 迁 CUDNN conv 分组通道一处用户边界（静态前置）；92a66390 迁 CUDNN conv backward-x dy rank 一处用户边界（静态前置）；1e3bab6e 迁 CUDNN conv backward-w 输入 rank 一处用户边界（静态前置）；5596563f 迁 CUDNN conv backward-w dy rank 一处用户边界（静态前置）；241ab528 迁 CUDNN RNN 输入 rank 一处用户边界（静态前置）；4858b0a2 迁 CUDNN RNN 输入通道 shape 一处用户边界（静态前置）；7a24ca0b 迁 cuFFT dtype 一处用户边界（静态前置）；040e44a0 迁 CUBLAS matmul 输入 rank 一处用户边界（静态前置），累计 114 处。C++ 具体类型、Python 跨 pyjt 可捕获、析构/信号防绕过和六十七组结构计数均有聚焦证据；fused AdamW 构造期长度负向与其结构/TU 证据已记录；var_slices 本批结构计数、`getitem_op.cc` TU 语法与字符串 slice 负向节点通过；set_data 本批结构计数、`var_holder.cc` TU 语法与两个负向节点通过；reuse_np_array 本批结构计数、`py_array_op.cc` TU 语法与两个负向节点通过；random 本批结构计数、`random_op.cc` TU 语法与无效 type 负向节点通过；py_caller 本批结构计数、`py_caller.cc` TU 语法与非字符串返回负向节点通过；unary 本批结构计数、`unary_op.cc` TU 语法与非法 op 负向节点通过；curand 本批结构计数与现有 dtype 负向静态合同通过，`nvcc -c` TU 语法通过，本机无 CUDA 未运行负向；CUDNN RNN dtype 本批结构计数、现有 bfloat16 负向静态合同与 descriptor 头 TU 语法通过，本机无 CUDA 未运行负向；CUDNN RNN x/weight dtype 本批结构计数、混合 dtype 负向静态合同与 `cudnn_rnn_op.cc` TU 语法通过，本机无 CUDA 未运行负向；Cutt transpose axes 本批结构计数、两个 axes 负向静态合同与 nvcc TU 语法通过，本机无 CUDA 未运行负向；CUBLAS matmul 本批结构计数、两个 dtype 负向静态合同与 nvcc TU 语法通过，本机无 CUDA 未运行负向；CUBLAS batched matmul 本批结构计数、两个 dtype 负向静态合同与 nvcc TU 语法通过，本机无 CUDA 未运行负向；CUBLAS acc matmul 本批结构计数、两个 dtype 负向静态合同与 nvcc TU 语法通过，本机无 CUDA 未运行负向；cuSPARSE CSR 本批结构计数、两个 dtype 负向静态合同与 nvcc TU 语法通过，本机无 CUDA 未运行负向；cuSPARSE COO 本批结构计数、两个 dtype 负向静态合同与 nvcc TU 语法通过，本机无 CUDA 未运行负向；NCCL reduce-scatter 本批结构计数与 shape 静态合同、nvcc TU 语法通过，本机无 NCCL 设备未运行负向；CUB cumsum 本批结构计数、rank-3 负向静态合同与 nvcc TU 语法通过，本机无 CUDA 未运行负向；CUB argsort/arg_reduce offsets dtype 本批结构计数、两个 int64 offsets 负向静态合同与双 nvcc TU 语法通过，本机无 CUDA 未运行负向；CUDNN conv format 本批结构计数、`Not a valid format` 静态合同与 nvcc TU 语法通过，本机无 CUDA 未运行负向；CUDNN conv backward-x format 本批结构计数、`Not a valid format` 静态合同与 nvcc TU 语法通过，本机无 CUDA 未运行负向；CUDNN conv backward-w format 本批结构计数、`Not a valid format` 静态合同与 nvcc TU 语法通过，本机无 CUDA 未运行负向；CUDNN conv3d input rank 本批结构计数与 rank 静态合同、nvcc TU 语法通过，本机无 CUDA 未运行负向；CUDNN conv3d weight rank 本批结构计数与 rank 静态合同、nvcc TU 语法通过，本机无 CUDA 未运行负向；CUDNN conv3d backward-x weight rank 本批结构计数与 rank 静态合同、nvcc TU 语法通过，本机无 CUDA 未运行负向；CUDNN conv3d backward-x dy rank 本批结构计数与 rank 静态合同、nvcc TU 语法通过，本机无 CUDA 未运行负向；CUDNN conv3d backward-w input rank 本批结构计数与 rank 静态合同、nvcc TU 语法通过，本机无 CUDA 未运行负向；CUDNN conv3d backward-w dy rank 本批结构计数与 rank 静态合同、nvcc TU 语法通过，本机无 CUDA 未运行负向；CUB argsort x/indexes rank 本批结构计数与 rank 静态合同、nvcc TU 语法通过，本机无 CUDA 未运行负向；CUB argsort indexes shape 本批结构计数与 shape 静态合同、nvcc TU 语法通过，本机无 CUDA 未运行负向；CUB argsort offsets rank 本批结构计数与 rank 静态合同、nvcc TU 语法通过，本机无 CUDA 未运行负向；CUB argsort offsets length 本批结构计数与长度静态合同、nvcc TU 语法通过，本机无 CUDA 未运行负向；CUB arg_reduce offsets rank 本批结构计数与 rank 静态合同、nvcc TU 语法通过，本机无 CUDA 未运行负向；CUB arg_reduce offsets length 本批结构计数与长度静态合同、nvcc TU 语法通过，本机无 CUDA 未运行负向；CUDNN RNN LSTM mode 本批结构计数与 mode 静态合同、nvcc TU 语法通过，本机无 CUDA 未运行负向；CUDNN RNN non-LSTM mode 本批结构计数与 mode 静态合同、nvcc TU 语法通过，本机无 CUDA 未运行负向；CUDNN RNN proj_size 本批结构计数与 proj_size 静态合同、nvcc TU 语法通过，本机无 CUDA 未运行负向；CUDNN RNN 第二处 proj_size 本批结构计数与 proj_size 静态合同、nvcc TU 语法通过，本机无 CUDA 未运行负向；CUDNN conv3d 分组通道本批结构计数与通道 shape 静态合同、nvcc TU 语法通过，本机无 CUDA 未运行负向；CUDNN RNN backward-x LSTM mode 本批结构计数与 mode 静态合同、nvcc TU 语法通过，本机无 CUDA 未运行负向；CUDNN RNN backward-x proj_size 本批结构计数与 proj_size 静态合同、nvcc TU 语法通过，本机无 CUDA 未运行负向；CUDNN RNN backward-x non-LSTM mode 本批结构计数与 mode 静态合同、nvcc TU 语法通过，本机无 CUDA 未运行负向；CUDNN RNN backward-x 第二处 proj_size 本批结构计数与 proj_size 静态合同、nvcc TU 语法通过，本机无 CUDA 未运行负向；CUDNN conv input rank 本批结构计数与 rank 静态合同、nvcc TU 语法通过，本机无 CUDA 未运行负向；CUDNN conv weight rank 本批结构计数与 rank 静态合同、nvcc TU 语法通过，本机无 CUDA 未运行负向；CUDNN conv 分组通道本批结构计数与通道 shape 静态合同、nvcc TU 语法通过，本机无 CUDA 未运行负向；CUDNN conv backward-x weight rank 本批结构计数与 rank 静态合同、nvcc TU 语法通过，本机无 CUDA 未运行负向；CUDNN conv backward-x dy rank 本批结构计数与 rank 静态合同、nvcc TU 语法通过，本机无 CUDA 未运行负向；CUDNN conv backward-w input rank 本批结构计数与 rank 静态合同、nvcc TU 语法通过，本机无 CUDA 未运行负向；CUDNN conv backward-w dy rank 本批结构计数与 rank 静态合同、nvcc TU 语法通过，本机无 CUDA 未运行负向；CUDNN RNN input rank 本批结构计数与 rank 静态合同、nvcc TU 语法通过，本机无 CUDA 未运行负向；CUDNN RNN input channel shape 本批结构计数与 shape 静态合同、nvcc TU 语法通过，本机无 CUDA 未运行负向；cuFFT dtype 本批结构计数与不支持 dtype 静态合同、nvcc TU 语法通过，本机无 CUDA 未运行负向；CUBLAS matmul input rank 本批结构计数与 rank 静态合同、nvcc TU 语法通过，本机无 CUDA 未运行负向，未迁调用点分类仍待领；a29e0f81 记录 CUDA 后端内部断言分类文档与结构门禁；b2308f41 扩展计划失败断言清单；91718e98 补充 CUDNN RNN 权重查询断言；be526722 补充 CUDNN RNN bias 查询断言；27cc72f2 补充 CUB CUDA 状态断言门禁；6ef94518 补充 CUBLAS 测试入口状态断言清单；54e8d545 补充 CUDNN 测试入口状态断言清单；8595e479 收束无新增安全用户边界说明；8af5fd8d 精确约束 CUTT 返回码内断言；30bfdb6e 精确约束 CUDNN RNN descriptor 内断言；46ab1e17 精确约束 CUDNN 计划 ASSERT(ok) 计数；9f6afe17 精确约束 CUB 测试入口 ASSERT 计数；23d70b26 追加分类文档，剩余断言不作用户错误迁移 |
| 2.20 | 信号处理器只做 `write` 与 `_exit`，符号化交给预建 helper 进程 | 已合并 | bindings | 上半 9b92f38d（去 stdio/LOGf/exit，标志改 volatile sig_atomic_t）；下半 640a4f07（符号化搬进崩溃前 fork 的 helper，经父进程 /proc/<pid>/maps 解析）；d874b01d 修 jit_key 用例（它原先靠信号处理器抛异常） |
| 2.21 | `DEFINE_FLAG_WITH_SETTER` 先赋值再调 setter，签名收新旧两值 | 已合并 | coreops | 14336afd |
| 2.22 | 环境变量统一 `JT_` 前缀 | 待领 | | |
| 2.23 | 布局收尾 | 待领 | | |
| 2.24 | `custom_data` 的最后一个用户：FusedOp 跨阶段 var 索引 | 待领 | | 依赖 3.11，需显式 `var→index` 映射并保持 relay/融合生成代码不变 |
| 3.01 | `Executor::run_sync` | 待领 | | |
| 3.02 | jit key 结构化 | 待领 | | |
| 3.03 | 三张 kernel 缓存表键改 `string` | 待领 | | |
| 3.04 | 求 jit key 改纯函数 | 待领 | | |
| 3.05 | 删除算子构造期回调执行器 | 待领 | | |
| 3.06 | 并行编译器修到可信 | 已合并 | coreops | e46ba4ec。持久线程池改 per-call future 并完整 get，异常以 exception_ptr 回主线程；任务显式携带原 JIT key，relay 成功后才发布；worker 数受 affinity/cgroup v2 限制，fork 后不再遗留幽灵线程，catch 不二次 prepare。fork 用例修前稳定退出 124；worker 上限、fork、prepare-once、失败归因 4 个 CPU 聚焦节点通过。长期 sanitizer/共享缓存压力仍按 KI-COMPILER-001 复查 |
| 3.07 | 执行器在设备等待段释放 GIL | 待领 | | |
| 3.08 | KernelIR 结构化 | 已合并 | codegen | 属性名 `62cdee84`、节点类型 enum `78410930`、pass 契约 `c3557a1c` |
| 3.09 | 死代码消除不再按「语句含 `void` 一词」删除 | 已合并 | codegen | 66e5a153 |
| 3.10 | 算子内标识符改名走结构化成员表并先做合法性校验，替代三个硬编码白名单与 `op{i}_` 盲目前… | 已合并 | codegen | 864fa52c |
| 3.11 | 生成源码里的结构体字节偏移改显式 setter，成员表用宏声明 | 已合并 | codegen | f32d3c83 |
| 3.12 | `float_atomic_fix_pass.cc:76-80`、`fake_main_pass… | 已合并 | codegen | `1ea90057`；CPU 定向 9 passed，CUDA 定向 3 passed，结构门禁 310 passed / 2 skipped |
| 3.13 | 循环维度身份用整数向量，`range10` 不再被拆成 `range1*range0` | 已合并 | codegen | bd5b5a67，用例修正 e1717fe5 |
| 3.14 | 两个同名 pass | 已合并 | codegen | 6c899325 |
| 3.15 | 一次编译只解析一遍 | 已合并 | codegen | 97cac22f；“只解析一遍”实测不值得，见下方结论行 |
| 3.16 | `token_replace_all` 不再用 CHECK 抛异常做循环终止 | 已合并 | codegen | 9aa683c4。显式终止批量替换，非法模板不再被正常路径异常吞掉；新增 2、C++ TEST 46、CPU op_compiler 5、真实 CUDA GPU3 1 项通过 |
| 3.17 | 只用于代码生成的 JIT 区段与普通 C++ 分离 | 已合并 | codegen | 2aa190af。KernelIR 逃逸 `_Pragma`，字符串/格式串保持原样，生成源码用 `#line` 指回原算子；968ae198 修复重复 `#line` 行号被登记成同名 IR scope 符号的回归，C++ TEST 11、scalar gradient 与 compiler contract 聚焦通过；未改 `jit_compiler.cc` |
| 3.18 | 删掉 `asm_tuner` 链路 | 待领 | | cb853074、acfed956 已合入前置：普通 CPU kernel 先绕过旧包装，随后 clang 输出 store 改为 `__builtin_nontemporal_store`，删除 `asm_tuner.py`、文本指令和专属测试；真实 broadcast 数值/生成代码通过，clang 汇编有 `movntss`。但同一 broadcast 冷编译只从约 1.00s 降至 0.80s，普通小 kernel 也仅从 0.89s 降至 0.69s，均约 20-22%，未达到 ≥50% 验收，仍需继续降低 CPU kernel 冷编译成本 |
| 3.19 | `event_queue` 异步基础设施修好并加测试，或删除 | 待领 | | |
| 3.20 | 执行器提供「提交部分图」显式接口，`jt.grad` 与 `Function` 回调降开销，让反… | 待领 | | |
| 3.21 | 每算子建图成本 | 待领 | | |
| 3.22 | CUDA 归约块内树形归约 | 待领 | | edf70f52 已合入 1/N：level-4 SharedReduce 改为 warp shuffle→每 warp 一个共享值→首 warp shuffle，两次 barrier；修复其对 ParallelPass 子节点位置与 `op` 字符串的旧假设，GPU3 六个生成/数值/梯度节点、CPU pass 契约通过。未改默认：四个 UNet 代表形状 30 次 profiler，新混合 17.21/16.32/16.39/25.53us（合计 75.45us），默认 warp 17.53/16.80/18.00/21.90us（74.23us）；混合总计慢 1.64%，末形状慢 16.6%，未满足无退化与完整 UNet 性能验收 |
| 3.23 | 融合逐元素 kernel 带宽效率 | 待领 | | |
| 3.24 | 布局收尾 | 待领 | | |
| 4.01 | 分配器 id 空间随分配器实例走，不再是进程静态 2M 单例 | 已合并 | device | 4e407447 |
| 4.02 | 合并多卡 | 已合并 | device | `ad9aab3a`（Var 带设备、算子在自己设备上跑、逐设备分配器与库句柄）、`c97b707a`（跨卡拷贝算子）、`93b48a8e`（torch facade）。选了什么、为什么，改写进 `device-placement.md` §5。**一处未达成**：跨卡拷贝的定序在本机不是回归网——8 张卡两两 `cudaDeviceCanAccessPeer` 全 0，驱动把跨卡拷贝经主机中转并自行与源卡串行，把 event 对整对删掉测试仍全过（实测）。测试写好了并会打印当前处于哪种情形，换到能 peer 的机器上才成为守卫。方法沉淀在 `agent/skills/multi-device-verification` |
| 4.03 | `BackendRegistry` | 待领 | | |
| 4.04 | `OpRegistry` | 待领 | | |
| 4.05 | Python 分派表 | 待领 | | |
| 4.06 | `jt.flags.backend_fallback ∈ {error, warn, allow… | 待领 | | |
| 4.07 | 后端配置改为返回 `BuildConfig` 值 | 待领 | | |
| 4.08 | 流与事件模型 | 已合并 | device | `0dfcb3dd` 每设备 copy/communication stream 与 ready/done event，接入 array H2D、fetch D2H、device_copy、NCCL collective；`78235157` 双 rank NCCL 用 rank 相关输入验证数值且 communication 双向依赖计数精确 +2。GPU 0/2：两 rank 各 1 passed，mixed-device H2D/fetch 2 passed，6.C16 下毒 1 passed，device_copy/multi-device 6 passed，既有 overlap 正确性 1 passed；未用负载敏感绝对墙钟阈值 |
| 4.09 | per-device 库句柄 | 已合并 | device | `13c28084`；4.02 已有五库 per-device 资源，本提交补齐每次执行前 SetStream。GPU 0/2 新增测试实际执行 cuBLAS/cuDNN/cuSPARSE/cuRAND/cuFFT 各两次并断言两卡逐库 bind 计数均 +2，1 passed；各库现有 wrapper 聚焦 5 passed；CPU 聚焦 1 passed |
| 4.10 | CUDA kernel 存放位置统一 | 待领 | | |
| 4.11 | ACL 改为注册表后端 | 待领 | | |
| 4.12 | 删除 `process_jittor_source` 与 `process_acl` | 待领 | | |
| 4.13 | 跨后端契约矩阵 | 待领 | | |
| 4.14 | `Module.cuda(i)`/`npu(i)`/`x.to(...)`/`x.cpu()` … | 已合并 | device | 14e5920e；修前 CPU 2 项、双卡 CUDA 4 项失败；修后新增 CPU 2 项、GPU 0/2 双卡 4 项及 4.02 聚焦回归 4 项通过；无 NPU 硬件，未做真 NPU 验证，无 ACL 时解析设备号后明确报能力错误 |
| 4.15 | 布局收尾 | 待领 | | |
| 5.01 | 114 个 `foo_` 就地方法改白名单显式声明 | 已合并 | pyops | 9d140c1c。85 个启发式生成别名收敛成显式白名单，错误别名归零，all_/any_ 非就地原语不再伪装；native 聚焦 20 passed/2 skipped |
| 5.02 | 视图与存储模型 | 待领 | | |
| 5.03 | 转置隐藏标记 | 待领 | | |
| 5.04 | 参数模型 | 已合并 | pyops | 3d40fa9e。`parameters`/`named_parameters`/`state_dict`/`named_buffers`/`_buffers` 共用一份角色遍历；绑定权重按对象身份去重而 state_dict 保留全部别名，BatchNorm buffer 按名字注册，查询不再改写 Var 名称。CPU `tests/nn` 182 passed/145 skipped，CUDA 聚焦 23 passed，Torch-shim 入口 1 passed，独立 PyTorch 2.12.1 语义对拍通过 |
| 5.05 | `eval()`/`train()` 只切 `is_train`，冻结统一由 `requires… | 已合并 | pyother | 4a8c4145 |
| 5.06 | hook 存实例级有序字典，多 hook、prepend/always_call 生效、可移除 … | 已合并 | pyother | 9117b843（含 `Var.register_hook` 返回 handle、`_dispatch_call` 接缝、weight_norm 的单 hook workaround 一并删掉） |
| 5.07 | `jt.Function` 每次调用创建一次性上下文对象，实例无状态 | 已合并 | pyother | 5c4e624b；0f639e5b（收尾：torch 兼容层的 ctx 记账跟着挪到一次性上下文上，`materialize_grads` 原本静默失效） |
| 5.08 | `flag_scope` 的备份改局部栈，`__call__` 每次新建 scope | 已合并 | | 5720e7e8 |
| 5.09 | 29 处融合 kernel 的启用条件由全局 `no_grad` 改为「输出不需要梯度」 | 已合并 | pyops | 11200c4f。native nn/misc 29 处统一按 grad mode 与递归输入 requires_grad 判定，无反向融合输出显式 stop_grad，fp16/bf16 cuDNN backward 放开；CPU/GPU契约 5、GPU norm 5、capability 7、CPU serving 9 项通过 |
| 5.10 | 索引与计数统一 int64 | 已合并 | pyops | 3e4d8a0b（`where_op.h` 的默认 dtype、randperm、topk 的空/非空两条分支、MaxPool2d/3d 与 AdaptiveMaxPool2d/3d 的 return_indices 全部 int64；顺带 `cub_where_op` 的计数与 free 大小、池化索引编码 `p*W+q`、repeat_interleave CUDA 快路径改 64 位并删掉 2^31 断言——该断言在 `misc/tensor_ops.py` 而不是审计写的 `pool/core_2d.py:198`。`jt.argsort`/`argmax`/`arange` 仍是 int32，理由见提交说明） |
| 5.11 | `amp_reg` 位常量命名导出，一律 `\ | 已合并 | pyother | 24a334cf；fc9244c4（收尾：用例的 level 切换改走 flag_scope，撞上 0.15 新加的 flag 泄漏规则） |
| 5.12 | matmul 四条路径共用能力表，dtype 用枚举不用子串 | 已合并 | pyops | 9d987034（`_cublas_can_take` 一个谓词供四处使用，判据是 `a.dtype == b.dtype and a.dtype.is_float()`；`bmm_transpose` 补上 dtype 守卫与 amp_reg。审计两处更正：「`"float" in dtype` 匹配 bfloat16/float64」属实但 cuBLAS 都支持、不是缺陷；「batched 只查 a 的 complex」属实但不可达。真正可达的是 `bmm_transpose` 完全没有守卫，整数/复数操作数在 CUDA 上撞 C++ 断言而同一乘积写成 matmul 就能算） |
| 5.13 | `unique` | 已合并 | pyops | 9c24a433（unique：四条路径合一，CPU 比较器不再把排序键截断成 int；根因不是注释说的「cub 只支持 int32」，而是存索引的输出 var 用了输入的 dtype，外加手工切分的 scratch 对不齐）43985e2c（isnan/isinf/isfinite 不再窄化成 float，float64 的 1e300 在所有后端都不是 inf）c8b4b206 + d6f08532（cumsum 一份实现、一条求导规则、一个 dim 契约，CPU 不再走 numpy 主机回调） |
| 5.14 | `Var.scatter` 改非就地 | 已合并 | | 0b75e187 |
| 5.15 | `.half()`/`.float16()` 删死的 amp 分支 | 已合并 | pyops | bf0317af。四种显式浮点 dtype 转换共用一条路径并覆盖持久/非持久 buffer，整数与 bool buffer 保持原 dtype；删除恒假且会改写整个类 `__call__` 的 amp 分支。新测试修前 5 failed/2 passed、修后 CPU/CUDA 各 7 passed |
| 5.16 | `state_dict(to="torch")` 用 `from_numpy`，不强制 floa… | 已合并 | pyops | b2238e7c。原回归 4 项中修前 2 红（int/bool 被压成 float32、大 int64 值改变）；本波回填复验 CPU 4 passed，真实 CUDA GPU 端 int64/bool dtype、大整数数值与实际计算通过 |
| 5.17 | 同一概念合并 | 已合并 | pyops | 1793f08f（平均池化：删 `pool/layers.py` 旧 AvgPool2d 并转发，2D/3D 同一套 `count_include_pad` 语义）3344cb40（`nn.Conv2d.execute` 委托 `functional.conv2d`，编译选项与输出尺寸校验合一）cd7ce682（BatchNorm/LayerNorm/GroupNorm 模块只做参数管理；`batch_norm(training=True)` 走融合 kernel；BN 的 sync 与非 sync 合并成一套数学——含审计 2026-09-03 补充的第三处：sync 分支的 `E[x²]-E[x]²` 只在 MPI 下跑，均值远大于标准差时相对误差约 7e-2） |
| 5.18 | 同一概念合并 | 已合并 | pyother | 40fa8695（efficientnet 投影层）37ac0ac5（models/_utils）4179c899（loss 的 _reduce）d5892775（分布类）d569f22d（旧式 scheduler）dd1cbe30（init 的 gain 表与 fan）96cb9b1c（linalg helper）f23dc9b8（normalize 合并到 torch 语义） |
| 5.19 | 被静默忽略的参数改为传非默认值时 warn 或 raise | 已合并 | pyops + pyother | 1710aef1（算子参数：relu/leaky_relu/silu/mish 的 inplace、instance_norm 与 InstanceNorm 的 running stats/momentum/is_train/sync、svd 的 compute_uv/driver、inv_ex 的 check_errors、ctc_loss 的 zero_infinity、sort 的 stable；topk 的 sorted 判为无需处理，见提交说明）。共用基础设施 `python/jittor/_arg_policy.py`。4cf6df28（实现 ResNet `zero_init_residual`）；211339c9（vjp/jvp strict、DataLoader pin_memory/persistent_workers、kaiming generator、fftfreq/rfftfreq dtype/device 与未知 kwargs）。统一回归 44 passed |
| 5.20 | import 期副作用删除 | 已合并 | pyother | 505a1155 |
| 5.21 | 六个 monkeypatch 安装器写成显式有序清单并加断言 | 已合并 | pyother | `3cd1a614`：新增 `_install_order.SEQUENCE` 显式声明十步安装顺序与 `record/verify` 运行时校验；`jt.sum`/`Var.sum` 共用 full-reduce 路径。`tests/core/test_install_order.py` 17 项、`tests/structure/test_install_order.py` 6 项在提交中通过 |
| 5.22 | `nn` facade 不导出 39 个下划线名，内部用模块局部名不经 `jt.nn.*` 晚绑… | 已合并 | pyops | 5d67f36b。源码 `jt.nn._*` 使用与 `dir(jt.nn)` 私有导出均为 0，后端私有覆盖迁入 `nn.backends.hooks`；结构 17 passed，CPU 25 passed/8 skipped |
| 5.23 | 根命名空间显式 `__all__` | 已合并 | pyops | d80d0b99。根星号来源归零，414 名运行时 `__all__` 与生成 pyi 顶层声明一致；namespace 13 passed，结构聚焦 4 passed |
| 5.24 | 10 个 `jt._*` 跨模块契约 | 待领 | | |
| 5.25 | `python/jittor/utils/` 拆散 | 待领 | | be2935f0、fdf3b759（部分：translator/server 已迁入 compat，jtune/nvtx 已迁入 jittor.tools，三个仓库脚本已迁入顶层 tools；utils 只剩四个由 C++/编译器硬编码引用的资源，待 3.18 落地后归 compiler 包） |
| 5.26 | 布局收尾 | 待领 | | |
| 6.C01 | `.item()` 对无符号 dtype | 已合并 | | 9b3023b1 |
| 6.C02 | `PySlice_Unpack` 返回值检查，三个变量初始化 | 已合并 | bindings | 78d08344 |
| 6.C03 | 整数提升 | 并入 2.16 | | |
| 6.C04 | 含 `void` 语句被删 | 并入 3.09 | | |
| 6.C05 | 融合边号 ≥256 回绕 | 已合并 | | 21a4f4fc |
| 6.C06 | `grad.cc:65-68` 判空对象改为 `dx` | 已合并 | | 4875a7aa |
| 6.C07 | 缺失梯度默认报错 | 已合并 | | 78c154e4 |
| 6.C08 | `grad.cc:146-261` 两趟遍历合一趟并快照结构，删无边界游标 | 已合并 | coreops | 096804a9。每个 gvar 局部快照 outgoing 与 op 输入输出后立即消费，删除 id_buffer 和无边界游标；动态新增输出回归修前进程终止，修后 CPU/GPU2 各1项及 autograd 各8项通过 |
| 6.C09 | `backward()` 可重复 | 已合并 | | 93b6e813 |
| 6.C10 | CUDA 分配钩子两张 map 用 `find` 加显式错误，释放后 `erase` | 已合并 | mem | `59c7a9b3` |
| 6.C11 | CPU 分配失败抛异常，返回值必须检查 | 已合并 | mem | `a683274e` |
| 6.C12 | `cuda_device_allocator.cc:32-37` 的 managed 回退放到 … | 已合并 | mem | `e48c52c2` |
| 6.C13 | 零字节分配不返回伪指针 `0x10` | 已合并 | mem | `b8b978e1` |
| 6.C14 | SFRL | 已合并 | mem | `a0da8374` 完成 SFRL 映射表清零、free/share_with 校验与五个分配器写回 allocation；`b0d90d44` 将 getitem/setitem 的别名判断改为 `7e223483` 引入的显式 share 环关系，并覆盖共享与非共享对象语义 |
| 6.C15 | `migrate_to_cpu/gpu` 迁移前检查 share_with 关系，整组迁移或拒绝 | 已合并 | mem | `7e223483`。Var 加共享环（`share_prev/share_next`），migrate 看到环就整组搬走并保持相对偏移。顺带两条：`ArrayOp::run()` 绕过 `free_var_mem` 换内存要自己摘环；裸分配器表达不了共享，新增 `Allocator::can_share()`，为假时退回旧行为**并告警**（不再静默断开） |
| 6.C16 | fetch 跨流 | 已合并 | mem | `9095484b`。**修法与任务行不同**：不是「记 event 让默认流等」——副流本来就在等默认流，再让默认流等副流等于取消掉异步重叠（`test_memcopy_overlap` 那条性能断言测的正是它）。改成在源块上多持一份引用直到主机回调之后；event 栅栏降级为 `can_share()` 为假时的兜底 |
| 6.C17 | `TempAllocator` 删遮蔽基类的 `used_memory`/`unused_mem… | 已合并 | mem | `4357c8bb` |
| 6.C18 | CachingBlock 保存底层 allocation 并原样回传，不再传 0 | 已合并 | mem | `74264dd3` |
| 6.C19 | 每个分配器一把锁并覆盖 `gc()` | 已合并 | mem | `6a73832b` |
| 6.C20 | swap | 已合并 | mem | `4b33609d`（文件名用运行期 pid 加随机 token、cudaMemcpy 查错、去静态 8 MB buffer）+ `2940d88d`（后半）。后半的核实结论：`save_mem` **已经**是编译期常量，「未完成特性挂在最热释放路径上」就现状而言不成立；真正坏的是 `export JT_SAVE_MEM=1` 从来没被翻译成 `-DJT_SAVE_MEM=1`，文档教的开法是空操作。已接上，开着时才进构建指纹 |
| 6.C21 | 检查 `NODE_MEMCHECK` 外 `check_graph` 静默空转 | 已合并 | mem | `2bce371e`。 绑定分区已让出（曾误领）。前置核实结论：`do_graph_check()` 前半段（从 hold_vars 反向遍历、重算 f/b/p）在任何构建下都真跑；只有后半段查悬挂节点的那个循环读 `lived_nodes`，而它只在 `-DNODE_MEMCHECK`（`compiler.py:1164`）下填充——所以 `check_graph=1` 在 release 下交付的是它宣称的一半。另：`Node::memcheck_all_exist()` 在出厂 object 里也是空的，它本该断言什么无法从 object 恢复（见 812714d5 还原者说明），不要当成还原时丢的。**做的是完整版而非最小版**（协调者确认）：登记表改成跟着 `check_graph` 走而不是跟着构建类型走，于是 release 下开 `check_graph=1` 两半都真查；`do_graph_check()` 返回悬垂扫描的节点数，扫到 0 时打一条每进程一次的警告说明为什么——不对称本身也可见了。关着时的开销量过：40 万个 Node 的构造从约 955 ns/个变成约 980 ns（+2.6%），在同进程 ±15% 的波动里 |
| 6.C22 | pyjt 关键字参数 | 已合并 | bindings | ed148a56 |
| 6.C23 | `is_type<NanoString>` 收窄 | 已合并 | bindings | f8f9de43 |
| 6.C24 | 带实例 `__dict__` 的类型加 `Py_TPFLAGS_HAVE_GC` 与 trave… | 已合并 | bindings | 4a30c5e4 |
| 6.C25 | 生成绑定补 `catch (...)` | 已合并 | bindings | b58ba756 |
| 6.C26 | `pyjt_compiler.py` 的 C++ 解析 | 已合并 | bindings | 4105d091 |
| 6.C27 | `Var.data` 返回的 numpy 视图 base 指向包裹该次 allocation 的… | 已合并 | bindings | 9504e520 |
| 6.C28 | 生成带「已构造」标志的 `tp_new` 或 `tp_dealloc` 先检查 | 已合并 | bindings | 8bd40d02 |
| 6.C29 | 标量转数组的全局 `tmp_data` 改自带 buffer | 已合并 | bindings | b57c31a1 |
| 6.C30 | `helper_cuda.h` 的 `peek` 去掉进程级闩 `peek_logged` | 已合并 | coreops | bcdf1593 |
| 6.C31 | 失败的 import jittor 在退出期 abort，父进程无声消失 | 已合并 | bindings | 64350894 |
| 6.P01 | 转置标记陈旧 | 并入 5.03 | | |
| 6.P02 | Function 实例复用、no_grad 泄漏、tied weight 参数集合 | 并入 5.07、5.08、5.04 | | |
| 6.P03 | H1 分组 conv3d 的 ww reindex 形状顺序 | 已合并 | pyops | a50c5678 |
| 6.P04 | H2 Pool3d `return_indices` 内核第三层循环变量 | 已合并 | pyops | 2fb2d15d |
| 6.P05 | H3 Pool3d CUDA 反向用 `pout_shape` 作上界 | 已合并 | pyops | 359031f4 |
| 6.P06 | H4 MaxUnpool2d/3d 在 `stride != kernel_size` 时用原始… | 已合并 | pyops | 70d97137（机制与审计描述不同：`xshape3` 本就是重建体宽度，真正错的是默认 `output_size`） |
| 6.P07 | H5 eigh 反向 `dout` 全零时写零 | 已合并 | pyops | aeeca502 |
| 6.P08 | H6 `_autograd_grad` 的 zip 用过滤后的 `new_grad_output… | 已合并 | pyops | e9c704cb |
| 6.P09 | H7 irfft 对实数输入与显式 `n` 的处理走 `:68-73` 的判别函数 | 已合并 | pyops | b59563c1（实数输入在默认 `n` 下原本就与 numpy 一致；错的是显式 `n`） |
| 6.P10 | H8 ReduceLROnPlateau 每轮从初始 lr 计算 | 已合并 | pyother | 634a8e8a |
| 6.P11 | H9 `unique(return_counts=True, return_inverse=Fa… | 已合并 | pyother | 7c854f1d |
| 6.P12 | H10 Adan 的 `clip_grad_norm` 移出 param_group 循环 | 已合并 | pyother | 888947fd |
| 6.P13 | H11 `zero_grad` 清缓冲而非只翻标志 | 已合并 | pyother | b116b545 |
| 6.P14 | H12 Adam 偏差修正用每 param 的步数 | 已合并 | pyother | 0d67526a |
| 6.P15 | H13 worker 异常不再变成给父进程发 SIGINT | 已合并 | pyother | 042bc2c7 |
| 6.P16 | H14 `mp_log_v` 做 int 转换 | 已合并 | pyother | f7162b68 |
| 6.P17 | H15 Pillow 版本用元组比较 | 已合并 | pyother | 03cf502d |
| 6.P18 | H16 `Dataset.__deepcopy__` memo 存对象不存 id | 已合并 | pyother | 9763203a |
| 6.P19 | H17 `LogitRelaxedBernoulli` 返回 logit | 已合并 | pyother | 2a76e252 |
| 6.P20 | H18 `ComplexNumber.__rsub__` 虚部符号、`__imatmul__` … | 已合并 | pyother | 8b36f3c4 |
| 6.P21 | H19 稀疏卷积重复坐标 CPU/CUDA 语义统一 | 已合并 | pyother | b3ebd1b5 |
| 6.P22 | H20 `to_dense` 对 COO 重复索引求和 | 已合并 | pyother | 9d1bf2a1 |
| 6.P23 | eigh 的特征向量梯度在 CUDA 上错约 60% | 已合并 | | d361100e |
| 6.P24 | Pool3d 的 count_include_pad 读原始参数 | 已合并 | | d221dbde |
| 6.P25 | Adan 偏差修正仍用全局 n_step；连带第一步 grad_diff 语义 | 已合并 | pyother | 2d5804a4 |
| 6.P26 | MaxPool3d 的 ceil_mode 输出尺寸比 torch 多一个平面 | 已合并 | pyops | f982a6b8。修前输出 `(4,4,4)` 对 Torch `(4,4,3)`；修后 CPU 18 passed/15 skipped，真实 CUDA GPU4 尺寸、索引往返和前后向 4 passed |
| 6.B01 | MPI 的 int64 改 `MPI_INT64_T` | 已合并 | dist | 03518707 |
| 6.B02 | ACL | 待领 | | 03daccfb 完成 tensor/workspace 前置：CreateAclTensor 返回真实状态；workspace 由 `exe.temp_allocator` 成对保存 owner/handle，失败前清空并抛。5388864c 完成第二代码切片：65 处 executeOp 共用的 `checkRet` 硬失败并带算子/CANN 状态，group/非 group 均检查注册表，fused 调度用 `current_op` 检查当前输入而不再被外层形参遮蔽。两组静态合同均修前 3 failed、修后各 3 passed；910B3 文档覆盖正常无 CPU fallback、三类故障归因和 workspace 进程退出释放。本机无 CANN/NPU；代码阶段完成，仍需 910B3 实机编译并验证正常路径、失败传播和释放后才能完成 |
| 6.B03 | HCCL 宏错误时抛而非 return | 已合并 | dist | c657ab01 |
| 6.B04 | 分布式一旦被请求，初始化失败硬失败 | 已合并 | dist | 8ae65e24 |
| 6.B05 | cuBLAS `use_tensorcore` 三目判断写反 | 已合并 | cudabk | 9f5c3e90 |
| 6.B06 | `var_broadcast` 用传入的 root | 已合并 | dist | 89dd014b |
| 6.B07 | cuDNN RNN（dropout 掩码/work_space/infer_shape 泄漏 + 按实际 dtype） | 已合并 | cudabk | f5540427、da5bcad4 |
| 6.B08 | cuSPARSE | 已合并 | cudabk | 44b8a8a6 |
| 6.B09 | curand 奇数长度用临时 buffer 不越界写 | 已合并 | cudabk | 08a1bd66 |
| 6.B10 | MPI fp16 归约统一标量参考实现加可选 SIMD 与运行期 CPUID 检测 | 已合并 | dist | 734d55a1 |
| 6.B11 | ACL 六个算子静默把输入升到 fp32 | 已合并 | dist | 492e5385 |
| 6.B12 | `cutt_transpose_op.cc:77` 的 `cudaGetLastError()`… | 已合并 | cudabk | 58215816 |
| 6.B13 | cuFFT `cufftCreate` 后被 `cufftPlanMany` 覆盖的句柄泄漏 | 已合并 | cudabk | 11697758 |
| 6.B14 | conv3d 三算子迁到 backend plan 缓存 | 已合并 | cudabk | 8432a181 |
| 6.B15 | MPI 同时识别 PMI_/SLURM_ 环境变量或要求显式声明 | 已合并 | dist | 956c4b23 |
| 6.B16 | `sync_run` 在 ACL 上实现或删 flag | 待领 | | 15bccb92 已合入 1/N 代码组织阶段：`BaseOpRunner::syncRun` 在 `sync_run=1` 时同步 `aclstream`，检查返回码并以算子名、数值码和 ACL 文本报错；静态合同修前 1 failed、修后 1 passed。Ascend 910B3 文档已给出 CANN/`npu-smi` 前置、`sync_run=1/0` 精确节点与禁止 CPU fallback 检查。本机无 NPU，仍需 910B3 实机验证同步/异步两条路径和失败归因后才能完成 |
| 6.B17 | 析构不得抛 | 已合并 | cudabk | 272f00ba |
| 7.01 | 「看起来支持其实空操作」一律改为实现或抛 `NotImplementedError`，需显式 `… | 已合并 | 兼容层分区 | ff395ecc b7c12ddc 0446217e 47012a27 46bc9ea7 49d41acf 9053a7c0 |
| 7.02 | DDP 真实梯度同步 | 已合并 | 兼容层分区 | 4f08f1da |
| 7.03 | 每个 torch API 一个模块级一等对象加保真度标注 | 待领 | | 3f009970 已合入 1/N：新增 exact/approximate/unimplemented 保真度注册与确定性报告；factories 最终拥有的 20 个 API 成为稳定模块级 callable。8589860b 把 compiler family 的 `compile/trace/script` 提升为稳定对象；6d65fdca 把 numerical owner 的 `eye` 提升并登记 approximate fidelity，CPU 3 项通过。52ddeabc 把 `empty_like` 收回 factory owner，登记保守 approximate fidelity，compiler 阶段只绑定稳定对象并删除临时闭包；5dc59d85 将 `vstack`/`row_stack`/`hstack`/`dstack`/`column_stack` 提升为 numerical 模块级稳定函数，登记保守 approximate fidelity，并以 1-D、2-D/混合 CPU NumPy shape/value对拍、身份与 metadata 共 7 项通过。630a3e44 将 `movedim`/`moveaxis` 提升为 numerical 模块级稳定函数，让 Var 方法共用内部实现避免递归，登记保守 approximate fidelity；身份、metadata、CPU 正负单轴/多轴及 Var 方法共 5 项通过。c38e1453 将 `unflatten`/`swapaxes`/`swapdims`/`ravel` 提升为 numerical 模块级稳定函数，让 Var 方法共用内部实现，登记保守 approximate fidelity；身份、metadata、CPU/Var shape-value 定点 3 项通过。d2c3b8f4 将 `copysign`/`xlogy`/`heaviside`/`signbit` 提升为 numerical 模块级稳定函数，让 Var 方法共用内部实现，登记保守 approximate fidelity；身份、metadata、CPU NumPy 对拍及 Var 方法定点 4 项通过。217e107f 将 `trace`/`diag_embed`/`diagflat` 提升为 numerical 模块级稳定函数，让 Var 方法共用内部实现，登记保守 approximate fidelity；身份、metadata、CPU/Var NumPy shape/value 定点 4 项通过。c0c6d283 将 `float_power` 提升为 numerical 模块级稳定函数，让 Var/root 共用内部实现，登记保守 approximate fidelity；模块身份、metadata、CPU NumPy value/float64 dtype 对拍 3 项通过。0ff1b6d1 将 `isclose`/`allclose` 提升为 numerical 模块级稳定函数，保留 `rtol`/`atol`/`equal_nan` 并让 `allclose` 返回 Python bool，登记保守 approximate fidelity；身份、metadata、CPU NumPy 与 bool 返回定点 3 项通过。af6280d7 将 `cdist`/`bucketize` 提升为 numerical 模块级稳定函数，登记保守 approximate fidelity；身份、metadata、CPU `cdist p=1/2` 与 `bucketize right=False/True`/dtype 定点 4 项通过。ac4877d6 将 `nansum`/`nanmean` 提升为 numerical 模块级稳定函数，让 Var/root 共用内部实现，登记保守 approximate fidelity；身份、metadata、CPU 全量/dim/keepdim/NaN-count 与 Var 方法定点 4 项通过。203d19c1 将 `aminmax` 提升为 numerical 模块级稳定函数，让 Var/root 共用内部实现，登记保守 approximate fidelity；身份、metadata、CPU 全量/dim/keepdim 与 Var 方法定点 3 项通过。9ada5bef 将 `pdist` 提升为 numerical 模块级稳定函数，让 Var/root 共用内部实现，登记保守 approximate fidelity；身份、metadata、CPU p=1/2 shape/value 与 Var 方法定点 3 项通过。2490ec9a 将 `logcumsumexp` 提升为 numerical 模块级稳定函数，让 Var/root 共用内部实现，登记保守 approximate fidelity；身份、metadata、CPU 1-D/2-D dim 与 Var 方法定点 3 项通过。1512f92c 将 `quantile` 提升为 numerical 模块级稳定函数，登记保守 approximate fidelity 并明确 NumPy CPU fallback/dtype/out/device 限制；模块身份、metadata、CPU q=0/.5/1 与 dim/keepdim 对拍 3 项通过。80b41079 将 `nanquantile` 提升为 numerical 模块级稳定函数，登记保守 approximate fidelity 并明确 NumPy CPU fallback/float32 与 device/layout/interpolation/out 限制；模块身份、metadata、CPU NaN q/dim/keepdim 对拍 3 项通过。7f2cbb83 将 `std_mean`/`var_mean` 提升为 numerical 模块级稳定函数，登记保守 approximate fidelity 并明确 correction/keepdim 限制；身份、metadata、CPU 基础值/tuple shape 定点 3 项通过。b1568308 将 `mv` 提升为 numerical 模块级稳定函数，保留 out identity 与 shape/size 错误语义并登记保守 approximate fidelity；身份、metadata、CPU value/out/Var 委托/非法输入定点 4 项通过。0d4828fb 将 `addmm` 提升为 numerical 模块级稳定函数，保留 alpha/beta 并登记保守 approximate fidelity；身份、metadata、CPU 默认/缩放及 Var 委托对拍 3 项通过。1cca275c 将 `mm` 提升为 numerical 模块级稳定函数，登记保守 approximate fidelity，保持现有 2-D matmul 与 out/device/layout/dtype 限制；身份、metadata、CPU NumPy shape/value 及 Var 委托定点 3 项通过。d8dbbb44 将 `trapz`/`trapezoid` 提升为 numerical 模块级稳定包装，登记保守 approximate fidelity，保留 x/dx/dim/out 语义；身份、metadata、CPU 1-D/2-D NumPy 对拍、Var 委托及 out identity 定点 4 项通过。a291194d 将 `masked_select` 提升为 numerical 模块级稳定函数，登记保守 approximate fidelity，保持 1-D 选择与 out/device/layout/dtype 限制；身份、metadata、CPU 2-D bool-mask NumPy 值及 Var 委托定点 3 项通过。8139d685 将 `narrow` 提升为 numerical 模块级稳定函数，登记保守 approximate fidelity，保留正/负轴与 Var 委托；身份、metadata、CPU 正/负 `dim/start` NumPy slice 对拍定点 3 项通过。ee5adc5b 将 `tile` 提升为 numerical 模块级稳定函数，登记保守 approximate fidelity，保留 tuple/list dims 与 Var 委托；身份、metadata、CPU NumPy.tile shape/value 定点 3 项通过。a2e86707 将 `diff` 提升为 numerical 模块级稳定函数，登记保守 approximate fidelity，保留 n/dim/prepend/append 语义；身份、metadata、CPU 1-D/2-D NumPy 对拍与 Var 委托定点 3 项通过。9ac4ae4f 将 `square` 提升为 numerical 模块级稳定函数，登记保守 approximate fidelity，保持元素级 x*x 与 Var 委托；身份、metadata、CPU NumPy 值定点 3 项通过。8647dc4d 将 `pairwise_distance` 提升为 numerical 模块级稳定函数，登记保守 approximate fidelity，复用原生 nn p-norm/keepdim 语义；身份、metadata、CPU p=2 值及 keepdim shape 定点 3 项通过。tensor 79、nn 126、module/cuda/data 尚未迁移，保持待领 |
| 7.04 | 激活显式、一次性、可查询 | 已合并 | compat | `f704b9d4`：删除 argv 源码嗅探与 `jt.flags` 代理，部署/运行时入口统一为幂等 `activate()`（`enable` 仅同对象别名），公开不可变 `activation_status()`；HOME/TMPDIR/NCCL/严格数学环境只在显式 preflight 中准备，并接入 `EXPLICIT_REQUIRES_GRAD` 策略。聚焦 bootstrap 41 passed，策略接线 1 passed，部署静态测试 15 passed；结构实算单例仍命中既有 JIT IR 重复行号失败，非本项激活回归。 |
| 7.05 | install 事务化 | 待领 | | |
| 7.06 | 依赖单向化 core→tensor→nn/optim→distributed→fsdp→适配器 | 已合并 | 兼容层分区 | 27c4bdeb |
| 7.07 | 第三方库补丁搬出 compat/ | 待领 | | |
| 7.08 | `torch.dtype` 改真正的对象而非 str 子类 | 待领 | | 9aaedba9 已合入 `torch.backends` 映射；dtype 真对象、完整 C++ dtype 边界迁移和占位 dtype 的计算/分配拦截仍需整体完成 |

**7.08 只做了三分之一，另两项仍待领**（兼容层分区，2026-09-03）：

- 已做：**`torch.backends.*` 映射表格化并单测** — `9aaedba9`。六种拼写合成两条状态。`fp32_precision` 原本是四个 backend 对象上的字面量 `"ieee"`（`_PrecisionBackend` 的**类属性**），读不反映 tf32 已打开、写它什么都不做；`get_float32_matmul_precision()` 读的则是一个`matmul.allow_tf32` 从不更新的独立字符串。四条缺陷都用探针在旧实现上逐条实测复现过，不是推断。
- 未做：**`torch.dtype` 改真正的对象** — **未动，且不建议顺手做**。`types.py` 的 `class dtype(str)` 里 str 继承是**承重**的，文件自己写明了理由：jittor 的 C++ 类型分发构造器要求 str/NanoString；而且 jittor **自己的 Python 代码**（`contrib.concat`、`linalg`、`nn`）会`str(var.dtype)` 再把结果直接喂回 C++ 分发。所以「入口处一次转换」要求先把**每一个** dtype 跨进 C++ 的边界找全再改；改一半会让错误的 dtype 静默流进算子。这是本任务里唯一一条「做一半比不做更糟」的，应整块领、单独排期。
- 未做：**占位 dtype 参与计算时抛 `NotImplementedError`**。占位清单在 `types.py:_make_dtypes` 的 specs 里已有注释标出。难点不在识别而在拦截点：这些 dtype 对象**必须**继续存在且可作字典键（transformers/safetensors/torchao 在 import 期就按它们建表），所以只能在「真的参与计算或分配」那一步抛，不能在被引用时抛。

| 7.09 | `torch.library` | 已合并 | compat | 99901e6c、d0a782a0。按张量真实驻留选择 CPU/CUDA 并排除 Meta，`register_autograd` 真正接入且模型特判移出通用注册层；线程局部 autocast dtype policy 进一步选择 AutocastCPU/CUDA，嵌套禁用与退出恢复普通路由。独立 PyTorch oracle 一致，CPU dispatch 8 passed、1 个未分配 CUDA 节点 skipped |
| 7.10 | `torch.compile`/`jit.trace`/`jit.script` 保留 pass… | 已合并 | 兼容层分区 | 3d898ece。语义参数拒绝、permissive allowlist/audit 与 ShapeProp ImportError 验收均有测试 |
| 7.11 | autograd 语义 | 待领 | | 7cc3fa71 已合入 create/retain、隐式输出、sum warn、saved version 等大部分语义；`is_leaf` 仍恒 True、`grad_fn` 仍恒 None，等待内核提供反向可达叶子查询 |
| 7.12 | 独立 torch 包 | 待领 | | |
| 7.13 | FSDP2 | 待领 | | 已合入 37c0aed4、c0e6e1ae、48da7360、873dd5cf；仍缺峰值显存达标、复用原生 optimizer 更新逻辑与 DeviceMesh 真实分组 |

**7.13 已合入四部分，其余待领**（兼容层分区，2026-09-03）：

- 已做：`37c0aed4` 按实现身份识别优化器；`c0e6e1ae` 修复 Torch 模式下 NCCL preflight；`48da7360` 拒绝未遵守的 mesh 并跨 rank 归约梯度范数；`873dd5cf` 在分片后释放聚合的 `full_param`。
- 未做：释放后峰值仍未低于未分片，每步仍增长 16 个分片大小的 Var；需继续沿已证实的引用环线索定位。
- 未做：`optimizer.py` 仍自行实现 SGD/Adam/AdamW 数学，没有做到“复用 Jittor optimizer、只替换梯度来源”。
- 未做：DeviceMesh 真实分组与多维切片依赖 8.08，当前仍明确拒绝。

| 7.14 | vLLM 边界检查把 `torch` 视作 jittor 别名 | 已合并 | 兼容层分区 | 178be65a |
| 7.15 | `_rebuild_tensor_v2` 按 stride 还原或报错 | 已合并 | | 7e7877c8 |
| 7.16 | compat/ 内 129 个 `except: pass` 与 258 个宽泛 except … | 已合并 | 兼容层分区 | 72dbc22d（+ 一次修复：`93b48a8e` [4.02 3/3] rebase 时把 cuda.py/types.py 整段解回 7.16 之前，`swallowed()` 24→0、16→0，全树违规回到 47 条；已用三方合并恢复，见提交说明） |
| 7.17 | `runtime.enable()` 只把 shim 的 site 目录加进 sys.path … | 已合并 | 兼容层分区 | d5c769fb |
| 7.18 | 布局收尾 | 待领 | | |
| 7.19 | 精度策略接线：Jittor 一档、torch 两档，底层 matmul/conv 分字段 | 待领 | | 依赖 8.03、7.08；需保持 shim 的卷积与 matmul 语义分离 |
| 7.20 | fp32 RNN 默认精度与 torch `cudnn.allow_tf32` 映射 | 待领 | | 依赖 8.03、7.19；需 CPU 递推与真实 CUDA 对拍 |
| 8.01 | 描述符与 workspace 一律 RAII | 已合并 | cudabk | afb08e88 |
| 8.02 | 集合通信走通信流加事件依赖，支持 `GroupStart/End` 桶化 | 待领 | | |
| 8.03 | 精度策略收敛 | 已合并 | cudabk | dab0690c |
| 8.04 | cuDNN 9 | 已合并 | cudabk | 7580b6e7（RNN v8 API）+ 9f2e7b80（版本闸门与 wheel 栈） |
| 8.05 | MKL | 待领 | | |
| 8.06 | ACL 去样板 | 待领 | device | 5be5fa15 建立 BaseOpRunner 统一 workspace/execute/error/可选同步尾部并迁 unary；86b31e14 迁 binary；b7c763bd 迁 ternary SWhere；b1d7bd5 迁四个单步 reduce owner；251e3e96 迁 Cumsum；c59e3948 迁 MatMul；90050ccb 迁 Expand；51103861 迁 Floor；9a5a8ac4 迁 NanToNum；5e831df1 迁 Triu；88d4d35f 迁 Sigmoid forward；b76f1b16 迁 Transpose/Permute；a180c691 迁 Softmax forward；072e05b9 迁 Embedding forward；15668ea3 迁 Roll；11481922 迁 Gather forward；80c5e565 迁 ClampTensor；6e1d462c 迁 Stack；9c348801 迁 Flip；9db57798 迁 Scatter；530bbc8f 迁 Concat；1c4b32d9 迁 SplitWithSize；69e5974a 迁 Nonzero；11fe7012 迁 Range；faac2700 迁 Dropout forward；910e2c49 迁 LeakyReLU forward；830907ff 迁 ArgReduce max/min；a293b615 统一 Random uniform/normal launcher；055cb64b 迁 UpsampleNearest2d forward；553b5ec1 迁 SiLU forward；600ee169 迁 BatchMatMul；8251d29d 迁 RotaryPositionEmbedding forward；71bab738 迁 Maxpool forward；16a89606 迁 Avgpool forward，保留 descriptors/poolCeil/divisor 与同步策略且 backward 不动；静态合同 35 passed，补 Ascend 910B3 上机说明。本机无 CANN/NPU，仍待实机验证，未铺其余 family、胖 AclOpFunctions、op_idx_map、属性 data 通道与描述符缓存 |
| 8.07 | conv 族共享描述符与计划层 | 已合并 | cudabk | 947f5223（反向只留 C++ 一份）+ 47f91130（计划请求一个构造函数） |
| 8.08 | `ProcessGroup` 对象替代全局唯一 communicator | 已合并 | dist | 82410549（NCCL env/file 与 MPI bootstrap 双卡通过；HCCL 对称实现未在 Ascend 真机验证） |
| 8.09 | NCCL | 已合并 | dist | f2d9c291, 95a1c956 |
| 8.10 | `distributed/launch.py:102-107` 改 `wait(timeout)… | 已合并 | dist | 925850b3 |
| 8.11 | 图同构优先：reduce 保留全尺寸输出，broadcast 去 rank 相关别名，flat 策略可配置 | 已合并 | dist | a1e769d5 |
| 8.12 | 算子内不再复用全局 jit key 缓冲做缓存键 | 待领 | | |
| 8.13 | cuTT 计划未命中时的 `cudaDeviceSynchronize` 删除或降流同步 | 已合并 | cudabk | c0d2cc5c |
| 8.14 | Corex | 待领 | | 32cb3f8f 已合入独立前置：`discover()` 通过 `COREX_HOME` 解析 `bin/clang++`，返回只读结构化结果；新增离线 fake compiler 探测 2 passed 与 Corex 上机指南。正式依赖 4.12 未满足，不删 `process_acl`；本机无 Corex/Iluvatar 硬件，仍需真实设备验证 |
| 8.15 | 多机 rendezvous | 待领 | dist（2/N） | 6d3b2ae3：TCPStore/FileStore 与 `env://`/`tcp://`；120174a6：NCCL WORLD unique id 经 Store 交换、错误 endpoint 有界超时。仍缺 HCCL、NCCL 子组 unique id、真实两机 collective 与跨机失败验收 |
| 8.16 | 多机启动器 | 待领 | | |
| 8.17 | 跨机网络与诊断 | 待领 | | |
| 8.18 | 多机 checkpoint | 待领 | | |
| 8.19 | 布局收尾 | 待领 | | |
| 9.01 | `import jittor` 不编译不下载 | 待领 | | 361d59b2、c4b21762 已合入 2/N：native import 不再探测 Torch 或无条件调用 NCCL/cuTT/MKL setup；显式分布式请求仍 fail-closed，CPU float32 batched matmul 首次按需 MKL，只读 HOME 配合可写 JITTOR_HOME 可离线导入。聚焦 2+1 passed；热缓存 import 实测 1.332 s，冷配置仍编译 174 个核心 TU、约 40.015 s，未达 <1 s 且核心编译未惰性化，保持待领 |
| 9.02 | `install_cuda.py:113-122` 的 `os.execl` 自重启删除，用 d… | 已合并 | 构建 | 6b45c078 |
| 9.03 | 构建期失败一律抛带上下文的 `RuntimeError`，不用 LOGf/裸 assert | 已合并 | 构建 | 9197c8c6 |
| 9.04 | 依赖跟踪改用编译器的 `-MD -MF` | 已合并 | 构建 | 65a2dc12（clean_cache 从一份布局定义生成）、2569fe3b（依赖跟踪、SHA-256、主机名/`-march=native`/git 分支/路径哈希位数，一个提交只让大家重编一次）。依赖跟踪走的是「扫描器认识 `#ifdef`」而不是 `-MD -MF`：`process()` 兼着 JT_XXX 宏发现（必须在编译前）与依赖跟踪（只能在编译后），拆开才可能用 depfile，已登记为 9.21 |
| 9.05 | 下载安全 | 已合并 | 构建 | e111ebcc |
| 9.06 | 删 cutlass 下载 | 已合并 | 构建 | 50673d69 |
| 9.07 | import 过程不反向写环境变量 | 待领 | | |
| 9.08 | 新架 GPU | 已合并 | 构建 | 2d71f792 |
| 9.09 | `cuda_wheel` 失败时 LOG.w 出原因，strict 为默认 | 已合并 | 构建 | c63dd809 |
| 9.10 | 2.0 版本策略 | 已合并 | | 77dcc747 |
| 9.11 | release 的 platform-validation 阶段跑 selftest | 已合并 | 构建 | 2af4658e |
| 9.12 | `extern/rocm/rocm_cache.tar.gz` 的预编译 .o 改从源码构建，或… | 已合并 | 构建 | 46cc77d5（源码不在本仓库，做不到从源码构建；按任务允许的第二条补了来源说明与字节钉定，并写清要怎样才算可接受） |
| 9.13 | README 加「首次运行会发生什么」 | 已合并 | 构建 | dad3cd26 |
| 9.14 | 一次性的构建前置条件检查 | 已合并 | 构建 | b2bd11fd（审计那 17 个失败点：4 个可操作 → 15 个「是」、2 个「部分」、0 个「否」） |
| 9.15 | noxfile | 已合并 | 构建 | 84c7f766 |
| 9.16 | `agent/scripts/check_repo_layout.sh` 收缩为少数真会复发的检… | 已合并 | build | 94944d28。266 行降到 68 行，删除历史路径/根目录清单与全树 grep，保留运行时资源、生成物污染、实验目录、模块包冲突和文档治理；0.18 s，相关结构测试 20 passed |
| 9.17 | 死代码 | 已合并 | 构建 | f99250bb |
| 9.18 | `disable_lock=1` 启用时明确告警并纳入缓存指纹 | 已合并 | build | 801dd80d。启用时打印并发损坏警告并进入独立构建配置指纹；默认锁定配置保持原缓存名。相关两文件 28 passed |
| 9.19 | 布局收尾 | 待领 | | |
| 9.20 | asm_tuner 非原子写 .s，并发编译读到截断汇编 | 已合并 | build | 1919b035。`pass_asm()` 写进带 pid 的临时文件后 `os.replace`；inode 回归 1 passed，原四 worker Dataset 复现用例 1 passed |
| 9.21 | 拆掉手写预处理器最后一块：process() 双职责分离 + depfile | 已合并 | build | 9a5f4e7c 拆出 `JT_*` 宏声明；237d6460 删除手写 include scanner，GCC/Clang 用原子发布的 `-MD/-MF` depfile，asm/dlink 仅首段编译保留依赖参数，MSVC 走独立 `/showIncludes` 构造与解析（单元契约覆盖，未做 Windows 实机）。宏展开/失活条件修前 1 failed，修后定向 5 passed、C++ TEST 通过、CPU 9 passed/1 skipped、实机 CUDA `-dc` dlink 通过 |
| 9.22 | 并发编译同一个算子读到写了一半的 `.so` | 已合并 | build | c4bbdd72。`cache_compile` 对 asm/dlink wrapper 的最终产物也用私有临时名加 rename，`.key` 同样原子替换；修前两个 wrapper inode 契约均失败，修后 4 passed，冷 CPU 聚焦 6 passed，实机 CUDA 普通 JIT 与 `-dc` dlink 均通过 |
| 9.23 | `run_child_script(timeout=N)` 不收孙进程 | 已合并 | bindings | 17e43c9a（进程组 + `os.killpg` + 有界 drain）。**更正**：任务描述里"`communicate()` 继续等"在 CPython 3.11 上不成立（3.11 的 `subprocess.run` 超时后只 kill+wait，不重新 drain，已实测）；稳定复现的是整棵子孙进程留存，默认 `timeout=600` 的用例因此要等满 10 分钟才失败 |
| 10.01 | `tools/run_test_suite.py` 拆成 `nox -s full` 周期性调度… | 已合并 | gates | `5501d0b6` 加入稳定 `nox -s full` 完整 CPU 门禁入口，保留 `cpu` 兼容 session；CPU workflow 改调用 `full`，结构合同确认委托和调度入口，定向 2 passed |
| 10.02 | 默认 `nox` 含 cpu 数值测试，或把默认改名为 static | 已合并 | gates | `151c5856`：`nox.options.sessions` 默认列表加入 `cpu`；新增 AST 结构合同确认默认数值门禁存在，定向 1 passed |
| 10.03 | optional/rocm/mpi/nccl 四个 session 排上 runner 或在文档… | 已合并 | gates | a1668aca。CUDA 可由维护者添加 `ci:cuda` 标签触发 PR 真机门禁；当前 runner 能力不覆盖 optional 依赖、ROCm、MPI 与双卡 NCCL，四项在测试支持矩阵中明确为 Manual，结构规则防文档/调度漂移。相关结构 22 passed |
| 10.04 | 假绿清理 | 已合并 | gates | 74cace5f。6 个首行 `return` 改严格预期失败并登记，4 个 `skipIf(True)` 清零；两条内存契约用短循环 RSS 上限进入 slow 层，负向自测证明真实保留会失败；AST 全树规则禁止复发。内存 2 passed，规则/负向 9 passed，旧禁用项 3 xfailed、4 prerequisite-skipped |
| 10.05 | 按 skip 原因分桶统计并在 CI summary 输出，对「本环境应能跑却 skip」设阈值 | 已合并 | gates | `1a423a16`：`tests/conftest.py` 按固定优先级（accelerator/backend/mpi/torch/network/manual/other）汇总 skip reason，CI summary 输出稳定 bucket；`JITTOR_TEST_REQUIRE_EXECUTION=1` 下 `other>0` fail-closed。`tests/structure/test_gate_scope.py` 合成重叠/unknown/阈值节点 2 passed。 |
| 10.06 | `expect_error` 带 `exc_type` 与 `match` | 已合并 | gates | 6753062d 严格 helper；49503f95 与 01536ba4 为全部 34 处旧调用固定异常类型/消息，AST legacy 计数 0，并修复 CUDA `Var` 误调用导致的假绿；796b5338 增加 OpInfo `ErrorInput`/生成式错误电池与覆盖率门禁，46/227（20.3%）。聚焦 helper 4 passed、六个最终调用节点 6 passed、OpInfo 错误电池 47 passed，真实 CUDA 负向节点 1 passed |
| 10.07 | Unary/Binary/Reduction 用 `OpDTypes.supported` | 已合并 | gates | 4af5fbcd。TestCommon 覆盖每个 OpInfo 声明的全部 dtype，BF16 输入保持原生 bfloat16；两条修前契约各失败，修后输入生成 7 passed |
| 10.08 | 已复现缺陷用 `xfail` 而非 `skip` | 已合并 | gates | d7f87e28。OpInfo `xfail` 改为 `pytest.mark.xfail(strict=True, raises=...)`；fft/ifft/rfft 不再以 `supports_autograd=False` 静默绕过，六个 CPU gradcheck/gradgradcheck 节点稳定复现 float64 输出无法 reinterpret 为 complex64 并全部 xfailed，monkeypatch 修复探针产生 XPASS(strict) 且退出非零。NPU crash/hang 隔离 skip 与数学、数值 harness、环境前置 skip 保持不变 |
| 10.09 | 公开 API 与 OpInfo 差集作为 structure 门禁一项 | 已合并 | gates | 8b76a79f。纠正审计把 stub 类方法/重载混进 536 个公开算子的口径；计划点名的 12 个高频项现全部结构化归属：nonzero/unique/einsum/rms_norm 新增独立 NumPy OpInfo，其余 8 项绑定可解析的现存 nodeid 与不适用通用 OpInfo 的理由。门禁修前准确报 4 项缺口，修后 1 passed；四个 CPU reference 节点 4 passed。注册表现为 231 实例/204 distinct name |
| 10.10 | gradcheck 加「故意写错导数应当失败」的负向自测 | 已合并 | gates | 3e83594d。故意把平方的 backward 写成 `3*x`，`gradcheck` 必须抛 `Jacobian mismatch`；定向 1 passed，相关结构 15 passed |
| 10.11 | 设备对拍加 dtype 轴 | 已合并 | gates | 4bf5830c。双方支持时生成 float32/int8/int16 轴，整数逐位比较，浮点容差按 `sqrt(reduce_size)*eps` 下限缩放，CuPy linalg 探针失败硬失败；真实 CUDA sum-int8/max-int16 2 passed |
| 10.12 | `retry` 装饰器记录并上报重试次数 | 已合并 | gates | 402d09ef。恢复成功与最终失败均报告准确 retries/attempts，并暴露调用、最近和累计重试计数；保留原异常并支持 kwargs。聚焦结构 8 passed |
| 10.13 | marker 真正建立 `-m "not slow"` 快门禁或删除 | 已合并 | gates | 821bb6ba。新增 AST 合同确认 smoke 的 native/torch 两次 `_run_pytest_once` 均传入 `-m not slow`，且 `SLOW_FILES` 仍被 gate 覆盖；定点结构节点 1 passed |
| 10.14 | notebook 门禁按 topic 参数化 | 已合并 | gates | 828bc272。fence/materialize/CPU smoke 生成独立 topic nodeid，smoke 共享模块级缓存；33/117 个 skip-execution 单元均带原因标签且低于 35% 上限。50 tests collected，结构/标签 20 passed，单 topic materialize 1 passed |
| 10.15 | 速度 harness 记录并断言两侧线程数、亲和掩码与精度策略 | 已合并 | gates | 9047897a。runner报告实际线程环境、affinity、runtime线程数与精度，harness要求两侧一致；速度类默认至少10次。纯结构契约3 passed、运行文件语法检查通过，未执行大模型 |
| 10.16 | 提供计时 API | 已合并 | pyother | f9f3a23d。`jt.benchmark` 冻结并预先物化输入池，至少一次 warmup 不计时，每轮递归保留 tuple/list/dict 全部输出并强同步后采样；无 Var 输出直接拒绝，返回不可变秒级统计。CPU 回归 3 passed，覆盖跨轮输入复用、嵌套输出、CSE/死码与未物化假快 |
| 10.17 | 异步错误 | 待领 | | |
| 10.18 | 结构测试预算转向核心 | 待领 | | |
| 10.19 | 每个带 `grad()` 的后端算子有对 CPU 参考的梯度单测 | 待领 | | 26f314fd 已建立并保留 26 项 extern `::grad()` inventory，且补齐 cuDNN 3D 小形状 forward/dx/dw CPU generic 对拍（manifest 结构 2 passed，node 收集 1 条，GPU5 定向 1 passed）；复核后撤回完成标记：HCCL 四项仍无 CPU 参考，多卡实机尚未验收，其中 `HcclAllGatherOp::grad()` 仍直接 `LOGf << "not implemented"`，`HcclAllReduceOp`/`HcclBroadcastOp`/`HcclReduceOp` 也需在 Ascend 910B3 多卡上补真实梯度与 CPU 对照。 |
| 10.20 | 给测试提供受支持的内省 API，替代 283 处 `jt.flags.*`、137 处 `com… | 待领 | | |
| 10.21 | import 方向做成 lint 规则 | 待领 | | |
| 10.22 | 多机门禁 | 待领 | | |
| 10.23 | 布局收尾 | 待领 | | |
| 11.01 | 删已被取代的绕过与死路径 | 待领 | | |
| 11.02 | 已提前为 0.20 | 并入 0.20 | | |
| 11.03 | 单文件异常拆分 | 待领 | | |
| 11.04 | 关键接口写成显式契约 | 待领 | | |

### 2026-09-04 第四十三波补充证据

- `8.06`：`1e8e90c6` 为 `aclnn.h` 增加 `#pragma once`，新增重复包含静态合同，1 passed；本机无 CANN/NPU，仍待 Ascend 910B3 实机。
- `2.19`：`45f77257`/`f76e3b90` 将 CUB argsort/arg_reduce 的 offsets dtype 边界改为 `USER_CHECK` 并记录 int64 负向与双 nvcc TU 语法通过；本机无 CUDA，仍待设备负向运行。
- `7.03`：`8647dc4d` 将 `pairwise_distance` 提升为模块级稳定对象并登记 conservative approximate fidelity；身份、metadata、CPU p=2/keepdim 三节点通过。

### 2026-09-04 第四十四波补充证据

- `8.06`：`553b5ec1` 将 SiLU forward owner 接入共享 launcher，backward/Swish/SwiGlu 保持原路径；结构合同 31 passed，本机无 CANN/NPU，仍待 Ascend 910B3 实机。
- `2.19`：`a3890dd9`/`b8e1f592` 将 cuDNN convolution forward 格式边界改为 `USER_CHECK`；结构合同与 nvcc TU 通过，本机无 CUDA 未运行负向，累计 80 处。
- `7.03`：`4a31179c` 将 `cosine_similarity` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；`py_compile`/diff-check 通过，三节点动态测试因首次 JIT 编译过久终止，未宣称通过。

### 2026-09-04 第四十五波补充证据

- `8.06`：`600ee169` 将 BatchMatMul 接入共享 launcher，保留 `cube_math_type` 与同步策略；结构合同 32 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`67e710b7`/`aae33e6a` 将 cuDNN convolution backward-x 格式边界改为 `USER_CHECK`；结构合同与 nvcc TU 通过，本机无 CUDA 未运行负向，累计 81 处。
- `7.03`：`93cd6a53` 将 `svd` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；`py_compile`/diff-check 通过，动态三节点因首次编译过久终止，未宣称通过。

### 2026-09-04 第四十六波补充证据

- `8.06`：`8251d29d` 将 RotaryPositionEmbedding forward 接入共享 launcher，保留三输入与同步策略；结构合同 33 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`0a0e820e`/`aad3ba0c` 将 cuDNN convolution backward-w 格式边界改为 `USER_CHECK`；结构合同与 nvcc TU 通过，本机无 CUDA 未运行负向，累计 82 处。
- `7.03`：`dc8cdfcb` 将 `svd_lowrank` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；`py_compile`/diff-check 通过，动态 JIT 未运行。

### 2026-09-04 第四十七波补充证据

- `8.06`：`71bab738` 将 Maxpool forward 接入共享 launcher，保留 descriptors、`poolCeil`、同步策略及 Avgpool/backward 原路径；静态合同 34 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`37004fe0`/`b48f8af7` 将 cuDNN conv3d 输入 rank 边界改为 `USER_CHECKop`；结构合同与 nvcc TU 通过，本机无 CUDA 未运行负向，累计 83 处。另：`broadcast_to` 源码实际 5 个检查，但 dimension map 仍期望 2（shape map 期望 5），相关门禁仍 1 failed，待专门修复。
- `7.03`：`cfe67a7e` 将 `pca_lowrank` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；`py_compile`/diff-check 通过，动态 JIT 未运行。

### 2026-09-04 第四十八波补充证据

- `8.06`：`16a89606` 将 Avgpool forward 接入共享 launcher，保留 descriptors、`poolCeil/divisor`、同步策略及 backward/其他 pool owner；静态合同 35 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`9d77a5a7`/`cf177243` 将 cuDNN conv3d backward-x 权重 rank 边界改为 `USER_CHECKop`；结构合同与 nvcc TU 通过，本机无 CUDA 未运行负向，累计 84 处。
- `7.03`：`727b440a` 将 `nan_to_num_` 提升为 numerical 稳定 in-place 对象并登记 conservative approximate fidelity；`py_compile`/diff-check 通过，因既有 NaN/Inf JIT abort 风险未运行动态测试。

### 2026-09-04 第四十九波补充证据

- `8.06`：`ba8e2621` 将 TruthReduce all/any 接入共享 launcher，保留双路径异常处理与同步策略；静态合同 36 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`e81ef514`/`a7f45f1f` 将 cuDNN conv3d backward-w 输入 rank 边界改为 `USER_CHECKop`；结构合同与 nvcc TU 通过，本机无 CUDA 未运行负向，累计 85 处。
- `7.03`：`602a813f` 将 `sparse_coo_tensor` factory 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态测试未运行。

### 2026-09-04 第五十波补充证据

- `8.06`：`230c0b69` 将 Conv2d forward 接入共享 launcher，保留 group/bias/descriptor 与同步策略，backward 不变；静态合同 37 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`496dd510`/`e7b10858` 将 cuDNN conv3d backward-x dy rank 边界改为 `USER_CHECKop`；结构合同与 nvcc TU 通过，本机无 CUDA 未运行负向，累计 86 处。
- `7.03`：`32064314` 将 `randint_like` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行。

### 2026-09-04 第五十一波补充证据

- `8.06`：`e86ccd11` 将 RmsNorm forward 接入共享 launcher，保留 `eps`、双输出与同步策略，gradient owner 不变；静态合同 38 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`f07cb966`/`54a88b42` 将 cuDNN conv3d backward-w 的 dy rank 边界改为 `USER_CHECKop`；结构合同与 nvcc TU 通过，本机无 CUDA 未运行负向，累计 87 处。
- `7.03`：`d9c7c6a2` 将 `det` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行。

### 2026-09-04 第五十二波补充证据

- `8.06`：`faf6745e` 将 RmsNormGrad 接入共享 launcher，保留多输入、双输出与同步策略，gradient owner 不变；静态合同 39 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`85ae0688`/`b5e00107` 将 cuDNN conv3d 权重 rank 边界改为 `USER_CHECKop`；结构合同与 nvcc TU 通过，本机无 CUDA 未运行负向，累计 88 处、四十一组证据。
- `7.03`：`9c469b37` 将 `inverse` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行。

### 2026-09-04 第五十三波补充证据

- `8.06`：`3581db5d` 将 Softmax backward 接入共享 launcher，保留 `dim` query 与同步策略；静态合同 40 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`1910f343`/`53db0066` 将 CUB argsort 的 x/indexes rank 边界改为 `USER_CHECK`，累计 89 处、四十二组证据；结构合同与 nvcc TU 通过，本机无 CUDA 未运行负向。
- `7.03`：`8bc2791e` 将 `take_along_dim` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行。

### 2026-09-04 第五十四波补充证据

- `8.06`：`5697f619` 将 Embedding backward 接入共享 launcher，保留 `numEmbeddings`、`paddingIdx`、`scaleGradByFreq` 与同步策略；静态合同 41 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`4fe6f687`/`4f605d00` 将 CUB argsort 循环内 x/indexes shape 边界改为 `USER_CHECK`，累计 90 处、四十三组证据；结构合同与 nvcc TU 通过，本机无 CUDA 未运行负向。
- `7.03`：`48c6fd73` 将 `log1p` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行。

### 2026-09-04 看板一致性修复与并发补充

- 修复 7.14 之后任务表的结构：补回计划中遗漏的 `0.22`、`2.24`、`7.19`、`7.20` 四行，移除 8.06 下只有两列的 `8.06 note` 伪行；增量证据继续集中在本文件末尾的波次小节。
- 主线已包含的并发提交：`166010a8`（CUB argsort offsets rank，2.19）与 `ccbc6132`（`reciprocal` 稳定对象，7.03）；两项保持原任务“待领”状态，作为前置证据记录。

### 2026-09-04 第五十五波补充证据

- `8.06`：`0b149241`/`a12a2fbe` 将 Dropout backward 接入共享 launcher，保留 `scale` query 与同步策略；静态合同 42 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`166010a8`（CUB argsort offsets rank）已进入主线，结构/TU 证据已记录，累计 91 处、四十四组证据；本机无 CUDA 未运行负向。
- `7.03`：`ccbc6132` 将 `reciprocal` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；静态身份/metadata、`py_compile`/diff-check 通过，动态 JIT 未运行。

### 2026-09-04 第五十六波补充证据

- `8.06`：`4f414054`/`14c30c38` 将 RotaryPositionEmbedding gradient 接入共享 launcher，保留四输入、三输出 query 与同步策略；静态合同 43 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`36502b8e`/`fcaa6cce` 将 CUB argsort offsets 长度边界改为 `USER_CHECKop`，并补齐此前 `166010a8` 漏记；累计 92 处、四十五组证据，结构与 nvcc TU 通过，本机无 CUDA 未运行负向。
- `7.03`：`742f1595` 将 `lerp` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行。

### 2026-09-04 第五十七波补充证据

- `8.06`：`f34ecce4`/`393a5f70` 将 Conv2d backward 接入共享 launcher，保留三输出 gradient query、descriptor cleanup 与同步策略；静态合同 44 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`193d5171`/`22ccddc5` 将 CUB arg-reduce offsets rank 边界改为 `USER_CHECKop`，累计 93 处、四十六组证据；结构与 nvcc TU 通过，本机无 CUDA 未运行负向。
- `7.03`：`25142db7` 将 `softmax` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行。

### 2026-09-04 第五十八波补充证据

- `8.06`：`1874f7ed`/`f92d4ffd` 将 UpsampleNearest2d backward 接入共享 launcher，保留 output/input-size RAII descriptor 与同步策略；静态合同 45 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`4c64067e`/`050da89a` 将 CUB arg-reduce offsets 长度边界改为 `USER_CHECKop`，累计 94 处、四十七组证据；结构与 nvcc TU 通过，本机无 CUDA 未运行负向。
- `7.03`：`d6bd24f1` 将 `log_softmax` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行。

### 2026-09-04 第五十五波补充证据

- `8.06`：`0b149241` 将 Dropout backward 接入共享 launcher，保留 `scale` query 与同步策略；静态合同 42 passed，本机无 CANN/NPU，仍待实机。

### 2026-09-04 第五十六波补充证据

- `8.06`：`4f414054` 将 RotaryPositionEmbedding gradient 接入共享 launcher，保留四输入、三输出 query 与同步策略；静态合同 43 passed，本机无 CANN/NPU，仍待实机。

### 2026-09-04 第五十七波补充证据

- `8.06`：`f34ecce4` 将 Conv2d backward 接入共享 launcher，保留三输出 gradient query 与 descriptor cleanup；静态合同 44 passed，本机无 CANN/NPU，仍待实机。

### 2026-09-04 第五十八波补充证据

- `8.06`：`1874f7ed` 将 UpsampleNearest2d backward 接入共享 launcher，保留 output/input-size RAII descriptor 与同步策略；静态合同 45 passed，本机无 CANN/NPU，仍待实机。

### 2026-09-04 第五十九波补充证据

- `8.06`：`2d8f415b` 将 LeakyReLU backward 接入共享 launcher，保留 negativeSlope、selfIsResult、scalar cleanup 与同步策略；静态合同 46 passed，本机无 CANN/NPU，仍待实机。
- `8.06`：`c33196b3` 将 SiLU backward 接入共享 launcher，保留同步策略；静态合同 47 passed，本机无 CANN/NPU，仍待实机。

### 2026-09-04 第六十波补充证据

- `8.06`：`d87bbd09` 将 Swish forward 接入共享 launcher，保留同步策略；静态合同 48 passed，本机无 CANN/NPU，仍待实机。

### 2026-09-04 第六十一波补充证据

- `8.06`：`744f6c6d` 将 Swish backward 接入共享 launcher，保留同步策略，SwiGlu 未迁；静态合同 49 passed，本机无 CANN/NPU，仍待实机。

### 2026-09-04 第六十二波补充证据

- `8.06`：`1f1ffec3` 将 LayerNorm forward 接入共享 launcher，保留 normalizedShape、eps、三输出与 descriptor cleanup；静态合同 51 passed，本机无 CANN/NPU，仍待实机。
- `8.06`：`ca40d0d6` 将 LayerNorm backward 接入共享 launcher，保留 normalizedShape/outMask、三输出 query 与 descriptor cleanup；静态合同 52 passed，本机无 CANN/NPU，仍待实机。
- `8.06`：`8e772a5b` 将 SwiGlu 接入共享 launcher，保留同步策略；静态合同 50 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`fbc69232`/`a4adb24b` 将 cuDNN RNN LSTM mode 用户边界改为 `USER_CHECKop`，累计 95 处、四十八组证据；结构合同与 nvcc TU 通过，本机无 CUDA 未运行负向。
- `7.03`：`b98cde25` 将 `relu` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行。

### 2026-09-04 第六十波补充证据

- `8.06`：`d87bbd09`/`55a81e8e` 将 Swish forward 接入共享 launcher，保留同步策略；静态合同 48 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`b1c604af`/`7c6420d3` 将 cuDNN RNN 非 LSTM mode 边界改为 `USER_CHECKop`，累计 96 处、四十九组证据；结构合同与 nvcc TU 通过，本机无 CUDA 未运行负向。
- `7.03`：`2bdc68a0` 将 `torch._shape_as_tensor` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行。

### 2026-09-04 第六十一波补充证据

- `8.06`：`744f6c6d` 将 Swish backward 接入共享 launcher，保留同步策略，SwiGlu 未迁；静态合同 49 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`6afd44df`/`b20ea9e2` 将 cuDNN RNN `proj_size==0` 用户边界改为 `USER_CHECKop`，累计 97 处、五十组证据；结构合同与 nvcc TU 通过，本机无 CUDA 未运行负向。
- `7.03`：本波复核剩余 API 后仅 `vmap` 仍是复杂闭包，已有原生 owner 的 API 不重复包装；未产生安全代码提交。

### 2026-09-04 第六十二波补充证据

- `8.06`：`8e772a5b`/`012dddf4` 将 SwiGlu 接入共享 launcher，保留同步策略；静态合同 50 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`57c6cd92`/`53895c66` 将 cuDNN RNN 第二处 `proj_size==0` 边界改为 `USER_CHECKop`，累计 98 处、五十一组证据；结构与 nvcc TU 通过，本机无 CUDA 未运行负向。
- `7.03`：`7a7ae622` 将 `outer` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行。

### 2026-09-04 第六十三波补充证据

- `8.06`：`1f1ffec3`/`f74043b9` 将 LayerNorm forward 接入共享 launcher，保留 `normalizedShape`、`eps`、三输出与 descriptor cleanup；静态合同 51 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`aae2f5bc`/`79269b83` 将 cuDNN conv3d 分组通道 shape 边界改为 `USER_CHECKop`，累计 99 处、五十二组证据；结构与 nvcc TU 通过，本机无 CUDA 未运行负向。
- `7.03`：`9a9011ce` 将 `isin` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行。

### 2026-09-04 第六十四波补充证据

- `8.06`：`ca40d0d6`/`2e92d162` 将 LayerNorm backward 接入共享 launcher，保留 `normalizedShape`、`outMask`、三输出 query 与 descriptor cleanup；静态合同 52 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`935bb1a9`/`bae4711f` 将 cuDNN RNN backward-x LSTM mode 边界改为 `USER_CHECKop`，累计 100 处、五十三组证据；结构与 nvcc TU 通过，本机无 CUDA 未运行负向。
- `7.03`：`b5dc26d7` 将 `tensordot` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行。

### 2026-09-04 第六十五波补充证据

- `8.06`：`3f0b8c7d`/`3c0f2115` 将 GroupNorm forward 接入共享 launcher，保留 group/eps、三输出 query 与同步策略；静态合同 53 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`b3826005`/`ae78d185` 将 cuDNN RNN backward-x `proj_size==0` 边界改为 `USER_CHECKop`，累计 101 处、五十四组证据；结构与 nvcc TU 通过，本机无 CUDA 未运行负向。
- `7.03`：`e0bc5294` 将 `repeat_interleave` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行。

### 2026-09-04 第六十六波补充证据

- `8.06`：`016fc62d`/`eb1e89cd` 将 GroupNorm backward 接入共享 launcher，保留 output-mask、group 属性、三输出 query 与 cleanup；静态合同 54 passed，本机无 CANN/NPU，仍待实机。
- `8.06`：`fc849c10` 将 MaskedSelect 接入共享 launcher，保留双输入 mask query 与同步策略；静态合同 57 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`35664df5`/`d3e786e2` 将 cuDNN RNN backward-x 非 LSTM mode 边界改为 `USER_CHECKop`，累计 102 处、五十五组证据；结构与 nvcc TU 通过，本机无 CUDA 未运行负向。
- `7.03`：本波复核剩余候选仅有复杂 `vmap` 闭包，未强行拆分，保持无新增代码提交。

### 2026-09-04 第六十七波补充证据

- `8.06`：`c4f0447c`/`fca8451c` 将 Avgpool backward 接入共享 launcher，保留 `countIncludePad/divisorOverride`、descriptor cleanup 与同步策略；静态合同 55 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`76dc9dc3`/`6baf9dd5` 将 cuDNN RNN backward-x 第二处 `proj_size==0` 边界改为 `USER_CHECKop`，累计 103 处、五十六组证据；结构与 nvcc TU 通过，本机无 CUDA 未运行负向。
- `7.03`：复核剩余 API 后仅 `vmap` 为复杂闭包，本波无安全小切片提交。

### 2026-09-04 第六十八波补充证据

- `8.06`：`a9d73aae`/`efb1b758` 将 Maxpool backward 接入共享 launcher，保留 pool descriptors、`poolCeil`、输出处理、cleanup 与同步策略；静态合同 56 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`408b4832`/`1de5551e` 将 cuDNN conv 输入 rank 边界改为 `USER_CHECKop`，累计 104 处、五十七组证据；结构与 nvcc TU 通过，本机无 CUDA 未运行负向。
- `7.03`：`9bd71961` 新增 `agent/design/vmap-owner-plan.md`，记录复杂 vmap 的 owner、Runtime 依赖、迁移边界与后续 CPU 验收节点；本波仅设计前置，未宣称实现完成。

### 2026-09-04 第六十九波补充证据

- `8.06`：`fc849c10`/`77e1d30d` 将 MaskedSelect 接入共享 launcher，保留双输入 mask query 与同步策略；静态合同 57 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`ceabd84c`/`3096b2f0` 将 cuDNN conv 权重 rank 边界改为 `USER_CHECKop`，累计 105 处、五十八组证据；结构与 nvcc TU 通过，本机无 CUDA 未运行负向。
- `7.03`：`30cd207f`、`27866dc2` 细化 `vmap` owner 的可验证契约与验收节点；仅设计前置，未修改 runtime，未宣称实现完成。

### 2026-09-04 第七十波补充证据

- `8.06`：`18fca063`/`029795fa` 将 Index 接入共享 launcher，保留 index query 与同步策略，SliceV2 未改；静态合同 58 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`44a80c8a`/`3259631f` 将 cuDNN conv 分组通道 shape 边界改为 `USER_CHECKop`，累计 106 处、五十九组证据；结构与 nvcc TU 通过，本机无 CUDA 未运行负向。
- `7.03`：`ed9b2010` 补充 `vmap` owner 提取协议、AST 完成门禁与 `VmapContext` 约束；仅设计前置，未修改 runtime。

### 2026-09-04 第七十一波补充证据

- `8.06`：`2e27d71b`/`e2b6e3f0` 将 SliceV2 接入共享 launcher，保留 begins/ends/steps/axes descriptors 与同步策略，Index/其他 owner 未改；静态合同 59 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`6727dd57`/`eb4db9b4` 将 cuDNN conv backward-x 权重 rank 边界改为 `USER_CHECKop`，累计 107 处、六十组证据；结构与 nvcc TU 通过，本机无 CUDA 未运行负向。
- `7.03`：`abaa242a`、`afa756bc` 连续补充 vmap 设计契约与 unsupported AST 静态门禁；仅设计/门禁前置，未修改 runtime，未宣称实现完成。

### 2026-09-04 第七十二波补充证据

- `8.06`：`ff26ab02`/`7457382d` 将 StridedSliceAssignV2 接入共享 launcher，保留 gradient memset 分支与 slice descriptor handling；静态合同 60 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`92a66390`/`5e720411` 将 cuDNN conv backward-x dy rank 边界改为 `USER_CHECKop`，累计 108 处、六十一组证据；结构与 nvcc TU 通过，本机无 CUDA 未运行负向。
- `7.03`：`41236df4`、`5a8b0115` 补充 vmap context 夹具契约、提取顺序、绑定与回滚步骤；仅设计前置，未修改 runtime。

### 2026-09-04 第七十三波补充证据

- `8.06`：`73b71c7d`/`b124efbf` 将 InplaceMaskedScatter 接入共享 launcher，保留 tracked base-to-output memcpy 依赖与同步策略；静态合同 61 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`1e3bab6e`/`f112976b` 将 cuDNN conv backward-w 输入 rank 边界改为 `USER_CHECKop`，累计 109 处、六十二组证据；结构与 nvcc TU 通过，本机无 CUDA 未运行负向。
- `7.03`：`05e9f37f` 补充 vmap 评审证据清单，覆盖 AST、closure/global、fidelity、聚焦节点与 skip 归因；仅设计前置，未修改 runtime。

### 2026-09-04 第七十四波补充证据

- `8.06`：`4eb360d7`/`2dc68144` 将 IndexPutImpl 接入共享 launcher，保留 index tensor-list handling 与同步策略，IndexPutImplAccumulate 未改；静态合同 62 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`5596563f`/`2702f7c6` 将 cuDNN conv backward-w dy rank 边界改为 `USER_CHECKop`，累计 110 处、六十三组证据；结构与 nvcc TU 通过，本机无 CUDA 未运行负向。
- `7.03`：`9ee118a0` 补充 vmap unsupported 行为矩阵，覆盖 extent/nested dim/非 bool/depth callback/out_dims；仅设计前置，未修改 runtime。

### 2026-09-04 第七十五波补充证据

- `8.06`：`f353076a`/`1cc7aa53` 将 IndexPutImpl accumulate 接入共享 launcher，保留 tracked output memset 与 index tensor-list dependency；静态合同 63 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`241ab528`/`7f7c9bbc` 将 cuDNN RNN 推理阶段输入 rank 边界改为 `USER_CHECKop`，累计 111 处、六十四组证据；结构与 nvcc TU 通过，本机无 CUDA 未运行负向。
- `7.03`：`ba76983c` 明确 vmap 仅做组织重构，不新增 kernel/设备传输/优化，并定义 CPU/CUDA/ACL 分层验收与 skip 归因；仅设计前置。

### 2026-09-04 第七十六波补充证据

- `8.06`：`3dd89256`/`90d73767` 将 AdamWList 各项更新接入共享 launcher，保留 fused D2D copy checks 与唯一同步点；静态合同 64 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`4858b0a2`/`1a9acf32` 将 cuDNN RNN 输入通道 shape 边界改为 `USER_CHECKop`，累计 112 处、六十五组证据；结构与 nvcc TU 通过，本机无 CUDA 未运行负向。
- `7.03`：`f6d3a435` 明确 vmap 稳定签名、内部 callback 注入和 unsupported kwargs 拒绝；仅设计前置，未修改 runtime。

### 2026-09-04 第七十七波补充证据

- `8.06`：`5f989f16`/`24a9e438` 将 FlashAttention forward 接入共享 launcher，保留 prefix/qstart/kvstart RAII descriptors 与同步策略，backward/KV-cache 未改；静态合同 65 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：本波复核剩余 CUDA/CUDNN/CUB/NCCL 断言均属内部不变量或后端运行失败，没有新增安全用户边界迁移。
- `7.03`：`d316a706` 补充 vmap AST 门禁输出契约，定义计数、禁止捕获、unsupported guard 与 fail-closed 证据格式；仅设计前置。

### 2026-09-04 第七十八波补充证据

- `2.19`：`a29e0f81`/`c9866b48` 新增后端内部断言分类文档与结构门禁，1 passed；用户边界累计保持 112 处、65 组，不改运行时错误语义。
- `7.03`：`d90a716a`、`9ee118a0`、`05e9f37f`、`41236df4`、`5a8b0115`、`ba76983c`、`f6d3a435`、`9bd71961`、`ed9b2010`、`d316a706`、`abaa242a`、`afa756bc` 逐步补充 vmap owner、context、unsupported、后端分层与 AST 评审契约；未修改 runtime，未宣称实现完成。
- `8.06`：本波暂未新增 ACL family；上一波 FlashAttention backward 已在前面 canonical 记录中，保持无 CANN/NPU 实机验证声明。

### 2026-09-04 第七十九波补充证据

- `8.06`：`e1470830`/`4e1f6ba0` 将 IncrementalFlashAttention 接入共享 launcher，保留 block-table、actual-sequence、cache-view cleanup 与同步策略，KVCacheMemcpy 未迁；静态合同 67 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`7a24ca0b`/`7c1565d2` 将 cuFFT jit_prepare unsupported dtype 边界改为 `USER_CHECK`，累计 113 处、六十六组证据；结构与 nvcc TU 通过，本机无 CUDA 未运行负向。
- `7.03`：`e8260779` 明确 vmap 版本兼容、kwargs 策略与退出标准；仅设计前置，未修改 runtime。

### 2026-09-04 第八十波补充证据

- `8.06`：本波复核确认标准 workspace/query/execute/sync owner 已全部迁移；剩余 KVCacheMemcpy 为逐 token `aclrtMemcpyAsync` 专用路径，不纳入通用 launcher。
- `2.19`：`040e44a0`/`d251d738` 将 CUBLAS matmul 输入 rank 边界改为 `USER_CHECK`，累计 114 处、六十七组证据；结构与 nvcc TU 通过，本机无 CUDA 未运行负向。
- `7.03`：`7d8fdd37` 补充 vmap 无可变全局、幂等 install、失败回滚与资源释放门禁；仅设计/门禁前置。

### 2026-09-04 第八十一波补充证据

- `2.19`：`91718e98`/`c041b0a7` 将 cuDNN RNN 权重查询内部断言纳入分类文档与结构门禁；1 passed，不改变用户错误语义或累计数。
- `7.03`：`79455300` 固定 vmap 首门禁夹具（seed=17、简单映射与 nested bool shape），要求记录 fixture/NumPy 期望/unsupported 矩阵；仅设计前置，未修改 runtime。
- `8.06`：复核确认标准 launcher owner 已穷尽，剩余 KVCacheMemcpy 等专用路径不纳入通用 launcher，本波无代码提交。

### 2026-09-04 第八十二波补充证据

- `8.06`：`6905864a` 明确标准 workspace/query/execute/sync owner 已迁移完毕，剩余 KVCacheMemcpy 为逐 token 专用 memcpy 路径，不纳入通用 launcher；本波无代码提交。
- `2.19`：`a29e0f81`/`be526722`/`8382497d` 新增后端内部断言分类文档与 CUDNN RNN bias/descriptor 门禁，结构门禁 1 passed；不改变用户错误语义，用户边界累计保持 114 处。
- `7.03`：`1abf0d75`、`d6626bd0`、`41236df4`、`5a8b0115` 补充 vmap 固定数据、context 夹具、handoff 证据模板等设计契约；未修改 runtime，未宣称实现完成。

### 2026-09-04 第八十三波补充证据

- `2.19`：`27cc72f2`/`23d70b26` 补充 CUB 状态内部断言分类门禁与说明文档；不改变运行时语义，用户边界累计保持 114 处。
- `7.03`：`de512c43` 补充 vmap AST 实现草案，定义 module/install/nested 计数与 binding 行号提取伪代码；仅设计前置，未修改 runtime。
- `8.06`：只读确认标准 launcher owner 已穷尽，KVCacheMemcpy 等专用 memcpy 路径不纳入通用 launcher，本波无代码提交。

### 2026-09-04 第八十四波补充证据

- `2.19`：`24848098`/`0bfb854e` 将 CUBLAS 测试入口返回码内部断言纳入分类门禁；1 passed，不改变运行语义或用户边界累计。
- `7.03`：`30f5e2de` 细化 vmap context 泄漏 AST 门禁，覆盖默认参数、注解、decorator、closure 白名单与模块全局扫描；仅设计/门禁前置。
- `8.06`：只读确认标准 workspace/query/execute/sync owner 已全部迁移，KVCacheMemcpy 等专用路径不纳入通用 launcher，本波无代码提交。

### 2026-09-04 第八十五波补充证据

- `2.19`：`54e8d545`/`4229cba5` 将 CUDNN 测试入口返回码内部断言纳入分类门禁；1 passed，不改变运行语义或用户边界累计。
- `7.03`：`40aed528` 补充 vmap fidelity registry 静态门禁，校验 implementation identity、approximate level、context/backend detail 与重复 install；仅设计/门禁前置。
- `8.06`：只读确认标准 ACL launcher owner 已穷尽，KVCacheMemcpy 等专用路径不纳入通用 launcher；本波无代码提交。

### 2026-09-04 第八十六波补充证据

- `2.19`：`d269a52a`/`8595e479` 收束后端内部断言分类说明与门禁，覆盖 CUDNN/CUBLAS/CUB 状态路径；不改变运行语义或用户边界累计。
- `7.03`：`d97d5620` 补充 vmap 发布检查清单，覆盖 clean import、重复 install identity、board/handoff 链接、回滚与无缓存产物；仅设计/门禁前置。
- `8.06`：只读确认标准 ACL launcher owner 已穷尽，KVCacheMemcpy 等专用路径不纳入通用 launcher，本波无代码提交。

### 2026-09-04 第八十七波补充证据

- `2.19`：`8af5fd8d` 精确约束 Cutt 返回码内部断言并纳入分类门禁；不改变运行语义或用户边界累计。
- `7.03`：`2b7b64ca` 补充 vmap 变更控制与 reviewer sign-off，明确 code/doc 分离、owner/context/unsupported 审阅与冲突规则；仅设计前置。
- `8.06`：只读确认标准 ACL launcher owner 已穷尽，KVCacheMemcpy 等专用路径不纳入通用 launcher，本波无代码提交。

### 2026-09-04 第八十八波补充证据

- `2.19`：`30bfdb6e`/`e853b873` 将 CUDNN RNN descriptor 内部断言纳入分类门禁；1 passed，不改变运行语义，用户边界累计保持 113 处。
- `7.03`：`ce1cbb5c` 补充 vmap metadata 兼容契约，固定 `_jittor_vmap_base`/`_jittor_vmap_specs` 身份、形状与嵌套层级；仅设计前置，未修改 runtime。
- `8.06`：只读确认标准 ACL launcher owner 已穷尽，KVCacheMemcpy 等专用路径不纳入通用 launcher，本波无代码提交。

### 2026-09-04 第八十九波补充证据

- `2.19`：`46ab1e17`/`20225dca` 精确约束 cuDNN plan `ASSERT(ok)` 内部断言计数并纳入门禁；1 passed，不改变运行语义或用户边界累计。
- `7.03`：`06899cc0` 补充 vmap `in_dims/out_dims` 的 int、None、tuple/list、负轴归一化矩阵；仅设计/门禁前置，未修改 runtime。
- `8.06`：只读确认标准 ACL launcher owner 已穷尽，KVCacheMemcpy 保持专用路径，本波无代码提交。

### 2026-09-04 第九十波补充证据

- `2.19`：`9f6afe17`/`143cb8e3` 补充 CUB 测试入口内部断言精确计数门禁；结构门禁 1 passed，不改变运行语义或用户边界累计。
- `7.03`：`9ad2c132` 补充 vmap 嵌套 metadata 深度契约，固定 specs 追加、base identity 与 batch shape 顺序；仅设计前置，未修改 runtime。
- `8.06`：只读确认标准 ACL launcher owner 已穷尽，KVCacheMemcpy 保持专用路径，本波无代码提交。

### 2026-09-04 第九十一波补充证据

- `2.19`：`bf5fef1e`/`a7bc9595` 收紧 CUBLAS/CUDNN 测试入口内部断言精确计数门禁；结构门禁 1 passed，不改变运行语义或用户边界累计。
- `7.03`：`ac38772d` 补充 vmap kwargs 兼容矩阵，明确 in/out_dims 归一化、randomness/chunk_size unsupported、未知 kwargs TypeError 与静态门禁；仅设计前置。
- `8.06`：只读确认标准 ACL launcher owner 已全部处理，KVCacheMemcpy 保持专用路径，本波无代码提交。

### 2026-09-04 第九十二波补充证据

- `2.19`：`8c2ebaa8`/`92216275` 将 CUB 测试 CUDA 状态断言收紧为精确计数门禁；结构门禁 1 passed，不改变运行语义或用户边界累计。
- `7.03`：`b233cf6a` 补充 vmap 标量/zero-dim 输出契约，固定 singleton 归一化、nested batch 轴与 `out_dims` 形状；仅设计前置，未修改 runtime。
- `8.06`：只读确认标准 ACL launcher owner 已穷尽，KVCacheMemcpy 保持专用路径，本波无代码提交。

### 2026-09-04 第九十三波补充证据

- `2.19`：`66fd400d`/`5599b11b` 精确约束 cuDNN convolution `best_algo_idx!=-1` 内部断言计数并纳入门禁；1 passed，不改变用户边界或运行语义。
- `7.03`：`aa882756` 补充 vmap autograd 契约，明确 loop/stack 梯度、bool fast path 非微分边界与 CPU gradient 节点草案；仅设计前置。
- `8.06`：只读确认标准 ACL launcher owner 已穷尽，KVCacheMemcpy 保持专用路径，本波无代码提交。

### 2026-09-04 第九十四波补充证据

- `2.19`：`9d2752e2`/`9ad1a807` 精确约束 cuDNN 3D convolution `best_algo_idx!=-1` 内部断言计数并纳入门禁；1 passed，不改变用户边界或运行语义。
- `7.03`：`ae5623e8` 补充 vmap 并发契约，明确 re-entrant 调用、context 生命周期隔离及线程安全 probe；仅设计前置，未修改 runtime。
- `8.06`：只读确认标准 ACL launcher owner 已穷尽，KVCacheMemcpy 保持专用路径，本波无代码提交。

### 2026-09-04 第九十五波补充证据

- `10.13`：`821bb6ba` 新增 smoke AST 快门禁，确认 native/torch 两次运行均传入 `-m not slow` 且 `SLOW_FILES` 仍被 gate 覆盖；定点结构节点 1 passed，任务已完整关闭。
- `2.19`：`903b8d3f`/`2a55244b` 精确约束 CUDNN backward-x `best_algo_idx!=-1` 内部断言计数；结构门禁 1 passed，不改变用户边界累计。
- `7.03`：`17a15406` 补充 vmap 资源边界/取消契约，覆盖 footprint 上限、超大 extent、异常清理和 hold-vars/fidelity 泄漏；仅设计前置。

### 2026-09-04 第九十七波补充证据

- `0.15`：只读核对 RingBuffer GIL/有界等待与长文件拆分方案。当前 `test_children_died` 需要 Linux timed-wait/GIL 安全实现；两个长 compat 文件无法在保持 nodeid/import 语义与 `loadfile` 覆盖的前提下安全拆分降时。smoke 仍约 390s、预算约 446s，未提交代码，任务保持待领。

### 2026-09-04 第九十八波补充证据

- `0.15`：`23814b9a` 已实现 Linux `RingBuffer::wait_pop_for`、等待阶段 GIL 安全拆分，并将 `test_children_died` 改为有界 timeout；worker-death 聚焦节点 1 passed（约 106s，含核心重编）。`SLOW_FILES` 尚未移除，smoke `<300s` 尚未重测，任务继续保持待领。

### 2026-09-04 第九十九波补充证据

- `0.15`：`d3f4853e` 修正 `wait_pop_for` 的单次绝对 deadline、EINTR/非零状态处理，并在恢复 GIL 后保留原异常（不再把 stop/其他错误统一改写为 timeout）；`ring_buffer.cc` 与 `py_ring_buffer.cc` TU 语法检查通过。固定 timeout 作用范围、Dataset 专用轮询、`SLOW_FILES` 与 smoke `<300s` 仍待专项验收，任务保持待领。

### 2026-09-04 第九十九波补充证据（修正版）

- `0.15`：`876ec09c` 恢复通用 `pop()` 无限等待/兼容行为，新增 Dataset 专用 `pop_for(timeout_ms)`；`wait_pop_for` 单次 deadline、伪唤醒/EINTR/stop/其他异常处理及 GIL 边界已修正，协议合同与 killed-worker 聚焦节点各 1 passed（后者 19.72s）。`SLOW_FILES` 尚未移除，smoke `<300s` 尚未重测，任务继续待领。

### 2026-09-04 第九十六波补充证据

- `10.05`：`1a423a16`/`f7f33f5b` 固定 skip reason bucket 优先级、CI summary 和 `other>0` fail-closed；合成结构测试 2 passed，任务已完整关闭。
- `0.15`：RingBuffer GIL/有界等待方案完成只读审计，当前未提交代码；仍需 Linux timed-wait/GIL 安全实现、worker death 聚焦节点与 smoke 重新测量，不能标完成。

### 2026-09-04 第九十三波补充证据

- `2.19`：`66fd400d` 补充 CUDNN convolution `ASSERT(best_algo_idx!=-1)` 精确计数门禁；结构门禁 1 passed，不改变运行语义或用户边界累计。

### 2026-09-04 第九十四波补充证据

- `2.19`：`9d2752e2` 补充 CUDNN 3D convolution `ASSERT(best_algo_idx!=-1)` 精确计数门禁；结构门禁 1 passed，不改变运行语义或用户边界累计。

### 2026-09-04 第九十五波补充证据

- `2.19`：`903b8d3f` 补充 CUDNN backward-x `ASSERT(best_algo_idx!=-1)` 精确计数门禁；结构门禁 1 passed，不改变运行语义或用户边界累计。
