# 整改看板

一行一个任务，与 [refactor-plan.md](refactor-plan.md) 的编号对应。领任务把状态改成「进行中」并写名字，
完成改成「已合并」并填提交号；推送冲突说明别人先领了。状态只有四种：待领 / 进行中 / 已合并 / 并入 X。

## 起点已知失败清单（归责之前先减掉这些）

**任何失败在算成回归之前，先确认它在分支起点是否也失败。** 分支起点是
`9eb696d9`（`origin/2.0`，即 `merge-base origin/2.0 origin/2.0-refactor`）。
没有这份清单，会把继承来的失败当成新回归、把责任安到无辜的提交上。

### A. 分支起点就存在的失败——不是任何 agent 引入的

| 用例 | 症状 |
| --- | --- |
| `tests/compat/torch/test_torch_compat.py` | `RandomOp` 子进程段错误 |
| `tests/compiler/test_atomic_tuner.py::TestAtomicTunerClass::test_atomic_tuner` | 第 4 项 `x.sum()+x.sqr().mean()` 期望两条 `atomictuner: move atomicAdd to loop -1`，实得 0 条。根因是 `032ecfe1`（2026-08-28，起点前 202 个提交）把 CUDA 全量归约改走 `nn/backends/full_reduce_cuda.py` 的 cub 两级折叠 code op，整条全归约不再进融合算子 JIT，AtomicTunerPass 根本看不到 atomic 语句。前三项 add/max/min（reindex_reduce）在起点与起点父提交上都通过 |

（这份表正在用一棵钉在 `9eb696d9` 的只读 worktree 实测补全，跑完会把失败 nodeid 逐条列全。）

**`test_atomic_tuner` 已定论，`9eb696d9` 洗清。** 两棵只读 worktree、两份独立 `JITTOR_HOME`、同一条用例、**串行**跑（并行会串号，见 skill）：`9eb696d9^`(`a88ae02a`) 与 `9eb696d9` 的失败**逐字一致**——同为第 69 行第 4 项 `AssertionError: (0, 2)`。WarpReducePass 挂在 `pass_manager.cc` 的 `AtomicTunerPass` **之后**，原子调优早已打完日志才轮到它改写，它不可能吃掉这些日志。真正的原因见上表那一行：`032ecfe1` 的全归约快路径绕开了整个 JIT。**该用例现在断言的是一条已经不存在的代码路径，属于过期断言，不是回归。**

### B. 已归责、修复进行中——不要重复归因

| 用例 | 引入提交 | 责任 |
| --- | --- | --- |
| `tests/compat/torch/test_torch_compat_interpolate.py::TestInterpolateBicubic::test_bicubic_constant_stays_constant` | `13ac1d14` [6.C05] | coreops，正改成变长编码（原属 3.02） |
| `tests/structure/test_runtime_composition_structure.py::test_moved_scope_state_stays_synchronized_with_the_root` | `956c4b23` [6.B15] | dist，同时会重写该用例本身 |

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
| `70d97137` | **822 passed / 816 skipped / 0 failed**（50 分） | 进行中 | 未开始 | 原生绿；收集总数 1540 → 1638，是有人加了测试 |

## 热点文件占有

第一波（2026-09-02 21:xx 派出）：九个 agent 并行，各自独占一张卡/一段核/一个 worktree
（`/home/zy/jittor-lab/refactor/<分区>`，分支 `wk/<分区>`，推送到 `2.0-refactor`）。

| 分区 | 当前占有者 | 任务 |
| --- | --- | --- |
| 核心节点 | — |  |
| 执行器 | — |  |
| 代码生成 | — |  |
| 类型与日志 | coreops (6.C01/05/06/07/09/30) | GPU3 c24-35 |
| 内存 | mem (6.C10–6.C20) | GPU2 c12-23 |
| 绑定 | bindings (6.C02/22/23/24/25/27/28/29) | GPU0 c0-11 |
| Python 核心 | — |  |
| Python 算子 | pyops (6.P03–6.P09) | GPU5 c36-47 |
| Python 其他 | pyother (5.18、5.19、6.P25) | GPU6 c64-75 |
| 兼容层 | compat (7.01) | CPU c96-103 |
| CUDA 后端 | cudabk (6.B05/07/08/09/12/13/14) | GPU7 c76-87 |
| ACL/ROCm/Corex | — |  |
| 分布式 | dist (6.B01/03/04/06/10/11/15) | CPU c88-95 |
| 构建 | build (0.07–0.11/0.17, 9.02–9.06/9.08/9.09/9.11/9.15/9.17, 9.04 部分) | GPU4 c48-63 |
| 门禁 | gates (0.01–0.04/06/12/13/18/19) | CPU c104-111 |

## 执行中出现的、需要认领的杂项

| 事项 | 现象 | 建议归属 |
| --- | --- | --- |
| wheel 内容基线过期 | **已处理**：基线本身没错，错的是那条断言写死了条目数——任何人加一个模块都会改变它。用当前源码树真构建一个 wheel 核对，45 个新增全是各分区加的合法源码、1 个删除是 9.17 删的 `flags.cc`，基线整个重新生成（861 条）；条目数那条断言换成「基线头部的 `# entries:` 必须与自身条目数一致」这条规则。注意基线是对**当时的源码内容**取的哈希，发版前需要再刷一次（办法写在提交说明里） | 构建，`f869cab8` |
| 结构测试子进程超时 flaky | 已处理，并已与 0.21 合并成一份实现：超时预算搬进 `tests/_helpers/child_process.DEFAULT_TIMEOUT`（600s，`JITTOR_TEST_CHILD_TIMEOUT` 与旧名 `JITTOR_TEST_SUBPROCESS_TIMEOUT` 都认，仍在门禁 `--timeout=900` 之内），`process_modes.SUBPROCESS_TIMEOUT` 随之删除——同一件事只留一处 | 门禁 gates，`46dbe946` |
| `split{i}` 与 `parallel` 不兼容 | 同时设这两个 loop option，`ParallelPass` 在 `ASSERT(def)` 上失败（`Check failed: def`）。`SplitLoopPass` 给内层循环的 range 是 `::min(range{i}-id{i}, stride{i})`，定义在外层循环里且随它变化，`ParallelPass` 在调用点 `func->find_define` 找不到、也无法在调用点求值。CUDA 恒走 `ParallelPass`，所以 CUDA 上任何 split 候选都必然编译失败。用例已钉住：`tests/compiler/test_reduce_tuner.py::test_a_split_candidate_would_not_compile_under_parallel` | 代码生成分区，1.04 的前置 |
| CUDA 归约需要的是线程分解候选，不是 CPU 那套 | `orderN` 候选实测五种形状全部不优于默认（最差 2.1 倍，破坏访存合并），`split{i}` 被上一条挡着，L1 分块尺寸对 GPU 无意义。真正有用的候选是 `ParallelPass` 里的线程分解，属于新工作 | 代码生成分区，待 1.04 前置解决后 |
| `para_opt_level=4` 的块内共享内存归约比默认慢 1.6–2.0 倍 | 实测四种 UNet 形状：默认（warp shuffle）15.7/14.0/15.0/18.1us，lvl 4（`SharedReducePass`）25.3/31.3/25.3/34.8us，不优化 157/92/159/171us。默认值保持 3。要提升需要「warp shuffle → 每 warp 一个值 → 共享内存 → 每输出一次原子」的混合实现，并且要有生态 harness 的端到端数据；数据与方法在 `agent/skills/cuda-reduction-strategy-comparison/` | 代码生成分区，新任务待派 |
| `tests/core/test_type_system.py` 一套门禁都不跑 | **已修**：0.04 之后 CPU 门禁的 torch 会话就是 `TORCH_MODE_PATHS` 本身，这个文件自然进来了。同一批还有 233 个此前一套 workflow 都碰不到的文件 | 门禁 gates，`6adbf488` |
| `test_atomic_tuner` 抓不到日志 | **已修**：根因确认为 `032ecfe1` 的 `full_reduce_cuda.py` 快路径猴补 `Var.sum`/`Var.mean`，全归约不再进 JIT；第 4 条用例改走 `jt.reduce` | codegen，`72f020b3` |
| `asm_tuner.py` 非原子写 `.s`，并发编译读到截断的汇编 | **已修**：`pass_asm()` 改成写 `<路径>.tmp.<pid>` 再 `os.replace`。判据是 inode——改名换 inode，原地重写不换，也就不会消掉那个窗口；用例 `test_asm_tuner.py::TestAsmTunerWritesAtomically` 钉住。缓存里已经存在的坏 `.s` 不会自动修复，删掉再跑 | 构建，`1919b035` |
| `tests/backends/cuda/test_backend_teardown.py` 过不了 0.21 的静态门禁 | `272f00ba`（6.B17）加的 `subprocess.run([sys.executable, ...])` 自己拼了 PYTHONPATH，但没走 `_helpers/child_process`，而 `46dbe946` 的静态检查禁止这么写。两个提交是并行落地的，谁都没错，只是撞上了。现症：`tests/structure/test_child_process_contract.py` 两个用例红（`test_no_test_names_the_interpreter_directly`、`test_every_child_launch_pins_this_tree`） | CUDA 后端 cudabk，改成 `run_child_script` 即可 |
| `jt.flags.nvcc_flags` 的拼法变了 | 9.08 之后架构 flag 是 `--generate-code=arch=...,code=...`，不再是 `-arch=compute_N -code=sm_N`。按后者做字符串匹配的地方要改 | 各分区自查，`2d71f792` |
| 全树跑时 `test_notebooks.py` 没有被当成 manual 跳过 | `tests/conftest.py` 的 `pytest_collection_modifyitems` 里，`test_notebooks.py` 的 `pytest.mark.manual` 是在 `SELECTION_IS_BROAD` 那段跳过判断**之后**才加上的，所以全树跑时它照跑不误——2026-09-03 的全树原生一遍里它实测 537 秒，是全树最慢的一项（第二名 289 秒）。顺序问题，不是标记问题 | 门禁 gates，随 0.13/0.15 |

## 任务

| 编号 | 任务 | 状态 | 负责 | 提交 |
| --- | --- | --- | --- | --- |
| 0.01 | `TestGradients` 改用 `only_for=("cpu",)` 显式实例化 | 已合并 | gates | aee8ecaa（+355deb6e） |
| 0.02 | 设备过滤后 bases 为空或方法数为 0 时生成器直接 raise | 已合并 | gates | e5eb0d05 |
| 0.03 | `tests/compiler/test_jit_tests.py` 进 CPU 门禁，并断言 … | 已合并 | gates | a5e7f654 |
| 0.04 | 门禁改为「整个 `tests/` 减显式排除清单」，排除项必须写理由 | 已合并 | gates | 6adbf488、689e206b |
| 0.05 | 生态对拍进 nightly | 待领 | | |
| 0.06 | `make_tensor` 种子改为 `hash(nodeid, shape, dtype)` … | 进行中 | gates |  |
| 0.07 | 缓存路径追加构建配置指纹 | 已合并 | 构建 | 82dfce6e、6379b2b5、6fdb3807、b25fcdfa（复验） |
| 0.08 | 锁统一为一种类型、一个 fd | 已合并 | 构建 | 460bead0 |
| 0.09 | 探测结果落盘 `cache_path/probe.json` | 已合并 | 构建 | 240a92a3 |
| 0.10 | 写缓存前检查可用磁盘空间，不足时给明确错误 | 已合并 | 构建 | 73eceeaf |
| 0.11 | 「jit_utils 已更新请重跑」改非零退出码 | 已合并 | 构建 | 7e8c7c74 |
| 0.12 | 14 处在用例里裸赋值 `jt.flags.*` 且无 tearDown 的测试改 `flag_… | 已合并 | gates | 26a20905 |
| 0.13 | conftest 的模式由显式环境变量决定，删除 `sys.argv` 嗅探 | 进行中 | gates |  |
| 0.14 | `_session_env` 不再 `os.environ.copy()` | 待领 | | |
| 0.15 | 门禁分两层 | 待领 | | |
| 0.16 | `test_device_parity.py` 按算子分片并行，不再在 `setUpClass`… | 待领 | | |
| 0.17 | `pyproject.toml` 的 `pythonpath` 改由 conftest 按环境变… | 已合并 | 构建 | b19d098f |
| 0.18 | 门禁每条目断言至少执行 1 个非 skip 用例 | 进行中 | gates | ee29bee3（记账与报告已合并；`EXECUTES_NOTHING` 待全树数据填完后开 `JITTOR_TEST_REQUIRE_EXECUTION`） |
| 0.19 | 结构测试从「精确清单」改成「规则」 | 已合并 | gates | c3bcd277 |
| 0.20 | 布局收尾 | 待领 | | |
| 0.21 | 测试起的子进程不带 PYTHONPATH，门禁机器上是假绿 | 已合并 | gates | 46dbe946、a5ce7310 |
| 1.01 | 把 `utils/data.gz` 解出的 `data.cc` 还原为可读的五个翻译单元 | 已合并 | codegen | ecb6a112（+72f020b3 用例） |
| 1.02 | `op_compiler.cc:30-69` 用正则给 `ParallelPass` 输出打补丁… | 已合并 | codegen | 3eb34e6a |
| 1.03 | 查明 `SharedReducePass` 在约 4900 个归约 kernel 里零命中的触发… | 已合并 | codegen | 3eb34e6a |
| 1.04 | `ReduceTuner::run` 不再对 CUDA 直接返回 | 已合并 | codegen | aebb1d73 |
| 1.05 | 布局收尾 | 待领 | | |
| 2.01 | Var 与 Op 各持自己的 flag 类型 | 待领 | | |
| 2.02 | 删除 `Node::custom_data` | 待领 | | |
| 2.03 | `tflag` 全局计数器加魔数改为 epoch 对象或局部集合 | 待领 | | |
| 2.04 | `Var::allocator` 去类型双关 | 待领 | | |
| 2.05 | 真正的 0 维张量 | 待领 | | |
| 2.06 | 边表由 list 加反向迭代器改 SmallVector，按下标 O(1) | 待领 | | |
| 2.07 | `hold_vars`/`sync_ptr` 析构里 `std::next(end())` 的 … | 待领 | | |
| 2.08 | `Node` 不再 include `pybind/py_var_tracer.h` | 待领 | | |
| 2.09 | `th_mode` 从 C++ 核心上移为 autograd 策略对象 | 待领 | | |
| 2.10 | 三套 liveness 计数 | 待领 | | |
| 2.11 | `VarHolder` 不再是执行触发点 | 待领 | | |
| 2.12 | 打破 `Executor ⇄ VarHolder` include 环 | 待领 | | |
| 2.13 | 执行相关全局状态 | 待领 | | |
| 2.14 | `src/misc/` 拆散 | 待领 | | |
| 2.15 | NanoString | 待领 | | |
| 2.16 | 类型提升表 | 待领 | | |
| 2.17 | 算子身份用注册期整型 id | 待领 | | |
| 2.18 | 算子注册表惰性初始化 | 待领 | | |
| 2.19 | 错误分两档 | 待领 | | |
| 2.20 | 信号处理器只做 `write` 与 `_exit`，符号化交给预建 helper 进程 | 待领 | | |
| 2.21 | `DEFINE_FLAG_WITH_SETTER` 先赋值再调 setter，签名收新旧两值 | 待领 | | |
| 2.22 | 环境变量统一 `JT_` 前缀 | 待领 | | |
| 2.23 | 布局收尾 | 待领 | | |
| 3.01 | `Executor::run_sync` | 待领 | | |
| 3.02 | jit key 结构化 | 待领 | | |
| 3.03 | 三张 kernel 缓存表键改 `string` | 待领 | | |
| 3.04 | 求 jit key 改纯函数 | 待领 | | |
| 3.05 | 删除算子构造期回调执行器 | 待领 | | |
| 3.06 | 并行编译器修到可信 | 待领 | | |
| 3.07 | 执行器在设备等待段释放 GIL | 待领 | | |
| 3.08 | KernelIR 结构化 | 待领 | | |
| 3.09 | 死代码消除不再按「语句含 `void` 一词」删除 | 待领 | | |
| 3.10 | 算子内标识符改名走结构化成员表并先做合法性校验，替代三个硬编码白名单与 `op{i}_` 盲目前… | 待领 | | |
| 3.11 | 生成源码里的结构体字节偏移改显式 setter，成员表用宏声明 | 待领 | | |
| 3.12 | `float_atomic_fix_pass.cc:76-80`、`fake_main_pass… | 待领 | | |
| 3.13 | 循环维度身份用整数向量，`range10` 不再被拆成 `range1*range0` | 待领 | | |
| 3.14 | 两个同名 pass | 待领 | | |
| 3.15 | 一次编译只解析一遍 | 待领 | | |
| 3.16 | `token_replace_all` 不再用 CHECK 抛异常做循环终止 | 待领 | | |
| 3.17 | 只用于代码生成的 JIT 区段与普通 C++ 分离 | 待领 | | |
| 3.18 | 删掉 `asm_tuner` 链路 | 待领 | | |
| 3.19 | `event_queue` 异步基础设施修好并加测试，或删除 | 待领 | | |
| 3.20 | 执行器提供「提交部分图」显式接口，`jt.grad` 与 `Function` 回调降开销，让反… | 待领 | | |
| 3.21 | 每算子建图成本 | 待领 | | |
| 3.22 | CUDA 归约块内树形归约 | 待领 | | |
| 3.23 | 融合逐元素 kernel 带宽效率 | 待领 | | |
| 3.24 | 布局收尾 | 待领 | | |
| 4.01 | 分配器 id 空间随分配器实例走，不再是进程静态 2M 单例 | 待领 | | |
| 4.02 | 合并多卡 | 待领 | | |
| 4.03 | `BackendRegistry` | 待领 | | |
| 4.04 | `OpRegistry` | 待领 | | |
| 4.05 | Python 分派表 | 待领 | | |
| 4.06 | `jt.flags.backend_fallback ∈ {error, warn, allow… | 待领 | | |
| 4.07 | 后端配置改为返回 `BuildConfig` 值 | 待领 | | |
| 4.08 | 流与事件模型 | 待领 | | |
| 4.09 | per-device 库句柄 | 待领 | | |
| 4.10 | CUDA kernel 存放位置统一 | 待领 | | |
| 4.11 | ACL 改为注册表后端 | 待领 | | |
| 4.12 | 删除 `process_jittor_source` 与 `process_acl` | 待领 | | |
| 4.13 | 跨后端契约矩阵 | 待领 | | |
| 4.14 | `Module.cuda(i)`/`npu(i)`/`x.to(...)`/`x.cpu()` … | 待领 | | |
| 4.15 | 布局收尾 | 待领 | | |
| 5.01 | 114 个 `foo_` 就地方法改白名单显式声明 | 待领 | | |
| 5.02 | 视图与存储模型 | 待领 | | |
| 5.03 | 转置隐藏标记 | 待领 | | |
| 5.04 | 参数模型 | 待领 | | |
| 5.05 | `eval()`/`train()` 只切 `is_train`，冻结统一由 `requires… | 待领 | | |
| 5.06 | hook 存实例级有序字典，多 hook、prepend/always_call 生效、可移除 … | 待领 | | |
| 5.07 | `jt.Function` 每次调用创建一次性上下文对象，实例无状态 | 待领 | | |
| 5.08 | `flag_scope` 的备份改局部栈，`__call__` 每次新建 scope | 待领 | | |
| 5.09 | 29 处融合 kernel 的启用条件由全局 `no_grad` 改为「输出不需要梯度」 | 待领 | | |
| 5.10 | 索引与计数统一 int64 | 待领 | | |
| 5.11 | `amp_reg` 位常量命名导出，一律 `\ | 待领 | | |
| 5.12 | matmul 四条路径共用能力表，dtype 用枚举不用子串 | 待领 | | |
| 5.13 | `unique` | 待领 | | |
| 5.14 | `Var.scatter` 改非就地 | 待领 | | |
| 5.15 | `.half()`/`.float16()` 删死的 amp 分支 | 待领 | | |
| 5.16 | `state_dict(to="torch")` 用 `from_numpy`，不强制 floa… | 待领 | | |
| 5.17 | 同一概念合并 | 待领 | | |
| 5.18 | 同一概念合并 | 进行中 | pyother | |
| 5.19 | 被静默忽略的参数改为传非默认值时 warn 或 raise | 进行中 | pyops（算子参数）+ pyother（其余） | 1710aef1（算子参数：relu/leaky_relu/silu/mish 的 inplace、instance_norm 与 InstanceNorm 的 running stats/momentum/is_train/sync、svd 的 compute_uv/driver、inv_ex 的 check_errors、ctc_loss 的 zero_infinity、sort 的 stable；topk 的 sorted 判为无需处理，见提交说明）。共用基础设施 `python/jittor/_arg_policy.py`，pyother 直接复用，不要另起近义模块 |
| 5.20 | import 期副作用删除 | 待领 | | |
| 5.21 | 六个 monkeypatch 安装器写成显式有序清单并加断言 | 待领 | | |
| 5.22 | `nn` facade 不导出 39 个下划线名，内部用模块局部名不经 `jt.nn.*` 晚绑… | 待领 | | |
| 5.23 | 根命名空间显式 `__all__` | 待领 | | |
| 5.24 | 10 个 `jt._*` 跨模块契约 | 待领 | | |
| 5.25 | `python/jittor/utils/` 拆散 | 待领 | | |
| 5.26 | 布局收尾 | 待领 | | |
| 6.C01 | `.item()` 对无符号 dtype | 待领 | | |
| 6.C02 | `PySlice_Unpack` 返回值检查，三个变量初始化 | 已合并 | bindings | 78d08344 |
| 6.C03 | 整数提升 | 并入 2.16 | | |
| 6.C04 | 含 `void` 语句被删 | 并入 3.09 | | |
| 6.C05 | 融合边号 ≥256 回绕 | 待领 | | |
| 6.C06 | `grad.cc:65-68` 判空对象改为 `dx` | 待领 | | |
| 6.C07 | 缺失梯度默认报错 | 待领 | | |
| 6.C08 | `grad.cc:146-261` 两趟遍历合一趟并快照结构，删无边界游标 | 待领 | | |
| 6.C09 | `backward()` 可重复 | 待领 | | |
| 6.C10 | CUDA 分配钩子两张 map 用 `find` 加显式错误，释放后 `erase` | 进行中 | mem | |
| 6.C11 | CPU 分配失败抛异常，返回值必须检查 | 进行中 | mem | |
| 6.C12 | `cuda_device_allocator.cc:32-37` 的 managed 回退放到 … | 进行中 | mem | |
| 6.C13 | 零字节分配不返回伪指针 `0x10` | 进行中 | mem | |
| 6.C14 | SFRL | 进行中 | mem | |
| 6.C15 | `migrate_to_cpu/gpu` 迁移前检查 share_with 关系，整组迁移或拒绝 | 待领 | | |
| 6.C16 | fetch 跨流 | 待领 | | |
| 6.C17 | `TempAllocator` 删遮蔽基类的 `used_memory`/`unused_mem… | 进行中 | mem | |
| 6.C18 | CachingBlock 保存底层 allocation 并原样回传，不再传 0 | 进行中 | mem | |
| 6.C19 | 每个分配器一把锁并覆盖 `gc()` | 进行中 | mem | |
| 6.C20 | swap | 进行中 | mem | |
| 6.C21 | 检查 `NODE_MEMCHECK` 外 `check_graph` 静默空转 | 待领 | | |
| 6.C22 | pyjt 关键字参数 | 已合并 | bindings | ed148a56 |
| 6.C23 | `is_type<NanoString>` 收窄 | 已合并 | bindings | f8f9de43 |
| 6.C24 | 带实例 `__dict__` 的类型加 `Py_TPFLAGS_HAVE_GC` 与 trave… | 已合并 | bindings | 4a30c5e4 |
| 6.C25 | 生成绑定补 `catch (...)` | 已合并 | bindings | b58ba756 |
| 6.C26 | `pyjt_compiler.py` 的 C++ 解析 | 已合并 | bindings | 4105d091 |
| 6.C27 | `Var.data` 返回的 numpy 视图 base 指向包裹该次 allocation 的… | 已合并 | bindings | 9504e520 |
| 6.C28 | 生成带「已构造」标志的 `tp_new` 或 `tp_dealloc` 先检查 | 已合并 | bindings | 8bd40d02 |
| 6.C29 | 标量转数组的全局 `tmp_data` 改自带 buffer | 已合并 | bindings | b57c31a1 |
| 6.C30 | `helper_cuda.h` 的 `peek` 去掉进程级闩 `peek_logged` | 待领 | | |
| 6.C31 | 失败的 import jittor 在退出期 abort，父进程无声消失 | 待领 | | |
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
| 6.P23 | eigh 的特征向量梯度在 CUDA 上错约 60% | 待领 | | |
| 6.P24 | Pool3d 的 count_include_pad 读原始参数 | 待领 | | |
| 6.P25 | Adan 偏差修正仍用全局 n_step；连带第一步 grad_diff 语义 | 进行中 | pyother | |
| 6.B01 | MPI 的 int64 改 `MPI_INT64_T` | 已合并 | dist | 03518707 |
| 6.B02 | ACL | 待领 | | |
| 6.B03 | HCCL 宏错误时抛而非 return | 已合并 | dist | c657ab01 |
| 6.B04 | 分布式一旦被请求，初始化失败硬失败 | 已合并 | dist | 8ae65e24 |
| 6.B05 | cuBLAS `use_tensorcore` 三目判断写反 | 待领 | | |
| 6.B06 | `var_broadcast` 用传入的 root | 已合并 | dist | 89dd014b |
| 6.B07 | cuDNN RNN | 待领 | | |
| 6.B08 | cuSPARSE | 待领 | | |
| 6.B09 | curand 奇数长度用临时 buffer 不越界写 | 待领 | | |
| 6.B10 | MPI fp16 归约统一标量参考实现加可选 SIMD 与运行期 CPUID 检测 | 已合并 | dist | 734d55a1 |
| 6.B11 | ACL 六个算子静默把输入升到 fp32 | 已合并 | dist | 492e5385 |
| 6.B12 | `cutt_transpose_op.cc:77` 的 `cudaGetLastError()`… | 待领 | | |
| 6.B13 | cuFFT `cufftCreate` 后被 `cufftPlanMany` 覆盖的句柄泄漏 | 待领 | | |
| 6.B14 | conv3d 三算子迁到 backend plan 缓存 | 待领 | | |
| 6.B15 | MPI 同时识别 PMI_/SLURM_ 环境变量或要求显式声明 | 已合并 | dist | 956c4b23 |
| 6.B16 | `sync_run` 在 ACL 上实现或删 flag | 待领 | | |
| 6.B17 | 析构不得抛 | 待领 | | |
| 7.01 | 「看起来支持其实空操作」一律改为实现或抛 `NotImplementedError`，需显式 `… | 已合并 | 兼容层分区 | ff395ecc b7c12ddc 0446217e 47012a27 46bc9ea7 49d41acf 9053a7c0 |
| 7.02 | DDP 真实梯度同步 | 待领 | | |
| 7.03 | 每个 torch API 一个模块级一等对象加保真度标注 | 待领 | | |
| 7.04 | 激活显式、一次性、可查询 | 待领 | | |
| 7.05 | install 事务化 | 待领 | | |
| 7.06 | 依赖单向化 core→tensor→nn/optim→distributed→fsdp→适配器 | 待领 | | |
| 7.07 | 第三方库补丁搬出 compat/ | 待领 | | |
| 7.08 | `torch.dtype` 改真正的对象而非 str 子类 | 待领 | | |
| 7.09 | `torch.library` | 进行中 | 兼容层分区 | |
| 7.10 | `torch.compile`/`jit.trace`/`jit.script` 保留 pass… | 进行中 | 兼容层分区 | |
| 7.11 | autograd 语义 | 进行中 | 兼容层分区 | |
| 7.12 | 独立 torch 包 | 待领 | | |
| 7.13 | FSDP2 | 待领 | | |
| 7.14 | vLLM 边界检查把 `torch` 视作 jittor 别名 | 进行中 | 兼容层分区 | |
| 7.15 | `_rebuild_tensor_v2` 按 stride 还原或报错 | 待领 | | |
| 7.16 | compat/ 内 129 个 `except: pass` 与 258 个宽泛 except … | 进行中 | 兼容层分区 | |
| 7.17 | `runtime.enable()` 只把 shim 的 site 目录加进 sys.path … | 进行中 | 兼容层分区 | |
| 7.18 | 布局收尾 | 待领 | | |
| 8.01 | 描述符与 workspace 一律 RAII | 待领 | | |
| 8.02 | 集合通信走通信流加事件依赖，支持 `GroupStart/End` 桶化 | 待领 | | |
| 8.03 | 精度策略收敛 | 待领 | | |
| 8.04 | cuDNN 9 | 待领 | | |
| 8.05 | MKL | 待领 | | |
| 8.06 | ACL 去样板 | 待领 | | |
| 8.07 | conv 族共享描述符与计划层 | 待领 | | |
| 8.08 | `ProcessGroup` 对象替代全局唯一 communicator | 待领 | | |
| 8.09 | NCCL | 待领 | | |
| 8.10 | `distributed/launch.py:102-107` 改 `wait(timeout)… | 待领 | | |
| 8.11 | `nccl_reduce`/`mpi_reduce` 非 root 不分配输出 | 待领 | | |
| 8.12 | 算子内不再复用全局 jit key 缓冲做缓存键 | 待领 | | |
| 8.13 | cuTT 计划未命中时的 `cudaDeviceSynchronize` 删除或降流同步 | 待领 | | |
| 8.14 | Corex | 待领 | | |
| 8.15 | 多机 rendezvous | 待领 | | |
| 8.16 | 多机启动器 | 待领 | | |
| 8.17 | 跨机网络与诊断 | 待领 | | |
| 8.18 | 多机 checkpoint | 待领 | | |
| 8.19 | 布局收尾 | 待领 | | |
| 9.01 | `import jittor` 不编译不下载 | 待领 | | |
| 9.02 | `install_cuda.py:113-122` 的 `os.execl` 自重启删除，用 d… | 已合并 | 构建 | 6b45c078 |
| 9.03 | 构建期失败一律抛带上下文的 `RuntimeError`，不用 LOGf/裸 assert | 已合并 | 构建 | 9197c8c6 |
| 9.04 | 依赖跟踪改用编译器的 `-MD -MF` | 部分合并 | 构建 | 65a2dc12（clean_cache 从一份布局定义生成；`-MD -MF`、hash64、主机名/`-march=native`/git 分支/路径哈希位数**未做**——每条都改变缓存路径或缓存键，该在一个提交里一起做，不要分四次各让所有人全量重编一次。审计里"删掉 helper_cuda.h 例外"那条已更正：裸删会让 CPU 构建整个失败，必须与 `-MD -MF` 同做） |
| 9.05 | 下载安全 | 已合并 | 构建 | e111ebcc |
| 9.06 | 删 cutlass 下载 | 已合并 | 构建 | 50673d69 |
| 9.07 | import 过程不反向写环境变量 | 待领 | | |
| 9.08 | 新架 GPU | 已合并 | 构建 | 2d71f792 |
| 9.09 | `cuda_wheel` 失败时 LOG.w 出原因，strict 为默认 | 已合并 | 构建 | c63dd809 |
| 9.10 | 2.0 版本策略 | 待领 | | |
| 9.11 | release 的 platform-validation 阶段跑 selftest | 已合并 | 构建 | 2af4658e |
| 9.12 | `extern/rocm/rocm_cache.tar.gz` 的预编译 .o 改从源码构建，或… | 待领 | | |
| 9.13 | README 加「首次运行会发生什么」 | 待领 | | |
| 9.14 | 一次性的构建前置条件检查 | 待领 | | |
| 9.15 | noxfile | 已合并 | 构建 | 84c7f766 |
| 9.16 | `agent/scripts/check_repo_layout.sh` 收缩为少数真会复发的检… | 待领 | | |
| 9.17 | 死代码 | 已合并 | 构建 | f99250bb |
| 9.18 | `disable_lock=1` 启用时明确告警并纳入缓存指纹 | 待领 | | |
| 9.19 | 布局收尾 | 待领 | | |
| 9.20 | asm_tuner 非原子写 .s，并发编译读到截断汇编 | 待领 | | |
| 9.21 | 拆掉手写预处理器最后一块：process() 双职责分离 + depfile | 待领 | | |
| 10.01 | `tools/run_test_suite.py` 拆成 `nox -s full` 周期性调度… | 待领 | | |
| 10.02 | 默认 `nox` 含 cpu 数值测试，或把默认改名为 static | 待领 | | |
| 10.03 | optional/rocm/mpi/nccl 四个 session 排上 runner 或在文档… | 待领 | | |
| 10.04 | 假绿清理 | 待领 | | |
| 10.05 | 按 skip 原因分桶统计并在 CI summary 输出，对「本环境应能跑却 skip」设阈值 | 待领 | | |
| 10.06 | `expect_error` 带 `exc_type` 与 `match` | 待领 | | |
| 10.07 | Unary/Binary/Reduction 用 `OpDTypes.supported` | 待领 | | |
| 10.08 | 已复现缺陷用 `xfail` 而非 `skip` | 待领 | | |
| 10.09 | 公开 API 与 OpInfo 差集作为 structure 门禁一项 | 待领 | | |
| 10.10 | gradcheck 加「故意写错导数应当失败」的负向自测 | 待领 | | |
| 10.11 | 设备对拍加 dtype 轴 | 待领 | | |
| 10.12 | `retry` 装饰器记录并上报重试次数 | 待领 | | |
| 10.13 | marker 真正建立 `-m "not slow"` 快门禁或删除 | 待领 | | |
| 10.14 | notebook 门禁按 topic 参数化 | 待领 | | |
| 10.15 | 速度 harness 记录并断言两侧线程数、亲和掩码与精度策略 | 待领 | | |
| 10.16 | 提供计时 API | 待领 | | |
| 10.17 | 异步错误 | 待领 | | |
| 10.18 | 结构测试预算转向核心 | 待领 | | |
| 10.19 | 每个带 `grad()` 的后端算子有对 CPU 参考的梯度单测 | 待领 | | |
| 10.20 | 给测试提供受支持的内省 API，替代 283 处 `jt.flags.*`、137 处 `com… | 待领 | | |
| 10.21 | import 方向做成 lint 规则 | 待领 | | |
| 10.22 | 多机门禁 | 待领 | | |
| 10.23 | 布局收尾 | 待领 | | |
| 11.01 | 删已被取代的绕过与死路径 | 待领 | | |
| 11.02 | 已提前为 0.20 | 并入 0.20 | | |
| 11.03 | 单文件异常拆分 | 待领 | | |
| 11.04 | 关键接口写成显式契约 | 待领 | | |
