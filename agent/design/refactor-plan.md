# Jittor 2.0 整改计划

2026-09-02。把 [代码库设计审计](codebase-audit/README.md) 七份分报告、[系统设计审计](system-design-audit.md)
与两份待决策方案里的每一条发现，转成可以分派给独立 agent 的任务，并排出先后顺序。
一行一个任务；每行给出改什么、改到什么程度、依赖谁、出处在哪、怎样算完成。

## 0. 怎么用这份文档

**出处缩写**（每行的「出处」列指向这些文档的小节；小节名就是那份文档里的二级标题）：

| 缩写 | 文档 | 缩写 | 文档 |
| --- | --- | --- | --- |
| 核心 | [codebase-audit/01-core-runtime.md](codebase-audit/01-core-runtime.md) | 测试 | [codebase-audit/05-tests.md](codebase-audit/05-tests.md) |
| Python | [codebase-audit/02-python-api.md](codebase-audit/02-python-api.md) | 后端 | [codebase-audit/06-backends.md](codebase-audit/06-backends.md) |
| 兼容 | [codebase-audit/03-compat-shim.md](codebase-audit/03-compat-shim.md) | 架构 | [codebase-audit/07-architecture.md](codebase-audit/07-architecture.md) |
| 构建 | [codebase-audit/04-build-tooling.md](codebase-audit/04-build-tooling.md) | 系统 | [system-design-audit.md](system-design-audit.md)（条目号 A1…F5） |
| 多后端 | [multi-backend-design.md](multi-backend-design.md) | 多卡 | [device-placement.md](device-placement.md) |
| 流水 | [pipelined-execution.md](pipelined-execution.md) | 卷积 | [cudnn-convolution-plans.md](cudnn-convolution-plans.md) |
| 布局 | [target-layout.md](target-layout.md)（目标目录树与每处搬动的理由） | | |

**每个任务的完成定义**（表格里的「验收」列只写该任务特有的部分，下面这些对所有任务生效）：

1. 凡是修「静默算错」的任务，必须先写一个**修前失败、修后通过**的测试，测试与修复同一提交。
2. 三套门禁全绿：原生（CPU）、CPU torch 模式、CUDA。改到哪一层跑哪一层是不够的——Python 层
   改动照样要过 CUDA 门禁，因为并行路径从不交叉验证正是审计的核心发现。
3. 新增测试必须位于门禁能触达的路径上（阶段 0 完成前请在提交说明里写明它属于哪个 session）。
4. 涉及的审计条目在对应报告里追加一行「已修：提交号」，不要删原条目。
5. 每个任务一个提交，提交后立刻推送；只 `git add` 自己改的文件，不用 `git add -A`。
6. **可复用的东西沉淀成 skill。** 做任务过程中凡是造出了下一个人还会用到的方法——一个复现脚本、
   一条对拍口径、一种定位手法、一组必须一起设的环境变量、一个「这样测才算数」的判据——就在
   `agent/skills/<kebab-case-名字>/SKILL.md` 里写成 skill，与任务同一提交。判据是**这件事第二个人
   做会不会重新踩一遍**：会，就写。写的是可执行的方法（命令、脚本、判据），不是叙述你干了什么；
   叙述归提交说明与看板。已有四个 skill 可作格式参照，frontmatter 只需 `name` 与 `description`。

**执行 agent 的硬性约束**（来自本轮踩过的坑，违反任何一条都会制造看起来像回归的假失败）：

- 实验与运行状态放在 `$JITTOR_LAB_ROOT` 之下，每个进程独立 `JITTOR_HOME` 与 `TMPDIR`；共享缓存
  的并发运行会互相损坏。
- 门禁运行期间不得修改 `python/jittor/src`：任何新起的进程都会在原目录重编核心。
- 不得 `kill -9` 门禁；必须杀时先删掉它的 JIT 缓存再重跑，否则下一轮会在无关算子上大面积报梯度不符。
- 门禁顺序用一个脚本串起来并等**哨兵文件**，不要用 `pgrep -f` 守卫（它会匹配到等待的 shell 自身）。
- pytest 忽略 `PYTHONPATH`（`pyproject.toml` 的 `pythonpath`）；测源码副本要传 `-o pythonpath=<副本>/python`。
- 不要往文档或规则里写个人目录、主机名、卡号。
- 与 PyTorch 对拍时 Jittor 一侧跑在开发环境，PyTorch 一侧跑在装有**真** PyTorch 的解释器（开发环境
  里的 `torch` 是 shim）。

### 执行方式：新分支、多 agent 并行、随时推送

**分支。** 全部工作在 `2.0-refactor` 上做（从 `2.0` 切出，已推送）。不开 PR、不等评审：一个任务一个提交，
提交即推送。推送前 `git pull --rebase origin 2.0-refactor`；禁止 force-push；禁止 `git add -A`；提交说明
以任务编号开头（`[2.04] Var::allocator 去类型双关`）。`2.0` 只在每个阶段收尾时由维护者合并，子 agent 不碰它。

**并行。** 编排 agent 按第 15 节的分区开子 agent：同一分区同一时刻只有一个 agent，分区内按「前置」列
顺序做；不同分区全部并行。并发上限是机器上的 GPU 数——每个子 agent 独占一张卡（`CUDA_VISIBLE_DEVICES`）
与一段核（`taskset`），独立的 `JITTOR_HOME`、`TMPDIR`（在 `$JITTOR_LAB_ROOT` 下按 agent 名分目录）和独立的
`git worktree`。不要等一个子 agent 的门禁跑完才开下一个不相关的任务。

**波次。** 波次只说明此刻哪些任务可领，子 agent 做完一个就领下一个，不等波次结束。

| 波 | 可领的任务 | 为什么现在能做 |
| --- | --- | --- |
| 一（立即） | 阶段 0 全部；阶段 6 全部（6.C、6.P、6.B）；7.01；9.03、9.05、9.06、9.09、9.17 | 阶段 6 的 69 条互不依赖、各碰一两个文件，是并行度最高的一批；7.01 与 9 的这几条不碰热点文件 |
| 二 | 阶段 1（一个 agent 独占 codegen 分区）；7.03–7.10；9 的其余；10 里不依赖架构的条目 | 阶段 0 落地后新测试能进门禁 |
| 三 | 阶段 2 → 3（核心与执行器分区内串行）、阶段 4（后端分区）、阶段 5（Python 分区）三路并行 | `data.gz` 已还原 |
| 四 | 7.11–7.13、阶段 8、10 的其余 | 架构任务提供了它们的前置 |
| 五 | 各阶段的布局收尾行、阶段 11 | 代码已经是一种形状 |

**领任务。** [`refactor-board.md`](refactor-board.md) 一行一个任务：`编号 | 任务 | 状态 | 负责 | 提交`。
领任务就是把状态改成「进行中」、写上自己的名字，提交并推送；推送冲突说明别人先领了，rebase 后换一个。
完成就是把状态改成「已合并」并填提交号。热点文件的当前占有者也记在板上。「并入 X」的行不领。

**门禁。** 子 agent 在自己的 worktree 里至少跑：本任务新增的测试、受影响目录的测试、smoke（0.15 落地后）。
另开一个**门禁 agent** 常驻：循环拉取 `2.0-refactor`，跑三套完整门禁，把提交号与结果写到板的顶部；
红了就 `git bisect` 定位到任务并把责任写到那一行，由领该任务的 agent 修。它定位期间其他 agent 照常推送
——不因为门禁在跑就停下，这是速度的来源，代价由门禁 agent 的 bisect 承担。

**沉淀。** 一个 agent 查清的东西，下一个 agent 不该重查。凡是可复用的调试方法、复现脚本、对拍口径、
环境前置、判据，都写进 `agent/skills/`（见第 0 节完成定义第 6 条），并在简报里点名你新增或更新了哪些。
编排 agent 每波结束时检查一次：这一波有没有出现三个以上 agent 各自重新查明的同一件事——有就说明
漏了一个 skill，补上。

**子 agent 的提示词必须包含：** 任务行原文、出处小节全文、本节与第 0 节的完成定义和硬性约束、它所在
分区的热点文件、分支与推送规则、板文件路径。**编排 agent 不做的事：** 不自己改代码，不等，不合并 `2.0`。

## 1. 排序原则

1. **先让改动可验证，再改任何东西。** 227 个算子的反向门禁当前实例化为零个用例，289 个测试文件里
   215 个不在任何门禁路径上，缓存目录不含构建配置指纹导致并发进程互相重编。在这之前落地的重构
   无法被证明没有破坏东西。阶段 0 因此排在最前，而且它的每一项都便宜。
2. **先还原看不见的核心。** `data.gz` 里的五个翻译单元（liveness、融合划分、三个 pass）是多设备、
   执行计划缓存、归约优化都要改的地方。不还原就是盲改。阶段 1 只有这一件事。
3. **架构改动前置，点修复落在新结构上。** 阶段 2–5 的架构任务会顺带消灭一批点问题（下表逐条标出
   「被 X 取代」），先修点问题会被重做一遍。但架构任务之间有严格依赖：0 维形状先于视图模型，
   `Var::allocator` 去双关先于设备字段，分配器 id 空间按实例先于多卡合并，Node 观察者接口先于
   拆 Executor。表里的「前置」列就是这些依赖。
4. **静默算错的单点修复与架构并行。** 它们各自只碰一两个文件，按「热点文件分区」（第 14 节）
   分配给不同 agent 即可同时推进；唯一规则是不碰阶段 2–3 正在重写的那几个文件。
5. **兼容层分两步。**「空操作改报错」不依赖任何架构，可以马上做，它把整类风险从「不报错结果错」
   变成「明确报错」；「独立 torch 包」要等核心提供存储视图、按值指定的上游梯度、反向可达叶子查询。
6. **分布式改造等设备与流模型。** 通信流、process group、事件依赖都建立在阶段 4 的设备字段
   和流模型之上，先做只会再返工一次。
7. **删除类任务随时可做**，但先删会让别的 agent 的分支冲突，所以集中在阶段 11 并留到相关重构合并之后。
8. **目录布局是每个阶段的收尾，不是开头。** 终点在 [目标目录布局](target-layout.md)；每个阶段表格的最后一行
   「布局收尾」只做 `git mv`、转发模块与结构测试同步，不夹带行为改动。把四份 CUDA kernel 挪进一个目录
   改不了「有四份」，所以搬动必须等该阶段把代码改成一种形状之后。

## 2. 已完成，勿重做

| 事项 | 提交 | 对应条目 |
| --- | --- | --- |
| 惰性图按段提前发射（`auto_flush_ops`，仅 CUDA），transformer 类用例追平 PyTorch | `780c1989`、`e2d98f9c` | 系统 A1（缓解）；流水 |
| `Tapes` 断言输入必须仍挂起；flush 排除 tape 输出 | `780c1989` | 系统 A3 |
| 二维卷积三算子改走 cuDNN backend API 与执行计划缓存；`grad()` 按布局串读维度；`jit_run` 权重维度按 KCRS | `cd66d6a3` | 系统 B1（2D）、B3；卷积 |
| `_is_scalar` 移到位 26，`grad.cc:51` 镜像加掩码；`tests/core/test_scalar_flag.py` | `048bee3f` | 核心§节点模型 前两行 |
| 反向图内提前发射：实测三个用例全部变慢，**不采用** | 无 | 系统 A2 |
| 参数注册规则改为「只在有正面证据时降级」（`_jt_plain_tensor`） | 见路线图 2026-08-30 | 兼容§张量语义（部分） |
| `WarpReducePass`：CUDA 归约 warp 内先规约，UNet 1.10×→1.06× | 本轮提交 | 核心§补充 CUDA 归约（部分，块级树形归约仍待做，见 3.22） |
| 多卡：两个完整实现在分支 `device-select` 与 `multi-device`，未合并 | 分支 | 多卡§5；系统 C1 |

## 3. 阶段 0 · 让改动可验证

目标：门禁真的跑到该跑的东西，缓存不再被并发进程互相损坏。全部完成前不要开始阶段 2 之后的任务。

| 编号 | 任务 | 前置 | 出处 | 验收 |
| --- | --- | --- | --- | --- |
| 0.01 | `TestGradients` 改用 `only_for=("cpu",)` 显式实例化（`tests/_helpers/device_types.py:57-65`、`test_ops.py:82,111`），并把 `JITTOR_TEST_DEVICES=cpu` 下的 `test_ops.py` 加进 CPU 门禁 | — | [测试](codebase-audit/05-tests.md)§算子证据链 | CPU 门禁日志里 gradcheck 用例数 > 200 |
| 0.02 | 设备过滤后 bases 为空或方法数为 0 时生成器直接 raise（`device_types.py:198-232`）；`get_all_device_types` 与 noxfile 的设备名枚举合一，`rocm`/`mpi` 不再返回空列表 | — | [测试](codebase-audit/05-tests.md)§算子证据链 | 故意配错设备名的用例报错而非 0 项通过 |
| 0.03 | `tests/compiler/test_jit_tests.py` 进 CPU 门禁，并断言 `len(names) > 0`（`src/test/` 9 文件 1033 行的 C++ 单测） | — | [测试](codebase-audit/05-tests.md)§完全没有测试保护的关键契约 | CPU 门禁跑到 expr/kernel_ir/sfrl 等 9 组 |
| 0.04 | 门禁改为「整个 `tests/` 减显式排除清单」，排除项必须写理由（`noxfile.py:152-289` 白名单废除）；`tests/backends/cpu`、`tests/data`、`tests/nn` 的 19 个文件、`test_cold_start_runtime.py` 随之进门禁 | 0.02 | [测试](codebase-audit/05-tests.md)§门禁范围；[构建](codebase-audit/04-build-tooling.md)§门禁与开发工具链 | CI 可达文件数从 74 → ≥ 280；排除清单每行有理由 |
| 0.05 | 生态对拍进 nightly：建一条装真 PyTorch 的 session，设 `REAL_TORCH_PYTHON`，`SPEED_RATIO=1.07`，`require_real_torch` 时断言 `skipped == 0` | 0.04 | [测试](codebase-audit/05-tests.md)§跳过条件与假绿、§对拍口径 | nightly 报告七个框架的对拍与速度比 |
| 0.06 | `make_tensor` 种子改为 `hash(nodeid, shape, dtype)` 的确定函数，失败信息打印种子（`_helpers/common.py:146,184`） | — | [测试](codebase-audit/05-tests.md)§测试自身的可靠性 | 同一用例 `-k` 单跑与全量跑输入一致 |
| 0.07 | 缓存路径追加构建配置指纹（nvcc_flags、cc_flags、cuda_archs、enable_lto、kernel_flags、shim 数学开关）；源码改动写新目录再切指针，不原地重建（`jittor_utils/__init__.py:480-525`） | — | [构建](codebase-audit/04-build-tooling.md)§缓存键与缓存布局；系统 D1 | shim 开与关不再共用 `jit/`；门禁运行中改源码不影响在跑进程 |
| 0.08 | 锁统一为一种类型、一个 fd（`lock.py:30` flock 对 `src/lock.cc:47` record lock）；加超时与持有者诊断；`query_cuda_cc` 子进程不再取同一把锁 | — | [构建](codebase-audit/04-build-tooling.md)§锁与并发；系统 D2 | 两个进程同时编译不再互相进入；孤儿锁 30 s 内报出持有者 pid |
| 0.09 | 探测结果落盘 `cache_path/probe.json`（g++/nvcc/python3-config/nvidia-smi/gdb/mpicc/hipcc/cuda cc），按工具链 mtime 失效 | 0.08 | [构建](codebase-audit/04-build-tooling.md)§首次导入做了多少事 | 热缓存下 `import jittor` 不再 spawn 探测子进程 |
| 0.10 | 写缓存前检查可用磁盘空间，不足时给明确错误 | — | 系统 D3 | 磁盘满时报 `No space` 而非散布段错误 |
| 0.11 | 「jit_utils 已更新请重跑」改非零退出码（`compiler.py:926-928`） | — | [构建](codebase-audit/04-build-tooling.md)§首次导入做了多少事 | CI 能看到失败 |
| 0.12 | 14 处在用例里裸赋值 `jt.flags.*` 且无 tearDown 的测试改 `flag_scope`；加静态检查禁止测试里裸赋值 flag | — | [测试](codebase-audit/05-tests.md)§测试自身的可靠性 | 检查脚本在门禁里跑 |
| 0.13 | conftest 的模式由显式环境变量决定，删除 `sys.argv` 嗅探（`tests/conftest.py:24-60,174`） | — | [测试](codebase-audit/05-tests.md)§测试自身的可靠性 | `-k`、xdist、IDE runner 下行为一致；native 会话的 `--collect-only` 能检查 Torch-mode 文件可导入 |
| 0.14 | `_session_env` 不再 `os.environ.copy()`；显式设定并断言线程数与亲和掩码（`noxfile.py:363`） | — | [测试](codebase-audit/05-tests.md)§测试自身的可靠性；系统 F2 | 宿主 `OMP_PROC_BIND` 不再泄漏进门禁 |
| 0.15 | 门禁分两层：smoke（< 5 分钟，进 PR）与 full（nightly，含并行编译器开启的一轮与 0.05）；同模式目标合并成一次 pytest 调用加 xdist；去掉 `stop_on_first_error` | 0.04、3.06 | [测试](codebase-audit/05-tests.md)§耗时分布；[构建](codebase-audit/04-build-tooling.md)§门禁与开发工具链；系统 F1 | PR 门禁 < 5 分钟；一轮能报出全部失败 |
| 0.16 | `test_device_parity.py` 按算子分片并行，不再在 `setUpClass` 关并行编译（`:172`）；目标 CUDA 门禁 < 40 分钟 | 3.06 | [测试](codebase-audit/05-tests.md)§耗时分布 | CUDA 门禁总时长 |
| 0.17 | `pyproject.toml` 的 `pythonpath` 改由 conftest 按环境变量决定，副本与工作树可用 pytest 验证 | — | 系统 F3；[构建](codebase-audit/04-build-tooling.md)§门禁 | `-o pythonpath` 不再必需 |
| 0.18 | 门禁每条目断言至少执行 1 个非 skip 用例；`tests/nn/test_attention.py` 与 `test_opt_state_dict.py` 两个恒绿条目处理 | 0.04 | [测试](codebase-audit/05-tests.md)§跳过条件与假绿 | 门禁 summary 含每条目的 passed/skipped |
| 0.19 | 结构测试从「精确清单」改成「规则」：`test_cleanup_structure.py` 的精确条目集合、`test_nn_structure.py:48-52,170-274` 的逐文件 import 与导出名清单、行数预算（`:1812-1821`）、`test_vllm_compat_structure.py:57-68` 的文件名与 300 行断言全部删除，只保留边界规则（import 方向、公开 API 快照、打包内容、循环依赖）；迁移守卫设过期时间。这是后面每一次目录搬动的前置 | — | [测试](codebase-audit/05-tests.md)§tests/structure 的成本；[架构](codebase-audit/07-architecture.md)§模块边界；系统 E4 | `tests/structure` < 2000 行；挪一个文件不需要改结构测试 |
| 0.20 | 布局收尾：`agent/` 与 `docs/` 定权威——`agent/design` 并入 `docs/architecture`，`agent/results` 并入 `docs/results`，`agent/` 只留 manuals/skills/scripts；删 `tools/services/legacy`、`tests/system/`；根目录 AWESOME 列表与 asv 配置归 `docs/`、`benchmarks/` | 0.19 | [布局](target-layout.md)§3 | 仓库只有一棵文档树 |
| 0.21 | 测试里起的子进程不带 `PYTHONPATH`，在 worktree/副本里测的是主树——**门禁机器上就是假绿**。已知一处：`tests/data/test_dataset.py::test_dataset_shuffle_mpi` 用 `mpirun -np 2 <裸 python> <script>`（同文件另两处已在 6.P15 改走 `run_child_script()`，这处因要拼 mpirun 命令行未改）。全树扫一遍 `subprocess`/`mpirun`/`os.system` 起的 python，统一走一个带 `PYTHONPATH` 的 helper，并加静态检查禁止裸起 | 0.13 | 2026-09-02 由 6.P15 的执行者发现；机理见 `agent/skills/jittor-worktree-verification` | 扫描脚本在门禁里跑；mpi 用例在有 mpi 的机器上真跑到本树代码 |

## 4. 阶段 1 · 还原不可见的核心

| 编号 | 任务 | 前置 | 出处 | 验收 |
| --- | --- | --- | --- | --- |
| 1.01 | 把 `utils/data.gz` 解出的 `data.cc` 还原为可读的五个翻译单元：`fuser.cc`、`node.cc`、`opt/pass/atomic_tuner_pass.cc`、`shared_reduce_pass.cc`、`parallel_pass.cc`（以头文件声明为对照反混淆 `x10364` 式标识符、十六进制串与改名宏，去掉 `_P(...)` 噪声）；删除 `compiler.py:1402-1429` 的解压编译路径、`use_data_gz` flag、`src/utils/vdp`、指向不存在的 `__data__` 回退分支 | — | [核心](codebase-audit/01-core-runtime.md)§可审计性；[架构](codebase-audit/07-architecture.md)§核心抽象 | 仓库无 `data.gz`；`data.o` 导出的 31 个符号全部来自源码；三套门禁不变 |
| 1.02 | `op_compiler.cc:30-69` 用正则给 `ParallelPass` 输出打补丁的 `fix_parallel_thread_ranges` 删除，累积逻辑在 pass 的 IR 层修正；`tests/compiler/test_parallel_pass.py:124-129` 改断言 IR 结果 | 1.01 | [核心](codebase-audit/01-core-runtime.md)§补充：代码生成与优化 pass | `:1156` 的子串嗅探消失 |
| 1.03 | 查明 `SharedReducePass` 在约 4900 个归约 kernel 里零命中的触发条件；修成可用的块内共享内存树形归约或删除（与 `WarpReducePass` 协调：warp 内归约后再做块内归约） | 1.01 | [核心](codebase-audit/01-core-runtime.md)§补充：CUDA 归约 | UNet 那个 `REDUCE_c` kernel 每输出只写一次；生成代码含块内归约 |
| 1.04 | `ReduceTuner::run` 不再对 CUDA 直接返回（`reduce_tuner.cc:14`）：给 CUDA 归约一套切分与顺序候选，或在注释里写明为何不需要并加测试 | 1.01、1.03 | [核心](codebase-audit/01-core-runtime.md)§补充：CUDA 归约 | 空间维归约与全归约各有一个走了非默认候选的用例 |
| 1.05 | 布局收尾：还原的 `fuser.cc`、`node.cc` 与 `src/` 根下其余 39 个文件进 `src/core/`；三个还原的 pass 进 `opt/pass/`；`src/test` 改 `src/tests`；`use_data_gz` 相关的 `utils/data.gz`、`vdp` 删除 | 1.01、0.19 | [布局](target-layout.md)§3 | `src/` 根下无 .cc/.h |

## 5. 阶段 2 · 核心数据结构（架构）

热点文件：`node.h`、`var.h/.cc`、`op.h/.cc`、`var_holder.h/.cc`、`grad.cc`、`graph.h/.cc`、`nano_string.*`、`utils/log.*`。同一时间只由一个 agent 改这些文件。

| 编号 | 任务 | 前置 | 出处 | 验收 |
| --- | --- | --- | --- | --- |
| 2.01 | Var 与 Op 各持自己的 flag 类型；位号由枚举连续生成并 `static_assert` 不重叠；`node.h` 里「6..22 共用、16..21 镜像 amp_reg」的注释契约变成类型 | 1.01 | [核心](codebase-audit/01-core-runtime.md)§节点模型；[架构](codebase-audit/07-architecture.md)§核心抽象 | 泛型读 Node flag 的代码不再编译通过 |
| 2.02 | 删除 `Node::custom_data`：`executor.cc:269,273,537`、`fused_op.cc:81-84`、`graph.h:115,143`、`grad.cc:120-129`、`memory_profiler.cc:88-94` 各用按 node id 索引的局部 vector | 1.01 | [核心](codebase-audit/01-core-runtime.md)§节点模型 | `custom_data` 字段不存在；两个遍历交错的测试通过 |
| 2.24 | **`custom_data` 的最后一个用户：FusedOp 的跨阶段 var 索引**。2.02 只清掉了五处遍历标记（审计描述的「任意两个遍历交错就互相破坏」由此消除），`fused_op.cc` 那处保留——它是 `(vid<<2) \| 不可融合位 \| 已访问位` 的持久映射，被 `do_jit_prepare`、`get_loop_options` 与 `context->vrm` 跨 pass 管线与 JIT 编译读取。彻底删除需要给 FusedOp 一份显式的 `var→index` 映射并改三处读取点，**其中 `vrm` 与生成代码的结构体偏移相关，与 3.11 在同一片代码上**——两边同时改会冲突，所以排在 3.11 之后 | 2.02、3.11 | [核心](codebase-audit/01-core-runtime.md)§节点模型（2026-09-03 由 2.02 的执行者查明第六个用法） | `Node::custom_data` 字段不再存在；融合与 relay 的行为不变（生成代码逐字比对） |
| 2.03 | `tflag` 全局计数器加魔数改为 epoch 对象或局部集合（`node.h:88`、`grad.cc:105,121,161`、`graph.cc:77`、`op.cc:135` 重入） | 2.02 | [核心](codebase-audit/01-core-runtime.md)§节点模型 | 嵌套 `run_sync` 不再使外层遍历失效 |
| 2.04 | `Var::allocator` 去类型双关：新增 `share_src` 与 `share_offset` 字段（`var.h:48`、`var.cc:118-124`、12 处调用）；`Var::allocation` 给初值 | — | [核心](codebase-audit/01-core-runtime.md)§节点模型、§补充：内存与分配器；[架构](codebase-audit/07-architecture.md)§核心抽象 | `share_with` 与 alloc 之间读 `allocator->is_cuda()` 不再是 UB |
| 2.05 | 真正的 0 维张量：`var.cc:107` 不再把空 shape 补成 `(1,)`；`array_op.cc:56-59` 的 `_is_scalar` 按 0 维判定；清掉 `var_holder.h:317-319`、`var_holder.cc:260-262` 两处注释掉的疤 | 2.01 | [核心](codebase-audit/01-core-runtime.md)§节点模型；[兼容](codebase-audit/03-compat-shim.md)§张量语义 | `np.array([2.0],f64) * f32` 提升到 f64；`x[0]` 是 0 维 |
| 2.06 | 边表由 list 加反向迭代器改 SmallVector，按下标 O(1)（`node.h:150-172`、`setitem_op.cc:61,170,179,204`） | 2.02 | [核心](codebase-audit/01-core-runtime.md)§节点模型 | UNet 一步执行器分配次数下降；门禁不变 |
| 2.07 | `hold_vars`/`sync_ptr` 析构里 `std::next(end())` 的 UB（`var_holder.cc:30-31,115-142`）改序号或哨兵 | — | [核心](codebase-audit/01-core-runtime.md)§节点模型 | `top_weak_sync` 有用例证明仍工作 |
| 2.08 | `Node` 不再 include `pybind/py_var_tracer.h`：追踪改为节点生命周期观察者接口，由 pyjt 层注册（`node.h:10,169`） | — | [架构](codebase-audit/07-architecture.md)§核心抽象、§分层与依赖方向 | 图数据结构可脱离 CPython 单测 |
| 2.09 | `th_mode` 从 C++ 核心上移为 autograd 策略对象（`var.cc:26`、`op.cc:142`、`node.h:47`、`mem_info.cc:73`，共 16 处） | 2.01 | [架构](codebase-audit/07-architecture.md)§分层与依赖方向 | 核心 grep `th_mode` 为 0 |
| 2.10 | 三套 liveness 计数（`node.h:120-133` 的 f/b/p 八条规则、8 个手工配对 own/release）封成带不变量断言的类型，或改为显式根集合加可达性 | 1.01、2.02 | [架构](codebase-audit/07-architecture.md)§核心抽象；[核心](codebase-audit/01-core-runtime.md)§可审计性 | `need_free()` 不再是手写布尔式；`check_graph` 在 release 构建也校验（含 6.C25） |
| 2.11 | `VarHolder` 不再是执行触发点：`add_hold_vars` 里的 auto_flush 与急切 sync 改为执行器的显式 `submit_pending()` 接口，由 Python 侧调度调用；删除 `flush_suspended` 补丁状态 | 2.10 | [架构](codebase-audit/07-architecture.md)§核心抽象；系统 A1（长期） | 构造 VarHolder 无执行副作用；错误不再需要从构造函数里躲开 |
| 2.12 | 打破 `Executor ⇄ VarHolder` include 环（`var_holder.h:11`、`executor.cc:24` 及 13 个核心 .cc） | 2.11 | [架构](codebase-audit/07-architecture.md)§分层与依赖方向 | 依赖图无环 |
| 2.13 | 执行相关全局状态（`hold_vars`、`exe`、`sync_ptr`、`tflag_count`、`use_cuda`、`device_id`、`sync_run`）收进一个 Runtime/Context 对象；78 个 flag 分成 `jt.config`（启动期只读快照）与 `jt.runtime`（运行期开关） | 2.03、2.11 | 系统 C2；[架构](codebase-audit/07-architecture.md)§公共 API；[核心](codebase-audit/01-core-runtime.md)§模块边界 | 用户不能在运行中改 `cc_flags`；`misc/cuda_flags.cc` 消失 |
| 2.14 | `src/misc/` 拆散：vendored miniz 移到 third_party；`nano_string`/`nano_vector` 并入 `src/type/`；NaN 检查归 debug | 2.13 | [架构](codebase-audit/07-architecture.md)§模块边界 | `misc/` 不再含核心状态与类型系统 |
| 2.15 | NanoString：索引位宽 7→8 并加注册期断言（`nano_string.h:107`、`nano_string.cc:194`）；名字写入 16 字节槽位加 ASSERT（`:221-227`）；dtype 表改运行期注册，后端可加 fp8/int4 | 2.14 | [核心](codebase-audit/01-core-runtime.md)§类型系统 | 注册第 129 个类型报错而非别名；注册 ≥16 字符名报错 |
| 2.16 | 类型提升表：区分 kind 与 width 的提升格替代「取最大字节数加与运算」（`nano_string.h:251-279,217-227`）；混合符号单独提一档，达上限退 float64；Python 层删除局部 cast 打补丁（`rad2deg`/`deg2rad` 等） | 2.15 | [核心](codebase-audit/01-core-runtime.md)§补充：绑定层（整数提升）；[Python](codebase-audit/02-python-api.md)§张量语义 | `uint8(200)+int8(1)` 得 int16 的 201；`uint8*(1/255.)` 得 float32；`int64*2.0` 与 torch 一致 |
| 2.17 | 算子身份用注册期整型 id：替换 25 处 `name()` 字符串比较（`nan_checker.cc:57-58`、`var_holder.cc:49,79`、`op_compiler.cc:146,148,943,1116`、`loop_var_analyze_pass.cc` 5 处、`conv_tuner.cc` 5 处）；删 `fast_strcmp` 越界读（`var_holder.cc:402-414`）；「必须保持挂起」做成节点标志 | 2.01 | [核心](codebase-audit/01-core-runtime.md)§JIT 键与代码生成；系统 A3 | 核心 grep `name() ==` 为 0 |
| 2.18 | 算子注册表惰性初始化；键对称（`op_register.cc:34` 存全名、`:15,38,43` 查截断名）；不在 `void*` 上用 RTTI 分派；73 处静态初始化期的 `get_op_info` 延迟到首次使用 | 2.17 | [核心](codebase-audit/01-core-runtime.md)§错误处理与失败模式 | 注册带点的名字能查到；静态初始化顺序不再是契约 |
| 2.19 | 错误分两档：用户错误（可捕获异常）与内部不变量（断言）；析构不得抛；生成的 `tp_dealloc` 异常时仍 `tp_free`；486 处 ASSERT/CHECK 与 62 处 LOGf 按此归类 | — | [核心](codebase-audit/01-core-runtime.md)§错误处理与失败模式 | 析构与信号处理器里 grep `LOGf` 为 0 |
| 2.20 | 信号处理器只做 `write` 与 `_exit`，符号化交给预建 helper 进程（`utils/log.cc:250-322`）；标志改 `volatile sig_atomic_t`；jit key 溢出改正常异常（与 3.02 配合） | 2.19 | [核心](codebase-audit/01-core-runtime.md)§错误处理、§补充：绑定层与失败模式 | 在 malloc 内崩溃能给出报告而非挂死 |
| 2.21 | `DEFINE_FLAG_WITH_SETTER` 先赋值再调 setter，签名收新旧两值（`log.h:228-242`）；删 `tracer.cc:137-139` 的手工回写；环境变量解析用 `from_chars` 加全串消费，失败 fail fast（`log.h:180-196`、`log.cc:173`） | 2.13 | [核心](codebase-audit/01-core-runtime.md)§补充：绑定层与失败模式 | `export log_v="1 "` 报错而非静默默认 |
| 2.22 | 环境变量统一 `JT_` 前缀（76 个小写同名变量保留并告警），构建期变量与运行期 flag 分两个命名空间，同名不再一个追加一个替换（`compiler.py:1116-1117` 对 `log.h:226`）；启动时把非默认 flag 打成一行摘要；生成自动导出的变量清单 | 2.13 | [构建](codebase-audit/04-build-tooling.md)§环境变量作为配置 | shell 里导出的 `name`/`debug` 不再改变框架行为 |
| 2.23 | 布局收尾：`src/misc/nano_*` 并入 `src/type/`，miniz 进 `src/third_party/`，`init.cc`/`cuda_flags.cc`/`profiler/`/`lock.cc` 进 `src/runtime/`，`pyjt/`+`pybind/` 进 `src/bindings/`；`src/misc/` 目录消失 | 2.13、2.14、2.15、0.19 | [布局](target-layout.md)§3 | `src/misc/` 不存在 |

## 6. 阶段 3 · 执行器与编译管线（架构）

热点文件：`executor.cc`、`fused_op.cc`、`op_compiler.cc`、`parallel_compiler.cc`、`jit_key.*`、`opt/kernel_ir.*`、`opt/pass_manager.*`、`opt/pass/*`、`opt/tuner/*`。

| 编号 | 任务 | 前置 | 出处 | 验收 |
| --- | --- | --- | --- | --- |
| 3.01 | `Executor::run_sync`（520 行）拆成 Planner（图→执行计划：融合划分、拓扑序、各段入口）与 Runner（分配、迁移、发射）；计划按图结构哈希缓存，训练循环逐步复用 | 2.02、2.03、2.11 | [架构](codebase-audit/07-architecture.md)§核心抽象；系统 A4；流水（末段） | UNet 每步执行器 CPU 时间由约 16 ms 降到计划命中后的发射成本；`executor.h` 不再有流水线补丁状态 |
| 3.02 | jit key 结构化：2 MB 无边界检查缓冲与 mprotect 守护页（`jit_key.h:16-18,129-198`、`jit_key.cc:36-43`）换成有长度检查、溢出抛正常异常的结构；融合边号变长编码，≥256 不再回绕（`fused_op.cc:181`、`executor.cc:86`）；`_` 开头 loop option 也入键或设置时拒绝（`fused_op.cc:206`） | 3.01 | [核心](codebase-audit/01-core-runtime.md)§JIT 键与代码生成 | 含 300 个算子的融合段与 200 个的键不同；溢出可捕获 |
| 3.03 | 三张 kernel 缓存表键改 `string`（`string_view_map` 的 SSO 悬垂，`misc/string_view_map.h:30-54`），加容量上限与 LRU；`FusedOpContext` 可释放；context 不持有指向栈上 `FusedOp` 的指针，relay 信息编译时固化成 POD（`executor.cc:539`、`fused_op.cc:223`、`parallel_compiler.cc:194,265-266`） | 3.01 | [核心](codebase-audit/01-core-runtime.md)§JIT 键与代码生成、§执行器 | 动态形状负载下表大小有界；ASan 无悬垂 |
| 3.04 | 求 jit key 改纯函数：`do_jit_prepare` 不再清 `_cuda`/`_cpu` 位（`op.cc:196-228`、`fused_op.cc:170-176`）；后端选择结果放执行计划；删 `op.cc:316-320` 的双键 | 3.01 | [核心](codebase-audit/01-core-runtime.md)§执行器 | `use_cuda=0` 下 prepare 过的算子打开 `use_cuda` 后能走 CUDA |
| 3.05 | 删除算子构造期回调执行器（`op.cc:127-137` 的 `exe.run_sync`）：动态形状走主机侧形状缓存或延迟形状；6 个含 `cudaDeviceSynchronize` 的算子文件改事件级等待；`.item()` 只等所需子图 | 3.01 | [核心](codebase-audit/01-core-runtime.md)§执行器；系统 A5 | 建图期间不再有嵌套 `run_sync` |
| 3.06 | 并行编译器修到可信：`has_error` 用 atomic、`error_msg` 加锁、等待改 join/future、异常用 `exception_ptr` 传递（`parallel_compiler.cc:113-130,226-227,308-309,327,369`）；线程数按 `sched_getaffinity` 与 cgroup 配额取且去 `static`（`:218`）；fork 后重建或禁用线程池（`:107-112`、`init.cc:102-109`）；catch 里不再二次 `do_prepare`（`:203,297`、`executor.cc:681-693`） | — | [核心](codebase-audit/01-core-runtime.md)§执行器；[构建](codebase-audit/04-build-tooling.md)§锁与并发；系统 D4 | 门禁与默认配置同为 `use_parallel_op_compiler=16`；工作线程抛出后主线程不自旋 |
| 3.07 | 执行器在设备等待段释放 GIL（`run_sync` 的 `cudaDeviceSynchronize`、`.item()`、`.numpy()`） | 3.01 | [核心](codebase-audit/01-core-runtime.md)§执行器 | 另一个 Python 线程在 sync 期间能运行 |
| 3.08 | KernelIR 结构化：节点类型用 enum，语义属性用带类型结构体，`get_attr` 对缺失键报错（`kernel_ir.h:23,33`、`kernel_ir.cc:50-52`）；13 个 pass 文件里 14 个字符串字面量契约改显式声明并由 pass manager 校验依赖顺序 | 1.01 | [核心](codebase-audit/01-core-runtime.md)§JIT 键与代码生成 | 拼错属性名编译期或注册期报错 |
| 3.09 | 死代码消除不再按「语句含 `void` 一词」删除（`kernel_ir.cc:865-871`）：解析时结构化识别 `(void)expr;` 并打属性；`reduce_op.cc:405`、`transpose_op.cc:123` 的依赖随之改写 | 3.08 | [核心](codebase-audit/01-core-runtime.md)§补充：代码生成与优化 pass | `memset((void*)p,0,n)` 保留在融合 kernel 里的用例 |
| 3.10 | 算子内标识符改名走结构化成员表并先做合法性校验，替代三个硬编码白名单与 `op{i}_` 盲目前缀（`op_compiler.cc:914-933,1074-1076`） | 3.08 | [核心](codebase-audit/01-core-runtime.md)§补充：代码生成与优化 pass | 含 `size_t`/`nullptr`/`return` 的 element 算子能编译 |
| 3.11 | 生成源码里的结构体字节偏移改显式 setter，成员表用宏声明；删 `compiler.py:431-440` 的正则扫头文件（`var_relay.cc:189-193`） | 3.08 | [核心](codebase-audit/01-core-runtime.md)§JIT 键与代码生成 | 给算子结构体加成员后旧缓存 kernel 不再写错位置 |
| 3.12 | `float_atomic_fix_pass.cc:76-80`、`fake_main_pass.cc:91-95` 名字解析失败改为错误 | 3.08 | [核心](codebase-audit/01-core-runtime.md)§JIT 键与代码生成 | 解析失败的 kernel 报错而非算错 |
| 3.13 | 循环维度身份用整数向量，`range10` 不再被拆成 `range1*range0`（`merge_loop_var_pass.cc:22-24,74-82,128`） | 3.08 | [核心](codebase-audit/01-core-runtime.md)§补充：代码生成与优化 pass | 10 维张量归约的对拍用例 |
| 3.14 | 两个同名 pass（`unroll_pass.h:13` 与 `expand_empty_block_pass.h:13` 都叫 `expand_empty_block`）改名；`get_pass` 按类型索引不用 C 风格下转型（`pass_manager.h:54,62`） | 3.08 | [核心](codebase-audit/01-core-runtime.md)§补充：代码生成与优化 pass | `exclude_pass` 能单独关 unroll |
| 3.15 | 一次编译只解析一遍：pass 管线解析后 clone IR，tuner 只改 loop_options（`pass_manager.cc:47`、`tuner_manager.cc:35-59`、`jit_searcher.cc:33-35`）；`jit_searcher.cc:58-61` 的 timeout 真正读取；`reorder_tuner.cc:22-24` 候选加上界 | 3.08 | [核心](codebase-audit/01-core-runtime.md)§补充：代码生成与优化 pass | 首次编译一个融合算子只解析一次 |
| 3.16 | `token_replace_all` 不再用 CHECK 抛异常做循环终止（`str_utils.cc:187,227-239`） | — | [核心](codebase-audit/01-core-runtime.md)§补充：绑定层与失败模式 | 正常路径零异常 |
| 3.17 | 只用于代码生成的 JIT 区段与普通 C++ 分离（或给 KernelIR 解析器加逃逸机制），写出算子源码编写规范与错误定位（`#include`、`_Pragma`、格式串 `%` 曾使 1199 个 CUDA 用例挂掉） | 3.08 | 系统 B5 | 含 `_Pragma` 的算子能编译或给出指向行号的错误 |
| 3.18 | 删掉 `asm_tuner` 链路（每个 CPU kernel 编译起一个 Python 进程，`jit_compiler.cc:253-256`、`asm_tuner.py:145-160`）：`use_movnt_pass.cc:24` 的 movnt 改 intrinsic 或编译器选项 | — | [构建](codebase-audit/04-build-tooling.md)§跨平台与死代码 | CPU kernel 冷编译时间下降 ≥ 50% |
| 3.19 | `event_queue` 异步基础设施修好并加测试，或删除（`executor.cc:704-707` 注释掉的 `run_sync`、`event_queue.h:26` 的 volatile、`tape_op.cc:38-44` 的 TODO） | 3.01 | [核心](codebase-audit/01-core-runtime.md)§错误处理与失败模式 | 不存在标注「cause hang」的死代码 |
| 3.20 | 执行器提供「提交部分图」显式接口，`jt.grad` 与 `Function` 回调降开销，让反向建图也能参与流水（上一轮在 `grad()` 里切碎反向图的做法已否决） | 3.01、2.11 | 系统 A2；流水 | Llama/UNet 反向段 GPU 空转下降且不慢于当前 |
| 3.21 | 每算子建图成本：边表分配（2.06）、jit key 拼接（3.02）、pyjt 绑定层开销逐项计量并降低；目标 UNet 前向建图 9 ms → ≤ 5 ms | 2.06、3.02 | 流水（末段）；系统 A4 | 分相计时报告 |
| 3.22 | CUDA 归约块内树形归约：在 `WarpReducePass` 之上加共享内存的块级归约（每输出只写一次），或让修好的 `SharedReducePass` 承担 | 1.03、1.04 | [核心](codebase-audit/01-core-runtime.md)§补充：CUDA 归约 | UNet 归约类 kernel 合计不慢于 PyTorch 的 1.13 ms |
| 3.23 | 融合逐元素 kernel 带宽效率：UNet 61 种融合 kernel 合计 4.47 ms、约 475 GB/s（峰值一半），逐 kernel 看生成代码与访存模式 | 3.08 | 流水（末段）；[核心](codebase-audit/01-core-runtime.md)§补充：CUDA 归约（末段的角色分解） | 逐元素类合计 ≤ PyTorch 的 3.07 ms |
| 3.24 | 布局收尾：`src/opt/`、`op_compiler.*`、`jit_key.*`、`jit_compiler.*`、`opt/kernel_ir.*` 进 `src/codegen/`；`src/ops/` 只留七个元算子，getitem/setitem/argsort/candidate/where/copy/clone/fused_adamw 等进 `src/ops/composite/` | 3.08、0.19 | [布局](target-layout.md)§3；[架构](codebase-audit/07-architecture.md)§核心抽象（元算子名不副实） | `src/ops/` 顶层只有元算子 |

## 7. 阶段 4 · 设备与后端注册表（架构）

| 编号 | 任务 | 前置 | 出处 | 验收 |
| --- | --- | --- | --- | --- |
| 4.01 | 分配器 id 空间随分配器实例走，不再是进程静态 2M 单例（`sfrl_allocator.h:35-38`）；索引前断言范围（`sfrl_allocator.cc:81-86`） | 2.04 | [核心](codebase-audit/01-core-runtime.md)§内存与分配器 | CPU 与每个 CUDA 设备各有独立 id 表 |
| 4.02 | 合并多卡：在 `device-select` 与 `multi-device` 之间选一个字段名与一条标量规则，跑三套门禁，确认结构测试只剩 2.0 上已有的三条失败，合并 | 2.04、4.01 | [多卡](device-placement.md)§5；系统 C1 | `Var` 有设备字段；`set_device` 不重启进程；两卡用例通过 |
| 4.03 | `BackendRegistry`：CPU 与 CUDA 作为前两个成员（device_count/allocator/set_device/memcpy/synchronize/stream），`use_cuda` 在其上重实现；`use_acl`/`use_rocm`/`use_corex`/`use_device` 别名标记废弃；配套契约测试 | 4.02、2.13 | [多后端](multi-backend-design.md)§3.1、§4 阶段 2；[架构](codebase-audit/07-architecture.md)§重复与一致性 | 全部门禁行为不变 |
| 4.04 | `OpRegistry`：执行器按（算子 id，后端）查 kernel；CUDA 库算子经注册表登记；核心 5 处 `has_op` 字符串查询与 tuner 两处写死名字（`conv_tuner.cc:371-395`、`matmul_tuner.cc:103`）改查表；`Op` 拆成 OpDef/Kernel/Codegen 三个接口由注册表组合 | 4.03、2.17、2.18 | [多后端](multi-backend-design.md)§3.2；[架构](codebase-audit/07-architecture.md)§分层、§核心抽象 | 核心不再认识任何可选后端的名字；`backend.supported_ops()` 可查 |
| 4.05 | Python 分派表：98 处 `use_cuda`/`is_cuda`/`has_acl` 判断（18 种写法）收进按（算子，设备，dtype）注册的表；`nn/backends/*.py` 的 39 处守卫删除；`jt.compile_extern.*` 的 globals 注入改后端能力查询接口；Conv 的 depthwise 分派移到 execute | 4.04 | [Python](codebase-audit/02-python-api.md)§后端分派；[架构](codebase-audit/07-architecture.md)§公共 API | `python/jittor` 内（compat/extern 外）直接判 `use_cuda` 的行数为 0 |
| 4.06 | `jt.flags.backend_fallback ∈ {error, warn, allow}`，默认 warn，harness 设 error；回退必须说明算子、后端与原因；异常不再作为路由机制（`acl_op_exec.cc:223`） | 4.04 | [多后端](multi-backend-design.md)§3.3 | harness 不再 grep `fallback cpu` |
| 4.07 | 后端配置改为返回 `BuildConfig` 值：三个 `*_compiler.py` 不再改写 `compiler` 全局（22 处）；`extra_core_files` 不由后端追加；`jittor_utils` 禁止 import jittor（环 1，`jittor_utils/__init__.py:731,807,810,869`）；后端用 entry point 发现并懒加载（corex/rocm 不再每次 import 探测） | 4.03 | [架构](codebase-audit/07-architecture.md)§分层；[构建](codebase-audit/04-build-tooling.md)§跨平台与死代码 | import 顺序不再决定行为 |
| 4.08 | 流与事件模型：每设备默认流之外的通信流与拷贝流，事件依赖，内存复用按事件而非流序；`fetch_op.cc` 的半套顺序保证补齐（`:121-122,156-159`，源块释放后立即回 free list） | 4.02 | 系统 C3；[核心](codebase-audit/01-core-runtime.md)§补充：内存与分配器 | H2D/D2H 与计算重叠的用例；fetch 不再读到被覆盖的数据 |
| 4.09 | per-device 库句柄（cuBLAS/cuDNN/cuSPARSE/cuRAND/cuFFT）加每次执行前 `SetStream`；全仓当前无 `cublasSetStream`/`cudnnSetStream` | 4.02、4.08 | [后端](codebase-audit/06-backends.md)§库句柄与资源生命周期；[多卡](device-placement.md)§2 | 两卡各自的句柄；流参数进 harness |
| 4.10 | CUDA kernel 存放位置统一：`extern/cuda/<lib>/ops`、`nn/backends/*_cuda.py`、`nn/*_cuda.py`、`src/ops` 的 `#ifdef HAS_CUDA` 四处 → 一处加注册表；`nn/` 根下 6 个 `*_cuda.py` 与 `kv_cache_acl.py` 移出，给 `nn/` 加 exact-entry 门禁 | 4.04 | [架构](codebase-audit/07-architecture.md)§重复与一致性、§模块边界 | 找一个算子实现只搜一个目录 |
| 4.11 | ACL 改为注册表后端：`.py`/`_op_acl.cc` 对转为注册项；`change_function()` 与 `warp()` 闭包删除；`post_process()` 不再改四个无关全局（需昇腾硬件） | 4.04、4.05 | [多后端](multi-backend-design.md)§4 阶段 4 | NPU 测试通过；`acl_compiler.py:803` 消失 |
| 4.12 | 删除 `process_jittor_source` 与 `process_acl`（整树文本替换、`WTF` 补丁）；ROCm 自己实现并注册（需 ROCm 硬件） | 4.11 | [多后端](multi-backend-design.md)§4 阶段 5；[架构](codebase-audit/07-architecture.md)§代码规模 | 核心源码不再是移植的输入 |
| 4.13 | 跨后端契约矩阵：一张算子矩阵对每个注册后端跑同一套对拍 | 4.11 | [多后端](multi-backend-design.md)§4 阶段 6 | 新门禁层 |
| 4.14 | `Module.cuda(i)`/`npu(i)`/`x.to(...)`/`x.cpu()` 语义落实（`core_api.py:1678-1685`、`tensor_ops.py:2604-2618,2849-2853`）：设备号生效，`.to()` 按 torch 签名解析不依赖 kwargs 顺序，`x.cpu()` 真迁移 | 4.02 | [Python](codebase-audit/02-python-api.md)§Module 与参数模型 | `x.to(device='cuda', dtype=float16)` 两者都生效 |
| 4.15 | 布局收尾：`python/jittor/extern/` 整体搬到顶层 `backends/<name>/`，每后端同一形状（build 片段 + kernels/ + 注册项）；`nn/backends/` 10 个与 `nn/` 根下 6 个 `*_cuda.py` 以及 `flash_attention.py` 的 CUDA 串进 `backends/cuda/kernels/`，Python 层只留调用；`python/jittor/src/` 搬出包到顶层 `src/`，`MANIFEST.in`/`pyproject` 的打包与 `jittor_path` 解析随 9.01 同一提交改 | 4.10、4.12、9.01、0.19 | [布局](target-layout.md)§3、§5 | `python/jittor/` 下无 .cc/.cu；`grep cuda_src` 于 `python/jittor` 为 0 |

## 8. 阶段 5 · Python 对象模型（架构）

热点文件：`_runtime/core_api.py`、`misc/tensor_ops.py`、`misc/indexing.py`、`nn/modules/parameter.py`、`__init__.py`。

| 编号 | 任务 | 前置 | 出处 | 验收 |
| --- | --- | --- | --- | --- |
| 5.01 | 114 个 `foo_` 就地方法改白名单显式声明（`__init__.py:135-154` 的首参数名启发式删除）；返回非 Var 的函数禁止生成（`isnan_`、`norm_`、`sort_`、`tolist_`…） | — | [Python](codebase-audit/02-python-api.md)§张量语义 | `dir(Var)` 中 `_` 后缀全部是真就地 |
| 5.02 | 视图与存储模型：`Var` 引入 storage（base+offset+strides），切片/view/expand 共享存储，所有就地操作作用于 storage；删 `var_holder.cc:406-441` 的十层下标链写回；`Var.view = reshape`（`core_api.py:722`）与 `expand = broadcast`（`shape_transforms.py:122`）改真视图 | 2.05、3.01 | [Python](codebase-audit/02-python-api.md)§张量语义；[兼容](codebase-audit/03-compat-shim.md)§张量语义；系统 E1 | `v=y[1:4]; v.assign(v+100)` 改到 y；expand 不物化 |
| 5.03 | 转置隐藏标记（`core_api.py:752-760`，`matrix.py:118-121,180-232` 消费）改为基于图结构的转置融合，或快路径校验版本号 | 5.02 | [Python](codebase-audit/02-python-api.md)§张量语义 | `at=a.transpose(); a.assign(0); matmul(at,b)` 得 0 |
| 5.04 | 参数模型：Parameter/Buffer 是类型；一份带策略参数的遍历，`parameters`/`named_parameters`/`state_dict`/`named_buffers`/`_buffers` 是它的视图（`core_api.py:1466,1525,1615,2212,2221`）；查询不改名（`:1518,1590`）；BatchNorm 改 `register_buffer`（`normalization.py:33-39`）；删 `_torch_registration_semantics` 模式位与内核里的 torch 分支（`core_api.py:1402-1405,2155-2173,2201-2207`、`parameter.py`） | — | [Python](codebase-audit/02-python-api.md)§Module 与参数模型；[兼容](codebase-audit/03-compat-shim.md)§张量语义 | 权重绑定时 `parameters()` 与 `named_parameters()` 数目一致；`bn.running_mean = zeros` 不进参数 |
| 5.05 | `eval()`/`train()` 只切 `is_train`，冻结统一由 `requires_grad` 表达；删按 `id(p)` 的备份还原（`core_api.py:2094-2123`） | 5.04 | [Python](codebase-audit/02-python-api.md)§Module 与参数模型 | 子模块 eval 后父模块 train 参数可训练 |
| 5.06 | hook 存实例级有序字典，多 hook、prepend/always_call 生效、可移除 handle；删类级 `__call__` 交换（`core_api.py:1833-1861`）；`Var.register_hook` 返回 handle 而非 swap（`:2460-2483`），与 Module 的 `_RemovableHandle` 统一 | — | [Python](codebase-audit/02-python-api.md)§Module 与参数模型、§jt.Function 的契约 | 连续注册两个 forward hook 都触发；装 hook 不影响其它实例 |
| 5.07 | `jt.Function` 每次调用创建一次性上下文对象，实例无状态（`core_api.py:2380-2418`）；`__call__` 接受关键字参数（`:2440-2442`） | — | [Python](codebase-audit/02-python-api.md)§jt.Function 的契约 | `f=Mul(); f(a,b); f(a,c); grad(o1,a)` 得 2 |
| 5.08 | `flag_scope` 的备份改局部栈，`__call__` 每次新建 scope（`core_api.py:104-112,137-167`）；`@jt.no_grad()` 递归后不再永久泄漏 | — | [Python](codebase-audit/02-python-api.md)§jt.Function 的契约 | 递归装饰函数返回后 `jt.flags.no_grad` 恢复 |
| 5.09 | 29 处融合 kernel 的启用条件由全局 `no_grad` 改为「输出不需要梯度」；`nn/backends/cudnn.py:41-44` 的 fp16/bf16 卷积不再只在 no_grad 下走 cuDNN | 5.04 | [Python](codebase-audit/02-python-api.md)§jt.Function 的契约 | `model.eval()` 无显式 no_grad 时融合 kernel 生效；fp16 训练卷积走 cuDNN |
| 5.10 | 索引与计数统一 int64（`where_op.h:30` nonzero、`tensor_ops.py:1850` randperm、`:1265` topk、`pool/core_2d.py:108,198`） | 2.16 | [Python](codebase-audit/02-python-api.md)§张量语义 | 与 torch dtype 一致；删 2^31 断言 |
| 5.11 | `amp_reg` 位常量命名导出，一律 `\|=`；`concatenation.py:49` 的 `amp_reg=4` 赋值改按位或；`array()`/`random()` 的 amp 逻辑合一（`core_api.py:417-424,459-466,471`） | — | [Python](codebase-audit/02-python-api.md)§后端分派 | `jt.concat` 不再改写用户 AMP 偏好 |
| 5.12 | matmul 四条路径共用能力表，dtype 用枚举不用子串（`matrix.py:99-256`）；`bmm_transpose` 与 `matmul` 的 amp 处理一致 | 4.05 | [Python](codebase-audit/02-python-api.md)§后端分派 | 同一 matmul 在四种设备/dtype 组合下累加精度有表可查 |
| 5.13 | `unique`（四条路径、主机同步、比较器截断 `tensor_ops.py:761-815`）、`cumsum`（numpy 回调对 CUB，`:1313-1373`）、`isnan/isinf/isfinite`（`:2567-2586` 非 ACL 内核 cast float）统一为算子层单一实现 | 4.04 | [Python](codebase-audit/02-python-api.md)§后端分派 | float64 `1e300` 在所有后端都不是 inf；unique 对 float 不残留重复 |
| 5.14 | `Var.scatter` 改非就地（`tensor_ops.py:2030` 内部先 clone） | — | [Python](codebase-audit/02-python-api.md)§张量语义 | `y=x.scatter(...)` 不改 x |
| 5.15 | `.half()`/`.float16()` 删死的 amp 分支（`core_api.py:2281-2292`），转换范围含浮点 buffer | 5.04 | [Python](codebase-audit/02-python-api.md)§Module 与参数模型 | `bn.half()` 后 running_mean 为 float16 |
| 5.16 | `state_dict(to="torch")` 用 `from_numpy`，不强制 float32（`core_api.py:1607`） | — | [Python](codebase-audit/02-python-api.md)§Module 与参数模型 | int/bool buffer 保持 dtype |
| 5.17 | 同一概念合并（一）：删旧 `pool/layers.py:20` AvgPool2d 并转发到 `nn/modules/pooling.py`，AvgPool3d 改用修正版；`nn.Conv2d.execute` 委托 `functional.conv2d`（含 `{"G","C"}` 编译选项与输出尺寸校验合一）；BatchNorm/LayerNorm/GroupNorm 模块只做参数管理；`batch_norm(training=True)` 走融合 kernel；BN 的 sync 与非 sync 分支同一套数学（`normalization.py:56-71`） | 5.04 | [Python](codebase-audit/02-python-api.md)§同一概念的多份实现 | `jt.nn.AvgPool2d` 与 `jt.pool.AvgPool2d` 同值；模块与 functional 生成同一 JIT kernel |
| 5.18 | 同一概念合并（二）：`functional/loss.py` 一个 `_reduce` helper（四种解析、三种错误行为）；`models/_utils.py` 单份 SqueezeExcitation/StochasticDepth/ConvNormActivation，修 `efficientnet.py:86-94` 投影层被强加 SiLU；分布类全部继承 `Distribution`，`sample` 在基类实现且 detach，`kl_divergence` 未覆盖时报错；legacy scheduler 继承新基类（`legacy_schedulers.py`）；`normalize` 合并到 torch 语义；`init.py` 两张 gain 表合一、fan 算法合一；`linalg.py` 12 份 helper 提到模块级 | — | [Python](codebase-audit/02-python-api.md)§同一概念的多份实现 | efficientnet b0–b7 与 torchvision 对拍；`reduction='MEAN'` 三个损失同一种错误 |
| 5.19 | 被静默忽略的参数改为传非默认值时 warn 或 raise：relu 等 `inplace`、instance_norm 的 running stats、resnet `zero_init_residual`、svd `compute_uv`/`driver`、inv_ex `check_errors`、ctc_loss `zero_infinity`、topk `sorted`、sort `stable`、kaiming `generator`、DataLoader `pin_memory`/`persistent_workers`、vjp/jvp `strict`（修 `grads_i is None` 判据）、fftfreq `dtype`/`device` | — | [Python](codebase-audit/02-python-api.md)§参数被接受后静默忽略 | 每个参数一条负向测试 |
| 5.20 | import 期副作用删除：模块顶层 `set_global_seed`（`tensor_ops.py:1879-1880`）；`PIL.Image.open` 永久替换（`dataset/dataset.py:39-40`、`utils.py:57-61`）改 opt-in 上下文管理器 | — | [Python](codebase-audit/02-python-api.md)§导入期的全局副作用 | import 前设的 `np.random.seed` 不被抹掉；`PIL.Image.open` 是原函数 |
| 5.21 | 六个 monkeypatch 安装器写成显式有序清单并加断言（`nn/__init__.py:104` → `__init__.py:109…219`）；`install_full_reduce_fast_path` 装在算子层，`jt.sum` 与 `Var.sum` 同一路径（`full_reduce_cuda.py:147-177`） | 4.05 | [Python](codebase-audit/02-python-api.md)§导入期的全局副作用 | `jt.sum is jt.Var.sum`；安装序列有测试 |
| 5.22 | `nn` facade 不导出 39 个下划线名，内部用模块局部名不经 `jt.nn.*` 晚绑定；先修 `shape_transforms.py:29,53` 对 `jt.misc.Sequence`/`jt.misc.np` 的依赖再给 `tensor_ops` 加 `__all__` | — | [Python](codebase-audit/02-python-api.md)§导入期的全局副作用 | `dir(jt.nn)` 无下划线名；`jt.np` 不存在 |
| 5.23 | 根命名空间显式 `__all__`（替代 5 次星号导入），`.pyi` 从 `__all__` 生成并加一致性门禁（`__init__.py:57-125`） | 5.22 | [架构](codebase-audit/07-architecture.md)§公共 API | `.pyi` 顶层名与 `__all__` 一致 |
| 5.24 | 10 个 `jt._*` 跨模块契约（`_torch_leaf_params`、`_active_optimizers`、`_current_optimizer`、`_torch_retained`…）收进 Runtime 对象 | 2.13 | [架构](codebase-audit/07-architecture.md)§公共 API | 根命名空间无下划线契约 |
| 5.25 | `python/jittor/utils/` 拆散：编译器资源归 compiler 包、`gen_pyi`/`local_doc_builder` 归 `tools/`、`pytorch_converter.py` 归 compat、`converter_server.py` 与其启动脚本合并 | 3.18 | [架构](codebase-audit/07-architecture.md)§模块边界 | `utils/` 不存在或只剩一个明确职责 |
| 5.26 | 布局收尾：`_runtime/core_api.py` 拆成 `_core/{var,module,function,flags,hooks}.py`；`misc/` 改名 `ops/` 并把 `tensor_ops.py` 按 indexing/reduction/shape/sort/scan 拆开；`pool/` 并入 `nn`；`linalg.py`/`distributions.py`/`init.py` 各自成包；`ccl/`、`loss3d/`、`math_util/`、`einops/` 进 `contrib/` 或分出去；`compiler.py`/`compile_extern.py`/`pyjt_compiler.py`/`cuda_wheel.py`/`install_cuda.py` 与 `python/jittor_utils/` 合成 `build/` 包；1.x 公开路径留 deprecated 转发模块 | 5.04–5.08、5.13、5.17、5.18、5.25、4.07、0.19 | [布局](target-layout.md)§3、§4 | 包根下只剩 `__init__.py`、`__init__.pyi`、`selftest.py`；无 > 1500 行的 .py |

## 9. 阶段 6 · 静默算错的单点修复（可并行）

每条都是独立提交，必须附修前失败的测试。标注「被 X 取代」的条目不单独修，等 X 落地时验证其测试通过。

### 9.1 核心 C++

| 编号 | 任务 | 前置 | 出处 | 验收 |
| --- | --- | --- | --- | --- |
| 6.C01 | `.item()` 对无符号 dtype：`ItemData data{}` 值初始化并补齐 uint8/16/32/64 分支（`var_holder.cc:284,305`、`py_converter.h:496-513`）；ROCm 上 bf16 的转换与 dtype 改写放同一分支（`:272-279`） | — | [核心](codebase-audit/01-core-runtime.md)§类型系统 | `jt.array(np.uint8([200])).item() == 200` |
| 6.C02 | `PySlice_Unpack` 返回值检查，三个变量初始化（`py_converter.h:158-167`） | — | [核心](codebase-audit/01-core-runtime.md)§绑定层 | `a[::0]` 抛 ValueError |
| 6.C03 | 整数提升 | 被 2.16 取代 | | |
| 6.C04 | 含 `void` 语句被删 | 被 3.09 取代 | | |
| 6.C05 | 融合边号 ≥256 回绕 | 被 3.02 取代；在此之前先加 `ASSERT(ops.size()+iv < 256)` 作临时防线 | [核心](codebase-audit/01-core-runtime.md)§JIT 键 | 超限报错 |
| 6.C06 | `grad.cc:65-68` 判空对象改为 `dx`（返回 nullptr 是 floor/round/ceil/mod/位运算的常规路径）；`grad.cc:262` 同 | — | [核心](codebase-audit/01-core-runtime.md)§自动微分 | 设 `compile_options` 后 `jt.floor(x)` 进反向不段错误 |
| 6.C07 | 缺失梯度默认报错（`op.cc:78`；`grad.cc:76-82` 按 var 名去重的进程级 map 删除） | — | [核心](codebase-audit/01-core-runtime.md)§自动微分 | 第二个无名 var 缺梯度仍报错 |
| 6.C08 | `grad.cc:146-261` 两趟遍历合一趟并快照结构，删无边界游标 | 2.02 | [核心](codebase-audit/01-core-runtime.md)§自动微分 | 反向中 `Op::forward` 加输出的用例 |
| 6.C09 | `backward()` 可重复：图释放与 `set_stop_grad` 分开（`grad.cc:281-294`，`var_holder.cc:183-186`） | — | [核心](codebase-audit/01-core-runtime.md)§自动微分 | 第二次反向报错或给出正确梯度，不再静默零 |
| 6.C10 | CUDA 分配钩子两张 map 用 `find` 加显式错误，释放后 `erase`（`executor.cc:721-733`） | — | [核心](codebase-audit/01-core-runtime.md)§执行器 | 传入未知指针报错；map 不增长 |
| 6.C11 | CPU 分配失败抛异常，返回值必须检查（`aligned_allocator.cc:16-28`、`var.cc:126-128`、`executor.cc:576-578`）；`aligned_alloc` 的 size 对齐要求满足 | — | [核心](codebase-audit/01-core-runtime.md)§内存与分配器 | CPU OOM 报 OOM 而非段错误 |
| 6.C12 | `cuda_device_allocator.cc:32-37` 的 managed 回退放到 LOGf 之前（LOGf 改 LOGw） | — | [核心](codebase-audit/01-core-runtime.md)§内存与分配器 | `managed_fallback` 真的回退 |
| 6.C13 | 零字节分配不返回伪指针 `0x10`（`cuda_device_allocator.cc:25,41`） | — | [核心](codebase-audit/01-core-runtime.md)§内存与分配器 | 零元素 Var 的 mem_ptr 为空 |
| 6.C14 | SFRL：`occupied_id_mapper` 用 `new …[N]()` 清零，free/share_with 先校验后解引用（`sfrl_allocator.cc:24-25,291-292,313-315`）；5 个分配器的 alloc 必须写回 `allocation`；`getitem_op.cc:515-518`、`setitem_op.cc:336-341` 的别名判断改用显式 share 关系 | 2.04 | [核心](codebase-audit/01-core-runtime.md)§补充：内存与分配器 | `use_sfrl_allocator=0` 与 `use_nfef_allocator=1` 下 setitem 对拍通过 |
| 6.C15 | `migrate_to_cpu/gpu` 迁移前检查 share_with 关系，整组迁移或拒绝（`allocator.cc:167-200`、`executor.cc:593-610`） | 2.04 | [核心](codebase-audit/01-core-runtime.md)§补充：内存与分配器 | 混合 CPU-CUDA 图里 inplace 别名不断开 |
| 6.C16 | fetch 跨流：拷贝后记 event，默认流等待（`fetch_op.cc:121-159`；`sfrl_allocator.cc:293-301`） | — | [核心](codebase-audit/01-core-runtime.md)§补充：内存与分配器 | 被 4.08 覆盖时验证其测试 |
| 6.C17 | `TempAllocator` 删遮蔽基类的 `used_memory`/`unused_memory`（`temp_allocator.h:29`） | — | [核心](codebase-audit/01-core-runtime.md)§补充：内存与分配器 | `device_mem_limit` 对 workspace 生效 |
| 6.C18 | CachingBlock 保存底层 allocation 并原样回传，不再传 0（`sfrl_allocator.cc:183,260,284`、`temp_allocator.cc:93-116`） | — | [核心](codebase-audit/01-core-runtime.md)§补充：内存与分配器 | 嵌套分配器配置通过 |
| 6.C19 | 每个分配器一把锁并覆盖 `gc()`（`sfrl_allocator.cc:241,307-310`、`temp_allocator.cc:41-110`）；`free_ratio=1` 的死策略改小或删 | — | [核心](codebase-audit/01-core-runtime.md)§补充：内存与分配器 | CPU 与 GPU 分配不再串行 |
| 6.C20 | swap：文件名用运行期 pid 加唯一 token（`mem/swap.cc:26,40`）、`cudaMemcpy` 查错（`:50`）、去静态 8 MB buffer（`:46`）；`save_mem` 在完成前用编译开关隔离出热释放路径（`var.cc:37-53`，`swap.h:37-58` 的 TODO） | — | [核心](codebase-audit/01-core-runtime.md)§内存与分配器 | fork 后父子不写同名 swap |
| 6.C21 | `check_graph` 在 release 构建里**只做一半而不自陈**。`do_graph_check` 含两个检查：f/b/p liveness 重算（沿 `inputs()` 反向走，无需注册表，**每种构建都工作**）与悬垂节点检查（遍历 `lived_nodes`，只在 `#ifdef NODE_MEMCHECK` 下填充，release 里恒空）。于是 `check_graph=1` 在 release 下**校验了不变量、静默跳过了泄漏检查**，用户拿到的是「全过」。**最小修法是让这个不对称可见**（跳过时明说跳过了什么、为什么），而不是把泄漏检查也搬进 release——那要给 `lived_nodes` 一套 release 下的填充机制，是另一件事，需要单独立项 | 2.10 | [核心](codebase-audit/01-core-runtime.md)§错误处理（2026-09-03 由 6.C31 的执行者核实并更正原描述） | release 下开 `check_graph=1` 会明确报出「liveness 已校验、泄漏检查因非 memcheck 构建跳过」 |
| 6.C22 | pyjt 关键字参数：长度检查计入 kw、未消费的 kwname 报错（`pyjt_compiler.py:106-113,166`）；转换错误检查移到关键字填充之后（`:193-196,219-223,714-717`）；重载先按 kwname 映射槽位再做类型检查（`:150`） | — | [核心](codebase-audit/01-core-runtime.md)§绑定层 | `v.detach(non_blocking=True)` 报错；`x.sum(dim=2**40)` 报 OverflowError；`x.sum(keepdims=True, dim=1)` 可用 |
| 6.C23 | `is_type<NanoString>` 收窄（`py_converter.h:238-250`）；`matched_overload` 移到转换成功之后 | 6.C22 | [核心](codebase-audit/01-core-runtime.md)§绑定层 | 传函数给 dtype 参数的错误信息指向参数 |
| 6.C24 | 带实例 `__dict__` 的类型加 `Py_TPFLAGS_HAVE_GC` 与 traverse/clear（`pyjt_compiler.py:874-876`） | — | [核心](codebase-audit/01-core-runtime.md)§绑定层 | `v.foo = v` 能被 GC 回收 |
| 6.C25 | 生成绑定补 `catch (...)`；`pyjt_console.h:531,533` 的 `throw new` 改 `throw` | — | [核心](codebase-audit/01-core-runtime.md)§绑定层 | 非 std::exception 不再 terminate |
| 6.C26 | `pyjt_compiler.py` 的 C++ 解析：`split_args` 数圆括号且不让 `>` 把 presum 变负；`find_bc` 跳过字符串与注释；`VarHolder *foo` 与多个 `=` 的情况加断言（`:72-86,215-218,321-338,389,400`）；长期换 libclang | — | [核心](codebase-audit/01-core-runtime.md)§绑定层 | 对每种合法写法有生成器单测 |
| 6.C27 | `Var.data` 返回的 numpy 视图 base 指向包裹该次 allocation 的胶囊（`var_holder.h:310-321`、`py_converter.h:471-488`） | — | [核心](codebase-audit/01-core-runtime.md)§绑定层 | `a=v.data; v.assign(other); a[0]` 不读已释放内存 |
| 6.C28 | 生成带「已构造」标志的 `tp_new` 或 `tp_dealloc` 先检查（`pyjt_compiler.py:875`；`ring_buffer.cc:73`） | — | [核心](codebase-audit/01-core-runtime.md)§补充：绑定层与失败模式 | `jittor_core.RingBuffer()` 抛异常不段错误 |
| 6.C29 | 标量转数组的全局 `tmp_data` 改自带 buffer（`numpy.h:125-131`、`py_converter.h:363-374`） | — | [核心](codebase-audit/01-core-runtime.md)§补充：绑定层与失败模式 | 一次调用两个标量参数正确 |
| 6.C30 | `helper_cuda.h` 的 `peek` 去掉进程级闩 `peek_logged`（`log.cc:24`），改限频或按 call site 去重 | — | [后端](codebase-audit/06-backends.md)§错误处理 | 第二次异步错误仍报出 |
| 6.C31 | **进程级 SIGCHLD 处理器让任何被信号杀死的子进程连带杀掉父进程，且不留输出。** 两个分区独立撞上同一机制：(a) CUDA 构建上任何「失败的」`import jittor` 都会在退出期 abort—— 全局 EventQueue 的 worker 线程只由**跑完**的 import 通过 `core.cleanup()` 注销；没跑完就是 `~std::thread` 落在 joinable 线程上 → `terminate called without an active exception`。再叠加 jittor 自己的 SIGCHLD 处理器（父进程 `_Exit(1)` 且不刷 stdio），后果是**子进程 import 失败 → 父进程无声消失，一行输出都不留**。要动核心的静态析构顺序。(b) 任何**崩溃类测试**的子进程会直接杀掉整个 pytest session 而不是让那条用例失败，零输出——写这类测试必须在中间隔一层 shell。合起来：**这个处理器把「子进程异常退出」变成了「父进程无声消失」，掩盖的正是最需要诊断的那类失败**。处理器本身也不是 async-signal-safe | 2.19（析构不得抛） | 2026-09-03 由 8.09 与 6.C25 的执行者独立发现，不在原审计里 | 构造一个必然失败的 import 与一个必然崩溃的子进程，两者的退出码与 stderr 都可读；父进程不消失 |

### 9.2 Python 层

| 编号 | 任务 | 前置 | 出处 | 验收 |
| --- | --- | --- | --- | --- |
| 6.P01 | 转置标记陈旧 | 被 5.03 取代 | | |
| 6.P02 | Function 实例复用、no_grad 泄漏、tied weight 参数集合 | 被 5.07、5.08、5.04 取代 | | |
| 6.P03 | H1 分组 conv3d 的 ww reindex 形状顺序（`functional/convolution.py:173-178` 对 `:164`） | — | [Python](codebase-audit/02-python-api.md)§已确认会静默算错 | 无 cuDNN 下分组 conv3d 与 CPU 参考对拍 |
| 6.P04 | H2 Pool3d `return_indices` 内核第三层循环变量（`pool/core_3d.py:84`） | — | 同上 | 对拍通过；无越界 |
| 6.P05 | H3 Pool3d CUDA 反向用 `pout_shape` 作上界（`core_3d.py:162-165`） | — | 同上 | CUDA 反向与 CPU 反向一致 |
| 6.P06 | H4 MaxUnpool2d/3d 在 `stride != kernel_size` 时用原始宽度解码（`pool/unpool.py:60-67`） | — | 同上 | 与 torch 对拍 |
| 6.P07 | H5 eigh 反向 `dout` 全零时写零（`linalg.py:581-586`） | — | 同上 | 输出不含未初始化内存 |
| 6.P08 | H6 `_autograd_grad` 的 zip 用过滤后的 `new_grad_outputs`（`autograd/functional.py:190-202`） | — | 同上 | vjp/jvp 种子梯度与输出对齐 |
| 6.P09 | H7 irfft 对实数输入与显式 `n` 的处理走 `:68-73` 的判别函数（`fft/__init__.py:150-162`） | — | 同上 | 与 numpy 对拍 |
| 6.P10 | H8 ReduceLROnPlateau 每轮从初始 lr 计算（`legacy_schedulers.py:69-77`） | 5.18 | 同上 | 降一次只乘一次 factor |
| 6.P11 | H9 `unique(return_counts=True, return_inverse=False)` 补分支（`tensor_ops.py:973-981`） | — | 同上 | counts 返回 |
| 6.P12 | H10 Adan 的 `clip_grad_norm` 移出 param_group 循环（`adan.py:70-75`） | — | 同上 | 裁剪只做一次 |
| 6.P13 | H11 `zero_grad` 清缓冲而非只翻标志（`optim/base.py:152-153,100`） | — | 同上 | step 后 `clip_grad_norm` 生效 |
| 6.P14 | H12 Adam 偏差修正用每 param 的步数（`adam.py:62,75`、`base.py:221`） | — | 同上 | 梯度累积写法下与 torch 对拍 |
| 6.P15 | H13 worker 异常不再变成给父进程发 SIGINT（`dataset.py:321-326`） | — | 同上 | worker 抛出的异常在主进程重新抛出 |
| 6.P16 | H14 `mp_log_v` 做 int 转换（`dataset.py:37`） | — | 同上 | `mp_log_v=0` 关闭日志 |
| 6.P17 | H15 Pillow 版本用元组比较（`transform/function_pil.py:541,652`） | — | 同上 | Pillow ≥10 判断正确 |
| 6.P18 | H16 `Dataset.__deepcopy__` memo 存对象不存 id（`dataset.py:468`） | — | 同上 | deepcopy 后引用正确 |
| 6.P19 | H17 `LogitRelaxedBernoulli` 返回 logit；`RelaxedOneHotCategorical` 的 log_prob 与 base_dist（`distributions.py:546-562`） | 5.18 | 同上 | 与 torch 对拍 |
| 6.P20 | H18 `ComplexNumber.__rsub__` 虚部符号、`__imatmul__` 操作数顺序（`legacy_complex.py:174-180,237-244`） | — | 同上 | 与 numpy 对拍 |
| 6.P21 | H19 稀疏卷积重复坐标 CPU/CUDA 语义统一；neighbors 缓存校验内容不只 shape（`sparse/convolution.py:74,137-138,232-235`） | — | 同上 | 重复坐标用例两后端一致 |
| 6.P22 | H20 `to_dense` 对 COO 重复索引求和；spmm 不再转稠密（`sparse/coo.py:60-72`） | — | 同上 | 与 scipy 对拍 |
| 6.P23 | **`eigh` 的特征向量梯度在 CUDA 上错约 60%**（对 numpy 相对误差，修 6.P07 前后都复现）。一直没被发现是因为 `tests/ops/test_linalg.py::TestBUG4_2Op` 设了 `jt.flags.use_cuda=1` 且从不还原，把 CUDA 泄漏给该文件后面每一个用例——即「本该在 CPU 跑的用例其实跑在 CUDA 上，而 CUDA 结果是错的」 | — | 2026-09-02 由 6.P07 的执行者发现并复现，不在原审计里 | CUDA 与 numpy 对拍通过；顺带修掉那处 flag 泄漏（属 0.12） |
| 6.P24 | `Pool3d.__init__` 的 `count_include_pad and padding != 0` 读的是原始参数，元组 padding 恒为真、`padding=0` 恒为假，与 torch 的 `count_include_pad` 语义不一致 | — | 2026-09-02 由 6.P04/6.P05 的执行者发现，不在原审计里 | 元组与标量 padding 各一个与 torch 对拍的用例 |
| 6.P25 | Adan 的偏差修正仍用全局 `self.n_step`（`optim/algorithms/adan.py:68`），与 6.P14 修掉的 Adam 同一根因：`n_step` 数的是 backward 次数，梯度累积写法下指数偏 k 倍。`base.py` 的 `Optimizer._advance_step_count(pg)` 已就位，改动一行。**但要连带定一个语义**：Adan 里 `if self.n_step>0` 决定第一步算不算 `grad_diff`，jittor 现在第一步 `grad_diff = g` 而官方实现是 `0`，改偏差修正会碰到它——两件事一起想清楚再改 | 6.P14 | 2026-09-02 由 6.P14 的执行者核实后留下，不在原审计里 | 梯度累积与不累积等价；第一步数值与官方 Adan 一致或明确写出为何不同 |
| 6.P26 | **`MaxPool3d` 的 `ceil_mode` 输出尺寸比 torch 多一个平面**（torch 的规则含「末窗完全落在 padding 内则丢弃」的修正，3D max 路径没有）。5.17 把平均池化统一到 `_avg_pool_nd` 时顺带核实到，但没有一起修：改它会改 `MaxPool3d` 的输出形状，**进而改 `MaxUnpool3d` 的默认 `output_size`**——前向编码的行宽与反解形状是一对，两者必须同改（6.P06 刚修过 unpool 的默认 `output_size`，改动要与那条对齐）。当前状态由 `tests/nn/test_pool3d.py` 里的「已知差异锁」用例钉住 | 6.P06、5.17 | 2026-09-03 由 5.17 的执行者核实后留下，不在原审计里 | 与真 torch 的 `MaxPool3d(ceil_mode=True)` 输出形状一致；`MaxUnpool3d` 往返仍然正确；已知差异锁用例改成正向断言 |

### 9.3 后端库

| 编号 | 任务 | 前置 | 出处 | 验收 |
| --- | --- | --- | --- | --- |
| 6.B01 | MPI 的 int64 改 `MPI_INT64_T`（`mpi_all_reduce_op.cc:85`、`mpi_broadcast_op.cc:70`、`mpi_reduce_op.cc:84`）；MPI/NCCL/HCCL 三份 dtype 表合一 | — | [后端](codebase-audit/06-backends.md)§分布式 | int64 all_reduce 对拍 |
| 6.B02 | ACL：`checkRet` 改抛（`base_op_acl.cc:115-124`）；`run()` 非 group 分支检查 `find()`（`:142-149`）；`CreateAclTensor` 返回真实状态（`utils.cc:124,151`）；`mallocWorkSpace` 失败置 nullptr 并抛，workspace 改走 `exe.temp_allocator`（`acl_jittor.cc:132-156`）；`acl_op_exec.cc:274` 的同名遮蔽 | — | [后端](codebase-audit/06-backends.md)§错误处理、§库句柄 | 65 处 executeOp 的失败都会抛 |
| 6.B03 | HCCL 宏错误时抛而非 return（`hccl_wrapper.h:33-47`）；集合通信失败让所有 rank 快速失败 | — | [后端](codebase-audit/06-backends.md)§错误处理 | 一个 rank 失败其余不挂死 |
| 6.B04 | 分布式一旦被请求，初始化失败硬失败（`compile_extern.py:972-973`、`fsdp2/common.py:20-31`） | — | [后端](codebase-audit/06-backends.md)§分布式 | 4 卡任务不会变成 4 个单卡 |
| 6.B05 | cuBLAS `use_tensorcore` 三目判断写反（`cublas_matmul_op.cc:122,125`、`cublas_batched_matmul_op.cc:150,153`）修正并加 4 种组合单测 | — | [后端](codebase-audit/06-backends.md)§精度 | 开 tensorcore 选 TENSOR_OP |
| 6.B06 | `var_broadcast` 用传入的 root（`mpi_wrapper.cc:271-280`） | — | [后端](codebase-audit/06-backends.md)§分布式 | 非 0 root 广播正确 |
| 6.B07 | cuDNN RNN：`work_space` 初始化为 nullptr（`cudnn_rnn_op.cc:179,226`）；描述符按实际 dtype（`cudnn_rnn_descriptor.h:94,105,135`）；dropout 状态按 (dropout, seed) 缓存跨调用复用（`:47-66`）；`infer_shape` 不创建描述符（`cudnn_rnn_op.cc:91-97`） | — | [后端](codebase-audit/06-backends.md)§库句柄、§精度 | fp16 RNN 对拍；每步 dropout 掩码不同 |
| 6.B08 | cuSPARSE：计算类型与 alpha/beta 用算出的 dtype（`cusparse_spmmcoo_op.cc:49-61`、`cusparse_spmmcsr_op.cc:50-67`）；COO 恢复 bufferSize 查询；每次调用的 `cudaMalloc/cudaFree` 改 `exe.temp_allocator` | — | [后端](codebase-audit/06-backends.md)§精度、§每次调用 | fp64 spmm 对拍 |
| 6.B09 | curand 奇数长度用临时 buffer 不越界写（`curand_random_op.cc:42-48`）；不支持的 dtype 显式拒绝 | — | [后端](codebase-audit/06-backends.md)§精度 | fp16 随机数报清晰错误 |
| 6.B10 | MPI fp16 归约统一标量参考实现加可选 SIMD 与运行期 CPUID 检测（`mpi_wrapper.cc:87-163`） | — | [后端](codebase-audit/06-backends.md)§精度 | x86 与 ARM 同结果 |
| 6.B11 | ACL 六个算子静默把输入升到 fp32（silu/softmax/sigmoid/relu/norms/where）改为声明支持的 dtype，不支持则报错 | — | [后端](codebase-audit/06-backends.md)§ACL | bf16 模型在这些点保持 bf16 |
| 6.B12 | `cutt_transpose_op.cc:77` 的 `cudaGetLastError()` 清错删除；`CUFFT_CALL`/`CUDA_RT_CALL` 改 `checkCudaErrors`（`cufft_utils.h:53-85`） | — | [后端](codebase-audit/06-backends.md)§错误处理 | cufftPlanMany 失败不再写进缓存 |
| 6.B13 | cuFFT `cufftCreate` 后被 `cufftPlanMany` 覆盖的句柄泄漏（`cufft_fft_op.cc:79-80`）；cuTT plan 缓存加容量上限与析构清理（`cutt_transpose_op.cc:66`、`cutt_wrapper.cc:30-32`）；两者键改 POD 并含 device | — | [后端](codebase-audit/06-backends.md)§库句柄 | 形状多样的 FFT 显存有界 |
| 6.B14 | conv3d 三算子迁到 backend plan 缓存；在此之前给 legacy 缓存键加 dtype 与命名空间（`cudnn_conv3d_op.cc:82-89,219-222`）；2D/3D 的 benchmark 判据统一 | — | [后端](codebase-audit/06-backends.md)§每次调用；系统 B1 | conv3d fp32/fp16 不共享算法 |
| 6.B15 | MPI 同时识别 PMI_/SLURM_ 环境变量或要求显式声明（`mpi_wrapper.cc:216`）；rank/world_size 单一来源（C++ 全局、`compile_extern.rank`、`jt.rank`） | — | [后端](codebase-audit/06-backends.md)§分布式 | srun 启动不再静默单卡 |
| 6.B16 | `sync_run` 在 ACL 上实现或删 flag（`base_op_acl.cc:106-113`） | — | [后端](codebase-audit/06-backends.md)§错误处理 | 文档承诺成立 |
| 6.B17 | 析构不得抛：六个 `*_wrapper.cc` 的静态析构改只记录的 peek，句柄销毁改显式 `shutdown()`；`cudnn_rnn_descriptor.h:68,99` 的 RAII 析构不抛 | — | [后端](codebase-audit/06-backends.md)§库句柄；系统 F5 | 退出期无 `terminate called without an active exception` |

## 10. 阶段 7 · Torch 兼容层

热点文件：`compat/torch/installers/*.py`、`compat/torch/__init__.py`、`compat/shim/*`。

| 编号 | 任务 | 前置 | 出处 | 验收 |
| --- | --- | --- | --- | --- |
| 7.01 | 「看起来支持其实空操作」一律改为实现或抛 `NotImplementedError`，需显式 `allow_stub` 才降级：`torch.autocast`（`grad.py:55-77`）、`load_state_dict(strict=)`（`nn.py:1353-1371`）、`torch.load(weights_only=, map_location=)` 与 `find_class` 兜底空类（`serialization.py:275-313`）、DataLoader `num_workers`（`data.py:215-247`）、SummaryWriter（`utilities.py:407-433`）、`tree_map`/`tree_map_only`（`:571-572`）、`dirac_`/`sparse_`、checkpoint、`update_bn`、`has_torch_function`、`opcheck`、`set_default_device`、`Event.elapsed_time`、`Var.backward(gradient=)`（`tensor.py:1059`）、`dist.nn.all_reduce`、dcp save/load、`new_subgroups_by_enumeration`、DeviceMesh、`DTensor.full_tensor`、TCPStore/FileStore、`set_device(i!=0)`（合并多卡前）；每条一个断言抛异常的负向测试 | — | [兼容](codebase-audit/03-compat-shim.md)§看起来支持其实是空操作、§分布式与 FSDP2、§dtype 与 device 映射 | 负向测试集；`tests/compat/torch` 有「未实现 API 清单」自动生成 |
| 7.02 | DDP 真实梯度同步：反向完成点做 all-reduce（hook 加 bucket 加初始广播），`no_sync()` 生效；做不到则构造时抛 `NotImplementedError`（`nn.py:132-137`；同步只在 `optim/base.py:206-215` 的 `opt.step(loss)` 路径） | 8.02 | [兼容](codebase-audit/03-compat-shim.md)§分布式与 FSDP2 | `loss.backward(); opt.step()` 在 2 卡上参数一致 |
| 7.03 | 每个 torch API 一个模块级一等对象加保真度标注（exact/approximate/unimplemented），install 只做绑定；拆掉 `_install_tensor_methods`（1386 行 79 闭包）、`_install_nn_extras`（1003/126）、`_install_module_methods`、`_install_cuda`、data 的 install；自动生成实现程度覆盖表 | 7.01 | [兼容](codebase-audit/03-compat-shim.md)§代码结构与测试、§架构判断 (3) | 任一 torch API 可单独 import 与单测 |
| 7.04 | 激活显式、一次性、可查询：删 `preflight.py:18,101-116` 的 argv 源码嗅探；import 期改写 HOME/TMPDIR/`use_nccl=0`/`--fmad=false` 移到显式启动器（`:347,365,367`，`_add_nvcc_flags`）；三条入口（`torch_init.py:15`、`runtime.enable()`、`jt.flags.torch_shim=1`）收敛为一个 `activate()`；flags 代理（`control.py`）改函数调用，`jt.flags` 保持原生对象 | — | [兼容](codebase-audit/03-compat-shim.md)§部署与激活；[架构](codebase-audit/07-architecture.md)§分层 | 注释里出现 `import jittor as torch` 的纯 Jittor 脚本不被接管；关 FMA 有记录 |
| 7.05 | install 事务化：要么全面可回滚（复用 `module_patcher.restore_method`，覆盖 jittor 模块属性、Var 类字典、`builtins.__import__`、`sys.meta_path`、os.environ、flags），要么明确硬失败；`run_optional` 的失败 warn 一次并可查询；install 加锁与幂等哨兵 | 7.03 | [兼容](codebase-audit/03-compat-shim.md)§install 的事务性 | 中途失败后进程处于已知状态 |
| 7.06 | 依赖单向化 core→tensor→nn/optim→distributed→fsdp→适配器：解开 `tensor.py:1096-1172` ↔ `fsdp2`、`distributed.py:220-223` ↔ `fsdp2.common`、`fsdp2/installer.py:49` ↔ `context` 的环；`nn.py:1237-1242` 与 `shard.py:383-398` 两处同时生效的 hook 合一；optimizers.py 里 31 处 fsdp 引用收口 | 7.03 | [兼容](codebase-audit/03-compat-shim.md)§分布式与 FSDP2 | import-linter 规则通过 |
| 7.07 | 第三方库补丁搬出 compat/：transformers 与 torchmetrics 的 `builtins.__import__` 串接（`utilities.py:61-76,196-209`）与私有函数替换（`:103-191`）改走 module_patcher 的 entry point 并带版本断言 | 7.05 | [兼容](codebase-audit/03-compat-shim.md)§vLLM / shim / 模块补丁的边界 | `builtins.__import__` 未被替换 |
| 7.08 | **⚠ `torch.dtype` 改真对象这一项不许做一半**（2026-09-03 执行中查明）：`class dtype(str)` 的 str 继承是**承重的**——jittor 自己的 Python 会 `str(var.dtype)` 再喂回 C++ 分发，改一半会让**错误的 dtype 静默流进算子**。这是全计划里唯一一条「做一半比不做更糟」的任务：要么连同所有 dtype 入口一次改完并有覆盖 C++ 分发的对拍，要么不动。`torch.backends.*` 的映射表格化已单独完成（六种拼写合成两条状态 + 13 条表驱动单测，其中 `fp32_precision = "bf16"` 被接受、存下、读得回来且不映射到任何东西这条已实测）。26 个占位 dtype 参与计算时报错这一项与 dtype 对象绑定，同样待整体做 | 8.03 | [兼容层](codebase-audit/03-compat-shim.md)§dtype 与 device 映射 | 改完之后 `isinstance(x.dtype, str)` 为 False 且 C++ 分发路径有对拍；未整体完成前保持现状 |
| 7.09 | `torch.library`：dispatch key 按驻留与 dtype 选择、Meta 排除（`library.py:69-70,212-223`）；`register_autograd` 记录的 backward 用 `jt.Function` 真正接入（`:235-236`）；删 `grouped_mm_fallback` 的模型特判（`:176-191`）移到 integrations | — | [兼容](codebase-audit/03-compat-shim.md)§自定义算子 | `impl("CPU")` 后注册 `"CUDA"` 不影响 CPU 张量；自定义反向被调用 |
| 7.10 | `torch.compile`/`jit.trace`/`jit.script` 保留 pass-through 但对 `fullgraph=True` 等语义性参数报错（`compiler.py:34-36,122-123`）；permissive finder 只覆盖已知 import-time 引用清单（`compiler.py:92,104-107`、`permissive.py:27-37`） | — | [兼容](codebase-audit/03-compat-shim.md)§自定义算子 | `from torch.fx.passes… import ShapeProp` 报 ImportError |
| 7.11 | autograd 语义：`needs_input_grad` 按签名归一（`autograd.py:127-131`）；`create_graph` 与 `retain_graph` 分开（`:237,244`）；多输出隐式求和改报错（`:233`）；`_sum_grad_to` 元素数不匹配至少 warn（`:93-94`）；saved_tensors 记版本号（`:62-69`）；`retain_graph` 默认 None（`tensor.py:1059-1062`）；`is_leaf`/`grad_fn` 由内核真实回答（`tensor.py:1260-1267`） | 需要内核的反向可达叶子查询 | [兼容](codebase-audit/03-compat-shim.md)§自定义算子、§张量语义 | 与真 torch 逐项对拍 |
| 7.12 | 独立 torch 包：`torch.Tensor` 持有 `jt.Var`，requires_grad/is_leaf/grad_fn/0 维/视图/存储是它的字段；删 `torch_init.py:16` 的 `sys.modules[__name__] = _jittor` 与 `compat/runtime.py:73`；205 个 `_torch_*` 属性名与 `_torch_index_parent`/`_torch_data_owner`/`_torch_0d`/`_torch_leaf_params`/`_torch_retained` 标记链并入一个 `TorchTensorState`；反向叶子由 requires_grad 加图连通性决定，不再是三个进程级 id 键字典 | 5.02、5.04、2.05、2.09、7.03、7.11 | [兼容](codebase-audit/03-compat-shim.md)§架构判断 (1)(2)；[架构](codebase-audit/07-architecture.md)§公共 API | `torch is jittor` 为 False；`grep _torch_` 于 `python/jittor`（compat 外）为 0 |
| 7.13 | FSDP2：反向不再需要同一 Var 对象后释放 `full_param`（`shard.py:361-366`、`:327` 的 `true_fsdp_flat_full_param`、`grad_sync.py:200`）；`fully_shard(mesh=)` 未支持的 mesh 直接拒绝（`api.py:180-181`、`shard.py:229-231`）；`clip_grad_norm_` 跨 rank 归约（`grad.py:149-162`、`nn.py:197-203`）；复用 jittor optimizer 只替换梯度来源（`fsdp2/optimizer.py:93-172`）；DeviceMesh 真实分组（依赖 8.08） | 7.06、8.08 | [兼容](codebase-audit/03-compat-shim.md)§分布式与 FSDP2 | FSDP 峰值显存低于未分片；自定义 Adam 子类可用 |
| 7.14 | vLLM 边界检查把 `torch` 视作 jittor 别名（`tests/structure/test_vllm_compat_structure.py:83-97`）；`compat/vllm/__init__.py:57,67` 不再改写 `jittor.__version__` | — | [兼容](codebase-audit/03-compat-shim.md)§vLLM / shim | 规则测试对 `torch.__version__ = …` 报红 |
| 7.15 | `_rebuild_tensor_v2` 按 stride 还原或报错（`serialization.py:260-268`） | 5.02 | [兼容](codebase-audit/03-compat-shim.md)§看起来支持其实是空操作 | 非连续视图保存的权重读出正确 |
| 7.16 | compat/ 内 129 个 `except: pass` 与 258 个宽泛 except 限定异常类型并至少 debug 打点 | 7.03 | [兼容](codebase-audit/03-compat-shim.md)§代码结构与测试 | `grep "except: pass"` 于 compat 为 0 |
| 7.17 | `runtime.enable()` 只把 shim 的 site 目录加进 sys.path 不插项目目录（`runtime.py:95-97`）；`_ensure_dir` 的 PermissionError 改明确诊断（`preflight.py:142`） | 7.04 | [兼容](codebase-audit/03-compat-shim.md)§vLLM / shim | 只读 HOME 下 import 给出可操作错误 |
| 7.18 | 布局收尾：`python/jittor/compat/` 搬到顶层 `compat/` 成独立 distribution `jittor-torch`（含 torch/shim/fsdp2/triton；vllm 适配器按既定方向分到自己的仓库）；`compat/shim/cpp_extension/src` 的打包随之迁移 | 7.12、0.19 | [布局](target-layout.md)§3；vLLM 适配器边界（路线图） | `pip install jittor` 不带 compat；`jittor-torch` 单独可装 |

## 11. 阶段 8 · 后端库与分布式（结构性）

| 编号 | 任务 | 前置 | 出处 | 验收 |
| --- | --- | --- | --- | --- |
| 8.01 | 描述符与 workspace 一律 RAII（`cudnn_conv_op.cc:122-125,366,380-386`、RNN 同）；卷积快路径直接由 Var 形状构造 `ConvPlanRequest`，不再建 legacy 描述符再销毁（`:143-246`） | — | [后端](codebase-audit/06-backends.md)§库句柄、§每次调用 | 异常路径 ASan 无泄漏；快路径每调用 CPU 时间 < 10 µs |
| 8.02 | 集合通信走通信流加事件依赖，支持 `GroupStart/End` 桶化；五个 NCCL 算子的默认流参数 0 改流（`nccl_*_op.cc`）；HCCL 每次集合通信的 4 次全设备/流同步删除 | 4.08 | [后端](codebase-audit/06-backends.md)§分布式 | 通信与反向计算重叠的 timeline |
| 8.03 | 精度策略收敛：`use_tensorcore` 0–3 多义编码、`cuda_allow_tf32`、`cuda_allow_cudnn_tf32` 改为 `float32_matmul_precision ∈ {highest, high, medium}`，matmul 与卷积共用；cuBLAS 三份 compute type 选择合一，删或合并 `cublas_acc_matmul` 第三份；默认值作为明确决策写进文档 | 6.B05 | 系统 B4；[后端](codebase-audit/06-backends.md)§精度 | 同一 fp16 矩阵乘在三条路径累加精度一致；shim 的 `set_float32_matmul_precision` 映射有表 |
| 8.04 | cuDNN 9：RNN 迁到 v8 RNN API 后放开 `compile_extern.py:337-341` 的版本拒绝；`jittor[cuda12]` 不再钉死 cuDNN 8.9（`pyproject.toml:37`、`cuda_wheel.py:37`） | 6.B07 | 系统 B2；[构建](codebase-audit/04-build-tooling.md)§安装 | 与现代 torch 共存的环境可安装 |
| 8.05 | MKL：oneDNN 迁 v3 API（`mkl_conv_op.cc:153-156`）并放开版本；primitive/pd/reorder 按形状缓存（`:119-175`）；matmul 支持 fp64/fp16/bf16 或在能力表声明（`mkl_matmul_op.cc:28`）；训练用 `forward_training` | 4.04 | [后端](codebase-audit/06-backends.md)§其余后端、§每次调用 | CPU 卷积每调用开销下降；能力表可查 |
| 8.06 | ACL 去样板：65 处 executeOp 尾巴收进 `BaseOpRunner::launch()`；`AclOpFunctions` 40 个 `std::function` 胖结构改类型擦除的单一 launcher，表改 extern 定义在一个 .cc 或注册宏（`acl_jittor.h:33-347`）；`op_idx_map` 删除（`utils.cc:42-104`）；算子属性走 `jt.code` 的 data 通道不拼进源码（`_code.py:47-59`、`pool_op.py:96-106` 等 30+ 处）；aclTensor 描述符按形状缓存、指针用 `aclSetTensorAddr`；`aclnn.h` 加 pragma once；`acl_op_exec.cc:621-632` 的 "cu" 前缀删算子改注册表 | 6.B02 | [后端](codebase-audit/06-backends.md)§ACL 三件套；[架构](codebase-audit/07-architecture.md)§重复 | `aclops/` 行数下降 ≥ 50%；自适应池化不再无界编译 |
| 8.07 | conv 族共享描述符与计划层（6 个文件的 12 行重复块）；`nn/backends/cudnn.py:8-16` 的 `_CudnnConv2d` 绕过删除（原缺陷已修）并补 CPU 参考对拍 | 6.B14 | [架构](codebase-audit/07-architecture.md)§重复；系统 B3 | 反向定义只有 C++ 一份 |
| 8.08 | `ProcessGroup` 对象替代全局唯一 communicator（`nccl_wrapper.h:45`、`hccl_wrapper.cc:24`）；子组 all-reduce 可用 | 4.02 | [后端](codebase-audit/06-backends.md)§分布式 | DDP 与张量并行共存的用例 |
| 8.09 | NCCL：rendezvous 加超时与失败路径并与 HCCL 版本抽成共享实现（`nccl_wrapper.cc:89-104`）；静态构造器里的 `cudaSetDevice` 加阻塞建通信器改显式 init（`:57,78,104,144`）；通信超时、watchdog、`ncclCommAbort` | 8.08 | [后端](codebase-audit/06-backends.md)§分布式 | 一个 rank 崩溃其余在超时内报错退出 |
| 8.10 | `distributed/launch.py:102-107` 改 `wait(timeout)` 轮询，任一非零退出立刻 kill 全部；每 rank 独立 JIT 缓存（`:90`）改共享缓存加锁或 rank 0 预热 | 0.08 | [后端](codebase-audit/06-backends.md)§分布式 | N 卡任务只编译一次 kernel |
| 8.11 | **图同构优先**（2026-09-03 裁决：原「非 root 不分配输出」与本行验收「跨 rank 图同构」方向相反，已更正）。三项：(a) `nccl_reduce`/`mpi_reduce` 的非 root **保留全尺寸输出、保留清零**，把「非 root 输出无意义」写进算子注释与文档；可选：debug 构建下填毒值而非零，让误用变响而形状不变。**不要**用缩短形状或 `share_with` 去省那块内存——那会让图的结构因 rank 而异，而这正是 (b) 要消除的。(b) `mpi_broadcast` 的 `share_with` 决策移出 `infer_shape`：`infer_shape` 只管形状，root 上原来的 `y->share_with(x)` 改成 jit_run 里 root 分支的一次 memcpy，`y` 永远是新输出，各 rank 的图逐字相同。(c) FSDP2 的两个硬编码经验常数（`world_size<=2 or total_numel<=1_000_000`）改可配置 | 8.09 | [后端](codebase-audit/06-backends.md)§分布式 | 三个 rank 的图逐字相同（可用 jit key 或图结构哈希断言）；非 root 输出的语义在注释与文档里写明 |
| 8.12 | 算子内不再复用全局 jit key 缓冲做缓存键（`cufft_fft_op.cc:72-73`、`cutt_transpose_op.cc:102-103`、`cudnn_conv_op.cc:276-277`、`cudnn_conv3d_op.cc:217-218`），改 POD 哈希 | 3.02 | [后端](codebase-audit/06-backends.md)§每次调用；系统 B6 | 算子执行期无字符串拼接 |
| 8.13 | cuTT 计划未命中时的 `cudaDeviceSynchronize` 删除或降流同步（`cutt_transpose_op.cc:115`） | — | [后端](codebase-audit/06-backends.md)§每次调用 | 首次转置不清空流水 |
| 8.14 | Corex：`check()` 只读、路径可配置（`corex_compiler.py:68,86,88`）；`process_acl` 同名改写随 4.12 删除 | 4.12 | [后端](codebase-audit/06-backends.md)§其余后端 | 探测无副作用 |
| 8.15 | 多机 rendezvous：NCCL unique id 经 TCP store 交换，契约为 `MASTER_ADDR/MASTER_PORT/RANK/WORLD_SIZE/LOCAL_RANK`，替代 `nccl_wrapper.cc:82-101` 的共享文件轮询（要求共享文件系统，120 s 后无失败路径）与只认 OpenMPI 的 mpirun 引导；`TCPStore`/`FileStore` 真实实现（`distributed.py:770-788` 当前是进程内字典，7.01 先改报错）；`init_process_group(init_method="env://" \| "tcp://…")` 生效；HCCL 同一套 | 8.08、6.B15 | [后端](codebase-audit/06-backends.md)§分布式；[兼容](codebase-audit/03-compat-shim.md)§分布式与 FSDP2 | 两台机器各 N 卡 all-reduce 对拍；`MASTER_ADDR` 写错在超时内报错而非挂死 |
| 8.16 | 多机启动器：`jittor.distributed.launch` 今天只有 `-n`（单机，`launch.py:50`），加 `--nnodes/--node_rank/--master_addr/--master_port`；shim 下 `torchrun` 可用（`is_torchelastic_launched` 恒 False，`distributed.py:519`）；`LOCAL_RANK` → 设备映射走 4.02 的 `set_device`，不再改写 `CUDA_VISIBLE_DEVICES` 重启进程 | 4.02、8.15、8.10 | [后端](codebase-audit/06-backends.md)§分布式；[多卡](device-placement.md) | `torchrun --nnodes=2 --nproc_per_node=N` 跑通 transformers 训练脚本，两机 loss 轨迹一致 |
| 8.17 | 跨机网络与诊断：`NCCL_SOCKET_IFNAME`/`NCCL_IB_*`/`HCCL_*` 透传并进启动摘要；跨机 all-reduce/all-gather 带宽微基准；通信超时报出对端 rank 与主机名（接 8.09 的 watchdog）；一个 rank 掉线其余在超时内退出 | 8.09、8.15 | [后端](codebase-audit/06-backends.md)§分布式 | 带宽基准进 nightly；掉线用例 |
| 8.18 | 多机 checkpoint：rank 0 保存、全 rank 加载的 `state_dict` 契约，FSDP 分片 checkpoint 的 `dcp.save/load` 真实实现（接 7.01 的报错），跨 rank 的 optimizer state 合并与重分片 | 7.13、8.15 | [兼容](codebase-audit/03-compat-shim.md)§分布式与 FSDP2 | 2 机保存、4 机加载的续训用例 |
| 8.19 | 布局收尾：`extern/mpi`、`extern/cuda/nccl`、`extern/acl/hccl` 进 `backends/comm/`；Python 侧的启动器、process group、rendezvous 收进 `python/jittor/distributed/`；`compat` 的 `distributed.py` 只做 torch 命名的委托 | 8.08、8.15、4.15、0.19 | [布局](target-layout.md)§3 | 三个通信后端同一目录形状 |

## 12. 阶段 9 · 构建、缓存与打包

| 编号 | 任务 | 前置 | 出处 | 验收 |
| --- | --- | --- | --- | --- |
| 9.01 | `import jittor` 不编译不下载：核心编译与 `setup_nccl/cutt/mkl` 移到显式 bootstrap 或首次算子调用（`compiler.py:1437`、`compile_extern.py:975-985`）；`FIX_TORCH_ERROR` 的无条件 `import torch` 改惰性且不改 `os.RTLD_GLOBAL`（`compile_extern.py:903-914`、`jittor_utils/__init__.py:738`） | 0.09 | [架构](codebase-audit/07-architecture.md)§分层；[构建](codebase-audit/04-build-tooling.md)§首次导入 | 离线只读环境能 import；热缓存 import < 1 s |
| 9.02 | `install_cuda.py:113-122` 的 `os.execl` 自重启删除，用 dlopen 绝对路径替代改 `LD_LIBRARY_PATH`；自动下载 CUDA 工具链默认关闭改显式命令，至少先打印大小并确认（`:187`、`compiler.py:1029-1030`） | — | [构建](codebase-audit/04-build-tooling.md)§首次导入 | shebang 脚本与 MPI rank 内 import 不重启进程 |
| 9.03 | 构建期失败一律抛带上下文的 `RuntimeError`，不用 LOGf/裸 assert（`compile_extern.py:42,263`、`compiler.py:942`） | — | [构建](codebase-audit/04-build-tooling.md)§首次导入 | `python -O` 下检查仍生效 |
| 9.04 | 依赖跟踪改用编译器的 `-MD -MF`（替代手写预处理器 `cache_compile.cc:176-178`，含 `helper_cuda.h` 例外与 `<...>` 不跟踪）；内容哈希换 xxhash64/SHA-256（`misc/hash.h:31-37`）；主机名移出缓存键、`-march=native` 的实际展开入键（`jittor_utils/__init__.py:488`、`compiler.py:1094`）；git 分支移出键改源码内容哈希且不写回 `os.environ`（`:503-519`）；项目路径哈希 ≥12 位（`:495`）；`clean_cache.py` 从同一份布局定义生成 | 0.07 | [构建](codebase-audit/04-build-tooling.md)§缓存键与缓存布局 | 改 `helper_cuda.h` 触发重编；集群节点间缓存可共享 |
| 9.05 | 下载安全：删进程级 TLS 降级（`jittor_utils/__init__.py:765-769`、`compiler.py:1398-1400`）；16 处 `extractall` 加 `filter="data"`；URL/文件名/SHA-256/平台一张 manifest 供 `compile_extern`、`install_cuda`、`install_msvc`、`pack_offline` 共用；MD5 换 SHA-256；删无人用的 `jittor_utils.download`；MD5 不匹配时删坏文件并提示 | — | [构建](codebase-audit/04-build-tooling.md)§第三方二进制下载 | `pack_offline` 产物在 CUDA 机器上真离线 |
| 9.06 | 删 cutlass 下载（`compile_extern.py:977,545-560`，全仓无引用）；NCCL 下载条件前移（`:597-605`）；MKL 安装期不再编译运行上游示例（`:106-107`）改 dlopen 加符号检查 | — | [构建](codebase-audit/04-build-tooling.md)§第三方二进制下载 | CUDA 门禁不再下载 cutlass |
| 9.07 | import 过程不反向写环境变量（`cache_name`、`cc_path`、`LD_LIBRARY_PATH` 无限追加、`NCCL_P2P_DISABLE`、`use_mpi`）；需传子进程的配置显式构造 env；删 `compiler.py:1057-1058` 的坏 `cuda_arch`；flag 组装收进一个函数一次性求值（`:1131,1229-1234`） | 2.22 | [构建](codebase-audit/04-build-tooling.md)§环境变量作为配置 | DataLoader worker 环境与父进程 import 前一致 |
| 9.08 | 新架 GPU：`-gencode arch=compute_X,code=[sm_X,compute_X]` 保留 PTX；`max_arch` 由 nvcc 查询（`compiler.py:1466-1486`）；`install_cuda.py:155` 版本表更新；「will be backward-compatible」的错误警告删除 | — | [构建](codebase-audit/04-build-tooling.md)§安装、§17 个失败点 | Blackwell 上有可 JIT 的 PTX |
| 9.09 | `cuda_wheel` 失败时 LOG.w 出原因，strict 为默认（`cuda_wheel.py:241-273`） | — | [构建](codebase-audit/04-build-tooling.md)§安装 | 组件不匹配可见 |
| 9.10 | 2.0 版本策略：`__version__ = '1.3.11.0'` 与分支名一致，缓存按版本分区 | 0.07 | [构建](codebase-audit/04-build-tooling.md)§安装 | 版本号与分支一致 |
| 9.11 | release 的 platform-validation 阶段跑 selftest（`.github/workflows/release.yml:145-190`）；selftest 扩成 conv+bn+optimizer 三步训练加关键子包 import 清单（当前 60 行只验 `[1,2,3]**2`） | — | [构建](codebase-audit/04-build-tooling.md)§安装；[测试](codebase-audit/05-tests.md)§完全没有测试保护 | 发布一个编不起来的 wheel 会被拦 |
| 9.12 | `extern/rocm/rocm_cache.tar.gz` 的预编译 .o 改从源码构建，或至少给构建脚本与来源说明 | — | [构建](codebase-audit/04-build-tooling.md)§安装 | wheel 内无来源不明的二进制 |
| 9.13 | README 加「首次运行会发生什么」（联网、1–2 GB 缓存、可能十几分钟、可能自动下载 CUDA、git 影响缓存路径）与离线安装说明 | 9.01 | [构建](codebase-audit/04-build-tooling.md)§安装 | 文档 |
| 9.14 | 一次性的构建前置条件检查：编译器、Python 头文件、OpenMP、磁盘空间、网络可达、CUDA 组件版本，一次报告全部缺失并区分可自动修复项（替代 1500 行模块顶层按序拦人） | 9.01 | [构建](codebase-audit/04-build-tooling.md)§17 个失败点 | 17 个失败点中「可操作」由 4 → ≥ 14 |
| 9.15 | noxfile：缓存目录按构建配置指纹命名并跨 session 共享，第三方包本地镜像预置（`noxfile.py:346-363`） | 0.07、9.05 | [构建](codebase-audit/04-build-tooling.md)§门禁 | 门禁不再每次重编核心与重新下载 |
| 9.16 | `agent/scripts/check_repo_layout.sh` 收缩为少数真会复发的检查，已删除路径交给 git 历史（40+ 条黑名单、8 组全树 grep） | — | [构建](codebase-audit/04-build-tooling.md)§门禁 | 脚本 < 100 行且 < 5 s |
| 9.17 | 死代码：Windows MinGW 分支未定义的 `link`（`compiler.py:83`）、`cuda_wheel` 的 Darwin 分支、`env_or_try_find` 重复定义（`compiler.py:949`）、`src/utils/flags.cc`（27 行全注释但被 flag 扫描器读到，导致 12 个 flag 双定义）、`tests/system/legacy` | — | [构建](codebase-audit/04-build-tooling.md)§跨平台与死代码；[架构](codebase-audit/07-architecture.md)§重复 | flag 扫描改预处理后扫描或宏注册 |
| 9.18 | `disable_lock=1` 启用时明确告警并纳入缓存指纹（`lock.py:18`、`src/lock.cc:31`） | 0.08 | [构建](codebase-audit/04-build-tooling.md)§锁与并发 | 告警可见 |
| 9.19 | 布局收尾：`tools/` 只留仓库工具（`gen_pyi`、`local_doc_builder`、`dumpdef`、release、benchmarks 脚本），`agent/scripts` 并入；`python/jittor/tools/` 只留给用户的 nvtx/jtune/tracer；`MANIFEST.in` 改为从 `pyproject` 的 package-data 生成 | 9.01、4.15、5.26、0.19 | [布局](target-layout.md)§3、§5 | `tools/` 与 `python/jittor/tools/` 职责不重叠 |
| 9.20 | **`utils/asm_tuner.py` 的 `pass_asm()` 非原子写 `.s`**：直接 `open(output_path,"w")` 截断再写。多 worker 并发编译同一个 kernel 进同一缓存目录时，有进程读到被截断的汇编，报 `unknown pseudo-op` / `end of file not at end of a line`。在三个互相独立的 `JITTOR_HOME` 上各复现一次，删掉那条缓存后单跑就过——**不是缓存损坏的偶然，是写法本身没有原子性**。改成写临时文件再 `os.replace`（0.07 的产物写入已经是这个模式，照抄）。注意 3.18 计划整条删掉 asm_tuner 链路，但那条周期长，这条是立刻能做的止血 | — | 2026-09-03 由 0.04 的全树运行暴露（`tests/data/test_dataset.py::TestDataset2::test_dataset_use_jittor`，`num_workers=4`） | 该用例在全树并发下稳定通过；写入路径无截断窗口 |
| 9.22 | **并发编译同一个算子时会读到写了一半的 `.so`**：`Cannot open library ...__opkey0_array...op.so: file too short`，且那个 `.so` 的 mtime 正好是报错的同一秒。4 个 dataset worker 并发编同一个 op 时复现。9.20 修的是 `asm_tuner` 写 `.s` 的原子性，这条是**产物 `.so` 本身**——0.07 已经把「写 `<目标>.tmp.<pid>` 再 rename」用在部分产物上，需要确认它覆盖了 JIT 算子的 `.so` 与它的 `.key`，没覆盖就补上。**判据**：报错信息里的 `.so` 事后是完整的、且 mtime == 报错时刻，就是这一类，不要当回归查 | 0.07、9.20 | 2026-09-03 由 5.20 的执行者在 `tests/data` 全量跑中发现 | 4 个 worker 并发编同一个 op 反复跑不再出现 `file too short` |
| 9.23 | **`run_child_script(..., timeout=N)` 在孙进程挂住时不会按时收尾**：`subprocess.run` 的 timeout 只杀直接子进程，而孙进程（如 dataset worker）还攥着 stdout 管道，`communicate()` 于是继续等——实测 `timeout=300` 跑到 10 分钟还在等。helper 需要用进程组（`start_new_session` 加 `os.killpg`）或在超时后主动关管道 | 0.21 | 2026-09-03 由 5.20 的执行者发现 | 构造一个孙进程挂住的用例，helper 在 timeout 内返回 |
| 9.21 | **拆掉 `cache_compile.cc` 里手写预处理器的最后一块**（9.04 只做到「让扫描器认识 `#ifdef`」这一步，手写扫描器仍在）。终态要求把 `process()` 现在兼的两个职责拆开：(a) **`JT_XXX` 宏发现**——扫源码里的 `#ifdef JT_XXX`、若环境里有值就注 `-DJT_XXX=<值>`；这件事**不需要准确的 include 图**，一次廉价文本扫描即可，且必须在编译前完成。(b) **依赖跟踪**——改用编译器的 `-MD -MF`，只能在编译后拿到。拆开之后 (b) 才可能真正换成 depfile。**注意一处循环**：直接给 `-M` 预处理一趟并不能解决 (a)，因为那一趟同样缺 `-DJT_XXX`，`#ifdef` 块照样看不见——所以必须是「(a) 独立的廉价扫描」而不是「(a) 复用 (b) 的预处理结果」。另需处理：`asm_tuner.py`/`dlink_compiler.py` 两个按字符串重写命令行的 wrapper（`run_and_install` 已为它们开了例外），以及 MSVC 无 `-MF`（只有 `/showIncludes`）要单独分支 | 9.04 | 2026-09-03 由 9.04 的执行者拆解出来；审计原文只写了 remedy「改用 `-MD -MF`」，没有看到 `process()` 的双职责 | 仓库内无手写 include 扫描器；改任意被包含的头（含 `<...>` 与条件包含）都触发重编 |

## 13. 阶段 10 · 测试体系补全

| 编号 | 任务 | 前置 | 出处 | 验收 |
| --- | --- | --- | --- | --- |
| 10.01 | `tools/run_test_suite.py` 拆成 `nox -s full` 周期性调度并入 CI，否则删除并在文档承认没有全量口径 | 0.15 | [测试](codebase-audit/05-tests.md)§门禁范围 | 全量口径有自动运行 |
| 10.02 | 默认 `nox` 含 cpu 数值测试，或把默认改名为 static（`noxfile.py:294`） | 0.15 | [测试](codebase-audit/05-tests.md)§门禁范围 | 本地 nox 全绿意味着能算对 |
| 10.03 | optional/rocm/mpi/nccl 四个 session 排上 runner 或在文档标注手动；CUDA 门禁对 labeled PR 触发（`cuda.yml:3`） | — | [测试](codebase-audit/05-tests.md)§门禁范围 | 文档与 workflow 一致 |
| 10.04 | 假绿清理：6 个首行 `return` 的用例改 `expectedFailure` 并登记，加禁止首行 return 的静态检查；4 个 `skipIf(True)` 中的 2 个内存泄漏测试改短循环加 RSS 阈值放 nightly | 0.18 | [测试](codebase-audit/05-tests.md)§跳过条件与假绿 | 内存释放契约有覆盖 |
| 10.05 | 按 skip 原因分桶统计并在 CI summary 输出，对「本环境应能跑却 skip」设阈值 | 0.15 | [测试](codebase-audit/05-tests.md)§跳过条件与假绿 | summary 有分桶 |
| 10.06 | `expect_error` 带 `exc_type` 与 `match`（61 处）；OpInfo 增加 `error_inputs_func` | — | [测试](codebase-audit/05-tests.md)§跳过条件与假绿 | 错误路径覆盖 > 15% |
| 10.07 | Unary/Binary/Reduction 用 `OpDTypes.supported`（`test_ops.py:59`）；bf16 用原生构造不再替换成 fp16（`_helpers/common.py:216-217`） | 0.01 | [测试](codebase-audit/05-tests.md)§算子证据链 | 声明的 dtype 都跑到 |
| 10.08 | 已复现缺陷用 `xfail` 而非 `skip`（definitions/ 下 9 处 skip）；skip 理由必须写 nodeid；63 个 `supports_autograd=False` 各指向 KI 编号或数学理由（fft/ifft/rfft 应可导） | 0.01 | [测试](codebase-audit/05-tests.md)§算子证据链 | `xfail(` 命中 > 0；XPASS 强制清理 |
| 10.09 | 公开 API 与 OpInfo 差集作为 structure 门禁一项；补 setitem/index_put/nonzero/unique/bincount/einsum/ctc_loss/rms_norm/rope/paged_attention/fused_moe/conv_transpose3d 的 OpInfo | 0.01 | [测试](codebase-audit/05-tests.md)§算子证据链 | 差集清单归零或每项有理由 |
| 10.10 | gradcheck 加「故意写错导数应当失败」的负向自测（`_helpers/gradcheck.py`） | 0.01 | [测试](codebase-audit/05-tests.md)§算子证据链 | 负向测试 |
| 10.11 | 设备对拍加 dtype 轴（int8/int16，`test_device_parity.py:192`）；容差按 `sqrt(reduce_size)*eps` 缩放或 per-OpInfo（`:159-163`）；linalg 的 cupy 探针失败改 error（`:63-89`） | 0.16 | [测试](codebase-audit/05-tests.md)§对拍口径与容差 | 窄整数 reduce 有对拍 |
| 10.12 | `retry` 装饰器记录并上报重试次数（`_helpers/retry.py`） | — | [测试](codebase-audit/05-tests.md)§测试自身的可靠性 | 不稳定率可观测 |
| 10.13 | marker 真正建立 `-m "not slow"` 快门禁或删除（`conftest.py:113-150` 打了 marker 而 `noxfile.py:410` 从不传 `-m`）；结构测试本身的压缩已提前为 0.19 | 0.15、0.19 | [测试](codebase-audit/05-tests.md)§tests/structure 的成本 | PR 门禁按 marker 筛选 |
| 10.14 | notebook 门禁按 topic 参数化；34 个 skip-execution 标签需写理由并设比例上限 | — | [测试](codebase-audit/05-tests.md)§完全没有测试保护 | 每 topic 独立结果 |
| 10.15 | 速度 harness 记录并断言两侧线程数、亲和掩码与精度策略；重复次数默认 ≥ 10 | 0.05 | 系统 F2 | 报告含这三项 |
| 10.16 | 提供计时 API：固定输入池、全量物化、剔除首编译；文档列出 CSE/死码消除/未物化三个陷阱 | — | 系统 F4 | `jt.benchmark` 或等价物 |
| 10.17 | 异步错误：低开销环形缓冲记录最近发射算子的 Python 位置，CUDA 异步错误报出来源 | 3.01 | 系统 F5 | 错误信息含算子的 Python 行号 |
| 10.18 | 结构测试预算转向核心：图不变量、liveness、执行器计划的属性测试（tests/core 7355 行对核心 37.5k 行） | 2.10、3.01 | [架构](codebase-audit/07-architecture.md)§代码规模；[测试](codebase-audit/05-tests.md)§tests/structure | 核心属性测试进 CPU 门禁 |
| 10.19 | 每个带 `grad()` 的后端算子有对 CPU 参考的梯度单测（`test_cudnn_conv_plan` 已示范） | 0.01 | 系统 B3 | 清单齐全 |
| 10.20 | 给测试提供受支持的内省 API，替代 283 处 `jt.flags.*`、137 处 `compile_extern`/`jt.compiler.*`、127 个文件触碰下划线名 | 2.13、4.05 | [架构](codebase-audit/07-architecture.md)§公共 API | 内部重构不再牵动大面积测试 |
| 10.21 | import 方向做成 lint 规则（import-linter 或 ruff 插件）；mypy 覆盖从 7 个文件扩到核心包（`pyproject.toml:70-91`） | 4.07、7.06 | [架构](codebase-audit/07-architecture.md)§代码规模 | 三个真环在 lint 里报错 |
| 10.22 | 多机门禁：两节点 smoke（需两台带 GPU 的 runner）跑 DDP 与 FSDP 各一个小模型并对拍单机结果；无硬件时单机多进程模拟 `world_size > 本机卡数` 的 rendezvous、超时与失败传播路径 | 8.15、8.16、8.17 | [测试](codebase-audit/05-tests.md)§门禁范围 | nightly 有多机条目；掉线用例在超时内结束 |
| 10.23 | 布局收尾：`tests/` 内部镜像源码目录（`tests/core` ↔ `src/core`，`tests/codegen` ↔ `src/codegen`，`tests/backends/<name>` ↔ `backends/<name>`，`tests/compat` 随 `compat/` 分出去）；`tests/system/` 删除 | 0.19、4.15、7.18 | [布局](target-layout.md)§2 | 每个源码目录有同名测试目录 |

## 14. 阶段 11 · 清理与删除（相关重构合并之后）

各阶段的「布局收尾」行（0.20、1.05、2.23、3.24、4.15、5.26、7.18、8.19、9.19、10.23）已经覆盖搬动；这里只剩删除与合并。

| 编号 | 任务 | 前置 | 出处 | 验收 |
| --- | --- | --- | --- | --- |
| 11.01 | 删已被取代的绕过与死路径：`nn/backends/cudnn.py` 的 `_CudnnConv2d`（8.07）、`var_holder.cc` 十层写回（5.02）、`change_function()`（4.11）、`process_jittor_source`（4.12）、`asm_tuner`（3.18）、event_queue 死代码（3.19） | 各自前置 | 各自出处 | grep 为 0 |
| 11.02 | 已提前为 0.20 | | | |
| 11.03 | 单文件异常拆分：`misc/tensor_ops.py` 2874、`_runtime/core_api.py` 2614、`installers/nn.py` 2454、`installers/tensor.py` 2413、`flash_attention.py` 2086、`compiler.py` 1500、`acl_compiler.py` 1397、`op_compiler.cc` 1171、`opt/expr.cc` 1180 按域下推一层 | 5.x、7.03 | [架构](codebase-audit/07-architecture.md)§代码规模 | 无 > 1500 行的源文件 |
| 11.04 | 关键接口写成显式契约（`executor.h` 40 行对 744 行实现、`allocator.h` 58 行下挂 8 个实现、compat 28k 行无契约）；`Installer`/`Backend` 协议类型统一 19 处 install 与 7 处 check 的签名 | 3.01、4.03、7.03 | [架构](codebase-audit/07-architecture.md)§代码规模、§重复 | 接口文件承载设计意图 |

## 15. 并行分区与热点文件

同一时间一个热点文件只由一个 agent 改。分区建议：

| 分区 | 文件 | 阶段 |
| --- | --- | --- |
| 核心节点 | `node.h`、`var.*`、`op.*`、`var_holder.*`、`grad.cc`、`graph.*` | 2 |
| 执行器 | `executor.*`、`fused_op.*`、`parallel_compiler.cc`、`jit_key.*` | 3 |
| 代码生成 | `op_compiler.cc`、`opt/kernel_ir.*`、`opt/pass/*`、`opt/tuner/*`、`opt/pass_manager.*` | 1、3 |
| 类型与日志 | `misc/nano_string.*`、`utils/log.*`、`init.cc` | 2 |
| 内存 | `mem/*`、`misc/cuda_flags.cc` | 4、6.C |
| 绑定 | `pyjt/*`、`pyjt_compiler.py`、`py_converter.h` | 6.C |
| Python 核心 | `_runtime/core_api.py`、`__init__.py`、`misc/indexing.py` | 5 |
| Python 算子 | `misc/tensor_ops.py`、`nn/functional/*`、`nn/modules/*`、`pool/*` | 5、6.P |
| Python 其他 | `optim/*`、`distributions.py`、`dataset/*`、`linalg.py`、`fft/*`、`sparse/*`、`models/*` | 5.18、6.P |
| 兼容层 | `compat/**` | 7 |
| CUDA 后端 | `extern/cuda/**` | 6.B、8 |
| ACL/ROCm/Corex | `extern/acl/**`、`extern/rocm/**`、`extern/corex/**` | 4.11、4.12、8.06 |
| 分布式 | `extern/mpi/**`、`extern/nccl/**`、`extern/hccl/**`、`distributed/*` | 6.B、8 |
| 构建 | `compiler.py`、`compile_extern.py`、`jittor_utils/*`、`install_cuda.py`、`cuda_wheel.py` | 0、9 |
| 门禁 | `noxfile.py`、`tests/conftest.py`、`tests/_helpers/*`、`.github/workflows/*` | 0、10 |

跨分区的任务（2.13 Runtime 对象、4.02 多卡合并、5.02 视图模型、7.12 独立 torch 包）每次只能有一个在进行，
并且开始前其它分区的进行中任务要先合并。
