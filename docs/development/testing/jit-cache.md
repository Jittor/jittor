# JIT 缓存与测试耗时诊断

[本栏目目录](../index.md) · [English](jit-cache.en.md)

本页用于排查“同一条测试为什么突然变慢”。前提是已按[测试体系](test-system.md)
选择范围，并能区分核心引导、外部库构建与测试执行。先确认缓存是否命中，再判断测试是否过重。

## 排查顺序

1. 记录提交、解释器、编译器、后端选择、编译选项、`JITTOR_HOME`、`cache_name`
   和实际 `cache_path`。同一台机器不等于同一份缓存。
2. 看慢段在核心 import、库加载、kernel 编译还是计算中。保存编译日志及子进程信息，
   不把等待构建锁直接判为计算卡死。
3. 必要时按原命令、原配置串行重跑一次，以冷/热差异判断编译成本；比较两轮的
   collected/executed、passed、skipped 和失败节点，确保工作量相同。
4. 只有确认热缓存后，才用 `--durations` 决定哪些测试需要优化或移出快速层。
5. 长运行使用 `-v` 留下逐条结果，并检查[会话完成哨兵](test-isolation.md)。
   超时退出后的不完整日志不能作为通过证据。

状态和原始日志放在 `$JITTOR_LAB_ROOT/_state/<topic>/<run>/`，不写入源码树。
测试与 benchmark 使用不同缓存。新增 JIT 运算或扩展的首次构建串行完成；
独立并行任务使用不同的 `JITTOR_HOME` 或 `cache_name`。

## 冷耗时归属与预热范围

首次触达一个 kernel 或库的用例会承担其编译成本，因此冷缓存的耗时排名受收集顺序影响。
它不能直接说明该测试自身的计算成本，也不能据此断言某个目录不存在慢测试。

融合 kernel 的 key 取决于具体子图：算子顺序、dtype、秩、stride 和广播形态等。
预热常见算子不能替代实际工作负载的融合图；库 `code` kernel 与单算子 kernel 则可能复用。

历史的一次 `tests/nn` 测量产生 604 个 JIT kernel，其中 436 个融合 kernel（72%）、
111 个库 kernel 和 57 个单算子 kernel。同一套用例冷/热耗时为 37:03 / 1:02，
结果均为 360 passed、71 skipped、948 subtests。这些数字说明归因方法，
**不是当前提交的性能基线，也不是任意模型的预热比例**。

## 缓存身份：编译器探测不能依赖 locale

`archXXXXXXXXXX` 来自 `cc -march=native -Q --help=target` 展开的目标指令集。
输出中的 enabled/disabled 状态可能被 gettext 翻译；直接哈希本地化文本会让同一台机器
因语言环境不同产生两棵缓存树。

维护中的约束是：

- 探测通过 `jittor_utils.c_locale_environment()` 固定为 C locale；调用者无需自行统一语言。
- memo 槽位使用 `target_arch:c-locale:<cc>`，避免复用旧 `target_arch:<cc>` 中的翻译文本。
- 修改探测方式时必须同时考虑已有 `probe.json`；只改新探测不能使旧 memo 自动失效。
- 其它 key 输入应是稳定的版本号、架构和 flag，避免加入本地化的自由文本。

定向回归为 `tests/build/test_build_config_cache.py` 中的
`test_the_instruction_set_key_does_not_depend_on_the_message_locale`。
诊断时对比实际 `arch*` 路径和探测槽位，不通过删除全部缓存掩盖身份问题。

## 外部库构建单独计时

`mkl`（oneDNN v3）是惰性加载的；仅 `import jittor` 或运行一次简单加法，
不能证明 CPU matmul/卷积所需的库已经构建。源码构建可能包含约 1200 个翻译单元。

oneDNN 构建并行度按核数和内存约束选择，默认上限 32；`JT_ONEDNN_BUILD_JOBS`
可显式覆盖。在资源较少的 runner 上不要照搬大机器的并行数。
历史同机隔离 A/B 中，并行数 4 的两次构建为 443/448 秒，32 为 87/87 秒；
这些是旧环境的观测，不能直接承诺新机器的耗时。

需要诊断库加载成本时，用 `probe_library(..., load=True)` 单独预热并计时。
这只把工作放到可归因、可设置超时的阶段，不减少总工作量。
**不要把 MPI/NCCL/HCCL 放进无其它 rank 的通用预热**：加载器可能初始化通信器并等待其它进程。

## 磁盘命中与构建锁

`Op::jit_run` 先查进程内 `jit_ops`；未命中才进入 `OpCompiler::do_compile`。
即使磁盘缓存完全热，新进程首次使用一个 key 时仍要经过后者。

磁盘命中走免锁路径：根据当前命令和输入算出 key，比较产物旁的 `<product>.key`，
匹配后直接 `dlopen`。真正构建在 `jit_compiler::compile()` 中持锁。
生成源码是内存输入，命中路径直接哈希字符串，不能为了检查命中而写入源码文件。

这条快路径依赖发布顺序：`cache_compile()` 先把完整产物原子 rename 到位，
再原子发布 `.key`，两者均在同目录操作。因此匹配的新 key 不得对应旧产物或半成品。
`tests/build/test_jit_cache_concurrency.py` 通过外部采样慢速重建和多进程重叠 kernel
集合检查发布顺序与数值一致性。这类定向并发测试不等于允许日常任务争用冷缓存。

验证锁行为时，先完成核心 import 和首次 `cutt` 等库加载：它们可能走另一条锁路径。
否则“持锁时命中慢”的测量可能实际是在等待引导或库构建。
Python 侧 `compile_custom_ops`、`compile_module`、`download_url_to_local` 的
`lock_scope` 仍可能覆盖整个调用，包括命中；它们与逐 JIT key 的快路径应分别分析。

## 并行与缓存复用的边界

已预热的相同配置可在运行器管理的 worker 间共享；真正需要编译时仍会竞争全局锁。
进程内并行编译由 `use_parallel_op_compiler` 控制，增加 pytest 进程不能消除构建锁。
独立任务仍按仓库协作规则隔离缓存，避免源码、环境和编译生命周期互相影响。

不要把硬链接复制热缓存当作默认的 worker 初始化方案。新 home 的 `jit_utils`
引导可能重建并要求重启（历史路径返回码 3），还可能使部分播种的 kernel 失效。
优先使用运行器管理的缓存，并明确记录源码和配置身份。

本页由原测试体系中的缓存调查整理而来。完整旧测量与环境语境见
[历史缓存章节](https://github.com/Jittor/jittor/blob/a3783a04a0fdf921abfe1e3c0d7c93256ee70bd3/docs/development/test-system.md)。
新的测量写入 GitHub issue/PR；改变缓存 key、锁或发布顺序后须重新验证，不能沿用历史速度结论。
