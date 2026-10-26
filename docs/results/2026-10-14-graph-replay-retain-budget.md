# 自动重放预算与 no-grad 边界

Status: 动态预算恢复已在 CPU 与 CUDA 上验证；默认策略未变。CUDA graph 相关回归通过。
通用 CPU 后端在既有回归中保留一个宽中间张量，改动前后完全相同；本报告不接受其 eager
规模保留的声明。

Baseline: `1e2ff1aa6b3a391c0c84747413c01bc2731e5626`，在拉取并整合
`origin/2.0-refactor` 的 `7ed9b3b52b658d2a3c1ac390eba09087c5e80f95` 之后。
实现：包含本报告的提交。维护者：Jittor 核心维护者。复查条件：自动重放资格、工作集
计量、记录生命周期、分配器策略或后端捕获支持发生变化。

## 现有策略与已验证的缺陷

TODO 中的 default/no-grad 策略已经存在：`61b4cb1db` 把默认记录预算从 64 提到
256 MiB；`29e5998b0` 把计量从累计分配改为峰值工作集。自动重放仍限制在重复的、最外层
`no_grad` 模块调用，且 Var 输入最多 4 MiB。超过记录预算时仍保留 executor 重放可用。
零表示无上限，显式 `jt.graph_replay` 没有自动预算。

最小的 CPU 与 CUDA 探针在默认预算下预热一个 1,024 字节的图，然后进入
`jt.flag_scope(auto_graph_replay_retain_bytes=1)`。缓存的 wrapper 保留了其 256 MiB
预算；CUDA 继续启动同一个已记录的图。有状态对象路径也保留了旧预算。新的回归测试在
修复前对这两个精确用例失败。

自动策略现在在下一次符合条件的调用时检测预算变化，通过当前 owner 释放每个现有
plain/stateful 捕获，并按新预算捕获。恢复作用域的旧预算也会重新捕获。测试比较变化
输入的输出，断言旧捕获被释放且记录被拒绝，并证明缓存更新恰好发生一次。独立的守卫
证明，预热后的推理捕获不会替换可微调用，且丢弃其模块会释放捕获。默认值与显式重放
语义不变。

## 测量与限制

使用 Native Jittor 2.0.0、Python 3.11.16、NumPy 1.26.4、GCC 11.5 以及一块
GeForce RTX 4090（计算能力 8.9，驱动 580.178.04）。CUDA 编译使用 nvcc 12.2.140
与现有 pip-cu122 库。CPU 显式禁用 CUDA；此环境缺少 oneDNN，所以 CPU matmul 使用
通用 kernel。检查了真实 CUDA 参数放置与非零 device-graph 句柄。数值验收运行禁止
后端回退，观察到零回退尝试。

测试与基准使用独立的 CPU/CUDA 缓存，首次编译串行。计时使用 `jt.benchmark`、12 个
物化的轮换输入、8 次预热与 25 次同步采样。每个表值是三个独立进程中位数的中位数，
基线/当前顺序交替。基线策略是从基线提交复制的字节一致副本，SHA-256 为
`fcb754d1dfe5e39683a994e6707c766a251a3c487b688477451a92a7f8b147ca`；只有那个
Python 策略 owner 不同，产物记录两个源码哈希。实测 native 数学策略为 `default`；
输入/输出一致性在计时之外对照每个后端的 eager 执行检查。

| 小 8x8 matmul/tanh 调用 | 基线，微秒 | 当前，微秒 |
| --- | ---: | ---: |
| CPU，禁用重放对照 | 34.33 | 32.88 |
| CPU，默认重放 | 31.01 | 30.87 |
| CUDA，禁用重放对照 | 35.65 | 35.41 |
| CUDA，默认重放 | 45.41 | 43.84 |

这些小行不能确立整体的 4% 改进。CUDA 默认重放在小行上比 eager 慢；预算改动修复策略
正确性，并不为每个工作负载选择最优执行模式。

宽 CUDA 调用有 256 字节输入、两个 32 MiB 权重、四个连续的 32 MiB 激活、storage
视图以及一个 8x8 输出。每个输出都用变化的固定输入与 eager 对照检查：

| 记录策略 | 微秒 | 调用后额外活动 GPU 池字节 | 额外 GPU 池峰值 |
| --- | ---: | ---: | ---: |
| 禁用重放 | 486.45 | 0 | 64 MiB |
| 16 MiB 预算：executor 重放 | 482.57 | 1 KiB | 64 MiB + 2 KiB |
| 64 MiB 预算：记录重放 | 451.72 | 64 MiB + 1.5 KiB | 64 MiB + 2 KiB |
| 默认 256 MiB：记录重放 | 452.79 | 64 MiB + 1.5 KiB | 64 MiB + 2 KiB |
| 无上限：记录重放 | 452.13 | 64 MiB + 1.5 KiB | 64 MiB + 2 KiB |

估计的记录工作集恰好为 64 MiB。私有输入与输出缓冲区在该估计之外增加固定开销。所有
策略在模块被丢弃时释放额外活动捕获存储；记录用例报告 41 次 graph 启动，16 MiB 用例
在 CUDA 上重放时明确拒绝记录。这些是活动池测量，不是 NVML 进程峰值，也不承诺进程
总显存被封顶。此工作负载没有提供提高或降低现有默认值的理由。

CPU 宽回归在权重之外保留 33,555,456 字节。改动前相同的策略以相同字节数失败同一个
小于 1 MiB 的断言。独立生命周期探针确认删除模块会释放它并把额外活动池字节归零。
该失败被保留，未标记 xfail 或削弱；通用 CPU 中间张量回收需要单独修复。

## 回归结果

- 新边界用例：CPU 四 passed/四 CUDA skips；CUDA 八 passed，无 skip，回退计数为零。
  修复前，两个动态预算 CPU 用例均失败。
- 五个相邻的 graph replay/capture/step 文件：CUDA 103 passed 且零回退；CPU 43 passed、
  59 accelerator skips，以及上述未变的宽保留失败。
- CPU `core` 白名单：native 98 passed、41 skipped、一个既有 xfail；Torch 224 passed、
  一个 accelerator skip。这些在隔离的热测试缓存中运行 `tests/_helpers/tiers.py` 的
  白名单，而非标准 runner 单独提供的冷会话。

Repository layout 与 `git diff --check` 通过。完整 `tests/structure`：1,426 passed、
4 skipped 与 4 failed，263 秒。失败是隔离环境中 ASV 不可用、既有 ACL 梯度清单
不一致、未登记的结果 toctree 页面以及生成的打包 manifest。共享 index/manifest 文件
不在本工作树的编辑所有权内，需要集成 owner 更新。这不是全绿结构声明；原始输出为
`logs/structure.log`。

## 复现与证据

使用可用的 native 环境，将 `PYTHONPATH` 固定到 checkout 的 `python`，并为每个后端
隔离 `HOME`、`JITTOR_HOME` 与 `TMPDIR`。CPU 使用 `JT_USE_CUDA=0 JT_BUILD_NVCC_PATH=''`；
CUDA 选择一张已分配的 GPU 并设置 `JT_USE_CUDA=1 JT_BUILD_NVCC_PATH=<nvcc>`。运行：

```bash
JITTOR_TORCH_SHIM=0 python -m pytest -q --rootdir=. \
  tests/core/test_graph_replay_retention.py \
  tests/core/test_graph_replay.py tests/core/test_graph_capture.py \
  tests/core/test_step_capture.py tests/nn/test_graph_replay_multi_output.py
```

改动前后探针、精确基线策略副本、测量脚本、原始样本与日志未版本化地位于
`$JITTOR_LAB_ROOT/_state/r19-graph-replay/`。关键证据：`logs/baseline-{cpu,cuda}.log`、
`logs/negative-budget-cpu.log`、`logs/positive-boundaries-{cpu,cuda}.log`、
`logs/graph-neighbors-{cpu,cuda}.log`、`logs/baseline-wide-cpu.log`、
`logs/wide-cpu-lifetime.log` 与 `artifacts/*.json`。基线与当前输出在每个计时进程中
独立检查；这是内部策略比较，不是 PyTorch 性能声明。
