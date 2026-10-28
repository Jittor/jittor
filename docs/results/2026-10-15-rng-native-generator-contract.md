# 原生 RNG seed 与显式 generator 契约

- 状态：以下列出的 CPU 与 CUDA 路径已验证；不做跨框架样本逐位一致声明。
- 复查日期：2026-10-15。
- Owner：Jittor runtime 与 Torch 兼容维护者。
- 初始复现基线：`4a4a480e628dae634a9d660f320c3aa957341e99`。
- 最终整合基线：`b2aeec10874136c1b4b16c1598241252a5f92c87` 加本次改动。`a5932248fdb06fb52293f5ddd8a357f4920195ff` 上的中间 CPU core 与 CUDA 结果保留其原始源码封存。同步保留了每个 owned 文件；最终设备与 CPU 检查在同步后的树上重新运行。
- 复查条件：seed 绑定、原生 generator engine、CUDA stream 预留、tensor factory placement 或 checkpoint RNG ownership 变化。

## 已确认的失败与最终行为

独立 binary PyTorch 2.11.0+cu128 接受 seed 区间 `[-2**63, 2**64-1]`，
把负 seed 映射进无符号 64 位 metadata，在不改动状态的情况下
拒绝相邻的两个越界值，并从 `manual_seed` 返回其默认 generator。
原 Jittor 前端通过 C++ `int` 绑定收窄了合法的大值；
`-1` 在状态恢复后可能改变其上报身份。仅 CUDA 的 seeding 还通过一个
便利函数重新 seed CPU/Python/NumPy。这些是执行结果，
不是从先前 PR 审计推断出的结论。

显式 shim generator 此前从 `numpy.random.Generator` 抽样，
即使其最终张量被拷到 CUDA 时也是如此。旧探针记录了
实际的 NumPy 调用。其 CUDA 状态张量也位于 CUDA 上。CUDA 设备上的输出指针
并不能确立 CUDA 抽样 provider。

前端现在在改动前校验 Torch seed 域，把全宽全局 seed
委派给既有原生 CPU owner 与 CUDA 重置回调，并返回
`torch.default_generator`。仅 CUDA 的 setter 保持 CPU/Python/NumPy 状态。
原生 CUDA 默认值为每设备保留一个 seed 与 cursor；legacy version-1
有符号 seed 的 CUDA checkpoint 仍可读，version 2 存储无符号 seed 位。

显式 generator 包裹既有原生 `jt.Generator`：CPU 上使用原生 CPU
engine，CUDA 上使用既有的基于计数器的 CUDA Philox kernel。
受支持的 factory 通过 random op 传递实际原生 handle；不增加
第二套 PRNG 或 NumPy 抽样算法。其不透明状态是一维
CPU ByteTensor。挂起的 CPU 抽样在捕获或重新 seed 前完成。
外来/损坏状态与设备不匹配在改动 owned 流之前失败。
既有原生状态解析器现在还会在提交替换 engine 前拒绝
尾随字节。

CUDA factory 回归还暴露了 `normal(Tensor mean, scalar std, generator=...)`
选择默认 CPU placement。factory 现在从 Tensor mean 或 std 推导
placement，包括关键字参数。两种形式都在其请求的原生
CUDA stream 上执行。

一个双设备 checkpoint 复现用 `2**64-1` 与 `2**63+17` 为设备
seed。原生 checkpoint 消费者把当前设备的 seed 拷进两个快照条目，
恢复了错误的第二个流。它现在按设备查询 seed。
第二个最小复现确认，包含有符号 `-1` 的旧原生 checkpoint tuple
无法再通过无符号恢复绑定。消费者现在校验 64 位区间，
并映射合法的 legacy 有符号位而不掩码溢出。legacy 续训与
非法边界用例在双设备测试中执行。checkpoint layout 与
FSDP/优化器数学不变。

## 验证

首次 JIT 与扩展编译串行执行。CPU 与 CUDA 有独立的
任务自有 home/缓存，core runner 分开 native 与 Torch 进程
缓存。CUDA 使用既有的严格 CUDA 12.2 wheel/工具链指纹
`pipcu122_a89fa4fed08c`，禁止后端 fallback。
实际分配在 host 读取前用 CUDA driver 设备 ordinal 查询检查。

| 范围 | 结果 | 证据 |
| --- | --- | --- |
| Torch seed/provider/state CUDA 矩阵，含两个物理设备与原生 checkpoint 消费者 | 96 passed，0 skipped | `logs/final-cuda-session-checked.log`, `final-cuda-session-checked.json` |
| 原生 Generator 与 RandomOp CPU/CUDA，配独立 Torch 2.13.0+cu130 统计参考 | 35 passed，0 skipped | `logs/final-native-session.log`, `final-native-session.json` |
| 独立 Torch 2.11 / 2.13 与修复后 shim 的 CPU seed API 比较 | 18 个 seed/state/error 行一致；10 个显式原生 handle 原语调用 | `cpu-api-comparison.json`, `reference-cpu.json`, `reference213-cpu.json`, `candidate-cpu-after.json` |
| 原生 checkpoint 双设备反例与修复后恢复 | 之前：第二个 seed 被错误复制；之后：两者精确，fallback 0 | `checkpoint-per-device-{before,after}.json` |
| Legacy 原生 checkpoint 负 seed | 之前：无符号绑定溢出；之后：无符号 metadata 与精确重放，fallback 0 | `checkpoint-legacy-{before,after}.json` |
| CPU Torch RNG 矩阵 | 53 passed，28 个加速器 skip；fallback 0 | `logs/final-cpu-session-retry.log`, `final-cpu-session.json` |
| 原生 CPU checkpoint 套件 | 30 passed，0 skipped；fallback 0 | `logs/final-checkpoint-cpu-session.log`, `final-checkpoint-cpu-session.json` |
| 最终计算/测试源码 structure | 添加本报告前 1486 passed，6 skipped | `logs/structure.log` |
| 中间基线下的 CPU core | 322 passed，42 skipped，1 xfailed | `logs/core.log` |
| 布局与空白 | 通过 | `logs/layout.log`；`git diff --check` |
| 针对原始受影响源码基线的 Ruff | 0 个新诊断；既有总数 132 到 131 | `ruff-comparison.json` |

CUDA 矩阵包含全宽正负 seed、非法 seed 原子性、default-generator
身份、Python/NumPy/全局流隔离、延迟抽样捕获、新 generator
恢复、pickling、损坏状态拒绝、设备不匹配拒绝，以及
12 种 factory 形式的实际原生 handle。`randint` 在一个小的
可表示区间上执行。七个 CUDA 扩展行覆盖混合原生/扩展续训、
预留取整、非法增量、无符号全宽 seed 位与恢复后的精确重放。

对两个设备，测试检查不同的 seed metadata、CPU ByteTensor
快照、真实分配 ordinal、未目标设备上的状态不变、逐设备精确
续训、全设备恢复、改动前拒绝畸形多状态、独立的显式
generator 状态，以及原生 checkpoint 快照/恢复。这是单主机
设备隔离，不是多主机结果。

更早的失败尝试均保留：私有解释器 bootstrap/import 顺序、
observer 挂在 Python 包而非实际核心 random 原语上、旧的错误预期
（`manual_seed` 返回 Torch 模块）、bare-CUDA 构造器的 `max`
被遮蔽、Tensor-normal placement 失败，以及扩展编译前缺少私有
pybind11 头文件。没有任何失败被转成宽松断言。
旧的共享 Torch-ref 安装随后缺少其共享内存 helper；未去改动它。
一个另行验证的完整 binary Torch 2.13 安装提供了最终 RandomOp
参考，并以相同结果重复了全部十八个 CPU seed API 行。
其二进制身份独立记录。之后一次 CPU 缓存路径切换在任何测试
运行前产生文档所述的 JIT 工具更新退出码 3；该次启动被保留，
同一源码在重建后重跑。

添加本报告后，定向文档/打包运行通过 21 项检查，
失败于两项中央登记检查：新页面不在 results toctree 与生成的
根 manifest 中。整合 owner 维护这两个共享文件；
本次独立改动不声称这些检查通过。此前的 1486/6 structure
结果适用于添加本报告之前未变的计算/测试源码。

在 `ab006244a34c550a5f7235b1f65ec7d4165bde9d` 上整合后，
本页在 results index 与生成的 manifest 中登记。全部 120 个封存
原始产物与全部 17 个已提交文件哈希均独立复检。整合后的
structure/tool 套件通过 1690 项检查，含六个声明/环境 skip；core 通过 322
项检查，含 42 个 skip 与 1 个既有预期失败。合并的 CPU RNG/pytree
后续通过 62 项检查，含 36 个加速器 skip。这些整合日志未版本化，
位于 `$JITTOR_LAB_ROOT/_state/rng-integration-ab006/`。更早的独立登记
失败保留为历史证据。

## 复现

使用私有 Python 环境，配仓库的 pytest/插件固定版本、NumPy、
扩展行所需的 pybind11，以及原生参考行所需的完整独立
binary PyTorch 安装。把 `PYTHONPATH` 设为本 checkout 的 `python`，
隔离 `HOME`、`JITTOR_HOME` 与 `TMPDIR`，并用规范的 `JT_BACKEND` /
`JT_USE_CUDA` 设置选择 CPU/CUDA。禁用 pytest 自动加载时，启用
`pytest_timeout` 与 `xdist.plugin`。

```bash
# Torch mode; CUDA build, two visible physical devices, strict fallback scope in tests.
JITTOR_TORCH_SHIM=1 python -m pytest -q \
  compat/tests/torch/test_generator_native_contract.py \
  compat/tests/torch/test_generator_streams.py \
  compat/tests/torch/test_torch_sampler_rng.py \
  compat/tests/torch/test_torch_fork_rng.py \
  compat/tests/torch/test_core_misc_owner.py \
  compat/tests/torch/test_torch_cuda_rng_state.py \
  compat/tests/torch/test_cpp_extension_philox.py

# Native mode, CUDA build with CPU ambient placement. Tests select CUDA where declared.
# REAL_TORCH_SITE must resolve to the actual complete binary package's site directory.
JITTOR_TORCH_SHIM=0 JT_USE_CUDA=0 use_cuda=0 \
  REAL_TORCH_SITE=<independent-torch-site> JITTOR_REQUIRE_REAL_TORCH=1 \
  python -m pytest -q tests/ops/test_random_generator.py tests/ops/test_random_op.py

python tools/run_test_suite.py --tier core --serial-compile
bash tools/check_repo_layout.sh
JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q tests/structure
```

原始 JSON、源码哈希、独立 runtime 身份、精确环境脚本、失败
运行与日志未版本化，位于 `$JITTOR_LAB_ROOT/_state/rng-generator-entry-contracts/20261015/`。
`baseline-source.json`、同步记录、`final-compute-source.json` 与
`post-sync-compute-source.json` 各自分开保存每个基线与 owned 源码快照；
`SHA256SUMS.json` 列出保留的原始产物，并显式排除
runtime 缓存与依赖。

## 边界

原生 CPU、原生显式 CUDA Philox、CUDA 默认 cuRAND 与
PyTorch 使用不同的算法/状态格式。此处验证的保证是
seed metadata、流 ownership、隔离与 runtime 内精确重放。
PyTorch 或旧 NumPy 支撑的 shim 状态字节无法被这些原生 owner
重放，会被拒绝。跨设备的显式状态迁移被有意拒绝。

这并未提升每个 factory 的保真度：原生整数抽样的既有
基于浮点的近似、大整数区间、稳定的 tie 处理、不支持的 `out=`
以及更广的 layout/内存格式语义都不在本结果内。本次测量中
default-generator 委派是一个单独的 Python 类。[default-generator 后续](2026-10-15-default-generator-state-contract.md)
统一了公共 Python 类、CPU 快照交换与私有拷贝语义；
完整的 C-extension ABI 一致仍不在这些结果内。共享源码改动
不产生 ROCm/NPU/Corex 硬件结论。legacy 回调 ABI 与 legacy
原生 `jt.set_seed(int)` 入口不变；其他 RNG 消费者（如
cuDNN RNN dropout）的全宽行为未在此执行。这些小的正确性
测试不引出 benchmark、速度或内存容量结论。
