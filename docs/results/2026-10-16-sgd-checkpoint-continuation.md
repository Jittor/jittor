# SGD 检查点续训与精度策略状态

- 状态：已验证有界的优化器续训；CPU 低精度一致性仍未关闭。
- 复查日期：2026-10-16。
- 维护者：原生优化器与 Torch 兼容性维护者。
- 原始全新进程测量：`72aebf9264f1360178ac876c88fedff4602bb6d5`。
- 维护版测试源码：`2715b895ee76d3f9d0971640bdfd62e83f9139f1`。
- 复查条件：SGD 更新、动量初始化、精度策略、优化器状态或序列化发生变化。

两个维护版测试在不改变优化器数学的前提下增加了检查点覆盖。可移植 Torch 契约覆盖
FP16/BF16/FP32、普通 SGD、带权重衰减与 dampening 的动量，以及 Nesterov。每个都使用两个
学习率不同的参数组，在三次真实反向/更新步之后保存，再将两个续训步与不间断执行进行对比。
恢复后的构造器默认值和参数值被故意设为不同，因此测试必须恢复检查点的组字段、完整参数和
动量。每次更新后已发布的梯度保持不变。

另一个原生测试检查 CPU 与 CUDA 上 FP32 的真实 `jt.save`/`jt.load`，分别使用
`torch_math=False` 和 `True`。加载进以相反策略构造的优化器会恢复保存的策略并精确续训。
缺少该字段的遗留状态保留构造器的策略。该原生测试是同进程序列化测试；下面单独的可移植
证据使用新的生产者/消费者进程。

八个实际全新 worker 覆盖 CPU 与 CUDA 上的候选项和真实 Torch。每个生产者写出九个真实
检查点文件，然后继续训练。其新的消费者读取同样的文件并以恢复的状态继续。独立复核重新
哈希了完整数组、检查点文件、七个原始源码收据（对照 Git blob）、worker 脚本、provider
收据和加载的框架核心二进制。它精确重算了全部 **504 个同运行时恢复/续训数组**，并从
损失系数直接重算了另外 **504 个已发布梯度数组**，包括 FP16/BF16 舍入。全部八个 PID 各不
相同。CUDA 观察包含真实的驱动指针属性；候选的整调用回退计数保持为零。

| 跨运行时范围 | 原始预算结果 |
| --- | --- |
| CUDA，生产者与消费者阶段共九种配置 | 18 匹配，0 失败；检查 336 个更新/梯度数组 |
| CPU，同样的配置/阶段 | 7 匹配，11 存在低精度差异；检查 336 个更新/梯度数组 |

包括初始与保存快照，独立复核还比较了每设备全部 444 个原始数组：CUDA 匹配 444/444；
CPU 匹配 302/444，保留 142 个低精度差异。

CUDA 的 FP16/BF16 比较是精确的；FP32 使用 `atol=2e-9, rtol=2e-6`。
CPU 不匹配影响六个低精度配置，其十二个阶段记录中有十一个未通过原始预算。它们仍未关闭，
且未从测试或原始结果中移除。两种设备上所有配置的同运行时精确续训均通过。完整的原始
参数组予以保留：候选组额外暴露 `n_step`，因此这不是完整的优化器 ABI 等价。真实参照是
独立的 Torch `2.13.0+cu130`，其实际 `_C` SHA256 为
`819a187ea3b1f4d781200fe73fba3264689835d2285f7e6d10b986257a802dce`。

```bash
SGD_CHECKPOINT_DEVICE=cpu JITTOR_TORCH_SHIM=1 PYTHONPATH=python \
  python -m pytest -q compat/tests/torch/test_sgd_checkpoint_continuation.py

# In a real CUDA-enabled environment:
SGD_CHECKPOINT_DEVICE=cuda JITTOR_TORCH_SHIM=1 PYTHONPATH=python \
  python -m pytest -q compat/tests/torch/test_sgd_checkpoint_continuation.py
JITTOR_TORCH_SHIM=0 PYTHONPATH=python \
  python -m pytest -q tests/optim/test_sgd_policy_checkpoint.py
```

原始的全新进程数据仍绑定到 `72aebf926`，即后来加入显式设备标签和原生测试之前。该 harness
使用未改动的可移植辅助函数；后续的测试提交不会给其测量重新贴标。
原始命令、失败、检查点和数组未纳入版本控制，位于
`$JITTOR_LAB_ROOT/_state/sgd-checkpoint-contract/20261016/`；独立复核位于
`$JITTOR_LAB_ROOT/_state/sgd-checkpoint-independent-review/20261016/`。
主整合记录位于 `$JITTOR_LAB_ROOT/_state/sgd-checkpoint-main-integration/20261016/`。

本结果不授予 Torch CPU 低精度一致性、分布式 SGD、整模型支持、其他加速器或统一的生态
L3 评级。

在维护版源码上的主整合通过可移植测试以及既有的 SGD 精度/行为邻接：CPU **20 / 7 个 CUDA
skip**，CUDA 可见 **26 / 1 个 CPU 域 skip**。原生检查点测试加上优化器核心与 RMSprop
邻接在 CPU 通过 **41 / 9 个 CUDA skip**，CUDA 可见 **50 / 0 skip**。CUDA 可见计数包含
原生测试的 CPU 对照；显式 CUDA 用例检查其真实设备。四个整调用守卫都报告零回退。
核心通过 **322 / 42 skip / 1 个既有 xfail**，合并结构门禁通过 **1628 / 6 skip**。新增
测试的 Ruff、仓库布局和 manifest 检查均通过。最终的可移植测试还在真实 Torch 中按设备
不变地通过九个用例，其独立的候选 CUDA 运行通过严格执行门禁。
