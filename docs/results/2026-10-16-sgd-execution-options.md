# SGD foreach 与 fused 执行选项

- 状态：KI-COMPAT-008 已关闭，限于经验证的 CPU/CUDA 选项契约。
- 复查日期：2026-10-16。
- 基线：`ae1df1983391d0696e72a0dec98696fe2236a1e3`，加上本报告的三处生产改动。
- 维护者：Torch 兼容性与原生优化器维护者。
- 复查条件：SGD 派发、参数组、状态导出或检查点加载发生变化。

`torch.optim.SGD` 现在接受仅限关键字形式的 `foreach` 与 `fused`，并在其状态字典中保留
构造器默认值和组覆盖。原生 SGD 所有者依据这些偏好选择其既有的批处理内核：`fused=False`
禁用该内核；`foreach=False` 会禁用它，除非 `fused=True` 显式请求。默认值和
`foreach=True` 允许后端选用其符合条件的实现。同时要求两个选项都为 `True` 时，抛出与
Torch 相同的 `RuntimeError`。

符合条件的 CUDA FP32 张量使用既有的批处理 SGD 内核。其他受支持的张量在其自身后端上保留
原生 JIT 更新。不引入新的优化器数学、主机更新或后端回退。原生构造器默认值保持不变。
状态导出还会为新加入的组保留继承的执行选项，恢复后的优化器使用其保存的组选项。

在此改动之前，全部八种合法选项组合在前端初始化器中都抛出 `TypeError`。维护版 CPU 回归
为 **27 个失败 / 27 个 CUDA skip**；独立 Torch 接受合法组合并拒绝冲突的那一对。改动后，
新测试在真实 Torch `2.13.0+cu130` 中不变地通过：CPU **27 / 27 个 CUDA skip**，CUDA 可见
**54 passed**。候选测试还包含既有的精度、检查点和 SGD 邻接：CPU **47 / 34 个 CUDA
skip**，CUDA 可见 **80 / 1 个 CPU 域 skip**。整调用候选守卫记录零后端回退。

契约覆盖普通 SGD、带权重衰减与 dampening 的动量，以及 Nesterov，使用两个参数和三次
反向/更新步。全部参数与动量数组与 30 个独立生成的真实 Torch FP32 基准数据精确一致。
输入是二进分数，因此相同的精确参照可在 CPU 与 CUDA 上检验每种执行选择。一次序列化状态
往返可恢复两个选项以及随后精确的参数/动量更新。组覆盖以及新加入组上的继承选项会单独
检查。

CUDA 观察检查真实的驱动指针属性。派发插桩统计调用次数并委派给原始原生内核：被禁用的
选择不产生批处理调用，启用的符合条件选择会写入两个参数，恢复后的优化器遵循保存的选择。
混合组用例只启动符合条件的那个组。测试在每次更新后保留原有的已发布梯度对象及其完整值。

```bash
JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q \
  compat/tests/torch/test_sgd_execution_options.py \
  compat/tests/torch/test_sgd_checkpoint_continuation.py \
  compat/tests/torch/test_torch_compat_sgd_precision.py \
  compat/tests/torch/test_torch_compat_optim.py::TestSGD
```

来自单独[检查点报告](2026-10-16-sgd-checkpoint-continuation.md)的 CPU 低精度跨运行时
差异仍未关闭。本结果不确立完整的优化器 ABI、分布式 SGD、其他加速器或整模型支持。
原始命令、失败、精确源码哈希、独立基准数据和加载的核心快照未纳入版本控制，位于
`$JITTOR_LAB_ROOT/_state/sgd-execution-options/20261016/`。

单独的 CPU 探针最初暴露了 **KI-COMPAT-015**：可移植 SGD 重新绑定了参数分配，使保留的
解绑别名失效。2026-10-25 的整合解决了该可移植 SGD 存储边界并验证 CPU/CUDA 别名；精确
的源码、设备与保留本地工作边界见[按日期记录的整合报告](2026-10-25-recent-pr-integration.md)。
上述历史性的选项选择与数值结果仍绑定其原始源码。干净的已提交融合 AdamW 以及围绕已预热
重放的待合入外部图有各自的限制，分别记录为 KI-COMPAT-016 和 KI-EXEC-015。

相邻原生优化器契约在 CPU 通过 **41 / 9 个 CUDA skip**，CUDA 可见 **50 / 0 skip**，
包含 CPU 对照。核心通过 **322 / 42 skip / 1 个既有 xfail**，合并结构门禁通过
**1628 / 6 skip**。新增测试的 Ruff 干净；被改动的遗留 API 文件保留八项既有 Ruff 诊断，
新增诊断为零。布局与 manifest 检查通过。
