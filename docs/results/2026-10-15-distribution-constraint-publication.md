# Distribution constraint 命名空间发布

- 状态：针对前端发布与边界检查已修复并验证；不声称通用 TensorDict 兼容性
  或完整 constraint 校验能力。
- 日期：2026-10-15。
- 基线：`be2ffcff831ca42639525e1c0508ba6ef04d0b77`；实现为本报告所在提交。
- 范围：CPU 与真实 RTX 4090 CUDA；ACL 与 ROCm 未验证。
- 维护者：Torch 兼容性维护者。
- 复查时机：distribution 模块克隆、原生 constraint 所有者、Torch 命名空间
  发布方式或参考版本发生变化时。

## 故障与修复

一个纯 Torch API 的 distribution 子类在任何 TensorDict 导入之前就在
`constraints.real` 处失败。原生命名空间已经暴露了 `real`、`interval` 和
`dependent_property`，但其前端克隆没有暴露其中任何一个。原生 constraints 是
带有类属性和静态方法的 `ModuleType` 子类；前端只复制了 `dir(instance)`，
而它列出的只是模块实例属性。随后安装器看到一个已存在的命名空间，
正确地避免了替换它。

克隆现在还会包含模块类型上的属性。这保住了原生 constraint 与工厂的所有权，
发布了 property 装饰器，并保持原生包不变。它没有增加任何数学实现，也没有对
原生宽松 constraint、整数支持或 event 维度作出更强的承诺。现有 distribution
保真度仍为 `APPROXIMATE`。

## 验证

回归用例在不导入任何下游库的情况下定义了一个带有 `arg_constraints` 和依赖型
`support` property 的 distribution 子类。它们检查规范模块标识、继承的所有者标识
以及未改动的原生命名空间。设备用例执行 interval、greater-than 和 less-than 谓词，
其中包含一个 NaN，并要求真实的 CPU/CUDA 放置以及零后端后备。
一个未安装 Jittor、也未安装 shim 的独立二进制 PyTorch 2.11.0+cu128 解释器
在两个设备上产生了完全相同的布尔结果。

```bash
JITTOR_TORCH_SHIM=1 PYTHONPATH=python JITTOR_TEST_DEVICES=cpu,cuda \
  python -m pytest -q \
  compat/tests/torch/test_distribution_package_identity.py \
  compat/tests/torch/test_distribution_frontend_owners.py \
  compat/tests/torch/test_torch_compat_distributions.py

python tools/run_test_suite.py --tier core --serial-compile
JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q tests/structure
bash tools/check_repo_layout.sh
```

- CPU distribution 选择集：34 通过，无跳过。
- 启用 CUDA 的 distribution 选择集：35 通过，无跳过，包含真实设备边界检查
  和既有 distribution 回归。
- 完整 CPU core runner：322 通过、42 跳过、一项既有预期失败；
  原生与 Torch 子进程均返回零。
- 完整结构选择集（显式加载 xdist/timeout 插件）：1,432 通过、六项跳过、无失败。
  两项跳过涉及该仅源码解释器中的 distribution 解析；其余为已声明或加速器能力跳过。
  布局与空白检查通过。

使用 Python 3.11、GCC 11.5 和 CUDA 12.2.140，CPU/CUDA 缓存相互独立；首次编译为
串行执行。未纳入版本控制的日志位于
`$JITTOR_LAB_ROOT/_state/refactor-goal/distributed/logs/`：
`constraints-publication-cpu-recheck.log`、`constraints-publication-cuda.log`、
`constraints-independent-torch.log` 和 `constraints-publication-core.log`。
首次结构调用在插件自动加载被禁用时漏掉了显式的 xdist 插件加载；其两项子进程记账失败
不能证明源码回归。修正后的运行为
`constraints-publication-structure-recheck.log`；显式的
`PYTEST_PLUGINS=xdist.plugin,pytest_timeout` 提供了已安装的插件。

TensorDict 0.10.0 的 `TruncatedNormal` 类定义暴露了这一通用发布缺陷。
端到端 TensorDict 索引、堆叠与训练需要各自的独立数值证据，不在本报告范围内。
