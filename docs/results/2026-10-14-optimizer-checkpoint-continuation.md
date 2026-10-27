# 优化器 checkpoint 与确定性 Accelerate 续跑

Status: 历史单进程数值验收。后续的 [RNG 与 FSDP2 FULL 续跑报告](2026-10-14-checkpoint-rng-fsdp2-continuation.md) 取代下面记录的 RNG 与多 rank 限制。ACL 仍未验证；本报告保留原始基线与证据。

Baseline: `cded3102037edc3663b8f2054a6ebc06da6ebfd2`（`origin/2.0-refactor`，实现前拉取）。实现：包含本报告的提交。维护者：Jittor 兼容性维护者。复查条件：checkpoint FQN 变换、优化器状态所有权、FSDP provider 变换、序列化允许列表、RNG owner 或下游版本发生变化。

基线无法从 `torch.distributed.checkpoint.state_dict` 导入 `get_optimizer_state_dict`；其组合 setter 也忽略优化器 checkpoint。实现增加了基于 FQN 的优化器 get/set、独立于活参数/组顺序的参数组恢复、多优化器、flattening、FULL FSDP provider 变换、host offload 以及 rank-zero broadcast。模型加载遵循 `strict`；模型 checkpoint 可以省略冻结参数。非空子模块选择显式抛出 `NotImplementedError`；这不是所有 DCP 选项的完整实现。

实际的 Accelerate 加载还暴露了缺失的 `torch.serialization.get_safe_globals`、`add_safe_globals`、`clear_safe_globals` 与 `safe_globals` API。它们的实现控制受限 unpickler；自定义类在注册前与临时允许列表结束后失败。没有引入下游 adapter 或替代优化器数学。

同样的七个测试用例在独立二进制 PyTorch 2.11.0+cu128 与 Jittor Torch 前端下运行，两侧均使用 Accelerate 1.10.1。Oracle 身份独立检查，其包使用原始二进制库。缺失的环境依赖在隔离的外置 overlay 中提供；共享环境未被修复。CPU 与 CUDA 使用不同的 JIT 缓存。两个后端都执行真实张量计算，禁止回退，观察到回退计数为零。

| 检查 | CPU | CUDA |
| --- | --- | --- |
| 独立 PyTorch / shim 用例 | 7 / 7 passed | 7 / 7 passed |
| 对照独立参考的最大绝对误差 | 1.49e-8 | 3.73e-9 |
| 后端回退计数 | 0 | 0 |

比较包含每个模型参数、优化器 moment 与 step、参数组学习率、续跑 loss、所有参数梯度以及输入梯度。输入与初始权重为固定本地数组；模型没有随机层。PyTorch 2.11 为 AdamW group 元数据增加 `decoupled_weight_decay=True`；shim 不暴露该较新字段。它已显式检查并从跨运行时结构相等性中排除；所有共享元数据与数值条目均比较。独立 PyTorch 目前在加载组合的 FULL-plus-flatten 选项时抛出 KeyError，因此可移植 flatten 门禁使用 `flatten_optimizer_state_dict=True, cpu_offload=True`，并单独测试 FULL offload。这未验证不支持的上游选项组合。

快照使用公开的 `torch.clone`：native `Var.clone` 是图/存储别名，因此不能建立不可变 Torch checkpoint 证据。setter 在安装已加载 tensor 时也使用公开的拷贝操作。

用部署的 shim 与隔离的 CPU/CUDA 环境复现功能选择，然后在独立 PyTorch 解释器中运行同一文件：

```bash
CHECKPOINT_DEVICE=cpu JITTOR_REQUIRE_ACCELERATE=1 \
  python compat/tests/torch/test_optimizer_checkpoint_state.py -v
CHECKPOINT_DEVICE=cuda JITTOR_REQUIRE_ACCELERATE=1 \
  python compat/tests/torch/test_optimizer_checkpoint_state.py -v
```

实现的聚焦序列化、checkpoint 与 FSDP 结构选择产生 63 passed 和一个既有加速器 skip。整合到 `7ed9b3b52b658d2a3c1ac390eba09087c5e80f95` 后产生 59 个 CPU 通过和一个因隔离环境缺少 `requests.HTTPError` 的失败。在该环境中提供依赖后，受影响的 Accelerate 用例单独通过。所有七个 CUDA checkpoint 用例在整合树上通过。

Repository layout 验证通过。整合推送前的完整 `tests/structure` 运行产生 1429 passed、4 skipped，以及 `test_backend_grad_contract.py::test_the_manifest_is_exactly_the_gradients_in_the_tree` 中的一个失败。其既有 `DropoutACL` 条目在 ACL 实现把 backward 移到 CodeOp 后不被直接 `grad` 方法扫描器识别。此改动不修改 ACL 或该 manifest；完整结构门禁仍失败。早期隔离运行的 ASV 与 xdist 依赖失败已在验证环境中解决，本次运行未再出现。

整合的 CPU `core` runner 两种进程模式均通过：native 产生 98 passed、41 skipped 和一个既有预期失败；Torch 产生 224 passed 和一个加速器 skip。合计：322 passed、42 skipped 和一个预期失败。一次初始调用在禁用 plugin autoload 时遗漏了显式加载的 pytest-timeout 插件；它在 native 测试执行前停止。上面修正后的完整调用退出码为零。

另一个真实 Accelerate RNG 续跑探针失败：Python 与 NumPy 流恢复，但 Torch CPU 流从种子而非保存位置续跑。兼容 API 明确宣称仅种子行为；此实现不修复它。Native CPU/ACL 快照 API 存在，需要单独验证的前端适配。CUDA 已有 cuRAND seed/offset 接口，但本报告不接受完整 RNG 恢复。失败的探针与原始日志保留在外置任务状态树下（`checkpoint-optimizer/rng_checkpoint_repro.py` 与 `checkpoint-optimizer/logs/targeted-rng-failure.log`）。此处的声明只覆盖确定性固定输入数值续跑，不覆盖完整 Accelerate 状态恢复。

## 2026-10-16：优化器构造前已发布的梯度

Status: 下述契约的 CPU/CUDA 行为已验证。Owner: Torch 优化器兼容维护者。Baseline: `4fd2a82951d2a5300774862685a0e661e364e93f`。实现: `dc6a178064c22bb5a31fc76bd55a879cc9497f5e`。复查条件：Tensor 梯度发布、优化器初始化、参数组或 native 惰性梯度创建发生变化。

构造优化器时，Parameter 可能已通过公开赋值或 `backward()` 拥有 `.grad`。在基线上，SGD/Adam/AdamW/RMSprop 与这两种梯度来源的全部八种组合静默跳过更新；把相同的梯度创建移到构造函数之后即可工作。独立二进制 Torch 2.11 在全部十六个 before/after 对照中更新。

构造函数现在把现有已发布梯度对象连接到 native 参数组，包括其他组中缺失的梯度。它保留对象/存储身份，使另一个优化器、原地梯度编辑与 `zero_grad(set_to_none=True)` 观察到同一状态。没有梯度时，原始 native 惰性初始化仍活跃。一次用 None 预填充所有组的中间尝试导致九个 native backward/step(loss) 回归；该尝试被拒绝，其失败日志保留。

新增的维护文件 `compat/tests/torch/test_optimizer_preexisting_grad.py` 每设备有十九个契约：两种梯度来源与构造函数顺序、两次连续更新、Adam step 状态、两个优化器共享梯度、跨组缺失梯度、带权重衰减的零梯度，以及无梯度 step。每个被检查的候选 tensor 检查其实际后端位置；回退被禁止。同一文件也在独立二进制 Torch 中运行。

| 范围 | CPU | CUDA 配置 |
| --- | --- | --- |
| 候选，新文件加四个相邻优化器/梯度文件 | 57 passed / 19 CUDA skips | 76 passed / 0 skipped |
| 独立 Torch，新文件的精确副本 | 19 passed / 19 CUDA skips | 38 passed / 0 skipped |

CUDA 配置从新文件运行 CPU 与 CUDA 参数；其十九个 CUDA 用例在加速器上执行。一些相邻测试显式测试 CPU，所以总数 76 不是仅 CUDA 测试的计数。完整候选 CUDA 调用在前后观察到回退计数 0。独立 Torch 版本分别为 2.11.0+cpu 与 2.11.0+cu128。候选 native CUDA 二进制的 SHA-256 为 `03b66666626c7fef5b1a2a9a645c5e80972cc9e8c8a316d106971db74c7d71ef`。

Core 验证：322 passed / 42 skips / 一个既有 xfail。最终生产文件与该 core 运行字节一致；此后只简化了测试设备参数化。Structure/layout/manifest 结果：`1628 passed / 6 skipped; layout and manifest checks passed`。新测试通过 Ruff；生产文件保留与拉取基线相同的八个既有 Ruff 诊断，没有新增诊断。

在隔离的 CPU/CUDA 环境中复现五文件选择：

```bash
JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q \
  compat/tests/torch/test_optimizer_preexisting_grad.py \
  compat/tests/torch/test_torch_compat_grad_management.py \
  compat/tests/torch/test_optimizer_parameter_rebinding.py \
  compat/tests/torch/test_optimizer_scheduler_api_owners.py \
  compat/tests/torch/test_torch_compat_optim.py
```

原始证据未版本化，位于 `$JITTOR_LAB_ROOT/_state/optimizer-existing-grad/20261016/`：基线与修复后的十六行探针、被拒绝的中间实现、冻结的实测源码、精确副本 oracle 绑定、CPU/CUDA 日志、整调用回退守卫与 core 门禁。一个初始探针通过了 RMSprop(weight_decay=0)，这是此前端不支持的 keyword，并大声失败；修正后的探针使用默认零衰减，不带该 keyword。此改动不声称该单独 keyword 缺口已修复。

## 2026-10-16：native RMSprop 耦合权重衰减

Status: 所述 FP32/FP64 CPU/CUDA 契约已验证。Baseline: `8333abbbb59776acde9fd622293353193a45c7eb`。实现: `b4b5b88e8bf3ed1c13b5c66c787d2fdcf4f0d8e7`；其精确的七 owner Git 绑定与原始证据一同保留。Owner: native 优化器 / Torch 兼容维护者。复查条件：RMSprop 算术、组覆盖、梯度所有权或 checkpoint 元数据发生变化。

这关闭了上面记录的权重衰减 keyword 缺口。基线把 `weight_decay=0` 当作未知 keyword 拒绝，并静默忽略仅通过参数组提供的衰减。最终基线选择有十五个失败的 CPU 契约，包括两个 FP32/FP64 仅组数值失败。更新现在在现有 native RMSprop owner 中，在累积平方平均并更新参数之前把 L2 项加到梯度上。已发布的 `.grad` 对象与字节保持完整。Torch 状态导出记录实际的全局/组衰减，允许已保存的非零衰减优化器在初始以零衰减构造的优化器中续跑。负与 NaN 全局衰减抛出 ValueError，与独立二进制 Torch 对照一致。

Native 位置参数顺序仍为 `(params, lr, eps, alpha, weight_decay)`；前四个参数保持旧顺序。这是 keyword 衰减接受，不声称支持每个 Torch RMSprop 选项或位置约定。新契约覆盖三次连续更新与平方平均状态、全局与每组覆盖（包括零）、缺失与零梯度、精确 checkpoint 续跑、梯度身份、dtype 与实际后端位置。可移植优化器契约共享一个回退守卫，而非复制它。

| 最终范围 | CPU | CUDA 配置 |
| --- | --- | --- |
| 六个相邻 Torch 优化器/梯度文件 | 73 passed / 33 CUDA skips | 106 passed / 0 skipped |
| Native 优化器核心加显式 RMSprop 设备契约 | 39 passed / 7 skips | 46 passed / 0 skipped |
| 独立 Torch 精确副本的两个可移植契约文件 | 35 passed / 33 CUDA skips | 68 passed / 0 skipped |

CUDA 配置包含显式选择的 CPU 对照。新 RMSprop 契约包含十四个公开 CUDA 用例与六个 native CUDA 用例，全部检查实际分配位置。两个最终 CUDA 调用都包裹在 fallback-error 作用域中，前后计数为零。最初的 native 仅 unittest 调用有 35 个成功断言但退出 1，因为其内部设备循环没有提供静态可识别的加速器用例。维护的 native 回归现在显式参数化设备并检查实际参数/状态存储；最终 46 用例门禁原样通过。一次单独的首轮结构运行拒绝了有意扩展的 native 签名。其精确签名契约现在包含追加的默认零参数，同时保留旧顺序；门禁未被绕过。

Core：322 passed / 42 skips / 一个既有 xfail。最终结构、布局与 manifest 结果记录在交付包中。新测试/支持文件通过 Ruff；编辑过的生产 owner 保留其基线诊断，无新增。此后续未执行 ACL 与 ROCm；这不授予这些设备或完整优化器 ABI 支持。

通过在上述五文件命令中加入 `compat/tests/torch/test_rmsprop_weight_decay.py` 复现，并在禁用 Torch shim 的情况下分别运行 native `tests/optim/test_optim_core.py tests/optim/test_rmsprop_device_contract.py`。使用隔离的 CPU/CUDA 单元缓存以及独立 Torch 解释器来运行可移植文件与 `_optimizer_contract_support.py` 的精确副本。

原始日志、被拒绝的中间验证、源码快照、精确副本绑定、整调用 CUDA 守卫、native 二进制身份与交付包未版本化地位于 `$JITTOR_LAB_ROOT/_state/rmsprop-weight-decay/20261016/`。
