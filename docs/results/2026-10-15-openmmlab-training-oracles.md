# 真实 MMEngine 与 MMCV 训练 owner 在 CPU 与 CUDA 上

- 状态：真实下游前向/反向与三步训练在两个设备上复现；限定的 tiny 组件场景，仅 L3。
- 测量基线：`2aa27ffe1ce1699c888575f3b5fb3092b1fc6eff`，每个数值子进程与专用 nox 执行前后均干净。
- 兼容前置：`99ee01f7c8d2d29056347846b2c99e4775e46c2a`；该源码基线包含训练探针、显式全梯度契约、registry 扫描器与 opt-in 门禁。
- 上游：MMEngine 0.10.7 位于 `390ba2fbb272816adfd2883642326d0fd0ca6049`；MMCV 2.2.0 位于 `3a5fb4b96ab1b241e7379adf7c326ef6edef6515`，两者都是 oracle 与候选共享的干净 release checkout。
- 维护者：Jittor core / Torch 兼容维护者。
- 复查条件：上游 registry、BaseModel/OptimWrapper/ConvModule 实现、BatchNorm 状态、兼容优化器/梯度、依赖版本或精度策略变化。

## 实际的下游 owner 与参数范围

两个用户模型都继承真实上游的 `mmengine.model.BaseModel`。其未修改的
`BaseModel.train_step` 负责预处理、`forward(mode='loss')`、标量/列表 loss 解析、
反向、通过 registry 构建的 `OptimWrapper` 做范数裁剪、真实 momentum SGD 步与
清零。一个小的 SGD 子类只在 `super().step` 前观测梯度；它
不实现任何优化器数学。三步中的每一步必须恰好调用一次 loss 前向与
优化器，并返回真实的 `loss`、`loss_regression`、`loss_regularization` 日志。

`mmengine_train_step` 使用 LayerNorm(6)、Linear(6,8)、SiLU 与 Linear(8,2)，输入 `[4,6]`，
输出 `[4,2]`，全部**六个**命名参数可训练。`mmcv_convmodule_train_step` 通过
`mmengine.registry.MODELS.build` 构建两个真实 `ConvModule` 实例：3→4 通道，使用 3×3
卷积/BN/ReLU，然后 4→4，使用卷积/GN（两组）/LeakyReLU、空间均值与一个
Linear(4,2) head。输入 `[2,3,8,8]`，输出 `[2,2]`，全部**八个**命名参数可训练。
目标是确定性的 float32 张量，不带梯度。输入样本可微。
不存在冻结参数、adapter、下游源码补丁或替代扩展。

每个轨迹在每次真实 `train_step` 后比较完整输入与目标、预测、解析日志、全部裁剪后
参数梯度、输入梯度、每个更新后参数、每个 momentum buffer 与每个
命名模型 buffer。两个模型分别得到 **75 / 102 个数组**。MMCV 在全部三步包含
BN running mean、方差与标量 int64 batch counter。
完整的对称键、精确 shape 与 dtype、有限值与一致固定输入都是
强制的。普通 CASES 另外确立直接前向/反向；普通的通用 SGD
训练并不确立该下游训练 owner。

## 固定平台与协议

| 字段 | CPU | CUDA |
| --- | --- | --- |
| 硬件 | AMD EPYC 9554 64-Core Processor；加速器可见性为空 | 一张分配的 NVIDIA GeForce RTX 4090，driver 580.178.04 |
| Oracle | 独立的 binary PyTorch 2.11.0+cu128，带真实 `_C` | 相同，CUDA 12.8 |
| 候选 | Jittor 2.0.0 Torch 兼容层 | 相同，实际 NVCC 12.2.140 |
| 依赖 | Python 3.11.16、NumPy 1.26.4、MMEngine 0.10.7、MMCV 2.2.0 | 相同的固定源码 |
| 精度 | TF32 matmul/cuDNN OFF，matmul highest | TF32 matmul/cuDNN ON，matmul high |
| 控制 | float32；cuDNN benchmark OFF；两个 runtime 线程，固定两 CPU 亲和 | 相同；实际 CUDA 常驻与零后端 fallback |

种子 0 导出完整参考权重与模型 buffer。种子 7 与 19 在两个 runtime
上复用完全相同的权重哈希；每个扰动的输出与一个被请求的梯度必须改变。
单独的实训练 owner 使用 seed-0 参考状态、三个输入 seed 1/2/3 与目标
seed 101/102/103，SGD 学习率 0.01、momentum 0.9 与 clip norm 1.0。三步后
六个/八个参数都必须改变。每个 benchmark 与 unit/gate 缓存都分开；首次编译
串行。设备分配诊断与所有可见 NVIDIA wheel metadata 保留在原始
platform JSON 中。并发卡占用不被当作性能测量。

共享 harness 测量 `max_abs_error / max(reference_field_max, gradient_block_max*1e-3,
1e-6)`。输出、loss、更新后参数与 buffer 使用各自的参考字段；一步中所有被请求
梯度共享梯度下限。现有正确性限值在两个设备上仍是：非梯度字段 0.002、
梯度 0.01。不增加阈值或改变模型形状
来掩盖差异。整数 BN 状态还必须保持精确标量形状与 int64 dtype。

## 测量结果

| Repo | 设备 | 每个 seed 0/7/19 的数组 | 跨 seed 最大绝对 / 归一化误差 | 三步数组 | 最大绝对 / 归一化轨迹误差 | 实际更新的参数 |
| --- | --- | --- | --- | --- | --- | --- |
| mmengine | CPU | 8 | 2.38418579e-07 / 3.09323537e-07 | 75 | 1.1920929e-07 / 5.21069171e-07 | 6 / 6 |
| mmcv | CPU | 10 | 2.38418579e-07 / 1.54802998e-06 | 102 | 2.38418579e-07 / 6.47097259e-07 | 8 / 8 |
| mmengine | CUDA | 8 | 0.000469446182 / 0.000338873152 | 75 | 0.000156402588 / 0.000432249954 | 6 / 6 |
| mmcv | CUDA | 10 | 9.93609428e-05 / 0.00031569602 | 102 | 8.55773687e-05 / 0.000624192525 | 8 / 8 |

公开 scorecard 审计接受 **4 条 L3 记录**，拒绝 0 条，重哈希 **180 个产物绑定**。一个如实重哈希、但声称 CUDA TF32 OFF/highest 的负向 platform 对两行 CUDA 均以 `Platform/measurement precision mismatch` 被拒绝；两行未改动的 CPU 仍被接受。随后每个原始产物绑定都在未改动的情况下重新哈希。

在同一干净基线上，实际的 `ecosystem_openmmlab` nox 会话在 CPU 与 CUDA 上
各自独立通过 **5 个测试、零 skip**：两个真实模型门禁加三个 BatchNorm 状态测试。
真实 PyTorch 解释器在每个设备上独立通过两个相关的 counter/state 测试。
第三个测试检查前端接入不改变原生 Jittor checkpoint 语义。

## 源码推导的限定分母与如实广度

仅用标准库的清单扫描干净 MMEngine / MMCV 包中全部 153 / 128 个受跟踪 Python 源码文件，
记录源码哈希、注册表达式与分支/函数/循环守卫。
字面注册循环、别名、类/函数装饰器与可选编译模块
分支都保留在声明中。任何声称轴中未解析的注册都按失败关闭处理。
MMEngine 有 25 个已解析声明点（一个条件式），MMCV 有 90 个（九个带守卫）；没有遗留
未解析的限定注册。MMEngine 其余 101 个注册点（含动态优化器
发现）被披露，但不为其编造分母。

| Repo | 声明源码轴 | CPU L3 成员 / 分母 | CUDA L3 成员 / 分母 | 已验证方法上下文 |
| --- | --- | --- | --- | --- |
| MMEngine | model_component | 1 / 7 | 1 / 7 | 该实际 train_step 用户模型中的 BaseDataPreprocessor |
| MMEngine | optim_wrapper | 1 / 5 | 1 / 5 | registry 构建的 OptimWrapper，裁剪 + momentum SGD |
| MMEngine | loop | 0 / 4 | 0 / 4 | 未验收 Runner loop |
| MMEngine | weight_initializer | 0 / 8 | 0 / 8 | 未验收 initializer 轴 |
| MMCV | model_component | 1 / 70 | 1 / 70 | registry 构建的 ConvModule，含所述 BN/GN 块 |
| MMCV | hook | 0 / 1 | 0 / 1 | 未验收 hook |
| MMCV | transform | 0 / 19 | 0 / 19 | 未验收 transform 轴 |

这些分母描述所述上游 SHA 下的限定源码声明，保留
条件与依赖扩展的条目。它们不是已安装 runtime 清单，也不是
整库 API 分母。自定义 Regressor 不计为上游 registry 成员；
ConvModule 内部使用的其他 norm/activation 别名保守地不单独计数。
每个被接受的覆盖行保留用例、完整 registry 引用、tiny shape 种类与
六个/八个完全可训练的参数名。其他加速器后端没有任何被接受的证据。
完整 Runner/dataset/hooks、分布式训练、编译版 `mmcv.ops`、生产规模、内存与
速度级别都不在本结果内。上游 inplace 激活请求会产生既有的
原生分配警告；比较的是输出/梯度正确性，不评分分配复用。

## 兼容缺陷与回归证据

修复前，独立的公共 API 复现在 CPU 与 CUDA 上都发现前端 BatchNorm 的
`num_batches_tracked` 继承了原生 `(1,)` int32 非持久状态，而真实 Torch 暴露的是
标量 int64 持久 buffer。修复仅限兼容 NN 构造 owner：
在被采用的前端实例上注册真正的标量后端 `array(0, dtype='int64').stop_grad()`，
或在禁用 running statistics 时为 `None`。原生 Jittor 状态不变。

所选设备回归覆盖 BatchNorm1d/2d/3d、真实训练前向后的计数 1/2/3、
eval 中不更新、持久 state_dict 与 torch.save/load_state_dict 往返、恢复后计数 4、
float64 模块转换保持 int64 counter、参数/buffer ownership 与原生/前端
隔离。除该 counter 契约外，不声称未测试的 running-statistics API 行为。

开发回归在源码提交之前运行，基于合并 `3156ee1f7ac11cc82acb6920f9ec611da9299fee`
加本批改动；正式数值与专用 nox 证据另外固定上述干净
测量 SHA。CPU core：**322 passed，42 skipped，1 个预期失败**；相邻 normalization/serialization/
counter 测试：**56 passed，六个加速器 skip**。Host/namespace/registry 负向契约：
**68 passed**。在修正外部测试环境后，完整 structure：**1453 passed，四项
skipped，两项 failed**，都是该源码分支上先前已封存的 Longformer 报告留下的本地
sealed-Longformer 文档索引/manifest 条目。这些失败不构成
结构套件全绿。本报告写就后，其余 14 个文档契约定通过；
可达性检查仍属中央报告登记的一部分。布局、Ruff 与 diff
空白检查通过。诊断日志把原始环境失败与模型验收证据
分开保留。

## 复现与产物归属

可复用的入口有
[`openmmlab_training_probe.py`](../../agent/skills/torch-compat-repo-runbook/scripts/openmmlab_training_probe.py)、
[`test_ecosystem_openmmlab.py`](../../compat/tests/torch/test_ecosystem_openmmlab.py) 与
`ecosystem_openmmlab` 会话。在独立的真实
Torch 与 Jittor 解释器中准备两个固定的下游 checkout，然后先运行 oracle 探针，
并把其 `.weights.npz` 提供给候选。使用专用会话时设置 `REAL_TORCH_PYTHON`、`JITTOR_CI_PYTHON` 与一个选定的
`JITTOR_TEST_DEVICES=cpu` 或 `cuda`；仅 CUDA 需要分配真实的可见设备与 NVCC。
该会话启用必需的依赖、真实 oracle 与非零执行 flag，并禁止
后端 fallback。CPU 在 import 框架前显式清除所有 CUDA 可见性/启用
变量。缺少必需库、不完整梯度、伪造 owner/更新、错误设备
metadata、缺少优化器状态、非对称键、广播形状与非有限数据都有
host 负向测试。

原始数据、精确命令与私有解释器/缓存位置保留在
`$JITTOR_LAB_ROOT/_state/downstream-evidence/real-runs/openmmlab-20261015/formal-2aa27ffe1c/`：
`measurement-source.json`、`platform-{cpu,cuda}.json`、固定清单、原始 seed/weight/train
NPZ 与 JSON、全部比较报告、nox 日志/源码检查、`scorecard-evidence.json`、
`scorecard.json`、`negative-precision-scorecard.json` 与负向结果。本受跟踪报告
不含原始张量。快照 skill/case discovery 审计列出两个普通 CASES；
专用门禁节点、源码 SHA 与实际 nox 执行另行记录并绑定。
仅静态 discovery 从不确立已执行的训练门禁。
