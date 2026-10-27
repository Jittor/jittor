# 四个 Torch like 工厂产生的独立叶子

- 状态：CPU/CUDA 定向行为已验证；报告已登记到中央 results 索引。
- 评审日期：2026-10-15；维护者：Torch 兼容性维护者。
- 基线：`1740cfc9e39b8ff7a55e379cdc0740bfcb8d87c6` 加上逐个哈希的工厂所有者
  与回归测试改动。这些是提交前的测量结果。
- 复查时机：工厂构造、原生叶子标识、梯度注册表、dtype 默认值、张量放置
  或设备测试参数化器发生变化时。

## 已验证的问题与修复

`empty_like` 绕过了共享的工厂选项适配器。省略或显式传入 false 的
`requires_grad` 仍会产生需要梯度的张量。显式 true 看似创建了叶子，但在反向之后
无法发布其自身梯度。`zeros_like`、`ones_like` 和 `full_like` 在请求梯度记录时暴露了
其内部的 broadcast/cast 计算图。另外，显式 `dtype=None` 在 `zeros_like` 和
`full_like` 中错误地使用了全局默认 dtype。

修复把所有选项处理保留在 `compat/torch/installers/factories.py` 中。
既有的空分配器现在委托给共享的构造器适配器。
这四个 like 工厂在走既有的 requires-grad 与叶子注册路径之前，会 detach 分配操作
内部的历史。显式 `dtype=None` 会保持输入 dtype。API 保真度分类仍为近似；
通过所测试的 contiguous/strided 选项并不构成对任意 layout、pinned memory
或 memory format 支持的证明。

## 独立验证

权威矩阵断言源码真实的 leaf/nonleaf 状态。它的四个工厂、三种梯度请求
（省略/true/false）以及两种来源类型在每个设备上产生 24 行。每一行候选结果都与
真实的独立 Torch 2.13.0+cu130 解释器比较。最初的诊断矩阵使用
`ones(..., requires_grad=True)` 作为其所谓的叶子；而候选方基于尺寸的 `ones`
本身也是 nonleaf。那些原始文件保持不变，但不用于声称叶子来源覆盖。修正后的矩阵
使用显式验证过的 `torch.tensor` 叶子，非叶子来源则通过乘法得到。

| 检查 | 结果 |
| --- | --- |
| 修正后的元数据矩阵，修复前 | 在 CPU 和 CUDA 上各有 10/24 行不匹配 |
| 修正后的元数据矩阵，修复后 | 在 CPU 和 CUDA 上 24/24 行匹配 |
| float32 输入、float64 全局默认下的显式 None dtype | 修复前每个设备 2/4 错误；修复后 4/4 正确 |
| 完整独立 Torch 契约文件 | CPU 与 CUDA 上 3 项测试通过 |
| 候选契约与构造器邻居 | 每个选中设备的会话中 14 通过，零跳过 |
| 原生与 Torch CPU core | 322 通过、42 项已声明/硬件跳过、1 项既有 xfail |
| 完整结构 | 1499 通过、4 项已声明/硬件跳过；2 处中央报告登记失败 |

三项既有契约测试包含全部 24 行以及 dtype 覆盖、显式 None、contiguous strided 选项
和 CPU/CUDA 放置覆盖。它们把新的物理指针与来源张量和第二个工厂结果分别比较，
然后验证写入不会影响另外两个张量。对于每个显式 true 请求，连续的独立 backward 调用
必须在结果上累计出恰好为二、然后为五的梯度，而来源与对等张量不保留梯度。
候选方还检查在声明设备上的实际分配，并拒绝后端后备。设备类由既有的
`instantiate_device_type_tests` 生成，固定为 CPU/CUDA，因此既有 CUDA 门禁使用其
正常的 `JITTOR_TEST_DEVICES=cuda` 选择。

两处结构失败仅指出新报告未出现在中央 results toctree 和 MANIFEST 中。
这些文件属于集成所有者。额外的真实 CUDA RMS 邻近会话通过全部 11 项公有/私有契约；
私有输出缓冲区探针现在在三种 dtype 下都与真实算子一致：requires-grad 为 false、
leaf 为 true，且没有 grad_fn。

CUDA 选择的邻居会话包含仅 CPU 的保真度检查和显式的宿主/设备传输；
其中三个新 CUDA 契约节点执行 CUDA 矩阵。CPU 验证隐藏 CUDA 并显式禁用其运行时。
候选方在 SM89 上使用 nvcc 12.2.140 的 CUDA strict kernel 数学；独立的 Torch wheel
使用 CUDA 13.0。这些确立正确性，而非速度测量。

```bash
CUDA_VISIBLE_DEVICES='' JT_USE_CUDA=0 JITTOR_TORCH_SHIM=1 \
JITTOR_TEST_DEVICES=cpu PYTHONPATH=python \
python -m pytest -q compat/tests/torch/test_like_factory_leaf_contract.py

JITTOR_TORCH_SHIM=1 JITTOR_TEST_DEVICES=cuda PYTHONPATH=python \
python -m pytest -q compat/tests/torch/test_like_factory_leaf_contract.py
```

在独立 Torch 解释器中直接运行同一个测试文件，不设置 `JITTOR_TORCH_SHIM`，
并设置 `JITTOR_TEST_DEVICE=cpu` 或 `cuda`。在该解释器中它会避免导入 Jittor 的
测试参数化器。需要已准备好的源码、工具链和独立缓存；这些命令不会安装依赖。

未纳入版本控制的证据位于 `$JITTOR_LAB_ROOT/_state/vllm-evidence/20261015/`。
`like-factory-validation-manifest-v1.json` 绑定修正后的 oracle/修复前/修复后矩阵、
可移植契约文件、验证日志和两个自有源码文件；SHA256 为
`d80e0b323db19432f296014c27e100ca0ca283cf5671adc7a1e6c49dc28cd9cf`。更早的启动器
与断言协议失败与最终成功的检查分开记录。

基于尺寸的 `ones(..., requires_grad=True)` 诊断仍是另一处未修复的独立边界。
本批次只覆盖四个具名的 like 工厂。它还修正了私有 RMS 输出缓冲区的分配，
但没有在该 adapter 内清除 requires-grad；前面的
[RMS 输出契约报告](2026-10-15-rmsnorm-triton-alias.md)保留其自身的源码与原始标识。
本次 API 修复不带来任何 Omni 模型/引擎验收、新的生态覆盖分子、训练层级或性能主张。

## 主树集成

集成自 `d87f6ba338114ca13e83a7b1072aac931f4175c3` 开始，合并了工厂改动
`7259036ccd0acbfaccb1be04f7bd9cea9d6a2784` 与自有 RMS 修复
`95518509d2038b4a64eef5efd713bb742b5a56f2`。全部 113 个原始产物哈希和七个自有源码
哈希经独立检查。重算保存的证据确认全部 48 行工厂结果和全部 30 个 RMS 数组都在其
原始数值预算之内；原始元数据与测量 SHA 保持不变。

集成后的工厂、构造器邻居、公共 RMS 和私有 RMS 测试在 CPU 上 20/20 通过，
在 CUDA 选择的会话中也 20/20 通过，零跳过。两个会话使用独立缓存；首次编译为串行。
中央 results 索引和生成的源码清单现在包含新报告和 adapter 测试。
集成日志与独立哈希/数值审计未纳入版本控制，位于
`$JITTOR_LAB_ROOT/_state/rms-factory-integration-d87/`。
