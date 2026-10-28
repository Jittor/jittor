# FSDP 优化器视图生命周期与原生 checkpoint 分派

- 状态：有界 CPU 与双 rank CUDA/NCCL 正确性已验证。
- 被测源码：`e5260eb1aa48a95d8106ba9901ec54d850f31dac`，在最终的 CPU、单 CUDA 和
  双 rank CUDA 运行中始终保持干净。
- 日期：2026-10-15。维护者：Jittor 兼容性维护者。
- 复查时机：Adam 参数缓存对齐、FSDP 视图刷新、checkpoint 分派或已安装 Module 的
  加载契约发生变化时。
- 范围：[原生 FSDP 报告](2026-10-15-native-fsdp.md)中记录的两处兼容性失败。
  其原始源码、原始张量、数值阈值和 BF16 限制仍为独立证据。

父级集成：交付的代码与依赖应用在
`71c5be74ef6991b75fa79d73fe52eb04c0319d59` 上，同时包含共享的 BF16 Linear 与 TP
变更。所有交付的源码文件都与其最终参考一致。生命周期清单中的三份文档、38 份日志和
七个 runner 均重新哈希成功。合并的 CPU 兼容性所有者通过 **18** 项，另有一项仅限 CUDA
的类被跳过；真实 CUDA 兼容性所有者通过 **24** 项，零跳过。CPU core 通过 **322** 项，
完整结构在中央文档登记后通过 **1470** 项、六项跳过。原始父级日志与审计未纳入版本
控制，位于 `$JITTOR_LAB_ROOT/_state/distributed-bf16-integration/`。更早的双 rank 结果
保留其被测源码，并不证明原生 FSDP 另一处未解决的 BF16 数值差异。

## 根因与变更

反向 mesh 的自定义 Adam 回归通过首次数值更新，但保留了已被取代的两个 shard 对象。
在每个 CUDA rank 上进行的真实 Python 引用者检查将它们追溯到参数组的
`_torch_adam_params` 元组。同一陈旧的标识缓存可能在下次更新时重置 moment 和
每参数步数。

现有的 Adam 缓存同步器现在接受同一逻辑参数的显式旧/新视图。FSDP 刷新只提供其实际
的条目替换；同步器仍执行其基于标识的重排和空槽初始化。本地测试检查：无需循环 GC 的
弱标记即时释放、moment/variance 对象标识、步数连续性、用户重排、全新的普通参数、
状态选择以及两次真实的 Adam/AdamW 更新。既有的 CUDA 回归在反向的双 rank mesh 上
独立对照 NumPy 检查三次全参数 Adam 更新。

checkpoint 分派器此前把 `strict=` 直接传给原生 `Module.load_state_dict`，
而后者没有该参数。当那个确切的原生方法仍然绑定时，分派现在改用安装环境的 Torch
Module 加载器。自定义覆盖保留其权威。严格键检查、形状检查、dtype 与保持分配的复制
仍由该既有加载器负责；测试检查变更前失败、复制别名与指针、可训练性和梯度。

嵌套 FSDP 回归还提供了一个完整的普通张量字典，却使用默认的
`full_state_dict=False`，同时期望 rank 零广播。一次独立的二进制 Torch 2.11.0+cu128
双 rank 运行在两个 rank 上都以 Tensor/DTensor 混合加载错误拒绝了该输入。同一个真实
模型在显式使用 `StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)`
时，在两个 rank 上都把 bias 恢复为 seven 并执行前向。仓库节点现在请求这些选项，
并复用既有的 FSDP 完整状态加载器。分派不会根据张量形状推断完整状态。

## 源码与环境

该分支获取了远端 `2c0d2681a7db089aa2e996f0ad8e15057fb8e35e`，在编辑前集成为
`361e49578cbc986d8a6a467cf66e1f09838f5d06`。随后它在集成以下完整依赖的同时，
保留了检出之外的本地改动：

- `a6b943d8ca6d6ae31936b152ee53905ebbd61dba`，张量列表清理所有者；
- `7b8230c19ab88f2d182233a24da231b622d77b6f`，保存储的集合通信与
  Module 加载所有者。

它们的本地等价提交为 `db8ba7bec` 和 `c33b72784`。它们是本报告的前置集成；
本报告不为其另外的下游引擎工作授予验收。该修复没有复制它们的加载数学。

候选执行使用 Python 3.11.16、NumPy 1.26.4、CUDA 12.2.140 和 NCCL 2.18.3。
CPU 运行显式禁用了 CUDA 可见性、CUDA 执行和 nvcc。CUDA 运行使用真实的分配设备、
strict kernel 数学以及独立的预热每 rank 缓存。最终的单 rank 与双 rank 运行设置
`backend_fallback=error`，并确认全局后备计数保持为零。首次编译为串行。
主机上既有的无关设备占用排除了性能结论。
独立参考方记录了其实际 Torch `_C` 二进制来源。

## 结果与限制

| 检查 | 结果 | 源码边界 |
| --- | --- | --- |
| 最终 CPU 加载/优化器/集合通信邻居 | 49 通过，零跳过 | 干净被测源码 |
| 最终单 CUDA 等价项 | 49 通过，零跳过；全局 F0 | 干净被测源码 |
| 既有双 rank FSDP CUDA 节点 | 每 rank 8 通过，零跳过；全局 F0 | 干净被测源码，真实 NCCL |
| 独立二进制 Torch 完整状态选项契约 | 默认在两个 rank 上均被拒；显式 full/broadcast 在两个 rank 上均通过 | 真实二进制 Torch，隔离解释器 |
| 原生/Torch 模式下的 CPU core | 322 通过、42 跳过、1 项预期失败 | 源码提交前的同一运行时字节 |
| 原生 FSDP + TP + ZeRO CPU 组合 | 51 通过，零跳过 | 源码提交前的同一运行时字节 |
| 原生 process-group CPU / 兼容 FSDP 邻居 | 12 / 20 通过 | 集成依赖字节 |
| 收集契约以及七个新增本地节点 | 41 通过，零跳过 | 迁移 Torch 测试之后 |
| 迁移之前的完整结构 | 1469 通过、4 跳过、3 失败 | 见下文的确切处理 |

可选的 Accelerate 节点在最终的 49 节点会话中被显式取消选择。此前一次 CPU 运行记录了
其缺失依赖跳过。双 rank 选择器显式排除 `test_two_dimensional_mesh_hybrid_gradient`，
后者至少需要四个 rank。最初的完整九节点运行在每个 rank 上记录了八项通过和该跳过，
并且在严格执行策略下正确地以非零状态退出。两项排除都不被接受为覆盖。

结构运行发现一个新增的 Torch 测试被放在原生收集路径下；将其移到
`compat/tests/torch/test_fsdp_optimizer_rebinding.py` 修复了该契约，并由随后的
41 节点运行验证。余下两处失败涉及已交付的原生 FSDP 文档在中央 toctree 和生成的
MANIFEST 中的登记。那些文件仍由父级集成所有者控制。此处不声称完整结构通过。

Ruff 对另外五个被改动文件通过。`optimizer_api.py` 保留了八个既有的 F401/E701
诊断，可从未修改的前置源码复现；本补丁未新增任何一项。空白与仓库布局检查通过。

新增的定向节点为 `compat/tests/torch/test_fsdp_optimizer_rebinding.py` 和
`compat/tests/torch/test_checkpoint_native_module.py`。请在分别配置的 CPU 和 CUDA
会话中运行它们，设置 `JITTOR_TORCH_SHIM=1`、`PYTHONPATH=python`，
并把 `JITTOR_TEST_DEVICES` 设为实际后端。CPU 必须如上显式禁用 CUDA。
既有的多 rank 入口为 `compat/tests/fsdp2/test_fsdp2_nccl.py`；
请使用[分布式验证流程](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/skills/jittor-distributed-verification/SKILL.md)，
配合独立的 rank 缓存和记录的双 rank 选择器。确切的外部 runner、环境文件和选择器
由下方的产物清单绑定。

## 原始证据

产物位于仓库之外的
`$JITTOR_LAB_ROOT/_state/native-fsdp/20261015/artifacts/compat-lifecycle/`：

- `SOURCE.json` 绑定干净源码 SHA 和十一个相关源码哈希。
- `AUDIT.json` 验证最终结果协议、全局 F0、源码哈希、参考选项行为以及原始原生
  FSDP 封存。
- `SHA256SUMS.json` 绑定三个元数据文件、38 份原始日志和七个 runner 或环境文件。
  SHA-256：
  `38a93c8ce1e8854d9f22034dc6e8a45dd9d52dd9f7bf5c464d8faef1b72500ac`。

清单保留了修复前/后的日志、失败的严格九节点调用、原始的参考检查以及最终干净测量。
先前的原生 FSDP 封存仍为
`9901d1ed7dd834dae4d11b4afdb501a37d2cdc3bc692019daad2d6a0c27c9a27`；
全部 236 个原始产物、13 份日志和五个 runner 都已独立重新哈希且仍然一致。
本次后续工作不重新指定其基线，也不改变待定的共享 Linear/BF16 数值审计。
