# Tensor detach 与工厂梯度边界

- 状态：CPU 与 CUDA 契约已验证；DeepSpeed 多 rank bf16 验收仍未完成
- 日期：2026-10-15
- 基线提交：`4a80afb6606c428c200ba84a29d178abe52284b9`
- 源码修复：`4d2936e4340995767694f9c1794e7b7690f8e803`
- 同步验证：`9f1dc97df75898791e7f8e33e537c09ba7b1afd0`，含远端 `aec3b683bffdd85fae42b3b94d8609c0eed7cf2f`
- 维护者：Jittor 核心与 Torch 兼容层维护者
- 复查条件：原生 view 来源、holder 重绑定、梯度发布、张量工厂或前向模式 AD 支持变化

## 已确认缺陷与 owner 修复

真实两 rank DeepSpeed 0.19.7 bf16 训练先到达原生 NCCL 初始化
与 SUM 归约，随后因缺少 `Tensor.detach_` 失败。独立
二进制 Torch CPU/CUDA 契约确认：它返回同一张量、保留
storage 与已有梯度、保留 detach 之前创建的消费者，
并拒绝 view，包括其临时 base 对象已销毁的 view。

兼容方法现在复用原生叶梯度策略或原生
共享 storage 的 detached 别名。原生只读来源查询在其弱写回 owner 过期后
仍保留 view 身份；storage 与 view 写回
归属未变。保真度仍为近似：被测试的反向模式
契约适用，而前向模式 AD 不支持。

默认 backward 此前还会发布带可微历史的非优化器梯度。
现在在 `create_graph=False` 时将其 detach；`create_graph=True`
保留历史并通过真实二阶导数测试。重复 backward
仍累加到同一梯度对象。

最后，在 CUDA 上构造 `torch.tensor(..., dtype=bfloat16, requires_grad=True)` 会产生
cast 非叶并遗漏其请求的梯度。工厂
现在在启用梯度之前 detach 其已完成的分配/转换。
`tensor(source)` 仍然复制并切断历史；`as_tensor(source, dtype=...)`
保留转换历史。已执行 CPU float32/float64 与 CUDA float32/float64/bf16；
本报告不声明 CPU bf16 算术。

## 验证

Python 3.11.16，Linux x86-64，RTX 4090/SM89，驱动 580.178.04。原生 CUDA 使用
真实 nvcc 12.2.140 与 provider `pipcu122_a89fa4fed08c`。独立参考
为二进制 PyTorch 2.11.0+cu128；其 `_C` 扩展与缺失的 shim 标记均已
检查。每个 runtime/设备与分布式 rank 使用独立外部缓存；
首次原生/NCCL 构建串行执行。并发的无关任务使得无法给出
性能结论。

原始日志未版本化，位于
`$JITTOR_LAB_ROOT/_state/refactor-goal/deepspeed/`。可移植复现使用
独立设备进程、`JITTOR_TORCH_SHIM=1`、固定源码的 `PYTHONPATH`
与任务本地 `JITTOR_HOME`/`TMPDIR`；CUDA 另设
`JITTOR_TEST_DEVICES=cuda`、已分配的 `CUDA_VISIBLE_DEVICES` 与真实 nvcc。

| 同步源码上的验证 | 结果 | 日志 |
| --- | --- | --- |
| `forbid_backend_fallbacks()` 内的新 detach/工厂 CUDA 契约 | 9 passed, zero skipped, fallback 增量为零 | `detach-factory-cuda-synced-no-fallback.log` |
| autograd/hooks/callbacks/optimizer/view/factory/dtype 回归，CPU | 170 passed, one existing xfail, zero skipped | `detach-expanded-cpu-synced-v3.log` |
| 同一扩展回归，CUDA | 170 passed, one existing xfail, zero skipped | `detach-expanded-cuda-synced-v2.log` |
| 官方 `python tools/run_test_suite.py --tier core`，CPU，两种进程模式 | 322 passed, 42 skipped, one existing xfail | `detach-core-cpu-synced-v2.log` |
| 完整 Torch 模式 `python -m pytest -q tests/structure` | 1460 passed, six skipped | `detach-structure-synced-v1.log` |
| 独立 Torch 分别执行同样七个 detach 与两个工厂方法，CPU 与 CUDA | 每设备 9 methods passed | `detach-oracle-contract-{cpu,cuda}.log`, `factory-leaf-oracle-{cpu,cuda}.log` |
| `bash tools/check_repo_layout.sh` | 报告登记前通过 | 终端输出 |

既有 xfail 为 `test_placeholder_dtypes_are_metadata_only_at_every_boundary`
（`KI-COMPAT-014`）。Core 的 skip 反映其既有的设备/opt-in 覆盖；六项
结构 skip 由两项 accelerator 检查、两项 declared 检查与
两项未安装 Jittor 发行物的隔离解释器组成。表中的计数
互相重叠，不可相加。

用
`python -m pytest -q compat/tests/torch/test_tensor_detach_inplace.py compat/tests/torch/test_tensor_factory_leaf.py` 运行新契约。
扩展命令另加既有的 gradient-hook、callback、task-participation、
autograd/semantics、view-storage、optimizer、autograd-library-owner、factory-fidelity、
dtype 与 tensor-to-dtype 测试文件。源码哈希在外部封存在
`detach-source-synchronized.json` 中。

首次 core 尝试在等待 fresh oneDNN 源码下载时被中断；
被接受的运行显式使用已存在的真实 oneDNN include/library 对。
更早的 CUDA 工厂失败与 DeepSpeed `detach_` 失败保留在外部
日志中。这些修复之后，真实两 rank bf16 backward 推进到下一个
缺失接口 `torch._foreach_zero_`；此处不声称已完成多 rank Engine 训练
或数值比较。该验收需要独立的
优化器分区、全梯度、训练与 fresh-Engine 续训证据。

## 主分支整合验证

改动整合于 `9cc83da6e2b32a170c23b7558ab445b75c5b4fd4` 之上。
全部七个源码与测试文件与上面列出的同步验证提交一致；
哈希记录在未版本化的
`$JITTOR_LAB_ROOT/_state/tensor-detach-integration/source-equivalence.json` 中。
在整合树上，CPU core 通过 322 项测试、42 项 skip 与一个既有
xfail，两个新契约文件分别在 CPU 与真实 CUDA 上通过九项测试，
零 skip，完整 Torch 模式结构套件通过 1460 项测试，
skip 仍为上述六项。仓库布局与差异检查通过。
对应日志为该外部整合目录中的 `core.log`、`cpu-contracts.log`、`cuda-contracts.log`
与 `structure.log`。这些检查验证整合后的张量改动；
多 rank DeepSpeed 验收仍待完成。
