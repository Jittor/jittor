# 架构迁移整合批次

2026-09-08，协调分区。状态：跨层代码、统一验证与打包核验完成；不代表全计划完成。
复查触发：存储ABI、dtype转换、执行提交或后端输入契约变化。

本批一起整合真实dtype、原生storage strides、执行边界、独立前端状态、FSDP架构与
唯一文档树。没有逐小改重新提交或跑完整套件。合并保留了共享Adam数学、mesh范数规则、
新dtype转换和显式视图写回，未用整文件取一侧解决冲突。

| 边界 | 稳定契约 |
| --- | --- |
| `src/core`、`src/runtime` | 构造只建图；提交时解析动态形状；设备等待经有入口锁的GIL作用域 |
| `Var` storage | 逻辑shape与物理span分离；strides由native拥有；expand及正步长基本切片共享存储 |
| `src/codegen`、七元算子 | 读取真实stride；未知布局支持的kernel在图中显式接ContiguousOp |
| dtype | metadata用dtype_name，真正消费用dtype_for_compute；Python与C++注册同一dtype类 |
| 独立Torch | 真Tensor/Parameter/Module类型；私有弱owner路由；不把状态表写回native模块 |
| FSDP | 真ProcessGroup/mesh；复用native更新数学；持久local shard拥有自己的allocation |
| 文档 | 看板、设计、结果归docs；Agent目录仅保留协作入口、skills、scripts |

详细接口分别见[存储](../architecture/storage-layout.md)、
[dtype](../architecture/torch-dtype-boundary.md)、
[提交边界](../architecture/dynamic-shape-submission.md)。

验证日志位于未版本化的`/home/zy/jittor-lab/_state/compat-dist-OIR3DI/`：

- `architecture-batch-cpu.log`：合并dtype/runtime/FSDP后37 passed、2个CUDA前置skip。
- `architecture-batch-cuda.log`：18 passed；CUB where还残留旧executor.h引用，已删重复旧include。
- `architecture-storage-cpu.log`：再整合strides后34 passed、2个CUDA前置skip、6 deselected。
- `architecture-storage-cuda.log`：最终存储5项和动态CUDA读回1项，6 passed、0 skipped。
- `gil-cuda.log`：握手/异常/无锁对照与真实CUDA并发和fetch回调，5 passed。
- `owner-small.log`：单步Adam数值及optimizer/Parameter释放通过，无native状态别名。

文档治理164个活跃Markdown、153文件相对链接与仓库layout检查通过。
隔离分区证据保留于[dtype契约](../architecture/torch-dtype-boundary.md)、
[FSDP记录](2026-09-08-fsdp-mesh-optimizer-lifetime.md)与[JIT/转置记录](2026-09-08-jit-target-and-transpose-views.md)。

最终整合后的四rank小回归每rank3 passed、0 skipped：自定义Adam三步/生命周期、
2×2 hybrid梯度、group-aware norm。日志为
`/home/zy/jittor-lab/_state/fsdp-fast-7-13/integrated-four-rank.log`。

统一打包产物（未版本化）：`/home/zy/jittor-lab/_state/integrated-package-SPQsqv/`。
core/compat wheel通过auditor、文件交集0；1156个生产文件与快照逐字节相同，874项
Python/C++源码覆盖完整，新增dtype/contiguous源码进入wheel与sdist。
compat sdist重建wheel所有成员字节与直接wheel相同。没有为打包再次运行JIT。

| 产物 | SHA-256 |
| --- | --- |
| core/jittor-1.3.11.0-py3-none-any.whl | 9c7b988d1daeaf78f3e4cfa538dee070fa47e0dd0c2369ea4245a74f9a1b3f18 |
| compat/jittor_torch-1.3.11.0-py3-none-any.whl | ea40ab2c67ae8fa97f9ec5d9a96a6567be3a7e4ebd48bc73874289ef6027a8c0 |

限制：NPU/ROCm无本批实机证据；ACL存储描述符仅桩SDK语法验证。未运行全量suite，
新包未另作安装后冷JIT复验。FSDP真实used高水位仍高于full，按用户要求性能调优后移。
安装延迟补丁、API模块级对象、其余工具/测试目录与部分接口契约仍有独立未完成任务。
