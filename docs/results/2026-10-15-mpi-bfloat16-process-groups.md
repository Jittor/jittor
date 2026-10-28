# CPU MPI BF16 通信与进程组

- 状态：有界通信验收；DeepSpeed CPU BF16 数值阶梯仍待完成。
- 日期：2026-10-15
- 基线：源码提交 `2de23a18e8788f901eaf08f9a0c4ef8e0d432876`；
  同步后的本地合并 `49a95fdde79869341af22eae98a7dd0972e01536` 包含
  远端 `824cb354d5cced0743b3207d50d518111ba59e76`。
- 维护者：Jittor 分布式与 compat 维护者。
- 复查时机：MPI 数据类型/操作表、通信子生命周期、集合通信梯度、原生进程组、
  compat 集合通信路由或保存储发布发生变化时。

## 已验证的行为

原生 MPI 所有者现在把 BF16 作为不透明的两字节元素传输，并使用由既有核心转换/舍入
类型构建的独立 BF16 SUM 回调。BF16 算术永不进入 FP16 回调。实际的 MPI 通信子来自
`MPI_Comm_split`，保留请求的 rank 顺序和独立上下文。所有 WORLD rank 必须以相同顺序
创建组。集合通信反向传递相同的通信子；mean 使用其大小，reduce/broadcast 使用其本地
root。通信子直到关闭前都由 worker 持有，与既有的组生命周期一致。

MPI gather 是真实的逐组 broadcast 组合。MAX/MIN/PRODUCT 是真实 gather 加上既有的原生
逐元素操作；它们被明确为组合实现，并拒绝可微输入。这不是 MPI all-gather 或非 SUM
融合 kernel 的主张。兼容层的扁平 gather 与分组 broadcast 转发给原生所有者。
既有的 compat 子组限制保持不变；DeepSpeed 所需的整 world 克隆在此覆盖。

可复用的[集合通信 oracle](../../agent/skills/jittor-distributed-verification/collective_oracle.py)
在隔离的运行时中执行相同的公共 API 主体。每个运行时/设备都有全新的产物目录；
独占的 rank 认领文件拒绝重复运行。
它检查 BF16 原始载荷（有符号零、次正规数、无穷和 NaN 载荷）、
有限 SUM 的并列情形、uint8 溢出、FP32/BF16 的 SUM/MAX/MIN/PRODUCT、Parameter/data/detach/
view 的保存储发布、更早克隆的不可变性以及可微 SUM。

| 候选方 | 独立 oracle | 实际 rank 数 | 比较字段数 | 结果 |
| --- | --- | --- | --- | --- |
| CPU / OpenMPI | CPU / Gloo | 2 | 每 rank 36 | 逐位一致 |
| CUDA / NCCL | CUDA / NCCL | 2 | 每 rank 36 | 逐位一致 |

全部 144 个比较数值字段匹配。元数据单独检查；MPI 不被呈现为原生 Gloo。候选输入/输出
在请求的设备上以 fallback-error 策略物化，计数增量为零。CUDA 回归执行了共享的非 SUM
折叠以及分组 gather/broadcast 路由。

## 平台与回归

Linux x86-64、Python 3.11.16、GCC 11.5；私有 OpenMPI 5.0.11。官方
[OpenMPI wheel](https://pypi.org/project/openmpi/5.0.11/) 的 SHA256 为
`838de0ec7fd719af19cde664cd6bf76eb170ed4769c279040c2db434d6db11b9`。
其完整的 `.data/data` 布局被解压到私有前缀，包括初次仅安装目标时遗漏的库。
MPICH 未被接受：既有的编译器探针使用了 OpenMPI 特有的 `--showme` 选项。
未修改任何共享环境。oneDNN 使用显式指定的既有安装。

独立 oracle 是二进制 Torch 2.11.0+cu128，带有其真实的 `_C` 扩展且无 shim 标记。
CUDA 在两块真实的 RTX 4090 / SM89 设备上运行，驱动 580.178.04，
使用原生 pip-cu122 工具链和 nvcc 12.2.140。这些 tiny 通信 fixture 不产生任何性能主张。

- 真实双 rank MPI 回归：**每 rank 20 通过、0 跳过**，包括带非零组内 root 的
  BF16 reduce/mean 及其反向、反向 rank 顺序、通信子独立性、非 SUM 的 AD 拒绝，
  以及既有的 dtype、FP16 标量/归约和 MPI 算子测试。
- Compat 路由所有者：**7 通过**；原生进程组宿主契约：**13 通过**。
- CPU core：**322 通过、42 跳过、1 项既有 xfail**，两种进程模式均如此。
- 完整结构：**1497 通过、6 跳过、2 项启动器失败**，原因是禁用插件自动加载时未显式
  加载 xdist。使用 `PYTEST_PLUGINS=pytest_timeout,xdist.plugin` 后，该文件整体通过
  **10/10**。更早的 core 启动器失败同样发生在测试执行之前，并已用 timeout 插件重跑。
  原始失败日志保持完整。
- 仓库布局检查通过。不作任何 smoke/full 运行时或非 x86/其他 MPI 实现、HCCL、
  加速器 MPI 子组或更大 world 的主张。

最终的文档/私有路径检查通过 18 项测试，另有一项待定的中央集成检查：本新报告必须由
父级集成所有者登记到 results toctree。共享索引与 release 清单被有意留给该所有者。

## DeepSpeed CPU 边界

官方 DeepSpeed 0.19.7（`f5af15c2b4da10411e340850c00ba0cb5b14fd87`）可以在二进制
Torch 中以两个 Gloo rank 在 CPU BF16 上执行既有的真实残差模型。这是通过实际的 Engine
backward/step/save/load 测试的，而不是从既有 runner 先前的仅 CUDA 守卫推断而来。

同一个候选模型、确定性权重、精度和六微步协议现在在 CPU MPI 上执行：
ZeRO1 检查 64 个严格续跑字段，ZeRO2 检查 56 个；全部 166 个本地 FP32 master 元素在
每个 rank 上都更新。这些是带有诊断元数据的 API 探针，不是既有门禁准入。
它们仍未通过未改动的外部数值预算（前向/状态 0.002，梯度 0.01）：

| 阶段 | 最差前向/状态 | 最差梯度，rank 0 / 1 |
| --- | --- | --- |
| ZeRO1 | 0.0717949 | 0.662354 / 0.719285 |
| ZeRO2 | 0.0717949 | 0.294964 / 0.456353 |

单独的算子探针使用确切的参考 BF16 输入和余切。
up/down/head Linear 的输出分别在 6/32、2/16 和 4/8 个元素上不同，
而它们孤立的输入/权重/bias 梯度逐位一致。LayerNorm 的孤立输出逐位一致，
但全部 16 个输入梯度元素和 4/8 个权重梯度元素不同（最大绝对差异分别为 `2.32756e-5`
和 `6.10352e-5`）。Bias 梯度匹配。这些探针保留中间值，明确属于诊断性质；
原始的 Engine 失败保持不变。两个 CPU 数学所有者都需要单独调查。
本 MPI 批次不授予任何 CPU 配对 scorecard、CUDA 覆盖分子或更广的 DeepSpeed 支持。

## 复现与证据

遵循[分布式验证 skill](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/skills/jittor-distributed-verification/SKILL.md)
了解独立的 rank 环境、串行首次编译以及真实 Torch 标识检查。在全新根目录中以
`--runtime jittor|torch` 和 `--device cpu|cuda` 运行 oracle；
精确比较每个数值 NPZ 字段，并单独审计元数据。原生 MPI 回归入口为：

```bash
mpirun -n 2 <private-rank-environment> <jittor-python> <native-pytest-wrapper> -q \
  tests/backends/comm/mpi/test_mpi_bfloat16_groups.py \
  tests/backends/comm/mpi/test_mpi_dtypes.py::TestMpiCollectiveDtypes \
  tests/backends/comm/mpi/test_mpi_half_reduce.py::TestMpiHalfReduce \
  tests/backends/comm/mpi/test_mpi_half_scalar.py \
  tests/backends/comm/mpi/test_mpi_op.py::TestMpiOps
```

原生 pytest 以 `JITTOR_TORCH_SHIM=0` 运行；compat 所有者测试与结构在各自独立的
Torch 模式进程中运行。Core 使用
`python tools/run_test_suite.py --tier core --serial-compile`。

未纳入版本控制的证据位于
`$JITTOR_LAB_ROOT/_state/refactor-goal/deepspeed/cpu-bf16/`：
`mpi-bfloat16-groups-v3-evidence.json` 的 SHA256 为
`7e72563a214d02929b8856a030a974a428efdcd7f0c8011c92db7b0c4f6d5ed7`。
它把 1,570 个运行时源码哈希绑定到源码提交、oracle 脚本哈希、
94 个原始证据哈希以及四次 rank 比较。`collectives-*-*-v2/` 包含最终原始数据包；
更早的 v1 产物保持不变。`audit-mpi-bf16-v3.py` 重算源码字节和数值相等性。
`launch-collective-oracle.py`、`run-native-pair.sh`、`native-env.sh`、
`run-cuda-collectives.sh` 和 `run-cpu-gates.sh` 保留确切的本地环境。

清单还保留了原生 Stage1/2 API 数据包、原始的 unsupported Gloo/BF16/group 探针、
其匹配的提交前 MPI 补丁以及孤立的 Linear/LayerNorm NPZ 文件。独立的 Torch CPU Engine
证据仍保留其原始 `f945b223cd4316feaa733556ef1413344f98e11f` 基线，位于
`cpu-bf16-api-oracle-v1-evidence.json`；它不会被重新标注为最终的配对门禁。

## 主树集成与启动器边界

集成自 `0751e425b0f197b8e7b388f218613378c71f701a` 开始，加上上述 MPI 源码和证据提交。
全部 1,570 个原始运行时源码哈希、94 个原始哈希和 144 个数值字段都经过独立复核。
随后主树 CPU 运行执行了两个真实 MPI rank：全部 72 个数值字段与保留的独立 Torch/Gloo
参考字节一致。既有 MPI 回归集每 rank 通过 20 项测试且无跳过；原生/compat 所有者契约
分别通过 13 和 7 项测试。Core 通过 322 项测试，42 跳过和一项既有 xfail。

CUDA 集成通过其文档化的等价入口
`python -m jittor.distributed.launch --standalone --backend nccl
--nproc-per-node 2` 使用公共 `jtrun` 所有者。任务本地的 rank 包装器只选择独立、
串行预热的缓存；`jtrun` 提供了 rank、设备和 rendezvous 设置。
两个真实 CUDA rank 都完成了全部 72 个数值字段，与保留的独立 Torch/NCCL 参考字节一致，
并检查了设备驻留和零后备。

主要的加速器训练入口仍为 `jtrun`。其当前支持的后端为 NCCL 和 HCCL。
此处的 CPU `mpirun` 证据验证的是保留下来的 MPI 后端；它不确立 `jtrun` 中的 CPU 支持，
也不确立新的 MPI 训练入口。上文的 CUDA 启动器测量与子进程原先的私有双进程启动相互
独立，保留各自的命令与日志。

一块最初分配的 CUDA 设备在串行预热期间、任何集合通信运行之前，就在
`cudaMemGetInfo` 中以 `cudaErrorECCUncorrectable` 失败。失败的预热被保留；
未执行设备重置。另一块已分配的设备通过了实际的 UUID/驻留检查以及随后的 `jtrun` 运行。
oracle 的两处仅排版编辑与最初封存的脚本保持相同的 AST。原始集成证据与命令未纳入版本
控制，位于 `$JITTOR_LAB_ROOT/_state/mpi-integration-282/`；该目录名标识集成任务，
而实际基线是上述完整 SHA。
