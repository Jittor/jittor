# 原生 ZeRO1 训练与 checkpoint 验证

状态：原生优化器状态分片已实现，并在 CPU 与
单机双 rank CUDA/NCCL 上验证。维护者：Codex 原生 distributed owner。
优化器算术、collective 调度、Store、checkpoint
或 trainability 语义变化后需复查。原生 FSDP/TP/ZeRO2/3 与多主机
验收不在本结果内。

## 源码与缺失能力

初始同步的远端为 `b930461562e73c67332581fe2384eaf8cfd7d838`；
既有原生 checkpoint 实现保留为
`82fb93150fc51e0608db7f95e08f8d48bb7f78ec`，并重新应用为
`fefcee969b96e61f1d592eee0f74c11ed4f95bc4`。停止 CPU 复现后，
合并了远端 checkpoint 整合 `82dda1ff5`，保留其报告
更新。最终受测源码基线为
`b8c150f07affd8b9651fba772f98816e61174be3` 加本次改动。原分支
引用与同步前证据均已保留。

三个 CPU 原生 Adam 步与独立 NumPy 轨迹一致，随后对
`jittor.distributed.ZeroRedundancyOptimizer` 得到真实的 `ImportError`。
新的 facade 在 backward/DDP 中使用既有的完整 base Optimizer，
并只用 owned 参数实例化既有原生算法。它
不 import 任何兼容 Torch 实现，也不产生重复的优化器
数学。原生 checkpoint preflight 在改动 model/RNG 之前，
把 rank 拥有的状态校验委派给优化器。

## 结果

私有 Python 3.11.16 环境使用 NumPy 1.26.4 与 pytest 7.4.4；
CUDA 使用 nvcc 12.2.140、严格 kernel 数学与隔离的 NCCL 2.18.3
runtime。源码 import 与每个 rank 的缓存都固定在本工作树。
每个 rank 的首次编译（含 BF16 预热）串行执行。

| 检查 | 结果 | 证据（位于 `$JITTOR_LAB_ROOT/_state/native-zero-training/`） |
| --- | --- | --- |
| 缺少公共入口，附带原生 Adam/NumPy 基线 | 预期 import 失败 | `logs/cpu-missing.log` |
| 原生 ZeRO CPU 加原生 checkpoint 回归 | 45 passed，0 skipped | `logs/cpu-zero-final.log` |
| 真实 CUDA/NCCL worker | 每 rank 24 passed，0 skipped，launcher exit 0 | `logs/two-rank-validated-launch.log`, `logs/two-rank-validated/rank{0,1}.log` |
| CPU core，两种进程模式 | 322 passed，42 skipped，1 个已知 xfail；两者 exit 0 | `logs/core-cpu.log` |
| 完整 structure | 1451 passed，6 skipped，3 failed；测试 basename 已修正且其两项检查通过，剩两项中央文档登记失败 | `logs/structure.log`, `logs/structure-registration-recheck.log` |
| 仓库布局与 diff whitespace | 通过 | `logs/layout.log` |

在完整 structure 门禁发现与 CPU 测试 basename 冲突后，CUDA 测试
被重命名为 `test_native_zero_nccl.py`。其内容哈希与
真实双 rank 运行时一致。剩余的 structure 失败需要
整合期间完成中央文档 toctree 与生成的 MANIFEST 登记；
此处不计为通过。`final-source.json` 记录
受测代码哈希，包括这次仅改路径的重命名。

CPU 覆盖三种算法、多个 group、两个 micro-batch 的累积、
冻结参数、延迟解冻与共享参数。普通轨迹
与未分片原生 owner 逐元素相等，并与
独立 FP64 NumPy oracle 一致。NumPy 显式实现原生 Adam 历史上的
epsilon-before-bias-scaling 约定；不把它当作
与 PyTorch Adam 逐位一致。

CUDA 覆盖 15 个算法/配置行、三个延迟解冻行、
三个删除 group override 行、BF16 SGD、tied/no-sync 与 rank
不匹配。每个 checkpoint 行训练三步、保存、再执行两步，
然后恢复到既有与全新 owner，并与完整续训精确一致，
包括 owned moment、梯度、模型值、步数
与 loss。被删除的 group LR override 通过保存的当前默认值续训。
所有设备工作都在 fallback-error 策略下运行，计数为零且不
import Torch。活跃输出、梯度、更新后参数与 moment
在读回前于设备上检查，包括 CUDA driver 指针 ordinal。
未触碰的冻结零张量单独物化并按
存储检查；host 快照不被描述为 GPU 计算。

在 15 个主要 CUDA 行中，相对独立参考的最大绝对误差
低于 `1e-7`；检查保持 `atol=2e-6`、`rtol=2e-5`，loss 绝对
误差低于 `2e-6`。精确续训使用数组相等，而不是这些容差。
BF16 SGD 使用可精确表示的 dyadic 输入与更新，检查
参数原始位，保留 BF16 momentum，并与独立 NumPy SGD 一致。

对一个八元素、两参数模型，Adam/AdamW 在 rank zero 上构造
12 个状态元素、rank one 上 4 个，而不是每个 rank 16 个。SGD
构造 6 个与 2 个，而不是每个 rank 8 个。构造器插桩证明
只有 owned 参数身份进入状态 buffer 创建；driver 检查
证明活跃的 owned buffer 是真实 CUDA 分配。rank 本地
checkpoint payload 计数一致。只有一个参数时，非 owner 构造
零个 moment，仍能正确训练/恢复。合并后的优化器状态
元素是复制式双 rank 总量的一半；这不是整进程
内存或性能结论。

NCCL 的实际 TRACE 记录每个 rank 有 463 次 AllReduce 调用：68 次
metadata reduction，以及 24 个工作负载所需的恰好 395 次梯度 reduction。
对三步、两参数的 DDP 累积窗口，Python owner spy
还报告六次梯度 reduction，而不用 DDP no-sync 累积时为 12 次。
物理调用排除了 owned 更新期间隐藏的第二次 reduction。

## 保留并修复的复现

第一个 CPU 矩阵发现：把可训练状态恢复到初始
冻结的目标上后，目标仍然冻结。原生 checkpoint helper 现在会在加载后
对既有目标对象重新应用保存的 flags，并双向验证。
错误的 partition/algorithm/moment/梯度 metadata 与意外的复制
moment 会在模型改动前失败。编辑导出的 ownership metadata
不再改变活跃 ownership。group override 删除会被转发并
恢复，不保留新对象上过期的 override。

一个未观测的 tied 常数梯度更新在 no-sync
拒绝之后最初会挂起。独立探针保留了原始进程栈，并定位
到 collective 梯度与非对称 owned 更新之间的边界。
在所有 rank 上显式做精确根梯度物化并加入就绪同步
解决了问题，未使用 NumPy 读回，也未改动数学。另一次运行暴露出
清理阶段 RPC/collective 的 GIL 竞争；对 peer 使用既有原生 Store 的
`arrive` 契约并由 root 负责清理后解决。预先 eager 加载 CUDA 模块
与冗余重启叶子都未能解决原始挂起，
因此未作为修复手段。原始失败运行保留在 `logs/two-rank-first/`、
`logs/two-rank-computed-storage/`、`logs/two-rank-gradient-boundary/` 与
`logs/tied-*` 探针中；最终完整证据是 `two-rank-validated`。

## 复现与限制

用 `JITTOR_TORCH_SHIM=0`、`JT_BACKEND=cpu`、
`JT_USE_CUDA=0`/`use_cuda=0` 两者都设、两个 NVCC 路径变量都为空、自有 `PYTHONPATH`
与私有 `JITTOR_HOME` 运行 CPU 原生检查。对 `tests/optim/test_native_zero.py`
与 `tests/serialization/test_sharded_checkpoint.py` 运行 pytest。双 rank 套件
使用 `python/jittor/distributed/launch.py --backend nccl -n 2 --device-ids=<two>`，
并对 `tests/backends/comm/nccl/test_native_zero_nccl.py` 运行 pytest，各 rank
使用独立 home/缓存，`JITTOR_ZERO_TEST_DIR` 指向共享 checkpoint 存储，
`JT_CUDA_KERNEL_MATH=strict`、fallback-error、不启用 shim。实验的
`cuda-env.sh` 与 `rank_entry.py` 保留精确命令与环境。
禁用 pytest 插件自动加载时，显式加载 `pytest_timeout,xdist.plugin`。
使用仓库固定的 pytest，而不是未经验证的更新大版本。

这是原生优化器状态训练分片，而不仅是文件分片。
参数/梯度保持复制，分片以整个参数为单位。
group 与参数顺序固定；checkpoint 的 world/ownership 必须匹配。
原生 DDP 的 trainability 在构造后固定。大模型峰值
内存、吞吐、其他算法/dtype/后端、TP、参数分片、
ZeRO 2/3 阶段与两台真实主机都未由本次运行验收。两个本地
worker 不是多主机证据。本报告不把任何未支持的能力登记
为完成。
