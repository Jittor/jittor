# 原生 TP linear 训练与 checkpoint 续训

状态：限定范围的原生实现；FP32 CPU singleton 与单机
双 rank CUDA/NCCL 验证。维护者：Codex 原生 distributed owner。
原生 collective adjoint、优化器梯度路由、Module
参数 ownership、checkpoint serialization 或 CUDA 后端变化后需复查。

初始同步基线：`0cddc76749e4f881d587a1f9ee865b9129464883`。
初始实现与测试保留为 `02864239b`。
所有测试进程停止后，工作树 fetch origin，在
`aec3b683bffdd85fae42b3b94d8609c0eed7cf2f` 新建分支并把改动
重新应用为 `e7cd1a467`。未使用 stash、丢弃本地工作或旧 owner 覆盖。
传入的 checkpoint optimizer preflight 与 trainability 修复
被保留。最终生产源码为 `416d1915e`（`e7cd1a467`、common-layout
校验 `17275bed3`、懒加载公共导出 `416d1915e`）。仅测试的
后续 `dc63a31a4` 把 fresh-process import 探针改为走
仓库子进程 helper；所有生产文件哈希保持
不变。该仅测试改动后重跑了 CPU/host 检查。
最终设备 observer 还断言实际 dtype 为 FP32。

## 已实现范围

规范的 native owner 是
`python/jittor/distributed/tensor_parallel.py`，通过
`jittor.nn.parallel` 导出为 `ColumnParallelLinear` 与 `RowParallelLinear`。
它 import 原生 Module、原生 linear 数学与原生 collective，不
import Torch 或兼容前端。见
[机制与调用契约](../notes/native-tensor-parallel.md)。

Column 的 weight/output feature 与 bias 是真正的本地张量；row
的 weight/input feature 是真正的本地，而 row bias 为复制。
原生 Function 提供通信 adjoint：copy 的反向为求和，
row reduce 的反向为恒等，gather 的反向为切片，scatter 的反向为
gather。复用原生优化器更新数学。被 TP 标记的参数
跳过 legacy 数据并行梯度 reduce 与周期参数
broadcast，后者的语义会混入无关的本地分片。对相同参数做 DDP
包裹，以及优化器注册的额外输入梯度，都被
显式拒绝。

高层原生训练 checkpoint 把每个真正分片的
模型张量与优化器 buffer 保存为 rank 本地叶子。它保留
对公共 partition metadata、row bias 及其 buffer、优化器 metadata 与普通
模型参数的复制一致性检查。它记录有序 FQN、rank/world、方向与
本地/全局形状，并在改动目标前校验 ownership。它
复用既有的原生 model/optimizer/RNG checkpoint owner。

## 独立参考与验收

提交的 fixture 用 NumPy float64 解析地计算
整个未分片 MLP：Linear(3,6)、tanh、Linear(6,4)、均方误差
以及学习率 .05、momentum .6、dampening .1、weight
decay .01 的原生 SGD 语义。输入与初始权重固定，同时覆盖
`(4,3)` 与 `(2,2,3)` 两种输入，以及 column gather/row scatter 两种选择。

独立安装的**真实 PyTorch 2.11.0+cu128**（经包路径
与实际 `torch.Tensor` owner 核实）在 **CPU float64** 上运行同一未分片模型与 SGD，
五步、两种输入形状。参考脚本
不 import 任何 Jittor runtime。在输出、输入梯度、全部四个
参数梯度、全部更新后参数、全部 momentum buffer 与 loss 上，
相对解析参考的最大绝对差异为
`1.11e-16`。这确立了独立参考；不代表
Torch CUDA 运行了本任务。

Jittor FP32 结果用 `atol=2e-6, rtol=2e-5`（loss 绝对误差 `2e-6`）把每个字段
与该未分片参考比较。checkpoint 验收
要求与未中断的同 runtime 续训**数组相等**，
不放宽跨框架容差，也不以
仅看配置的断言代替。

四个训练行各自运行三步、保存、再继续两步，
加载全新的 model/optimizer 对象并重复这两步。每个 rank
检查完整的复制输出/输入梯度、其全部本地参数
梯度/更新/momentum，以及复制的 row bias。另有两个独立的
column/row 测试检查加权 loss 的 adjoint 与不可整除 feature
的拒绝。负向 checkpoint 测试故意让复制的 row bias 漂移，或
跨 rank 构造不一致的全局 partition。两者都必须在
两个 rank 上失败且不产生完整 manifest。只有 layout 的 rank 字段是
rank 本地；公共等分 partition schema 保持复制。

这个小 MLP 每个 rank 保存 **28 个参数元素**，未分片参考为 52 个：
column `(3,3)` 与 `(3,)`，row `(4,3)` 与 `(4,)`。
两个层实例都恰好只有自己的两个本地参数 Var。CUDA
observer 同步实际输出、输入梯度、全部参数
梯度、更新后参数与 momentum buffer，断言 FP32，在指针物化**之前**
检查原生 CUDA dispatch 与设备位置，
并查询 CUDA driver 指针归属。两个 rank 使用各自独立的
可见 CUDA 设备。这些是存储与执行断言，不是推断的
内存节省，也不是整进程峰值内存测量。

## 结果与保留的失败

| 运行 | 源码 / 结果 |
| --- | --- |
| 缺少公共 owner 复现 | 初始基线：原生 import 抛出 `ModuleNotFoundError` |
| 初始 CPU TP + checkpoint | 29 passed，无 skip |
| 初始 CUDA 最小 column adjoint | 每 rank 1 passed，真实 NCCL |
| 初始 CUDA 完整 | 每 rank 6 个计算/训练用例通过；负向检查正确拒绝了漂移，但仅因测试正则写的是 `replicated` 而实际为 `replicas` 而失败 |
| 修正后的负向测试 | 每 rank 1 passed；无完整 manifest |
| 整合 CPU TP + checkpoint + ZeRO | 53 passed，无 skip |
| 最终 CPU owner，含 fresh-process 懒加载 import 回归 | 54 passed，无 skip，4.59 s |
| 整合 CUDA 完整 | 每 rank 8 passed，0 skipped，80.42 s；launcher exit 0 |
| 懒加载导出后的最终 CUDA | 每 rank 8 passed，0 skipped，4.26 s；launcher exit 0 |
| 最终 core CPU | 322 passed，42 skipped，1 个既有 xfail；两种进程模式 exit 0 |
| 初始完整 structure | 1457 passed，6 skipped，4 failed：两个真实 import 循环回归，加上未登记的文档与生成的 MANIFEST |
| Import/子进程回归复检 | Lint：三项契约全部通过；定向 structure：24 passed，0 skipped |
| 布局 | 通过。中央文档 toctree 与生成的 MANIFEST 仍是整合方负责的门禁 |

所有成功的 CUDA 计算都在 `forbid_backend_fallbacks()` 下运行，并
断言全进程 fallback 计数精确为零。四个
训练用例还监视实际的 native optimizer legacy collective
owner：**零次梯度 reduce、零次参数 broadcast**，
包括 `param_sync_iter=1` 的情况。非零的 off-rank 输入/参数梯度
贡献与未分片参考比对共同要求真实的 TP 通信。不声称
任何物理 NCCL 流量或时延。

完整 structure 运行抓到一个真实的 eager 再导出回归：import
`nn.parallel` 会把 `distributed` 带进核心初始化循环，
使循环模块从 165 增加到 168。公共 TP 导出现在改用模块
`__getattr__`，保留规范 owner 而不在核心 import 时加载它。
一个 fresh-process 行为测试守住该初始化顺序。其启动
使用仓库 helper 固定本 checkout 并套用标准
子进程超时。import 契约重新报告 165 个循环模块；allowlist 未变。
上文最终的 CPU、CUDA 与 core 重跑都包含该修复。原先四个
structure 失败保留；那次运行中两项中央文档/manifest
登记缺失；下文整合检查覆盖了它们的完成。六个
structure skip 包括未安装源码解释器中的两项包解析检查、
两项声明的 skip 与两项加速器 skip。

显式 checkpoint-replica 探针首先证明：把整个
model/optimizer 当作 rank 本地会错误接受 bias 漂移。最终
实现只标记真正分片的叶子，并保留副本
完整性检查。第二个真实双 rank 探针还发现：当整个 layout 都是
rank 本地时，会接受不兼容的全局 column 形状；修复后只有其 rank 字段为本地，
并校验全部公共 layout metadata。那次被错误接受的 checkpoint 与日志保留在
Git 之外。第一次 CPU 调用使用了不存在的测试路径，收集到零个
测试；修正后的调用才是 29/53 通过的证据。
第一个 fixture 尝试了 SGD 不支持的构造参数
`param_sync_iter`；现在改为构造后设置既有优化器属性。
早期一次 collective 预热漏掉显式 NCCL 后端 loader；一次
post-sync singleton 预热漏掉其必需的 rendezvous 环境。
两者都在测试前失败、被保留并已修正；随后的串行
预热成功加载 NCCL（`ncclGetVersion=21803`）。这些失败尝试都不计入验收。

## 复现与原始证据

原始文件、日志、参考身份、环境脚本与隔离 JIT
缓存**未版本化**，位于
`$JITTOR_LAB_ROOT/_state/native-tensor-parallel/`。工作树通过
`PYTHONPATH=<worktree>/python` 固定；原生模式使用
`JITTOR_TORCH_SHIM=0`。使用 Python 3.11.16、NumPy 1.26.4、pytest 7.4.4、CUDA nvcc
12.2.140、GCC 11.5 与 NCCL 2.18.3。未修改
共享环境。每个 CUDA rank 有自己的 `HOME`、`TMPDIR`、`JITTOR_HOME` 与
`cache_name=native-tp`；launcher 使用已预热的 rank-zero 缓存。CPU、
CUDA-rank-zero 与 CUDA-rank-one 的首次编译串行执行。CPU 测试
与 core-runner 缓存也分开。后者是从该
owner 早先不活跃的测试缓存复制到新 state 根目录；不共享锁或
可变缓存 inode，runner 重建了变更过的源码。

定向 CPU 命令，在规范与 legacy 两个 flag 中都关闭 CUDA：

```bash
JITTOR_TORCH_SHIM=0 JT_USE_CUDA=0 use_cuda=0 PYTHONPATH=python \
  <native-python> -m pytest -q tests/nn/test_native_tensor_parallel.py \
  tests/serialization/test_sharded_checkpoint.py tests/optim/test_native_zero.py
```

实际 CUDA 命令在逐 rank 串行预热之后
使用原生 launcher。未版本化的 `rank_entry.py` 在 import Jittor 前设置
各 rank 专属的 cache/home/temp 路径，然后调用 pytest，不替换
原生 runtime。`JITTOR_TP_TEST_DIR` 指定一个新的共享 checkpoint 根目录。

```bash
<native-python> python/jittor/distributed/launch.py --backend nccl -n 2 \
  --device-ids=<first>,<second> --timeout 180 --log-dir <logs> -- \
  <native-python> <state>/rank_entry.py -x \
  tests/backends/comm/nccl/test_native_tensor_parallel_nccl.py
```

state 目录下的主要日志索引：

- `logs/missing.log`, `logs/cpu-first.log`, `logs/cpu-second.log`,
  `logs/cpu-checkpoint.log`, `logs/cpu-leaf-fix-corrected.log`;
- `logs/real-torch-identity-overlay.log`, `logs/real-torch-reference.json`;
- `logs/warm-rank0.log`, `logs/warm-collectives-rank0-loaded.log`,
  `logs/warm-rank1.log`（各自独立的 rank 缓存）；
- `logs/two-rank-minimal/rank{0,1}.log`,
  `logs/replica-drift-before/rank{0,1}.log`,
  `logs/two-rank-first-full/rank{0,1}.log`,
  `logs/two-rank-negative-corrected/rank{0,1}.log`,
  `logs/layout-drift-before/rank{0,1}.log`;
- `logs/integrated-cpu-owners.log`, `logs/integrated-cpu-final.log`,
  `logs/two-rank-integrated-final-launch.log`,
  `logs/two-rank-integrated-final/rank{0,1}.log`, `final-source.json`,
  `logs/nccl-library-identity-corrected.json`, `evidence-index.json`;
- `logs/core-cpu.log`, `logs/structure.log`, `logs/layout-final.log`;
- `logs/lazy-cpu-owners.log`, `logs/lazy-cpu-owners-helper.log`,
  `logs/lazy-core-cpu.log`,
  `logs/import-layering-fixed.log`, `logs/lazy-structure-import.log`,
  `logs/lazy-structure-import-child.log`,
  `logs/two-rank-lazy-final-launch.log`,
  `logs/two-rank-lazy-final/rank{0,1}.log`, `lazy-final-source.json`,
  `lazy-evidence-index.json`, `accepted-source.json`,
  `logs/layout-lazy-final.log`.

## 主分支整合验证

整合后的树使用远端 `d62e0deba6b6dc74e4e4d9aaa12bb6baefec2d95`，
含 CUDA RNG 与 tensor factory 修复。全部 1113 个原生 runtime 文件
与七个 TP 源码/测试文件与独立验证过的
`642c60b1bc2007315b4c717722c962223d0dafb2` 树逐字节一致。分别在
串行预热缓存后，两个真实 CUDA/NCCL rank 再次以零 skip 通过全部八个用例。
每个 rank 的四个训练行保持 28/52 个 owned 参数元素、
零 fallback、零 legacy reduction/broadcast 与精确续训。
根验证重新哈希了两个 rank 日志与源码快照。

在整合后的主树上，CPU core 通过 322 个测试、42 个 skip 与
1 个既有 xfail；TP/checkpoint/ZeRO CPU owner 通过 54 个测试、无 skip。
在文档索引与生成的 manifest 更新后，完整的 Torch 模式 structure
套件通过 1461 个测试、六个 skip。布局与 diff
检查通过。此前的 structure 失败不计为通过。

未版本化的 CPU 整合日志为原 state 根目录下的
`logs/main-core.log`、`logs/main-cpu-owners.log` 与 `logs/main-structure.log`。
新基线 CUDA 证据位于
`$JITTOR_LAB_ROOT/_state/native-tensor-parallel-bf16/`：`frozen-fp32-source.json`、
`frozen-fp32-acceptance.json`、`main-source-audit.json` 与
`logs/frozen-fp32-two-rank/rank{0,1}.log`。尽管该 state 目录名如此，
本次验收仅限 FP32；BF16/AdamW 仍是独立工作。

## 限制

这完成的是真实原生 **column/row linear TP 子集**，不是 R17 整体的
FSDP/TP/ZeRO/多节点目标。它验证的是 WORLD-only 等分 feature partition、
CPU singleton 与单机双 rank CUDA、FP32、复制的输入/loss，
以及固定 world 下 momentum SGD 的 checkpoint 续训。通用 TP
算子、attention/vocabulary/sequence parallelism、非均匀或空
feature partition、混合精度、更大 world、DP/TP/FSDP/ZeRO 混合、
checkpoint resharding 与多主机 rendezvous/训练都
不在本次验收范围。Adam 族数学仍在其既有 owner 中，不新增
TP 专属数值验收结论。不声称吞吐或 allocator 峰值
改善。R17 多主机验收仍需不同的真实主机，
这两个本地 rank 无法满足。
