# 整改分区的 CUDA 可用性实机核实

- Status: Accepted；推翻看板与交接文档里 83 处「本机无 CUDA」记述
- Date: 2026-09-04
- Baseline: `cb343f79`（`2.0-refactor`）
- Owner: Jittor 2.0 整改协调者
- Review when: 开发机换卡、换驱动、换 CUDA 工具链，或 `install_cuda` 的
  pip wheel 选择逻辑变化

## 结论

整改用的这台开发机**有可用的 CUDA**，任意整改分区都能在真实 GPU 上编译并算对。
看板与交接文档里累计 **83 处**「本机无 CUDA 未运行负向」「无 CUDA 实机」的记述
**不成立**：`refactor-board.md` 37 处、`refactor-handoff.md` 45 处「本机无 CUDA」，
外加 `refactor-handoff.md` 1 处「无 CUDA 实机」。

这不是文书问题。受影响最大的是任务 `2.19`：它已迁移 114 处用户错误边界，其中约
60 处属于 CUDA 后端算子（curand、cuDNN RNN/conv/conv3d、cuTT、cuBLAS
matmul/batched/acc、cuSPARSE CSR/COO、CUB cumsum/argsort/arg_reduce、cuFFT、
NCCL），证据一律只有「结构计数 + nvcc TU 语法」两种**静态**验证。而 `2.19` 改的是
`ASSERT` → `USER_CHECK`/`USER_CHECKop`，后者定义在
`python/jittor/src/utils/log.h:210-216`，抛的是 `jittor::UserError`。
**「抛出的是可被 Python 捕获的异常而不是 abort 掉进程」只能在运行时证明**，结构
计数与 TU 语法检查都证明不了。对应的运行时负向用例就在 `tests/backends/cuda/`
下（33 个文件），从未因缺硬件而不可用。

整改计划第 0 节完成定义第 2 条要求「三套门禁全绿：原生（CPU）、CPU torch 模式、
CUDA」，并注明「改到哪一层跑哪一层是不够的——因为并行路径从不交叉验证正是审计的
核心发现」。以「本机无 CUDA」为由跳过 CUDA 一档，正是这条要求要防的情况。

## 环境

| 项 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090 × 8，各 24564 MiB |
| 驱动 / NVRM | 595.84 |
| nvcc | 12.2.140（`/usr/local/cuda/bin/nvcc`） |
| jittor 探测到的 cuda archs | `[89]`（sm_89） |
| nvcc 支持的 archs | 50…87、89、90 |
| CUDA 缓存键 | `cu12.2.140_pipcu122_b9c03cb2d2fb_sm_89` |
| 主机内存 | 1007.70 GB，编译用 16 进程 |

`$JITTOR_LAB_ROOT/refactor/_home/<分区>/.cache/jittor/probe.json` 里本来就记着
`cuda_archs = ["89"]` 与 nvcc 版本——即这些分区**在自己的缓存里已经探测到过
CUDA**，「无 CUDA」的结论不是探测结果，是执行者的判断。

## 结果

在 `coord` 分区（当时是 `2.0-refactor` 的干净 checkout）用一张空闲卡执行：

| 检查 | 结果 |
| --- | ---: |
| `jittor.__file__` 指向本分区 worktree | 是 |
| `jt.has_cuda` | `1` |
| `jt.compiler.nvcc_path` | 非空 |
| `jt.flags.use_cuda`（`flag_scope` 内） | `1` |
| float32 matmul `(256,512)@(512,128)` 对 NumPy 最大误差 | `4.58e-05` |
| 上式 `sum(c*c)` 对 `a` 的梯度最大误差 | `1.46e-03` |
| `argmax(dim=1)` 与 NumPy 逐元素一致 | 是 |
| 全归约 `sum` 对 NumPy 误差 | `3.05e-05` |
| CUDA 冷编译总耗时 | 约 53 s（`jittor_core` 177 个 TU 约 23 s） |

冷编译 53 秒这一项值得单独记：AGENT-BRIEF 写的「首次 `import jittor` 约 10 分钟」
是**核心从零冷编**的量级，而在已有 CPU 侧缓存的分区里追加 CUDA 一档只需约 1 分钟。
「跑 CUDA 太贵」不足以成为跳过它的理由。

## 误判的成因

AGENT-BRIEF 第 1 节为 CPU-only 任务给了加速写法
`JITTOR_TEST_DEVICES=cpu nvcc_path=""`。带上它之后 `jt.has_cuda` 必然是 `0`。
执行者若在这个环境里读 `jt.has_cuda`，就会得到「本机无 CUDA」，且此后每一波都
照抄前一波的措辞——`2.19` 的 114 条记录里这句话重复了 60 余次，从未被复核。

判据：**`jt.has_cuda == 0` 只说明当前进程没启用 CUDA，不说明机器没有 CUDA。**
要判断机器有没有，看 `nvidia-smi` 与 `probe.json` 的 `cuda_archs`，或在**不设**
`nvcc_path=""` 的进程里读。

## 命令口径

```bash
cd "$JITTOR_LAB_ROOT/refactor/<分区>"
PYTHONPATH="$JITTOR_LAB_ROOT/refactor/<分区>/python" \
JITTOR_HOME="$JITTOR_LAB_ROOT/refactor/_home/<分区>" \
TMPDIR="$JITTOR_LAB_ROOT/refactor/_tmp/<分区>" \
CUDA_VISIBLE_DEVICES=<派给你的卡> \
nvcc_path=/usr/local/cuda/bin/nvcc \
PATH=/usr/local/cuda/bin:$PATH \
taskset -c <派给你的核> python <验证脚本>
```

验证脚本先打印 `os.path.dirname(jittor.__file__)` 自检（必须是本分区 worktree，
否则测的是 editable 安装指向的主树），再在 `jt.flag_scope(use_cuda=1)` 内做
matmul、`jt.grad`、`argmax` 与全归约，各自与 NumPy 对拍。

**不要**在验证 CUDA 的进程里设 `JITTOR_TEST_DEVICES=cpu` 或 `nvcc_path=""`。

## 边界

本报告只证明**这台开发机的 CUDA 可用、且分区内能算对**，不代表：

- `2.19` 那 114 处迁移是正确的——它们的运行时负向验收另行执行，结论以那次为准；
- CUDA 全量门禁通过——设备对拍全量外推约 3.5 小时（见任务 `0.22`），本次未跑；
- 其它硬件可用。**本机确实缺** Ascend/CANN/NPU、ROCm、Corex，以及多机所需的第二台
  机器。这几类后端按已批准的降级口径只做代码组织、公共接口与上机文档，看板须保留
  待实机状态。看板里「本机无 CANN」（113 处）、「本机无 NPU」「本机无 Corex」
  「本机无 NCCL 设备」等记述**是成立的**，不在本报告推翻范围内。
