---
name: jittor-distributed-verification
description: 在单机上验证 Jittor 分布式改动（MPI / NCCL 集合通信、rank 与 world_size、rendezvous）。用于改动 extern/mpi、extern/cuda/nccl、extern/acl/hccl、compile_extern 的分布式分支或 distributed/launch.py 之后，需要真正跑起多进程对拍而不是只读代码的场合。也说明哪些后端在本机根本无法验证。
---

# 单机验证 Jittor 分布式改动

**先读 [`../jittor-worktree-verification/SKILL.md`](../jittor-worktree-verification/SKILL.md)**。
分布式验证全是多进程，每个 rank 各自 `import jittor`，PYTHONPATH 错了就是每个 rank 都在
测主树。

## 第 0 步：先确认能验证什么

按顺序问三件事，别跳。

**1. 有没有 mpirun？** 系统目录里**没有**——MPI 是随 conda 环境装的，所以
`which mpirun` 在默认 PATH 下会空手而归，很容易误判成"这台机器没有 MPI"。

```bash
ls "$(dirname "$(which python)")"/mpirun "$(dirname "$(which python)")"/mpicc
"$(dirname "$(which python)")"/mpirun --version
python -c "from mpi4py import MPI; print(MPI.Get_library_version()[:40])"
```

**2. Jittor 编译时带上 MPI 了吗？** `setup_mpi()` 只在 PATH 里找得到 `mpicc` 时才编 MPI 算子。
所以**跑任何 MPI 验证都必须先把环境的 bin 放进 PATH**，否则 `has_mpi` 是 False，
测试全部 skip，而 skip 会以绿色退出——看起来通过了，其实一条没跑。

```bash
PATH=<env>/bin:$PATH PYTHONPATH=<worktree>/python JITTOR_HOME=... TMPDIR=... \
python -c "import jittor as jt; print('has_mpi', jt.compile_extern.has_mpi)"
```

**判据**：`has_mpi True`。若为 False，本次「测试全绿」不算数。

**3. 有没有对应硬件？**

| 后端 | 本机 | 能验证到什么程度 |
| --- | --- | --- |
| MPI（CPU 集合通信） | 有 | **完全可验证**：多进程真跑，和 numpy 期望值对拍 |
| NCCL（CUDA） | 有 GPU | **可编译 + 可单 rank 跑**；多 rank 数值对拍需要多张卡 |
| HCCL / ACL（昇腾） | **无** | **编都编不了**。改动只能标「未在硬件上验证」，且必须在提交说明里写明 |

没有硬件时不要假装验证过。结论写「未验证」比写一个跑不到的测试诚实。

## MPI：多进程真跑（最有价值的一档）

仓库里已有的入口模式：测试文件里放两个类，一个 `@unittest.skipIf(not jt.in_mpi, ...)`
装真正的断言，另一个调 `run_mpi_test(N, "<name>")` 负责用 mpirun 把自己再拉起来。
新测试要在 `tests/_helpers/distributed.py` 的 `_MIGRATED_TEST_PATHS` 里注册，否则
`run_mpi_test` 抛 `unknown migrated MPI test`。

```bash
cd <worktree>
JITTOR_HOME=... TMPDIR=... PATH=<env>/bin:$PATH nvcc_path="" JITTOR_TEST_DEVICES=cpu \
  python -m pytest tests/distributed/test_mpi_dtypes.py -q
```

要看**每个 rank 各自**的结果（外层 pytest 会把内层输出吞掉），直接跑内层：

```bash
JITTOR_HOME=... TMPDIR=... PATH=<env>/bin:$PATH nvcc_path="" \
  mpirun --allow-run-as-root -np 2 python -m pytest -q <worktree>/tests/distributed/<file>.py
```

内层不需要额外设 `PYTHONPATH`：子进程还是 pytest，且 `cwd`/参数落在仓库内，
rootdir 会重新解析。**换成裸 `python script.py` 就必须自己传 `PYTHONPATH`。**

### 写分布式数值测试的口径

- **期望值从 rank 推出来，不要从"另一次运行"推**。例如每个 rank 造
  `arange(1,33) * (rank+1)`，则 all_reduce 的期望是 `arange(1,33) * n(n+1)/2`。
  这样一个错的 datatype 会给出结构性错误的数，而不是"看起来差不多"的数。
- **向量要够长**（≥32 个元素）。类型宽度算错（比如把 8 字节当 16 字节）在短向量上
  可能刚好落在同一块分配里而不崩。
- **dtype 必须显式写**：`jt.array(v, dtype="int64")`。
  `jt.array(np.arange(8, dtype="int64"))` 会静默变成 int32，于是 int64 的 bug 测不到。
- **每个 dtype 都要测**。dtype 表是逐条手写的，错一条只影响一条。
- **断言要 `assert_array_equal` 而不是 `allclose`**（整数类型上 allclose 会放过一些错）。

### 已知的真实例子（6.B01）

`mpi_all_reduce` / `mpi_reduce` / `mpi_broadcast` 三个算子把 int64 映射成
`MPI_DOUBLE_INT`。那是 MAXLOC 用的 (double,int) 二元组、含填充 16 字节，不是整数类型。
按 `count = 元素数` 传给 MPI，读越界且结果错。

复现（修前）：

```bash
mpirun -np 2 python -c '...'   # 两个 rank 的 int64 all_reduce 都返回全 0
```

修前跑测试的两条典型症状，两条都要能看到才算复现到位：

- `mpi_all_reduce` / `mpi_broadcast`：**不报错**，返回全 0（静默算错）
- `mpi_reduce`：`MPI_ERR_OP: invalid reduce operation`——这条恰好证明了那个类型
  是个 MAXLOC 二元组（`MPI_SUM` 对它无定义），而不只是"宽度不对"

## NCCL：单卡也能做编译验证

NCCL 算子是 JIT 编译的，**不真正执行一次就不会编译**，所以「改完读一遍代码觉得没问题」
完全不能替代验证——漏一个枚举名要到运行时才炸。

不需要多卡：用 MPI-free 的 env/file rendezvous 起 **world_size=1**，通信器照样建得起来，
五个算子全部会 JIT 编译并真正调用 NCCL。world_size=1 时 all_reduce/broadcast/reduce 是
恒等，够验证「编得过 + NCCL 接受这个 datatype」；数值上的 N-rank 对拍交给 MPI 那一档。

```bash
CUDA_VISIBLE_DEVICES=<一张卡> nvcc_path=/usr/local/cuda/bin/nvcc \
PATH=/usr/local/cuda/bin:$PATH PYTHONPATH=<worktree>/python \
JITTOR_HOME=... TMPDIR=... \
JT_NCCL_WORLD_SIZE=1 JT_NCCL_RANK=0 JT_NCCL_LOCAL_RANK=0 \
JT_NCCL_ROOTINFO_FILE=$TMPDIR/nccl_root.bin use_nccl=1 \
python <你的脚本>.py
```

脚本里**必须**设 `jt.flags.use_cuda = 1`，否则算子走 CPU 分支，直接报
`Op nccl_all_reduce doesn't have cpu version`，一行 NCCL 代码都没编。

`bfloat16` 不能用 numpy 造（`np.dtype("bfloat16")` 不存在）：
用 `jt.array(x.astype("float32")).cast("bfloat16")`，读回来前先 `.float32()`。

## 别的坑

- **第一次多进程跑很慢**：`jittor.lock` 是 flock 互斥的，N 个 rank 的首次编译是**串行**的。
  按单进程耗时设超时一定会误判成挂死。用哨兵文件等，别用 `pgrep`。
- **别 `kill -9` 正在编译的 rank**：留下损坏的 JIT 缓存，下次在毫不相干的算子上大面积报错。
- **`JT_NCCL_WORLD_SIZE` / `JT_HCCL_WORLD_SIZE` 一旦设了，`use_mpi` 会被强制关掉**
  （`compile_extern.py` 里显式 `os.environ["use_mpi"]="0"`），MPI 算子根本不会编译。
  想同时验证 MPI 和 NCCL，必须分两次跑，不能塞进同一个进程。
- **`mpirun` 会重置环境**：验证用的所有变量都要通过 `mpirun` 之前的 env 传进去，
  确认每个 rank 日志里的 `cache_path:` 指向自己的 `JITTOR_HOME`。
