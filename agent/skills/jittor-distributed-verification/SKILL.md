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

## 分布式里最难测的一类：静默不执行

分布式的坏结果很少是异常，多数是**该做的事没做**：集合通信被跳过、参数没广播、
梯度没归约。这类缺陷跑起来一切正常、loss 曲线好看，只有模型是错的。写测试时
**必须让"跳过"变成可观测的**，否则测试和缺陷一起绿。

判据：**问「如果这段代码什么都不做，我的断言会失败吗？」** 答不上来就重写断言。

三种把"跳过"变可观测的做法：

1. **让本该被调用的东西不存在**。例如验证 `mpi_param_broadcast` 有没有真的走分布式
   分支：在 `use_mpi=0` 的子进程里跑（`Var` 上没有 `mpi_broadcast`），
   走进去 -> `AttributeError`，跳过 -> 静悄悄返回。于是打印
   `ATTEMPTED` / `SKIPPED` 两种结果都能看见，而不是「没报错就算过」。
2. **初值不能是零，也不能是期望值**。广播/归约的测试里，非 root 的初值填 `-1`
   这种既不是期望值也不是 0 的数。填 0 的话，「广播没执行」和「广播执行了但结果是 0」
   分不开——6.B01 修前 int64 all_reduce 返回的正是全 0。
3. **期望值要依赖 rank**。所有 rank 用同一份输入时，「没通信」和「通信了」结果一样。

同理，**`skipped` 不是 `passed`**。`has_mpi` 为 False 时整个文件被 skip，pytest 照样
EXIT=0、照样绿。每次跑完都要看清 `N passed` 里的 N 是不是你以为的那些用例，
skip 的数量有没有突然变大。

## 「看起来是局部的，其实是全局的」

这个形状在 Jittor 这套代码里反复出现，而且每次都表现为**静默的错结果**而不是报错。
认出它比修它重要。三种变体：

**1. 副本冒充读通道。** 一个名字看着是"当前值"，其实是某次 import 时抄的快照。
之后有人改了真正的来源，快照不会跟着变，读它的代码就悄悄走错分支。
- 实例：`_runtime/core_api.py` 顶上 `from jittor import *` 抄走一份 `in_mpi`，
  `Module.mpi_param_broadcast()` 读的是它。任何在 import 之后才打开分布式的路径
  （torch 的 NCCL installer 就是）都改不到这份快照，于是参数广播**直接 return**，
  每个 rank 保留自己的随机初始化。
- **判据**：`"x" in vars(mod)` 为真 -> 是副本；为假而 `mod.x` 能取到 -> 是读通道
  （模块级 `__getattr__`）。
- **改法**：让所有读取点走读通道（PEP 562 的模块 `__getattr__`），唯一来源只有一处。
  注意**给模块属性赋值会遮蔽 `__getattr__`**，把副本又造回来——写入必须写唯一来源。

**2. 全局开关被当成局部设置。** `jt.flags.use_cuda = 1` 之类，设了不还原就泄漏给
后面所有用例。表现是"本该在 CPU 跑的用例全在 CUDA 上跑"，而那条路径的梯度恰好是错的。
- **改法**：try/finally 还原，并在测试里**断言还原成功**。

**3. 上下文由调用方式推断。** 不是快照也不是开关，而是"进程的语义取决于你怎么调它"。
- 实例：`tests/conftest.py` 按 `sys.argv` 决定整个进程的 torch shim 模式。选择集合里
  只要有一个路径命中 `TORCH_MODE_PATHS`（含 `tests/compat/torch`），**整个进程**就设
  `JITTOR_TORCH_SHIM=1`，惰性求值、归约默认值、梯度语义全部换一套。于是
  `pytest tests/core tests/nn tests/compat/torch` 会让 core 和 nn 在 shim 语义下跑，
  产生一大批**假失败**。
- **判据**：加一个目录进选择集合，前面那些目录的结果会不会变？会，就是这个形状。
- **改法（测试侧）**：按语义分组跑，不要混：
  ```bash
  JITTOR_TORCH_SHIM=0 pytest tests/core tests/nn tests/optim tests/distributed
  JITTOR_TORCH_SHIM=1 pytest tests/structure tests/compat/torch
  ```
  归责之前先确认失败不是这么来的。

### 怎么写抓得到它的测试

**只写唯一来源，查所有读取点。** 绝不「写几份、查几份」。

一条真实的反例：结构用例本来是
```python
core_api.in_mpi = True; jittor.in_mpi = True; compile_extern.in_mpi = True   # 写三份
with scope: assertFalse(三份)                                                # 查三份
```
它看起来在守护「三个视图保持同步」，实际守护的是「我刚手动同步过的三份现在还一致吗」
——**三份互相独立的快照也照样通过**。也就是说它验不出它声称要验的东西。改成
写 `compile_extern.in_mpi` 一处、断言三个读取点都跟着变之后，旧实现下它才会失败。

## 别的坑

- **第一次多进程跑很慢**：`jittor.lock` 是 flock 互斥的，N 个 rank 的首次编译是**串行**的。
  按单进程耗时设超时一定会误判成挂死。用哨兵文件等，别用 `pgrep`。
- **别 `kill -9` 正在编译的 rank**：留下损坏的 JIT 缓存，下次在毫不相干的算子上大面积报错。
- **`JT_NCCL_WORLD_SIZE` / `JT_HCCL_WORLD_SIZE` 一旦设了，`use_mpi` 会被强制关掉**
  （`compile_extern.py` 里显式 `os.environ["use_mpi"]="0"`），MPI 算子根本不会编译。
  想同时验证 MPI 和 NCCL，必须分两次跑，不能塞进同一个进程。
- **`mpirun` 会重置环境**：验证用的所有变量都要通过 `mpirun` 之前的 env 传进去，
  确认每个 rank 日志里的 `cache_path:` 指向自己的 `JITTOR_HOME`。
