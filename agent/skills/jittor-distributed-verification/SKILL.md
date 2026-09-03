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

## 单卡就能复现「一个 rank 起不来，其余全挂」

多卡失败模式里最贵的一类不需要多卡去复现：**等一个永远不会到的 peer**。
非 0 rank 在 rendezvous 上等，根本走不到第二张卡，所以一张卡就够。

```bash
CUDA_VISIBLE_DEVICES=<一张卡> nvcc_path=/usr/local/cuda/bin/nvcc \
PATH=/usr/local/cuda/bin:$PATH PYTHONPATH=<worktree>/python \
JITTOR_HOME=... TMPDIR=... use_nccl=1 \
JT_NCCL_WORLD_SIZE=2 JT_NCCL_RANK=1 JT_NCCL_LOCAL_RANK=0 \
JT_NCCL_ROOTINFO_FILE=$TMPDIR/never_written.bin \
JT_RENDEZVOUS_TIMEOUT_S=5 \
python -c "import jittor"
```

**判据**：约 5 秒内退出，且错误里同时有「rendezvous」「rank 1」和那个路径。
`JT_RENDEZVOUS_TIMEOUT_S` 生效本身就是一半的验收——8.09 之前的轮询是写死的
6000×20ms，任何环境变量都改不动它。

把 `JT_NCCL_ROOTINFO_FILE` 整个不设，是同一个缺陷的另一张脸（不等待，直接拿
未初始化的 id 去建通信器）。两条都要测。

去掉 `JT_RENDEZVOUS_TIMEOUT_S`、把 rank 改回 0、world_size 改回 1，就是
happy path，必须仍然通过——只测失败分支的话，「无条件抛」也能全绿。

### 写这类子进程测试之前，先知道这两个坑

两个都会让**测试进程自己消失**，而不是给你一条失败。

**1. 子进程被信号打死会连带打死父进程。** jittor 在 `utils/log.cc` 里装了 SIGCHLD
处理器：任何 `si_code != CLD_EXITED` 的子进程都被当成 OOM，父进程直接
`_Exit(1)`。`_Exit` **不刷 stdio**，所以 pytest 会**一个字都不输出**、退出码 1。
它看起来不像崩溃，看起来像什么都没发生。

**2. `import jittor` 失败会在退出时 abort（CUDA 构建）。** 全局 `EventQueue`
起了一个 worker 线程，注销它的 `core.cleanup()` 只由**跑完**的 import 挂上
atexit。没跑完时 `~std::thread` 落在 joinable 的线程上：
`terminate called without an active exception` + SIGABRT，接着触发第 1 条。
这是 jittor 自己的退出期缺陷，和被测的东西无关。

于是这类测试要两层包装（`tests/distributed/test_nccl_rendezvous_timeout.py`
是现成的样板）：

- **最里层**（真正 import jittor 的那个）：`try: import jittor / except: 打印
  traceback，flush，`os._exit(9)`。绕开解释器退出期，就没有第 2 条。
  用一个**独特的退出码**而不是「非 0」——它证明失败是以**可捕获的 Python 异常**
  回来的，而不是从静态构造器里 unwind 进动态链接器变成 terminate。
- **中间层**：一个**不 import jittor** 的普通 python，用 subprocess 跑最里层，
  把被信号打死的孙进程翻译成 shell 风格的 `128+signum` 退出码。它没有 SIGCHLD
  处理器，所以第 1 条打不到 pytest。少了这层，拿这个测试去跑**修复前**的代码
  不会得到一条红的断言，只会得到「pytest 凭空消失」。

## 两卡验证「一个 rank 死了，其余怎么办」

这条要真两张卡，`tests/distributed/test_nccl_watchdog.py` 是现成的样板。步骤与判据：

1. **先各自预热**。每个 rank 用 `JT_NCCL_WORLD_SIZE=1` 单独跑一遍（`cache_name=nccl<r>`
   一 rank 一个缓存，和 `jittor.distributed.launch` 一致）。不预热的话，冷编译会和
   rendezvous 的超时赛跑。
2. **`NCCL_P2P_DISABLE=1`**。本机 `nvidia-smi topo -p2p r` 全是 CNS（任何一对 GPU 都没有
   peer access），而 NCCL 把被拒的 peer access 当致命错误，报的是
   `unhandled cuda error`。`_skip_nccl_p2p_without_peer_access()` 在能看见整张设备表时会
   替你设上；每个 rank 只看得见一张卡，它判断不出来，所以手写实验必须自己设。
3. **按 pid 杀**，绝不用 `pkill -f`——模式会匹配到你自己的 shell 和脚本文本。
4. **等的是哨兵**（双方都打印了 `STEP 5`），不是 sleep。

**判据**：幸存 rank 在预算内退出，**且消息里指名是哪个 rank 没了**。只说「通信超时」等于
没说——N 个 rank 全都这么说，运维还是不知道该看谁的日志。

### 已知：`ncclCommGetAsyncError` 抓不到同机的这一类

实测（NCCL 2.18.3，两张 4090，`NCCL_P2P_DISABLE=1`）：kill 掉一个 rank 之后，幸存 rank 的
通信器**整整两分钟停在 `ncclSuccess`**，kernel 一直在转。同机 + 无 peer access 走的是共享
内存传输，**没有 socket 可断**，异步错误就不会被置位。所以只靠 `ncclCommGetAsyncError`
的 watchdog 在单机多卡上是**恒绿的死代码**——写完必须真杀一个 rank 验证，不能读代码通过。

补的办法是心跳文件（`<rootinfo>.hb<rank>`，见 `nccl_wrapper.cc`）：不挑传输方式，而且能
说出是哪个 rank。判定陈旧要用**本机 steady clock 上「这个文件多久没变过」**，不要拿文件
mtime 去和本机时钟比——共享文件系统差几秒就会让所有 peer 看起来都死了。

### 这类测试必须多一层 conductor

上面「子进程被信号打死会连带打死父进程」那条在这里是**必然**触发的：测试本身就要
`SIGKILL` 一个 rank。第一版直接从 pytest 里杀，得到的不是一条红断言，是
**pytest 凭空消失、零输出、退出码 1**。

所以：pytest → **一个不 import jittor 的 conductor 进程** → 两个 rank。conductor 负责启动、
等哨兵、按 pid 杀、按预算等幸存者，最后打印一行 JSON 给 pytest 断言。它没有 jittor 的
SIGCHLD 处理器，所以孙进程被信号打死打不到 pytest。

## ProcessGroup：两卡也能证明不是 WORLD 的别名

子组测试不一定要四张卡。两 rank 足以区分真实 communicator 与两种常见假实现：

1. 所有 rank 先以同一顺序创建一个 rank 顺序反转的全员组，再依次创建每个 rank 的
   singleton 组。`new_group` 是 collective；只让成员创建会让后续 group id 漂移。
2. 在反转组上做一次 sum，断言 local rank 与全局 rank 相反，并断言它的后端句柄不是
   WORLD 的句柄。
3. 每个 rank 在自己的 singleton 组上对 `100 + rank` 做 sum，结果必须仍是
   `100 + rank`。若底层算子忽略 group、仍使用 WORLD communicator，结果会变成所有
   rank 的和；若 Python 直接把 singleton 当恒等，这条数值会绿，但第 2 步的独立句柄
   断言会红。
4. 最后再在 WORLD 上归约 `rank + 1`。它必须得到 `n(n+1)/2`，证明建组和子组通信没有
   覆盖 DDP 使用的默认 communicator。

环境/文件 rendezvous 与 MPI bootstrap 都要各跑一次同一用例：前者覆盖按 group id 派生
的 root-info 文件，后者覆盖以组首 rank 为 root 的 MPI 广播。只跑其中一条不能证明另一条
不会在建组时挂住。

## TCPStore/FileStore：用两个普通进程验，不要用两份字典自证

store 的跨进程契约不需要 GPU 或 mpirun。父进程启动两个普通 Python 子进程，rank 0 写一个
非空 payload、rank 1 读取后写 ack；两边再对同一个计数键各 `add(1)`，等两个到达键后断言
最终值为 2。这样同时抓住「每个进程各有一份字典」、阻塞 `get/wait` 无效和 `add` 非原子。

两个子进程共享同一份 JIT 缓存时，**必须先顺序预热一次 import**。否则 master 进程可能持着
`jittor.lock` 进入 TCPStore 构造并等待 client，client 却在等这把锁完成 import，表象是 store
挂死，实质仍是编译锁死锁。预热输出还要断言 `jittor.__file__` 属于目标 worktree。

`init_process_group(init_method=...)` 不能只测 URL parser 的返回值：初始化之后从
`distributed_c10d._get_default_store()` 取回默认 store，再做同一轮跨进程 payload/ack，才能
证明 `env://` / `tcp://` 真进入了 store 机制而不是解析后丢弃。

## 「等别的 rank」和「拿着编译锁」不能同时发生

`jittor.lock` 是**整个缓存目录一把 flock**，而 `import jittor` 从头到尾都握着它
（`jittor/__init__.py` 的 `with lock.lock_scope():`）。所以：

> **任何会阻塞等待其他 rank 的动作，都必须在释放编译锁之后做。**

否则就是死锁：这个 rank 拿着锁等别人，别人要拿这把锁才能编译、才能走到会合点。
`compile_custom_ops` 里那句 `with lock.unlock_scope():` 包住 dlopen（注释写的是
"unlock scope when initialize"）就是为这个存在的——通信器原本由 dlopen 期的静态构造器建，
所以正好落在锁外。**把它改成显式调用就同时把它挪回了锁内**，2 卡冷缓存 MPI 跑立刻死锁。

现在 `setup_nccl()` / `setup_hccl()` 都用 `lock.unlock_scope()` 包住 init 调用，
`misc/file_rendezvous.h` 里的 `rendezvous_require_unlocked()` 在真去等之前检查一次，
拿着锁就直接报错而不是挂死。

### 认出它（症状是「什么都没有」）

死锁期间**没有任何输出**：一个 rank 100% CPU（OpenMPI 的集合通信是忙等），
其余 0% 睡在 flock 上。三步确认：

```bash
# 1. 谁在忙、谁在睡
ps -o pid,stat,pcpu,wchan:30 -p <每个 rank 的 pid>
#    R + 99% = 忙等在 MPI 里；S + hrtimer_nanosleep + 0% = 在排队等锁

# 2. 锁在谁手里（jittor 会把持有者写进锁文件本身）
find $JITTOR_HOME -name jittor.lock -exec sh -c 'echo "== $1"; head -c 400 "$1"' _ {} \;
#    输出 {"pid":..., "time":..., "cmd":...}

# 3. 那个 pid 是不是正好就是「忙等的那个」。是 -> 就是这个死锁。
```

**判据**：持有锁的 pid 同时是 `R`+99% 且没有编译器子进程
（`ps --ppid <pid>`）、缓存里几分钟没有新文件
（`find $JITTOR_HOME/.cache -newermt "-3 minutes" -type f`）——那它不是在编译，
是在等人。

`JT_LOCK_TIMEOUT`（默认 1800s）最终会让等锁的一方报错，但等半小时才知道太贵；
上面三步一分钟就能定性。

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
- **别在同一个 `JITTOR_HOME` 里同时跑两份活**（哪怕都是你自己的）。那把锁是整个缓存目录
  一把，不是每个 cfg 一把：一个 `mpirun -np 2` 的首次编译会让你另开的那条 pytest 干等十几
  分钟，看起来完全像挂死。分辨方法——
  `ps -o stat,pcpu,wchan:30 -p <pid>`：**`S` + `hrtimer_nanosleep` + 0% CPU 是在排队等锁**，
  `R` + 99% 才是真在编译。要并行就给第二条活另一个 `JITTOR_HOME`（代价是一次全量重编）。
- **`has_mpi` 变了会换 cfg 目录，也就是全量重编一次核心。** 所以「加上 PATH 再跑一遍
  MPI 版」不是在原缓存上多跑几个用例，是从零编一棵树。排时间的时候按十分钟起算。
- **别 `kill -9` 正在编译的 rank**：留下损坏的 JIT 缓存，下次在毫不相干的算子上大面积报错。
- **`JT_NCCL_WORLD_SIZE` / `JT_HCCL_WORLD_SIZE` 一旦设了，`use_mpi` 会被强制关掉**
  （`compile_extern.py` 里显式 `os.environ["use_mpi"]="0"`），MPI 算子根本不会编译。
  想同时验证 MPI 和 NCCL，必须分两次跑，不能塞进同一个进程。
- **`mpirun` 会重置环境**：验证用的所有变量都要通过 `mpirun` 之前的 env 传进去，
  确认每个 rank 日志里的 `cache_path:` 指向自己的 `JITTOR_HOME`。
