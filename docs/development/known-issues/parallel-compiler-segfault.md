# 并行编译器：文件级死锁与算子级段错误

- 状态：**文件级编译进程池的死锁已修复**（2026-09-16）；并行**算子**编译器的段错误 /
  状态损坏在非 Jupyter 负载下仍未解决
- 上次复查：2026-09-16
- 基线：`4b01857b`
- Owner：编译器 / 执行器维护者
- Workaround：
  - 状态损坏（算子级）：`jt.flags.use_parallel_op_compiler = 0`
  - 死锁（文件级，修复前）：`DISABLE_MULTIPROCESSING=1`。
    **`use_parallel_op_compiler=0` 对这个死锁无效**——它根本不经过 `run_cmds`，
    见下面的对照数据
- 退出条件：在开启并行编译的情况下，一个最小化压力测试可重复通过，且无缓存损坏、
  无明显性能回退（死锁一项已由下面的回归测试与对照数据关闭）

这一页是**单个问题的调查记录**，不是已知问题的清单。问题总账在
[`agent/manuals/known-issues.md`](https://github.com/Jittor/jittor/blob/master/agent/manuals/known-issues.md)，
本页对应其中的 `KI-COMPILER-001`；两边的状态、workaround 与退出条件必须一致，
改一边就改另一边。总账里其余条目只有条目本身，没有对应的长文页面。

文件名是历史遗留（早期只知道段错误那一面）。这里记录的是**两个不同的失败**，它们共用
"并行编译"这个名字，但机制、触发条件和修法都不一样：

| | 文件级 | 算子级 |
| --- | --- | --- |
| 代码 | `python/jittor/build/utils/__init__.py` 的 `run_cmds()` | `src/core/parallel_compiler.cc` |
| 并行原语 | `multiprocessing.Pool` | `std::thread` / `std::async` |
| 开关 | `DISABLE_MULTIPROCESSING=1` | `use_parallel_op_compiler` |
| 失败 | **死锁**（已修复） | 段错误 / 堆损坏（未解决） |

## 已修复：文件级编译进程池的死锁

### 症状

裸 `python3 某个脚本.py`，在需要真正编译的时候（冷缓存、改了 `src/`、新的自定义算子）
**稳定卡死**，不是偶发：

- 父进程 `S (sleeping)`，`/proc/<pid>/wchan` 是 `unix_stream_read_generic`；
- 一个子进程 `hrtimer_nanosleep`；
- **`cc1plus` 进程数为 0**——根本没有在编译，所以"编译很慢"是错的判断；
- 同样的编译工作在 `python3 -m pytest` 下**成功**。

### 最小复现

三行脚本，一个冷缓存：

```bash
export JITTOR_HOME=/tmp/ki-compiler-001
export cache_name=ki-compiler-001
rm -rf "$JITTOR_HOME/.cache"
cat > /tmp/bare.py <<'PY'
import jittor as jt
print((jt.ones((3, 3)) * 2).sum().item())
PY
python3 /tmp/bare.py          # 修复前：永远不返回
```

修复前，这个脚本在上限为 1800s、900s、600s、300s、240s、240s、600s 的**七次**运行里
**一次也没有前进过**，`cc1plus` 始终为 0。两个对照能让同一份冷缓存跑通：把入口换成
`python3 -m pytest`（它的 `__main__` 有 `__spec__`），或者给脚本加上
`if __name__ == "__main__":` 保护（被重跑时脚本体不执行）。这个差别正是机制的入口。

### 机制

三件事叠在一起，缺一不可：

1. **`run_cmds()` 是在持有 `jittor.lock` 的时候建进程池的。** 它由
   `import jittor` 经 `build_core()` 调到，而整个 `jittor/__init__.py` 的主体跑在
   `with _lock.lock_scope():` 里面（`__init__.py:103`）。这个进程**握着那把文件锁**。

2. **Python 3.14 把 Linux 上 `multiprocessing` 的默认启动方式从 `fork` 改成了
   `forkserver`。** 本机实测 `multiprocessing.get_start_method()` 返回 `forkserver`。

3. **`forkserver` 会重新执行 `__main__`。** `ForkServer._preload_modules` 恒为
   `['__main__']`；`spawn.get_preparation_data()` 在 `__main__.__spec__` 为
   `None`（也就是 `python3 脚本.py` 这种跑法）时走 `init_main_from_path` 分支，于是
   fork server 启动后第一件事就是
   `spawn.import_main_path()` → `runpy.run_path(脚本, run_name='__mp_main__')`——
   **把用户的脚本原样再跑一遍**。

于是形成一个闭环：被重跑的脚本走到它自己的 `import jittor`，在 `lock.py:_acquire()`
里轮询等待父进程握着的那把锁；父进程在 `forkserver.connect_to_new_process()` 里等
fork server 交回 worker 的 pid，而 fork server 永远到不了 `accept()` 那一步。

```
父进程   持有 jittor.lock ──等待──> fork server 交回 worker pid
                                        │
fork server  重跑脚本 → import jittor ──等待──> jittor.lock
```

两边都不会动，编译器一次也没有被启动。

### 证据

fork server 的命令行里直接写着要重跑哪个文件：

```
python3 -c 'import sys; from multiprocessing.forkserver import main;
  main(14, 15, ['__main__'], ..., **{..., 'main_path': '/tmp/bare.py', ...})'
```

同一份卡死现场，`python3 -m pytest` 的 fork server 命令行里**没有 `main_path`**——
因为 `pytest/__main__.py` 有 `__spec__`，走的是 `init_main_from_name` 分支。

`faulthandler` 打出的两个栈把环闭上（fork server）：

```
forkserver.py:230 main -> spawn.py:307 import_main_path
  -> spawn.py:297 _fixup_main_from_path -> runpy.run_path
  -> bare.py:1 <module> -> jittor/__init__.py:103 <module>
  -> build/utils/lock.py:269 lock_scope.__enter__
  -> build/utils/lock.py:232 Lock.lock -> build/utils/lock.py:176 _acquire
```

以及父进程：

```
bare.py:1 <module> -> jittor/__init__.py:107 -> build/compiler.py:1402 build_core
  -> build/compiler.py:102 compile_backend_sources
  -> build/utils/__init__.py run_cmds -> pool.py __init__ -> _repopulate_pool
  -> popen_forkserver.py:51 _launch -> forkserver.py:106 connect_to_new_process
  -> connection.py:995 answer_challenge -> connection.py:226 recv_bytes
```

`/proc/locks` 说明谁握着锁——只有父进程，fork server 只是打开了同一个文件：

```
29: FLOCK  ADVISORY  WRITE 812441 00:6a:56864199665 0 EOF
```

`hrtimer_nanosleep` 对应 `_acquire()` 里 `time.sleep(0.05 / 0.5)` 的轮询。

**一个红鲱鱼**：现场还有第三个进程，`python3 <脚本>`、`pipe_wait`、父进程是主进程。
它不是死锁的一环，是 `src/utils/tracer.cc` 里那个**预先 fork 的符号化助手**，按设计
就一直阻塞在 `read(request_fd, ...)` 上。归因时不要把它算进去。

### 修复

`run_cmds()` 用 `_main_module_not_reexecuted()` 把整个进程池的生命周期（创建**以及**
使用）包起来：临时给 `__main__` 一个只需要回答 `.name` 的替身 `__spec__`，于是
`get_preparation_data()` 走 `init_main_from_name` 分支，fork server 拿不到
`main_path`，每个 worker 的 `_fixup_main_from_name('__main__')` 也直接返回。

编译池要发给 worker 的东西（`do_compile`、`pool_initializer`）都是
`jittor.build.utils` 的属性，没有任何东西住在 `__main__` 里，所以子进程什么也没丢。

这段代码**本来就存在**，只是被 `if os.name == 'nt'` 圈在 Windows 上，注释写着
"a hack way to by pass windows multiprocess spawn init_main_from_path"。Python 3.14
把 `forkserver` 变成 Linux 默认之后，同一个绕法在所有平台都成了必需品。作用域也从
"只包住 `Pool()` 的构造"扩大到"包住整个 `run_cmds`"，这样 `_maintain_pool` 在任务
执行期间补建的 worker 也会得到同样的准备数据。

回归测试：`tests/build/test_compile_pool_main_reexec.py`。它钉住的是**真正阻断环路的
那条性质**——没有 `init_main_from_path` 到达子进程——而不是死锁本身，后者无法在进程内
复现。

### 那把全局构建锁：它解释什么，不解释什么

`jittor.lock` 是一把进程级的 flock，Python 和 C++ 共用同一个描述符和同一个
`_has_lock` 标志（`Lock.bind_core` → `set_lock_fd`）。它出现在这个问题的两端，
所以值得把它的作用范围说清楚，免得把没发生的事算进机制里。

**它确实是环的一半。** `run_cmds()` 建进程池时这个进程握着锁——这是上面第 1 条。

**但"子进程继承了描述符和已持有标志，于是父子互等"不是这里发生的事。**

- 卡住的那个子进程是 **fork server**，它由 `util.spawnv_passfds()` 以
  `close_fds=True` `posix_spawn` 出来，只传 listener / alive / authkey 三个 fd。
  它是一个**全新的解释器**，既没有继承 `lock_fd`，也没有继承 `_has_lock`。
  实测它持有的 `jittor.lock` 描述符是**它自己在重跑脚本时打开的**，它的栈也停在
  `_acquire()` 里——它在**申请**锁，不是在继承锁。
- `/proc/locks` 里自始至终只有一个 FLOCK 持有者，就是父进程。
- 反过来说，**继承 `_has_lock=1` 从来不会造成死锁**：子进程会以为自己已经持有而
  直接跳过申请。这正是 Python 3.14 之前用 `fork` 启动方式时发生的事，也正是这个
  死锁在 3.14 之前不存在的原因。（它换来的是另一类风险：一个自以为持锁、实际没锁的
  编译子进程。本次改动不触碰这一点，只记录在案。）
- 现场那个 `pipe_wait` 的子进程是 `tracer.cc` 的符号化助手，不是池 worker。

**"裸 python 死锁而 pytest 成功"也不是持锁状态的差别。** 两者都从
`import jittor` 的同一个 `lock_scope()` 里进入 `run_cmds`，持锁状态完全一样。差别
只在 `__main__.__spec__`：fork server 的命令行里有没有 `main_path`。这一条有直接
证据（见上面的"证据"一节），不需要推测。

**锁对并行度的真实影响在进程之间，不在进程之内。** `_has_lock` 是一个普通的全局
`int`，不是 thread_local：`parallel_compile_all_ops` 在主线程上先取了
`lock_guard`（`parallel_compiler.cc:210`），16 个 worker 线程随后在
`OpCompiler::do_compile`（`op_compiler.cc:1361`）里各自构造的 `lock_guard` 就都看到
`_has_lock != 0` 而变成空操作。所以**这把锁不会把同一个进程里的 16 个 worker 串行
化**。它串行化的是**共用同一个缓存的其它进程**——而且是按"整个编译批次"为粒度。
pytest-xdist 的 8 个 worker 共用一个缓存时正好落在这个粒度上，这是"xdist 8 worker
只比串行快 7%"值得优先验证的解释。

顺带记一个**未修的隐患**：`_has_lock` 被主线程写、被 16 个编译线程读，没有任何同步，
是一处真实的数据竞争。目前只因为"主线程一定先取外层 `lock_guard`"而没有暴露；一旦
哪个 worker 在 `_has_lock == 0` 时进入 `OpCompiler::do_compile`，它取到的进程级
flock 会被另一个 worker 的 `lock_guard` 析构函数解掉。这属于本页未解决的那一半
（算子级），不在本次改动范围内。

### 试过但否定的方向

- **`jt.flags.use_parallel_op_compiler = 0`**（总账里原本记的 workaround）：无效。
  这个 flag 只管 `parallel_compiler.cc` 的 `std::thread`，`run_cmds()` 根本不看它。
  对照数据见下。
- **把模块级的 `if os.name=='nt' and _inheriting: DISABLE_MULTIPROCESSING=1` 改成
  跨平台**：不够。被重跑的脚本卡在 `jittor/__init__.py:103` 的锁上，**早于**
  `jittor.build.utils` 被导入，那个守卫来不及生效。而且 `_inheriting` 在
  `spawn._main()` 里对**每一个**编译 worker 都为真，跨平台化会顺带给所有 worker 设上
  `log_silent=1`，把编译诊断藏起来。因此保持 Windows 专用。
- **改用 `fork` 启动方式**：能绕开，但方向是错的。仓库里已有的 MPI 逃生通道正是因为
  `fork()` 在 MPI_Init 之后不安全才存在；CPython 也正因为多线程进程里 `fork()` 不安全
  才改的默认值。往回退等于把一个更难诊断的问题请回来。
- **加超时 / 重试**：不可接受。这是一个闭合的资源环，超时只能把"永远不动"换成"很久
  之后失败"，编译依然一次都没发生。

### 对照数字

同一台机器、同一份脚本、每次都从 `rm -rf $JITTOR_HOME/.cache` 的冷缓存开始：

```
                     修复前                      修复后
trial 1     240s 到期，DONE=0，cc1plus=0      61s，跑完
trial 2     240s 到期，DONE=0，cc1plus=0      61s，跑完
```

240s 是这一轮的上限；同一个复现此前还在 1800s、900s、600s 的上限下各卡过一次，六次
运行里**没有一次编译过任何一个文件**。修复后两次都是 61s，其中约 27s 是
`jittor_core` 那 222 个文件的冷编译。

对照运行（修复后，同一份代码）：

```
python3 guarded.py            5.7s   （脚本带 if __name__ == "__main__" 保护）
python3 bare.py（热缓存）      1.7s
python3 -m pytest tests/core  37.3s
```


哪个开关真正管用（同样冷缓存、修复前的代码）：

```
W1  use_parallel_op_compiler=0    rc=124（600s 到期），DONE=0   —— 无效
W2  DISABLE_MULTIPROCESSING=1     rc=0，336s，DONE=1           —— 有效
```

W1 卡在和默认设置完全相同的位置（日志最后一行都是 `Create cache dir ... /checkpoints`），
现场同样有一个带 `main_path` 的 fork server。这不是巧合：`run_cmds()` 的代码里没有任何
一处读 `use_parallel_op_compiler`，那个 flag 只喂给 `src/core/parallel_compiler.cc`。
W2 反过来直接把 `run_cmds` 换成串行 in-process 编译，环就不存在了——代价是 222 个文件
串行编译，336s 对并行的 61s。


## 未解决：并行算子编译器的段错误

某些大模型与设备一致性负载在**多个融合运算并发编译**时，可能以分配器错误或段错误终止。
该失败对时序敏感，而且可能在**破坏状态的那次编译之后**才暴露出来——所以**最后打印的
算子名不是可靠的归因**。

数据集、ACL 和一致性测试路径已经成功使用串行编译。这是一项**围堵措施，不是"运行时算子
执行有缺陷"的证据**。

### 已解决的 Jupyter 子问题

此前的 Jupyter 复现有一个独立且已被证实的原因。并行算子编译器用的是 `std::thread`，
**并不 fork 编译 worker**。但 Jittor 仍会在 ipykernel 内安装一个进程级的 `SIGCHLD`
处理器，只要有任何子进程被信号杀死就让宿主快速退出。一个只包含 `import jittor` 和一个
无关的、被 `SIGKILL` 的子进程的最小进程，复现出了带 `si_code=CLD_KILLED` 与
`si_status=SIGKILL` 的退出。

现在 Jupyter kernel 保留它已有的 SIGCHLD 处置方式（就像它已经保留 SIGINT 那样），而
Jittor 保留 SIGILL 与 SIGBUS 的故障诊断。确定性的宿主归属回归测试和定向的 CPU/CUDA
冷算子探测都通过。

但**之后一次完整的 nbclient 冒烟仍然在 transformer 注意力负载下、八个编译 worker 时
死了两次**，其中一次还设了 `JT_NO_SIGNAL_HANDLER=1`；同一个冒烟在串行编译下通过。
因此在这个更大的问题未关闭之前，**离线 notebook 门禁保持串行**。

### 当前假设

主要假设是：**并行编译 worker 与进程级的编译/缓存加锁之间缺少一道同步边界**。某个
worker 可能观察到由拥有进程建立的锁状态，从而错误地认为共享的编译器或 relay 状态是
受保护的；随后的并发修改就表现为堆损坏。

对非 Jupyter 负载而言**这仍然只是假设**：relay 组的归属、确切的共享可变对象、以及第一次
非法访问，都还没有用 sanitizer 轨迹演示出来。因此**加一把大锁不是可接受的修法**——它
可能让嵌套编译死锁、抹掉本意的并行性，或者根本保护不了多进程的缓存访问。

### 并行算子编译器值多少

一个专用的探针（`use_cuda=0`，50 个互不相同的 CPU 融合核，独立的 `JITTOR_HOME`，
每次测量前清空 `jit/` 目录，两轮）：

```
          单进程编译 50 个核     两个进程各编译 50 个互不相同的核（同一缓存）
par=16          7s / 7s                       14s / 13s
par=0          43s / 42s                       88s / 86s
```

两个结论，方向相反，都要看：

- **进程内：并行算子编译器值大约 6 倍**（7s 对 42s）。所以"它根本不带来加速、默认
  开着只剩风险"这条假设**不成立**，本次不改 `use_parallel_op_compiler` 的默认值。
- **进程间：完全串行。** 两个进程编译**互不相交**的核，用时等于两者之和
  （≈ 2×），而不是两者的最大值——在一台 384 核、load 约 24 的机器上。无论
  `par=16` 还是 `par=0` 都是 2×，说明串行化来自那把进程级的构建锁本身，与批次粒度
  无关。**这是"pytest-xdist 8 个 worker 只比串行快 7%"最值得先验证的解释**：worker
  之间共用一个缓存，编译阶段被这把锁排成了一队。

换句话说，要提升编译吞吐，该动的是**跨进程的锁粒度**（例如按产物加锁而不是按批次 /
按算子加锁），而不是把 `use_parallel_op_compiler` 默认关掉——后者会白白丢掉进程内的
6 倍。跨进程锁粒度不在本次改动范围内。

### `45:32` 对 `3:39` 是缓存冷热，不是这个 flag

整套 `tests/ops`，**每次都从同一份热缓存快照 `rm -rf` 后恢复**再跑：

```
                          耗时    通过/失败/跳过        并发编译器数(峰值/均值)
修复后 par=16 (1)         183s    740 / 4 / 91                18 / 2.9
修复后 par=0  (1)         219s    740 / 4 / 91                19 / 3.6
修复后 par=16 (2)         216s    740 / 4 / 91                17 / 3.4
修复后 par=0  (2)         204s    740 / 4 / 91                 4 / 2.1
修复前 par=16             201s    740 / 4 / 91                 3 / 1.3
```

五次运行的 `FAILED` 集合**逐条相同**（4 个既有的 `test_matmul` 失败），通过与跳过的
条数也相同——本次改动没有回归。耗时 183~219s 之间的差异全在噪声里：**在同一份缓存
状态下，`use_parallel_op_compiler` 对整套 `tests/ops` 没有可测量的影响**。

对照之下，同一套测试从**冷 JIT 缓存**（核心已编译、`jit/` 为空）跑，74% 的进度就花了
47 分钟——因为每个 CUDA 核都要重新过一遍 nvcc。这就是 `45:32` 对 `3:39` 的真实来源：
第一次跑把缓存喂热了，第二次跑的 3 分多钟和并行/串行无关。**先并行、后串行地各跑一次，
量到的是缓存预热。**


## 复现协议

使用专用的状态目录，并保留确切的提交、编译器、设备和环境：

```bash
export JITTOR_HOME="$JITTOR_LAB_ROOT/_state/parallel-compiler/repro/jittor-home"
export cache_name=parallel-compiler-repro
python -m pytest -v tests/backends/parity/test_device_parity.py
```

用默认的并行编译设置跑一次，再用下面的设置跑一次：

```python
import jittor as jt

jt.flags.use_parallel_op_compiler = 0
```

**两个变体不要并发运行，也不要共用同一个缓存。** 一次有用的复现要记录：缓存是否是冷的、
worker 数量、最后一次完成的编译、信号/回溯，以及重复的串行运行是否干净。

**并且：两个变体必须从同一个缓存快照出发。** 先跑并行、再跑串行，然后把第二次的耗时
当成串行的功劳，量到的是缓存预热，不是编译策略——`45:32` 对 `3:39` 那一组数字就是这么
来的。做法是先把缓存目录 `cp -a` 出一份快照，每个变体开跑前 `rm -rf` 再恢复。

## 调查计划

1. 在保持冷缓存失败的前提下把负载最小化；
2. 用稳定的标识符为编译任务创建、relay 组归属、缓存锁获取和 worker 完成打点；
3. 在 AddressSanitizer 或 ThreadSanitizer 下运行最小化的原生编译路径；
4. **在改动同步之前**先定位第一次非法访问或竞争；
5. 施加范围最窄的归属或加锁修复，并补一个确定性的压力回归测试。

## 验收门槛

一个修复必须同时证明：

- 冷缓存与热缓存的重复压力运行不再崩溃；
- 带超时的测试显示无死锁；
- 两个进程分别使用独立缓存与共享缓存配置时不会损坏产物；
- 并行编译时间相对记录的基线**没有明显回退**；
- 恢复并行编译后，编译器、设备一致性和代表性模型测试全部通过；
- **串行 workaround 与问题总账条目在同一次改动中一并移除**。

在这些证据出现之前，优先考虑确定性验证的调用方可以显式关闭并行编译器，**并应在结果中
报告这一选择**。
