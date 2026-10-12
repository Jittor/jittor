---
name: hang-that-holds-the-gil
description: 定位「进程再也不返回」的挂死——Jittor 的无界阻塞调用（RingBuffer 的 `pop()`/`push()` 等）是在 C++ 里等条件变量且**不释放 GIL**，所以 Python 的 signal handler、定时器、看门狗线程一律不会跑，gdb 又常因 ptrace_scope 附不上。给出能拿到 Python 栈的两种手段、把挂死变成有界失败的测试写法，以及多进程唤醒路径的核查清单。遇到 timeout 124 收尾、pytest 卡住不动、「worker 报错了但父进程不返回」时读这一篇。
---

# 挂死在一个握着 GIL 的调用里

## 症状与第一判据

`timeout 100 python x.py` 以 124 收尾，或 pytest 停在某条用例上不动。
**先分清是"慢"还是"挂"**：`cat /proc/<pid>/wchan`。

- `futex_wait_queue` / `do_futex_wait` → 在等条件变量或锁 → 是挂死。
- `poll_schedule_timeout` / `pipe_read` → 在等 I/O（子进程、管道）。
- 有 CPU 占用（`ps -o pcpu`）→ 只是慢。

## 拿到 Python 栈：`faulthandler`，不是 gdb

两个原因让常规手段失效：

1. **gdb 附不上。** `/proc/sys/kernel/yama/ptrace_scope=1` 只允许祖先进程附加，
   而你新起的 gdb 不是那个挂死进程的祖先；没有 root 就改不了。
2. **Python 的 signal handler 不会跑。** 阻塞发生在 C++ 里且**没有释放 GIL**
   （`src/bindings/pyjt/py_ring_buffer.cc` 里只有有界的 `pop_for` 包了
   `Py_BEGIN_ALLOW_THREADS`，`pop()`/`push()` 仍握着 GIL 无限等），解释器停在字节码之间，
   `signal.signal` 注册的处理器、`threading.Timer`、任何 Python 看门狗线程都不会被调度。

`faulthandler` 两个入口都在 C 层，不需要 GIL，所以两个都能用：

```python
# 事后问它："你卡在哪"——给已经挂住的进程发 SIGUSR1
import faulthandler, signal
faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
```
```bash
kill -USR1 <pid>     # 栈打到该进程的 stderr（重定向到文件才看得到）
```

```python
# 事前设一个有界看门狗：N 秒后打印所有线程的 Python 栈并退出
import faulthandler
faulthandler.dump_traceback_later(60, exit=True)
...
faulthandler.cancel_dump_traceback_later()
```

栈顶那一行就是答案。上一次它指向 `dataset.py:753 in __iter__`——也就是
`w.buffer.recv()`，而事故报告说的是 `idqueue.pop()`：**差一行，根因完全不同**。
不要用报告里的行号开工，自己打一次栈。

## 把挂死写成一条有界的测试

一条会挂的用例比一条失败的用例贵得多：它拖满整个 timeout，还常常把整个 session 挂住。
所以**给子进程自带看门狗**：

```python
SRC = """
import faulthandler
... 起 worker、准备好之后 ...
faulthandler.dump_traceback_later(60, exit=True)   # 不要在 import jittor 之前设：
                                                   # 冷缓存要编译十分钟，会误杀
<被测的循环>
faulthandler.cancel_dump_traceback_later()
print("STILL_ALIVE")
"""
r = run_child_script(SRC)
assert r.returncode == 0, r.stderr.decode()[-4000:]   # 失败时输出里就有卡住的那一行
```

看门狗要**放在 worker 起来之后**，不要覆盖 `import jittor` 的冷编译时间。

`timeout` 一定要能真收尾：`subprocess.run(timeout=N)` 只杀直接子进程，
dataset worker 这类孙进程会继续跑并攥着 stdout 管道。用
`tests/_helpers/child_process.py`（`start_new_session` + `os.killpg`），不要自己起。

## 多进程唤醒路径的核查清单

「子进程发现错误 → 叫醒父进程」的实现，逐条问：

1. **父进程此刻在等哪个对象？** 不是"在等"，是"在等**哪一个**"。往 A 上推消息、
   而父进程睡在 B 上，等于没推。这是最常见的一种，且看起来完全正确。
2. **入队的是事实还是承诺？** 只有"数据已提交"才可以入队。dataset 的 worker 曾经
   在**加载这一批之前**就把自己的 id 推给父进程，于是父进程 pop 到 id、检查错误
   （还没发生）、进 `recv()` 阻塞——之后的错误报告推给了它已经不再看的队列。
   不变式要写成："队列里有 id ⇒ 要么一整批可读，要么错误已存好"。
3. **先存后叫。** 先把错误写进共享槽，再叫醒；反过来会让父进程醒来看见空槽。
4. **共享内存的条件变量能跨进程叫醒吗？** Jittor 的 `RingBuffer` 可以：`is_stop`
   与两个 cv 都在 `mmap(MAP_SHARED)` 里、`pthread_condattr_setpshared`。所以**将死的
   子进程可以直接 `buffer.stop()`**，让睡在 `recv()` 里的父进程抛出 `runtime_error("stop")`。
   注意 `stop()` 用的是 `pthread_cond_signal`（只叫醒一个），多个等待者时不够。
5. **子进程被信号杀死（SIGKILL/OOM/段错误）这条路谁兜？** 它跑不到任何 Python 收尾
   代码：不写共享槽、不 `stop()`、不推队列。兜底只能来自父进程自己的**有界等待**：
   `RingBuffer::wait_pop_for`（`src/runtime/ring_buffer.h`）用 `pthread_cond_timedwait`
   等，`PyMultiprocessRingBuffer::pop_for(timeout_ms)` 在等的期间释放 GIL、超时抛
   `ring buffer pop timed out`；dataset 的主循环用 `idqueue.pop_for(5000)` 取 worker id，
   超时交给 `_raise_worker_death` 报错。`tests/data/test_dataset.py::TestDatasetSeed::test_children_died`
   杀掉一个 worker 后断言子脚本在 20 s 内以非零退出、stderr 里有那句超时；
   `tests/structure/build/test_ring_buffer_timeout_contract.py` 钉住「`pop()` 仍无界、
   `pop_for` 显式有界」这个契约。新加的阻塞等待照这个形状写：**有界、释放 GIL、超时抛错**，
   不要指望信号或看门狗线程来救。
6. **不要退回"给父进程发信号"。** Jittor 装了进程级 SIGCHLD/SIGINT 处理器：
   SIGINT 分不清是不是用户按了 Ctrl-C，处理器会直接退进程；旧的 SIGCHLD 处理器
   会让父进程**无声消失**。挂住很糟，无声消失更糟。
