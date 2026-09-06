---
name: process-global-state-and-optin
description: 证明一段代码在"不该动的地方"留下了全进程状态，并把它改成显式 opt-in 且可撤销。用于 import 期副作用（顶层调用改了别人库的全局状态）、猴补丁（替换第三方模块的属性且无卸载接口）、装在类上而非实例上的 hook，以及"把共享实例状态改成每次调用的一次性上下文"之后的收尾检查。判据是：改完之后进程里还看得见你来过吗？父进程观察不到自己的 import，所以这类测试一律起子进程。
---

# 全进程状态：怎么证明它在，怎么改成 opt-in

这类缺陷的共同形状是**写在了活得太久的对象上**：模块顶层、第三方模块的属性、类对象。
症状永远是静默的——没有异常、没有日志，只是别人的东西变了。

## 1. 先判断它属于哪一类

| 写在哪 | 典型写法 | 谁受害 |
|---|---|---|
| 模块顶层 | `set_global_seed(...)` 直接写在 `.py` 里 | 任何 import 到它的进程，包括间接 import |
| 第三方模块属性 | `setattr(PIL.Image, "open", timer)` | 全进程所有用 PIL 的库 |
| 类对象 | `cls.__call__, cls.__hooked_call__ = ...` | 该类**此后所有实例**，且沿 MRO 波及子类 |
| 共享实例 | `self.x = x` 存反向中间量 | 同一实例的上一次调用 |

前三类都不可撤销，第四类是"下一次调用覆盖上一次"。修法都是把状态挪到寿命正确的对象上，
并给出显式的进入/退出。

## 2. 修前失败怎么证：父进程观察不到自己的 import

**测 import 期副作用必须起子进程。** pytest 进程早就 `import jittor` 过了，在里面写
`np.random.seed(0); import jittor` 什么都测不到——jittor 已在 `sys.modules` 里，第二次
import 是空操作。

用 `tests/_helpers/child_process.py` 的 `run_child_script`（**不要裸起子进程**，那会导入
主树而不是你的 worktree，见 `jittor-worktree-verification`）：

```python
from _helpers.child_process import run_child_script

PROBE = '''
import numpy as np
np.random.seed(0)
expected = np.random.rand(5).tolist()
np.random.seed(0)
import jittor as jt              # 被测的那一下
got = np.random.rand(5).tolist()
assert got == expected, "%r != %r" % (got, expected)
print("DONE")
'''
done = run_child_script(PROBE, text=True, merge_stderr=True, name="probe")
assert done.returncode == 0, done.stdout
```

要点：
- **先设种子、取参考、再重设种子、再 import**。只写"import 后 rand 不等于某个常数"会误报。
- 断言 `"DONE"` 在输出里。子进程可能因为别的原因退出 0（见 AGENT-BRIEF §7 的 SIGCHLD 陷阱）。
- 同一套写法测 `random`、`cupy`；分别写，别合成一条——合成之后你不知道是哪一个漏了。

## 3. 猴补丁：光看"是不是同一个对象"不够

`PIL.Image.open` 被换成一个 `HookTimer` **实例**（不是函数）之后，坏掉的不止是身份：

```python
import inspect, pickle, PIL.Image
original = PIL.Image.open
import jittor.dataset                      # 被测的那一下
assert PIL.Image.open is original          # 身份
assert inspect.signature(PIL.Image.open) == inspect.signature(original)
assert pickle.loads(pickle.dumps(PIL.Image.open)) is original   # 按引用可 pickle
```

第二、三条是真正会咬人的：`inspect.signature` 在装饰器、序列化、参数校验里到处都是，
而一个普通对象既没有 `__name__` 也不能按引用 pickle。所以**即使确实需要包一层，也要包成
`functools.wraps` 过的函数**，属性看起来仍然像它替换掉的东西。

## 4. opt-in 的三条最低要求

```python
with jt.dataset.time_image_open() as timer:   # 1. 显式进入
    ...
assert PIL.Image.open is original             # 2. 出了作用域必须还原
```

3. **可嵌套**。共享的 hook 对象被两层 `with` 用到时，内层退出不能把外层的 hook 拆掉——
   用安装深度计数，不要用布尔 `installed`：

```python
def install(self):
    self._depth += 1
    if self._wrapper is not None: return self
    ...
def uninstall(self):
    if self._depth == 0: return
    self._depth -= 1
    if self._depth: return
    ...
```

还原时先确认**当前值还是自己装的那个包装**；不是就别还原，只告警——盲目还原会把别人后装的
补丁静默删掉，那正是你在修的那类缺陷。

## 5. 改成"每次调用一次性上下文"之后，必须做的收尾

把 `self.x` 挪到 per-call ctx（`object.__new__(type(self))` + 浅拷 `__dict__`）之后，
**去找所有还在往实例上记账的包装层**。它们会分成两种，只有一种会当场报错：

- 在调用**之前**写的：上下文是在调用开始时从实例拷出来的，所以它**碰巧还能用**。
  测试是绿的，但同一实例并发/重入时就串了。
- 在调用**之后**写的：上下文早就拷完了，`grad()` 里 `getattr(self, "...", None)`
  **永远是 None**，整块逻辑静默失效。

grep 的落点是"包装了 `__call__` 又在里面写 `self.`"：

```bash
grep -rn "_orig_.*call\|__call__ =" --include=*.py python/ | grep -v test
```

修法不是把 ctx 暴露成 `self._last_ctx`（那又回到共享状态），而是**给一个能在指定上下文上
跑一次调用的接缝**：

```python
def __call__(self, *args, **kw):
    return self._new_call_context()._run_call(*args, **kw)

def _run_call(self, *args, **kw):   # self 就是那个一次性上下文
    ...
```

包装层于是自己建 ctx、往 ctx 上记账、再 `ctx._run_call(*args)`，forward 和 backward
读到的是同一个对象。

## 6. 在 `python/jittor/_runtime/core_api.py` 里写代码的一个地雷

这个文件顶上有 `from jittor import *`，**它把好几个内建名字重新绑定成了 jittor 的
算子**：`bool`、`int`、`float`、`abs`、`max`、`min` 都不再是 Python 内建。所以

```python
return bool(d.get("_forward_hooks"))     # RuntimeError: Wrong inputs arguments,
                                         # Please refer to examples(help(jt.ops.bool))
```

报错信息完全不提"你以为在调内建"，只说某个算子的重载都不匹配。写这个文件时：

- 需要真值就直接把表达式交给 `if` / `or` 链，不要 `bool(...)`；
- 需要真正的内建，用文件里已有的别名（例如 `ori_int`），或者 `import builtins`。

文件里已经有注释警告过 `bool`，但只在一处；改这个文件之前先
`grep -n "ori_int\|rebinds the name" python/jittor/_runtime/core_api.py`。

## 6.5 全进程状态改成「可回滚 ledger」之后怎么验

当撤销面不止一个属性（模块属性 + `os.environ` + flags + `sys.meta_path` + `sys.modules`
+ `sys.path`），把它们记进一本账再逆序回放是唯一可审计的做法。这类改动**有三个判据是
「跑绿」覆盖不到的**：

**（1）中途失败必须逐项复原，不是「回滚到第一个冲突就算了」。**
回滚是逆序走的，所以在第一个冲突处 `return`/`raise` 会把**更早**的全部改写留在生效状态——
恰好是这本账想消除的半改写状态。正确做法是继续撤销其余条目，把撤不掉的一次列全：

```python
conflicts = []
for entry in reversed(self._entries):
    try:
        self._undo(entry)
    except TransactionConflict as conflict:
        conflicts.append(str(conflict))
if conflicts:
    self.state = "failed"          # 不能留在 "open"
    raise TransactionConflict("; ".join(conflicts))
```

测法是记三条（属性、env、属性），只让**最后**记的那条被外人改写，然后断言前两条已复原、
第三条保持外人的值、`commit()` 拒绝、`retry()` 接受。

**（2）「没泄漏全局锁」必须从别的线程探。**
`threading.RLock` 对**持有它的那个线程**是可重入的：

```python
lock.acquire()                          # 泄漏的那一下
assert lock.acquire(blocking=False)     # 仍然 True，什么都没证明
```

真检测器要起线程，并且**在获取它的那个线程里释放**（RLock 拒绝跨线程 release）：

```python
def install_lock_is_free(timeout=2.0):
    outcome = []
    def probe():
        acquired = Transaction._lock.acquire(timeout=timeout)
        outcome.append(acquired)
        if acquired:
            Transaction._lock.release()
    t = threading.Thread(target=probe); t.start(); t.join(timeout + 5.0)
    return bool(outcome) and outcome[0]
```

见 `tests/_helpers/install_lock.py`。同一个 helper 反着用一次：事务开着的时候断言
`not install_lock_is_free()`，否则「加了锁」这句话也没被证明。

**（3）失败路径要用 `finally`，并且要构造「回滚本身抛异常」。**
`rollback(); release(); state.pop(...)` 顺序写下来看着没问题，但回滚是**最可能抛**的那一步
（外部改写就抛）。注入方式：让一个安装步骤先经 ledger 写一个值、再把它改成别人的值、然后抛：

```python
def steal(context):
    context.state["_install_transaction"].mutate_attr(root, "_owned", "ours")
    root._owned = "someone else"
    raise RuntimeError("injected midway failure")
```

断言三件事：抛的是 `TransactionConflict`、事务状态是已知的（`failed`）、
`context.state` 里不再有 ledger 句柄、锁在别的线程看来是空的。

**（4）写入口的 helper 会在 install 之外被调到。**
`_set_use_cuda` 之类的 helper 同时服务安装期和 `torch.zeros(device="cuda")` 运行期。
所以 helper 里**必须查事务状态**，不能只查「有没有事务」——已 commit / 已 rollback 的账
再 `record()` 会抛 `RuntimeError: transaction is committed`，于是一次失败的安装能让此后
每一次 CUDA 工厂调用都报错。定向测法是把已关闭的事务塞进 context 再调 helper，
断言值写进去了：

```python
@pytest.mark.parametrize("closed", ("committed", "rolled_back"))
def test_writes_ignore_a_ledger_that_has_already_closed(closed):
    ...
```

## 7. 判据

- 子进程里 import 之前设的第三方随机种子，import 之后仍然复现。
- 被猴补丁的属性在作用域外 `is` 原对象，且 `signature`/`pickle` 与原对象一致。
- 给一个实例装 hook 之后，**另一个同类实例**的行为不变（类级别安装的判据）。
- 同一个 Function 实例连续调两次，第一次的反向仍然拿第一次的张量。

前三条都不是"跑绿了"能覆盖的：装了 hook 的那条路径本身能跑通，坏的是没装的那些。
