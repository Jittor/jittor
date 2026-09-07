---
name: graph-inference-to-explicit-record
description: 把一段「靠遍历算子图反推出语义关系」的代码改成「在关系产生的地方记下来」。包含为什么推断版的覆盖面无法审查、扁平化视图链以避免强引用改变存活计数、在同一个进程里隔离测量新热路径钩子的开销，以及「负控制用例和正用例一样重要」。改 var_holder.cc 的 cascade 写回、transpose 隐藏标记（5.03）、_torch_index_parent 链（7.12）之前读。
---

# 推断出来的关系，改成记录下来的关系

这个仓库里有一族相同形状的缺陷：**某个语义关系在产生时没有被记下来，于是后来需要它的代码
去遍历算子图把它猜回来。** 三个实例：

- `var_holder.cc` 的 `cascade_setitem_root`：`y[1][2] = c` 要写回 `y`，靠从结果 Var 往回
    10|  走 `getitem` 算子链找一个带 `holder` 的祖先；
- `core_api.py` 的转置隐藏标记（5.03）：`matmul` 要知道操作数是不是某个基张量的转置；
- compat 的 `_torch_index_parent`（7.12）：同一件事在 Python 层又做了一遍。

## 1. 推断版的真正问题不是慢，是覆盖面无法审查

`cascade_setitem_root` 的循环条件是 `n<10`，并且只接受 `vs.n == 1 && slices[0].is_int()`。
翻译成人话：**最多十层，每层必须是单个整数**。所以

```python
y[1][2] = c        # 走到了
    20|v = y[1:4]; v[0] = c  # 走不到，静默丢掉
```

而这一点在代码里**没有任何地方写着**。它不是注释里的一条限制，它是两个 `if` 的副产品。
没有测试会红，因为没人为「推断失败」写过用例——你不知道要为哪些形状写。

**判据**：一段推断代码的规格 = 它能识别的模式集合。如果这个集合只能靠读 `if` 反推出来，
那它就是不可审查的，无论它现在对不对。

记录版没有这个问题：`v = y[expr]` 的时候把 `(y, expr)` 存下来，覆盖面就是「所有走过这个
入口的表达式」，一句话说得清。

    30|## 2. 记录在哪一层：产生名字的那一层，不是产生值的那一层

这个坑值得单独说。jittor 有两个 getitem 入口：

- `jt.core.ops.getitem(x, s)` / `Var.getitem`：**算子**，做的是 gather；
- `Var.__getitem__`：**张量 API**，做的是「取一个视图」。

视图是关于**两个名字**的断言，不是关于一次计算的断言，所以记录点在 `__getitem__`
（本仓库是 `misc/indexing.py`），不在算子绑定里。这么分还有两个好处：`_is_basic_index`
这类「哪些索引算视图」的判断本来就在那一层；而已有的、直接调算子的测试
    40|（`tests/core/test_cascade_setitem_query.py`）不会因为你改了语义而被牵动。

## 3. 视图链要在创建时扁平化，否则你被迫持强引用

`inner = y[1][2]` 里的中间量 `y[1]` **在表达式结束时就死了**。所以

```cpp
struct VarView { VarHolder* base; VarSlices slices; };   // 天真版
```

对 `inner` 来说 `base` 是一个已经析构的 holder。两条出路：

    50|- **持强引用**：`inner` 让 `y[1]` 活着。torch 就是这么做的，但在这个仓库里它会**改变
  每一个切片过的测试的存活 Var 计数**——而 `tests/core` 有一簇用例正是在断言这些数
  （见 `jittor-refactor-gates` §6），你会同时收获一批看不懂的红。
- **扁平化**：记录的时候如果 `base` 自己也是视图，就把它的步骤链接过来，记到**根**上：

```cpp
struct VarView { VarHolder* base; vector<VarSlices> steps; };  // base 是根
```

  中间量随便死。写回的时候用 `getitem(root, steps[0])` 把它**重建**出来——它的值本来就
  恒等于这个。

    60|扁平化还顺带把「弱引用」变成可以接受的选择：记录唯一依赖的 holder 是**调用者自己命名的
那一个**，而一个没有名字的基张量，本来也只能通过这个视图读回来。

副产品：写回不再需要递归，也不需要深度上限。

## 4. 负控制用例和正用例一样重要

改的是 `assign`——**每个 `x.foo_()` 都从这里过**。所以测试里必须有等量的「不该发生」：

```python
def test_advanced_indexing_is_a_copy_not_a_view(): ...   # y[jt.array([1,3,5])] 不写回
def test_assign_on_a_non_view_still_rebinds(): ...       # (y+0).assign(...) 不碰 y
    70|def test_a_view_of_a_dead_base_still_assigns_locally(): ...  # 基没了也不崩
```

第一条尤其重要，而且**有过真实代价**：compat 里 `_torch_getitem` 的注释记着，给高级索引
的结果也留父指针，会在 Gaussian Splatting 的稠密化里跨代拉出一条优化器状态链。
torch 的规则（只有基本索引是视图）不是风格，是这个后果的解药。

## 5. 怎么量新钩子在热路径上的开销：同进程里关掉谓词

`x[i]` 是训练循环里最热的绑定之一，加一次记录要给出数字。**跨进程跑两个版本在这台机器上
量不出来**（常驻十几个 agent，负载 10 以上）。同一个进程里做 A/B：

    80|```python
from jittor.misc import indexing
with_view = bench(lambda i: x[i % 64])
indexing._is_basic_index = lambda idx: False    # 同一条代码路径，只是不记录
no_view   = bench(lambda i: x[i % 64])
indexing._is_basic_index = orig
again     = bench(lambda i: x[i % 64])          # 复测，确认不是漂移
```

第三次复测不能省：它把「关掉之后变快了」和「第二次总是更快」区分开。
    90|实测本例：记录一次视图 0.37 µs，占 `x[i]` 全程 1.97 µs 的 19%。

**不要拿 `x.getitem(i)` 当对照**——它绕过了整个 `__getitem__` 的 Python 规范化，
量出来的 1.2 µs 差值里只有 0.37 µs 是你的。

## 6. 旧的推断路径先留着，别和新记录一起跑

新旧两条路都能写回时，最坏的结果不是慢，是**写两遍**。把新路径设成优先并让它**吞掉**
旧路径的入口条件，而不是并列：

   100|```python
needs_cascade = not x._is_view() and x._needs_cascade_setitem()
```

这样有记录的走记录，没记录的逐字维持原状——回归面收敛到「有记录」这一类。
旧的 C++ 入口留给 11.01 那一轮统一删，它还被别人的测试直接调用着。
