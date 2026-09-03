---
name: jittor-op-parity-oracle
description: 给 Jittor 的 Python 层算子做数值对拍（correctness oracle）。当你要证明某个算子「静默算错」或「已经修对」时用它——如何选对拍基准（numpy/scipy 优先，必要时用独立进程里的真 PyTorch）、为什么绝不能在 Jittor 环境里 import torch、容差怎么定、fail-before/pass-after 怎么做、以及 code-op 内核（jt.code）出错时的取证手法。
---

# Jittor 算子对拍口径

修「静默算错」类缺陷时，**"改完测试绿了" 不算证据**。证据是：同一个测试在修之前失败、
修之后通过，且期望值来自一个与 Jittor 无关的独立实现。本 skill 给出选基准、起进程、
定容差、做取证的固定做法。

## 0. 先确认你测的是哪一份代码

jittor 通常是 **editable 安装**（site-packages 里的 `.pth` 指向某一个固定的源码树）。
在另一个 worktree / 副本里裸跑 `python -c "import jittor"` 或 `python script.py`，
导入的是 `.pth` 指向的那一份，**不是你刚改的那一份**，于是"验证通过"毫无意义。

- `pytest` 是安全的：`tests/conftest.py` 把本 checkout 的 `python/` 放在 sys.path 最前，天然导入当前树。
  要测别的副本用 `JITTOR_SOURCE_ROOT=<那个副本> pytest ...`。
- **所有手写脚本、`python -c`、子进程都必须显式带** `PYTHONPATH=<你的源码树>/python`。

第一次验证前先跑这句，打印的路径必须是你正在改的那棵树：

```bash
PYTHONPATH="$TREE/python" python -c "import jittor, os; print(os.path.dirname(jittor.__file__))"
```

## 1. 选对拍基准：三档，从上往下选

| 档 | 用什么 | 适用 | 代价 |
|---|---|---|---|
| A | **numpy / scipy 直接算**（含手写的教科书式 N 重循环） | conv、pool、fft、linalg、reduce 等有闭式定义的算子 | 0，同进程 |
| B | **有限差分（对 A 的前向做数值微分）** | 反向/梯度 | 0，但要挑点 |
| C | **真 PyTorch，独立子进程** | 只有 torch 定义了语义的算子（`return_indices` 的索引编码、`MaxUnpool` 的重复索引累加规则、`autograd.functional.vjp/jvp` 的签名语义…） | 要起子进程 |

优先 A。A 的手写参考实现要**直白到不可能和被测实现共享 bug**：不复用 Jittor 的
reindex/broadcast 语义，就写 for 循环，`float64` 累加。

档 B 的写法：不要对整个张量做有限差分（慢且噪声大），只挑 2–4 个坐标：

```python
loss = (op(x) * seed).sum()          # seed 是固定的随机张量，别用 全 1
gx, = jt.grad(loss, [x])
fd = (ref_loss(x + eps*e_idx) - ref_loss(x - eps*e_idx)) / (2*eps)   # eps=1e-4，参考实现走 float64
```

## 2. 真 PyTorch 只能在独立进程里跑

**Jittor 开发环境里的 `torch` 是 Jittor 自己的 shim。** 在里面 `import torch` 得到的是
被测对象本身，拿它当 oracle 等于自己和自己比，任何错都对拍不出来。

做法：把 oracle 脚本写到临时目录，用**另一个** conda env 的解释器起子进程，只通过
stdout 上的 JSON / `.npy` 文件交换数据。

```bash
# 找到真 torch 的解释器；REAL_TORCH_PY 由调用方提供，不要把路径写进仓库
"$REAL_TORCH_PY" -c "import torch, sys; \
    assert 'jittor' not in torch.__file__, torch.__file__; \
    print(torch.__version__, torch.__file__)"
```

上面那句断言是**必须的**：先证明拿到的不是 shim，再信它的输出。

固定骨架（放到 scratch 目录，不进仓库）：

```python
# oracle.py —— 只用 torch + numpy，把结果存成 npz
import numpy as np, torch
d = np.load(IN_NPZ)
...
np.savez(OUT_NPZ, y=y.detach().numpy(), gx=x.grad.numpy())
```

```python
# 测试侧
subprocess.run([os.environ["REAL_TORCH_PY"], "oracle.py", in_npz, out_npz], check=True)
```

测试用例本身**不要**依赖真 torch（CI 里没有）：把 oracle 跑出来的数字**固化成常量**
写进测试，或者换成一个能算出同样结果的 numpy 参考实现。固化时在注释里写清楚是哪个
torch 版本、哪段脚本算出来的。仓库里已有的
`tests/_helpers/torch_runtime.py`（`modules_available` / `import_torch_modules` +
`REAL_TORCH_SITE`）是给"进程里预装了二进制 torch"的那种会话用的，没预装时它会让整个
用例 skip——所以**不要**把新的对拍用例挂在它上面，否则日常跑的是 skip 不是绿。

## 3. 容差

- float32 前向：`rtol=1e-4, atol=1e-4`。
- float32 反向 / 有限差分：`rtol=2e-3, atol=2e-3`（有限差分本身就只有 3–4 位有效数字）。
- **凡是要证明"确定为某个值"（例如「梯度必须恰好是 0」「不是未初始化内存」）的，用
  `assert (arr == 0).all()`，不要用 allclose。** 未初始化内存有相当概率碰巧接近 0。
- dtype 陷阱：`jt.array(np.ones(4, "float64"))` 会静默降成 float32。要 float64 必须
  `jt.array(v, dtype="float64")`。
- **测试互相污染**：`jt.flags.use_cuda` 是进程全局的，某些既有用例把它置 1 后不复原，
  于是同文件里后面的用例全跑在 CUDA 上。写新用例时在 `setUp`/`tearDown` 里存取复原它，
  否则单跑绿、全量跑红。**修的时候用 `@jt.flag_scope(use_cuda=1)` 装饰器，不要裸赋值。**
  验证泄漏有没有真的修掉，用一个不经过 pytest 的小脚本：

  ```python
  import unittest, jittor as jt, importlib.util
  spec = importlib.util.spec_from_file_location("m", "tests/<...>.py")
  m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
  unittest.TextTestRunner(verbosity=0).run(
      unittest.TestLoader().loadTestsFromTestCase(m.<那个类>))
  assert jt.flags.use_cuda == 0, "use_cuda leaked!"
  ```

## 4. 「未初始化内存」类缺陷的判据

只跑一次看到 0 不算数。判据是**投毒**：先让被测缓冲区里装满非零的脏数据，再跑被测路径，
看输出是否被确定性地写成 0。

```python
# 让分配器大概率复用一块刚被写满非零值的显存/内存
junk = [jt.ones(shape) * 12345.0 for _ in range(8)]
for j in junk: j.sync()
del junk
out = op_under_test(...)          # dout 全零的分支
assert (out.numpy() == 0).all()   # 必须是 ==，不是 allclose
```

配套做法：重复 N 次（`for _ in range(20)`），每次都必须恰好为 0。一次侥幸不是证据。

更强的一招（不用投毒也成立）：把"多加了一个零贡献项"的结果与"没加这一项"的结果做
**逐位相等**断言（`assert_array_equal`）。零贡献加上去必须不改变任何一位；只要有一位变了，
那一项就不是零。

`jt.numpy_code` 的 backward 拿到的 `out` 是**未初始化**的：每条分支都必须 `np.copyto(out, ...)`，
`if` 没有 `else` 就是漏写。触发方式：让某个输出的 `dout` 全零但仍在图里，例如
`loss = f(w).sum() + (v * jt.array(np.zeros(...))).sum()`——只用 `f(w)` 不够，
那样反向根本不会被调用。

## 5. fail-before / pass-after 的机械做法

**不要用 `git stash`。** stash 栈存在公共的 `.git` 里，多个 worktree 共用同一个栈，
`pop` 出来的可能是别人的改动。用补丁文件：

```bash
git diff <被改的源文件> > "$TMPDIR/wip.patch"   # 测试文件保持在树里，它是新增的
git checkout -- <被改的源文件>
<跑测试>                                        # 必须看到 FAILED
git apply "$TMPDIR/wip.patch"
<跑测试>                                        # 必须全绿
```

失败原因要看一眼：如果修前是 `ImportError` / 形状不匹配之类的**别的**原因，说明测试
测的不是这个缺陷。反过来，修前"挂死 / 段错误"也是有效证据（越界循环就是这样），
用 `timeout N` 记录退出码 124 即可，不必要求它以 assert 失败。

## 6. `jt.code` 内核（code op）缺陷的取证

Pool/unpool 这类算子的 kernel 是 f-string 拼出来的 C++。定位手法：

1. **先把 kernel 源码打出来读**，不要盯着 Python 猜：
   `python -c "...; m = Pool3d(...); print(m.execute.__doc__)"` 不管用，直接
   `sed -n` 读生成 kernel 的那段 f-string，把 `{...}` 手工代入。
2. **CPU 与 CUDA 是两段独立的源码**（`cpu_src`/`cpu_grad_src` 对
   `cuda_src`/`cuda_grad_src`），2D 与 3D 又是两个文件。**修 3D 时一定要和 2D 对读**，
   一致的地方不一致就是 bug。
3. `cuda_grad_src` 里 `out` 是**输入的梯度**（形状同输入），`pout` 是**前向输出**，
   `dout` 是前向输出的梯度。反向 kernel 的循环上界必须是 `pout_shape*`，
   用 `out_shape*` 就会多跑/少跑（3D 的 CUDA 反向就踩了这个）。启动配置用 `pout_shape*`
   而 kernel 循环用 `out_shape*` 是同一个 bug 的两半，看到不一致就是它。
4. 无上界的 C++ 循环（写错循环变量）会越界读，表现是**段错误或整机卡住**，不是
   "结果不对"。所以这类改动要在**带 timeout 的子进程**里先复现，别在主进程里跑：
   `timeout 120 python repro.py; echo "exit=$?"`（124 = 超时 = 死循环）。

## 6.5 「索引编码 / 解码」类缺陷的定位顺序

pool→unpool、argmax→scatter 这类算子，索引是一个**扁平下标**，编码和解码各用一次形状。
出错时按这个顺序查，**不要一上来就改解码表达式**：

1. 编码那一侧用的是谁的宽度？（pool 前向用的是 `in0_shape*`，即**原始输入**的 H/W）
2. 解码那一侧的形状是怎么来的？**优先怀疑「输出形状的默认值」**，而不是解码表达式本身。
   解码表达式用输出形状是对的；错的往往是输出形状被默认成了别的东西。
3. 解码越界会发生什么？`reindex_reduce` 把越界目标当 overflow **静默丢弃**，
   所以症状是「一部分值凭空消失」而不是报错。判据：`out.sum() == pooled.sum()`。
4. 反解 kernel 尺寸的默认输出形状必须**反演前向公式**：`(pooled-1)*stride + kernel`
   （torch 的 `_unpool_output_size`）。写成 `pooled*stride` 时 stride==kernel 下恰好相等，
   所以只有 stride != kernel 才暴露——这类「默认参数只在常用配置下正确」的 bug
   要专门构造 stride != kernel 的用例。

## 7. `reindex` 与 `reindex_reduce` 里 `xshape`/`yshape` 的方向是反的

这是最容易改错方向的地方，实测结论（见 `python/jittor/src/ops/*.cc`）：

| op | `xshape*` | `yshape*` | 迭代变量 `i0..iN` 走的是 |
|---|---|---|---|
| `reindex` | **输入**的形状 | **输出**的形状 | 输出形状 |
| `reindex_reduce` | **输出**的形状 | **输入**的形状 | 输入形状 |

30 秒验证法（别靠记忆）：

```python
x = jt.ones([2, 3])
for name in ("xshape1", "yshape1"):
    print(name, x.reindex_reduce("add", [2, 7], ["i0", f"({name})"]).numpy())
# 写到第 3 列的那个 name 等于输入的 shape[1]==3；全 0 的那个越界了，等于 7
```

## 7.5 `use_cuda` 打开时，`jt.numpy_code` 里的 `np` **不是 numpy，是 cupy**

`pyjt/py_converter.h` 在 `use_cuda` 为真时 import 的是 `cupy`，并通过
`init_cupy.numpy2cupy` 把 `data` 里的每个数组换成 `cupy.ndarray`。所以回调里的
`np.linalg.*`、`np.einsum`、`np.copyto` 全都是 cupy 的实现。影响到**所有**
`numpy_code` 算子（linalg 全家、cumsum 的 CPU 路径、gamma 采样……）：

- 同一个 `jt.linalg.*` 在 CPU 上是 LAPACK、在 CUDA 上是 cuSOLVER，**是两套库**。
- 回调里若用了 cupy 没有的 numpy API，只在 CUDA 上炸。
- 调试时 `data["inputs"][0].flags['WRITEABLE']` 这类写法在 CUDA 上直接 KeyError——
  这也是判断"我现在拿到的是 cupy 还是 numpy"最快的一招。

**对拍推论：分解类算子（eigh/eig/svd/qr）不能用"和 numpy 的 U/V 逐元素相等"做判据。**
特征向量只定义到每列一个符号（重根还差一个子空间内的旋转），LAPACK 与 cuSOLVER 不
约定同一个符号。要断言的是**符号不变量**：

- 特征值 / 奇异值本身；
- 重建：`v @ diag(w) @ v.T == a`、`u @ diag(s) @ vh == a`；
- 正交性：`v.T @ v == I`；
- 梯度的**自洽性**：用该设备**自己返回的** `v` 代进闭式公式，而不是用 numpy 的 `v`；
- 跨设备比较时，损失必须是符号不变的（例如对重建矩阵求和，而不是 `(v*seed).sum()`）。

拿 numpy 的 `v` 当基准去对 CUDA 的梯度，会得到一个漂亮但完全错误的"相对误差 60%"
结论——那是符号约定不同，不是梯度算错了。

## 8. 「只在默认参数下正确」是这一类缺陷最常见的形状

反复出现的模式：某个**改变输出形状的参数**（`n`、`output_size`、`stride`、`groups`、
`ceil_mode`…）不给时恰好正确，一给非默认值就静默算错。原因是实现把「默认值下成立的
恒等式」写死了。

- `irfft(x, n)`：只有 `n == 2*(len-1)` 时对。正确做法是**先把半谱缩放到 `n//2+1`
  再做共轭镜像**；对镜像完的全谱缩放会从共轭对中间切开。
- `MaxUnpool(kernel, stride)` 的默认 `output_size`：只有 `stride == kernel` 时对。
- 分组卷积：`groups == 1` 的分支被测过，`groups > 1` 的分支从没跑通过。

**做法**：对每个影响输出形状的参数，列出至少三档——小于默认、等于默认、大于默认，
奇偶各一。已有的 opinfo / 回归样本往往**只覆盖默认值**（jittor 的 irfft opinfo 样本
全部是 `n == 2*(half-1)`），所以"既有测试全绿"完全不能说明这条路径被测过。
先去数一下样本里那个参数取过几个不同的值。

## 9. 两个收尾动作，做完修复别忘了

### 9.1 结构测试可能把 buggy 的值钉死了

`tests/structure/test_*_structure.py` 里有大量"默认实例的字段快照"。它们是照着**当时的
实现**写出来的，所以一个字段如果本来就算错，快照里存的就是错值。改了构造函数的字段
计算之后，这类测试会失败——**先判断是"我改坏了"还是"快照本来就钉的是 bug"**，
后者要同步更新快照并在旁边写清楚为什么变。别把它当成回归就把改动撤回去。

同类还有：`test_package_import_direction` 这种"实现模块只准 `import jittor`"的规则，
所以别在 `jittor/pool/*.py` 里随手 `import numpy`——纯 Python 算得出来就纯 Python 算。

### 9.2 已知差异要用"锁"钉住，不要留白

修完一处后往往还剩一批"知道不对但不属于本任务"的组合（例如 ceil_mode 的输出尺寸
规则）。两种做法都不好：跳过（差异从此隐形）、或者顺手一起改（越界）。用**已知差异锁**：

```python
def test_still_diverges_from_torch(self):
    """Known gap lock -- NOT a statement that this behaviour is right.
    If you fix it: this test fails; delete it and fold these cases into the
    parity test above."""
    diverging = [combo for combo in ALL if jittor_shape(combo) != torch_shape(combo)]
    self.assertEqual(diverging, [ ...穷举出来的那几个... ])
```

它同时防两件事：差异变大（回归）和差异被修好却没人更新测试。对拍测试本身则只比较
**双方都有定义**的那部分（例如只比 torch 真正会输出的那些平面），不要用 nan 兜底。

### 9.3 布尔参数被折进启发式条件，是"标量/元组不等价"的典型来源

`self.count_include_pad = count_include_pad and padding != 0` 这种写法读的是**原始入参**：
`padding=0` 恒为假、`padding=(0,0,0)` 恒为真（元组永远不等于 int），于是同一个语义的两种
写法走不同分支、算出不同的数。查这类问题的固定动作：**把每个"看起来只是校验/优化"的
`and`/`or` 条件拆开，对标量与元组各跑一遍，断言两者结果相同。**

## 10. 惰性图会把"就地改写"藏起来——测"有没有污染输入"必须先 sync

判断 `y = x.op(...)` 有没有偷偷改写 `x`，**不能**这样写：

```python
before = x.numpy().copy()
y = x.op(...)
assert (x.numpy() == before).all()      # ← 假绿！
```

写入发生在惰性图真正执行的时候。上面这句在 `y` 还没被求值时读 `x`，读到的当然是旧值；
就地实现同样能通过。正确写法是**先把结果物化，再读输入**：

```python
before = x.numpy().copy()
y = x.op(...)
y.sync()                                 # 先让写入真的发生
assert (x.numpy() == before).all()
```

`Var.scatter` 的就地缺陷就是这样漏掉的：读 `x` 在前、读 `y` 在后时结果是 0（干净），
调换顺序后 `x` 变成 15（被污染）。**"污染输入"这类缺陷的症状是顺序相关的，不是恒定的**，
所以复现脚本要把两种顺序都跑一遍并打印出来。

顺带：`Var.setitem` 本身是就地的（`y = x.setitem(...)` 之后 `x` 被改，且 `y is x` 为假），
所有转发到 setitem 的 API 默认都会继承这个语义。要 out-of-place 就得先 `clone()`。

## 11. 「同一概念的多份实现」怎么证明它们不等价

这一类任务（5.17/5.18 家族）最容易走偏的地方是：**把两份实现跑同一个用例、都绿，就当它们等价**。
不是。那只说明这个用例选在了它们重合的地方。

**判据：默认参数与良态输入下，两份实现往往恰好相等。要专门去找让它们分开的那个输入。**

实测例子，全部来自 2026-09-03 的 5.17：

| 两份实现 | 在什么输入下相等（藏住） | 在什么输入下分开（暴露） |
|---|---|---|
| BatchNorm 的 sync 分支（`E[x²]-E[x]²`）与非 sync 分支（两遍法 `mean((x-mean)²)`） | mean 0 / std 1：相对误差 < 1e-5 | **mean 100 / std 0.05**：相对误差约 7e-2。判据是 `var/mean²` 掉到 float32 的 1e-7 以下，两个平方项就在它们仅有的 7 位上相等，差出来的是舍入噪声 |
| `jt.pool.AvgPool2d` 与 `jt.nn.AvgPool2d` | `count_include_pad=True` 或 `padding=0`：完全相同 | **`padding>0` 且 `count_include_pad=False`**：旧实现根本没读这个 flag |
| `Pool.__init__` 的 `count_include_pad and padding != 0` | `ceil_mode=False`：该字段压根没人读，标量 0 与元组 (0,0) 走同一条路 | **`ceil_mode=True`**：才有代码去读它，元组恒真、标量 0 恒假的坑才发作 |
| `nn.Conv2d.execute` 与 `nn.functional.conv2d` | 任何合法输入：数值相同 | **非法输入**（3 维、通道数不符、输出尺寸 ≤ 0）：一个抛 ValueError，另一个报 unpack 错或撞 C++ 断言 |

**做法**：写对拍用例之前，先问「这两份实现在哪个参数上分叉」，然后**沿那个参数取值**，
而不是沿数据取值。分叉常见于：

- 只在某个分支里被读到的布尔参数（`count_include_pad`、`ceil_mode`）；
- 只在某个部署下才跑的分支（`sync`、`jt.in_mpi`、`use_cuda`）——这类最危险，
  因为**它在开发机上永远不跑**；
- 数值条件数（均值远大于标准差、极小方差、极大动态范围）；
- **非法输入**：两份实现的校验往往不一样，而校验差异不会被任何数值对拍发现。

## 12. 「两个拼写必须是同一份实现」的断言强度怎么选

合并重复实现之后，正确的回归断言是**逐位相等**（`assert_array_equal`），不是 `allclose`：
它们如果真是同一张图，就该逐位相同；容差会放过「又抄了一份、数值接近」的回归。

但 CUDA 上有三处例外，实测（不要靠猜，按下面的顺序量一遍）：

1. **前向逐位相等**：CPU 与 CUDA 都成立，这是最强也最稳的断言，优先靠它。
2. **反向在 CUDA 上不逐位可复现**。同一个调用跑两次，梯度最后一位就可能不同。已量到的：
   - 深度可分离 conv 的权重梯度（反向用 atomicAdd 累加）——**不可复现**；
   - 普通与分组 conv、`_ln_normalize` 系列——**可复现**。
   所以「CUDA 反向一律给容差」是偷懒，「CUDA 反向一律逐位」会 flaky。**写一条用例把
   哪个 kernel 不可复现量出来**，用它给容差背书，并让它在 kernel 变确定时失败提醒收紧。
3. **一边走融合 kernel、另一边走通用路径时，连中间统计量都不逐位相等**。BN 的
   `sync=False` 在 CUDA 上取融合 kernel，于是同一个 `jt.mean` 因为图的融合方式不同，
   归约顺序也不同，均值差 6e-8。这时 CPU 用逐位、CUDA 用容差，并在注释里写明
   「CPU 上它们是同一张图，逐位相等才是能抓住回归的那条断言」。

## 13. 只在 MPI 下才跑的分支，怎么在单机上测

`sync = self.sync and jt.in_mpi` 这种分支在开发机上永远是 False，于是它的缺陷
只能在分布式作业里暴露。不要为此去起 mpirun——**把 world size 伪造成 1**：

```python
identity = lambda self, op: self
original_reduce = getattr(jt.Var, "mpi_all_reduce", None)
original_in_mpi = jt.compile_extern.in_mpi
try:
    jt.Var.mpi_all_reduce = identity      # 一个 rank 的 all-reduce 就是恒等
    jt.compile_extern.in_mpi = True       # 见下
    assert jt.in_mpi
    ...                                   # 跑 sync 分支
finally:
    ...                                   # 逐个还原
```

这同时也是**判据本身**：一个 rank 的 all-reduce 不改变任何值，所以 sync 分支与非 sync
分支必须给出相同的输出与相同的梯度。它们不同，就说明 sync 只是名义上「多做一次通信」，
实际上是**另一份实现**。

⚠ **`jt.in_mpi` / `jt.rank` / `jt.world_size` 必须写 `jt.compile_extern.*`，不能写 `jt.*`。**
它们由 `compile_extern.distributed_state_getattr` 通过模块级 `__getattr__` 提供；给
`jt.in_mpi` 赋值会在 `jittor.__dict__` 里留下一个条目，**永久遮蔽那个访问器**，
后面所有读到的都是这个陈旧副本（6.B15 的提交说明写明了这一点）。


## 14. 「某个后端上这个算子是错的」——别信注释里的原因，自己按 dtype 扫一遍

多后端算子里最常见的注释是「X 后端只对 Y 支持得对，其余绕开」。这类注释**记录的是当时
观察到的症状，不是原因**，而绕行代码往往比缺陷本身危害更大（`unique` 的绕行把 float
输入送进了一个会截断排序键的 CPU 内核）。

**做法：把 Python 层的分派器整段旁路掉，让每个 dtype 都真的走进那个后端内核，扫一遍。**

```python
import inspect
src = inspect.getsource(jt.misc.unique)
head = src.index("    if jt.flags.use_cuda and")     # 分派器开头
tail = src.index("    temp_shape = None")            # 真正的实现开头
ns = {"jt": jt}
exec(compile(src[:head] + src[tail:], "<inner>", "exec"), ns)
inner = ns["unique"]                                  # 没有分派器的同一个函数
for dtype in ("int8","int32","int64","float16","float32","float64"):
    for use_cuda in (0, 1):
        ...
```

扫出来的形状本身就是诊断：

| 观察到的形状 | 通常的原因 |
|---|---|
| **恰好只有一个 dtype 对**，且那个 dtype 与索引类型相同 | 某个存索引的输出 var 用了 `输入.dtype`，kernel 往里 memcpy 原始 int32 字节；输入是 int32 时重解释是恒等的 |
| 只有偶数长度对，或只有 4 字节 dtype 对 | 手工在一块内存里切出来的 scratch 缓冲区对不齐（`(T*)(p + n)`）。改成两次独立 `alloc`，让分配器各自对齐 |
| 大值错、小值对 | 中间量被截断（`int lhs = @in(a,i)`），或计数用了 `int` |

第一行是 2026-09-03 的 `unique` 实测：注释说「cub 只对 32 位整数键排得对」，
实际 cub 没问题，错的是 `jt.code` 的 `output` 用了 `input_sorted.dtype`。

## 15. CUDA 的异步错误会落在无辜的用例上

`invalid configuration argument`、`misaligned address` 这类错误在**下一个同步点**才报，
所以 pytest 指的那个用例常常不是肇事者，而且它之后**整个进程的 CUDA 上下文都废了**，
于是同一次运行里后面所有 CUDA 用例一起变红（2026-09-03 一次 5 文件运行里 19 个失败，
真正的原因只有一个）。

**定位**：
1. 看失败用例**前面**那一个（同一个类里按方法名字典序，跨类按文件里的定义顺序）。
2. 有嫌疑的用例后面加 `jt.sync_all()`，让它在自己那里失败。
3. 一个文件一个进程重跑（`pytest 单个文件`），级联就消失了，只剩真失败。

**最常见的两个源头**：
- **零大小张量导致 0 个 block 的启动**。`BlockScanKernel<<<batch_num, ...>>>`，
  `batch_num` 来自 `shape[0]`；空输入压成 2 维就是 0 行。空输入要在 Python 层挡掉，
  不要让它进 kernel。
- **手工切分的 scratch 缓冲区不对齐**（见上一节）。

顺带：`.numpy()` 只同步**那一个** var 的图，不足以把异步错误逼出来；`jt.sync_all()` 才是。

## 16. 依赖 NaN/Inf 语义的表达式不能写成普通 Var 运算

jittor 的融合内核用 `-Ofast` 编译（`compiler.py` 里 `kernel_opt_flags += " -Ofast "`），
`-Ofast` 蕴含 `-ffast-math`、进而 `-ffinite-math-only`——**编译器可以假设不存在 NaN 和 Inf**，
于是 `x >= 0 || x <= 0`、`|x| == inf` 这类写法允许被折叠成常量。

所以：
- 任何 `isnan/isinf/isfinite/nan_to_num` 一类的语义**必须**进一个显式压低优化级别的
  `jt.code`。`misc/tensor_ops.py` 的 `_simple_for` 就是干这个的
  （`flag_scope(compile_options={"FLAGS: -O2 ":1})`）；
- 拿「用原始比较写一遍」当对拍基准也不行——它在 CPU/CUDA 上不可信。
  能对拍的只有 **dtype 策略**和**普通值上的答案**，把 NaN/±Inf 那几个元素排除掉，
  并在测试里写明排除的理由（这本身就是「两种写法必须分开」的证据）。
- CUDA JIT 还带 `--use_fast_math`（`compiler.py` 的 `nvcc_flags`），同样的道理。

## 17. 写 dtype 相关测试时的三个 jittor 陷阱

1. `jt.array(np.zeros(4, "float64"))` **静默变 float32**，int64 变 int32。
   要 `jt.array(v, dtype="float64")`——否则「1e300 被判成 inf」这类用例根本构造不出来。
2. `jt.array(v, dtype="bfloat16")` 直接 **TypeError**（numpy 没有 bfloat16）。
   要 `jt.array(np.zeros(shape, "float32")).cast("bfloat16")`。
3. **`jt.misc._foo` 不是自动可见的。** `python/jittor/misc/__init__.py` 是个 facade，
   下划线开头的名字要写进那份**显式再导出清单**才存在；而 `tensor_ops.py` 内部正是靠
   `jt.misc._foo` 做晚绑定的。新加私有 helper 忘了加清单，报的是
   `AttributeError: module 'jittor.misc' has no attribute '_foo'`，而且只在真正调用时才炸。
   `tests/structure/test_misc_structure.py` 只校验**非**下划线的公开面，挡不住这个。

## 18. 索引 dtype 的取证：dtype 本身就是那个错答案

「索引应该是 int64」听起来像形式问题，直到你把它接到算术上。jittor 按**字节宽度**提升，
所以 `index * scalar` 停在索引自己的 dtype 里：

```python
idx = mask.nonzero().reshape((-1,))     # 四个元素，值 0..3
(idx * 1000000000).numpy()
# int32 索引 -> [0, 1000000000, 2000000000, -1294967296]
# int64 索引 -> [0, 1000000000, 2000000000,  3000000000]
```

四个元素就能证明，不需要 2^31 个。**凡是「某个 dtype 应该更宽」的任务，都先找这种
「小输入 + 大标量」的算术，它比构造大张量便宜几个数量级。**

真需要跨过 2^31 时（比如证明一条 CUDA 快路径的 `int` 索引会回绕）：挑 int8 输出，
2^31 个元素只要 2.1 GiB，24G 卡上 6.5 秒。这类用例加 `@pytest.mark.manual`
（`JITTOR_TEST_MANUAL=1` 才跑），并在 docstring 里写清楚显存开销和实测耗时。

## 19. 工作树里混了两个任务时，怎么拆成一个任务一个提交（禁止 stash）

会话中断或者一时手快，很容易在同一个文件里叠了两个任务的改动。`git stash` 是禁止的
（所有 worktree 共用一个栈）。可行的做法是**按 hunk 建索引**：

```bash
git diff -- path/to/file.py > all.patch
python3 split_hunks.py all.patch out          # 按 hunk 内容分类，见下
git apply --cached out.taskA.patch            # 只把 A 的 hunk 放进索引
git add <A 的其它文件>
git commit -m "[A] ..."                       # 工作区仍然是全量，不受影响
git apply --cached out.taskB.patch
git commit -m "[B] ..."
```

分类脚本按 hunk 正文里出现的标识符归类，并且**要求每个 hunk 恰好命中一类**，
命中 0 类或 2 类就报错退出——这条断言比分类规则本身重要，它挡住「某个 hunk 被悄悄
漏掉或进错提交」。

最后一步必须校验：`diff <(git show :path/to/file.py) 你实际测过的那份`，
确认最终索引与跑绿的那份逐字节相同。

预防办法还是 brief 第 9 节那条：**一条任务做完就提交**，别让两个任务的 WIP 同时留在树里。
