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

- `pytest` 是安全的：`pyproject.toml` 里有 `pythonpath = ["python"]`，天然导入当前树。
  要测别的副本才需要 `-o pythonpath=<那个副本>/python`。
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
