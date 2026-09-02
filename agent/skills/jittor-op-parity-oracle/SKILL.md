---
name: jittor-op-parity-oracle
description: 给 Jittor 的 Python 层算子做数值对拍（correctness oracle）。当你要证明某个算子「静默算错」或「已经修对」时用它——如何选对拍基准（numpy/scipy 优先，必要时用独立进程里的真 PyTorch）、为什么绝不能在 Jittor 环境里 import torch、容差怎么定、fail-before/pass-after 怎么做、以及 code-op 内核（jt.code）出错时的取证手法。
---

# Jittor 算子对拍口径

修「静默算错」类缺陷时，**"改完测试绿了" 不算证据**。证据是：同一个测试在修之前失败、
修之后通过，且期望值来自一个与 Jittor 无关的独立实现。本 skill 给出选基准、起进程、
定容差、做取证的固定做法。

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
写进测试，或者用 `unittest.skipUnless(os.environ.get("REAL_TORCH_PY"), ...)`。固化时在
注释里写清楚是哪个 torch 版本、哪段脚本算出来的。

## 3. 容差

- float32 前向：`rtol=1e-4, atol=1e-4`。
- float32 反向 / 有限差分：`rtol=2e-3, atol=2e-3`（有限差分本身就只有 3–4 位有效数字）。
- **凡是要证明"确定为某个值"（例如「梯度必须恰好是 0」「不是未初始化内存」）的，用
  `assert (arr == 0).all()`，不要用 allclose。** 未初始化内存有相当概率碰巧接近 0。
- dtype 陷阱：`jt.array(np.ones(4, "float64"))` 会静默降成 float32。要 float64 必须
  `jt.array(v, dtype="float64")`。

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

## 5. fail-before / pass-after 的机械做法

```bash
git stash push -q <只 stash 被改的源文件>        # 测试文件是 untracked，不会被 stash
<跑测试>                                        # 必须看到 FAILED，并确认失败原因就是该缺陷
git stash pop
<跑测试>                                        # 必须全绿
```

失败原因要看一眼：如果修前是 `ImportError` / 形状不匹配之类的**别的**原因，说明测试
测的不是这个缺陷。

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
