---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# 循环网络：为什么普通 RNN 记不住，LSTM 能记住

到这里为止，前面所有教程的网络都有一个共同点：**输入的长度是固定的**。一张 16×16 的图、
一句话的定长编码、一个 batch 的特征向量——网络一次看完全部输入，输出一个结果。

这一篇处理**长度不固定**的输入：一段文字、一个时间序列、一串点击。这类数据不能一次性
喂进去，因为后面的位置要依赖前面的位置。处理它的基本结构是**循环**：让网络带着一个
**隐藏状态**（hidden state）一步一步地读，每读一个位置就更新一次状态。

这一篇会：

* 把「循环」这件事**手写出来**，并证明手写的和内置的 `nn.RNN` **逐位相同**——内置的
  没有魔法，它就是那个循环；
* 用 `jt.grad` 直接量出梯度穿过整条链之后还剩多少，看到它**随长度指数衰减**；
* 手写一个 LSTM，说明它凭什么能救场；
* 用一个**能测的记忆任务**做一个决定性实验：**同一个任务，短依赖两个都能学会，
  长依赖普通 RNN 完全学不会、LSTM 照样学会**；
* 讲清楚**截断 BPTT** 在做什么——它不是近似求导的技巧，而是明确地**剪断梯度链**。

> 全篇 CPU 可跑，不需要数据下载。训练部分实测 40 秒左右。

```{code-cell} ipython3
import time

import numpy as np
import jittor as jt
from jittor import nn

jt.flags.use_cuda = 0        # 本篇用 CPU；有显卡时可以改成 1
jt.set_global_seed(0)
```

## 1. 序列进来是什么形状

序列张量比图片多一个维度。Jittor 沿用的是「**时间在前**」的布局：

| 形状 | 含义 |
| --- | --- |
| `(seq_len, batch, feature)` | 默认布局，`batch_first=False` |
| `(batch, seq_len, feature)` | `batch_first=True` 时用的布局 |

**这是第一个坑。** PyTorch 的 `nn.LSTM` 默认也是「时间在前」，所以严格说不算差异；
但很多人习惯写 `batch_first=True`，然后在别处忘了它，于是把 `(batch, seq, feat)` 当成
`(seq, batch, feat)` 用。程序不会报错，只会算出静悄悄错掉的结果。

```{code-cell} ipython3
x = jt.randn((7, 2, 3))          # (seq=7, batch=2, feature=3)

rnn = nn.RNN(3, 5)               # input_size=3, hidden_size=5
out, hn = rnn(x)
print("默认布局      out", out.shape, " hn", hn.shape)
print("  out 是每一步的隐状态；hn 只存最后一步")

lstm = nn.LSTM(3, 5, batch_first=True)
xb = jt.randn((2, 7, 3))         # 同样的数据，(batch=2, seq=7, feature=3)
out_b, (hn_b, cn_b) = lstm(xb)
print("batch_first   out", out_b.shape, " hn", hn_b.shape, " cn", cn_b.shape)
print("  注意 hn/cn 不受 batch_first 影响，永远是 (层数, batch, hidden)")
```

LSTM 比 RNN 多返回一个 `cn`：它有两份状态（`h` 和 `c`），后面会讲为什么。

## 2. 手写循环：把「记忆」写出来

「隐藏状态」听起来抽象，其实就是**一个向量，每步被更新一次**。普通 RNN 的更新规则只有
一行：

$$h_t = \tanh\big(x_t W_{ih}^\top + b_{ih} + h_{t-1} W_{hh}^\top + b_{hh}\big)$$

`x_t` 是这一步的输入，`h_{t-1}` 是上一步的状态。读完整条序列，最后的 `h` 就是「读到了
什么」。把它写出来。

```{code-cell} ipython3
HID = 5

# 取一个内置 RNN 的权重，用来逐位对照
ref = nn.RNN(3, HID)
w_ih, w_hh = ref.weight_ih_l0, ref.weight_hh_l0
b_ih, b_hh = ref.bias_ih_l0, ref.bias_hh_l0
print("weight_ih_l0", w_ih.shape, " (hidden, input)")
print("weight_hh_l0", w_hh.shape, " (hidden, hidden)")


def rnn_by_hand(seq):
    """seq: (seq_len, batch, feature) -> 最后一步的隐状态 (batch, hidden)"""
    h = jt.zeros((seq.shape[1], HID))
    for t in range(seq.shape[0]):
        h = jt.tanh(seq[t] @ w_ih.transpose(0, 1) + b_ih
                    + h @ w_hh.transpose(0, 1) + b_hh)
    return h


ref_out, ref_hn = ref(x)          # 用同一个模块，保证权重是同一份
diff = float((rnn_by_hand(x) - ref_hn[0]).abs().max())
print("\n手写 vs nn.RNN 的最大差异:", diff)
assert diff < 1e-6, "手写循环和内置实现不一致"
print("所以 nn.RNN 就是这个循环，没有任何额外的东西")
```

差值是 `6e-08` 量级——浮点累加的误差，不是算法差异。**这条结论很重要**：内置模块不是
一个黑盒，它只是把循环封装起来了。于是后面所有关于梯度的推理，都可以直接对着这个循环做。

## 3. 梯度要穿过整条链，而它会消失

现在看这个循环在反向传播时发生了什么。损失在最后一步，梯度要回到第一个时间步，必须
**逐个 Jacobian 相乘**：

$$\frac{\partial h_T}{\partial h_1} = \prod_{t=2}^{T} \frac{\partial h_t}{\partial h_{t-1}},
\qquad \frac{\partial h_t}{\partial h_{t-1}} = \mathrm{diag}\big(\tanh'(\cdot)\big)\, W_{hh}^\top$$

`tanh'` 最大是 1，`W_hh` 初始化时谱半径小于 1。**一堆小于 1 的数连乘，长度一上去就归零。**
这叫**梯度消失**（vanishing gradient）。这个式子也解释了为什么 `T` 稍大一点，第一个
时间步就什么都学不到。

不要停在公式上——直接量。

```{code-cell} ipython3
def grad_to_step0(kind, T):
    """d(loss)/d(x[0]) 的模长：有多少信号能传回第一个时间步。"""
    jt.set_global_seed(1234)     # 三种结构用同一份初始化分布
    cell = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[kind](2, 8)
    seq = jt.randn((T, 4, 2))
    out, _ = cell(seq)
    g = jt.grad(out[-1].sum(), seq)
    return float(np.linalg.norm(g[0].numpy()))


LENGTHS = (5, 20, 50)
print("%-5s %-12s %-12s %-12s" % ("结构", "T=5", "T=20", "T=50"))
grads = {}
for kind in ("rnn", "lstm", "gru"):
    grads[kind] = [grad_to_step0(kind, T) for T in LENGTHS]
    print("%-5s %-12.2e %-12.2e %-12.2e" % (kind, *grads[kind]))
print("\n这是 |d(loss)/d(x[0])|——传回第一个时间步的信号强度")
```

读这张表要注意两点，**不要过度解读**：

* **三种结构都在衰减。** 门控不是让梯度不消失，只是让它**慢得多**。从 T=5 到 T=50，
  普通 RNN 掉了 15 个数量级（约 `1e-01 → 1e-17`，已经掉进浮点噪声），GRU 只掉了 8 个
  （约 `1e-01 → 1e-08`）。同样的长度，门控结构多保住**好几个数量级**的信号。
  （具体的首位数字每次运行会差一点，数量级是稳定的。）
* **衰减速度决定了「能学多长的依赖」。** 这个数掉到浮点噪声以下，那个时间步就对训练
  完全没贡献了——不管它的信息有多重要。

画出来更直观。

```{code-cell} ipython3
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 3.2))
for kind in ("rnn", "lstm", "gru"):
    ax.plot(LENGTHS, grads[kind], "o-", label=kind.upper())
ax.set_yscale("log")
ax.set_xlabel("序列长度 T")
ax.set_ylabel("|d(loss)/d(x[0])|")
ax.set_title("梯度传回第一个时间步还剩多少")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()
```

## 4. 门控：手写一个 LSTM

LSTM 的核心改动是**多加了一条状态线** `c`（cell state），它的更新是**加法**：

$$c_t = f \odot c_{t-1} + i \odot g, \qquad h_t = o \odot \tanh(c_t)$$

三个门（`i` 输入门、`f` 遗忘门、`o` 输出门）都是 sigmoid，取值在 0~1 之间，作用是
**控制信息通过多少**。关键在于 `c_t` 的更新里，`c_{t-1}` 前面乘的是一个**门**而不是
**权重矩阵**：

$$\frac{\partial c_t}{\partial c_{t-1}} = f$$

所以梯度回传时连乘的是一串门（每个都在 0~1），而不是一串权重矩阵乘 `tanh'`。
如果网络学会让 `f` 接近 1，梯度就能几乎无损地传很远——**这就是「记忆通道」**。
这句话在下一节的实验里会变成一个可以读的数。

```{code-cell} ipython3
lref = nn.LSTM(2, 6)
lih, lhh = lref.weight_ih_l0, lref.weight_hh_l0
lbi, lbh = lref.bias_ih_l0, lref.bias_hh_l0
print("LSTM 的 weight_ih_l0", lih.shape, " = 4 * hidden，四个门的权重拼在一起")


def lstm_by_hand(seq, h, c):
    outs = []
    for t in range(seq.shape[0]):
        y = seq[t] @ lih.transpose(0, 1) + lbi + h @ lhh.transpose(0, 1) + lbh
        i, f, g, o = jt.chunk(y, 4, dim=1)      # 按门切开
        c = jt.sigmoid(f) * c + jt.sigmoid(i) * jt.tanh(g)   # 加法更新
        h = jt.sigmoid(o) * jt.tanh(c)
        outs.append(h)
    return jt.stack(outs, dim=0), h, c


seq = jt.randn((7, 3, 2))
b_out, (b_h, b_c) = lref(seq)
m_out, m_h, m_c = lstm_by_hand(seq, jt.zeros((3, 6)), jt.zeros((3, 6)))
print("\n手写 vs nn.LSTM:")
print("  out 最大差异 %.3e" % float((b_out - m_out).abs().max()))
print("  h   最大差异 %.3e" % float((b_h[0] - m_h).abs().max()))
print("  c   最大差异 %.3e" % float((b_c[0] - m_c).abs().max()))
print("两处 sqrt 之和的差异是浮点误差；结构完全一致")
```

## 5. 一个能测的记忆任务

公式讲完了，现在需要一个**能把「记不住」变成数字**的任务。这一篇用的任务叫
**带标记的值回忆**：

* 序列长 `T`，每一步是一个 2 维向量 `[值, 标记]`；
* **只有一个**位置 `k` 的标记是 1，其余是 0；
* 目标：在读完**整条序列之后**，输出第 `k` 步的那个值。

要做到这一点，网络必须在第 `k` 步把值存住，然后**一路带到最后**。信息要在状态里活过
`T - k` 步。`k` 是均匀随机的，所以总有样本需要在状态里存活很久——**这个任务直接考察
「能记多长」**。

注意非标记位置的值是**随机噪声**，不是 0。这一点是设计过的：如果它们都是 0，网络只要把
所有值加起来就能交差，根本不需要记忆。有了噪声，「求和」这条路被堵死。

```{code-cell} ipython3
def make_batch(T, B, seed):
    """返回 (输入, 目标, 标记位置)。输入形状 (T, B, 2)。"""
    rng = np.random.RandomState(seed)
    values = rng.uniform(-1, 1, (B, T)).astype("float32")
    flag = np.zeros((B, T), dtype="float32")
    pos = rng.randint(0, T, size=B)
    flag[np.arange(B), pos] = 1.0
    seq = np.stack([values, flag], axis=2)                 # (B, T, 2)
    target = values[np.arange(B), pos].reshape(B, 1)
    return jt.array(np.transpose(seq, (1, 0, 2))), jt.array(target), pos


x_demo, t_demo, p_demo = make_batch(T=8, B=3, seed=0)
print("输入", x_demo.shape, "（seq, batch, feature）")
print("目标", t_demo.shape, " 标记位置", p_demo)
print("目标就是标记位置上的那个值:", np.round(t_demo.numpy().ravel(), 3))
print("\n常数基线：永远输出 0，MSE ≈ 1/3，命中率 0")
```

两个网络。`RNN` 是上面那个手写循环，`LSTM` 是手写门控——都从零开始，用同一套代码路径，
只换状态更新规则。

```{code-cell} ipython3
HIDDEN = 32


class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(2, HIDDEN)
        self.rec = nn.Linear(HIDDEN, HIDDEN)
        self.head = nn.Linear(HIDDEN, 1)

    def execute(self, seq):
        h = jt.zeros((seq.shape[1], HIDDEN))
        for t in range(seq.shape[0]):
            h = jt.tanh(self.inp(seq[t]) + self.rec(h))
        return self.head(h)


class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(2, 4 * HIDDEN)
        self.rec = nn.Linear(HIDDEN, 4 * HIDDEN)
        self.head = nn.Linear(HIDDEN, 1)

    def execute(self, seq):
        h = jt.zeros((seq.shape[1], HIDDEN))
        c = jt.zeros((seq.shape[1], HIDDEN))
        for t in range(seq.shape[0]):
            y = self.inp(seq[t]) + self.rec(h)
            i, f, g, o = jt.chunk(y, 4, dim=1)
            c = jt.sigmoid(f) * c + jt.sigmoid(i) * jt.tanh(g)
            h = jt.sigmoid(o) * jt.tanh(c)
        return self.head(h)
```

训练和评估。评估时按**标记位置**把样本分成两组：`开头`（标记落在前四分之一）需要
跨过最长的一段，是最难的一类。

```{code-cell} ipython3
def train_and_report(kind, T, steps=400, lr=3e-3, batch=32, seed=0):
    jt.set_global_seed(seed)
    net = RNNModel() if kind == "rnn" else LSTMModel()
    optim = nn.Adam(net.parameters(), lr=lr)

    t0 = time.time()
    for step in range(steps):
        seq, target, _ = make_batch(T, batch, seed=1000 + step)
        loss = ((net(seq) - target) ** 2).mean()
        optim.step(loss)
    elapsed = time.time() - t0

    seq, target, pos = make_batch(T, 256, seed=99)
    pred = net(seq).numpy().ravel()
    truth = target.numpy().ravel()
    err = np.abs(pred - truth)
    early = pos < max(1, T // 4)
    return {
        "sec": elapsed,
        "mse": float((err ** 2).mean()),
        "recall": float((err < 0.2).mean()),
        "early": float((err[early] < 0.2).mean()),
        "n_early": int(early.sum()),
    }
```

## 6. 短依赖：两个都学得会

先跑最简单的版本，`T = 8`。梯度到第一个时间步的路径只有 7 步，还没衰减到不能用。

```{code-cell} ipython3
results = {}
for kind in ("rnn", "lstm"):
    results[("short", kind)] = train_and_report(kind, T=8)
    r = results[("short", kind)]
    print("T=8  %-5s  %5.1fs  MSE %.4f  命中率 %.3f  开头位置 %.3f (%d 个)"
          % (kind, r["sec"], r["mse"], r["recall"], r["early"], r["n_early"]))
```

两个都学会了。普通 RNN 的 MSE `0.011`、命中率 `0.92`；LSTM 更好（MSE `0.002`、
命中率 `1.00`），但**都是「学会了」**。到这里看不出谁不行。

## 7. 长依赖：普通 RNN 完全学不会

把 `T` 从 8 提到 32，其它全部不动。

```{code-cell} ipython3
for kind in ("rnn", "lstm"):
    results[("long", kind)] = train_and_report(kind, T=32)
    r = results[("long", kind)]
    print("T=32 %-5s  %5.1fs  MSE %.4f  命中率 %.3f  开头位置 %.3f (%d 个)"
          % (kind, r["sec"], r["mse"], r["recall"], r["early"], r["n_early"]))
```

**普通 RNN 完全崩了。** MSE `0.3254`，而「永远输出 0」这个常数基线的 MSE 是 `1/3 ≈ 0.3333`
——它一点都没学到，命中率 `0.199` 也正是「输出一个很小的随机数、偶尔碰巧落进容差」的水平。
它连训练集都没拟合上。

LSTM 在同一份数据、同样的步数下 MSE `0.0081`、命中率 `0.957`，**开头位置满分**。

```{code-cell} ipython3
rnn_long = results[("long", "rnn")]
lstm_long = results[("long", "lstm")]
rnn_short = results[("short", "rnn")]
lstm_short = results[("short", "lstm")]

assert rnn_short["recall"] > 0.4, "T=8 时普通 RNN 应该能学会"
assert lstm_short["recall"] > 0.8, "T=8 时 LSTM 应该能学会"
assert rnn_long["mse"] > 0.2, "T=32 时普通 RNN 应该学不会（接近基线 0.333）"
assert rnn_long["recall"] < 0.5, "T=32 时普通 RNN 的命中率应该接近随机"
assert lstm_long["recall"] > 0.7, "T=32 时 LSTM 应该仍然能学会"
assert lstm_long["early"] > rnn_long["early"] + 0.4, "长依赖上两者的差距应该很明显"

print("任务越长，两者的差距越大：")
print("  T=8   普通 RNN %.3f  ->  LSTM %.3f" % (rnn_short["recall"], lstm_short["recall"]))
print("  T=32  普通 RNN %.3f  ->  LSTM %.3f" % (rnn_long["recall"], lstm_long["recall"]))
print("\n注意 T=8 时两者都行——RNN 不是「不能用」，而是「只能记住短的」。")
```

这张对照表是**整个结构选择的依据**：

| | T=8 | T=32 |
| --- | --- | --- |
| 普通 RNN | 命中率 0.922 | 命中率 **0.199**（等于没学） |
| LSTM | 命中率 1.000 | 命中率 **0.957** |

它和 §3 的梯度表是同一件事的两种量法：那里量的是**梯度还剩多少**，这里量的是
**能不能学会**。

## 8. 截断 BPTT：主动把梯度链剪短

§3 的结论有一个直接的工程后果：既然梯度本来就传不远，**何必把整条链都建出来**。
把序列切成固定长度的段，每段结束后把隐藏状态**从计算图里摘出来**，只在这个段内反向传播。

被摘出来的状态不参与求导——这就是**截断 BPTT**。先确认摘掉确实切断了梯度。

```{code-cell} ipython3
w = jt.array(np.array([[3.0]], dtype="float32"))
x2 = jt.array(np.array([[2.0]], dtype="float32"))

print("不摘:  d(w*x)/dx =", float(jt.grad((w * x2).sum(), x2).item()))
print("detach: d(w*x)/dx =", float(jt.grad((w * x2.detach()).sum(), x2).item()))
print("stop_grad: d(w*x)/dx =", float(jt.grad((w * x2.stop_grad()).sum(), x2).item()))
print("\n.detach() 和 .stop_grad() 都在这里、都能剪断；没有 stop_grad_（原地版本）")
```

`jt.grad` 穿过被剪断的路径返回 **0**，不是报错——所以剪错位置会**静默**地让一部分
参数拿不到梯度。现在量一下「剪断」到底剪掉了多长的路径。

```{code-cell} ipython3
def steps_the_gradient_reaches(T, chunk):
    """返回哪些时间步的梯度还能传到最终损失（chunk=0 表示不截断）。"""
    jt.set_global_seed(7)
    hid = 8
    cell = nn.LSTMCell(2, hid)     # Cell 接收「一步」：(batch, feature) -> (h, c)
    head = nn.Linear(hid, 1)
    h, c = jt.zeros((1, hid)), jt.zeros((1, hid))
    marked = []
    for t in range(T):
        x_t = jt.randn((1, 2)) * 1.0
        x_t.start_grad()
        marked.append(x_t)
        h, c = cell(x_t, (h, c))
        if chunk and (t + 1) % chunk == 0 and t + 1 < T:
            h, c = h.detach(), c.detach()          # 只在段与段之间剪
    loss = head(h).sum()
    return [t for t, x_t in enumerate(marked)
            if float(np.abs(jt.grad(loss, x_t).numpy()).max()) > 0]


T = 12
print("T=12 不截断    梯度能到的步:", steps_the_gradient_reaches(T, chunk=0))
print("T=12 每 4 步剪   梯度能到的步:", steps_the_gradient_reaches(T, chunk=4))
print("T=12 每 6 步剪   梯度能到的步:", steps_the_gradient_reaches(T, chunk=6))
```

不截断时 12 步全部可达；每 4 步剪一次，**只剩最后一段 `[8, 9, 10, 11]`**；改成每 6 步，
剩的就是 `[6, 7, 8, 9, 10, 11]`。前面那些步的输入梯度全是 0，它们的贡献被彻底丢掉了。

> 两处实现细节值得记下来。第一，`nn.LSTM` 是**序列**模块（吃整条序列，返回 `out` 和
> 最终 `hn`），要自己驱动循环用 `nn.LSTMCell`（吃一步的 `(batch, feature)`，返回
> `(h, c)`）——这正是 §4 手写循环对应的内置版本。第二，**剪的位置是「段与段之间」**：
> 上面那个 `t + 1 < T` 不能省，否则最后一段的输出在算损失**之前**就被摘掉了，整个损失
> 会和所有输入脱钩（`jt.grad` 全返回 0，一个警告都不会有）。

这就是截断 BPTT 的真实代价与收益：

| | 不截断 | 截断（每 k 步） |
| --- | --- | --- |
| 梯度能回传的步数 | 全部 `T` | 只有最后 `k` 步 |
| 计算图大小 / 显存 | 随 `T` 线性增长 | 固定在 `k` |
| 长距离依赖 | 理论上可以，实际会衰减 | **明确放弃** |

**所以它是超参，不是免费的近似。** `k` 太小，模型学不会任何超过 `k` 步的依赖；
`k` 太大，又回到了显存和速度的问题。取 `k` 时要想的是「这个任务真正需要记多久」，
而不是「越大越好」。

## 9. 内置模块的形状与层序

实际写代码时用内置的 `nn.LSTM` / `nn.GRU` 就够了，不必手写循环。有两个返回值的细节
值得当面确认一次，因为弄错同样不会报错。

```{code-cell} ipython3
deep = nn.LSTM(3, 5, num_layers=2)
seq = jt.randn((9, 2, 3))
out, (hn, cn) = deep(seq)
print("两层 LSTM:  out", out.shape, " hn", hn.shape)
print("|out[-1] - hn[-1]| =", float((out[-1] - hn[-1]).abs().max()))
print("|out[-1] - hn[0]|  =", float((out[-1] - hn[0]).abs().max()))
print("hn[0] 是第 1 层，hn[-1] 才是最后一层（输出层）")
```

* `out` 是**最后一层**每一步的输出，形状 `(seq, batch, hidden)`；
* `hn` 按**层**索引，`hn[0]` 是第 1 层、`hn[-1]` 是最后一层；
* `out[-1]` 恰好等于 `hn[-1]`（差异 `0.0`）；只有一层时 `hn[0]` 就是 `hn[-1]`，
  所以单层网络里写 `hn[0]` 也对——**层数一变就会错**。

最后一层之外，中间层的输出不返回；要拿中间层输出得自己逐层调。

## 10. 同一件事，用 PyTorch API 再写一遍

循环网络在两种前端里的写法几乎逐字对应，因为 `nn.LSTM` 的签名是一致的。要留意的还是
那一件事：**激活必须在创建任何张量之前**，所以这一节自成一段。

```{code-cell} ipython3
from jittor.compat.shim import activate
activate()

import torch
import torch.nn as tnn

torch.manual_seed(0)

lstm_t = tnn.LSTM(2, HIDDEN, batch_first=True)
head_t = tnn.Linear(HIDDEN, 1)

seq_np, target_np, pos_np = make_batch(T=32, B=8, seed=5)
seq_t = torch.tensor(np.transpose(seq_np.numpy(), (1, 0, 2)))   # -> (batch, seq, feature)

out_t, (hn_t, cn_t) = lstm_t(seq_t)
pred_t = head_t(out_t[:, -1])          # batch_first 布局：直接取最后一列
loss_t = ((pred_t - torch.tensor(target_np.numpy())) ** 2).mean()
loss_t.backward()

print("torch 侧 out", tuple(out_t.shape), " hn", tuple(hn_t.shape))
print("torch 侧 损失 %.4f" % float(loss_t))
print("梯度回传到输入门权重了吗:", lstm_t.weight_ih_l0.grad is not None)
```

对照表：

| | 原生 Jittor | PyTorch 兼容层 |
| --- | --- | --- |
| 布局 | 默认 `(seq, batch, feat)`，可 `batch_first=True` | 同左 |
| 取最后一步 | `out[-1]`（时间在前） | `out[:, -1]`（batch 在前） |
| 梯度 | `optim.step(loss)` 内部完成 | 显式 `loss.backward()` |
| 剪断状态 | `.detach()` 或 `.stop_grad()` | `.detach()` |

`batch_first` 会改变 `out` 的索引方式，这就是 §1 那个坑在兼容层里的样子：**同一个
`out[-1]`，布局不同含义就不同。**

## 11. 什么时候用 RNN，以及检查清单

序列建模现在的主流是注意力（见 [从零实现 Transformer](transformer.md)），但循环结构
没有消失——它在**短序列、流式输入、状态可以持续维护**的场景里依然简单好用。选型的
依据就是这一篇量的那个数：**依赖有多长**。

```{code-block} text
1. 形状先确认布局             —— 默认 (seq, batch, feature)，别把两种搞混
2. 隐藏状态就是「一个向量每步更新一次」 —— 内置模块没有魔法，就是 §2 那个循环
3. 梯度回传是逐 Jacobian 连乘   —— 所以随长度指数衰减，§3 的数是可量的
4. 依赖超过约 20 步就别用普通 RNN —— §7 的实验：T=32 时它连训练集都拟合不上
5. 门控改的是「连乘什么」       —— 门而不是权重矩阵，所以衰减慢好几个数量级
6. 截断 BPTT 是明确放弃长依赖   —— 剪断处梯度为 0，不报错；k 是超参
7. out[-1] 对应 hn[-1]，不是 hn[0] —— 层数大于 1 时写错不会报错
8. 判断「学没学会」要有常数基线   —— 本篇的基线是「永远输出 0」，MSE ≈ 1/3
```

第 4 条和第 8 条是这一篇最实用的两条。**「命中率 0.199」这个数单独看毫无意义**——
它看着不像 0，很像「学了一点」。是把它和「常数基线 1/3、命中率 0」放在一起，
才知道那其实是**完全没学到**。

这和生成模型那几篇是同一类纪律：
[WGAN-GP](wgan_gp.md) 里「W 为负时不要相信它的数值」、
[VAE](vae.md) 里「KL 为 0 时采样图看着不错但其实塌了」——**先想清楚哪个数能证明它
学会了，再去看那个数。**

## 接下来

* [从零实现 Transformer](transformer.md)：用注意力一次性看到整条序列，绕开逐 Jacobian 连乘。
* [自动求导](autograd_details.md)：`jt.grad` 的完整语义，以及它什么时候静默返回 0。
* [JIT、算子融合与异步执行](jit_and_async.md)：为什么循环里的每一步都值得融合。
* 回到 [教程总目录](README.md)。
