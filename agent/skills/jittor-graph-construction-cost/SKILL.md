---
name: jittor-graph-construction-cost
description: 量并降低「算子之上」的主机开销——Python 前端把一步图建出来要多少钱。用于 batch 1 decode、小算子、launch-bound 这类主机受限场景；也用于判断某个前端改动到底值不值。改 python/jittor/{_runtime/dispatch,_core/{var,function,module},nn/functional,ops/indexing} 之前读。
---

# 图构建的主机成本

`jittor-core-planning-cost` 量的是 `run_sync` 里的规划与发射，那是 C++ 侧。
**这份量的是它上面那一层**：从 `model(x)` 到「图已经建好」之间，Python 花了多少。
两者不能互相代替——一个主机受限的 decode 步长里，执行器只占小头。

## 0. 先确认自己在哪个区间

```
一步的节点数 × 单个元算子的构建时间   vs   一步的墙钟
```

实测（RTX 4090 / H20 都一样，与卡无关）：**一个裸的 binary 元算子 1.60 us**。
8 层 d=512 的 transformer，batch 1 seq 1，一步 354 个节点 = 0.57 ms 的地板，
而实际构建 2.99 ms。**差的 5 倍全在算子之上的 Python**，不在 kernel、不在执行器。

设备受限的区间（batch 8 seq 256）这一切都藏在异步队列后面，量不出来也不该动。
**先把两个区间分开，再决定要不要做。**

## 1. `auto_flush_ops`：不关掉，"只构图"的计时全是错的

`jt.flags.auto_flush_ops` 在 CUDA 上**默认 128**。每创建这么多算子，
`Executor::submit_pending` 会：

1. **遍历所有存活的 `VarHolder`**，挑出待执行的；
2. 把它们全部发射出去（不等设备）。

于是一段「只构图、不同步」的计时里同时混进了核函数发射、和一次
**O(存活 holder 数)** 的遍历。而典型的微基准正好把 N 个结果留在列表里：

```python
out = [fn() for _ in range(N)]      # holder 数随 N 线性增长
```

**代价随 N 变化，于是"每次调用多少微秒"这个数根本不是常数。**

同一批量测，同一棵树：

| 行 | auto_flush=128 | auto_flush=0 |
| --- | --- | --- |
| `nn.Linear(512→512)` on (1,1,512) | 34.7 us | 20.8 us |
| `nn.LayerNorm(512)` | 55.5 us | 39.2 us |
| 一个 transformer block | 644 us | 370 us |

**规矩**：构图基准一律 `jt.flags.auto_flush_ops = 0`；端到端基准一律用默认值，
并且说明用的是哪一个。两个数不能混在同一张表里。

## 2. 必须是 `use_cuda=1`，哪怕不碰设备

派发层按 **Var 的放置**选内核。`use_cuda=0` 时 `select_kernel("matmul", ...)`
返回 `None`，矩阵乘走通用的 broadcast+mul+sum 路径——**那是一条完全不同的
Python 路径**，量它得到的结论对 GPU 不成立（实测 Linear 26.3 us vs 34.7 us，
连方向都可能反）。

配套的坑：`bench_models.py` 这类基准脚本在 **import 时**就写 `jt.flags.use_cuda = 1`。
`jt.flags.use_cuda = 0` 写在 import 之前会被它覆盖回去。**把 flag 的赋值放在所有
import 之后，并且把实际值打印出来**：

```python
from bench_models import Tiny          # noqa: E402  —— 这一行会改 flag
jt.flags.use_cuda = int(os.environ.get("PROBE_USE_CUDA", "1"))
jt.flags.auto_flush_ops = int(os.environ.get("PROBE_AUTO_FLUSH", "0"))
print(f"[probe] use_cuda={jt.flags.use_cuda} auto_flush={jt.flags.auto_flush_ops}")
```

## 3. 基准表要带对照行

前端改动是一堆互不相关的小改，噪声和真实收益同量级。**表里必须有几行是
"这次不该动"的**，它们不动才说明别的行动了不是噪声：

| 行 | 改前 | 改后 | 倍数 |
| --- | --- | --- | --- |
| transpose 5d | 10.38 | 6.20 | 1.67 |
| getitem `t5[0]` | 4.20 | 3.24 | 1.30 |
| *reshape 5d（对照）* | 2.30 | 2.34 | 0.98 |
| *`x + x`（对照）* | 1.599 | 1.607 | 1.00 |
| *`x * 0.5`（对照）* | 5.358 | 5.360 | 1.00 |

同一份基线连跑两遍，微观行差 ≤0.5%、step/block 行差 ≤3%，这就是判据的分辨率。

## 4. 不用查的两件事

- **建节点的代价与待执行图的大小无关。** 把 K 个 binary 串成一条未执行的链，
  K 从 1 到 1024，每算子 1.57–1.63 us，释放 0.12 us/算子。所以"图大了变慢"
  这个假设不必再查。
- **numpy 标量 vs python 标量的分派往返不是瓶颈。** `np.float32(0.5) * x`
  比 `0.5 * x` 只贵 0.5 us（6.02 vs 5.49），而两者都比 `x + y` 贵 3.8 us——
  贵在节点数，不在 numpy 让路的那一步。

## 5. 钱在哪：前端开销的五种形状

一步里 9138 次 Python/C 调用建 354 个节点 = 每节点 26 次调用。按 `cProfile`
的 `print_callers` 归因（不是 `print_stats`——要的是调用点，不是函数），
反复出现的是这五种：

1. **每调用重做的注册表查询。** 每个矩阵乘要问两次 `get_library_ops("cublas")`
   （一次判断能否派发、一次真正调用），派发层每个候选还要再问；每次三层函数帧、
   一把 RLock、三次字典查找。**给已定型的答案加备忘，带 `enabled` 策略的名字
   除外**（它读环境变量，可能在两次调用之间翻转）。
2. **ABC 的 `isinstance`。** `numbers.Integral`、`collections.abc.Sequence`
   会进 `ABCMeta.__instancecheck__`，一次约 0.35 us。一次 5 维 transpose 要
   做 10 次。**先 `type(x) is int` 快判，判不了再问 ABC**——语义不变。
3. **函数体里的 `import`。** `import jittor as jt` 写在函数里是为了绕开循环
   依赖，但它是一条 IMPORT_NAME：一次 `x[i]` 原来要跑四条。**在模块级留一个
   `_jt = None` 惰性绑定。**
4. **每调用重建的常量。** `gelu` 每次构造三个 numpy 标量；`_jittor_dtype_name`
   被调用三次，其中两次作用在它自己返回的*字符串*上；`layer_norm` 把
   `normalized_shape` 为两个 relay 各重建一次。
5. **只为被丢掉而构造的算子对象。** `BinaryOp` 的广播分支为两个操作数各建一个
   `BroadcastToOp`（其中一个只是把输入原样转发），再建第二个 `BinaryOp` 并转发
   给它。一次 `x * 0.5` 要构造 5 个算子对象，只留下 3 个节点。

## 6. 判据：少建节点，不是把节点建得更便宜

拆一次标量乘法（`x * 0.5`，5.36 us）：

| 组成 | us |
| --- | --- |
| `jt.array(0.5)`（array 节点） | 2.24 |
| `jt.broadcast_var(s, x)`（broadcast 节点） | 1.38 |
| `jt.binary(x, bs, 'mul')`（binary 节点） | 1.41 |

三项之和 5.30，与整体 5.36 吻合。**每个节点约 1.6 us，array 因为多两次分配再贵
0.6 us。** 所以前端优化的上限就是"省掉多少个节点"：`nn.Linear` 上那对
reshape（3 维→2 维→3 维）是 5.8 us、一个 block 23 us，让 cublas 直接吃 rank>2
就全部消失；`Function` 的录带是每个输入/输出一个 tape 节点，一次 layer_norm 占
11.8 us，`jt.no_grad()` 把它整个省掉（19.9 → 8.1 us）。

把节点建得更便宜（省一次 `PyDict_New`、省一次分配）是 0.1 us 量级的事，
**做之前先确认没有一个"少建一个节点"的选项还没做。**

## 7. 对拍的坑：一侧为空的 diff 和"全是回归"长得一模一样

前端改动要证明的是「失败集合与分支起点逐 nodeid 相同」。跑两棵树、`comm -13`
出回归列表——但如果基线那一侧根本没跑起来（`nvcc` 写不进一个没人建过的
`TMPDIR`，收集阶段就退出了），`comm` 会把被测侧的全部失败当成回归列出来。

**摘要里必须同时打印两侧的 passed/failed 计数**，两侧都非零才看 diff：

```
=== baseline 48055a75 ===  22 failed, 300 passed, 81 subtests
=== modified ===           22 failed, 300 passed, 81 subtests
=== only in modified (regressions) ===
（空）
```

同样地：新建的 worktree 要自己 `mkdir -p "$TMPDIR" "$JITTOR_HOME" "$HOME"`，
`env-*.sh` 只是 export，不会建目录。
