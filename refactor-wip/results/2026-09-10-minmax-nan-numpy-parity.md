# `maximum`/`minimum` 与 `max`/`min` 对齐 NumPy 的 NaN 与符号零语义（KI-BACKEND-004）

- Status: 已修复并验证；遗留一项 CPU 归约吞吐代价，见 KI-OPS-006
- Date: 2026-09-10
- Baseline: `cb0b5890d`，外加 KI-BACKEND-005 的 `-Ofast` → `-O3`（前置条件）
- Owner: 算子与 CUDA 后端维护者
- Review when: `common_op_type.cc` 的 `maximum`/`minimum` 行、
  `type/minmax_compute.h`、两个 pass 里匹配这些拼写的模式，或
  `cuda_atomic_max`/`min` 的有序整数编码发生变化

## 结论

`maximum`/`minimum`/`jt.max`/`jt.min` 现在在 **CPU 与 CUDA 上都逐位符合 NumPy**：
NaN 从任一操作数传出、在归约里既能进入累加器也能留住，零的符号按 NumPy 的
`>`/`<` 规则决定，整数行为逐位不变。

代价只有一处，实测而非推断：**CPU 上 float32 `max`/`min` 归约慢 1.9-2.0 倍**
（14.3 → 7.25 GB/s）。逐元素算子、CUDA 全部路径、以及最接近真实场景的 softmax
都测不出退化。剩下的 2 倍不是比较本身，而是归约核用运行期 stride 索引导致整个
循环不向量化——同一表达式开 8 路部分和、stride 为 1 时可以打平 `std::max`。

## 复现（修复前）

`f = [nan, -inf, -0.0, 0.0, inf]` 对 zeros，float32：

| | 逐元素 `maximum(f, 0)` |
| --- | --- |
| CPU | `[nan, 0.0, -0.0, 0.0, inf]` |
| CUDA | `[0.0, 0.0, 0.0, 0.0, inf]` |
| NumPy | `[nan, 0.0, 0.0, 0.0, inf]` |

把操作数换个次序 `maximum(0, f)`，CPU 变成 `[0, 0, 0, 0, inf]`——**CPU 的“传播
NaN”是位置巧合**：`std::max(a,b)` 是 `a<b?b:a`，与 NaN 的比较全为假，返回的是
写在前面的那个操作数。

归约在两个设备上一致地错。一个 NaN 混在 1.0 里：

| n | `jt.max` CPU | `jt.max` CUDA | NumPy |
| --- | --- | --- | --- |
| 5 | 1.0 | 1.0 | nan |
| 4,096 | 1.0 | 1.0 | nan |
| 1,048,576 | 1.0 | 1.0 | nan |

`jt.min` 同。累加器 `tmp = max(tmp, b)` 里进来的 NaN 永远排第二，所以任何规模、
任何设备都吞掉。

## NumPy 的判据（实测，不是印象）

```
np.maximum(-0.0, 0.0) = 0.0   signbit=0      np.minimum(-0.0, 0.0) = 0.0   signbit=0
np.maximum(0.0, -0.0) = -0.0  signbit=1      np.minimum(0.0, -0.0) = -0.0  signbit=1
```

NumPy 只用 `>`（或 `<`）判断，因此**符号零的结果与操作数次序有关**。这不是需要
额外处理 signbit 的场景：`a > b ? a : b` 原样复现它。修复前 CPU 给
`maximum(-0.0, 0.0) = -0.0`、CUDA 两个次序都给 `+0.0`，两个都错。

## 修改

1. `src/type/minmax_compute.h`（新增）：
   `_max(a,b) = ((a > b) | (a != a)) ? a : b`，`_min` 对称。整数上 `a != a` 恒假，
   编译器删掉，剩下的就是原来的三目。
2. `src/type/common_op_type.cc`：CPU 表与 CUDA 表的 `maximum`/`minimum` 都改成
   `jittor::_max`/`jittor::_min`；`post_pass` 按 `_floor_mod`/`_signed_pow` 的写法
   插入头文件。
3. `parallel_pass.cc` / `atomic_tuner_pass.cc`：这两个 pass **按字面拼写**匹配
   max/min 来路由原子操作，不同步改就会丢原子路径。新增 `jittor::_max(...)` /
   `jittor::_min(...)` 两条模式；`std::max` / `::max` 两条保留，因为 float16 表
   仍然发这两种拼写。`expr::match` 吃得下限定调用——生成的 CUDA 核里
   `cuda_atomic_max` 仍在（见下），而且**匹配不上不会静默降级**，会走到
   `LOGf << "Expr not match"`。
4. `src/type/cpu_atomic.h`：`cpu_atomic_max`/`min` 自旋锁下改用 `_max`/`_min`。
5. `backends/cuda/kernels/core/cuda_atomic.h`：见下一节。

## CUDA 原子：先测再改，测出来必须改

只改表和 pass 之后，CUDA 归约仍然错，而且错得有规律：

| | `+nan` | `-nan` |
| --- | --- | --- |
| `jt.max` | nan ✓ | **1.0** |
| `jt.min` | **1.0** | nan ✓ |

`cuda_atomic_max/min(float*)` 是在有序整数编码上做 `atomicMax`。
`floatToOrderedInt` 把**正 NaN 映射到 `+inf` 之上**、**负 NaN 映射到 `-inf`
之下**，所以每个操作各自只带得动一个符号的 NaN。改法是把任何 NaN 编成该操作
的极值键（max 用 `0x7FFFFFFF`，min 用 `0x80000000`，两者解码回来都是 NaN），
于是任何 NaN 在两个方向上都压过所有实数，已经在累加器里的 NaN 也不会被挤掉。
解码不变，`fix_float` 与 `float_atomic_fix_pass` 不受影响。

同时改了 `shared_reduce_max/min`（SharedReducePass 用它做块内折叠）和
setitem 用的 `cuda_atomic_max_rmw/min_rmw`——最后一个是因为 CPU 的
scatter-maximum 走公共表，不改它就会新造一个 CPU/CUDA 分歧。

## 验证

`tests/ops/test_minmax_nan_propagation.py` 重写，22 条：逐元素覆盖全部浮点类别
与两种操作数次序（float32/float64）、符号零两种次序、`n` = 5/4096/1048576 两种
NaN 符号的 `max`/`min`、64×4096 沿维归约（并行/原子路径）、float64 归约、
int8/int16/int32/int64/uint8 逐位不变，外加条目要求的 **device-parity 类**：同一
批操作数在两个设备上跑，互相对比也和 NumPy 对比。

- 未修复的源码上：**16 failed, 6 passed**（回退确认变红）。
- 修复后：22 passed，两个设备。
- `tests/ops/test_ieee_arithmetic.py` 6 passed（前置条件：`x != x` 对 NaN 返回 1.0）。
- `tests/ops/test_reduce_accuracy.py` 全绿，`sum`/`mean` 的精度契约未受影响。

生成核对照（同一 op key，修复前后）：

```
before  cpu   ... OP_maximum ... : std::max x1
after   cpu   ... OP_maximum ... : jittor::_max x1, minmax_compute.h x1
before  cuda  ... OP_maximum ... : cuda_atomic_max x1, ::max( x5
after   cuda  ... OP_maximum ... : cuda_atomic_max x1, jittor::_max x1, minmax_compute.h x1
```

`cuda_atomic_max` 仍在 —— 原子路径没丢。

## 性能（`benchmarks/reductions.py` + `benchmarks/operators.py`，前后交错 3 轮，取各自最好一轮）

CPU：

| | before | after | |
| --- | --- | --- | --- |
| `max` 1M | 14.29 GB/s | 7.26 GB/s | **1.97x 慢** |
| `max` 16M | 13.99 GB/s | 7.25 GB/s | **1.93x 慢** |
| `min` 1M | 14.33 GB/s | 7.26 GB/s | **1.97x 慢** |
| `min` 16M | 13.89 GB/s | 7.25 GB/s | **1.92x 慢** |
| `sum` 1M（对照） | 24.53 GB/s | 24.21 GB/s | 1.01x |
| `sum` 16M（对照） | 18.67 GB/s | 18.41 GB/s | 1.01x |
| 逐元素 `maximum` float32 16M | 1287 GB/s | 1352 GB/s | 无退化 |
| 逐元素 `maximum` int32 16M | 1426 GB/s | 1440 GB/s | 无退化 |
| 逐元素 `minimum` float32 16M | 1403 GB/s | 1253 GB/s | 噪声范围内 |
| softmax (16,128,1024) | 1.175 ms | 1.207 ms | +2.7% |
| layernorm（对照） | 1.690 ms | 1.757 ms | +4.0% |

softmax 的 `max(-1)` 正是这个归约；它动得比对照还小，所以真实场景测不出代价。
逐元素 1M 一档在两侧都在 205-435 GB/s 之间跳（内存池的冷热），跨度盖过差值，
所以以 16M 一档为准。

CUDA：全线打平。

| | before | after |
| --- | --- | --- |
| `max` 16M | 1819.6 GB/s | 1829.4 GB/s |
| `min` 16M | 1812.9 GB/s | 1846.9 GB/s |
| `sum` 16M（对照） | 727.9 GB/s | 723.9 GB/s |
| 逐元素 `maximum` float32 16M | 868.8 GB/s | 869.7 GB/s |
| softmax | 57 µs | 56 µs |
| layernorm（对照） | 63 µs | 64 µs |

## 那 2 倍是什么，不是什么

独立微基准，16.7M float32，单线程，`g++ 12.3 -O3 -march=native`：

| 写法 | stride 编译期为 1 | stride 运行期未知 |
| --- | --- | --- |
| `std::max(a,b)` | 13.9 GB/s | 13.7 GB/s |
| `((a>b) \| (a!=a)) ? a : b`（已采用） | 9.8 GB/s | 4.9 GB/s |
| `(a>b \|\| a!=a) ? a : b` | 6.9 GB/s | — |
| `if (a!=a) ...; if (b!=b) ...;`（初稿） | 2.4 GB/s | — |
| 上面已采用写法的 8 路部分和 | **13.7 GB/s** | 4.3 GB/s |

两点结论。**拼写值 4 倍**：`|` 换掉 `||` 去掉了向量化器过不去的短路分支，三段
`if` 是最差的一种——初稿正是三段 `if`，在树内实测 3.65 GB/s，换成 `|` 之后
7.25 GB/s。**剩下的 2 倍不是比较**：同一表达式 8 路部分和与 `std::max` 齐平，
但只在 stride 编译期已知时如此。

归约核索引写的是 `op0_xid = id0 * op0_xstride0`，`op0_xstride0` 是
`storage_stride(0)`，运行期值。g++ 对它**认识**的归约会按 stride 做 loop
versioning（这就是 `std::max` 在未知 stride 下还有 13.7 GB/s 的原因），对不认识
的不做。

**实测而非预测**：把 `BlockedReductionPass` 扩到 `maximum`/`minimum`（`zero`
取累加器自身、折叠改成 `_max`/`_min`）已经实现并测过，结果是 **更慢**，
7.25 → 3.58 GB/s——没有向量化可解锁，分块记账是纯开销。该改动**没有进树**。

## 与 KI-OPS-006 记的 7.1-7.5x 为什么不一致

两头都变了。旧数字测在 `1e25ff68a`，基线是 `-Ofast`，`std::max` 被重结合并
向量化到 28-31 GB/s；KI-BACKEND-005 去掉该 flag 后，同一表达式在 `-O3` 下是
14 GB/s。旧实现是指数位的位测试套在 `if` 链里，测得 3.9-4.9 GB/s；现在的比较是
7.2。分子变大、分母变小，2.0 倍而不是 7.4 倍。两个数字对各自的树都不算错。

## 未随之解决的

- **KI-OPS-008**（CPU 归约的有限单位元）不受影响，已复测：CPU `jt.max` 对全
  `-inf` 的 float32 仍返回 `-3.4028235e38`，CUDA 仍返回 `-inf`，
  `test_minmax_reduction_identity.py` 的 strict xfail 仍然 xfail。错的是
  `init_maximum`/`init_minimum` 行，不是旁边的 `maximum`/`minimum` 行。
- **float16/bfloat16 表**（`src/type/fp16_op_type.cc`）仍发 `::max` /
  `std::max<float>`，没有跟着改，因此 fp16 的 NaN 行为未变。本条目的判据不含
  fp16；要改需要先确认 `_max<__half>` 在 `__host__ __device__` 下的实例化。
- **KI-BACKEND-007**（CUDA `std`/`norm` 吞 NaN）是组合归约，不经过这一行，未受
  影响。

## 未版本化

原始日志、编译缓存与微基准源码在
`$JITTOR_LAB_ROOT/_state/minmax-cache/`，worktree 在
`$JITTOR_LAB_ROOT/_state/minmax-wt/`。
