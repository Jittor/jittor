# 数值契约

- 状态：已接受
- 上次复查：2026-09-11
- 基线：`2d716db31`
- 复查触发：本页任何一条对应的测试文件变红、或其断言被放宽时

这一页回答一个问题：**同一段代码在 CPU 和 CUDA 上、在不同的调度设置下，什么必须
一样，什么允许不一样。**

它写给已经跑通模型、开始怀疑"两个设备的结果对不上是不是我用错了"的人。每一条都是
三段式——**你会看到什么 / 为什么 / 怎么办**——后面跟着**钉住它的测试文件**。这些数字
全部来自实测，出处写在每条里；测试文件是可执行的那一份，本页变旧时以它为准。

按现象索引：

| 你看到的 | 看哪一条 |
| --- | --- |
| `1/0` 得到 `nan`、被完全 mask 的 attention 行没有变成 `nan` | {ref}`IEEE 特殊值算术 <numerics-ieee>` |
| 极小的正数在 CUDA 上变成 0、`log` 给 `-inf`、`count_nonzero` 少数一个 | {ref}`CUDA 的次正规数 <numerics-subnormal>` |
| `x.max()` 看不见张量里的 NaN；`maximum(-0.0, 0.0)` 的符号 | {ref}`maximum/minimum 与 max/min <numerics-minmax>` |
| float64 的 `log`/`sin` 在 CUDA 上精度只有 float32 | {ref}`float64 一元数学 <numerics-float64>` |
| 只改了 `auto_flush_ops`，loss 一位不差而梯度变了 | {ref}`卷积算法的自动选择 <numerics-autotune>` |
| `profile_scope` 报告 0 行；换个 flag 结果就动 | {ref}`auto_flush_ops 的后果 <numerics-flush>` |
| 一千六百万个数求和，CPU 比 NumPy 差很多 | {ref}`CPU 归约的分块累加 <numerics-reduce>` |

还没有成立的部分单列在{ref}`最后一节 <numerics-open>`——**那里的每一条都是当前会算错的**。

(numerics-ieee)=
## IEEE 特殊值算术

**你会看到什么。** 无穷和 NaN 参与的算术，在 CPU 与 CUDA 上都按 IEEE-754 给出
定义好的答案，与 NumPy 一致：

| 表达式 | 结果 |
| --- | --- |
| `inf - inf`、`inf / inf`、`0 * inf`、`inf + -inf` | `nan` |
| `1 / 0` | `inf` |
| `-1 / 0`、`-inf / 0` | `-inf` |
| `inf * 1` | `inf` |
| `nan + 1`、`nan * 0` | `nan` |

`x != x` 作为"这是不是 NaN"的判定也仍然成立。

**为什么值得单独说。** 2026-09-10 之前不是这样：CPU 的 JIT kernel 用 `-Ofast`
编译，它蕴含 `-ffast-math`、进而 `-ffinite-math-only`——一个"任何操作数都不会是
无穷或 NaN"的承诺。编译器按这个承诺做变换，而**真的是无穷**的操作数就走到变换
后的代码碰巧产生的那条路上：float32、长度 ≥ 4（向量化路径）时 `1 / 0` 得 `nan`、
`-inf / 0` 得 `nan`。单个元素是对的，所以标量试探看不见。

危险的不是明显坏掉的那种。一个被完全 mask 掉的 attention 行会减去它自己的 `-inf`
最大值；那里出现有限值，产生的是一个形状良好但错误的 softmax，而不是一个一眼可见
的 `nan`。CUDA 一侧当时是对的，于是两个设备对同一个表达式给出不同答案。

**怎么办。** 不用做什么：CPU kernel 现在编译在 `-O3`（提交 `1e50d76c5`，
`KI-BACKEND-005`）。唯一要注意的是**自己往 `cc_flags` / `kernel_flags` 里加
`-Ofast` 会把这件事撤销**，而失效是静默的。

关于代价：这次替换在逐元素链、`exp`/`sqrt` 链、`sum`、matmul 四个形状上**没有实测
到代价**（4M 逐元素链 0.000320 → 0.000324 s，`sum` 4M 0.000737 → 0.000719 s）。
有一个形状付了钱：float32 的 `max`/`min` 归约在 `-Ofast` 下被重结合并向量化，
28–31 GB/s，在 `-O3` 下是 14 GB/s。这一项记在 `KI-OPS-006` 里，没有被抹平。

**测试**：[`tests/ops/test_ieee_arithmetic.py`](https://github.com/Jittor/jittor/blob/master/tests/ops/test_ieee_arithmetic.py)——
十条 IEEE-754 有确切定义的表达式，两个设备，**长度取 8**（长度 1 时即使旗标是错的
也全过），外加 `x != x` 这个判定本身。
[`tests/structure/codegen/test_kernel_math_flags.py`](https://github.com/Jittor/jittor/blob/master/tests/structure/codegen/test_kernel_math_flags.py)
是它便宜的同伴：点名那个旗标，让重新引入时的失败说清改了什么，并且单独断言优化
等级仍然存在——否则"删掉旗标什么也不放"会以把事情变得更糟的方式满足前一条。

(numerics-subnormal)=
## CUDA 的次正规数被刷成零

这是一个**决定**，不是缺陷，但此前没有写在任何地方，于是每次设备对拍撞上它都读
起来像缺陷。

**你会看到什么。** float32 的最小正规数是 `1.18e-38`。比它更小的非零值（次正规
数）在 CPU 上按 IEEE-754 保留，在 CUDA 上**默认被刷成零**：

| 值 | CPU | CUDA 默认 | CUDA `strict` | NumPy |
| --- | --- | --- | --- | --- |
| `1e-45`（次正规） | 保留 | **0.0** | 保留 | 保留 |
| `1e-40`（次正规） | 保留 | **0.0** | 保留 | 保留 |
| `1e-30`（正规） | 保留 | 保留 | 保留 | 保留 |

**为什么。** 根因是 nvcc 的 `--use_fast_math`，它蕴含 `-ftz=true`。后果不止于值
本身：输入在进入函数**之前**就已经是零了，所以 `log(1e-45)` 在 CPU 上给正确的
`-103.28`，在 CUDA 默认档下给 `-inf`；`count_nonzero([1e-45])` 在 CPU 上是 1，在
CUDA 默认档下是 0。一个下溢到次正规区间的梯度在 CUDA 上恰好是零、在 CPU 上是一个
极小的非零，两个设备于是对同一个模型走上不同的更新路径——在有人把它们并排比较
之前，这件事是不可见的。

**怎么办。**

```python
jt.flags.cuda_kernel_math = "strict"     # 去掉 --use_fast_math
```

这个开关按进程生效（`src/runtime/jit_policy.cc`），切换时会先把待执行的图排空。
实测它**完全恢复**次正规数：上表 `strict` 一列是量出来的，不是推断的。

代价：2026-09-10 在 RTX 4090 上量过，两轮交替、取最小值：

| 算子（16M float32） | 默认 | strict |
| --- | --- | --- |
| `divide` | 218.6 µs | 218.8 µs |
| `sqrt` | 148.1 | 148.8 |
| `exp` | 150.0 | 150.2 |
| `log` | 148.1 | 148.2 |
| `mul-add`（对照，不受 fast-math 影响） | 218.6 | 218.6 |

**在这个规模下代价测不出来**：差异是 0.1–0.8%，和不受 fast-math 影响的对照组同
量级。原因是这些核受访存带宽限制，fast-math 省下的 ALU 周期本来就被访存掩盖了。

计算密集的情形**没有得出结论**：把 `exp/log/sqrt` 串成 8 层和 32 层的链之后，同一
个档位在两轮之间的差异（1.01e-3 与 4.16e-4，2.4 倍）**大于**两个档位之间的差异，
那是融合方式的波动而不是策略的效果。所以这里只能说**没测出来**，不能说"没有代价"。
要给出一个可靠的数字，需要一个融合形态固定的基准。默认值因此没有动：没有任何证据
说刷零是错的取舍。

**测试**：[`tests/backends/parity/test_subnormal_contract.py`](https://github.com/Jittor/jittor/blob/master/tests/backends/parity/test_subnormal_contract.py)
把上面那张表断言下来。它不是在说哪一种行为更对，而是在说**当前的行为是什么**：
默认档刷成零、`strict` 保留、CPU 始终保留。三条里任何一条变了，那个测试就红——包括
"有人把默认改成了 strict"这种好的改变，那时应该改的是这份文档和那个测试，而不是
让它们继续描述一个已经不成立的世界。它的牙齿在于两个档位**必须真的不同**：把
`strict` 用例指向默认档它就红。

(numerics-minmax)=
## maximum/minimum 与 max/min：NaN 与符号零

**你会看到什么。** 逐元素的 `maximum`/`minimum` 和 `max()`/`min()` 归约都按
**NumPy 的语义**处理 NaN 与符号零，两个设备一致：

- **NaN 会传播**。`x.max()` 在张量含 NaN 时返回 `nan`，无论 NaN 在哪个位置、
  哪个符号、张量多大。
- **符号零是顺序相关的，这是故意的**。NumPy 用 `>` 单独决定 `maximum`，所以
  `maximum(-0.0, 0.0)` 是 `+0.0`、`maximum(0.0, -0.0)` 是 `-0.0`，`minimum`
  镜像。Jittor 现在逐位复现这个。
- 整数不受影响。

**为什么值得单独说。** 2026-09-10 之前三方都不一致。`f = [nan, -inf, -0.0, 0.0, inf]`
对零、float32：CPU 给 `[nan, 0.0, -0.0, 0.0, inf]`，CUDA 给 `[0.0, 0.0, 0.0, 0.0, inf]`，
NumPy 给 `[nan, 0.0, 0.0, 0.0, inf]`。归约更糟且**两个设备靠一致地错而一致**：含
一个 NaN 的数组上 `jt.max`/`jt.min` 在 n = 5、4096、1048576 都返回 `1.0`，NumPy 返回
`nan`。`x.max()` 是判断张量有没有坏掉的常用手段，而它当时根本看不见 NaN。

根因是两种拼写，各自都"显然"，但都不是按 NaN 行为选的：`std::max(a, b)` 是
`a < b ? b : a`，对 NaN 的比较全为假，于是返回**写在前面**的那个操作数；归约折的是
`acc = max(acc, x)`，到来的 NaN 永远在第二位，所以被丢掉。CUDA 的 `::max` 落到
`fmaxf`，即 IEEE 的 `maxNum`，**故意**返回非 NaN 的那一个，于是两个位置都丢。

**怎么办。** 不用做什么，`KI-BACKEND-004`（提交 `0f037d790`）已修。顺带修好的还有
`KI-BACKEND-007`：`jt.std([nan, 1.0, 2.0])` 此前在 CPU 上是 `nan`、在 CUDA 上是
`0.0009999999310821295`，`norm` 是 `nan` 对 `1e-15`；它们是经过 `maximum`/`minimum`
这一行的复合归约，那一行修好之后两个设备都给 `nan`。

代价是实测的，并且**不是零**：CPU 的 `max`/`min` **归约**慢 1.9–2.0 倍
（1M：14.29 → 7.24 GB/s；16M：13.99 → 7.25 GB/s；`sum` 作为对照 1.01 倍）。逐元素的
`maximum`/`minimum` 与最接近的真实场景 softmax 都没有可测的代价。剩下的那一半原因
不是比较本身，是 reduce kernel 用运行期 stride 索引挡住了向量化——八路部分和在单位
stride 下与 `std::max` 持平。这一条记在 `KI-OPS-006`，尚未关闭。

**测试**：[`tests/ops/test_minmax_nan_propagation.py`](https://github.com/Jittor/jittor/blob/master/tests/ops/test_minmax_nan_propagation.py)，
22 个用例，含一个**设备一致性类**——同样的操作数在两个设备上跑，互相比较也与 NumPy
比较，所以将来的分歧不会因为"每个设备各自看起来都合理"而溜过去。在未修的树上 22 条
里有 16 条失败。

(numerics-float64)=
## float64 的一元数学在 CUDA 上不再被窄化

**你会看到什么。** float64 的 `log`、`sqrt`、`sin`、`tan`、`tanh`、`arctan`、
`arcsin`、`exp`、`erf` 等在 CUDA 上**以 float64 计算**，与 CPU 和 NumPy 逐位相同。

**为什么值得单独说。** 2026-09-11 之前不是。CUDA 的表达式表里有十九项把函数名写成
**只吃 float 的** C 变体——`::logf`、`::expf`、`::sinf`、`::tanhf`、`::erff` 等
——不管操作数是什么 dtype。float64 的操作数于是被窄化成 float32、在单精度下求值、
再放大回去。大多数输入盖得住它；它在答案本身低于 float32 分辨率的地方露出来：

```
log(1 + 2**-51)   CPU 4.4408920985006252e-16   CUDA 0.0   NumPy 4.4408920985006252e-16
```

`1 + 2**-51` 在 float32 里**正好是 1.0**，`log(1.0)` 是零。不是"略有偏差"，是没了。

**怎么办。** 不用做什么，`KI-OPS-007`（提交 `2d716db31`）已修：同一套 dtype 分派
加到那十九项上。float32 的操作数模板发射的文本与之前**逐字相同**，所以单精度路径
按构造不变——实测 float32 的 `log`/`exp`/`sqrt`/`sin`/`tanh` 在 100 万元素上与
NumPy 的差距不变（约 1e-7，那是 `--use_fast_math` 造成的，改前改后都在），每次调用
23–25 µs 也不变。

**测试**：[`tests/ops/test_float64_unary_math.py`](https://github.com/Jittor/jittor/blob/master/tests/ops/test_float64_unary_math.py)，
两个设备。输入是挑过的：float32 的答案要**定性地**错（等于零，或等于输入本身），
而不只是精度低一点——容差型的测试在旧构建上有好几项是能过的。它同时钉住"这些输入
在 float32 下真的会塌缩"和"float32 没有变"。

(numerics-autotune)=
## 卷积算法的自动选择会改变梯度

这一条和上面几条不同：它改变的不是某个算子的语义，而是**同一份代码在不同调度下会
训练出不同的权重**，而且**前向的 loss 一位不差**——任何盯着 loss 的检查都不会响。

**你会看到什么。** 实测，RTX 4090，五个 bottleneck 块、598 万个参数元素：

| | `auto_flush_ops=0` | `=64` | `=256` |
| --- | --- | --- | --- |
| loss | 23080.078125 | 23080.078125 | 23080.078125 |
| 梯度范数（算法缓存 100，默认） | 45839.379 | **45822.207** | 45839.379 |
| 梯度范数（算法缓存 100000） | 45839.379 | 45822.207 | **45822.207** |

相对差 `3.8e-4`，**确定性**（同一配置三次同一个数）——不是竞态，是真的在算不同的
东西。

**为什么。** 卷积算法不是按启发式选的，是**实测候选者再挑最快的那个**，结果按形状
缓存（`backends/cuda/kernels/cudnn/cudnn_conv_op.cc`；代码注释写明理由：启发式在
寻常 fp32 3×3 上会挑到慢 2–3 倍的算法）。于是选中哪个算法取决于两件与数学无关的事：

* **实测时显存里有什么**——`auto_flush_ops` 改变执行调度，也就改变驻留；
* **在此之前见过多少个形状**——缓存满了会静默退回启发式
  （`if (fwd_algo_cache.size()>=max_cache_size) benchmark = false;`，
  `max_cache_size` 默认 100），所以 `set_algorithm_cache_size` 也是一个会改变数值
  的旋钮。

**怎么办。** 要可复现的梯度，先关掉自动调优：

```python
import jittor as jt
jt.cudnn.set_benchmark(0)                    # 强制走确定性的启发式
# Torch 前端等价写法：
# torch.backends.cudnn.benchmark = False
```

实测这样之后梯度不再随 `auto_flush_ops` 变化（残差 `2e-6`，那是不同批边界带来的
重结合）。代价是卷积可能选到较慢的算法——上面那条代码注释说的正是这个，所以这是
**取舍而不是纯改进**，默认值没有动。

可对照的是 PyTorch 的 `torch.backends.cudnn.benchmark`：它有同样的性质，而且在文档
里写明了。这一节存在的理由就是把这里的也写下来。

**一条给做对照实验的人的纪律**：任何**跨越会改变显存驻留的 flag** 的数值对照，都
必须先 `set_benchmark(0)`。不这样做，`3.8e-4` 这个带宽会把更小的真实差异盖掉，也
会把自动调优的波动误判成被测改动的效果——`KI-EXEC-001` 的第一次修复就是这样被错误
地否决的。

**测试**：[`tests/backends/cuda/test_autotuning_isolation.py`](https://github.com/Jittor/jittor/blob/master/tests/backends/cuda/test_autotuning_isolation.py)
钉住的是**边界**而不是行为：关掉自动调优后，梯度**不得**随调度 flag 变化（这一条
有牙齿——如果将来是**别的**东西开始改变数值，它会红）；以及无论开关，loss 都必须
逐位相同。它**不**断言"开着调优时梯度会不同"——那取决于机器、形状和 cuDNN 版本，在
实测恰好稳定的硬件上断言它会造成假红。

尚未解释的残余：关掉调优后，四个 bottleneck 块在 `auto_flush_ops=32` 上仍给
33063.723，别处是 33063.688，相对 `1.1e-6`——比算法效应小三百倍，落在"累加顺序变了"
的量级。没有继续追，记下来是为了不把它当成零。这条是 `KI-EXEC-003`，未关闭。

(numerics-flush)=
## auto_flush_ops 的后果

`jt.flags.auto_flush_ops = N` 让待定图提前发射：每创建 `N` 个算子就把当前所有待定
工作交给执行器。它的作用和收益见[流水式惰性执行](pipelined-execution.md)；这里只讲
它在数值和观测上的后果，因为它读起来像一个纯调度旋钮，而它不是。

**它默认是 128，只对 CUDA 生效。**

三件已经发生过的事：

1. **它会改变梯度**——不是它自己算错，是它改变显存驻留，而 cuDNN 的算法自动选择
   依赖驻留。见{ref}`上一节 <numerics-autotune>`。这是**目前唯一仍然开着的那一件**。

2. **它曾经让 ResNet50 类的骨干网在 CUDA 上段错误**。五个 bottleneck 块在
   `auto_flush_ops` 1、16、32 和 **128（发布默认值）** 上确定性段错误，而 64 和 256
   碰巧不会——看起来像"超过某个图规模阈值"，其实阈值不是规模，是切分点第一次落在
   一条 tape 上。根因是 `Tapes` 是一个**纯控制算子**（没有 `run`、没有 `jit_run`，
   不读取输入的任何字节），而 `run_exec_plan` 按**计算算子**的规则对待它：迁移每个
   输入到设备、断言每个输入有存储。整张图在一个批次里时两者从不相遇。已修
   （`KI-EXEC-001`，提交 `47cedd14d`）：0、1、16、32、64、128、256、512 全部不崩，
   loss 逐位相同，关掉 cuDNN 自动调优后跨设置的梯度残差最大 `2e-6`。

3. **它曾经让 `jt.profile_scope` 返回一份空报告，而且不说话**。在 `with` 之前构造的
   图可能已经跑完了，于是超过阈值的图得到 0 行、结果却是对的。已修
   （`KI-EXEC-002`，提交 `3b4e646e5`）：`profile_scope` 现在在其作用域内把
   `auto_flush_ops` 置 0（除非调用方显式覆盖），64 和 32 片的用例从 **0 行变成 9 行**；
   而在作用域打开之前就跑掉的工作**无法找回**，那时它抛一个 `RuntimeWarning` 点名
   原因——"跑得很快"和"什么都没测到"此前是同一份输出。

**怎么办。**

- 要**可复现的训练数值**：`jt.cudnn.set_benchmark(0)`。单独设 `auto_flush_ops=0`
  也能让一次运行内部自洽，但那是在放弃流水化的收益，而且不阻止别的驻留变化。
- 要**剖析**：把图建在 `profile_scope` 里面。那本来也是该测的窗口。
- 要**对照两个 flag 的数值效果**：先关自动调优，见上一节。

**测试**：[`tests/backends/cuda/test_auto_flush_graph_split.py`](https://github.com/Jittor/jittor/blob/master/tests/backends/cuda/test_auto_flush_graph_split.py)。
每个设置跑在**自己的进程**里，因为失败形态是段错误——在进程内它结束的是整个会话而
不是一个用例，后面的测试会静默地不运行。它断言 loss 逐位相同**并且**比较梯度：一个
"止住崩溃但反向读了错字节"的修法能过"跑起来没有"的检查，在这里过不去。它关掉 cuDNN
自动调优，以免 `3.8e-4` 的带宽藏住更小的东西。

(numerics-reduce)=
## CPU 归约的分块累加

**你会看到什么。** CPU 上 float32 的 `sum`/`mean` 的相对误差**不随元素个数增长**，
并且比 NumPy 更小：

| n 个 `0.1` 相加 | CPU 改前 | CPU 现在 | CUDA | NumPy |
| --- | --- | --- | --- | --- |
| 65,536 | 6.17e-4 | **5.96e-7** | 1.49e-7 | 1.49e-7 |
| 1,048,576 | 9.86e-3 | **5.96e-7** | 1.49e-7 | 9.69e-7 |
| 16,777,216 | 1.53e-1 | **5.96e-7** | 4.47e-7 | 1.61e-5 |

**为什么值得单独说。** 改前是**单个串行累加器**，误差与元素个数成正比：1677 万个
`0.1` 相加偏差 **15%**；100 万个 `0.1` 的 `mean` 是 0.09975。这不需要特殊输入，
一个按 batch 求平均的 loss 就会撞上。

**怎么办。** 不用做什么。`KI-BACKEND-006`（提交 `fc2bfe46f`）把它改成"块内多路部分
和 + 块间成对折叠，折叠栈跨越整个归约循环嵌套"。这次修复**同时更准也更快**：同一个
核 64M float32 从 4.90 到 **18.38 GB/s**（3.7–4.9 倍），从"比 NumPy 慢 4.4 倍"变成
"与 NumPy 同级"。唯一实测到的代价是大体积 fused 归约核的 JIT **编译**时间 +18%
（每个核形状一次，落盘缓存）。CUDA 一侧逐位不变——这个 pass 对加速器直接返回。

**已知边界，请照着读**：沿**最外维**归约的形状不在改写范围内。`(262144, 64)` 沿
dim 0：改前改后同为 2.47e-3。这不是退化——**NumPy 在同一形状上给出一模一样的
2.47e-3**，它也是每列一条串行链——而是落在"被归约的维度是最内层"这一前提之外。

`max`/`min` 归约也不在范围内：它们的折叠严格可结合，重排没有精度收益。它们的问题
是别的，见{ref}`上面那节 <numerics-minmax>`和下面的开放项。

**测试**：[`tests/ops/test_reduce_accuracy.py`](https://github.com/Jittor/jittor/blob/master/tests/ops/test_reduce_accuracy.py)，
9 个用例，CPU 与 CUDA 两套。断言的是**增长的形状**而不是阈值：16M 上的相对误差要在
64K 上的 10 倍以内（串行累加器在这个 256 倍区间上给出 248 倍，分块给出 1 倍），
并且在同一输入上 NumPy 误差的 10 倍以内；还有一个三维嵌套形状，专门抓"只改了最内层
循环"的修法。固定阈值会随机器和编译器变红变绿而与缺陷无关，所以一个都没有用。

(numerics-open)=
## 还没有成立的部分

以下每一条**当前都会算错**，都已复现、都还没修。对拍时撞上它们不是你的错。

| 现象 | 范围 | 条目 |
| --- | --- | --- |
| 全是 `-inf` 的 float32 张量，`jt.max` 返回 `-3.4028235e38` 而不是 `-inf`（`jt.min` 镜像）。一个被完全 mask 的 attention 行正是这个输入 | **CPU**；CUDA 正确 | `KI-OPS-008` |
| `digamma(nan)` 返回 `-inf` 而不是 `nan`；`digamma(-0.0)` 返回 `-inf` 而不是 `inf` | **CPU**；CUDA 正确 | `KI-OPS-011` |
| float32 的 `max`/`min` 归约比修 NaN 之前慢约 2 倍（只是吞吐，答案是对的） | CPU | `KI-OPS-006` |
| 关掉 cuDNN 自动调优之后，梯度仍有 `1.1e-6` 的残余随 `auto_flush_ops` 变动 | CUDA | `KI-EXEC-003` |

`KI-OPS-008` 的绕行：在 CPU 上，把等于 `numpy.finfo(dtype).min` 的 `max()` 结果
（或 `min()` 的 `.max`）当成"可能其实是无穷"来处理，或者把这个归约放到 CUDA 上。

完整的条目——严重度、证据、owner 和退出条件——在仓库的
[`agent/manuals/known-issues.md`](https://github.com/Jittor/jittor/blob/master/agent/manuals/known-issues.md)。

## 相关

- [float32 累加精度](float32-precision-policy.md)：TF32 / bf16 档位，以及惰性图下
  这些设置什么时候被读取。
- [流水式惰性执行](pipelined-execution.md)：`auto_flush_ops` 是干什么用的。
- [设备与放置](device-placement.md)：先确认两边真的在你以为的设备上，再对拍。
