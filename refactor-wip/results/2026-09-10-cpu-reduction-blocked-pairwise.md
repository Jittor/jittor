# CPU float32 归约改为分块 + 成对折叠（KI-BACKEND-006）

- Status: 已修复并验证
- Date: 2026-09-10
- Baseline: `860863960`（origin/2.0-refactor）
- Owner: CPU 后端与归约维护者
- Review when: `BlockedReductionPass` 的适用条件、`WAYS`/`PER_WAY`/`BIG_BODY`
  常量，或 `ReduceAccumulatorPass` 产出的 `kir::reduce_acc` 契约发生变化

## 结论

CPU 上 float32 的 `sum`/`mean` 用**单个串行累加器**求和，误差与元素个数成正比：
16,777,216 个 `0.1` 相加偏差 **15%**。改成「块内多路部分和 + 块间成对折叠、
折叠栈跨越整个归约循环嵌套」之后：

- 16.7M 上最坏情况相对误差 **1.5e-1 → 6.0e-7**，且在 65,536 → 16.7M 的 256 倍
  区间内**不再增长**（NumPy 成对求和在同一区间是 1.5e-7 → 1.6e-5）；
- 同一个核的吞吐 **4.9 → 18.4 GB/s（64M）**，即快 3.7-4.9 倍；
- CUDA 一侧逐位不变。

**修复没有代价**：它同时更准、更快。唯一实测到的代价是大体积 fused 归约核的
JIT 编译时间 +18%（见「代价」一节），这是发射代码变多的直接后果。

## 复现（修复前，`860863960`）

`n` 个 float32 `0.1` 相加，相对于精确值的相对误差：

| n | Jittor CPU | Jittor CUDA | NumPy |
| --- | --- | --- | --- |
| 65,536 | 6.17e-4 | 1.49e-7 | 1.49e-7 |
| 1,048,576 | 9.86e-3 | 1.49e-7 | 9.69e-7 |
| 16,777,216 | **1.53e-1** | 4.47e-7 | 1.61e-5 |

`mean` 同源：1M 个 `0.1` 的均值在 CPU 上是 0.09975（2.47e-3 相对误差）。

16.7M 随机 float32（67 MB），同机同轮：

| | 吞吐 | 相对误差 |
| --- | --- | --- |
| Jittor CPU `sum` | 4.45 GB/s | 7.74e-5 |
| NumPy `sum`（成对） | 21.2 GB/s | 7.99e-7 |

三张表都与 KI-BACKEND-006 条目记录的数字一致，起点确认。

## 为什么 `-Ofast` 没有救它

`-Ofast` 蕴含 `-ffast-math`，编译器**被允许**重结合归约、拆成多个累加器并向量化。
它没有这么做：用生成出来的核加 `-fopt-info-vec-all` 重新编译，g++ 12.3 报

```
src/ops/reduce_op.cc:396:39: missed: couldn't vectorize loop
src/ops/reduce_op.cc:399:77: missed: not vectorized: no vectype for stmt: _40 = *_42;
```

把该核里的 `op0_xstride0 = op0_x->storage_stride(0)` 换成字面量 `1` 再编译，同一
循环立刻 `optimized: loop vectorized using 32 byte vectors`。**运行期 stride 是
阻碍向量化的原因**；1.5e-1 的偏差确实来自一条完整的串行依赖链，不是 fast-math
重结合后的残留。

这一点必须单独验证，否则容易被自己骗：把同样的循环抄成一个独立的小 `.cc`
（stride 作为参数），g++ 会给它做 loop versioning 并向量化，误差降到 1.71e-2、
吞吐 27 GB/s——那是编译器的功劳，与本次改动无关。**改动的收益必须在真实核里测**，
下面所有数字都是。

## 改了什么

新增 `BlockedReductionPass`（`src/codegen/opt/pass/blocked_reduction_pass.{h,cc}`），
在 pass 流水线最后一步运行；`ReduceAccumulatorPass` 增加一个标记告诉它哪些循环
可以改写。

1. **`ReduceAccumulatorPass`** 仍照旧把 `yp[yid]` 提升成寄存器累加器（`CpuParallelPass`
   依赖这一步）。新增的是：当该循环里**每一条**累加语句都属于浮点**加法类**归约
   （`ns_add` / `ns_mean`，输出 dtype 为非复数浮点）时，在循环节点上写下
   `kir::reduce_acc`，值是累加器变量名的逗号分隔列表。归属通过 `op<N>_` 前缀
   映射回 `FusedOp::ops[N]` 判断，取不到就不标记。
   `maximum`/`minimum`/位运算折叠本身严格可结合，重排没有收益；`multiply` 的重排
   不是任何人要求的；整数求和与顺序无关。这些一律不动。

2. **`BlockedReductionPass`** 找到带该标记的最内层循环，向外爬过其余**被归约的
   维度**（判据：该维度的下标不出现在累加器的声明和回写语句里，且该层循环体只有
   下一层循环），然后把整个嵌套替换成一段不透明的原文：

   ```
   {
     <累加器声明>                       // 原来的 loop->before
     decltype(acc) stack[64];  int top = 0;  long long blocks = 0;
     decltype(acc) a_0 .. a_7 = decltype(acc)(0);
     for (<外层被归约维度…>) {
       if (range >= 16) {
         for (base = 0; base < range; base += 512) {
           a_* = 0;
           for (; i + 8 <= end; i += 8) { <循环体 ×8，各自写 a_0..a_7> }
           for (; i < end; i += 1)       { <循环体 ×1，写 a_0> }
           <平衡树折叠 a_0..a_7 → a_0>
           <按 blocks 的低位把 a_0 压栈，遇到等权条目就合并>
         }
       } else if (range > 0) {
         a_0 = 0; for (i = 0; i < range; i += 1) { <循环体 ×1> } <压栈>
       }
     }
     while (top > 0) { top -= 1; acc = stack[top] + acc; }
     <回写>                              // 原来的 loop->after
   }
   ```

   三个设计点，每个都对应一次实测到的错误：

   - **部分和的初值是加法单位元，不是累加器。** 第一版拿 `acc` 去初始化八路部分和，
     在多维归约里 `acc` 已经装着前面所有外层迭代的小计，于是每块把它多加八次——
     `(4096,4096).sum()` 直接返回 `-inf`。累加器只在最后的 drain 里被读一次。
   - **折叠栈跨越整个嵌套，而不是每层一个。** 只对最内层分块时，
     `(32768,512).sum()` 的误差是 2.58e-4、`(32,224,224).sum()` 是 3.18e-3——外层
     仍然是串行链，只是被内层长度除了一下。爬到最外层被归约维度之后两者都回到
     1.9e-7 ～ 6.0e-7。一个按 batch 求平均的 loss 正是嵌套而不是平铺循环。
   - **发射的文本自带一对花括号。** `KernelIR::to_string` 给带 before/after 的
     *循环*节点补花括号，给普通语句节点不补；不自己补的话累加器声明会泄漏到外层
     作用域，撞上生成器在同一作用域里重复写的 `auto op0_yid`，编译报
     `conflicting declaration`。

3. **发射规模随循环体大小调整**：循环体超过 400 字符（fused 归约，例如 softmax
   backward）时改用 2 路部分和、块长 128；块内链长 `PER_WAY = 64` 不变，所以精度
   不变，变的只有指令级并行度和发射的代码量。理由与代价见下节。

pass 放在流水线最后（`SolveConflictDefinePass` 之后、`FakeMainPass` 之前）：
`CpuParallelPass`、`InsertProfileLoopPass`、`SolveConflictDefinePass` 看到的仍然是
它们各自被写来处理的那种普通单累加器循环，改写之后留下的是一个不透明文本节点，
下游没有任何 pass 需要理解这个形状。`check_cache` 模式和加速器后端直接跳过。

新增属性 `kir::reduce_acc` 已按 `tests/structure/codegen/test_pass_attr_contracts.py`
的要求在两个 pass 的 `reads`/`writes` 里声明。

## 修复后：精度

`n` 个 `0.1` 相加（最坏情况，误差全部同号）：

| n | CPU 改前 | CPU 改后 | CUDA 改前 | CUDA 改后 | NumPy |
| --- | --- | --- | --- | --- | --- |
| 65,536 | 6.17e-4 | **5.96e-7** | 1.49e-7 | 1.49e-7 | 1.49e-7 |
| 1,048,576 | 9.86e-3 | **5.96e-7** | 1.49e-7 | 1.49e-7 | 9.69e-7 |
| 16,777,216 | 1.53e-1 | **5.96e-7** | 4.47e-7 | 4.47e-7 | 1.61e-5 |

`mean` 三个规模全部 1.49e-7（改前 1.54e-4 / 2.47e-3 / 3.98e-2）。
16.7M 随机输入相对误差 7.74e-5 → **2.86e-8**（NumPy 7.99e-7）。

同一个 16.7M 元素换成不同形状，仍是最坏情况输入：

| 形状 | n | Jittor CPU 改前 | 改后 | NumPy |
| --- | --- | --- | --- | --- |
| `(16777216,)` | 16.7M | 1.53e-1 | **5.96e-7** | 1.61e-5 |
| `(4096, 4096)` | 16.7M | 1.53e-1 | **5.96e-7** | 1.61e-5 |
| `(512, 32768)` | 16.7M | 1.53e-1 | **5.96e-7** | 1.61e-5 |
| `(32768, 512)` | 16.7M | 1.53e-1 | **5.96e-7** | 1.61e-5 |
| `(8, 2097152)` | 16.7M | 1.53e-1 | **5.96e-7** | 1.61e-5 |
| `(64, 262144) dims=[1]` | 262K | 2.58e-4 | **5.96e-7** | 2.98e-7 |
| `(32, 224, 224)` | 1.6M | 3.18e-3 | **1.9e-7** | 1.95e-6 |
| `(32,64,224,224) dims=[0,2,3]`（BN 统计量） | 1.6M | 1.10e-5 | **1.9e-7** | 1.95e-6 |
| `(262144, 64) dims=[0]` | 262K | 2.47e-3 | 2.47e-3 | 2.47e-3 |

**已知边界**：最后一行 `(262144, 64) dims=[0]` 沿最外维归约，最内层循环跑的是
输出维度，循环里没有可提升成寄存器的累加器，`ReduceAccumulatorPass` 不标记、
本 pass 也就不介入，改前改后同为 2.47e-3。**NumPy 在同一形状上给出一模一样的
2.47e-3**——它也是每列一条串行链——所以这不是相对 NumPy 的退化，而是落在
「被归约维度是最内层」这一前提之外的形状。条目的退出条件是「与 NumPy 同一数量级」，
这里是相等；要更好需要另一种改写（把输出维度分块驻留寄存器），不在本条范围内。

## 修复后：吞吐

`benchmarks/reductions.py`（`ReductionBenchmarks.track_bytes_per_second`，jittor 后端，
随机 float32；该 harness 自带「十次归约要花约十倍时间」的线性自检，因此报出来的
不是被 lazy 图吞掉的次数）：

| CPU | 改前 GB/s | 改后 GB/s | 比值 |
| --- | --- | --- | --- |
| `sum` 1M | 4.80 | **23.40** | 4.9x |
| `sum` 16M | 4.87 | **18.14** | 3.7x |
| `sum` 64M | 4.90 | **18.38** | 3.7x |
| `max` 1M | 14.33 | 14.29 | 1.00x |
| `max` 16M | 10.42 | 13.91 | （改前该点偏低，属噪声） |
| `max` 64M | 13.90 | 13.74 | 0.99x |
| `min` 1M | 14.25 | 14.23 | 1.00x |
| `min` 16M | 13.52 | 13.87 | 1.03x |
| `min` 64M | 13.91 | 12.85 | 0.92x |

`max`/`min` 不在改写范围内，它们的浮动是同一台机器上的运行间噪声，用来判断
`sum` 的 3.7-4.9 倍不是机器整体变快了。

同机 NumPy 单线程 `sum` 在 16.7M 上是 20.5-22.5 GB/s。改后 CPU `sum` 在 16M 是
NumPy 的 0.85 倍、在 1M 是 1.1 倍——**从「比 NumPy 慢 4.4 倍」变成「与 NumPy 同级」**，
同时误差比 NumPy 小 27 倍。剩下的差距来自上面那条：核里的地址仍然用运行期 stride
计算，g++ 不向量化，八路部分和拿到的是标量指令级并行而不是 SIMD。把 stride 变成
编译期常量是另一件事，不在本条范围内。

CUDA 未受影响（本 pass 对加速器直接返回）：

| CUDA | 改前 | 改后 |
| --- | --- | --- |
| `sum` 16M | 688.9 / 696.4 GB/s | 697.9 / 684.7 GB/s |
| `sum` 64M | 742.4 / 735.7 GB/s | 745.0 / 737.0 GB/s |

（每列两次独立测量。1M 规模上 `max`/`min` 在同一份构建的两次测量之间就能从
105 波动到 144 GB/s，那是启动延迟，不作结论。）

## 代价：JIT 编译时间

发射的代码是原循环体的 (路数 + 1) 份，这是唯一实测到的代价。冷编译单个核，
三次取最小：

| 核 | 改前 | 八路固定 | 按体积调整（最终） |
| --- | --- | --- | --- |
| `sum(4096x4096)` 全归约 | 912 ms | 997 ms | 977 ms |
| `sum(256x768) dim1` | 923 ms | 1020 ms | 1061 ms |
| `mean(256x768) dim1` | 913 ms | 1062 ms | 1097 ms |
| `max(256x768) dim1`（未改写） | 889 ms | 912 ms | 892 ms |
| fused softmax backward | 1011 ms | **1495 ms (+48%)** | **1198 ms (+18%)** |

八路固定时，大体积 fused 核的编译时间涨了近一半。按循环体大小把路数降到 2
（块长同比降到 128，块内链长不变、精度不变）把它压回 +18%，小核仍走八路。
这是每个核形状一次、落盘缓存的成本。

## 回归测试

`tests/ops/test_reduce_accuracy.py`，9 个用例，CPU 与 CUDA 两套。

断言的是**增长的形状**而不是阈值：

- `rel_err(16M) <= 10 × max(rel_err(64K), float32 eps)`——串行累加器在这个 256 倍
  区间上给出 248 倍，分块给出 1 倍；
- `rel_err(16M) <= 10 × max(NumPy 在同一输入上的 rel_err, eps)`；
- 一个三维嵌套形状 `(n>>12, 64, 64)` 的同样断言，专门抓「只改了最内层循环」的修法；
- CPU 与 CUDA 在 16.7M 上必须一致到 `100 × eps`——即条目要求的、大到足以暴露该
  缺陷的 parity 用例。

固定阈值会随机器和编译器变红变绿而与缺陷无关，所以一个都没有用。
构造数据后按 KI-DTYPE-002 断言拿到的 dtype 确实是 float32。

**把修复回退后测试确实变红**（在 `860863960` 原始源码上单独运行该文件）：

```
5 failed, 4 passed
FAILED ...TestReductionErrorDoesNotGrowCpu::test_mean_error_does_not_grow_with_size
   0.0398 not <= 0.00154 : mean relative error grows with size:
   ['0.000154', '0.00247', '0.0398'] over n=[65536, 1048576, 16777216]
FAILED ...TestReductionErrorDoesNotGrowCpu::test_nested_sum_error_does_not_grow_with_size
   0.153 not <= 0.00617 : nested sum ... ['0.000617', '0.00986', '0.153']
FAILED ...TestReductionErrorDoesNotGrowCpu::test_sum_error_does_not_grow_with_size
FAILED ...TestReductionErrorDoesNotGrowCpu::test_sum_error_is_near_numpy_at_sixteen_million
   0.153 not <= 0.000161 (NumPy 1.61e-05 on the same input)
FAILED ...TestLargeReductionAgreesAcrossDevices::test_sixteen_million_element_sum_agrees
   CPU and CUDA disagree by 0.153: 1935089.0 against 1677720.875
```

四个通过的正是 CUDA 那四个——CUDA 本来就是对的，这也说明这些断言没有把两侧
一起放过。修复后同一文件 **9 passed**（CPU 与 CUDA 各自单独确认）。

## 其它验证

- **形状/dtype 扫描**（自建，未版本化）：1D/2D/3D/4D、全归约与按维归约、
  `sum`/`mean`/`max`/`min`/`prod`/`any`/`all`、float64/float16/int32、
  fused `(x*y).sum()` / `(x*x).mean()` / 范数、`n = 0,1,511,512,513,1023,1024,4095,4096`
  边界、以及 `jt.grad`。误差按 `sum(|x|)` 归一（对求和误差而言这才是条件数；
  按结果归一会在结果接近零时报出与核无关的巨大相对误差）。CPU 与 CUDA 各 0 失败。
- **层级 spot check**：BatchNorm（训练输出与 running_mean）、LayerNorm、softmax、
  Frobenius 范数、`grad(mean(x*x))`，CPU 0 失败，最大相对误差 8.4e-7。
- **`tools/semantic_divergence_probe.py`**：CPU 侧 `stability` 类的
  `mean of a constant does not drift` 由 MISMATCH（0.09975 vs 0.1）变为 OK，
  MISMATCH 总数 5 → 4，没有新增项；CUDA 侧不变。

## 基线对照

`tests/ops` + `tests/opinfo`（native 模式，`-p no:randomly`，同一命令、
各自独立的 `JITTOR_HOME`）：

| | 改前 | 改后 |
| --- | --- | --- |
| 结果 | 249 failed, 328 passed, 81 skipped, 6 xfailed, 29 errors | 258 failed, 328 passed, 81 skipped, 6 xfailed, 29 errors |

逐条 nodeid 对照：**改前的 249 条一条不少、一条不多地出现在改后**；
新增的 9 条**全部**来自本次新加的 `tests/ops/test_reduce_accuracy.py`。

这 9 条需要解释，因为它们单独跑是 9 passed。原因与本改动无关：
`tests/ops/test_numpy_code_op.py`（改前改后同样失败）在会话中把 CUDA context 打成
`cudaErrorIllegalAddress`，此后同一进程里的每一个测试都以 CUDA 错误失败——
基线那 249 条里的绝大多数也是这个级联。取该文件在收集顺序中的前三个邻居一起跑即可
复现：

```
FAILED tests/ops/test_numpy_code_op.py::TestCodeOp::test - RuntimeError: cudaErrorIllegalAddress
...
tests/ops/test_reduce_accuracy.py:80: RuntimeError: CUDA error ... cudaMemGetInfo
19 failed, 1 passed
```

`tests/structure`（shim 模式，该目录要求 `JITTOR_TORCH_SHIM=1`）：改前改后同为
**72 failed, 1256 passed, 2 skipped**，抽样对照的失败 nodeid 全部落在改前集合内；
其中 `tests/structure/codegen` 26 项全过（pass 属性契约、`kir` 名称声明、流水线
顺序都在这里）。`bash tools/check_repo_layout.sh` 通过。

## 未版本化产物

复现脚本、扫描脚本、benchmark 驱动、编译时测量、两侧完整测试日志与 probe JSON 位于
`$JITTOR_LAB_ROOT/pairwise-sum/`，隔离运行状态位于
`$JITTOR_LAB_ROOT/_state/pairwise-sum/`。本报告的结论不依赖这些文件仍然存在。

## 环境

单机，8 张 NVIDIA GeForce RTX 4090，CUDA 12.2.140，g++ 12.3.0，Python 3.11.15，
NumPy 单线程。CUDA 测量固定使用 7 号卡（worktree 侧用 6 号）。
