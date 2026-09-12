# CUDA 元算子：发射配置、strided 下标与标量常量融合

- 状态：部分完成 —— 三处改动已实现并在 H20 上测得；第三处同时修掉一个静默算错；完整 CPU/CUDA 门禁未跑完
- 日期：2026-09-12
- 基线提交：`d40a2e97`
- 分支：`perf/metaop-speed`（前两项）、`perf/scalar-fusion`（第三项）
- Owner：Jittor 核心维护者
- 复查触发：新增 CUDA 架构、`broadcast_to` 的存储语义变化、或发射配置再次调整

## 问题

元算子（unary/binary/ternary/reduce/broadcast/reindex）在 CUDA 上离硬件上限还有多远，
差在哪里，以及缩小这个差距能给整网带来多少加速。

## 环境与测量口径

H20（sm_90，标称约 4 TB/s）、CUDA 12.9、g++ 12.3.1、Python 3.12。工作树与运行状态
在 `$JITTOR_LAB_ROOT` 下，未版本化。

**这台机器上有别的进程长期占满全部 8 张卡**，因此任何一次同步都要额外付约 2.3 ms
的固定停顿——真 PyTorch 在同一张卡上量到同样的停顿，所以这不是 Jittor 的问题，
但它会毁掉「一个算子对一次同步」的测法。口径因此定为：

- 每个用例发射 R 个**互不依赖**的同类算子，只同步一次，报**斜率** `(t(R2)-t(R1))/(R2-R1)`；
- R1 与 R2 在**同一轮内交替**测量并取配对差的**中位数**。分别取两个窗口的最小值会把
  安静的 R2 和繁忙的 R1 配到一起，实测出现过负斜率；
- 制造「互不依赖」**不能靠加一个算子**。最初用 `a+b+float(i)` 区分实例，profiler 显示
  那个标量自己编成了一个 `array` kernel，占掉一次运行 44% 的记录条目。改为预先分配
  R 份不同输入；
- 整网 A/B 在**同一进程内**切换 `cuda_thread_num` 并交替测量，避免重编译带来的位置偏差。

## 改动

### 1. `cuda_thread_num` 默认 1024 → 256

`ParallelPass` 把这个值同时写进 kernel 的 `__launch_bounds__` 和发射形状。1024 是
硬件允许的最大 block，既压占用率也压每线程寄存器上限。

十三种元算子形状各扫 1024/512/256/128/64：**256 在每一种上都是最优或与最优同等，
没有一种形状变慢**；512 次之；64 与 128 在按轴归约上变差。`cuda_block_num` 保持 2048
（更大的线程预算实测更差）。

机制上有一点要写清楚：**起初怀疑是 1024 的 `__launch_bounds__` 造成寄存器溢出，
ptxas 否掉了这个假设**——ternary 与五算子融合链在两个 bound 下寄存器数相同（28/32）、
零溢出。收益来自占用率与波次量化，以及线程预算从 2M 降到 512K 后每线程处理的元素
变多。两个因素都有贡献：固定预算 2M 时仅收窄 block 是 1.11x，再把预算降到 512K 是 1.33x。

`SharedReducePass` 自己写死的 1024 一并接到同一个取值函数（`cuda_block_width`）。
它只在 `para_opt_level >= 4` 生效（默认 3），但比 `__launch_bounds__` 宽的 block
根本发不出去，不能留着。

### 2. 逐元素元算子的 strided 下标不再逐轴做整数除法

生成的 kernel 原本把循环坐标压平成线性下标，再用每轴一次 `%` 和一次 `/` 解回来——
而坐标本来就在手里。把「哪些轴真的推动物理下标」按操作数编进 JIT key 之后：

- 掩码外的轴整项消掉；
- 某轴的除法只在还有更低的轴要读余数时才保留；
- 第 0 轴不需要取模（此时线性下标已小于该轴范围）。

于是行广播只剩一次取模、列广播只剩一次除法、秩 1 视图两者全无（原本各是两次）。
ternary 原本三个操作数共用一个 `STRIDED` 标志，一个 strided 的条件会把完整解码
强加给通常连续的两个值操作数，现在按操作数分开并共用一条余数链。

**要注意**：单独量这一项的收益不大。独立 CUDA 微基准里，把两次 div/mod 换成一次取模
只有 **1.13x**——这条 kernel 访存受限，整数除法大半被掩盖。它真正的价值是和第 1 项
叠加后广播用例的稳定性（见下）。

### 3. 一元素 expand 保留可融合形式（标量常量不再单独成核）

`broadcast_to` 在本分支的存储边界重构（`7e83d6da`）里变成了**存储描述符**：输出与输入
共享存储、广播轴步长为 0、`set_type(OpType::other)`。对真实广播这是对的，不再物化。

但它同时切断了标量常量的融合。`x * 0.5` 的图是 `array(1元素) → broadcast_to → binary`，
而 `count_fuse` 的 `edge_fusable` **在看 `_force_fuse` 之前**就先拒绝任何一端是
`OpType::other` 的边。`ArrayOp` 对一元素输出显式设 `_force_fuse` 并注明
「Fused scalar values are emitted inside generated kernels」——正是要这条融合。

实测对照重构前的提交 `845dd6c6`：

| | 生成的 kernel |
| --- | --- |
| `845dd6c6`（重构前） | 1 个：`array + broadcast_to + binary` 同核，标量是 **kernel 参数** |
| `d40a2e97`（重构后） | 2 个：`kernel<<<1,1>>>` 写 4 字节，再一个 binary 逐元素读回 |

代价按发射次数计（Transformer 训练一步，profiler 的计数口径）：

| 类别 | 修复前/步 | 修复后/步 |
| --- | ---: | ---: |
| **一元素常量单独成核** | **130.0** | **1.0** |
| 融合 elementwise | 115.9 | 164.9 |
| cublas GEMM | 72.0 | 72.0 |
| 其他 | 36.0 | 36.0 |
| 含归约的融合 | 16.0 | 16.0 |
| **合计** | **369.9** | **289.9** |

**每步 370 次发射里有 130 次只是往显存写一个标量。** 修法是：一元素来源不走存储
描述符（共享 4 字节没有收益），保留计算形式，由 fuser 折叠进消费者。`is_storage_view()`
改为按 `infer_shape` 决定的成员返回，与 `GetitemOp` 的 `storage_view` 同一模式。

#### 同一处还是一个静默算错

这条本来只当作性能改动，实测发现它修掉了一个**静默错误结果**。
一元素 expand 走存储描述符时，`jt.zeros/ones/full` 返回的张量其存储就是那个常量的
四个字节、全零步长——形状是 `(6,4)` 而缓冲区不是。Jittor 自己的算子读步长，看不出问题；
**任何拿它的指针按形状写入的外部库都会错**，而且不报任何错。

`cusparse_spmmcsr` 正是这样用的：`cusparseCreateDnMat(&matC, os[0], os[1], os[1],
outputVar->ptr<T>(), ...)` 把它当稠密矩阵写入。基线上单独跑
`tests/backends/cuda/test_cusparse_{op,dtype}.py` 是 **11 failed / 4 passed**，
改动后 **15 passed**。最小复现（基线树）：

| `output` 的来源 | 结果 |
| --- | --- |
| `jt.zeros((3,3))` | **MISMATCH**，读回全 0 |
| `jt.array(np.zeros((3,3)))` | ok |

Python 侧可见的判据是 `.data` 的 numpy 步长：基线上 `jt.zeros((4,3),'float32')`
报 `(0, 0)`、`C_CONTIGUOUS=False`，改动后是 `(12, 4)`、连续。
回归测试 `tests/core/test_constant_tensor_storage.py` 直接断言这条不变量
（修前失败、修后通过，与修复同一提交），并同时钉住「多元素 expand 仍然是视图」。

## 结果

### 元算子（n=2^22 float32，µs/算子，越小越好）

同一张卡背靠背测得。`jt 最终` = 三项改动全部生效。

| 用例 | jt 基线 | jt 最终 | 加速 | eager torch | torch.compile |
| --- | ---: | ---: | ---: | ---: | ---: |
| unary_abs | 25.11 | 14.06 | 1.79x | 12.61 | 12.50 |
| unary_exp | 25.22 | 14.10 | 1.79x | 12.62 | 12.61 |
| binary_add | 27.71 | 17.92 | 1.55x | 17.29 | 17.13 |
| binary_mul | 27.72 | 18.05 | 1.54x | 17.17 | 17.20 |
| ternary | 82.85 | 21.74 | **3.81x** | 18.44 | 18.24 |
| fused_mul_add | 27.93 | 18.07 | 1.55x | 29.25 | 17.23 |
| fused_chain5 | 30.09 | 19.28 | 1.56x | 106.11 | 17.19 |
| reduce_all | 13.81 | 13.42 | 1.03x | 12.33 | 51.21 |
| reduce_lastaxis | 13.76 | 10.93 | 1.26x | 17.85 | 7.49 |
| reduce_firstaxis | 13.11 | 11.06 | 1.19x | 12.33 | 53.51 |
| bcast_row | 76.05 | 15.28 | **4.98x** | 23.35 | 40.89 |
| bcast_col | 75.42 | 15.40 | **4.90x** | 23.41 | 12.72 |
| strided_slice | 23.46 | 11.55 | 2.03x | 13.76 | 11.52 |

**基线的广播用例是双峰的**：同一配置多次运行里 `bcast_row` 报过 32.8 µs 也报过 76.1 µs。
上表取的是与其它列同一轮的那次。按 32.8 算，这两行的加速是约 2.1x 而不是 4.9x；
改动后不再出现双峰。引用这两行时要连口径一起说。

**必须和 `torch.compile` 比，不能只和 eager 比。** `fused_chain5` 对 eager torch 快 5.5x
里有很大一部分只是「Jittor 融合成 1 个 kernel，而 eager 发 5 个」；inductor 同样会融合，
对它是 19.28 对 17.19，Jittor 略慢。真正站得住的对比是：改动后 Jittor 在 13 个用例里
有 7 个不差于 eager torch，其余在 15% 以内；对 torch.compile 则各有胜负——
它在 unary 与按最后一维归约上更快，Jittor 在整体归约、按第一维归约和行广播上更快。

### 整网

元算子快 1.5–3.8x，**整网只有 1.00–1.41x**，因为 MLP/CNN 由 cublas/cudnn 主导。
Transformer 的收益随模型变窄而增大（同进程内 A/B，`cuda_thread_num` 1024 对 256）：

| 模型/模式 | tn=1024 | tn=256 | 加速 |
| --- | ---: | ---: | ---: |
| mlp/infer | 2.141 ms | 2.139 | 1.00x |
| mlp/train | 6.617 | 6.562 | 1.01x |
| cnn/infer | 9.691 | 9.671 | 1.00x |
| cnn/train | 11.233 | 11.206 | 1.00x |
| transformer/infer | 8.884 | 7.848 | 1.13x |
| transformer/train | 27.231 | 23.636 | 1.15x |

| transformer 宽度（4 层，seq 256，训练） | tn=1024 | tn=256 | 加速 |
| --- | ---: | ---: | ---: |
| d=256 | 13.714 ms | 9.759 | **1.41x** |
| d=512 | 27.234 | 23.630 | 1.15x |
| d=1024 | 31.544 | 30.629 | 1.03x |
| d=2048（2 层） | 61.257 | 58.041 | 1.06x |

标量融合单独的整网收益（两棵树在同一张卡上交替，取各自最小值）是 **1.01–1.05x**
（transformer 训练 1.05x 最高），发射受限的 decode 形态（b1 s1 d512 8 层）
5.09 → 4.92 ms，**1.035x**。少发 22% 的 kernel 只换来个位数百分比，因为发射是异步的，
多数不在关键路径上——这一点要如实说。

## 为什么之前没发现：4090 的结论不能搬到 H20

已有 skill `cuda-elementwise-bandwidth-roofline` 在 **RTX 4090** 上的结论是
「逐元素类整体已经贴着屋顶线（1086 GB/s，ratio 0.84），想更快只能少搬字节」。
那个结论在它的机器上是对的：4090 实测 copy 带宽 916.7 GB/s，访存先到顶，
发射配置不是瓶颈。

H20 的带宽约是它的 4 倍，同样的 kernel 形状只跑到 1300–1800 GB/s，
**发射配置反而成了先到的那个瓶颈**。所以「元算子已经贴着屋顶」这句话是
**依机器而定的**，换一代硬件要重测。

## 验证

- `tests/backends/cuda` + `tests/backends/parity/test_dtype_coverage.py`：补丁前后
  失败集合**逐 nodeid 相同**（各 31 条，无新增、无消失）。基线与补丁树各自独立运行、
  各用各的编译缓存。
- 下标特化：一份覆盖秩 1–5、逐轴广播、15 种多轴广播掩码、全零掩码、非广播切片、
  转置、ternary 三个操作数各自 strided、以及反向的对拍，**纯下标用例逐位相等**
  （`exp` 与求和顺序两条按 1 ulp 容差）。
- 一元素 expand 的静默算错：`tests/core/test_constant_tensor_storage.py` 在基线树
  11 条 subtest 失败、在补丁树全过；`tests/backends/cuda/test_cusparse_{op,dtype}.py`
  单独运行由 11 failed/4 passed 变为 15 passed。
- `tests/backends/cuda`（补丁树 21 failed / 284 passed）与基线（31 failed / 279 passed）
  逐 nodeid 对比：**无新增失败**，并且上面 11 条 cusparse 由失败转为通过。
- 一元素 expand：33 项定向对拍全过，覆盖标量常量在各算术位置、
  计算得到的一元素来源（reduction 回灌、softmax 形态）、消费者不是逐元素算子
  （reduce/matmul/getitem/transpose/直接读取）、多元素 expand 仍是视图、
  四种 dtype、秩 1–5、以及三类梯度。
- **未完成**：`tests/backends/parity/test_device_parity.py` 与 `tests/ops/test_ops.py`
  （约 227 个生成用例的 CPU/CUDA 逐算子对拍）在基线树与补丁树上并行运行中，本报告
  写作时约 9%，两边失败模式逐字符一致。完整 CPU 门禁、ROCm 与 NPU 未跑。

### 一次自己造成的假失败，记下来避免重复

第一轮 `tests/backends/cuda` 补丁树比基线多 2 条失败，都在 `test_softmax_cuda_grad.py`，
报 `Jit var YSMASK not found`。原因不是改动：**门禁还在跑的时候我改了 `src/ops/binary_op.cc`**。
`op_compiler` 在 JIT 时直接读 `.cc` 源文件，于是已加载的核心（旧 `jit_prepare`，key 里
没有 YSMASK）配上了新模板。计划文档第 0 节写明「门禁运行期间不得修改 src」，
这条规则就是为此存在。之后的改动改在**另一棵工作树**里，重跑后两条失败消失。

## 已定位但未实现

按算子的**建图**成本（不同步，只建节点，µs/算子）：

| 算子 | jittor | eager torch | 比值 |
| --- | ---: | ---: | ---: |
| binary Var+Var | 5.31 | 4.33 | 1.23x |
| **binary Var×标量** | **10.31** | 5.16 | **2.00x** |
| unary abs | 4.75 | 4.80 | 0.99x |
| **reshape** | 3.38 | 0.72 | **4.7x** |
| **reduce sum** | 11.86 | 5.06 | **2.34x** |
| ternary（含比较） | 7.35 | 9.55 | 0.77x |

1. **`Var × 标量` 要建三个图节点**（array + broadcast_to + binary），torch 只分派一次。
   本报告第 3 项去掉了多余的 *kernel*，但节点还在，每个标量运算多约 5 µs 主机时间。
   要去掉需要给 BinaryOp 一个立即数操作数，牵涉 JIT key、反向和所有后端，
   是一次独立的大改，本轮没有做。
2. 发射受限形态（b1 s1 d512 8 层）下 jittor 一步 5.05 ms、其中 **96.9% 是建图**，
   同结构的 eager torch 一步 1.43 ms。**主机路径慢约 3.5x** 是这一类负载的主要差距，
   与 kernel 无关。
3. `reduce sum` 与 `reshape` 的建图成本同样明显偏高，未定位。

另外，向量化访存（float4）在独立微基准里对标量访存有 1.31–1.52x，但改动后
Jittor 的逐元素已到 2800 GB/s（eager torch 2911、torch.compile 2938），
余量不大，未实现。

## 复现

```bash
export JITTOR_LAB_ROOT=/root/jittor-lab
source $JITTOR_LAB_ROOT/_state/metaop/env-scalar.sh
cd $WT
CUDA_VISIBLE_DEVICES=<一张卡> $PY $JITTOR_LAB_ROOT/metaop-perf/bench3.py --n $((1<<22))
CUDA_VISIBLE_DEVICES=<一张卡> $PY $JITTOR_LAB_ROOT/metaop-perf/check_strided.py
CUDA_VISIBLE_DEVICES=<一张卡> $PY $JITTOR_LAB_ROOT/metaop-perf/check_scalar.py
```

对照口径：`bench3_torch.py`（eager）与 `bench3_torch_compile.py`（inductor）在装有真
PyTorch 的解释器里跑同一组用例、同一套斜率方法。原始日志、编译缓存与工作树在
`$JITTOR_LAB_ROOT/_state/metaop/` 下，未版本化。
