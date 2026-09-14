# Jittor vs 真 PyTorch 2.9.1：现在差在哪

- Status: 进行中。vs eager PyTorch 赢 6、平 5、输 1（本轮起点赢 2 平 1 输 9），
  唯一那条输的（prefill 训练）由 torch 侧 19% 的抖动主导，比值在 0.95–1.12 之间摆。
  vs 最好的 torch.compile 模式赢 5 平 3 输 4
- Date: 2026-09-14
- Baseline: `perf/metaop-broadcast`（基于 `d40a2e97`）
- Oracle: PyTorch 2.9.1 + CUDA 12.9，`allow_tf32=False`（jittor 也不用 tf32）
- Review when: 图构建改成可重放，或 `auto_flush_bytes` 的判据变化

## 怎么测的

**每个 case 独立进程，而且要按"整段"预热。** 一步的预热完全不够：b8s256 的
训练一段一段量下来是 111.6 → 44.4 → 40.3 → 20.5 ms/step 才稳定，所以只热三步
就开始计时，十一轮 best-of 会横跨两个状态，落在哪边全看运气——同一份代码因此
量出过 44.6 和 20.5 两个值，我一度把它当成"双峰噪声"，又一度把它错当成某个
改动的功劳。现在两边都先跑四整段再计时。

**标签用整词匹配，不要用子串。** `"b1s1"` 是 `"b1s128"` 的子串，于是 prefill
一直在用 decode 的步数。两边同样，所以比较仍然公平，但这类 bug 会让人以为
自己在量另一件事。



六个 model/shape 组合 × 推理/训练，**每个 case 独立进程**、两边交替跑同一张卡、
取多轮最小值。一个进程跑完全部 case 会让结果失真：同一个 b8s256 训练在两次
运行里测出 44.6 与 20.8 ms，因为前面的 case 留下的分配器状态不一样。

两边的模型定义逐行对齐，同步纪律也对齐：都在 `no_grad` 下推理，都只在一段
`steps` 结束时同步一次，让主机的发射开销与设备执行重叠。

## 现在的位置

连跑三轮取中位数。torch 侧同时量了 eager 和 `torch.compile`（`default` 与
`reduce-overhead`，后者用 CUDA graphs）。

| case | jittor | pt eager | pt compile | vs eager | vs 最好的 compile |
| --- | --- | --- | --- | --- | --- |
| tf b8s256 训练 | 19.57 * | 40.14 | 39.26 | **0.49** | **0.50** |
| cnn-6conv 推理 | 8.93 | 14.12 | 13.92 | **0.63** | **0.64** |
| tf d1024 decode 推理 | 0.44 | 0.68 | 0.58 | **0.65** | **0.75** |
| tf decode b1s1 推理 | 1.19 | 1.43 | 0.83 | **0.83** | 1.44 |
| cnn-6conv 训练 | 23.99 | 28.23 | 27.74 | **0.85** | **0.86** |
| mlp-1024x4 推理 | 0.37 | 0.40 | 0.38 | **0.93** | **0.98** |
| mlp-1024x4 训练 | 1.17 | 1.20 | 1.23 | 0.97 平 | **0.95** |
| tf b8s256 推理 | 13.17 | 13.41 | 13.00 | 0.98 平 | 1.01 平 |
| tf decode b1s1 训练 | 6.64 | 6.59 | 4.65 | 1.01 平 | 1.43 |
| tf d1024 decode 训练 | 3.53 | 3.49 | 3.55 | 1.01 平 | 0.99 平 |
| tf prefill b1s128 推理 | 2.86 | 2.82 | 1.78 | 1.02 平 | 1.61 |
| tf prefill b1s128 训练 | 8.02 | 7.58 | 7.57 | 1.06 ** | 1.06 |

**vs eager：赢 6、平 5、输 1**（本轮起点是赢 2、平 1、输 9）。
**vs 最好的 compile 模式：赢 5、平 3、输 4。**

`*` b8s256 训练在 19.6 / 42.9 两个状态之间跳，这一轮落在快的一半。

`**` prefill 训练这条**给不出判定**：jittor 侧稳定在 8.013–8.021，而 **torch 侧在
7.26 和 8.64 之间摆（19% 的幅度）**，九次历史里比值从 0.95 到 1.12，中位数约 1.02。
它落在哪边取决于 torch 那一次跑成什么样。

两边的强弱是互补的：

- **compile 强在主机受限的 transformer 小步**——decode 推理 1.43 → 0.83、prefill
  2.82 → 1.78、decode 训练 6.59 → 4.65。这也是 jittor 离得最远的三条（1.4–1.6x）。
- **jittor 强在设备受限的地方，而 compile 帮不上 torch**——cnn 推理 compile 之后
  反而略慢，b8s256 训练 39.26，mlp 训练 compile 之后变慢。这几条领先 15–37%。
- **`reduce-overhead` 在全部六条训练用例上失败**，报
  `accessing tensor output of CUDAGraphs that has been overwritten by a
  subsequent run`——复用输出缓冲区导致的别名。训练侧的 compile 实际只有 default
  可用。jittor 的回放不会有这个失败模式：它每次返回新建的 Var（量下来和轮换缓冲区
  一样快，1.182 对 1.183 ms，所以没有理由换那个契约）。

## 最后一条输的：d1024 decode 训练，成因分散

五轮中位数、两边都热十段：jittor 3.50-3.59，torch 3.01-3.05，**1.17**。
（prefill 训练 1.03 卡在平/输边界上；为它单独跑过七对交替，比值 1.029，算平。）

把这一步拆开：优化器 0.146 ms，前向+反向 3.512。所以差的 0.50 ms 全在前向和反向。
同一个前向在三种条件下：

| | ms |
| --- | --- |
| 带梯度（cuBLASLt 被梯度守卫挡掉） | 1.806 |
| `no_grad`，eager，cuBLASLt 开 | 1.378 |
| `no_grad`，eager，强制可移植 Linear | 1.400 |
| `no_grad`，图回放 | 0.852 |

于是三笔账是清楚的：**cuBLASLt 在这个前向上只值 0.022 ms**（所以给它补反向没有意义，
试过 `jt.Function` 那条路是负收益，训练 8.02 → 10.30，已撤回）；**autodiff 建带值
0.428 ms**；**图回放值 0.526 ms，但回放只管推理**——训练的前向要建带，回放的图产不出
反向需要的带。

那 0.428 ms 拆到调用剖面上（grad 开减 grad 关）是：`Var.tape` 多 40 次 113 us、
`Function._run_call` 多 8 次 82 us、`tape_together` 多 12 次 35 us，加上 grad 开着
时走的是另一条路径（可移植 Linear、训练版 layer_norm）。但**逐个验证下来都是空的**：
关掉 layer_norm 的训练 Function，三段量法（开/关/再开）显示 3.661 / 3.502 / 3.504
——「关」和第二次「开」相同，差值是顺序偏差不是 Function。

把这 0.50 ms 的每一块都量过之后：

| | 实测 |
| --- | --- |
| 优化器（fused SGD 之后） | 0.146 ms，已是最小（重绑 0.006、zero_grad 0.018） |
| cuBLASLt 在带梯度的前向上 | 0.022 ms |
| `jt.Function` 机制（tape / context / tape_together） | 约 0.10 ms |
| layer_norm 的训练 Function 换成可移植路径 | 约 0 |
| 用 `jt.Function` 给 cuBLASLt 补反向 | **负收益**，已撤回 |

Function 机制的单次成本是量出来的：`x.tape()` 1.10 us、`_new_call_context()`
0.36 us、整个 `f(x)` 10.81 us 而它的 `execute` 只是一个 3.69 us 的算子——机制本身
7.1 us 一次。d1024-L4 一步 12 次（LayerNormCUDA 8 + CodeSoftmax 4），合计约 0.10 ms。

**先前写「剩下是每算子的 autodiff 记账」，那是错的。** 直接量：一个普通算子在
grad 开和关下的成本差约等于零（`x * 2` 是 3.96 us 对 no_grad 的 5.66 减去 0.92 的
scope，即 -0.79 us，在噪声里）。记账本身不花钱。

真正的去处是**grad 开着时 layer_norm 和 softmax 换成了训练版内核**，而这两条的主机
成本比 torch 高得多：

| 算子（主机时间） | jittor | torch | 倍数 |
| --- | --- | --- | --- |
| layer_norm (1,1,1024) grad on | 46.4 us | 11.0 us | **4.2x** |
| layer_norm (1,128,512) grad on | 45.7 | 10.8 | **4.2x** |
| softmax (1,8,1,1) grad on | 30.8 | 5.5 | **5.6x** |
| layer_norm no_grad | 23.2 | 9.4 | 2.5x |

d1024-L4 一步 8 个 layer_norm + 4 个 softmax，按这个差距就是约 380 us——正是那 500 us
缺口的主体。

层层拆开 jittor 的 layer_norm（这一组不带 `.sync`，量的是纯主机时间）：

| | us |
| --- | --- |
| `nn.layer_norm` 整个 | 28.3 |
| `_layer_norm_cuda`（relay + 派发 + Function + code op） | 24.9 |
| &nbsp;&nbsp;其中 `cls.apply(x, w, b)` | 17.6 |
| 外层的形状校验与 `fp32_guard` | 3.3 |
| `_output_requires_grad` | 0.24 |
| `_supports_layer_norm_training` | 0.86 |
| Function 类的 `lru_cache` 命中 / 实例化 / `_new_call_context` | 0.10 / 0.16 / 0.35 |
| `jt.code` 三入三出（对照，含 sync） | 14.1 |

所以外层的校验和派发都不是问题（3.3 + 0.9），**成本在 `jt.Function` 的机制加上三输出
的 code op**。torch 的整个 layer_norm 是 11 us。

在真实训练步里量过这两条的上限（把它们换成一次乘法，答案是错的，只为定尺寸；
基准取「开/关/再开」里稳定的那两次，不取第一次）：

    as shipped                    3.525 / 3.528 ms
    layer_norm -> 一次乘法         3.204      => layer_norm 整个值 0.322 ms
    softmax    -> 一次乘法         3.350      => softmax    整个值 0.175 ms

两条合起来 0.497 ms，正好是缺口的大小——**但那是它们的全部成本，不是可省的部分**。
按上表 2.5-4.2 倍的差距，追平 torch 能省其中约 60-75%，即 0.33-0.38 ms，把 3.53
带到 3.15-3.20，对 torch 的 3.03 是 **1.04-1.06**。也就是说：**即使把 layer_norm 和
softmax 做到和 torch 一样快，这一条仍然翻不过来。**

修法本身是清楚的——把训练版 layer_norm 的 `jt.Function` 换成 code op 上的
`cuda_grad_src`，省掉 context、四次 `tape`、`tape_together` 和 `_grad` 间接层，按
实测的 Function 机制成本值约 72-128 us 一步；softmax 同理。**它值得做**，因为
`no_grad` 的 layer_norm 也比 torch 慢 2.5 倍，那是每个 transformer 推理都在付的。
但它不是这一条 case 的答案，而把一个三输出算子的反向从 python `grad()` 改写成
`cuda_grad_src` 需要完整的验证周期。

唯一的结构性出路是让图回放覆盖训练，而那需要参数原地更新——在惰性图里，优化器写 p
的时候前向/反向可能还没执行，jittor 每次新建 Var 正是为了避开这个读写冲突。那是另
一个量级的改动。

## 更正：先前那张 torch.compile 的对比是错的

两个独立的错误，都是我的 harness 的问题，方向都对 jittor 有利。

**一、compile 侧的训练循环和 eager 侧不一致。** eager 用
`opt.zero_grad(set_to_none=True)`，我给 compile 侧写成了 `False`——后者要为每个
参数跑一次清零核。**这正是我先前报告的「`reduce-overhead` 在全部六条训练用例上
失败（CUDA graphs 输出被覆盖）」的成因。** 改成 `True` 之后它全部正常工作，而且很
快。我据此写过「jittor 的回放不会有这个失败模式，是鲁棒性优势」——**那个说法是我
自己的 bug 造成的，收回。**

**二、compile 的数字是单独一趟量的，跨了机器状态。** 这台机器有两个相差约 2.2 倍
的性能状态，两个框架都会落进去。验证过：只要两条臂挨着量，比值是稳的（b8s256 推理
0.95–0.99、cnn 0.63–0.65，绝对值同时跳 2 倍而比值不动）；但跨趟量就不成立——同一个
`vs_torch_pt.py` 在两个 harness 里量到 13.4–13.9 和 6.25。

改正后重量（jittor / eager / compile / compile+CUDA graphs 四条臂在同一个 case 内
交替，再用该轮的 jittor/eager 比值对照五轮基准，丢掉两条臂落在不同状态的轮次）：

| case | vs 最好的 compile |
| --- | --- |
| cnn-6conv 推理 | **0.67 赢** |
| cnn-6conv 训练 | **0.92 赢** |
| mlp-1024x4 推理 / 训练 | 0.97 / 0.99 平 |
| tf decode b1s1 推理 | 1.42 输 |
| tf prefill b1s128 训练 | 1.42 输 |
| tf d1024 decode 训练 | 1.58 输 |
| tf prefill b1s128 推理 | 1.61 输 |
| tf decode b1s1 训练 | **3.01 输**（6.62 对 2.20） |

**vs torch.compile：赢 2、平 2、输 5**（12 条里 9 条状态一致可用），不是先前写的
赢 5 平 3 输 4。b8s256 训练三轮都状态不一致、判不了；它的 compile 臂一致是 0.53，
jittor 应该在那条大赢，但不计入。

vs **eager** 的对位表不受这两个错误影响：那个 harness 在每个 case 内交替跑两边，
比值经上面的校验大多与基准一致。

## 那个「双稳」不是双稳，是我预热不够

b8s256 训练在 19.6 和 42.9 两个值之间跳，我先前当成机器的双稳态、在表里标了
「给不出比值」。把它一段一段量出来就清楚了（每段 5 步）：

    region 0-1:  203.5, 109.2     编译
    region 2-5:   42.0, 42.5, 42.5, 42.6     一个平台
    region 6:     29.7            过渡
    region 7+:    19.6            稳定，之后一直平

**要七段才稳定，而 harness 只热四段**，于是它有时停在 42 ms 的平台上。不是随机的
双稳，是一条需要 35 步才走完的收敛曲线。两边的预热都改成十段之后，这一条稳定回到
0.49，而 torch 侧也因此 settle 得更好——`tf-d1024-L4 decode 训练` 的 torch 读数从
3.02–3.52 收紧到 3.01–3.05，于是那一条从「平」变成 jittor 真的输 1.17。是更长的
预热把两边的真实稳态都量出来了。

顺带：**jittor 的这条训练要 35 步才进入稳态，torch 几乎一开始就稳**。这本身是用户
会感觉到的差别，短跑会一直停在 2 倍的开销上；成因还没查（看起来是分配器状态）。

## 已落地：优化器不再按参数一个一个来

训练步里最大的一块不是模型，是优化器。8 层 d512 的一步量出来是
fwd+bwd 6.33 + 更新循环 1.11 + `zero_grad` 1.03 = 8.49 ms，torch 一整步 6.61 ms。
两处都改掉之后是 6.51 ms。

**`zero_grad` 每步重建 96 个零。** `g.update(jt.zeros_like(g).stop_grad())` 对每个
梯度做一次，而 `post_step` 每步都调它。`zeros_like` 是标量的 broadcast，不带存储，
所以花掉的不是显存而是主机时间。改成每个梯度的零建一次、之后复用（没有任何路径
原地写梯度缓冲：`update`/`assign`/`+=` 都是重绑 holder）：**1.031 -> 0.037 ms**。

**CUDA 的 fused SGD。** 可移植的更新是每个参数两个 elementwise 算子加一次 holder
重绑。PyTorch 不付这笔，因为它的 SGD 是 `foreach`：整个表一次 launch。jittor 此前
只有 ACL 有 fused optimizer，CUDA 没有。现在每个参数的指针、梯度指针、输出指针和
长度装进一个按值传的参数结构，一个核走完全部；结构大小就是一批的上限（CUDA 给核
4 KB 参数空间，按 96 个一批切）。同时让 `momentum == 0` 也走 fused——原来只在有
momentum 时才问，理由是无 momentum 的捷径"已经是一趟"，但那是**每个参数**一趟。

**decode b1s1 训练步 8.49 -> 7.41 -> 6.51 ms（torch 6.61）。**

写这个核时撞到的两条 code op 约束，都写进注释了：`cuda_src` 是拼进
`CodeOp::jit_run` 函数体里的，`struct` 和 `__global__` 要放 `cuda_header` 才在文件
作用域；结构体成员不能叫 `out`，模板在函数体外做 `#define out out0`，`args.out[k]`
会被改写成 `args.out0[k]`。

## 已落地：自动 flush 按工作量而不是算子个数

见提交 `自动 flush 改成按待执行的工作量决定`。`auto_flush_ops` 原来只数算子：
满 128 个就把待执行的全部发射出去。但**一次 flush 会把这一步的图从中间切成
两批**，而融合是在一批之内决定的——跨过切口的 elementwise 链就变成两个核，
第二批还要再付一次规划。算子个数分不出这值不值：batch1 seq1 的 decode 和
batch8 seq256 的一步建的是**同样 ~100 个算子**，但前者只有 200 KB 待执行输出，
后者有几十 MB。

于是加一条尺寸判据 `auto_flush_bytes`（默认 8 MB），两个都满足才 flush。
主机受限的场景拿到 1.09–1.16x，设备受限的回到原水平，Resnet50 训练的流水
一分不少（146.1 对 145.9 ms；完全关掉是 154.7）。

顺带否掉的两条：
- **单纯提高算子阈值到 512**：主机受限有效，但 b8s256 亏 3%（15.07 → 15.51）。
- **降低算子阈值**（有尺寸判据护着）：单调变差，b8s256 在 af=8 是 16.60、
  af=128 是 15.05。flush 得越勤，被切断的融合越多。

## decode 那 2.25x 在哪：量过，不是猜的

一步 3.08 ms，按阶段拆（临时在 executor 里加计数器量的，量完撤掉）：

| 阶段 | ms | 占比 |
| --- | --- | --- |
| **Python 建图** | **2.382** | **70%** |
| 真正发射核函数 `execute_prepared` | 0.536 | 16% |
| 发射循环里除发射之外的部分 | 0.245 | 7% |
| `build_exec_plan`（阶段 2-5） | 0.117 | 3% |
| JIT key + 分配 | 0.127 | 4% |
| *PyTorch 的一整步* | *1.443* | |

**jittor 光是执行器就已经等于 torch 的一整步，建图是白加的。**

规划只占 0.117 ms——先前怀疑"规划是 O(n²)"是错的，`jittor-core-planning-cost`
里记的量级是对的。派发也不是热点：`select_kernel` 直接量是 2.1 us/次、
120 次/步 = 0.25 ms，cProfile 的 27% 是被 profiler 放大的。

建图为什么贵，一句话：**一步 6426 次 Python/C 调用建 ~100 个算子，
平均每个算子 64 次调用**，而 torch 的分派是 2-3 次。它是摊开的，没有哪一处
占大头——今晚把十处每算子开销拿掉也只换来 1.26x。

设备端没有问题：b8s256 下逐个算子对拍，十个算子全部在 0.96–1.07 之间。

### 前端到底占多少：把它整个拿掉量一次

"建图贵"还可以是两件事：贵在 `nn.Module`/functional 这一层，还是贵在它下面
的 C++ 算子构造。直接量：同一个 block 建两遍，一遍走正常前端，一遍用这棵树
暴露的最低层调用（`cublas.cublas_matmul`、`jt.transpose`、`jt.nn.layer_norm`）
拼出**同一张图**，只计构图：

| | us/block |
| --- | --- |
| 正常前端（nn.Module / functional） | 251.5 |
| 最低层调用，同一张图 | **156.3** |

**把整个前端拿掉省 38%。**（第一次量成 29% 是错的：那一版的"最低层"仍然在调
`jt.nn.layer_norm` / `nn.softmax` / `nn.gelu`，它们各自还带着自己的派发与
Function 机制。这一版直接落到它们底下的 `jt.code`。）

剩下的 156 us 是 C++ 侧的算子构造本身。前端微优化的天花板是 2.38 ms 里的
0.9 ms（整步的 30%）——值得拿，但拿完也不够。

按这个地板算：完美前端 → 建图 1.25 ms，整步 1.25 + 1.02 = **2.27 ms**，
对 torch 的 1.37 ms 仍然是 1.66x。

**所以 decode 要赢 torch，光把前端做薄是不够的，必须不再每步重建图。** 两档：

1. **录制回放**：录一次算子序列，之后在 C++ 侧回放。省掉全部 Python，但仍然
   每步构造一遍 C++ 算子（约 176 us/block 里的大部分）。估计整步落到 1.6 ms
   上下——追平，未必超过。
2. **静态图**：同一张图留着不拆，每步只换输入缓冲区重新执行。构图趋近于零，
   整步趋近 1.03 ms 的执行器成本，**这一档才真的赢**。代价是要让算子可重复
   执行，并且要有形状/控制流的护栏。

这两档都是特性，不是调参。上面的预算说明它们各自值多少。

## 这一节落地的四处（都是通用改动，不需要用户改代码）

### rank≥5 的 permute：内层循环没有被并行，一个 warp 一次都不合并

`ParallelPass` 把线程发给最外面 `max_parallel_depth` 层循环，CUDA 上默认 4 层。
秩 5 以上的 permute 于是把**最内层——也就是连续那一维——留在每个线程里串行跑**，
拿到线程的四层全是外层。相邻线程因此隔着一整行落地，没有一次访存是合并的。

attention 的 qkv permute 正好是这个形状
（`qkv.reshape(b,s,3,h,d/h).transpose(2,0,3,1,4)`）。b8s256 d512 下它搬 12 MB
用了 166 us——152 GB/s，这张卡远不止：

    max_parallel_depth=4（默认）  165.9 us   152 GB/s
    max_parallel_depth=5           62.3 us   404 GB/s
    max_parallel_depth=8           62.2 us   405 GB/s

2.7 倍，且 5 就是整个循环嵌套，所以 6 和 8 一分不多。

选项设在算子上而不是抬全局默认，并且**只给 broadcast 形态**：那一种是纯
permute，每层循环互相独立，全给线程不花钱；reduce 形态会融进可能带规约的核，
把被规约的轴并行掉意味着原子加，以及一个每次运行都可能不同的浮点求和顺序——
不该为一个 permute 悄悄换这个。秩 4 以内本来就整个嵌套都并行，不动。

nsys 对得上：b8s256 一步的设备时间 8.161 → 7.706 ms（torch 7.187），非 GEMM
那一半 1.335 → 0.883（torch 0.716），permute 核掉出前十。

### `+ - * /` 上那层只为复数标量存在的 python 包装

`1j * x` 得能用，哪怕 `x` 是 float32、事先没有任何东西说复数要来。原来的做法是
在八个算术运算符上装一层 python 包装去接这种操作数——于是**每个模型的每一次
加减乘除都为它付一个 python 栈帧**：一步 decode 88 次，0.160 ms，占 5.9%。

转换搬到转换器里（`py_array_op.cc` 与 `py_converter.h` 的 `is_type`——少了后者，
运算符连自己的重载都匹配不上，报"Wrong inputs arguments"）。用 `PyComplex_Check`
而不是 `CheckExact`：numpy 的 complex128 标量是 python complex 的子类，而它正好
是类型表唯一给不出名字的复数 numpy 标量（NPY_CDOUBLE 映到 `ns_void`）；complex64
不是子类，走数组那条路本来就对。

窄化到 complex64 不是这里的选择：`dtype_infer` 对任何复数组合都答 complex64，
而且它在看别的之前先看复数。包装服务过的十四种操作数/运算符组合行为不变，
`jt.array(1j)` 从报错变成能用。

### 小块 H2D：阻塞式拷贝在这台驱动上要 2.3 ms

纯 CUDA 量的，不涉及任何框架：

    raw cudaMemcpy H2D    2048 B : 2344.1 us
    raw cudaMemcpy H2D   65536 B : 2348.5 us
    raw cudaMemcpy H2D  131072 B :   15.8 us
    cudaMemcpyAsync(0)    2048 B :    1.5 us

断崖正好在 64 KB 这个 pageable 暂存阈值上，八张卡都一样，**PyTorch 也一样付**
（2 KB 的 `t.copy_(cpu_tensor)` 量到 2354 us）——所以这是驱动，不是谁写错了。
代价在**等**：异步发出去只要 1.5 us，小块 D2D 和 D2H 分别是 2.3 和 9.0 us。

`backend_copy` 本来就有 `ordered`，文档写着"ordered 的设备拷贝改为建立流依赖"，
但只有 D2D 认它。H2D 给同样的待遇：要 `ordered` 的调用方只需要字节在读它的核
之前落地，那是流顺序，不是主机等待。Pageable 的源在 `cudaMemcpyAsync` 返回前
就被拷进驱动自己的缓冲，所以调用方可以立刻复用——这里拿 2000 次"返回即刻覆写
源缓冲"验过，没有一次读到撕裂。Pinned 没有这个保证，那条仍然等。

### `jt.graph_replay`：不再每步重建图

上一节的预算说"这一档才真的赢"。做出来了，但**没有赢**——下面是改正后的数。

`jt.graph_replay(module, example)` 录一次推理调用再重放，每步把新输入写进录下来
的缓冲区。**两边同一套纪律**重量：每个 case 每条臂各自一个进程、每次调用换输入、
300 次一段、七轮取最小，每个答案都和 eager 对过。

| case（ms/次前向） | jittor eager | jittor + 回放 | torch | 取优/torch |
| --- | --- | --- | --- | --- |
| tf-d512-L8 decode b1s1 | 2.54–2.60 | **1.184–1.192** | 1.362–1.370 | **0.87** |
| tf-d512-L8 prefill b1s128 | 2.81–2.90 | **2.194–2.207** | 2.777–2.781 | **0.79** |
| tf-d1024-L4 decode b1s1 | 1.33–1.38 | **0.771** | 0.701–0.724 | 1.10 |
| mlp-1024x4 b256 | **0.336** | 0.374 | 0.367 | **0.91** |
| cnn-6conv | **4.12** | 4.155 | 13.96 | **0.30** |

decode 和 prefill 的推理**因此从输变成赢**；d1024 decode 从 1.99 收到 1.10，仍然输。
mlp 和 cnn 用 eager 就已经赢，回放对它们不划算（mlp 还略亏）。

b8s256 推理**不列**：它一次前向会落在相差约 2 倍的两个状态上（14.06 ms 或 6.49），
**eager 侧和回放侧一样会跳**。b8s256 训练跨会话在 20.0 / 43.6 之间跳是同一回事。

注意这张表的段长是 300 次，和上面对位表用的 `steps_for(label)` 不同——jittor 对段长
比 torch 敏感得多（cnn 推理在这里是 4.12，对位表里是 9.42；torch 两种量法都是 14.0）。
两边同一纪律时上表成立；**对位表那一张仍然是 eager 的、按原纪律的**。

### 两个把我骗过去的量法（都写进代码注释了）

**一、把录下来的那个 Var 再喂回去，等于什么都没量。** 参数就是录制时那个 Var 时
`_replay_once` 会跳过输入拷贝，而**没有写入输入，图就不会重新执行**——这一次调用
塌缩成把它已经持有的输出拷出来，而且**答案仍然是对的**，因为那个输入的答案没变。
b8s256 这样"回放"量到 6.8 ms 对 eager 的 14.0，我据此写下过一个 2 倍的结论，
错的。喂变化的输入再量：15.7 ms，比 eager 还慢。

**二、两条臂放在同一个进程里量的是顺序。** 后跑的那条起步就是热的。prefill 那次
回放被判定为"更慢"因而回退——两条臂跑的是**完全相同的 eager 代码**——第二条量到
1.57 ms，第一条 2.74 ms。现在每条臂各自一个进程，wrapper 自己那次对时也改成两条
臂交替各量两轮取最小。

### 护栏

录下来的图失效的每一种方式都是静默的——它不报错，它继续用上一次算出来的值回答。
每一条都是先看着它错了才写的：

- **录制必须还没被 finish。** 从留着的图里读一个值会把它 finish 掉，任何收走它的
  批次也会。`clone()` 录制输出就是这样：之后什么都不再重算，回放永远返回第一个
  输入的结果，每次 0.08 ms。于是有了 `Var.is_finished`，以及把拷贝放在
  `keep_graph` 里做。
- **失效时必须把图交回执行器跑一次（`keep_graph` 关着）才算释放。** 只把 python
  引用丢掉不够：节点是故意留着没 finish 的，而没 finish 的节点会一直 pending，
  之后每一次 sync 都会把它收走再跑一遍。两次重录之后设备上就在跑三份模型——nsys
  数到一步 321 个核，eager 是 130。
- **参数按 holder 身份比，不按值比**：`update` 会把 holder 重新绑到新 Var，而录
  下来的图继续读它录的那个。整套护栏一次 6.9 us，不是开销所在。
- 形状、dtype、参数个数、training 模式。
- **带随机数的图直接拒绝**（回放会重复同一次抽样）；**录制过程中把值读回主机的**
  也拒绝（它的 python 路径依赖张量的值），后者靠副作用检测——读回会把正在录的图
  finish 掉。

回放不是白来的：运行时本来就重叠得好的图，回放反而略亏。所以第一次录制会和 eager
对时，慢了就永久回退并在 `refused` 里说明；持平仍然回放。`measure=False` 可以跳过
这次对时（它要多花约四十次前向），代价是没有任何东西再拦着它比 eager 慢。

一个还在的限制：录制只活到下一个收走它的批次为止，`jt.sync_all()` 就会。长循环里
每几百次会重录一次（`stats` 看得到）。要让录制真正长命，需要把"保留"标在节点上而
不是做成全局 flag——那是核心生命周期的改动，这一节没有做。

## 还没收口

## b8s256 那 1.12x：一半是核函数，一半是发射空隙

**先纠正一个错的结论。**"逐算子和 torch 持平"是错的——那次是用
"调一次算子测一次墙钟"量的，对便宜的算子量到的是 **Python 分派**不是设备时间
（torch 的 softmax 被量成 59.8 us，而它的 profiler 说那个核只有 12 us）。

两边改用同一种办法重量：把上百次同一个算子排进队列、不中途同步，再除
——主机跑得比设备快的时候，每次的墙钟就是设备时间。torch 侧三轮完全稳定，
jittor 侧除 softmax 有一轮跳动外也稳定：

| 算子（b8 s256 d512） | jittor | torch | 倍数 |
| --- | --- | --- | --- |
| gemm 512→512 | 134.8 | 134.7 | **1.00** |
| transpose+reshape | 26.0 | 26.2 | **0.99** |
| x + x | 24.2 | 22.7 | 1.07 |
| **layernorm** | 44.1 | 29.7 | **1.49** |
| **softmax** | 53.2 | 32.2 | **1.65** |

**gemm 和简单 elementwise 是真的持平**，差距集中在两个手写核上。

**layernorm 那 1.49x 的成因已经看到了**：`layer_norm_cuda.py` 里的核把
`x[row * hidden + j]` 读了**三遍**——一遍求均值、一遍求方差、一遍归一化写出
——加上写出一共 4 趟访存；torch 的 `vectorized_layer_norm` 用 Welford 在寄存器
里一趟读完，共 2 趟。4/2 与实测的 1.49x 对得上（缓存吃掉了一部分重读）。
修法是把一行先读进寄存器再复用。**这一条后来做了**（提交
`layer_norm 的核把一行读进寄存器复用`）：44.1 → 27.4 us，反超 torch 的 29.7。
double 回退路径仍然重读内存。

**整体对账**（一步）：jittor 自己的逐算子设备时间加起来约 13.67 ms，墙钟
15.06，空隙约 1.4 ms（9%）；torch 的设备时间 12.84（它的 profiler 读数），
墙钟 13.40，空隙 0.56 ms（4%）。所以 1.66 ms 的差距大致是
**0.83 ms 核函数 + 0.84 ms 发射空隙**，两边各一半。其中已定位到具体算子的是
layernorm 0.23 ms + softmax 0.17 ms。

## 这台机器上量不了的东西

八张卡都被别人占着（`nvidia-smi` 全部 100%）。几十微秒量级的单核对比不可
复现：同一个 gelu 在两次运行里量到 53.2 和 34.3 us，还出现过"一个表达式比
它自己的子集更快"。**比值**（两边紧挨着测）稳定，绝对值不稳定，所以上表只
用比值。内置 profiler 也不能用：它会把每个算子重测，一步报出 306 ms 而墙钟
只有 15 ms。
- **`2fc3837d` 那条设备故障**：见
  [2026-09-13 的报告](2026-09-13-host-path-per-node-cost.md)。它在这条分支上，
  修掉之前整叠不能合。

## 整步训练回放：能做对，但只值 1.08x（做过，放弃了）

前面写过「回放只管推理，训练的前向要建带」。那个说法不完整：**可以回放整个训练
步**——前向、反向、参数更新一起捕获，反向核函数就已经在图里，回放时不需要任何带。

搭出来了，也验证了。机制上有三处是必须的，每一处我都先踩了一次：

1. **叶子必须是自己持有的私有缓冲。** `opt.step` 把参数 holder 重绑到更新后的
   Var，所以捕获前从模块上取的句柄在捕获后指向的是**更新后**那个；往它里面喂回
   更新是空操作，参数永远落后一步（实测差值 2.4e-5，正好等于一次 `lr*grad`）。
2. **更新要落在叶子缓冲里**，用已经暴露的 `share_with`：更新输出与叶子共用存储，
   执行图就是原地推进参数，不需要任何回写。（回写那条路走不通：48 次
   `_copy_into` 会把整张保留图重跑 48 次，量到 74.7 ms 一步。同理，48 次
   `sync` 是 61 ms——必须 `jt.sync(list)` 一次成批。）
3. **回放的 sync 要开 `keep_graph`**，否则第一次就把图收掉，之后每次回放都是
   空转，而参数冻在第一步——答案看着对，其实是静默的错。

正确性是对着 eager 验的，一进程一臂（两条臂放同一个进程里没法验：eager 自己的
`jt.sync` 是 **weak sync**，会把待执行的捕获图一起收走）：

    30 步之后 worst |replay - eager| = 0.000e+00   BIT-IDENTICAL

**然后量速度，结论是否定的：**

| tf-d1024-L4 decode b1s1 train | ms/step |
| --- | --- |
| jittor eager | 3.530 |
| jittor 整步回放 | 3.274 |
| *torch eager* | *3.01–3.05* |

**1.08x，不是能翻盘的量级。** 中途有一次量到 1.543 ms（1.96x），那是错的：那个
进程里 eager 那条臂已经把捕获图收掉了，"回放"没在跑完整的步。**该读数作废。**

为什么只有 8%——把执行器按阶段计时（临时在 `Executor::run_sync` 里加计数器，量完
撤掉），稳态每步：

| 阶段 | eager | 整步回放 |
| --- | --- | --- |
| `top_weak_sync` | 0.4 us | 0.1 us |
| `build_exec_plan` | 210 us | 172 us |
| 编译 + JIT key | 32 us | 45 us |
| **`run_exec_plan`** | **1436 us** | **3728 us** |
| 执行器合计 | 1.68 ms | 3.95 ms |

回放确实把 Python 建图（eager 那 3.53 里约 1.85 ms）整个拿掉了，**但保留图的
`run_exec_plan` 反而是 eager 的 2.6 倍**，把省下的又吃回去了。不是融合垮了——
回放 206.8 kernel/步、设备 455 us，eager 188.9、415 us，差 9%。多出来的是启动
循环本身在一张大得多的 plan 上的开销。

所以这条路的奖金还在，只是门在别处：**如果保留 plan 的 `run_exec_plan` 能降到
eager 的 1.44 ms，整步回放就是约 1.65 ms，对 torch 是 1.8x。** `exec_plan.h` 自己
写着「这个切分是跨步复用 plan 的前提」——那是下一步该动的地方，不是建图。

还有一条必须记下的**危险**：保留下来的训练图会改写自己的叶子，所以它**不是幂等
的**——任何无关的 weak sync 把它扫进去重跑一次，就是悄悄多走了一个优化器步。
推理回放没有这个问题（重跑只是重算同一个答案）。要把这个做成能发布的自动特性，
得先在 core 里让 weak sync 跳过被捕获的训练图。

## 这一条到底输在哪：两边都是主机受限，而 jittor 的设备端快 2.3 倍

| tf-d1024-L4 decode b1s1 train | kernel/步 | 设备时间/步 | 整步 wall |
| --- | --- | --- | --- |
| jittor | 188.9 | **0.415 ms** | 3.53 ms |
| torch | 196.0 | 0.963 ms | 3.03 ms |

发射次数几乎一样，jittor 的核函数快 2.3 倍，**输的 0.5 ms 全部在主机侧**。

顺带把 `jt.Function` 的机制成本拆干净了（layer_norm (1,1,1024)，纯主机时间）：

| | us |
| --- | --- |
| `cls.apply(x, w, b)` 整体 | 18.12 |
| 其中 4 次 `Var.tape()` | 4.75 |
| `tape_together` | 1.64 |
| `cls()` 每次新建实例 | 0.89 |
| `__call__` + `_new_call_context` | 0.98 |
| 掩码/isinstance/列表等 | 2.19 |
| **机制合计** | **11.05（占 61%）** |
| 真正的算子（`execute` 里那个 3 输出 code op） | 7.07 |

`tape()` 是最大的一块，而 TapeOp 不做任何计算（`share_with` 之后只是个别名），
1.19 us 一个纯粹是算子构造。但一步 12 次 Function 也只有约 0.10 ms，**不值得为它
去动 tape 的图语义**——省掉输入侧的三次 tape 需要放弃「输入侧 stop_grad 边界」，
而 `GradHooker` 那种直接返回入参的 Function 会因此被错误地 stop_grad。

## 每算子地板：CUDA 上 7.9 us，而发射本身只要 2.6 us

前一节说「奖金在 `run_exec_plan`」。量到底之后，它不是 `run_exec_plan` 的结构问题，
而是**每个 CUDA 算子的固定开销**。用一条 200 个 `stop_fuse` 过的平凡算子的链，
捕获后回放（所以测量里没有任何建图），每算子的主机时间：

| | us/算子 |
| --- | --- |
| jittor，CUDA | **7.89** |
| jittor，CPU 后端（同一条链） | **1.05** |
| 裸 `cudaLaunchKernel`（本机实测） | **2.0–2.6** |

**和图的大小无关**（N=50/200/400 分别是 8.11/7.90/7.75），**和算子种类无关**
（python 标量操作数 7.37、Var 操作数 7.96、一元 7.36）。所以这是一条地板，不是
某个算子的问题：d1024-L4 一步约 190 个算子，光这条地板就是 1.5 ms（整步 3.53）。

拆到阶段（临时在 `run_exec_plan` 和 `execute_fused_prepared` 里加计数器，量完撤掉）：

| | us/算子 |
| --- | --- |
| 5 个 RAII scope + LaunchRecord | 0.27 |
| 输出 `alloc` | 0.17 |
| `prepare_execution` | 0.16 |
| JIT key `to_string` | 0.04 |
| migrate 检查 + `record_active_launch` | 0.14 |
| `jit_fused_ops` 查表 | 0.06 |
| **生成代码 `entry(this)`**（四行加一次 `<<<>>>`） | **5.38** |

真实模型里按算子类型分（d1024-L4 一步）：融合 JIT 核 81.8 个 × 13.94 us、
cuBLAS 72 个 × 12.56 us、存储视图等 140 个 × 2.03 us。

排除掉的几条（都实测过，都不是）：
- **dlopen**：同一个 kernel 编进主程序 2.657 us，放进 dlopen 的 .so 也是 2.657 us。
- **遗留默认流的隐式同步**：进程里另外开 8 条阻塞流，stream 0 的发射还是 2.590 us。
- **队列打满导致发射阻塞**：一轮 30000 次发射也只是 2.588 → 2.978 us。
- **python 标量常量**（`x * 1.000001` 的 array op）：换成 Var 操作数或一元算子，
  每算子成本不变。
- **profiler / trace 钩子**：`profiler_enable`、`trace_py_var` 都是 0，
  `record_and_run` 在关闭时是直通。

## 出路：CUDA Graph（已量过奖金，也定位了唯一的阻碍）

既然整步的图已经验证可以捕获回放（上一节，30 步逐位相同），而它每次回放执行的是
**同一串 kernel、打在同一批缓冲上**——那正是 CUDA Graph 的形状。本机实测：

    200 次单独发射    334.7 us   (1.67 us 一个)
    一次 cudaGraphLaunch  2.6 us   (0.013 us 一个)

**发射开销塌缩 128 倍。** 而且捕获之后根本不走 `run_exec_plan`，上面那 7.89 us 的
地板整个消失。按这一条 case 估：3.53 ms → 设备时间 0.455 ms 加上 python 外壳，
**对 torch 的 3.03 是 4 倍以上**，并且对每一个主机受限的模型都成立。

唯一的阻碍已经定位清楚，也测过了：

    legacy default stream (0)       不可捕获
    cudaStreamPerThread             可捕获
    显式 non-blocking stream        可捕获

jittor 生成的是 `func<<<p1,p2>>>`——没有流参数，落在**遗留默认流**上，而 jittor
编译 CUDA 时没有加 `--default-stream per-thread`。所以要做的是：

1. nvcc 加 `--default-stream per-thread`，一处改动就让每个生成 kernel 和每个手写
   CUDA 算子都落到 `cudaStreamPerThread`（发射成本实测不变，2.574 对 2.588 us）；
   cuBLAS/cuDNN 还要显式 `cublasSetStream(handle, cudaStreamPerThread)`。
   这一条会去掉遗留流的隐式同步语义，**必须跑完整门禁对照 447 条基线**。
2. 捕获期间不能有任何 `cudaMalloc`（实测：捕获中分配直接报
   "operation not permitted when stream is capturing"）。保留图第二次回放时所有
   var 都已持有内存，正好满足；但分配器必须确认不会在捕获期间向驱动要内存。
3. 第二次回放时 `cudaStreamBeginCapture` / `EndCapture` / `Instantiate`，之后
   `cudaGraphLaunch`。失效条件与现有回放相同，再加上「捕获期间发生过分配」。

这一段**没有实现**。它是三处改动加一轮完整门禁，而上面每一条数字都是为了确认它
值得做、以及唯一的阻碍是什么——不是猜的。

## CUDA Graph：把每算子的主机成本从 7.9 us 变成整步一次 2.24 us

上一节定位到每算子地板 7.9 us（裸发射 2.6，CPU 后端 1.05），并写了出路是
CUDA Graph、唯一阻碍是遗留默认流不可捕获。**做完了。** 三段改动，每段单独验证。

### 一、所有 CUDA 工作挪到 `cudaStreamPerThread`

`compute_stream()` 原来返回 `nullptr`（遗留默认流），整个流抽象就这一处定义。
改成 `cudaStreamPerThread`，配上 nvcc 的 `--default-stream per-thread`（把全部
104 个 `<<<>>>` 一次性映射过去）和主机侧的
`-D__CUDA_API_PER_THREAD_DEFAULT_STREAM=1`，再把少数显式写出遗留流的地方补上：
4 个库句柄（cuBLAS/cuDNN/cuRAND/cuSPARSE，cuFFT 本来就绑流）、cuBLASLt 的两次
`cublasLtMatmul`、以及 driver/setitem/cutt/curand 里的几处 `cudaMemcpyAsync`。

**这一条必须做全**：`cudaStreamPerThread` 与遗留流**互不同步**，漏一个就是无序
并发，不报错也不打印。`grep cudaStreamPerThread` 就是这个集合的审计清单。

意外的是它本身就是一笔通用收益——遗留流每次发射都要和上下文里其它阻塞流做隐式
排序：

| | 之前 | 之后 |
| --- | --- | --- |
| 每算子地板（200 个算子的链，回放） | 7.89 us | **4.25 us** |
| d1024-L4 整步回放 | 3.235 ms | 2.769 ms |

### 二、捕获原语

`BackendOps` 加四个函数指针（begin/end/launch/release），CUDA 后端实现，
`graph_capture.h` 暴露给 python。空的捕获**被拒绝而不是返回一个什么都不做的
图**——那种图会一直答上一次的结果，是静默的错。

捕获中途踩到一个必须修的：**array 算子每次执行都用同步 `cudaMemcpy` 把主机端的
标量常量搬上去**（每个 `x * 2` 都建一个 array op，保留图每次重跑都会重搬），而
捕获期间同步拷贝非法，报 `cudaErrorStreamCaptureImplicit`。改成捕获中走流序拷贝。

### 三、结果

**整个训练步捕获成回放**（前向+反向+更新），一进程一臂对着 eager 跑 30 步：

    worst |graph - eager| after 30 steps = 0.000e+00   BIT-IDENTICAL

| tf-d1024-L4 decode b1s1 train | ms/step |
| --- | --- |
| jittor eager | 3.527 |
| jittor 整步回放（执行器） | 2.781 |
| **jittor CUDA Graph** | **2.186** |
| *torch eager* | *3.01–3.05* |

**1.39x，最后那条输的变成了赢。** 拆开看：

    一次 graph_launch（主机）        2.24 us
    单次启动 + 等设备                3.344 ms
    流水化                           2.186 ms/step

**主机成本从 3.5 ms 降到 2.24 us**，这一步现在完全是设备墙钟（0.455 ms 是核函数
执行时间，其余是设备侧逐核的调度间隔）。torch 在同样的核函数上是 3.03 ms 主机、
0.963 ms 设备——两边原本都卡在主机上，jittor 现在落到了设备地板上。

### 已经接进自动路径的部分

现有的推理 `GraphReplay`（`auto_graph_replay`，已带全部守卫）第三次回放时录一张
设备图，之后每次调用是一次 launch。输出落进本 wrapper 自己的缓冲（录进图里），
调用方仍然拿到新建的 Var——`_copy_into(src, sync_src=False)` 是为此加的：录好的
图已经产出了字节，再 sync 捕获输出就等于把整张图又跑一遍。

    tf-d1024-L4 decode b1s1 infer   0.852（旧回放） -> 0.684（换流） -> 0.538
    tf-d512-L8  decode b1s1 infer                                   -> 0.575

门禁：`tests/backends/parity/test_device_parity.py` + `tests/ops/test_ops.py`，
改动树与改动前那个提交**失败集合逐行相同**（各 1615 failed / 52 passed），
零新增、零修复。新增 `tests/core/test_graph_capture.py` 8 条。

### 还没做的：训练步的自动化

训练那 1.39x 是**验证过但还没自动化**的。自动化需要在 core 里定出「一步」的边界
（`opt.step()` 是天然的锚点），并解决上一节记下的那条危险：保留的训练图会改写
自己的叶子，**不是幂等的**，任何无关的 weak sync 扫到它就是悄悄多走一个优化器步。
推理那条没有这个问题（重跑只是重算同一个答案），所以先接了推理。

## 更正：我报过的 "1615 failed" 是我自己的 harness 造的

我的门禁把 `tests/backends/parity/test_device_parity.py` 和 `tests/ops/test_ops.py`
写进**同一次 pytest 调用**。Torch 兼容模式是进程全局的——它改惰性执行、reduce 默认值
和梯度语义——而前一个文件不会把它改过的状态放回去。于是后一个文件整片变红。

量给自己看（同一个 31% 进度点，同一棵树）：

| 命令里的文件顺序 | 失败 | 通过 | 失败率 |
| --- | --- | --- | --- |
| parity 在前（我的门禁） | 563 | 6 | **99%** |
| ops 在前 | 117 | 459 | **20%** |

`tests/ops/test_ops.py` 的文件头里写的运行方式就是单独跑；`refactor-dispatch.md`
也早就写过「把它们混进一次选择，正是同一个测试会因为命令行里和它一起写了哪个目录
而时过时败的原因」。两处我都没读。

**我该早点起疑**：1818 条里 1615 条失败本身就不合常理，而我只盯着「两棵树的失败
集合是否逐行相同」就放过去了。差集比较在那个口径下仍然成立（两条臂用的是同一个坏
顺序），但绝对数字是我制造的。

连带**收回一条结论**：先前记的「98 条 device-parity 由这批改动转绿」是在同一个坏
顺序下量的，不算数。正确口径（一个文件一个进程，`metaop-perf/run_gate2.sh`）重测。

## `tests/ops/test_ops.py` 不是「失败」，是**把进程打死**

按正确口径（一个文件一个进程）跑，`test_ops.py` 在 **73%** 处停住：两棵树、两次运行，
日志字节数完全相同，没有 pytest 汇总行——确定性崩溃，不是超时。崩溃前完成 1163 条，
第 1164 条是：

    tests/ops/test_ops.py::TestGradientsCPU::test_gradcheck_interpolate_bilinear

**后面约 400 条从来没有跑过**，所以此前任何「N failed」都不是这个文件的真实数字。

### 最小复现（12 行）

    x = jt.array(...)                                   # (1,1,3,3) float64
    y = nn.interpolate(x, size=(4,4), mode="bilinear", align_corners=False)
    y.numpy()                                           # 执行前向
    jt.grad(y.reshape(-1)[0], [x], retain_graph=True).numpy()   # 通过
    jt.grad(y.reshape(-1)[1], [x], retain_graph=True).numpy()   # SIGSEGV

三个条件缺一不可：**前向必须先被执行**（不执行则不崩）；必须是
`bilinear + align_corners=False + 上采样`（nearest、align_corners=True、下采样都不崩，
区别是这一支多了 `x.clamp(0, h-1)`）；必须是**第二次** `jt.grad`。
`mul` / `relu` / `sum` / `matmul` 在同样的形状下都不崩。

这是每一个 Jacobian / per-sample-gradient 循环的形状，不是边角。

### 崩在哪：规划器假设「输入 var 一定有生产者」

用 `cc_flags=" -g "` 重编取到行号（无符号构建会把内联归错地方，先前它一直指向
`count_fuse`，是误导）。逐个补判空之后崩溃点会前移，得到一条链：

    src/core/exec_plan.cc:343   opi = v->input(); opi->batch_index_at(tt)
    src/core/fuser.cc:225       producer = var->input(); producer->batch_index_at(tt)
    src/core/fuser.cc:133       func(var, var->input(), ...)   -> edge_fusable 解引用
    src/core/exec_plan.cc:278   同上
    src/core/exec_runner.cc:320 v->allocator->is_cuda()        -- var 没有内存

前四处是同一个模式：**批次里存在生产者已经执行完并被释放的 var**，而规划器到处默认
`v->input()` 非空。`exec_plan.cc:145` 那处已经写对了，注释就叫 `continue if is boundary`
——其余几处只是忘了写。

**但补完前四处之后冒出第五个**（`var->allocator` 为空），这说明逐点判空是在修症状：
把这些 var 标成「已物化」会改变分配语义。正确的修法在更上游——批次收集阶段就该把
这类 var 放进输入前缀（`start_var_num`），那里所有路径本来就处理妥当。

**没有修完，也没有提交任何猜测性的补丁。** 手上有精确的最小复现和这条链，下一步是
去看 `build_exec_plan` 的 BFS 为什么会把一个 `is_finished()` 的生产者的输出 var 收进
批次（`retain_graph=True` 下的 liveness 语义），而不是继续往规划器里加判空。

## 把 `test_ops.py` 的失败从 286 降到 56（其中 28 条是环境缺 cupy）

先绕开那条会打死进程的 `interpolate_bilinear`（上一节），拿到这个文件**真实**的
数字，再逐块查。三块，都不是「N 个独立的算子 bug」：

| 修的东西 | 性质 | 消掉 |
| --- | --- | --- |
| 参考里的 `np.atleast_1d` | 测试侧：对一个**已经修好**的旧行为的迁就 | 189 |
| `keepdim` / `keepdims` | **jittor 自己的 API 不一致** | 21 |
| 多输出 reduce 的解包 | 测试侧：只认 namedtuple，不认普通元组 | 20 |

    286 -> 97 -> 76 -> 56 failed        1150 -> 1363 passed

### 一、`np.atleast_1d`：迁就一个已经不存在的限制

参考被包在 `np.atleast_1d` 里，注释写着「jittor 没有 0-d 标量，全量 reduce 返回
`(1,)`」。**那句话过时了**：实测 sum/mean/prod/max/min/std/var/median/all/any/
count_nonzero 的全量 reduce 全部返回 `()`，和 numpy、torch 一致。于是这个 lift 不再
是在掩盖 jittor 的限制，而是**凭空造出一个分歧**——参考说 `(1,)`，算子说 `()`，
28 个算子的每一个全量 reduce 样本都在形状上失败，而且**失败发生在比较任何数值之前**，
所以它还顺带藏住了这些算子里可能真正的错误。

`tensordot` 和 `kthvalue` 那两处 `atleast_1d` 留着：jittor 在那里确实返回 `(1,)`，
参考与之相符，本来就没失败。（那是另一个问题，不在这一轮。）

### 二、`keepdim` / `keepdims`：六个算子六种口径

`keepdims` 是 jittor 和 numpy 的拼法，`keepdim` 是 torch 的。原生算子两种都收
（pyjt 的 `get_hash_condition` 把一个映到另一个），但 python 层的包装各写各的：

| | `keepdims` | `keepdim` |
| --- | --- | --- |
| sum / mean / max / min / prod、all_ / any_ | 收 | 收 |
| all / any | **都不收** | **都不收** |
| argmax / argmin / var | 收 | 不收 |
| std | 不收 | 收 |

`std` 和它正上方的 `var` 正好相反。不管调用者选哪一个拼法，总有算子会拒绝。现在
六个都两种都收。

### 三、多输出 reduce：只认 namedtuple

jittor 的 `argmax/argmin/argsort` 返回 `(indices, values)`，`sort/topk/kthvalue`
返回 `(values, indices)`——都是**普通二元组**，而 harness 只会解 namedtuple
（`hasattr(actual, "values")`）。于是拿一个二元组去和一个数组比，形状就差了一维。

要比的那一半**每个算子不同**，所以判断写在 OpInfo 的 `op=` 上，`kthvalue` 本来就是
这么做的。我第一次图省事在 harness 里写了「二元组就取 [0]」的通用规则，结果
**把 76 改成了 108**——`split`、`chunk`、`slogdet` 的元组本身就是答案。撤回重做。

### 剩下的 28 条（去掉 28 条 cupy 缺失）

    16  reinterpret_view_op.cc:58 byte size mismatch      -- 真算子 bug
    20  'float' object has no attribute 'astype'          -- 测试侧
    10  cross_entropy_loss 不接受 label_smoothing          -- 缺功能
     8  norm_p2 形状 (2,3) vs ()                           -- 默认 dim 不一致
     8  median 形状 () vs (1,)                             -- 还有一处参考没改到
     4  rms_norm gradcheck / gradgradcheck                 -- 真反向问题
     2  exec_runner.cc:440 融合算子执行失败
