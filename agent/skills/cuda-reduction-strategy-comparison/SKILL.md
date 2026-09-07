---
name: cuda-reduction-strategy-comparison
description: 比较 Jittor CUDA 归约的三条策略（每线程原子、warp shuffle 归约、块内共享内存树形归约）该用哪条，以及怎么把它们量准。用于改 WarpReducePass / SharedReducePass / ReduceTuner、调 para_opt_level、或怀疑某个归约 kernel 慢的场合。含四种配置的实测数、切换开关的准确写法、为什么手写的 wall-clock 微基准在这里必然测错、以及「Jittor 的归约比 PyTorch 快/慢多少」这句话该怎么对齐口径才成立。
---

# CUDA 归约：三条策略，怎么选，怎么量

## 谁在改那条 atomicAdd

生成的 CUDA 归约 kernel 结尾是每个线程把私有部分和直接原子加到输出元素上。
有三个 pass 会动这条语句，按 `pass_manager.cc` 里的顺序：

| 顺序 | pass | 做什么 | 开关 |
| --- | --- | --- | --- |
| 117 | `AtomicTunerPass` | 把累加提到循环外，每个输出地址一次原子而不是每次迭代一次 | 总是开 |
| 118 | `SharedReducePass` | 重排线程范围让一个 block 覆盖整个归约，再以 warp shuffle → 每 warp 一个共享值 → 首 warp shuffle 折叠，`if (threadIdx.x==0)` 写一次 | **`para_opt_level >= 4`，默认 3，即默认关** |
| 120 | `WarpReducePass` | warp 内 `__shfl_down_sync` 折叠，每 warp 一次原子 | 默认开，`no_warp_reduce` 关 |

**先确认你以为在跑的那条真的在跑**——这三条的开关条件都不写在一起：

```python
jt.flags.para_opt_level          # 3 = 默认，SharedReducePass 直接 return
jt.flags.compile_options = {"no_warp_reduce": 1}   # 关 WarpReducePass
```

判据看**生成的源码**，不要看名字：`shared_reduce<` 在不在、`_wr_mask` 在不在。

## 实测数（RTX 4090，float32，profiler 报的设备时间，每 kernel 平均）

四维张量沿 `(0,2,3)` 归约（diffusers UNet 反向里最多的那一类）：

| 形状 | 只有原子 | + warp（今天的默认） | + 块内共享内存（lvl 4） |
| --- | --- | --- | --- |
| 8×384×32×32 | 157.0us | **15.7us** | 25.3us |
| 8×128×64×64 | 92.1us | **14.0us** | 31.3us |
| 16×192×32×32 | 159.2us | **15.0us** | 25.3us |
| 32×64×56×56 | 171.0us | **18.1us** | 34.8us |

结论，会反直觉，记下来省得重测：

1. **warp shuffle 比块内共享内存快 1.6–2.0 倍**。共享内存版本要一个 1024 项的
   `__shared__` 数组、六次 `__syncthreads()`，尾部那段 `warpReduce` 还走 volatile
   共享内存（每步一次读一次写）；shuffle 版本五条 `__shfl_down_sync`，全在寄存器里。
   「块内树形归约」听起来更高级，实测更慢。
2. 两者都远好过不优化（6–10 倍）。所以**「SharedReducePass 从没生效过」不等于
   「归约没优化」**——问题在 9eb696d9 之后已经被 WarpReducePass 解掉了。
3. 精度相反：块内树形归约 relerr ~2.3e-7，warp ~3.5e-7，纯原子 ~1.8e-6。
   要可复现的求和顺序时这一条才是理由。
4. **两个 pass 会改同一条语句。** SharedReducePass 留下的是
   `if (threadIdx.x == 0) atomicAdd(...)`，WarpReducePass 匹配的是
   `startswith(code, "atomicAdd")`，会把这条已经只剩一个活跃 lane 的原子再包一层
   shuffle。运行期 `__activemask() != 0xffffffff` 会走回退，**结果对但全是死代码**，
   实测每 kernel 多花 1.3–1.9us。判据：lvl 4 的源码里同时出现 `shared_reduce<`
   与 `_wr_mask` 就是踩上了。

## 拿哪些形状去量：合成的四个形状和真实网络的七个形状答案相反

`reduce_ab.py` 把两条策略在同一个进程里对同一批张量跑一遍，同时给设备时间和相对
float64 的误差（归约改的是求和顺序，只有时间的表毫无意义）：

```bash
python reduce_ab.py --shapes unet             # UNet 一步真正用到的七种归约
python reduce_ab.py --shapes representative   # 上一节那四个合成形状
```

RTX 4090、best-of-30、同一次运行内（2026-09-06 实测）：

| 形状集 | warp 合计 | 块内混合合计 | 混合 / warp |
| --- | ---: | ---: | ---: |
| `representative`（8×384×32×32 一类，归约 (0,2,3)） | 64.50us | 73.91us | **1.146** |
| `unet`（下表七种） | 91.08us | 88.21us | **0.968** |

**两者方向相反，而且相反得有道理。** 合成那四个形状只有 64–384 个输出、每个输出上百
万个元素；块内路径给一个输出一个 block，于是整卡只跑 64–384 个 block，SM 喂不饱。
真实 UNet 的归约输出多、每个输出短，正好是块内路径合适的形状。

**推论：拿"代表形状"下结论之前，先用 `profiler_record_shape=1` 把目标负载真正用到的
形状读出来。**（怎么读：`cuda-elementwise-bandwidth-roofline` §8。）

`large_diffusers_unet2d` 一步里代码生成器发出的全部归约，就是 `--shapes unet` 那张表：

| 形状 | 归约维 | 每步次数 | 是什么 |
| --- | --- | ---: | --- |
| `[4,256,384]` | 0,1 | 24 | 注意力块里线性层的偏置梯度 |
| `[4,128,64,64]` / `[4,384,16,16]` / `[4,256,32,32]` | 2,3 | 17 | ResnetBlock 里 `hidden + temb[:,:,None,None]` 的广播梯度 |
| `[4,C]` | 0 | 19 | 时间嵌入线性层的偏置梯度 |
| `[4,384,256]` | 0,2 | 6 | 注意力输出投影的偏置梯度 |
| `[4,32,12,256]`（`mean`） | 2,3 | 18 | **六个注意力 GroupNorm 的回退**，见下面的口径一节 |

## 不要用整网梯度 diff 当归约改动的数值判据

默认的 warp 路径结尾是每 warp 一次 `atomicAdd`，求和顺序本来就每次不同。实测在
`large_diffusers_unet2d` 一步上：**同一条策略跑两遍**，270 个梯度里最坏的
`max|Δ|/max|g|` 是 **1.8**；warp 与块内两条策略之间是 **1.6**。自噪声比信号还大，
这个对比什么都判不出来。

同理，跨进程比 `loss` / `grad_checksum` 也不行：即便 `jt.set_global_seed` 之后模型
权重一致，卷积算法选择随负载变化，而 `(output*weights).sum()` 这种 loss 会抵消掉约
16 倍量级，于是输出上 3e-4 的差异在 loss 上就是 0.5%。实测同一配置两次 loss
13.718 与 13.653。

**能判的只有一种：逐形状对更高精度的参考。**`reduce_ab.py` 用 float64 的 numpy 做
参考，上面十一种形状两条策略的相对误差都 ≤ 6.9e-7。

## 3.22 两级混合路径复测（2026-09-03）

把旧的 1024 项共享树改成“两级 warp shuffle，中间只交换每 warp 一个值”后，level 4
路径从六次 `__syncthreads()` 降到两次。GPU profiler、30 次、同一输入的直接 A/B 为：

| 形状 | 默认 warp-only | 两级混合 level 4 | 混合 / warp |
| --- | ---: | ---: | ---: |
| 8×384×32×32 | 17.53us | 17.21us | 0.982 |
| 8×128×64×64 | 16.80us | 16.32us | 0.972 |
| 16×192×32×32 | 18.00us | 16.39us | 0.910 |
| 32×64×56×56 | 21.90us | 25.53us | 1.166 |
| 合计 | **74.23us** | 75.45us | **1.016** |

两边相对 NumPy 误差均不超过 `3.7e-7`，生成源码分别命中 `_wr_mask` 与
`shared_reduce<`/`if (threadIdx.x == 0)`。

**这张表的结论在 2026-09-06 被真实负载推翻了一半。** 上面那四个形状不是
`large_diffusers_unet2d` 用的形状（见「拿哪些形状去量」一节）。在整网上直接量
——`profile_step.py --flag para_opt_level=4 --compile-option reduce_lvl4=1`，
`classify.py` 的 `reduce` 角色：

| 配置 | reduce 角色（每次独立运行） | 均值 |
| --- | ---: | ---: |
| warp 默认（五次） | 568.3 / 566.7 / 571.5 / 560.3 / 562.4us | **565.8us** |
| 块内混合 level 4（三次） | 527.9 / 517.8 / 514.3us | **520.0us** |

即混合路径在真实 UNet 上**快 8.1%**，与 `reduce_ab.py --shapes unet` 的 0.968 同向。
只看通用求和（把六个注意力 GroupNorm 的回退摘掉）是 486.0 → 436.4us，**快 10.2%**。

**即便如此也没改默认**，理由换了一条：省下的约 45us 是整步的 0.2%，只占对齐口径下与
PyTorch 归约差距（profiler 口径 320us、nsys 口径 739us）的 6%–14%，够不到 3.22 的验收；
而在输出少、每输出长的形状上它仍慢到 1.39 倍，翻默认会伤到别的负载；`para_opt_level=4`
又是个粗开关，同时改了 AtomicTunerPass 的行为，本来就不是"只开 SharedReducePass"的
干净方式。它保留在 `para_opt_level >= 4` 作为 opt-in。

## 口径：说「Jittor 的归约比 PyTorch 快/慢」之前必须先对齐

这是 3.22 收口时踩到的坑，值得单独记：**两个框架把"归约"切成 kernel 的方式完全不同，
按 kernel 名字分桶得到的两个数可以几乎没有交集。**

3.23 留下的一句结论是「归约类 Jittor 0.57ms 对 PyTorch 1.20ms，快一倍以上」。
2026-09-06 把两个桶各自拆开看，它们装的是：

- **Jittor 的 0.57ms** = 代码生成器的 `reduce` 角色，仅此而已。其中 0.49ms 是通用求和，
  0.08ms 是六个注意力 GroupNorm 的回退。**不含**手写的 GroupNorm（1.72ms）、
  不含手写的卷积偏置梯度 `channel_bias_backward`（0.26ms）、不含手写 softmax（1.59ms）。
- **PyTorch 的 1.12–1.20ms** = `classify_torch` 的 `reduce/norm` 桶 = 0.65ms 真归约
  （两族 `reduce_kernel<...sum_functor...>`）**加上** 0.47ms 的 GroupNorm **逐元素仿射
  写回**——三个 `GroupNorm*KernelImplInternal` 的 `elementwise_kernel`，只因为符号名里
  有 "Norm" 就被归进了归约。而 PyTorch 真正的 GroupNorm **统计量归约**
  （`RowwiseMomentsCUDAKernel`、`ComputeInternalGradientsCUDAKernel`、
  `GammaBetaBackwardCUDAKernel1`、`ComputeBackwardFusedParamsCUDAKernel`，合 0.74ms）
  一个都不在这个桶里，它们落在 `other`。

### 对齐的做法：按语义配对，并核对次数

用 `profile_step_torch.py --attribute` 拿到每个 CUDA kernel 是哪个 aten 算子发的
（一个 `reduce_kernel` 符号同时服务卷积偏置梯度和普通 `sum`，只看符号名分不开），
再和 Jittor 侧按语义配对。**配对成立的判据是每步调用次数对得上**：

| 语义 | Jittor | PyTorch | 次数 |
| --- | --- | --- | --- |
| 卷积偏置梯度 | 手写 `channel_bias_backward` | `aten::convolution_backward>aten::sum` | **51 : 51** |
| 其余通用求和 | 代码生成 `reduce` 的 66 次 + `full_reduce_*` | 裸 `aten::sum` 61 次 + 注意力反向的 sum 6 次 | **67 : 67** |
| GroupNorm（统计量 + 仿射写回） | 手写 `group_norm_{forward,backward_x,backward_affine}` 35 个 + 代码生成回退 6 个 | `aten::native_group_norm(_backward)` 的八种 kernel | **41 : 41** |

三行的次数全部对上，这个划分才可以引用。

融合与不融合的一侧要整体对整体：Jittor 的 `group_norm_forward` 一个 kernel 里做完两次
块归约再写回，PyTorch 拆成 `RowwiseMoments` + `ComputeFusedParams` + 一个 apply，
所以只能三个一起算，不能只挑「统计量」那部分比。

### 对齐后的实测（RTX 4090、TF32、UNet 一步、2026-09-06）

**每个数字都要多跑几次。** 本节第一版发出去的那一组来自单次运行，而那次 `uptime` 一分钟
负载是 22（八个分区并行），手写 GroupNorm 报了 1715us；同一配置在负载 9–13 时的三次运行
是 1533 / 1535 / 1544us。**profiler 的逐算子测量对机器负载敏感，单次运行会高估 15%。**
下表列范围，不列单值。

| 类别 | Jittor profiler（三次） | Jittor nsys（两次） | PyTorch 2.12.1 CUPTI（三次） |
| --- | ---: | ---: | ---: |
| 卷积偏置梯度（51 : 51 次） | 257.0–258.8us | 363.6 / 364.5us | ┐ |
| 其余通用求和（67 : 67 次） | 486.2–494.1us | ~488us | ┘ 652.6 / 671.8 / 681.9us |
| 通用求和小计（118 : 118 次） | **744.9–753.2us** | **~853us** | **652.6–681.9us** |
| GroupNorm 全部（41 : 41 个） | **1532.9–1543.8us** | **~1850us** | **1275.5–1315.6us** |
| **归约类合计** | **2279–2297us** | **~2705us** | **1928–1998us** |
| 同次整步 | 21.19–21.35ms | 23.09ms | 21.11–21.95ms |

**Jittor 慢 15%（profiler 口径）到 36%（nsys 口径）**，不是「快一倍以上」。两种量法在
手写 kernel 上系统性地差一截（nsys 量的是真实流水里的 kernel 时长，profiler 逐算子加同步
单独量），**但方向与归因完全一致，所以结论不依赖选哪一种**。严格同口径的一对是
**nsys 对 CUPTI**（两边都是真实流水的 kernel 轨迹）。

**差距的 75% 在 GroupNorm**（profiler 口径 +241us、nsys 口径 +555us，两种量法算出来的
占比都是 75%），其中约 1.5–1.8ms 跑在 `backends/cuda/kernels/nn/group_norm_cuda.py` 的
手写 CUDA 里，**不在代码生成器里**；通用求和那一栏只差 79–184us。所以「给代码生成的归约
加块内树形归约」这条路最多够到差距里的 45us。

### 不可对齐的一项要单列，不要塞进合计

Jittor 手写的 attention softmax（`softmax_cuda.py`，前后向 1586 / 1411us）在
PyTorch 侧**没有对应的独立 kernel**：PyTorch 走 memory-efficient attention，softmax
在 `fmha_cutlassF/B` 内部，和 QK^T、PV 两个矩阵乘同在一个 kernel（合 3081us）。
把它计进任何一边都会让对比失真，只能单列并说明原因。

## 手写 kernel 少存一个中间量：三类证据都要，缺一条判不了

`group_norm_ab.py` 是 8.20 用的那套，改任何一个手写 norm kernel 都可以照抄：

```bash
python group_norm_ab.py --what accuracy     # float32 对 float64 NumPy 参考
python group_norm_ab.py --what memory       # 分配器被要走多少字节
python group_norm_ab.py --what time --repeats 3
```

三件必须一起做的事，每件都对应一种「看起来成立其实没成立」的结论：

1. **量真实负载的形状组合，不是一个代表形状。** 脚本里的 `SHAPES` 是
   `large_diffusers_unet2d` 一步真正提交的 12 种形状与各自的次数（35 次），用
   `profiler_record_shape=1` 读出来的（读法见
   `cuda-elementwise-bandwidth-roofline` §8）。拿 `4x128x64x64` 乘 35 得到的是另一个数。
2. **显存收益用 `use_stat_allocator=2`，不是 `=1`。** `=2` 挂在缓存分配器**之上**，
   数的是图要了多少字节；`=1` 挂在下面，数的是没命中设备缓存的部分，会随缓存状态跳。
   要证明「少了一份全尺寸中间量」，用 `=2`：它是精确值，两次运行一模一样。
   **驱动反向要绕开 loss**：`(y * cotangent).sum()` 自己就会分配全尺寸中间量，把
   要测的东西埋掉。脚本直接调 `Function.execute` / `Function.grad`。
3. **时间用整网 in-situ，而且 before / after 必须交替跑。** 这一条是本波踩出来的，
   代价是一整轮白测：先跑三次 before 再跑三次 after，期间机器一分钟负载从 4.29 单调
   涨到 7.02、并发编译进程从 7 涨到 20，结果 after 里混进一个 1575.3 us 的离群值，
   区间**重叠**，什么都判不出。**改成 before/after 交替（B,A,B,A,…）跑四对**，同一台
   机器同一段时间（负载 1.82–3.65、编译进程 0–4），三项全部区间不重叠。
   负载趋势落在一条臂上和落在两条臂上，是「有结论」与「没结论」的差别。

**别忘了证明两次比的是两份不同的源码。** jt.code 的 jit key 里带着 CUDA 源码原文，所以
「before 报告确实描述 before」是可以证明的，不用靠记性：

```python
keys = [r[name_index] for r in report["records"] if "group_norm" in r[name_index]]
print(sum("xhat[base + j] = out1_type" in k for k in keys))   # 12 before, 0 after
```

实测踩到过：一个 before 批次跑了一半时源码已经被改了，靠这一句才确认前三次都是旧码。

### 8.20 的实测结果（RTX 4090，UNet 一步，2026-09-07）

手写 GroupNorm 的 forward 原来除了 `y` 还物化一份全尺寸 `xhat` 存给反向；改成只存
`mean` 与 `rstd`（每组两个 float）、反向从 `x` 重算，和这里的 LayerNorm / BatchNorm
以及 torch 的 `native_group_norm` 一致。`--match group_norm`（35 个算子、70 条记录，
八次运行 calls 都是 70）。**四对交替**，负载 1.82–3.65、编译进程 0–4：

| | before 四次 | after 四次 | 屋顶线下界 |
| --- | ---: | ---: | ---: |
| forward | 540.3 / 541.6 / 542.3 / 569.0 | 445.3 / 445.4 / 446.3 / 446.5 | 667.6 → 445.1 us |
| backward | 905.9 / 907.2 / 908.8 / 909.7 | 918.6 / 920.6 / 925.4 / 925.9 | 667.6 us（不变） |
| 合计 | 1447.5 / 1449.1 / 1452.0 / 1476.2 | 1363.9 / 1366.0 / 1371.7 / 1372.5 | 1335.2 → 1112.7 us |

均值 **合计 1456.2 → 1368.5（−87.7 us，−6.0%）**，三项区间都不重叠。拆开看：

- **forward −102.4 us（548.3 → 445.9，−18.7%）**，这是全部收益。
- **backward +14.7 us（907.9 → 922.6，+1.6%）**，是收益的对价。**重算不是免费的**：
  它读 `x` 与原来读 `xhat` 的字节数确实一样，但每个元素多两次浮点运算，而
  `backward_x` 有两个循环要重算。这一项小，但四对交替下区间不重叠，是真的。

**为什么少搬 222 us 的量只快了 102 us，值得记：** profiler 自己的字节列说 DRAM 下界
降了 222.5 us（正好是 203,948,032 B ÷ 916.7 GB/s，即一份全尺寸 `xhat`），但 forward
改前的 ratio 是 548.3 / 667.6 = **0.82**——**那份写有一大半根本没到 DRAM，被 72 MB L2
吃掉了**。改后 forward 的 ratio 是 445.9 / 445.1 ≈ **1.00，正好贴在屋顶线上**：
它已经不可能再快，除非再少搬字节。**下界降多少 ≠ 时间省多少，要先看改前的 ratio。**

显存（本条唯一与负载无关的精确量）：同一批 35 个算子的 forward + backward，分配器
被要走 **611,953,152 B（583.6 MiB）→ 408,023,040 B（389.1 MiB）**，即
**3.001 → 2.001 份**全尺寸副本（一份 = 203,948,032 B = 194.5 MiB）。分配**次数**不变
（210 次），少的是尺寸不是数量。两次运行逐字节可复现。

数值：**36 个数组（9 种形状/分组 × y/dx/dweight/dbias）改前改后逐位相同**。因为重算
用的是同样三步 float 运算、同样顺序，而原来存下来的 `xhat` 也是 float32——注意这一条
只在快路径注册为 `dtypes=("float32",)` 时成立，若将来放开 float16，存下来的 `xhat` 会
被舍入而重算的不会，那时就不再逐位相同（会更准，但不再是「同一个数」）。对 float64
NumPy 参考的最差相对误差改前改后同为 **1.135e-3**，出在退化的 `(1,4,1,1)`、
`num_groups=2`（每组两个元素，`dx` 整个是抵消），其余 ≤ 2.15e-7，无一条变差。

**「xhat 占 3.22 缺口 75%」这个归因不成立，而且量级错了一个数量级。** 同场次同负载下
PyTorch 2.12.1+cu126 的 GroupNorm 合计（八种 kernel、328 次发射、41 个算子）是
**1288.1 us**。所以：

- GroupNorm 改前对 PyTorch 的差距是 **+168.1 us**，不是 8.20 描述里写的「绝对量
  1.5–1.8 ms」——1.5–1.8 ms 是 GroupNorm 的**总耗时**，不是**差距**，原描述把两者混了。
- 不物化 `xhat` 值 **87.7 us**，即这个差距的 **52%**，只堵了一半多一点。
- 改后仍是 **1368.5 对 1288.1，慢 80.4 us（+6.2%）**，**8.20 的验收未达成**。

剩下的全在 backward：922.6 us 对 667.6 us 的下界（**ratio 1.38**），而 forward 已经
贴在屋顶线上（ratio 1.00），不可能再从 forward 拿到东西。`group_norm_backward_x`
走三遍 `grad_y`、两遍 `x`，而 torch 的 `ComputeInternalGradientsCUDAKernel`
（410.8 us）加一次 apply 各走一遍。头两个归约循环都在读同一份 `grad_y`，合成一趟是
下一个杠杆，属于另一条改动。

## 3.22 两级混合路径复测（2026-09-03）

## 怎么量：不要写 wall-clock 微基准

Jittor 是惰性图。`t0=time(); y=x.sum(); t1=time()` 量到的是**建图**的时间，
kernel 还没跑；加一个 `.sync()` 又会把编译时间算进去。手写的微基准在这里
**必然**测错，而且错得像是「优化生效了」。

用 profiler 的设备时间，并且把编译赶到测量之外：

```python
jt.flags.use_cuda = 1
a = jt.random(shape); a.sync()
jt.reduce(a, "add", dims).sync()      # 先编译一次，不计入
jt.sync_all(True)
with jt.profile_scope(rerun=0) as rep:
    for _ in range(30):
        jt.reduce(a, "add", dims).sync()
    jt.sync_all(True)
hdr, rows = rep[0], rep[1:]
ni, ci, ti = hdr.index("Name"), hdr.index("Count"), hdr.index("TotalTime")
for r in rows:
    if "reduce" in r[ni]:
        print(float(r[ti]) / int(r[ci]) / 1000.0, "us")
```

三个要点：

- **`rep[0]` 是表头**，`rep[1:]` 才是数据行；`rep[i][hdr.index("FileName")]` 是
  生成源码的路径，拿它去 `open().read()` 检查 `shared_reduce<` / `_wr_mask`。
- **每次改 `para_opt_level` 都要换一个 `compile_options` 值**（例如
  `{"test_xxx": <序号>}`），否则第二次拿到的是缓存里上一个 level 编出来的 kernel，
  你会看到「改了没反应」。
- 每次都同时算一遍与 numpy 的相对误差。归约改的是求和顺序，**只看时间不看数值
  的对比毫无意义**——一个算错的 kernel 通常也更快。

## 走不走 JIT：先确认这个形状还在代码生成器里

`nn/backends/full_reduce_cuda.py` 把 `jt.Var.sum` / `jt.Var.mean` 猴补成了两级
CUB 折叠，**全量归约（不指定 dim）根本不进代码生成器**。所以

- 想量代码生成器产出的归约，用 `jt.reduce(x, "add", dims)`，不要用 `x.sum()`；
- 反过来，`x.sum()` 慢不慢与这三个 pass 无关，去看那个快路径。

同一个语义有两条实现、测试钉在其中一条上，是这个仓库反复出现的形状。量之前
先确认你打中的是哪一条。
