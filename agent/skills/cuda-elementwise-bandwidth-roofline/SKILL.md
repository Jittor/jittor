---
name: cuda-elementwise-bandwidth-roofline
description: 量一个网络里逐元素/访存受限 CUDA kernel 花了多少时间、离带宽上限还有多远，以及怎么和真 PyTorch 做同口径对比。用于任务 3.23 这类「某类 kernel 只跑到峰值一半」的性能结论、改融合逐元素代码生成之后要出 before/after、或者要判断一个 kernel 到底是访存受限还是别的原因。含屋顶线怎么量、Jittor profiler 的两个会让数字差两倍的坑、nsys 与 profiler 怎么互校、以及角色分类的口径。
---

# 逐元素 kernel 的带宽与屋顶线怎么量

「这类 kernel 合计 X ms、只有峰值一半带宽」这种结论，**换一个人量会得到不一样的数**，
因为有四个地方各自能把结果改掉两倍以上：分母用标称还是实测、profiler 的 rerun 因子、
每个 kernel 算作哪一类、以及拿 Jittor 的哪种测量去比 PyTorch 的哪种测量。
本 skill 把这四件事钉死。同目录四个脚本就是可执行形式。

归约类不在这里，看 `cuda-reduction-strategy-comparison`。

## 0. 环境

```bash
cd <worktree>
PYTHONPATH=<worktree>/python \
JITTOR_HOME=<自己的> TMPDIR=<自己的> \
CUDA_VISIBLE_DEVICES=<自己的卡> nvcc_path=/usr/local/cuda/bin/nvcc \
PATH=/usr/local/cuda/bin:$PATH taskset -c <自己的核段> python <脚本>
```

**`PYTHONPATH` 不能省**：开发环境里 jittor 是 editable 安装、`.pth` 指向主树，
裸跑 `python 脚本.py` 量的是别人的代码，而且是静默的。每次量之前先
`print(os.path.dirname(jittor.__file__))` 确认。**性能任务里测错树 = 整份数据作废。**

性能测量期间那张卡要独占，前后各看一眼 `nvidia-smi`。

## 1. 分母：屋顶线用实测的 copy 带宽，不用标称值

```bash
python roofline_copy.py --mb 512
```

一个 grid-stride 的 `float4` 拷贝，读一遍写一遍。RTX 4090（sm_89）实测
**916.7 GB/s**（标称 1008 GB/s 的 91%）；`scale`（读-乘-写）914.9 GB/s，
两者在 0.2% 以内，说明确实是访存受限而不是别的。

**把这个数记下来当分母。** 说「只有峰值一半」时分母必须是这个实测值，
否则同一份数据能得出不同结论。

脚本里那句 `assert np.allclose(got, a*2)` 不是摆设：一个什么都没做的 kernel
会报出非常漂亮的带宽。

## 2. 分子：每个 kernel 的字节数从 profiler 拿，不要自己按 shape 算

`jt.profile_scope` 报告里的 `Input`/`Output` 两列是**字节每秒**，由
`FusedOp::statistics` 算出：只统计外部输入（`type==0`）与输出（`type==2`）的 var 大小，
**融合掉的中间量不计**。所以 `(Input+Output) × AvgTime` 正是「一个完美实现至少要
搬多少字节」——正好是屋顶线的分子，自己按 shape 数一遍只会数错。

由此：

    实测带宽 = 字节 / 实测时间
    时间下界 = 字节 / 916.7 GB/s
    ratio    = 实测时间 / 时间下界

**ratio < 1 是正常的，不是算错了。** 4090 有 72 MB L2：广播输入、生产者-消费者相邻的
kernel 有相当一部分流量根本没到 DRAM，于是「实测带宽」会超过 DRAM 上限。
判读方式：**ratio > 1.2 才值得看**，ratio ≈ 1 的 kernel 已经贴着屋顶，
再优化它只能靠**少搬字节**（改融合），不是靠改 kernel。

## 3. 会让你差两倍的那个坑：profiler 的 rerun 因子

`Profiler::record_and_run` 把每个算子跑 `1 << r` 次再取平均，其中

    r = NanoVector::get_nbits(rerun+1) - 2
    get_nbits(v) = 65 - lzcnt64(v)        // 即 v.bit_length() + 1

所以 **`r = bit_length(rerun+1) - 1`**，不是 `- 2`。`rerun=31` 时每个算子跑 **32** 次，
不是 16 次。把它算错，报告里每一个「每步耗时」和「每步调用次数」都会差**正好两倍**，
而报告内部完全自洽——总时间对得上、百分比对得上、什么都不会报警。

**判据：拿 nsys 的 `Instances` 除以步数，去对 profiler 的 `Count / reps`。对不上就是这个。**
`classify.py` 里的 `reps_per_step()` 已经是对的，注释里写了推导。

## 4. 两种测量都要做，它们答的不是同一个问题

```bash
# A. profiler：每个算子单独测，给出身份（融合了哪些元算子）与字节数
python profile_step.py --mode profiler --out base.json --tag baseline

# B. nsys：真实流水执行的 kernel 时长
nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop -t cuda \
     -o jt_step --force-overwrite true \
     python profile_step.py --mode nsys --steps 10
nsys stats --report cuda_gpu_kern_sum --format csv --output jt_kern jt_step.nsys-rep
```

A 会在每个算子外面加 `cudaDeviceSynchronize`，所以它**只**适合算「这个 kernel 相对
它自己的屋顶线如何」；拿它的总和去和 PyTorch 的 profiler 总和比是不对的。
B 没有算子身份也没有字节数，但时间是真的。

`classify.py nsys` 用 **jit key 的哈希**把两者接起来：生成的文件名里是
`..._hash_<h>_op.cc`，生成的 kernel 符号是 `func_<h>_0`，同一个 `<h>`。

```bash
python classify.py jittor base.json --achievable-gbps 916.7
python classify.py nsys jt_kern_cuda_gpu_kern_sum.csv --jittor base.json \
       --steps 10 --achievable-gbps 916.7
```

**互校判据**：两边的 elementwise 合计应当在 10% 以内（实测 3.29 对 3.37 ms）。
差两倍 → 看第 3 节。差得没有规律 → 你的 nsys 捕获范围没盖住稳态步。

再加一条独立的清醒剂：`--mode profiler` 会同时打印**不带 profiler 的墙钟**。
GPU 时间必须小于墙钟。实测 22.0 ms 对 31.9 ms，合理；如果 GPU 时间反而更大，
先怀疑 rerun 因子。

## 5. 角色分类：不分类的总和没有意义

`classify.py` 把每个 kernel 归入六类，**这个划分本身是结论的一部分**，
换个划分数字就变了，所以引用数字时要连划分一起说：

| 角色 | 判据 | 为什么单独一类 |
| --- | --- | --- |
| `elementwise` | 代码生成器产出的融合 kernel，op 列表里没有 `reduce` | 3.23 说的就是这一类 |
| `reduce` | op 列表里有 `reduce` | 3.22 的地盘，两边不要混 |
| `indexing` | `getitem` / `setitem` / `transpose` | 代码生成器产出但不是「融合逐元素」 |
| `handwritten:code` | `jt.code`，nsys 里符号在 `jittor::` 命名空间 | `nn/backends` 手写的 CUDA，不是代码生成器的产物 |
| `library:*` | cudnn / cublas / cub / cutt … | 与代码生成无关 |

`fuse_transpose` **算 elementwise**（它是融合进逐元素链的），裸 `transpose` 算 indexing。

PyTorch 侧同口径：

```bash
REAL_TORCH_PYTHON -m ... python profile_step_torch.py --out torch.json
python classify.py torch torch.json
```

PyTorch 的 `other` 一类（GroupNorm 的 `ComputeInternalGradientsCUDAKernel` 之流）
**要计入逐元素类**——计划里「PyTorch 的 3.07 ms」正是 `elementwise + other`
（本机实测 2.20 + 0.84 = 3.04 ms）。把它漏掉会让 PyTorch 看起来快 27%。

## 6. 实测基线（RTX 4090，TF32，`large_diffusers_unet2d` 一步前向加反向）

一步的设备时间，nsys / profiler 两法：

| 角色 | nsys | profiler |
| --- | ---: | ---: |
| `library:cudnn` | 11.65 ms | 11.86 ms |
| `handwritten:code` | 4.13 ms | 3.58 ms |
| **`elementwise`** | **3.37 ms** | **3.29 ms** |
| `library:cublas` | 3.05 ms | 2.10 ms |
| `reduce` | 0.57 ms | 0.59 ms |
| `indexing` | 0.25 ms | 0.62 ms |
| 合计 | 23.02 ms | 22.03 ms |

同机 PyTorch 2.12.1+cu126 一步 22.41 ms：conv/gemm 18.17、elementwise 2.20、
reduce/norm 1.20、other 0.84。

**两条会反直觉的结论，记下来省得重测：**

1. **逐元素类整体已经贴着屋顶线**（1086 GB/s，ratio 0.84）。「只有峰值一半」
   这个说法在今天的树上不成立。想让这一类更快，唯一的方向是**少搬字节**
   （更好的融合、不物化中间量），不是「把 kernel 写得更快」。
2. **归约类今天已经比 PyTorch 快一倍以上**（0.57 对 1.20 ms）——`WarpReducePass`
   之后的事。3.22 的验收目标「不慢于 PyTorch 的 1.13 ms」看起来已经达到，
   但它的口径把 GroupNorm 的归约算在 `handwritten:code` 里，引用前先看第 5 节。

## 7. 逐 kernel 归因：按「超出下界的量」排序，不要按耗时排序

按耗时排序，头几名永远是那几个搬得最多的大 kernel，而它们往往已经贴着屋顶。
`classify.py` 默认按 `step_us - floor_us` 排序，这才是**可回收的时间**。

本机基线上，49 种融合逐元素 kernel 里正向超出合计约 0.59 ms，只有三个来源：

| 来源 | 代价 | 归属 |
| --- | ---: | --- |
| **float64 标量除法**（两族 kernel，593 / 397 GB/s） | 0.55 ms | torch shim，见下 |
| 裸 `transpose`（565 GB/s，写合并读不合并） | 0.10 ms | `src/ops/transpose_op.cc` |
| 一堆几乎不搬数据的小 kernel（约 60 次、每次约 1.6 µs） | 0.23 ms | 纯 launch 延迟，改不动 |

**float64 那条值得单独记**：`python/jittor/compat/torch/installers/tensor.py`
的 `_make_truediv` 对「float32 张量 ÷ Python float」**故意加宽到 float64**
（注释说是为 1-ulp 对齐 PyTorch）。sm_89 的 FP64 是 FP32 的 1/64，
于是 diffusers `ResnetBlock2D` 那句 `(input + hidden) / self.output_scale_factor`
（`output_scale_factor` 默认就是 `1.0`）在整张特征图上跑双精度除法。
把 `use_wide` 临时改成 `False` 实测：逐元素类 **3.29 ms → 2.73 ms**（−16.8%），
整步 22.03 → 21.29 ms。这条改动由兼容层分区决定（有 bit-exact 用例钉着它，
`tests/compat/torch/test_torch_compat_promotion.py`），不要顺手改。

**判据：看到某个融合 kernel 的带宽只有同类的一半，先去生成的 `.cc` 里搜 `float64`。**
路径就在 profiler 报告的 `FileName` 列里。

## 8. 定位一个 kernel 是谁生成的

`--flag trace_py_var=2` 在 shim 模式下会崩（`jittor.utils` 没有 `_pytree`），别用。
可用的两条：

- `--flag profiler_record_shape=1`：`FileName` 列后面附上 `shapes:` 段，
  按形状就能认出是哪一层（本机 `[4,48,256,256]` 一眼就是 16×16 分辨率、48 头的
  注意力分数矩阵，它是逐元素类里最大的一块：softmax 那两个 kernel 合计 1.2 ms，
  而 PyTorch 走 memory-efficient attention 根本不物化它）。
- 直接读 `FileName` 指向的生成源码，看 `#line` 指回哪个算子的 `.cc`。

## 9. 改了代码生成之后

before/after 必须是**同一个脚本、同一台卡、同一个 `--tag` 之外全同**的两次
`profile_step.py --mode profiler`，并且：

- `grad_checksum` 与 `loss` 两个值都会打印出来。**数值先看这两个**，
  改生成代码最容易的失败模式是「快了但算错了」，只看时间的对比毫无意义。
- 每一轮换一个只进 jit key、不进生成文本的标记（`--flag` 加一个整数 flag），
  否则第二轮直接命中缓存里第一轮的 `.cc`，diff 全绿而你什么都没测到
  （见 `jittor-core-cpp-edit-loop` §7）。
- 生成源码的逐字节 diff 与 profiler 数字一起放进提交说明。

## 10. 已知会挡路的东西

- **shim 的 `Tensor.backward()` 在这个 UNet 的 CUDA 图上直接 abort**
  （`node.h Check failed: value_ > 0  backward liveness release without a matching owner`），
  `_ecosystem_runner.py --runtime jittor --device cuda` 同样崩。`jt.grad(loss, params)`
  提交的是同一批反向算子且不走那条路径，脚本因此用 `jt.grad`。看板上有这条。
- 第一条命令可能拿到 `jit_utils was rebuilt ... rerun the same command`，
  原样重跑，不需要清缓存。
- shim 模式与非 shim 模式的 cfg 哈希不同，交替跑会各重编一次核心（约 40 s）。
  量的时候固定一种模式。
