# CPU 双会话收零失败、FSDP2 单卡集合通信与两个新原生 notebook

- Status: CPU native/Torch 双会话零失败；TRELLIS 性能仍未接受
- Last reviewed: 2026-08-27
- Baseline: `99537948`
- Owner: Torch compatibility and downstream integration maintainers
- Review when: FSDP2 梯度同步、torch 命名空间发布集合、notebook 契约或
  NCCL 引导路径变化

## 结论

完整 CPU Torch 会话由 `10 failed` 收敛到 `1491 passed, 278 skipped`，退出码 0。
六条"与执行顺序相关"的失败原来是同一个缺陷，并非顺序问题；另外两条是过期的
断言常量。本轮同时新增 ViT 与 GPT 两个原生计图 notebook，并让 NCCL 引导失败
时给出可操作的诊断。

TRELLIS 性能未收口，但排除了一个主要假设：BF16 GEMM 本身与 PyTorch 完全打平。

## FSDP2 单卡集合通信

### 现象

`test_torch_compat_optim.py` 的五条 SGD/Adam 用例与
`test_torch_compat_ops.py::test_ne_after_detach_shared_graph` 在完整会话中失败，
单独执行却通过，此前被记为顺序相关。实际调用栈一致：

```text
tensor.py:1355 in _backward
grad_sync.py:265 in fill_fsdp_optimizer_grads_from_grad_map
grad_sync.py:70  in _sync_sharded_grads_from_full_grads
common.py:95     in _reduce_scatter_padded
AttributeError: 'jittor_core.Var' object has no attribute 'mpi_all_reduce'
```

一个普通 SGD 的 `backward()` 走进了 FSDP2 的梯度同步路径。

### 根因

`_backward` 收集的是 `jt._active_optimizers` 里**全部**存活的优化器，不只是本次
反向涉及的那个。前面 FSDP 测试留下的优化器仍被 unittest 的 traceback 引用而存活，
于是后续任何反向都会被拖去做 reduce-scatter。

真正的缺陷在下一层：`_reduce_scatter_padded` 与 `_all_gather_shards` 在
`world_size == 1` 时仍无条件寻找 NCCL/MPI，找不到就抛错。单卡下 reduce-scatter
没有可归约的对端，rank 0 的分片就是整个补齐后的梯度，all-gather 同理，两者都是
恒等操作。单进程 `fully_shard` 是受支持的用法——ms-swift 与 verl 的 FSDP2 路径在
CPU 上正是如此——不应因此失败。同文件的 `_globally_used_grads` 早已有相同的
`world_size <= 1` 短路。

### 修复

两个函数在 `world_size <= 1` 时直接返回输入。此外，单卡下若某次反向完全没有触及
某个 state 的参数，跳过该 state，而不是把零梯度归约进去、覆盖上一轮留下的
`true_fsdp_last_flat_grad`。多卡仍须参与，因为对端在等待。

### 验证

- `test_torch_compat_{fsdp2,optim,ops}.py` 依此顺序同跑 `67 passed`；
- 完整 CPU Torch 会话 `1491 passed, 278 skipped`，退出码 0；
- 真实双 GPU `tests/distributed/test_fsdp2_nccl.py` 两 rank 各 `4 passed`，
  确认多卡路径未受影响。

## AMP level 4 下任何带 Linear 的模型都会中止

`jt.flags.auto_mixed_precision_level` 是文档化的用户可见开关（`0` 不用 fp16，
`1-3` 保留级别，`4` 偏向 fp16 但 `sum`/`exp` 等仍用 fp32，`5` 追加 array 转换，
`6` 全部偏向 fp16）。在 CPU 上把它设为 `4`，一次普通的 `nn.Linear` 前向即中止：

```text
Execute fused operator failed. fused_op:(broadcast_to, broadcast_to,
binary.multiply, reduce.add)
[Input]: float32[16,8]  [Output]: float16[16,16]
var_relay.cc:48 Check failed p.first->size(1024) == p.second->size(512)
```

`mkl_matmul` 与 `cublas_matmul` 都以第一个操作数的 dtype 创建输出。AMP level 4 把
reduce 的输出改成 float16 而操作数仍是 float32，于是 relay 要分配两倍字节；
`VarRelayManager::add_relay_group` 断言两者尺寸相同，不匹配的后果是整个融合算子
中止，而不是放弃这次 relay。`MatmulTuner` 此前只检查操作数是浮点、CPU 上宽度为 4，
从未检查输出。

现在按两个 relay 算子自身的契约设卡：操作数宽度须相同（cublas 断言 dsize 相等，
mkl 断言均为 4），输出 dtype 须等于第一个操作数。不满足就不提出 relay，交给融合
kernel 写出所需 dtype。

| 场景 | 修复前 | 修复后 |
| --- | --- | --- |
| `nn.Linear` 前向 | 断言失败 | `float16` |
| `nn.Sequential` 前向 | 断言失败 | `float16` |
| 前向加 loss | 断言失败 | 正常 |
| 三步 SGD | 断言失败 | `0.86572 / 0.86328 / 0.86133` |

fp32 路径不受影响：参考轨迹 `0.8967 / 0.89419 / 0.89171`，`64x64` fp32 matmul 对
NumPy 最大误差 `7.629e-06`，原有 `relay0` 候选断言仍通过。

conv tuner 看似有同一处缺口，但 CPU 上 AMP level 4 的 Conv2d 前向、反向、
conv+BatchNorm 与 conv+Pool 均实测通过，没有可复现的问题，因此未改动。

新增两条回归：一条直接构造 AMP 下的元算子 matmul 并对拍 NumPy，一条覆盖真实
`nn.Linear` 训练。

## 两个 CPU 会话的当前结果

在 `f162d01c` 上，四类门禁均零失败（CUDA 与 NCCL 于 `99537948` 复验）：

| 门禁 | 结果 | 用时 |
| --- | --- | ---: |
| native CPU（`JITTOR_TORCH_SHIM=0`） | `728 passed, 699 skipped`，退出码 0 | `18m31s` |
| torch CPU（`JITTOR_TORCH_SHIM=1`） | `1496 passed, 278 skipped`，退出码 0 | `2m15s`（热 cache） |
| `nox -s cuda`（空 cache 冷构建） | 五组合计 `559 passed, 1 skipped`，退出码 0 | `2h` |
| `test_fsdp2_nccl.py`（真实双 GPU） | 两 rank 各 `4 passed` | `3m26s` |

CUDA 组的分解为 `97 passed, 1 skipped` / `6 passed` / `227 passed` / `2 passed` /
`227 passed`；该会话从空 cache 起，包含全部 CUDA 扩展的冷编译。

一处容易误判的现象记录在此：`tests/data/test_dataset.py` 的 worker 测试在本机与
CUDA 扩展编译（16 个并发编译进程）、notebook 门禁和 GPU 基准同时运行时会挂死，
且 `--timeout` 的 signal 方式打不断它。同一文件单独执行为 `8 passed, 5 skipped`，
用时 `17.69s`；空载重跑完整会话也全过。这是负载争抢下的 fork/handshake 饥饿，
不是代码回归——判定套件结果时不要与真实失败混为一谈。

## 过期的断言常量

`test_install_context.py` 把 torch 命名空间的模块数钉在 186，实际为 189。同一处的
`set(after) == set(baseline)` 通过，说明命名空间自身一致；多出的三个是
`_composable.fsdp` 一系有意发布的条目。哨兵保留，数字更新并写明何时才该改。

`test_torch_compat_fsdp2.py` 的 nccl 断言改为跟随 `jt.has_cuda`。

## TRELLIS BF16 GEMM 与 PyTorch 打平

按 `20260826-perf` 的算子 profile，`cublas_matmul`（BF16、`Trans_b:T`）占已 profile
算子总时间的 `65.5%`（`3844.6ms / 5867.7ms`）。取该 profile 中占比最高的八个形状，
在同一张 RTX 4090 上分别用 Jittor 与真实 PyTorch 计时，每形状 5 次预热、30 次测量：

| M | K | N | Jittor | PyTorch |
| ---: | ---: | ---: | ---: | ---: |
| 3540 | 1536 | 4608 | `333.8us` / `150.1` TFLOPS | `334.6us` / `149.8` TFLOPS |
| 3540 | 1536 | 8192 | `566.3us` / `157.3` TFLOPS | `566.7us` / `157.2` TFLOPS |
| 3540 | 4096 | 1536 | `301.5us` / `147.8` TFLOPS | `301.5us` / `147.7` TFLOPS |
| 3540 | 1536 | 1536 | `124.7us` / `134.0` TFLOPS | `124.4us` / `134.3` TFLOPS |
| 4096 | 1536 | 4608 | `376.3us` / `154.1` TFLOPS | `377.0us` / `153.8` TFLOPS |
| 4096 | 1536 | 8192 | `647.0us` / `159.3` TFLOPS | `641.6us` / `160.7` TFLOPS |
| 4096 | 4096 | 1536 | `326.7us` / `157.8` TFLOPS | `324.0us` / `159.1` TFLOPS |
| 1029 | 1536 | 3072 | `77.1us` / `126.0` TFLOPS | `76.5us` / `127.0` TFLOPS |

每个形状差异都在 `1%` 以内。因此 `1.093x` 的端到端差距不在 GEMM，剩余候选是
非 GEMM kernel（`1.81s`）与图构建/派发开销（约 `1.6s`），后续应从这两处入手，
而不是继续调 cuBLAS 计算类型或算法选择。

## TRELLIS 基线在空闲 GPU 上重测：`1.045x`，且差距已定位

`benchmark_jittor.sh` 与 `benchmark_torch.sh` 的 `CUDA_VISIBLE_DEVICES` 默认值都是
`1`，而本机 GPU 1 上有一个无关的常驻进程（`serve_g1_real_policy.py`，已运行 6.5
天，占用 `12262 MiB`）。此前记录的 `7.5149s` 对 `6.8778s` 即在该被争抢的卡上测得。

改到空闲的 GPU 4，同一条对齐 tape、同样的 1 次预热加 3 次测量、同样的
`--profile-pipeline --skip-glb`：

| 后端 | 三次测量 | 中位数 |
| --- | --- | ---: |
| Jittor | `6.8678 / 6.8390 / 6.8470s` | `6.8470s` |
| PyTorch | `6.5525 / 6.5499 / 6.5687s` | `6.5525s` |

比值 `1.045x`，差 `0.2945s`。此前的 `1.093x` 中有一部分是共享 GPU 带来的噪声，但
真实差距仍然存在。**今后 TRELLIS 基准必须显式指定空闲 GPU。**

### 差距按阶段分解

顶层阶段互不重叠，可以相加（数值为剔除预热后 3 次测量的均值）：

| 阶段 | Jittor | PyTorch | 差 | 比值 |
| --- | ---: | ---: | ---: | ---: |
| `preprocess_image` | `0.0365` | `0.0270` | `+0.0095` | `1.352x` |
| `get_cond` | `0.0490` | `0.0488` | `+0.0002` | `1.004x` |
| `sample_sparse_structure` | `2.3770` | `2.4813` | `-0.1043` | `0.958x` |
| `sample_shape_slat` | `2.5080` | `2.3239` | `+0.1841` | `1.079x` |
| `sample_tex_slat` | `1.4440` | `1.3137` | `+0.1303` | `1.099x` |
| `decode_latent` | `0.4360` | `0.3203` | `+0.1157` | `1.361x` |
| 合计 | `6.8505` | `6.5150` | `+0.3355` | |

稀疏结构采样上 Jittor 已经**更快**。差距集中在两处 SLat 采样（各约 `1.08-1.10x`）
与 decode（`1.361x`，比值最差）。

### decode 内部：调用级 profile 在这里不可用

调用级 profile（`--profile-decode-calls`）给出的比值分别是
`SparseUnetVaeDecoder.forward` `1.460x`、`SparseConvNeXtBlock3d.forward` `1.221x`、
`SparseResBlockC2S3d.forward` `1.535x`、`SparseChannel2Spatial.forward` `1.743x`、
`SparseLinear.forward` `2.308x`。**这些数字不能用来定位差距。**

原因在 profiler 自身：`CallProfiler.record` 在每个被包裹的调用前后都调用 `sync`。
对 PyTorch 这只是插一个 stream 同步，对 Jittor 却是强制刷图——正好废掉跨模块算子
融合，而那是 Jittor 的主要优化手段。包裹的粒度越细，惩罚越重，于是最小的模块得到
最难看的比值。`SparseLinear` 排在最差不是因为它慢，而是因为它最小。

同一张 GPU 上单独测这些原语可以确认：

| 原语 | Jittor | PyTorch |
| --- | ---: | ---: |
| linear `[1485494, 64] -> 64` | `1.668ms` | `2.974ms` |
| linear `[1485494, 64] -> 4` | `0.503ms` | `0.966ms` |
| linear `[3540, 1024] -> 1024` | `0.118ms` | `0.204ms` |
| gather `[28320, 512]` | `0.147ms` | `0.126ms` |
| gather `[132632, 256]` | `0.342ms` | `0.300ms` |

底层 kernel 上 Jittor 在大 linear 上明显更快，gather 慢约 `15%`。因此 decode 的
`1.361x` 不是 kernel 吞吐问题，而是每个算子的构图与派发开销——与 BF16 GEMM 那节的
结论一致。

阶段级 profile 仍然可用：它每轮只同步 6 次，代价可以忽略。后续定位应当使用不引入
同步的手段（Jittor 自己的算子 profile 只统计 kernel 时间），而不是继续细分包裹。

### 派发开销也不是原因

上一节把矛头指向"每算子的构图与派发开销"。直接测量否定了这个假设。在同一张 4090
上跑一个纯 launch-bound 的深 MLP（每层 linear 加 relu，深度 32，50 次测量）：

| 形状 | Jittor | PyTorch | 比值 |
| --- | ---: | ---: | ---: |
| depth 32, width 32, batch 1 | `15.6us/op` | `11.1us/op` | `1.41x` |
| depth 32, width 32, batch 64 | `13.5us/op` | `13.0us/op` | `1.04x` |
| depth 32, width 512, batch 64 | `14.2us/op` | `13.6us/op` | `1.04x` |
| depth 8, width 4096, batch 512 | `223.6us/op` | `220.3us/op` | `1.01x` |

只有最极端的 batch 1、width 32 才拉开到 `1.41x`，且绝对差仅 `4.5us`；其余形状在
`1.01-1.04x`。TRELLIS 的算子远大于此，因此不存在一笔均摊到每个算子的固定税。

于是三个候选依次被排除：GEMM 打平、per-op 派发打平、gather 只慢约 `15%`（在
TRELLIS 中 `getitem` 全部 kernel 时间仅 `28.7ms`，`15%` 不足 `4ms`）。剩下的
`0.2945s` 目前无法用现有工具进一步归因，需要不引入同步的细粒度手段。TRELLIS 性能
因此仍未接受，但排除掉的这三条可以省去后来者的重复工作。

## ViT 的缺口不在 CUDA GEMM

`2026-08-26-common-network-training-trajectories.md` 记 ViT 为 `1.33x`，并断定
「ViT 剩余缺口需要更强的 CUDA GEMM backend/algorithm」，依据是 profile 中 Jittor
每 step 的普通 GEMM 约 `27.97ms`、PyTorch 约 `16.11ms`。这个结论不成立。

### GEMM 本身在 fp32 与 TF32 下都打平

同一张空闲 4090，ViT-base 的四个主力形状（`M=1576`，即 batch 8 的 197 token）：

| M | K | N | Jittor TF32 off | PyTorch TF32 off | Jittor TF32 on | PyTorch TF32 on |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1576 | 768 | 2304 | `141.7us` | `142.3us` | `90.4us` | `90.0us` |
| 1576 | 768 | 3072 | `186.5us` | `194.6us` | `133.2us` | `132.5us` |
| 1576 | 3072 | 768 | `210.8us` | `211.8us` | `163.1us` | `162.6us` |
| 1576 | 768 | 768 | `57.8us` | `57.8us` | `45.0us` | `44.7us` |

三种转置布局同样打平（`133.7 / 133.0 / 133.5us` 对 `132.3 / 132.4 / 132.5us`），
而完整的 linear 前向加反向 Jittor 反而更快（`482.4us` 对 `533.5us`）。两侧脚本都
已开启 TF32，不存在配置不对称。

### Jittor 也没有多做 GEMM

反证：设 PyTorch 每 step 的 GEMM 为 `G`。若 Jittor 的 GEMM 工作量是两倍，则由
PyTorch 步长 `23.7ms = G + 非 GEMM` 可得 `G` 约 `19.7ms`，Jittor 步长至少
`39.4ms`；实测只有 `29.4ms`。按 FLOP 计算也一致：ViT-base 一步约 `803 GFLOP`
（12 层，每层前向 `22.3 GFLOP`，反向两倍），在实测的 `41-56 TFLOPS` 下 GEMM 下限
为 `14-19.5ms`，两侧都被它主导。

### 因此缺口在 GEMM 之外

空闲 GPU 上重测：Jittor `0.0294s`，PyTorch `0.0237s`，`1.24x`，差 `5.7ms`。单个
GEMM 打平、GEMM 工作量相同，这 `5.7ms` 只能落在 attention 的 batched matmul、
transpose、LayerNorm、逐元素算子与 host 间隙上。文档中「需要更强的 GEMM backend」
因此是误判——同一节记录的 cuBLASLt 实验反而退化（`132us` 到 `381us`），正是缺口
不在 GEMM 时应有的结果。

### 关于 jt.profile_scope 的绝对值

原结论所依据的 `27.97ms` 来自 `jt.profile_scope(2, 5)`。该 profiler 逐算子同步并
重跑，既加开销又阻止 kernel 重叠：同一个 ViT step，`profile_scope(0, 1)` 报出
`56.17ms` 的算子时间，而实测步长只有 `29.4ms`；把 rerun 由 1 改为 5，每种转置的
Count 由 `144` 变为 `288`。它的绝对时间与调用计数都不能直接当作 step 的真实值，
两侧 profiler 的数字更不可直接相减——PyTorch 侧脚本 profile 的是 `slots` 四步，
Jittor 侧只有一步。

## 剩余性能差距的共同成因：构图不与 GPU 执行重叠

用 nsys 对两侧同等地测量 ViT（同一张空闲 4090，各 10 次预热加 30 次计时）：

| | Jittor | PyTorch |
| --- | ---: | ---: |
| GPU kernel 总时间 | `961.7ms` | `953.7ms` |
| 其中 GEMM | `707.2ms` | `661.3ms` |
| 其中 fmha（PyTorch 的 memory-efficient attention） | — | `147.4ms` |
| 其他 kernel | `254.5ms` | `144.9ms` |
| kernel 数 | `32511` | `38026` |

**两侧的 GPU 计算量几乎相同（差 `0.8%`）。** 顺带一提，Jittor 用显式 batched GEMM
实现的注意力（`63.1ms`）比 PyTorch 在 fp32 下选中的 fmha（`147.4ms`）更省。

差距因此不在 kernel，而在时间线。把一步拆成两段：

| 阶段 | Jittor | PyTorch |
| --- | ---: | ---: |
| Python 侧（Jittor 构图 / PyTorch 发射） | `9.6ms`（GPU 全程空闲） | `17.7ms`（GPU 已在跑） |
| 执行与等待 | `19.7ms` | `5.9ms` |
| 合计 | `29.4ms` | `23.6ms` |

结论与直觉相反：**Jittor 的 Python 开销比 PyTorch 低 `45%`，执行阶段 `19.7ms` 也比
PyTorch 的整步 `23.6ms` 快。** 全部 `5.7ms` 的劣势来自那 `9.6ms` 构图期间 GPU 无事
可做——lazy 图在 sync 之前不向设备发射任何东西，而 PyTorch 每个算子即时入队，
Python 时间与 GPU 执行天然重叠。

佐证：把同步频率降低，差距按预期缩小——每步同步 `0.0295s`，每两步 `0.0272s`，
每四步 `0.0254s`，逼近 `24.0ms` 的 kernel 下限。

若构图能与执行重叠，一步应接近 `max(9.6, 19.7)`，即约 `19.7ms`，反而比 PyTorch
快约 `17%`。这同一成因也适用于 TRELLIS 与 transformer 训练的剩余差距，此前对这些
差距的 kernel 级归因（「需要更强的 GEMM backend」）方向有误。

### 可重叠的是跨步，不是步内

两次尝试划出了边界。在前向中每隔 `1/2/3/4/6` 层插入一次 `jt.sync`（其
`device_sync` 默认为 false，发射后即返回）让 GPU 提前开工：中位数全部为 `0.0294s`，
与不插入完全一致。原因是 `Tensor.backward` 在构建反向图之前已经先 sync 了一次前向
（用以物化自定义 CUDA 扩展的乱序写入），前向本来就与反向构图重叠了。

而降低同步频率始终有效（每步 `0.0295s`、每两步 `0.0272s`、每四步 `0.0254s`）。
两者合起来说明：可以重叠的是**上一步的反向执行与下一步的构图**，而非步内。真实
训练循环在这里必须拿到梯度才能更新，这正是结构差别所在——PyTorch 的反向全程异步，
Python 立即继续；Jittor 在 sync 处让主机停下来等。

### 空隙的确切位置

nsys 的 kernel 时间线给出了确切结论：Jittor 每步恰好有**一个**约 `6.23ms` 的连续
空隙（30 步对应 30 个 `>1ms` 的空隙），位置固定——紧接在前向第一个 kernel
（patch embedding 的 cuDNN 卷积）之前。PyTorch 没有这样的大空隙，它的空闲分散在
`8971` 个 `10-100us` 的小间隙里，属于 nsys 自身的插桩开销。

即：不是发射慢，而是每步开头有一次成块的等待——lazy 图在 sync 之前不向设备发射
任何东西。

### 可选的执行流水线

`Module.set_execution_pipelining(n)` 在模块边界上，当距上次发射新增的算子数达到
`n` 时调用一次 `jt.sync`（`device_sync` 为 false，发射后即返回），让 GPU 在 Python
继续构图时先开工。默认 `0` 即关闭。

触发条件必须用"新增算子数"而非"存活算子数"：后者包含图仍持有的一切，一旦越过
阈值就会每次调用都发射，反而没有任何两个算子能融合——实测只有 `3.7%` 收益。

ViT-base（batch 8，空闲 4090）每步中位数：

| 阈值 | 中位数 | 相对 PyTorch `0.0237s` |
| ---: | ---: | ---: |
| `0`（关闭） | `0.0295s` | `1.245x` |
| `100` | `0.0263s` | `1.110x` |
| `200` | `0.0260s` | `1.097x` |
| `400` | `0.0263s` | `1.110x` |
| `800` | `0.0284s` | `1.198x` |
| `1600` | `0.0296s` | `1.249x` |

取 `200` 时快 `11.9%`。代价是融合：发射点两侧的算子无法融合，浮点累加分组随之
改变，各档 loss 相对差不超过 `8.08e-06`。因此默认关闭。

ViT 由此从 `1.24x` 降到 `1.097x`，仍未达到"不慢于 PyTorch"的验收线。

### 两处 kernel 缺陷：Jittor 的计算量现已低于 PyTorch

排查空隙时发现两个与空隙无关、但各自很严重的 kernel 缺陷。

**全量归约的原子争抢。** 生成的整张量 reduce 让每个线程都对同一个输出元素做
`atomicAdd`；按代码生成器选取的线程数，那是二十六万次对同一地址的原子操作，完全
串行。float32 的 `sum` 无论 1.21M 还是 8.4M 元素都是约 `727us`——耗时跟的是原子
次数而非数据量，同机 PyTorch 为 `10-11us`。逐元素运算本身两边持平。新增的
`full_reduce_cuda` 改为两级折叠（每 block 一个 partial，再由单 block 汇总），
`sum` 降到 `38.1us`，且因为去掉了原子，结果逐次可复现；相对 float64 参考的误差
也由 `2.50e-05` 降到 `1.68e-07`。

**LayerNorm 参数梯度的非合并访存。** 原实现每个 block 负责一个 channel、线程沿
row 走，warp 内相邻 lane 的地址相隔 `hidden` 个 float，各自落在不同 cache line。
改为 lane 沿 channel、row 分摊到 `threadIdx.y` 后，一个 warp 的读取是一次连续事务，
且 weight/bias 由同一遍扫描得出。

同一段 40 步 ViT-base 的 nsys 前后对比：

| | 修复前 | 修复后 |
| --- | ---: | ---: |
| GPU kernel 总时间 | `961.7ms` | `895.7ms` |
| 每步 | `24.04ms` | `22.39ms` |
| `layer_norm_backward_affine` | `43.6ms` | `7.1ms` |
| 原子归约 kernel | `27.2ms` | `0.2ms`（新 kernel） |

同机 PyTorch 每步 `23.84ms`。**至此 Jittor 的 GPU 计算量已低于 PyTorch**，
剩下的差距完全在主机与设备的重叠上。

### 当前 ViT 数值

kernel 修复后重新扫描流水线阈值（空闲 4090，batch 8）：

| 阈值 | 每步中位数 | 相对 PyTorch `0.0237s` |
| ---: | ---: | ---: |
| `0`（关闭） | `0.0279s` | `1.177x` |
| `100` | `0.0252s` | `1.063x` |
| `200` | `0.0249s` | `1.051x` |
| `300` | `0.0247s` | `1.042x` |
| `400` | `0.0248s` | `1.046x` |
| `800` | `0.0268s` | `1.131x` |

ViT 由本轮开始的 `1.24x` 降到 `1.042x`。kernel 下限对应 `0.94x`，即余下约
`2.4ms` 仍是没能重叠的主机时间；验收线（不慢于 PyTorch）尚未达到。归约与
LayerNorm 两处修复后，各阈值的 loss 逐位相同，因为新归约是确定性的。

TRELLIS 同期由 `1.0443x` 微降到约 `1.040x`：它没有 ViT 那种成块的单点空隙，
流水线对它基本无效。

### 余下 2.2ms 空隙的确切位置

开启流水线（阈值 300）后把一步拆开计时：

| 阶段 | 阈值 0 | 阈值 300 |
| --- | ---: | ---: |
| 前向构图 | `4.50ms` | `6.34ms` |
| 反向构图 | `4.95ms` | `3.39ms` |
| 末尾同步 | `18.35ms` | `14.89ms` |
| 合计 | `27.80ms` | `24.62ms` |

GPU kernel 为 `22.39ms`，合计 `24.62ms`，故仍有约 `2.2ms` GPU 空闲。按各段时长
推算，它落在**前向 kernel 跑完到反向图建好之间**：前向的 GPU 工作约 `7.5ms`，
而主机在 `6.34ms` 处结束前向构图后还要花 `3.39ms` 建反向图，这段没有任何东西可
发射。`loss.backward()` 经 `jt.grad` 一次性建出整张反向图，中间没有模块边界，
Python 侧的钩子够不着。

### 发射时机已到最优：两次调度实验都无收益

先量出计数的实际含义：`number_of_lived_ops` 统计的是存活算子而非待执行算子，步与
步之间几乎不变（`757` 对 `758`）。逐层追踪一步的轨迹：每个 encoder 层加 `72` 个
算子，12 层后到 `1639`，**反向只加 `36` 个**（`jt.grad` 的时间花在图分析而不是建
算子上），同步后回到 `758`。

据此，阈值 300 恰好是每 `4.2` 层发射一次，末尾正好留下 4 层的工作交给
`Tensor.backward` 里的那次 sync 释放，用来掩护反向构图——这解释了它为何最优。

两次改进尝试都没有收益，均已回退：

- 图排空后让第一次 flush 用八分之一门槛（假设空隙在步开头）：各阈值数值与不加时
  逐位相同，说明开头并非瓶颈；
- 发射间隔随迭代递增（前期早发射、后期少发射以留更大尾部）：`0.0249s`，比固定
  阈值的 `0.0247s` 略差。

### 轴向归约不是同一个问题

ViT 反向中最大的非 GEMM kernel 是沿前面所有维归约到最后一维的 `reduce`（每个
Linear 的 bias 梯度，每步 72 次、`0.861ms`），生成的代码同样用 `atomicAdd`，看起来
和整张量归约是同一个缺陷。按同样的合并访存 tiling 写了一版快路径，数值正确（相对
float64 参考误差约 `1.4e-07`），但实测**比原生慢 2.4 到 5 倍**：

| 形状 | 快路径 | 原生 |
| --- | ---: | ---: |
| `1576x768` | `35.0us` | `14.8us` |
| `8192x1024` | `30.0us` | `12.6us` |
| `256x4096` | `41.0us` | `8.1us` |

原因是两者的争抢程度完全不同：整张量归约把二十六万次原子压在**同一个地址**上，而
轴向归约分摊到几百上千个输出，每个输出的争抢有限，原生路径本来就接近带宽上限；
再加上 `jt.Function` 与 `jt.code` 的构造开销，快路径反而更慢。该改动已回退。

顺带确认：ViT 反向里的这个归约由 `jt.grad` 内部生成，并不经过 `Var.sum`，Python
层的钩子本来也拦不到它。

### 模块树遍历：单遍改写会踩掉 dfs 的多态

`parameters()` 与 `zero_grad()` 各约 `0.6ms`，PyTorch 对应为 `0.215/0.243ms`；
两者每步各调用一次，合计约 `0.7ms` 的差，正是需要削减的量级。profile 显示每次
`parameters()` 里 `isinstance` 被调用 `3923` 次、`startswith` `1240` 次，因为每个
模块的 `__dict__` 被扫两遍——`dfs` 挑子模块一遍，callback 挑 Var 一遍。

据此写了一版专用的单遍递归取代 `dfs` 加 callback，`parameters()` 由 `0.593ms` 降到
`0.024ms`。但它是错的：`Module.dfs` 是多态的，`ModuleList` 与 `Sequential` 覆盖了它
并从 `layers` 而非 `__dict__` 取子模块，绕开 `dfs` 就丢掉了这个覆盖——ViT-base 的
参数数由 `200` 塌成 `8`。回归测试立刻抓到，改动已回退。

要保留这项优化就必须仍然走 `dfs`，也就回到了两遍扫描的结构；真正的做法是让 `dfs`
一次分类出子模块与其余条目并把结果交给 callback，那是对一个公共遍历接口的改动，
影响本文件中十余处调用点。

### 稳态吞吐：此前的比值来自每步强制同步的测法

前面所有 ViT 数值都来自一个每步调用 `jt.sync_all(True)` 的计时循环。真实训练循环
不会这样：PyTorch 的 `backward`/`optimizer.step` 全程异步，Jittor 的
`jt.sync(grads)` 同样不等设备。每步强制排空是测量伪影，而且对惰性构图的惩罚更重
——它恰好把上一轮的执行与下一轮的构图之间的重叠切断，而那正是本文档量出的
`2.298ms` 步边界空隙的来源。

改用稳态吞吐重测：预热 10 步，之后连跑 30 步、末尾只同步一次，取三轮最好值，两侧
同一协议、同一优化器（SGD `lr=1e-4`）、同样计入优化器更新。

| 网络 | Jittor | PyTorch | 比值 |
| --- | ---: | ---: | ---: |
| GPT-2 | `0.0362s` | `0.0415s` | `0.872x` |
| ConvNet | `0.0140s` | `0.0156s` | `0.897x` |
| ViT | `0.0242s` | `0.0246s` | `0.984x` |
| Llama | `0.0384s` | `0.0392s` | `0.980x` |

ConvNet 起初测得 `1.006x`，原因是配置不对称：参考侧设了
`torch.backends.cudnn.benchmark = True`，而 Jittor 侧没有对应设置。它不是 `jt.flags`
上的开关（我先前写的 `jt.flags.cudnn_benchmark` 并不存在，静默无效），入口是
`jt.cudnn.set_benchmark(1)`，且必须在 cudnn 扩展加载之后调用。补上之后 Jittor 为
`0.0140s`，三次完全一致。

连同此前已接受的 UNet `0.79x`，四个常见网络在稳态吞吐下均不慢于 PyTorch，
todo 第五项的真实规模性能门禁据此达成。

TRELLIS 不受这一项影响：它是单次端到端推理，本来就没有每步同步的伪影。开启
cuDNN autotune 与流水线后 Jittor 三次为 `6.8035 / 6.8006 / 6.8143s`。

参考侧是否被削弱也查过了，结论对本项不利、如实记录：脚本默认
`--torch-float32-matmul-precision highest`，即 PyTorch 的 float32 matmul 走纯 fp32，
而 Jittor 的 `tensorcore=2` 让 float32 matmul 以 bf16 计算——这一处不对称本来对
Jittor 有利。实测把参考侧改成 `high`（TF32）并不会更快（`6.5576s` 对 `6.5139s`），
因为 TRELLIS 以 bf16 为主，float32 matmul 只占算子时间的 `0.9%`。参考侧没有被削弱。

但这一轮 PyTorch 跑出了比先前记录的 `6.5525s` 更好的成绩（`6.5127 / 6.5139 /
6.5422s`）。按最好对最好，`6.8006s` 对 `6.5127s`，即 `1.044x`——比先前报出的
`1.038x` 略差，以此为准。
它的算子时间已由基线的 `5867.7ms` 降到 `4869.8ms`，其中 bf16 GEMM 占 `67%` 且与
PyTorch 打平，其余分散在两万八千次小的融合算子上，没有单点杠杆。

工作量等价性单独核对过：两侧从同一份权重出发（200 个参数逐一加载），五步 SGD 的
loss 轨迹为

    Jittor  -2210.5356  -100857.84  -331381.88  -30094.90  -851.2244
    PyTorch -2210.0493  -100866.73  -331384.63  -30113.85  -851.1620

相对差 `8.3e-06` 到 `6.3e-04`，与 TF32 GEMM 在这条量级剧烈变化的轨迹上的累积一致。

两个指标都记录在案：稳态吞吐是训练关心的量，而每步强制同步下的单步延迟仍为
`1.042x`。前者说明 ViT 与 GPT-2 已不慢于 PyTorch，ConvNet 差 `0.6%`。

### CPU 的差距：长融合算子没有被并行化

CPU UNet 此前记为「慢于 PyTorch、性能未接受」，本轮量出具体数字与成因。同一协议
（预热 3 步后连跑 10 步，取两轮最好值）下 Jittor 为 `3.4038s`，PyTorch 为
`0.7051s`，即慢 `4.83x`。

一步的算子 profile（合计 `5856ms`）显示差距高度集中：

| 每步 | 调用数 | 每次 | 算子 |
| ---: | ---: | ---: | --- |
| `829.5ms` | 12 | `69.1ms` | `array + broadcast_to + ... + reduce` 融合 |
| `628.6ms` | 12 | `52.4ms` | 同类 |
| `602.8ms` | 12 | `50.2ms` | 同类 |
| `134.5ms` | 12 | `11.2ms` | 同类 |
| `529.2ms` | 100 | `5.3ms` | `mkl_conv_backward_x` |
| `484.9ms` | 100 | `4.8ms` | `mkl_conv` |

四个同类融合算子共 48 次调用占 `2195ms`，即 `37.5%`，比全部卷积（`1369ms`）还多。
读它们生成的源码可知原因：**里面一个线程构造都没有**——`thread`、`parallel`、`tid`
的出现次数为 `0`，是一重普通的串行嵌套循环。这台机器有 32 个物理核。

内容上它是 UNet 注意力块的 softmax 反向（`exp`、除法、沿末轴归约）融合成 13 个
算子的一个 kernel。

并行化与否有规律可循：该 CPU cache 中 2091 个生成 kernel 只有 45 个含
`#pragma omp`，而它们的 op key 都很短——单个 `binary.add`、单个 `reduce`。也就是说
`ParallelPass` 会并行化短融合，但对这种十余个算子的长融合放弃了。该 pass 与
`AtomicTunerPass` 一样在 `data.o` 中没有源码，无法从仓库侧调整它的判据。

试过用 `jt.flags.no_fuse` 把融合整体关掉以换取并行：每个算子都要重新 JIT 编译，
在 2400 秒内没有跑完；何况全局关闭融合本就不是可交付的配置。

结论：CPU 真实规模性能的主要成因已定位到「长融合算子不被并行化」，修法在闭源的
`ParallelPass` 里。

### 空隙的分布，以及为什么调度改不动它

开启流水线（阈值 300）后用 nsys 量出每步的空闲分布——每类空隙每步恰好出现一次：

| 每步 | 位置 |
| ---: | --- |
| `2.298ms` | 上一轮最后一个 kernel 到本轮第一个 kernel（步边界） |
| `0.624ms` | loss 归约到反向第一个 kernel（反向构图窗口） |
| `0.427ms` | 前向中段 |

步边界占七成，起因是固定阈值必须先攒够一个阈值的算子才发射——300 个算子相当于
四个 transformer 层的 Python 时间，设备在这段时间里没有任何东西可做。

据此做了三次调度改进，全部无收益，均已回退：

| 变体 | 阈值 300 时 |
| --- | ---: |
| 固定阈值 + weak sync（已提交的版本） | `0.0247s` |
| 开头一次低门槛发射 | `0.0251s` |
| 开头三次递增门槛 | `0.0250s` |
| 关闭 weak sync（只发射本模块的依赖） | `0.0250s` |

需要说明，先前那次"排空后低门槛"的实验其实是死代码：后来量到
`number_of_lived_ops` 在步间稳定在 `757/758`，`lived < mark` 从不成立。这一轮改用
`Tensor.backward` 作为迭代边界信号（训练循环每轮恰好调用一次），信号是可靠的，
结果仍然没有改善。

原因是每次发射都有固定的主机成本，而收益又与一次释放多少工作成正比：关掉
weak sync 后每次只发射本模块的依赖，设备更快再次空转，结果反而更差；更小的阈值
（`25`、`50`）同理。两端都被验证过，`0.0247s` 是这条路的下界。

### 为什么剩下的部分需要引擎改动

要消掉这 `2.2ms`，只能让反向图在构建过程中分批发射。`jittor::grad` 在整个构建过程
中持有 `vector<Node*> sorted` 与 `vector<Var*> gvars` 这些裸指针，并且**把自己的
索引存放在 `Var::custom_data` 里，而执行器也使用同一个字段**。中途调用执行器会踩坏
这套索引记账，还会释放它仍在遍历的算子——这与先前在 `add_hold_vars` 里异步发射
导致堆损坏是同一类问题。因此这属于执行模型的设计改动，不是一处补丁，本轮不在没有
充分验证的前提下引入。

### 一次失败的尝试及其原因

试过在 `add_hold_vars` 中按待执行算子数触发一次 `device_sync=false` 的异步发射，
让已就绪的图先上 GPU。它以堆损坏告终（`free(): double free detected in tcache 2`），
关掉 `weak_sync`（避免重入改写 `hold_vars`）也一样。执行器无法在 VarHolder 尚未构造
完成时被重入，这不是一处小补丁能解决的，需要正经的流水线设计。改动已回退，未留在
仓库中。

## NCCL 引导诊断

本机的 P2P 矩阵全为 `CNS`，NCCL 的 p2p transport 报
`Cuda failure 'peer access is not supported between these two devices'` 并直接失败，
不回退共享内存。它抛出的 `unhandled cuda error` 既未说明原因也未说明解法；NCCL
自己的解释走 stderr，被 pytest 捕获后用户只看到 `Fatal Python error: Aborted`。

两处 `ncclCommInitRank` 现统一经由 `init_nccl_comm`，失败时先记录 NCCL 的错误
字符串并指出 `NCCL_P2P_DISABLE=1` 与 `NCCL_DEBUG=INFO`，再让原错误传播。实测
命中该路径会打印提示；设 `NCCL_P2P_DISABLE=1` 后双 rank 初始化成功。

该 P2P 限制属于本机当前驱动状态，不是 Jittor 缺陷；多卡门禁在此机器上需要
`NCCL_P2P_DISABLE=1`。

## 新增原生 notebook

两个 topic 均登记进 `_TOPICS` 与 `_SMOKE_TOPICS`，因此结构契约与离线 CPU
smoke 都覆盖它们；`tests/integration/test_notebooks.py` `5 passed`。

`vit_training.md`：patch embedding、class token、可学习位置编码、完整写出的多头
自注意力、pre-norm 块，三次 AdamW 更新。loss 轨迹 `1.78485 / 1.55026 / 1.38720`，
注意力每行和为 1，恢复 state 后 logits 完全一致。

`gpt2_training.md`：decoder-only 语言模型，重点是因果掩码为何必须下三角，以及
key/value 缓存为何只改变开销。任务为 `a,b,a,b,...`，除第一个位置外全部可解析确定。
120 步 AdamW 后 loss 由 `2.7793` 降到 `0.0860`，初值与 `ln(16) = 2.7726` 相符；
第一个位置之后准确率 `100%`；带缓存的贪心生成精确得到 `[2, 3, 2, 3]`；缓存与整段
重算的 logits 最大差 `1.04e-6`。

权重共享下 embedding 初始化单独说明并落到代码：logit 是同一矩阵两行的内积，方差
为 1 时初始 loss 约 18 而非 2.77，故按 GPT-2 取 `0.02`。

## 复现

```bash
# CPU Torch 会话
JITTOR_TEST_DEVICES=cpu JITTOR_TORCH_SHIM=1 REAL_TORCH_SITE= \
  python -m pytest tests/compat/torch -q

# 多卡 FSDP2（本机需要 NCCL_P2P_DISABLE=1）
NCCL_P2P_DISABLE=1 JITTOR_NCCL_WORLD_SIZE=2 python -m nox -s nccl

# notebook 契约与离线 smoke
python -m pytest tests/integration/test_notebooks.py -q
```

GEMM 基准脚本、算子 profile 与原始日志留在 `$JITTOR_LAB_ROOT` 之下，不纳入仓库。

## 遗留

- TRELLIS 端到端仍为 `1.093x`，方向已收窄到非 GEMM kernel 与派发开销；
- 真实 NPU、ROCm 设备套件；
- Transformer 训练与 CPU UNet/MMCV 的真实规模性能差距。
