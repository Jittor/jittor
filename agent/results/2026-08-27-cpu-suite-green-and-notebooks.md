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

## 两个 CPU 会话的当前结果

在 `99537948` 上，两个会话均零失败：

| 门禁 | 结果 | 用时 |
| --- | --- | ---: |
| native CPU（`JITTOR_TORCH_SHIM=0`） | `726 passed, 699 skipped`，退出码 0 | `18m12s` |
| torch CPU（`JITTOR_TORCH_SHIM=1`） | `1491 passed, 278 skipped`，退出码 0 | `2m27s`（热 cache） |
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

### decode 内部

调用级 profile（同样剔除预热）显示 decode 的差距并不弥散，而是几个模块：

| 调用 | Jittor | PyTorch | 比值 | 每轮调用数 | 每轮差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `SparseUnetVaeDecoder.forward` | `287.7ms` | `197.0ms` | `1.460x` | 2 | `+0.1813s` |
| `SparseConvNeXtBlock3d.forward` | `7.355ms` | `6.025ms` | `1.221x` | 64 | `+0.0851s` |
| `SparseResBlockC2S3d.forward` | `21.94ms` | `14.29ms` | `1.535x` | 8 | `+0.0612s` |
| `SparseChannel2Spatial.forward` | `1.787ms` | `1.026ms` | `1.743x` | 16 | `+0.0122s` |
| `SparseLinear.forward` | `0.830ms` | `0.360ms` | `2.308x` | 8 | `+0.0038s` |

（这些调用互相嵌套，不能相加。）`SparseChannel2Spatial` 的热点是缓存命中路径上的
一次 gather——`x_feats[idx * factor ** DIM + subidx]`；`SparseLinear` 是稀疏特征上的
一次 linear，`2.308x` 是全部条目中比值最差的一项，最值得单独追查。

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
