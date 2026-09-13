# batched Linear 的两个展平节点：去掉之后图小了 13%

- 状态：已实现并测得；真实网络对拍与逐算子对拍在跑，结论另附
- 日期：2026-09-12
- 基线提交：`2cd9d6d0`（本轮改动为 `077e97ad`）
- 分支：`perf/matmul-rank`
- Owner：Jittor 核心维护者
- 复查触发：`cublas_matmul` 的形状契约变化、`matmul`/`matmul_transpose` 的路由变化，
  或 CPU 侧也要走同一条路时

## 问题

一个 transformer decode 步（b1 s1 d512 8 层，开自动微分）建 **408** 个图节点，
而同结构的 eager PyTorch 只分派约 318 次；主机路径慢约 2.9x，且这一步的 **96.9%**
是建图。节点普查显示其中 **80 个是 `reshape`**，而 `reshape` 是纯视图：
`is_storage_view()` 为真、不生成任何 kernel、`infer_shape` 之外几乎不做事。

## 这 80 个里的 64 个来自一处

`matmul` 与 `matmul_transpose` 把 rank>2 的左操作数送进二维 cuBLAS kernel 的办法是
先展平、算完再还原：

```python
aa = a.reshape((-1, a.shape[-1]))
cc = jt.nn.matmul_transpose(aa, b)
return cc.reshape(a.shape[:-1] + (-1,))
```

每次两个节点，而 `nn.Linear` 每次前向都付这一对。**`nn.Linear` 走的是
`matmul_transpose`，不是 `matmul`**——这一点最初找错了地方，改 `matmul` 时节点数
一个没少。本测例 4 个 Linear × 8 层 × 2 = 64，加上模型里显式写的 2 × 8 = 16，
正好是 80。该分支自带注释 `TODO:ugly implementation for tuner`。

## 改动

一个稠密行主序的 `(d0,..,dn-1,m)` **就是**它的 `(d0*..*dn-1,m)` 展平：同一个指针、
同一个 leading dimension，cuBLAS 看到的也只有展平后的 extent。所以让
`CublasMatmulOp` 接受它已有的 rank，契约是：

- 每个操作数把前导维压平成二维（`rows = prod(shape[:-1])`, `cols = shape[-1]`），
  套用原有的二维语义；
- 输出取贡献行的那一侧的前导维：`trans_a` 为假时输出是 `a.shape[:-1] + [k]`；
  为真时行来自 `a` 的最后一维，输出退回 rank 2，这正是 b 的梯度
  `matmul(a, dout, 1, 0)` 需要的形状。

`a` 必须真的稠密，展平才描述得了它，所以 strided 视图保留原来的 reshape 路线。
前向的 rank-2 行为逐字不变（展平是恒等）。

**`rank 1` 是这里唯一的坑。** 它也会走到 `matmul_transpose` 的同一个分支，而它的展平
`(1,m)` 是 reshape 要**补出来**的一行、不是 kernel 能从 buffer 上读出来的 rank。
第一版用 `len(a.shape) != 2` 判断就踩了，`tests/ops/test_matmul.py::test_linear1d` 抓到，
改成 `> 2`。

### 为什么这不是新引入的设备分叉

先前这项被我自己以「只能在 CUDA 侧改、会让 CPU 与 CUDA 的图形状不一致」为由否决过。
复查后这个理由不成立：`matmul` 的图形状**本来就依设备而定**——`len_a == 2 and len_b == 2`
走 `_matmul_2d_cublas`（CUDA 专有），`len_a >= 3 and len_a == len_b` 走
`select_kernel("batched_matmul", ...)`（按设备选），都选不上才落到通用的
broadcast+乘+求和路径。本改动是在既有的设备专有快路径上少建两个节点，
不是新增一条分叉。CPU 侧仍走 reshape。

## 结果

| | 改前 | 改后 |
| --- | ---: | ---: |
| 一步节点数（开自动微分） | 408 | **354**（−13.2%） |
| 其中 `reshape` | 80 | 16 |
| decode b1s1 d512 8层 | 4.93 ms | **4.45 ms（1.10x）** |
| 同上，`jt.no_grad()` | 4.13 ms | **3.77 ms（1.09x）** |
| prefill b1s32 d512 8层 | 5.39 ms | **4.91 ms（1.09x）** |
| mlp / cnn / transformer（infer 与 train，大 batch） | — | **1.00x** |

decode 与 prefill 各测三轮，三轮都复现。

**整网那一行是结论的一半，不是噪声**：MLP、CNN 和大 batch transformer 由 cublas/cudnn
主导，少掉的 54 个节点在主机侧，被设备执行重叠掉，没有可测变化——也没有回退。
**这项改动只在主机是关键路径时兑现**（decode、prefill、小模型、大量小算子）。

## 剩下的节点在哪

改动后一步 354 个节点：

| 类型 | 个数 | 占比 | 是什么 |
| --- | ---: | ---: | --- |
| `binary` | 88 | 24.9% | 真实逐元素运算 |
| `broadcast_to` | 64 | 18.1% | 32 个配对标量常量，32 个是 Linear 的偏置广播 |
| `array` | 32 | 9.0% | 标量常量 |
| `cublas_matmul` | 32 | 9.0% | 真实 GEMM |
| `tape` + `tapes` | 55 | 15.5% | 自动微分记账；推理用 `jt.no_grad()` 不会有 |
| `getitem` | 24 | 6.8% | qkv 的三次切分，视图 |
| `reshape` | 16 | 4.5% | 模型里显式写的 |
| `fuse_transpose` | 16 | 4.5% | 注意力的两次转置 |

下一步的候选，按实测排序：

1. **标量常量仍是两个节点**（`array` + `broadcast_to`，共 64 个 = 18%）。
   逐段实测：binary 本身 5.32 µs、expand 节点 +1.99 µs、新建 array 节点 +2.91 µs。
   一个「按形状产常量」的单算子能把它降到一个节点，但要新增一个 op（注册、
   codegen、grad），预计只值 1.5-2%，并且**不能顺手让 `jt.zeros/ones/full` 也用它**
   ——那会重新引入它们缓冲区不是自己形状的静默错误（见
   `2026-09-12-cuda-metaop-launch-index-scalar.md`）。
2. **纯视图仍各付一个完整图节点**（`getitem` 24 + `reshape` 16 + `fuse_transpose` 16
   = 15.8%）。要整体抹掉需要让视图成为 Var 的属性而不是一个 Op，
   而 Op 节点承载的是自动微分的边——对图模型的根本改动。
3. **单节点成本本身**：一步 354 个节点对 4.588 ms 是每节点约 13 µs，而孤立测一个
   binary 的建图是 5.31 µs。余下的在 Python 层（`nn.Module` 调用、functional 包装、
   `_runtime/dispatch.py`）。改动后 cProfile 总时间 0.500 → 0.387 s（−23%），
   但剩下的成本已经摊平，没有单一主导项：最大一项是 `nn/_bindings.py` 的 binary
   包装占 19%，而它的 tottime 里大部分是它调用的 C++ 建图本身。

## 验证

- `tests/ops/test_matmul_higher_rank.py`（随本改动新增）：各 rank 的前向、
  `matmul_transpose`（Linear 用的拼写）、**rank 1**、strided 左操作数的两种形态
  （切片与转置视图）、两个梯度、batched `nn.Linear` 的前向与三个梯度、
  float32/float64 dtype，CPU 与 CUDA 各一遍——**15 passed / 30 subtests**，
  float64 对 numpy 逐位相等。
- `tests/ops/test_matmul.py` 与 `tests/nn/test_bmm.py`：失败集合与改动前**逐 nodeid
  相同（各 7 条）**。rank-1 的那条 bug 就是这里抓到的。
- 真实网络的前向/反向对拍（`tests/models/test_network_parity.py` 与
  `test_network_training_parity.py`，ResNet/ViT/GPT-2/diffusion UNet，
  对独立二进制 PyTorch）与逐算子 CPU/CUDA 对拍（约 227 个生成用例）在两棵树上运行中。
  注意这两个对拍需要 `REAL_TORCH_SITE` 指向真 PyTorch 的 site-packages，
  不设会 16 个全部 skip 而门禁照样报绿。
- **未跑**：`tools/run_test_suite.py` 的完整维护口径、ROCm、NPU。本改动只影响
  CUDA 的 `cublas_matmul` 与 `matmul`/`matmul_transpose` 的路由；CPU 走原路线。

## 复现

```bash
export JITTOR_LAB_ROOT=/root/jittor-lab
source $JITTOR_LAB_ROOT/_state/metaop/env-mm.sh
cd $WT
CUDA_VISIBLE_DEVICES=<一张卡> $PY -m pytest -q tests/ops/test_matmul_higher_rank.py
CUDA_VISIBLE_DEVICES=<一张卡> $PY $JITTOR_LAB_ROOT/metaop-perf/bench_decode.py
```

节点普查用 `jittor_core.number_of_lived_ops()` 的差值加 `jt.dump_all_graphs()`，
在**不同步**的情况下建一步的图；`jt.no_grad()` 下这个计数不可用（图会被及时释放，
dump 只看到碎片），那里只能比时间。
