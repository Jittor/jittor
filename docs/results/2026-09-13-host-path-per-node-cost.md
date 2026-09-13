# 主机受限步长：把每算子的 Python 开销从图构建里拿掉

- Status: Done（Python 路径）；C++ 侧的每节点成本与 `perf/matmul-rank` 仍在查
- Date: 2026-09-13
- Baseline commit: `48055a75`（分支 `perf/host-path`），其下的基线是 `d40a2e97`
- Owner: 元算子/计算图性能
- Review when: 前端再加一层派发，或 `auto_flush_ops` 的默认值变化

## 问题

batch 1 的 decode 是**主机受限**的：一步构 354 个节点要 2.99 ms，而一个裸的
binary 元算子只要 1.60 us。354 × 1.60 us = 0.57 ms，也就是说一步里有 2.4 ms
不是建节点，是节点之上的 Python。

先确认这不是图变大导致的：把 K 个 binary 串成一条未执行的链，逐 K 测每个
算子的构建时间，从 K=1 到 K=1024 稳定在 1.57–1.63 us，释放时间 0.12 us/算子。
**建节点的代价与待执行图的大小无关**，所以 8 倍的差额全在前端。

## 测量方法（含一个会让结论完全失效的陷阱）

`jt.flags.auto_flush_ops` 在 CUDA 上默认 128：每创建这么多算子，执行器就
**遍历所有存活的 VarHolder** 并把待执行的全部发射出去。于是

- 一段"只构图、不同步"的计时里其实混进了核函数发射；
- 而且那次遍历是 O(存活 holder 数)，在 `out = [fn() for _ in range(N)]`
  这种把 N 个结果留在列表里的微基准里，代价随 N 增长——测出来的
  "每次调用多少微秒"会随 N 变化。

同一批量测，`auto_flush_ops` 开着测得 Linear 34.7 us、LayerNorm 55.5 us，
关掉之后是 20.8 us 和 39.2 us。本文所有构图数字都在 `auto_flush_ops=0`、
`use_cuda=1` 下测（派发按 Var 的放置走，所以必须是 CUDA），端到端数字在默认
值下测。

噪声对照：基线连续跑两遍，微观行差 ≤0.5%，step/block 行差 ≤3%。

## 一个 block 的主机时间去向（改动前，370 us）

| 项 | 每次 | 次数 | 占比 |
| --- | --- | --- | --- |
| Linear(512→512/1536) | 21.7 us | 4 | 23% |
| LayerNorm | 36.9 us | 2 | 20% |
| gelu | 27.0 us | 1 | 7% |
| softmax | 23.1 us | 1 | 6% |
| batched matmul | 10.2 us | 2 | 6% |
| transpose 5d | 10.4 us | 2 | 6% |
| getitem | 4.2 us | 3 | 3% |
| 其余（reshape、残差、缩放） | | | 4% |
| 未计入（block 自身 Python + 隔离测量的缓存偏暖） | | | 25% |

把 Linear 的 20.8 us 再拆开（use_cuda=1，auto_flush 关）：

| 组成 | us |
| --- | --- |
| cublas 算子本身 | 1.24 |
| select_kernel 与其周边帧 | 4.14 |
| reshape 对（3 维→2 维→3 维） | 5.79 |
| bias 加法（broadcast + binary 两个节点） | 5.31 |
| `_check_matmul_shapes` | 1.64 |
| `Module.__call__` | 2.64 |

即：真正的算子占 6%，其余 94% 是前端与额外节点。

## 改了什么

见提交 `主机路径：把十处每算子的 Python 开销从图构建里拿掉`。十处，按收益排序：

1. **backend_libraries 备忘**：每个矩阵乘要问两次 `cublas`（一次判断能否
   派发、一次真正调用），派发层每个候选还要再问一次；原来每次三层函数帧、
   一把 RLock、三次字典查找。带 `enabled` 策略的名字（只有 mkl，它读模块
   全局和环境变量）一律不进备忘，任何注册都清空备忘。
2. **transpose 的整数置换快路径**：`numbers.Integral` 是 ABC，原来每个轴
   都要进 `ABCMeta.__instancecheck__`，之后还要建一个字典查重复。
   10.4 → 6.2 us。
3. **dispatch**：`jittor.core` / `core.Var` / `core.dispatch_context` 只取
   一次；内联 `_canonical_backend` 和已解析候选表的读取；容器参数（形状、
   置换、dim 列表）走一条不建环检测集合的扁平遍历；dtype 过滤改用
   frozenset 的 `issuperset`；`optional` 包装内联 `try_dispatch`。
4. **`_check_matmul_shapes`**：任一操作数 ≤2 维时批次维循环不可能执行，
   但原来仍要建两个切片、两个 reversed、一个 zip 和一个 enumerate 才发现
   这一点。`nn.Linear` 正是这个形状。
5. **Function**：一个 Function 不是参数容器（`dfs` 在这里是空实现），
   `Module.__setattr__` 为每个 Var 属性做的参数分类是死工作，而
   `self.saved = x, mean, rstd` 正是融合 layer_norm/softmax 保存反向输入的
   写法；`Sequence` 是 ABC，两处 isinstance 先看确切类型；无关键字参数时
   不进 `_reject_var_keywords`。
6. **indexing**：`jittor` 模块只绑定一次（一次 `x[i]` 原来要跑四条
   IMPORT_NAME），`_is_plain_int` 对确切 int 立即返回。4.20 → 3.24 us。
7. **layer_norm / gelu 的每调用重建**：`normalized_shape` 不再为两个 relay
   各重建一次，`dims` 只在真的走通用路径时才建；`_jittor_dtype_name` 原本
   调用三次，其中两次作用在它自己返回的*字符串*上；gelu 的三个 numpy 标量
   常量按计算 dtype 只留两套。

## 结果

构图（us/次，use_cuda=1，auto_flush_ops=0）：

| 行 | 改前 | 改后 | 倍数 |
| --- | --- | --- | --- |
| decode 一步（8 层） | 2991.4 | 2516.8 | **1.19** |
| 一个 block | 370.4 | 311.0 | **1.19** |
| transpose 5d | 10.38 | 6.20 | 1.67 |
| matmul_transpose（2 维） | 7.18 | 5.03 | 1.43 |
| getitem | 4.20 | 3.24 | 1.30 |
| softmax | 23.05 | 19.12 | 1.21 |
| layer_norm | 34.56 | 29.33 | 1.18 |
| Linear | 21.68 | 19.54 | 1.11 |
| gelu | 26.97 | 25.34 | 1.06 |
| *reshape 5d（对照）* | 2.30 | 2.34 | 0.98 |
| *x + x（对照）* | 1.599 | 1.607 | 1.00 |
| *x \* 0.5（对照）* | 5.358 | 5.360 | 1.00 |

末三行是对照：这些路径没有被动到，它们不动才说明上面的变化不是噪声。

端到端（GPU，ms/步，含执行，默认 `auto_flush_ops=128`）。第一轮在卡被别人
占着的时候测，设备受限的两行完全不可用（同一配置两次测出 15.1 和 22.5 ms）；
后来卡空下来又完整测了一遍，四行都可用，并且把 `metaop-cpp` 连测两次确认
复现（3.512 / 3.514，差 0.06%）：

| 场景 | 起点 48055a75 | + Python 改动 | + 单侧广播 | 累计 |
| --- | --- | --- | --- | --- |
| decode b1s1 d512 L8 推理 | 4.128 | 3.641 | **3.512** | **1.18x** |
| decode b1s1 d512 L8 训练 | 10.589 | 10.034 | 9.805 | 1.08x |
| prefill b1s128 推理 | 4.325 | 3.960 | 3.708 | **1.17x** |
| *batch b8s256 推理（对照）* | 15.126 | 15.106 | 15.060 | **1.00x** |

最后一行是对照，也是这轮最该看的一行：**设备受限的场景一点没变**。主机路径
的改动只该在主机是瓶颈的时候有用，这一行证明它在别处既没变快也没变慢。
训练那一行只有 1.08x，因为一步里 `jt.grad` 和优化器的时间不在这条路径上。

## 验证

`tests/runtime/{python_dispatch,dispatch_context,domain_dispatch,
native_op_dispatch,capability_queries,native_backend_registry}`、
`tests/ops/{matmul,matmul_dispatch,transpose_op,slice,index_op,index_bounds,
gather_index_contract}`、`tests/nn/{linear,norm,normalization_dispatch,
acl_registry_routing}`、`tests/autograd` 全量，与分支起点 `48055a75` 逐 nodeid
对比：两侧都是 22 failed / 300 passed / 81 subtests，**失败集合完全相同**。
那 22 条在分支起点就是红的。

第一次跑这个对比时基线侧收集了 0 个测试就退出了——`nvcc` 写不进一个没人
建过的 `TMPDIR`，而脚本仍然照常输出了一张"回归"表，表里列的其实是被测侧
全部 22 条失败。**一侧为空的 diff 看起来和"全是回归"一模一样**，所以摘要
里现在同时打印两侧的 passed/failed 计数。

## 剩下的空间

按同一套测量，改完之后一个 block 仍是 311 us / 约 44 个节点 = 7.1 us/节点，
而裸算子是 1.6 us。剩下的按大小排：

1. **每个节点固定的 1.6 us**。`x * 0.5` 是三个节点（array + broadcast +
   binary），`x + bias` 是两个；`BinaryOp` 的广播分支还会为两个操作数各建
   一个 `BroadcastToOp`，其中一个只是把输入原样转发出去，再建第二个
   `BinaryOp` 并转发给它——一次标量乘法要构造 5 个算子对象才留下 3 个节点。
   要更快只能**少建节点**，不是把节点建得更便宜。
2. **matmul 的 reshape 对**（每个 batched `nn.Linear` 5.8 us，一个 block
   23 us，7%）。`perf/matmul-rank` 让 cublas 直接吃 rank>2，正是去掉这一对；
   见下。
3. **Function 的录带**：一次 layer_norm 里录带占 11.8 us（3 个输入 + 1 个
   输出各一个 tape 节点，加 `tape_together`），是它 33.5 us 的 35%。推理侧
   `jt.no_grad()` 已经把这部分完全省掉（19.9 → 8.1 us）。
4. **再往下就是"重放"级别的改动**：同样形状的步长反复走同一段 Python，
   把算子序列录一次、之后在 C++ 侧回放，才能逼近 0.57 ms 的地板。这是
   torch.compile / CUDA Graph 那一档的工作量，本轮没有做。

## 少建一个节点：BinaryOp 的单侧广播

第 1 条「剩下的空间」里最便宜的一条已经做了。`BinaryOp` 的广播分支原来为
**两个**操作数各建一个 `BroadcastToOp`，而 `BroadcastToOp(x, y, {})` 自己会问
`need_broadcast`，答案是否就把输入原样转发出去——也就是说，对每一个
`x * 0.5`、每一个 `x + bias`、每一个 `x * mask`，总有一个操作数的
`BroadcastToOp` 纯粹是建出来扔掉的。改成先问同一个谓词、只给真正需要的那一侧
建：

```cpp
VarPtr xh, yh;
Var* xp = x;
Var* yp = y;
if (y->num < 0 || BroadcastToOp::need_broadcast(x, y->shape)) { xh = ...; xp = xh; }
if (x->num < 0 || BroadcastToOp::need_broadcast(y, x->shape)) { yh = ...; yp = yh; }
```

`y->num < 0` 那一半不能省：形状未定的操作数，`BroadcastToOp` 的构造函数本来
就不会转发，这里也必须照建。

这不是一个新形状：同一个文件旁边的 `TernaryOp` 本来就是这么写的——
`if (bx2) cc = make_broadcast(...)`、`if (bx) xx = make_broadcast(...)`，
四个方向各判一次，只给需要的那个建。`BinaryOp` 是这三个元算子里唯一
无条件建两个的那一个。

实测（同一台机、同一张卡、前后各测一次）：

| 行 | 无此改动 | 有此改动 | 倍数 |
| --- | --- | --- | --- |
| `x + bias`（广播） | 4.073 | 3.470 | **1.17** |
| `x * 0.5`（标量） | 5.350 | 4.697 | **1.14** |
| gelu(2048)（四个标量二元） | 25.33 | 22.74 | 1.11 |
| decode 一步（8 层） | 2532.6 | 2377.5 | **1.07** |
| 一个 block | 310.4 | 299.0 | 1.04 |
| *`x + x`（对照）* | 1.595 | 1.622 | 0.98 |
| *getitem（对照）* | 3.175 | 3.251 | 0.98 |
| *reshape（对照）* | 2.307 | 2.237 | 1.03 |

连同前面的 Python 改动，decode 一步的构图从 2991 us 降到 2378 us，**1.26x**。

## `perf/matmul-rank` 的现状：671 条已经收敛成一条

不再是"六百多条神秘失败"。按第一条**额外**失败往回查：

- 那一条是 `test_device_parity.py::TestDeviceParity::test_inner`，**单独跑就能
  复现**，8 次里 8 次失败（最早的 15 次里 13 次，前两次通过是因为那时相关
  kernel 还没编出来）。之后的六百多条是连带：一次失败把 CUDA 上下文打坏，
  剩下的全部跟着红。
- **炸在反向，不在前向**：
  `jt.grad(loss, diff)` → `sfrl_allocator.cc:305: mem_ptr does not belong to
  allocation`。前向的值是对的。
- **和 rank>2 那条快路径无关**：`sample_inner` 喂的是 A:(3,4)、B:(2,4)，两个
  都是 rank 2，`len_b == 2 and len_a > 2` 根本不成立。
- 三方对照把它夹住了：

  | 树 | 内容 | `test_inner` |
  | --- | --- | --- |
  | `d40a2e97` | 未改基线 | 通过 |
  | `perf/matmul-on-baseline` (`a03c88ef`) | 基线 + matmul 改动 | 通过（整套 121，逐 nodeid 同基线） |
  | `perf/host-path` (`6ae82f2b`) | 基线 + 发射配置/strided 下标/标量融合 | 通过（它是第 115 号测试，在那次跑到的 172 号之内，逐字符与基线一致） |
  | `perf/matmul-rank` (`e426abf8`) | 上面两叠**都有** | **失败 8/8** |

  也就是说：**两叠改动各自干净，叠在一起才炸**，而且触发点是一个 rank-2 的
  矩阵乘。
- 还没收口的是：把同样的形状、同样的梯度、同样的 cotangent 写成一个不带测试
  框架的脚本，**它通过**（rank2/rank3/rank4 的前向反向、`matmul_transpose`、
  `matmul(a, b.transpose(-1,-2))` 全对）。所以触发还需要框架里的某个状态，
  下一步要在框架内部打点，而不是继续在外面凑复现。

**结论：这条不落地。** 它值一个 block 的 7%（每个 batched `nn.Linear` 省掉
一对 reshape），但带着一条能复现的分配器不变式崩溃不能合。上面那张表和
`test_inner` 这个最小入口是下一位接手时省下来的那几个小时。


## 未解决：这一叠改动本身在冷 reference cache 下有一条设备级故障

**这条比上面所有加速都重要，先写下来。**

现象：`tests/backends/parity/test_device_parity.py` 在 **reference cache 是冷的**
时候，第 6 号测试 `test_affine_grid` 会死在一条设备级
`cudaErrorIllegalAddress`，之后整个文件跟着红（241 failed / 5 passed，约 11 分钟）。
未改基线不会。

为什么是冷 reference cache 才触发：这个文件的 CPU 参照值是 `_cpu_oracle` 在
**同一个进程里**用 `_run(op, sample, use_cuda=0)` 现算的，算过一次就落盘。所以
冷 cache 的一轮里，每个测试都会先在 CPU 上建一遍图、再在 CUDA 上建一遍；
cache 热了之后 CPU 那一半根本不跑。而 **`source_fingerprint()` 覆盖整个
`python/jittor`**——任何一行 Python 改动都会让全部 reference 失效，所以每次改完
第一次跑必然是冷的。

对照（都是全新缓存、冷 reference cache、六个测试就够看出来）：

| 树 | 卡 | 第 6 号 `test_affine_grid` |
| --- | --- | --- |
| `d40a2e97` 未改基线 | GPU 7 | 通过（整份文件 11 failed，是已知红灯） |
| `d40a2e97` 未改基线 | GPU 1 | 通过（`......FF..FFFF` 正是基线花样） |
| 带这一叠的任意树（`perf/host-path`、`perf/metaop-broadcast`） | GPU 1 / GPU 7 | **失败**，其后全部连带 |

所以既不是卡，也不是门禁框架，**是树**。而且它早于今晚的工作：今晚的两个提交
各自相对自己的分支起点都是逐 nodeid 干净的，但**分支起点本身不干净**。

已经排除的：
- 不是 JIT 缓存损坏——全新缓存、串行预热之后照样复现；
- 不是 `perf/matmul-rank` 的 matmul 改动——不带那条改动的树一样炸，而且
  同样的 `cudaErrorIllegalAddress` 与同一串 launch candidates
  （`setitem id=30`、`cublas_batched_matmul id=43`、`fused_op fused_ids=[10,2]`）；
- 不是"在一个进程里先 CPU 后 CUDA"这件事本身——把广播二元、strided 读、
  reduce 和 matmul 写成一个来回切 40 轮的脚本，两棵树都干净。所以触发还需要
  前六个测试里某个具体算子。

**还欠一步**：`9a60c3e9`（发射配置 + strided 下标特化）与 `2fc3837d`（标量融合）
两个提交里是哪一个。四点对照（基线 / 9a60c3e9 / 9e17002d / 今晚的 tip，每个跑
前六个测试）正在跑。

**对今晚成果的影响**：性能数字不受影响（都是构图与端到端计时，不依赖这条路径），
两个提交相对各自起点的门禁也不受影响。但**这一叠在这条故障定位并修掉之前不该合**。
