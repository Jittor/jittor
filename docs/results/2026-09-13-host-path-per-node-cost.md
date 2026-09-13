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

端到端（GPU，ms/步，含执行，默认 `auto_flush_ops=128`），两个方向各跑一遍
以排除先后顺序偏差：

| 场景 | 改前 | 改后 | 倍数 |
| --- | --- | --- | --- |
| decode b1s1 d512 L8 推理 | 4.132 / 4.076 | 3.639 / 3.568 | **1.14** |
| decode b1s1 d512 L8 训练 | 10.601 / 10.581 | 10.055 / 10.379 | 1.02–1.05 |
| prefill b1s128 推理 | 4.329 / 4.591 | 3.947 / 4.623 | 不可用 |
| batch b8s256 推理 | 15.1 / 16.3 | 15.2 / 22.5 | 不可用 |

只有 decode 推理这一行在两个方向上都复现。后两行是设备受限的，本机的 GPU
同时被别人占着（同一配置两次测出 15.1 和 22.5 ms），**这两行不构成任何结论**
——既不能说明变快，也不能说明没变慢。设备受限的场景本来就不该被主机路径的
改动影响，这个预期没有被证实也没有被证伪。

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

## `perf/matmul-rank` 的现状

- matmul 的 rank>2 改动**单独打在未改基线 `d40a2e97` 上**（分支
  `perf/matmul-on-baseline`，`a03c88ef`）：121 failed / 654 passed，与基线
  对照组**逐 nodeid 完全相同**。这个 op 改动本身是干净的。
- 同样的改动在 `perf/matmul-rank`（`bd43655e`，且已补上
  `push_back_check_overflow`）上、用全新缓存、串行预热之后，仍然是
  671 failed。但失败的形状说明问题不在 matmul：前面若干条是基线本来就红的
  （`all() got an unexpected keyword argument 'keepdim'`、fft 对拍），随后
  出现一条 **`cudaErrorIllegalAddress`（code 700，设备级）**，此后 600 多条
  全部失败——包括 `test_isnan`、`test_log` 这类不可能与矩阵乘有关的。
  也就是说：**一次非法访存把上下文打坏了，剩下的全是连带**。
- 所以现在要回答的是"matmul 之下的那一叠改动（发射配置、strided 下标、
  标量融合）在整套设备门禁上是否干净"，`perf/host-path` 带着同一叠改动，
  它的全量门禁正在跑。在那个结果出来之前，不能说 matmul 有问题，也不能说
  它没问题。
