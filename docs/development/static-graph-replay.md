# 让重复的步长不再每次用 Python 重建图

- 状态：已实现（`35cedb1a3` 推理图重放，`2f794e289` 训练步捕获）
- 上次复查：2026-10-05
- 依据：[2026-09-14 的对位报告](../results/2026-09-14-jittor-vs-pytorch.md)、
  [2026-09-24 的真实模型对位](../results/2026-09-24-torch-compat-real-models.md)
- 复查触发：捕获的护栏、自动策略的门槛，或 `jt.profile` 的重放计数发生变化时

## 为什么是这条

对真 PyTorch 的十二项对位里，规律是**设备受限的赢、主机受限的输**：卷积 0.66/0.85、
b8s256 训练 0.51 都赢；decode（batch 1 seq 1）输 1.97，d1024 decode 输 2.05。
主机受限那几条的成因量过五遍，结论一致：

1. **阶段拆解**：decode 一步 2.70 ms 里，Python 建图约 1.6 ms，发射核函数
   0.54 ms，规划只有 0.12 ms。
2. **逐算子设备时间**（两边同法量：把上百次同一算子排进队列不中途同步再除）：
   gemm 1.00x、transpose 0.99x、elementwise 1.07x——设备侧不是问题。
3. **节点数**：省掉一个 broadcast 节点只值整步的 2%，节点数不是杠杆。
4. **前端天花板**：把 `nn.Module` / functional 整层拿掉，一个 block 从
   202.1 降到 151.4 us——前端只占建图的 25%。
5. **地板**：即使前端为零，decode 也要 1.21（建图）+ 1.0（执行器）= 2.2 ms，
   对 torch 的 1.37 仍然是 1.6x。

所以剩下的全部在「每一步都用 Python 重新构造同一张图」这件事本身。`ExecPlan` 早就是
一个可以留着的值（`src/core/exec_plan.h`：规划只读图并写 batch 编号，不分配、不发射），
缺的是**让图在执行之后活下去、能换输入、并且在不再成立时拒绝回放**。

## 现在有什么

两个入口，都是"捕获一次、之后重放"，区别在于被捕获的东西会不会更新状态：

| 入口 | 捕获什么 | 实现 |
| --- | --- | --- |
| `jt.graph_replay(module, *example_inputs)` | 一次**推理**调用：模块是输入与参数的函数，调用结束什么都不留下 | [`python/jittor/_runtime/graph_replay.py`](https://github.com/Jittor/jittor/blob/master/python/jittor/_runtime/graph_replay.py) |
| `jt.capture_step(fn)` | 一整步**训练**：前向、反向、优化器更新，连同它改写的参数、动量与归一化统计量 | [`python/jittor/_runtime/step_capture.py`](https://github.com/Jittor/jittor/blob/master/python/jittor/_runtime/step_capture.py) |

Torch 前端把 `torch.compile(..., mode="reduce-overhead")` 接到这两者上：包模块走推理
重放，包函数走整步捕获。

此外有一条**默认开启**的自动策略（`JT_AUTO_GRAPH_REPLAY=1`）：它只看
`no_grad` 下、最外层的模块调用，只在相同形状连续出现两次之后介入，只接受 Var 输入
合计小于 `auto_graph_replay_bytes`（默认 4 MB）的调用，并且只在工作集不超过
`auto_graph_replay_retain_bytes`（默认 256 MB）时录成设备图。**输入大小是"重放省下的
建图"与"多出来的输入输出拷贝"之间的廉价判据**：实测 1 MB 输入的 mlp 0.14 -> 0.05 ms、
batch 1 的 ResNet-50（0.6 MB）5.9 -> 0.7 ms、SD1.5 一个去噪步（0.3 MB）42 -> 20 ms 都
值得进，8 MB 输入、设备受限的 mlp（0.49 -> 0.52 ms）不值得。其余一切照常执行。

## 护栏：捕获什么时候作废、什么时候根本不许

一张捕获的图是**一组形状、一组参数、一条 Python 路径的照片**，它自己看不出这些变了。
因此每次调用都检查，不再成立就丢弃重捕，而不是拿旧答案应付：

- 参数的形状、dtype、个数；
- 模块是不是还在同一个 training 模式；
- 有没有参数被重新绑定（优化器更新、load、设备迁移）——图读的是它捕获的那些参数 Var，
  更新会把它们换掉，否则就会**悄悄用旧权重回答**；
- 捕获的图是不是还没被 finish。**这条最要紧**：一张 finish 过的图不会报错，它会一直用
  上次算出来的值回答。

两种情况是**根本不可重放**，捕获直接拒绝而不是答错：图里**抽了随机数**（重放会重复
同一次抽样），以及被追踪的调用**把值读回主机**（那之后的 Python 路径取决于张量的值，
下一次调用可能走另一条）。训练步捕获还要求 Python 侧的账目自己申报——
`live_array()`（每次重放前重填的小数组，如融合 AdamW 的 step 与学习率）、
`on_replay()`（每次重放前跑的钩子）、`guard()`（读数变了就重捕）、`refuse()`（直说这步
不能重放）。设备录图在每次发射前还要核对叶子有没有迁移（`graph_leaves_moved`）。

`replay.stats` / `step.stats` 报告到底发生了什么——重放了几次、退回了几次、为什么退回。
`jt.profile` 的 `prof.replay` 读同一组计数（见[性能与显存画像](../notes/profiling.md)）。
**静默退回比不重放更坏**，所以这些计数是结论的一部分，不是可选的调试信息。

## 它值多少，以及两种会骗到自己的量法

同一个 decode 步，eager 与重放在**各自独立的进程**里测、每次换输入、每个答案都和 eager
对过：

| tf-d512-L8 decode b1s1 | 耗时 |
| --- | --- |
| eager | 2.54–2.67 ms |
| 重放 | 1.57–1.79 ms |
| 参考 PyTorch | 1.37–1.43 ms |

约 1.5x，跨运行可复现；差距关掉了大部分，但没有关完。其它形状**没有**列进来，因为在
这台机器上测不稳：b8s256 前向在两个相差约 2x 的状态之间跳（14.06 ms 或 6.49 ms），
eager 一侧也一样，prefill 同理。**量你自己的模型。**

两个把作者本人骗过的量法：

- **把捕获的那个 Var 再喂回去等于什么都没测。** 输入就是捕获的 Var 时重放跳过输入拷贝，
  没有新输入写进去，图根本不重新执行——调用退化成把它已经持有的输出拷一份，而且答案
  还是对的。b8s256 这样"重放"出 6.8 ms 对 eager 的 14.0 ms。
- **在一个进程里测两臂测的是顺序。** 后跑的那一臂从热状态开始：让两臂跑完全相同的
  eager 代码（拒绝重放），第二次测出 1.57 ms，第一次 2.74 ms。

**重放不是白拿的钱。** 运行时本来就重叠得好的图重放会更慢——一个 mlp 前向 eager
0.33 ms、重放 0.42 ms，一个卷积栈持平。`measure=True` 可以在捕获时自动 A/B 一次、
输了就永久拒绝，但它**默认关闭**：这次测量扰动它所测的进程，让之后每次重放发射 219 个
kernel 而不是 132 个，包装器因此慢 30%——比它要保护的那点边际更大。

## 不要从哪里开始

- **不要从"把前端做薄"开始**。天花板量过：前端只占建图的 25%，全拿掉也不够。
- **不要用 `Op::duplicate()` 做回放**。它在 `Op` 基类里返回 `nullptr`，
  全树只有 `BroadcastToOp` 和 `ReindexOp` 两个实现了，不是通用原语。
- **不要指望减少节点数**。省一个 broadcast 节点值整步的 2%。
- **不要事后撤销 finish**。`finish_pending_liveness()` 开头就是
  `if (is_finished()) return;`，清掉标志会让它**第二次释放同一批输入的 pending
  liveness**，引用计数下溢、随后 use-after-free。保图是**第一次执行就不 finish**
  （`keep_graph`），不是执行完再撤销。

## 测试

`tests/core/test_graph_replay.py`、`tests/core/test_graph_capture.py`、
`tests/core/test_graph_replay_retention.py`、`tests/core/test_step_capture.py`
与 `tests/nn/test_graph_replay_multi_output.py` 覆盖护栏、拒绝与多输出结构；
Torch 前端那条路径在 `compat/tests/torch/test_torch_compile_replay.py`。
**每条重放的结果都要与 eager 对过**——重放的失败形态是"答得快但答的是旧值"，
它不会自己报错。
