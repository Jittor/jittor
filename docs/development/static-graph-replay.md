# 让重复的步长不再每次用 Python 重建图

- Status: 设计与预算，未实现
- Date: 2026-09-14
- 依据：[2026-09-14 的对位报告](../results/2026-09-14-jittor-vs-pytorch.md)

## 为什么是这条

对真 PyTorch 2.9.1 的十二项对位里，规律是**设备受限的赢、主机受限的输**：
卷积 0.66/0.85、b8s256 训练 0.51 都赢；decode（batch 1 seq 1）输 1.97，
d1024 decode 输 2.05。主机受限那几条的成因量过五遍，结论一致：

1. **阶段拆解**：decode 一步 2.70 ms 里，Python 建图约 1.6 ms，发射核函数
   0.54 ms，规划只有 0.12 ms。
2. **逐算子设备时间**（两边同法量：把上百次同一算子排进队列不中途同步再除）：
   gemm 1.00x、transpose 0.99x、elementwise 1.07x——设备侧不是问题。
3. **节点数**：省掉一个 broadcast 节点只值整步的 2%，节点数不是杠杆。
4. **前端天花板**：把 `nn.Module` / functional 整层拿掉，一个 block 从
   202.1 降到 151.4 us——**前端现在只占建图的 25%**（本轮开始时是 38%，
   已经拿掉一部分）。
5. **地板**：即使前端为零，decode 也要 1.21（建图）+ 1.0（执行器）= 2.2 ms，
   对 torch 的 1.37 仍然是 1.6x。

所以剩下的全部在「每一步都用 Python 重新构造同一张图」这件事本身。

## 预算

decode b1s1 d512 L8，一步 2.70 ms：

| | ms |
| --- | --- |
| Python 建图 | ~1.6 |
| 规划（build_exec_plan） | 0.12 |
| 发射循环（含 0.54 的真正发射） | ~0.9 |
| **回放后应剩** | **~1.0** |
| torch | 1.37 |

**如果重复步长不再重建图，decode 由输 1.97x 变成赢约 1.37x。** prefill、
d1024 decode、以及所有 batch 1 的推理同理。

## 这棵树已经为它准备好的部分

`src/core/exec_plan.h` 的注释写得很明白：

> The plan is a value. Building it reads the graph and writes only the batch
> numbering (`Node::batch_index` under `stamp`); it allocates nothing on the
> device, runs no kernel, and moves no memory. Everything that does is the
> Runner's job. That split is what lets `run_sync` be read as two halves and is
> **the precondition for reusing a plan across steps.**

也就是说 `build_exec_plan`（规划）与 `run_exec_plan`（执行）已经分开，
`ExecPlan` 是一个可以留着的值。缺的是「留着它，并且让图活下去」。

## 缺的三件事，以及各自的难点

### 1. 让图在执行之后活下去

`run_exec_plan` 结束时对每个算子调 `finish_pending_liveness()`，它会
`flags.set(NodeFlags::_finished)` 并**释放输入的 pending liveness**——这正是
中间结果被回收的机制。要回放就不能释放。

- **难点**：不能简单地在回放前把 `_finished` 清掉再跑一遍。
  `finish_pending_liveness` 开头是 `if (is_finished()) return;`，清掉标志会让它
  **第二次释放同一批输入的 pending liveness**，引用计数下溢、随后
  use-after-free。这是静默内存损坏，正是本项目最不能接受的失败形态。
- **可行的方向**：给 `run_exec_plan` 一个「保图」模式，**第一次执行就不 finish**
  （而不是事后撤销）。持有方（Python 侧握着全部中间 Var）负责让它们别被回收。
  `Var::alloc` 在 `mem_ptr` 非空时直接返回 true，所以重跑不会重新分配。

### 2. 换输入

回放要能把新的输入数据写进原来的输入缓冲区。`x.assign(...)` 会建新算子、
`jt.array(...)` 会建新 Var，两者都重新引入图构造。需要一个「就地覆写已分配
缓冲区」的入口。

### 3. 护栏

形状/dtype 必须与录制时一致；控制流分支改变就必须重录；`no_grad`、RNG、
`flag_scope` 这些进程级状态改变也要使回放失效。**没有护栏的回放会静默算错，
比慢更糟。**

## 不要从哪里开始

- **不要从"把前端做薄"开始**。天花板量过：前端只占建图的 25%，全拿掉也不够。
- **不要用 `Op::duplicate()` 做回放**。它在 `Op` 基类里返回 `nullptr`，
  全树只有 `BroadcastToOp` 和 `ReindexOp` 两个实现了，不是通用原语。
- **不要指望减少节点数**。省一个 broadcast 节点值整步的 2%。
