# 并行算子编译器的段错误

- 状态：非 Jupyter 负载下仍未解决；Jupyter 的 SIGCHLD 子问题已修复
- 上次复查：2026-08-22
- 基线：`137f9dd1`
- Owner：编译器 / 执行器维护者
- Workaround：`jt.flags.use_parallel_op_compiler = 0`
- 退出条件：在开启并行编译的情况下，一个最小化压力测试可重复通过，且无死锁、无缓存
  损坏、无明显性能回退

这一页是**单个问题的调查记录**，不是已知问题的清单。问题总账在
[`agent/manuals/known-issues.md`](https://github.com/Jittor/jittor/blob/master/agent/manuals/known-issues.md)，
本页对应其中的 `KI-COMPILER-001`；两边的状态、workaround 与退出条件必须一致，
改一边就改另一边。总账里其余条目只有条目本身，没有对应的长文页面。

## 症状

某些大模型与设备一致性负载在**多个融合运算并发编译**时，可能以分配器错误或段错误终止。
该失败对时序敏感，而且可能在**破坏状态的那次编译之后**才暴露出来——所以**最后打印的
算子名不是可靠的归因**。

数据集、ACL 和一致性测试路径已经成功使用串行编译。这是一项**围堵措施，不是"运行时算子
执行有缺陷"的证据**。

## 已解决的 Jupyter 子问题

此前的 Jupyter 复现有一个独立且已被证实的原因。并行算子编译器用的是 `std::thread`，
**并不 fork 编译 worker**。但 Jittor 仍会在 ipykernel 内安装一个进程级的 `SIGCHLD`
处理器，只要有任何子进程被信号杀死就让宿主快速退出。一个只包含 `import jittor` 和一个
无关的、被 `SIGKILL` 的子进程的最小进程，复现出了带 `si_code=CLD_KILLED` 与
`si_status=SIGKILL` 的退出。

现在 Jupyter kernel 保留它已有的 SIGCHLD 处置方式（就像它已经保留 SIGINT 那样），而
Jittor 保留 SIGILL 与 SIGBUS 的故障诊断。确定性的宿主归属回归测试和定向的 CPU/CUDA
冷算子探测都通过。

但**之后一次完整的 nbclient 冒烟仍然在 transformer 注意力负载下、八个编译 worker 时
死了两次**，其中一次还设了 `JT_NO_SIGNAL_HANDLER=1`；同一个冒烟在串行编译下通过。
因此在这个更大的问题未关闭之前，**离线 notebook 门禁保持串行**。

## 当前假设

主要假设是：**并行编译 worker 与进程级的编译/缓存加锁之间缺少一道同步边界**。某个
worker 可能观察到由拥有进程建立的锁状态，从而错误地认为共享的编译器或 relay 状态是
受保护的；随后的并发修改就表现为堆损坏。

对非 Jupyter 负载而言**这仍然只是假设**：relay 组的归属、确切的共享可变对象、以及第一次
非法访问，都还没有用 sanitizer 轨迹演示出来。因此**加一把大锁不是可接受的修法**——它
可能让嵌套编译死锁、抹掉本意的并行性，或者根本保护不了多进程的缓存访问。

## 复现协议

使用专用的状态目录，并保留确切的提交、编译器、设备和环境：

```bash
export JITTOR_HOME="$JITTOR_LAB_ROOT/_state/parallel-compiler/repro/jittor-home"
export cache_name=parallel-compiler-repro
python -m pytest -v tests/backends/parity/test_device_parity.py
```

用默认的并行编译设置跑一次，再用下面的设置跑一次：

```python
import jittor as jt

jt.flags.use_parallel_op_compiler = 0
```

**两个变体不要并发运行，也不要共用同一个缓存。** 一次有用的复现要记录：缓存是否是冷的、
worker 数量、最后一次完成的编译、信号/回溯，以及重复的串行运行是否干净。

## 调查计划

1. 在保持冷缓存失败的前提下把负载最小化；
2. 用稳定的标识符为编译任务创建、relay 组归属、缓存锁获取和 worker 完成打点；
3. 在 AddressSanitizer 或 ThreadSanitizer 下运行最小化的原生编译路径；
4. **在改动同步之前**先定位第一次非法访问或竞争；
5. 施加范围最窄的归属或加锁修复，并补一个确定性的压力回归测试。

## 验收门槛

一个修复必须同时证明：

- 冷缓存与热缓存的重复压力运行不再崩溃；
- 带超时的测试显示无死锁；
- 两个进程分别使用独立缓存与共享缓存配置时不会损坏产物；
- 并行编译时间相对记录的基线**没有明显回退**；
- 恢复并行编译后，编译器、设备一致性和代表性模型测试全部通过；
- **串行 workaround 与问题总账条目在同一次改动中一并移除**。

在这些证据出现之前，优先考虑确定性验证的调用方可以显式关闭并行编译器，**并应在结果中
报告这一选择**。
