# 下游 P1：分配器并存上限、数值协议与 transpose 的运行时置换

- Status: 已修复并验证（CPU；transpose/协议另在真实 CUDA RTX 4090 上验证）
- Date: 2026-10-09
- Baseline: `6a850ea5`，此后与上游 `2.0-refactor` 合并
- Owner: Jittor 核心维护者
- Review when: `mem/allocator/sfrl_allocator.*` 的 id/键设计、`Var` 的数值协议绑定、
  transpose/fuse_transpose 的 JIT key 与模板、`CpuParallelPass` 的工作量估计发生变化

## 结论

下游量子线路库对 Jittor 核心列出的三项 P1 全部处理：

| 问题 | 根因 | 处理 |
| --- | --- | --- |
| 分配器 `block id limit extended`，之后同进程运算全部失败 | 不是 id 泄漏：id 一直在回收。30 比特张量网络采样的懒执行图在峰值同时持有约 800 万个小缓冲（batch=32 约 25 万、128 约 92 万，与 batch 成正比），超过硬上限 `1<<21`；这个上限来自把 `origin_size * ID_LIMIT + id` 压进一个 `size_t` 的池键 | 池键改为 `(size, origin_size, id)` 三元组，`ID_LIMIT` 提到 `1<<40`，只防计数器损坏 |
| `complex(v)`、`np.asarray([复数 0 维 Var, ...])` 失败 | `Var` 没有 `__complex__`，退回 `__float__` | 增加 `Var.__complex__` |
| `Var @ 外部对象` 抛内部 `AttributeError` | `__matmul__` 是 Python 层绑定，直接把任何对象交给 `matmul` | 与 torch 一致：`Var @ ndarray` 转换为 Var；其他非 Var 返回 `NotImplemented`（对方的 `__rmatmul__` 执行，没有就是 `TypeError`）；`ndarray @ Var` 为 `TypeError`；`@=` 同样处理 |
| transpose 的置换写进 JIT key，每个新置换现场编译 | `transpose`/`fuse_transpose`（含 CUDA 后端实现）的 key 有每轴 `AXES` | key 只保留 dtype、维数和形式；置换在主机侧换算成每轴 stride 作为运行时参数传入（stride 本来就是运行时值） |
| 上一项修复后，6 比特 DMCircuit 热身后仍为 300–390 ms | 与编译无关：一个 [4,4,64] 的复数乘加（约 1K 次）在 128 核上开了线程组。`CpuParallelPass` 把嵌套里所有内层循环的上界相乘，归约的两个兄弟循环（清零行、累加行）被算成乘积，越过 65536 门槛 | 工作量改为沿路径相乘、兄弟循环相加 |

## 验证

环境：128 核 CPU 主机（负载约 23）、RTX 4090，conda `jittor` 与下游 `tensorcircuit`
环境（以 editable 方式使用本工作树），每条线独立缓存。日志在
`$JITTOR_LAB_ROOT/_state/tc-p1/`（未版本化）。

| 检查 | 结果 |
| --- | --- |
| 30 比特 GHZ，`allow_state=False`，1024 次张量网络采样（下游官方用例同参数） | 完成，924 s，峰值 RSS 26.6 GB，`0…0`/`1…1` = 507/517；修复前在 `tot_block_id < ID_LIMIT - 1` 处失败 |
| C++ `sfrl_holds_more_than_two_million_live_allocations`（2^21+1000 个并存分配，地址假分配器，不占内存） | 通过 |
| 下游 `jitcount.py`：每次迭代新增 kernel / 耗时（默认线程数） | dense 6：+0 / 3–4 ms；DM 4：+0 / 7–8 ms；DM 6：+0 / 11–12 ms（修复前每次 +1 个 kernel、0.7–4.6 s；同机 PyTorch DM 6 为 6–7 ms） |
| 随机 12 维置换（float32/complex64；纯置换、两种 fuse 形式、梯度） | 热身后 +0 编译、约 1 ms |
| `complex(v)`、`np.asarray`/`np.real`（complex64/128）、`Var @ ndarray`/`ndarray @ Var`/自定义对象/`@=` | 与 torch 合同一致 |
| 新增 `tests/ops/test_transpose_runtime_permutation.py`，扩充 `tests/ops/test_binary_reflected_and_complex_order.py` | CPU 与 CUDA 通过 |
| 定向回归（cpu_parallel_pass、transpose、reduce、高维张量、JIT C++ 测试） | 通过；另见下方既有失败 |

## 已知边界与后续

- **峰值内存**：采样用例 26.6 GB 来自懒执行图在一次 `numpy()` 前累积全部样本的
  中间结果；这次只去掉了错误的上限，没有改变执行顺序。
- **与本次改动无关、未在原始基线上单独复验的失败**（oneDNN 已加载时出现）：
  `tests/codegen/test_jit_tests.py::test_fused_op_relay_matmul`（relay 读到的
  `mkl_matmul_op.cc` 源码只有 `#include <cmath>`，`jit_run` 在 `#ifdef JIT_cpu`
  内）、`tests/ops/test_matmul.py` 的 `test_matmul`/`test_matmul_type`（同一 relay
  未命中）、`tests/ops/test_misc_issue.py` 的 `test_mkl_conflict1/2`（子进程里
  `jt.mkl_ops` 为 None）。这些路径都不在本次改动范围内。
- `TempAllocator` 仍有同样的 `1<<21` 键设计；它只持有少量 workspace，未改。
- ROCm/NPU 未验证。
