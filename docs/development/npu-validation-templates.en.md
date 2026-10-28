# NPU defect and performance record templates

- Status: reusable templates; placeholders are not validation results
- Review date: 2026-09-10
- Corresponding baseline: `2.0-refactor` / `1a1f175e`
- Owner: NPU backend and Torch compatibility maintainers
- Recheck when: the standalone compatibility package entry point, the NPU gates, the fallback
  policy or the performance protocol changes

For validating native Jittor and the standalone Torch compat layer on Huawei Ascend.
Environment setup and real ACL probing follow the [Ascend guide](../guides/ascend-910b.en.md);
test ownership follows the [test system](test-system.en.md), and timing constraints follow the
[benchmarking guide](../performance/benchmarking.en.md). This page only provides record formats;
it does not maintain a second task board or issue ledger.

## How to use it and evidence requirements

Copy the templates below into the topic's existing maintainer result report; write a new topic
as `docs/results/YYYY-MM-DD-topic.md`, with an opening metadata block stating status, date,
baseline commit, validation scope (device, software versions, what was and was not run),
maintainer (a role, not a person) and recheck conditions, and add it to the toctree and index
table of [`docs/results/index.md`](../results/index.en.md). Reports keep only reproducible
conclusions; do not let them become project history, and delete or archive them once stale.
Active defects maintain owner, evidence links, workaround and exit criteria only in the
[known-issues ledger](https://github.com/Jittor/jittor/blob/master/agent/manuals/known-issues.md);
task status is recorded in GitHub issues and PRs, and PR descriptions carry the validation
evidence. After a fix, keep the before/after evidence in the report and remove the resolved
entry according to the ledger rules.

Raw stdout/stderr, pytest XML, device snapshots, profiles, timing JSON/CSV, caches and
binaries go under `$JITTOR_LAB_ROOT/_state/ascend-validation/<run-id>/`, marked
**unversioned**. Reports keep only brief conclusions, runnable commands, relative artifact
paths and necessary checksums; do not write personal home directories, machine addresses or
fixed device numbers. External artifacts that are lost must be regenerable from the record.

Native, standalone compat and the real PyTorch oracle each use a separate process. Native mode
clears the shim activation variables; compat explicitly activates the standalone namespace and
records `torch is not jittor`. Record the source, version and installation method of both the
core package and the standalone `jittor-torch` package, and do not treat
`import jittor as torch` as compat. A real PyTorch/`torch_npu` oracle must be shown to come from
a binary implementation and must not resolve to the Jittor shim. Each process uses its own
`JITTOR_HOME` or `cache_name`; test caches must not be shared with benchmarks; the first JIT or
extension compile completes serially. Use the environment managed by the runner and do not
modify the user's `HOME`.

Set `backend_fallback=error` before starting every NPU validation process, and record
`has_acl`, `use_acl`, `use_cuda`, device residency and the values of
`jt.core.backend_fallback_count()` before and after. Computation and synchronization must
complete inside a `forbid_backend_fallbacks()` scope; merely creating lazy tensors, a successful
import, device-memory usage or the absence of fallback log lines are all insufficient proof of
NPU execution. Report unverified, environment-blocked, failed, skipped, xfailed and passed
separately; an all-skip run cannot be recorded as a pass.

## Template one: NPU defect record

The `<...>` fields below are filled in by the actual run; do not reuse pass counts or device
information from an older report.

```text
标题：<BUG-ID> <触发操作及观察到的错误>
状态：<已复现 / 修复待验证 / 已修复 / 环境阻塞>
严重性 / owner：<...>
记录时间（含时区）/ 复查触发条件：<...>
关联：<问题总账条目、GitHub issue/PR、已有报告>

基线与环境
  核心分支 / 完整提交 SHA / dirty diff 或补丁 SHA256：<...>
  独立 compat 包版本 / 提交 / 安装来源：<...>
  修复提交或待提交补丁 SHA256：<...>
  OS / 架构 / Python 可执行文件标识与版本 / NumPy / pytest：<...>
  NPU 型号 / 驱动 / 固件 / CANN / ccec / 宿主编译器：<...>
  可见设备映射与分配方法：<映射证据保存在外部快照，不硬编码设备编号>
  设备占用 / 健康状态 / 并发负载：<...>
  模式与包身份：<native / independent compat；实际 import 来源与身份断言>
  状态目录 / JITTOR_HOME / cache_name / 模式与后端环境变量：<...>

最小复现
  输入：<shape、dtype、stride、seed、具体边界值及必要权重校验和>
  步骤：<完整启动命令、环境前置、版本化测试 node id 或最小代码>
  预期：<独立 NumPy/数学/二进制 PyTorch 参考及容差依据>
  实际：<错误值、最大绝对/相对误差、异常或退出码；不可只贴最后一行>
  频率：<失败次数 / 重跑次数；冷/热缓存>
  真实 NPU 证据：<has_acl/flags、驻留断言、同步位置、回退计数 before/after>
  原始产物：<未版本化相对路径、日志/输入/设备快照 SHA256>

归因与修复
  所属层：<原生算子 / ACL backend / compat 语义 / 测试或环境>
  根因：<证据支持的因果关系；未确认处明确标出>
  修改：<源码路径及行为变化；为何在此层修复>
  workaround：<适用范围；不允许以静默 CPU fallback 作为通过>
  未覆盖边界 / 风险：<...>

修前修后与回归（每种受影响模式单独填写）
  native NPU：<修前失败证据 → 修后通过；命令与测试计数>
  independent compat NPU：<修前失败证据 → 修后通过；命令与测试计数>
  共享 CPU 语义 / 相关加速后端：<命令与结果，或明确未验证原因>
  单测新增或修改：<node id；断言独立数值/梯度、设备执行与回退增量>
  skip / xfail / 未执行节点：<数量、具体原因及已有问题关联>
  验收条件：<修前复现、修后回归、零回退、退出码、必要性能检查>
  结论：<只声明本记录实际验证的 dtype、shape、模式和后端>
```

Where differentiable computation is involved, check the forward pass, the input gradients and
all affected parameter gradients separately; CPU numerical gradients cannot substitute for NPU
backward-execution evidence. For diagnosis, `sync_run=1` can be used to localize the problem;
re-verify on the normal asynchronous path afterwards. Do not eliminate a regression by
loosening tolerances, widening skips or catching backend exceptions.

## Template two: NPU performance report

```text
标题：<操作/模型> 原生 Jittor 与独立 Torch compat 的 NPU 性能
状态 / owner / 日期（含时区）/ 复查条件：<...>
目标：<延迟 / 吞吐 / 峰值内存；预先声明阈值和比较口径>
环境与基线：<按缺陷模板记录；每个运行时分别填 SHA/dirty 补丁与包身份>
比较组：<native Jittor / independent compat / 可选真实 PyTorch + torch_npu>
正确性前置：<相同输入/权重的输出与梯度误差、容差、NPU 执行和零回退证据>

工作负载
  算子或模型 / batch / shape / dtype / layout / seed：<...>
  inference 或 training / grad / AMP / optimizer / 编译与融合选项：<...>
  输入与权重来源 / 校验和 / 状态重置方法：<...>
  输入常驻设备与输出保留方法：<...>

计时协议
  命令与脚本版本：<可执行命令、参数及版本化脚本路径>
  每组独立状态目录与缓存：<测试、native、compat、oracle 分开>
  冷启动定义：<新进程/空编译缓存/已有系统缓存，明确哪些成本计入>
  热路径：<串行首次编译、预热次数、每轮次数、独立重复轮数>
  同步：<计时前及计时后同步；包含全部输出与梯度的实体化>
  计时窗口：<是否包含构图、前向、反向、更新、H2D/D2H、编译与数据读取>
  时钟 / 单位：<例如 perf_counter、ms/step>
  测量顺序：<同一设备顺序交错各运行时；设备负载、温度与主机线程配置>
  内存口径：<allocated/reserved/设备总占用；峰值采样方法与范围>
  无效样本判据：<预先约定；干扰或失败单独记录，不能只删慢样本>
  逐算子诊断同步状态：<性能运行不启用 sync_run=1 或其它诊断开关>

原始产物（未版本化）
  相对根目录：<...>
  每轮原始时延 / 环境快照 / profile / correctness / stdout / stderr：<路径与 SHA256>
```

Fill in the summary table from the actual repeated samples; leave untested cells as "not
measured" rather than `0`. p95 needs enough samples, otherwise report the sample count and the
range. Cold starts belong in their own row, not mixed into hot-path statistics.

| case / mode | baseline or patch | samples | hot-path median ms | p95 ms | throughput and unit | peak memory and measure | fallback delta | correctness |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `<case> / native` | `<SHA>` | `<n>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |
| `<case> / independent compat` | `<SHA>` | `<n>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |
| `<case> / real torch_npu（可选）` | `<version>` | `<n>` | `<...>` | `<...>` | `<...>` | `<...>` | `不适用` | `<...>` |

The conclusion states the comparison formula, for example `compat median / native median`
(below 1 means compat is faster), together with run-to-run variation, the acceptance threshold,
bottleneck evidence and the applicable scope. Before and after a fix, the workload, hardware
and protocol must be identical; results from a dirty tree are temporary evidence and must not
masquerade as a publishable performance baseline for a clean commit. When correctness fails,
a fallback occurred, the environment was disturbed or the oracle is missing, state explicitly
that the performance conclusion is not acceptable; do not let timing numbers paper over the
failure state.
