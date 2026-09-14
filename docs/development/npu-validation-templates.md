# NPU 缺陷与性能记录模板

- 状态：可复用模板；占位符不是验证结果
- 复查日期：2026-09-10
- 对应基线：`2.0-refactor` / `1a1f175e`
- Owner：NPU 后端与 Torch 兼容层维护者
- 复查触发：独立兼容包入口、NPU 门禁、回退策略或性能协议变化

用于华为昇腾上的原生 Jittor 和独立 Torch compat 验证。环境配置及真实 ACL
探测按 [昇腾指南](../guides/ascend-910b.md)，测试归属按
[测试体系](test-system.md)，计时约束按 [性能测试指南](../performance/benchmarking.md)。
本页只提供记录格式，不维护第二份任务看板或问题总账。

## 使用方式与证据要求

复制下面的模板到该主题既有的维护者结果报告；新主题按当前仓库约定放入
`refactor-wip/results/YYYY-MM-DD-topic.md`。活跃缺陷只在
[已知问题总账](../../agent/manuals/known-issues.md) 维护 owner、证据链接、workaround
和退出条件；任务状态只更新 [唯一看板](../../refactor-wip/architecture/refactor-board.md)。
修复后在报告保留前后证据，按总账规则移除已解决条目。

原始 stdout/stderr、pytest XML、设备快照、profile、计时 JSON/CSV、缓存和二进制
放在 `$JITTOR_LAB_ROOT/_state/ascend-validation/<run-id>/`，标记为**未版本化**。
报告仅保存简要结论、可执行命令、相对产物路径和必要的校验和；不要写入个人主目录、
机器地址或固定设备编号。外部产物丢失时应能通过记录重新生成。

原生、独立 compat、真实 PyTorch oracle 各用独立进程。原生模式清除 shim 激活变量；
compat 明确激活独立命名空间并记录 `torch is not jittor`。记录核心和独立 `jittor-torch`
包的来源、版本及安装方式，不把 `import jittor as torch` 当作 compat。
真实 PyTorch/`torch_npu` oracle 需证明来自二进制实现，不能解析到 Jittor shim。
各进程使用独立 `JITTOR_HOME` 或 `cache_name`，测试缓存不得与 benchmark 共享；
首次 JIT/扩展编译串行完成。沿用运行器管理的环境，不修改用户的 `HOME`。

所有 NPU 验证进程启动前设置 `backend_fallback=error`，记录 `has_acl`、
`use_acl`、`use_cuda`、设备驻留及 `jt.core.backend_fallback_count()` 的前后值。
计算与同步必须在 `forbid_backend_fallbacks()` 作用域内完成；仅创建惰性张量、
import 成功、显存占用或没有回退日志均不足以证明 NPU 执行。
未验证、环境阻塞、失败、skip、xfail 和通过分别报告；全 skip 不能记为通过。

## 模板一：NPU 缺陷记录

以下字段中的 `<...>` 由实际运行填写，不要沿用旧报告的通过数或设备信息。

```text
标题：<BUG-ID> <触发操作及观察到的错误>
状态：<已复现 / 修复待验证 / 已修复 / 环境阻塞>
严重性 / owner：<...>
记录时间（含时区）/ 复查触发条件：<...>
关联：<问题总账条目、看板任务、已有报告>

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

涉及可导计算时分别检查前向、输入梯度和全部受影响参数梯度；CPU 数值梯度不能替代
NPU 反向执行证据。诊断时可用 `sync_run=1` 定位，再用正常异步路径复验。
不要通过放宽容差、扩大 skip 或捕获后端异常来消除回归。

## 模板二：NPU 性能报告

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

汇总表按实际重复采样填写；未测留作“未测”，不能填 `0`。p95 必须有足够样本，
否则报告样本数与范围。冷启动单列，不混入热路径统计。

| case / 模式 | 基线或补丁 | 样本数 | 热路径 median ms | p95 ms | 吞吐与单位 | 峰值内存与口径 | 回退增量 | 正确性 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `<case> / native` | `<SHA>` | `<n>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |
| `<case> / independent compat` | `<SHA>` | `<n>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |
| `<case> / real torch_npu（可选）` | `<version>` | `<n>` | `<...>` | `<...>` | `<...>` | `<...>` | `不适用` | `<...>` |

结论填写比较公式，例如 `compat median / native median`（小于 1 表示 compat 更快），
以及轮间波动、接受阈值、瓶颈证据和适用范围。修复前后必须使用相同工作负载、硬件与
协议；脏树调查结果标为临时证据，不能冒充某个干净提交的可发布性能基线。
正确性失败、有回退、环境受干扰或 oracle 缺失时，明确性能结论不可验收，勿以耗时
数字覆盖失败状态。
