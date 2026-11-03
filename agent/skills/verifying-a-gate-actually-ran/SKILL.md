---
name: verifying-a-gate-actually-ran
description: 核对门禁的来源、收集集、执行结论与跳过原因，识别少跑、空跑和错误缓存造成的假绿。
---

# 证明门禁确实执行了

## 验证证据

- 先按[隔离验证](../isolated-validation/SKILL.md)记录 SHA、dirty 补丁、解释器和实际包路径。
  子进程、rank 与主进程都要核对，源码测试和 wheel 测试分别记录。
- 用维护中的 runner 选择 native/Torch 模式，查看实际选择集；仅 `--collect-only` 不证明执行。
- 保存退出码、收集的 nodeid、每个 setup/call/teardown 结论和 skip 原因。非零退出、worker 崩溃、
  超时或缺少结论都不能因部分用例通过而算整轮通过。
- xfail 与环境 skip 分开；严格 xfail 的意外通过要暴露。硬件验收必须要求目标用例实际运行。
- 冷缓存与热缓存的结果不直接比较。记录配置指纹和编译发生与否；不同隔离目录不要求路径逐字相同。

## 比较两轮

使用 `tools/gate_conclusion_diff.py record --out <外部结果.json> -- <pytest参数>`
记录，再用 `compare <基线.json> <候选.json>` 比较。
在基线和候选各自的目录调用各自的工具；record 使用脚本所属仓库作为工作目录，
从另一棵树调用它并不等于测那棵树。核对测试路径和 `JITTOR_SOURCE_ROOT`。先看 `--help` 确认所需模式和参数。
新增用例通过可重复的 `--expect-new NODEID` 显式声明；不靠总通过数或失败数推断等价。

串行改为 xdist 时同时核对收集集与结论集。controller 不执行 worker fixture；汇总应来自
worker 回传的 reports/collection，而非 controller 的空本地集合。用一个故意少跑或失败的
小会话验证统计器能拒绝它；相关回归见 `tests/structure/test_gate_conclusion_record.py`。

基线也失败时，先比较同输入、同模式、同缓存温度下的具体失败，不能自动称作“历史问题”。
改变同步后的源码或环境就重新验证受影响范围，保留旧结果原基线。
