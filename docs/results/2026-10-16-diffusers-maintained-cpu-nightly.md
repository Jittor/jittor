# Diffusers 完整 CPU nightly 的独立环境验收

- 状态：本地完整维护门禁通过；不是 GitHub scheduled 作业执行记录。
- 日期：2026-10-16。
- 维护者：Jittor 核心维护者。
- 实测源码：`d63976115fb6a31d94c4474871b07f2aba69d7e7`，干净 worktree。
- 范围：官方 Diffusers 0.35.1 的既有完整 tiny UNet CPU 三阶段协议。
- 复查条件：nightly 安装闭包、来源校验、UNet 协议、Torch 或 Diffusers 固定版本变化。

## 维护入口与环境

`ecosystem.yml` 增加 `diffusers` profile，`tools/provision_ecosystem_nightly.py`
建立全新的 candidate、oracle、controller 三个 prefix，再调用
`nox -s ecosystem_diffusers_source`。维护入口最终执行 `tools/run_diffusers_gate.py`；
没有用 import/preflight 或缩小模型替代完整协议。既有 74 个 nox 函数的 AST 不变。

本次从独立健康 Python 3.11 引导，全依赖正常解析安装。三个 prefix 的实际 `pip check`
均通过，oracle 实际导入 Torch 2.11.0+cpu 的 `_C`，无 shim，CUDA 不可见。
本地可复用的是已哈希的官方 wheel 和干净上游 Git 输入；没有复制 JIT 缓存或安装目录。
单测和 benchmark 的 HOME/JITTOR_HOME 分开，首次 benchmark JIT 实际编译。
原 controller 未装 pytest 的 host 命令失败保留，host 验证使用独立健康 oracle 环境。

## 完整协议与独立复算

原 seed 0/7、144 个参数和 sample 输入梯度、三步 SGD（lr=0.001）保持不变。
六个独立 worker 完成 base、perturb、train，nox 自然退出 0，包含首次编译约八分钟。
独立读取原始 NPZ/JSON 复算如下：

| 证据 | 实际结果 |
| --- | --- |
| 两端完整数组对 | 1,165 |
| 实际 CPU tensor observations | 2,618 |
| 每 worker 来源文件闭包 | 1,612 |
| 两端三步参数更新复算 | 864 |
| 最大绝对误差 / 最大全局归一误差 | 4.959106e-5 / 9.408136e-7 |
| 最大 SGD 重放舍入误差 | 3.725290e-9 |

保持原逐阶段 `.002 absolute OR .01 global-reference-scale` 预算；SGD 重放使用
`2*float32_eps*(abs(old)+abs(lr*grad))` 的舍入界限。参数和输入梯度完整且非零，
扰动确实改变输出，所有 worker fallback count 为 0。

逐侧将实际安装文件与原始官方 wheel 比较：Torch 各 11,945 个非 RECORD 文件、
Diffusers 各 702 个；真实 pip RECORD 的所有带哈希行分别为 11,949 和 706，全部核实。
没有将 archive RECORD 覆盖安装 receipt；生成 pyc 仅核实对应源码，不冒称 wheel 载荷。
完整源码另按 Git mode/blob/content 核实 2,916 文件；候选 JSON 的 binary 字段不可用，
不据此宣称已捕获其 native binary SHA。真实 oracle `_C` 的原始哈希另行核实。

## 证据及范围

原始产物未版本化，位于 `$JITTOR_LAB_ROOT/_state/diffusers-nightly/20261016/`：
`fresh-v1/provision.json`、`fresh-v1/logs/024.log`、`fresh-v1/maintained-output/`、
`fresh-nightly-terminal-v1.json`、`nightly-cpu-independent-v1.json` 和
`nightly-installed-independent-v1.json` 保留真实安装、命令、终态和独立复算。
`nightly-evidence-seal-v1.json` 封存 395 文件 / 82,710,906 字节，SHA256
`3abfe7fe0acba477941a5d9c4694a5252a10c69fb3366678eef446f214a30179`；全量二次哈希通过。

这只验收既有完整 CPU 组件 nightly。既有 CPU/CUDA 配对来源结论见
[官方安装验证](2026-10-16-diffusers-official-installation.md)，其原 b212 源码不重标为本轮 d639。
不新增整库覆盖分子，不授 pipeline、CUDA 新测、L4 性能或 GitHub scheduled 已执行结论。
