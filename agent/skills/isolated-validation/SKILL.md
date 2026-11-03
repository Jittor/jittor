---
name: isolated-validation
description: 为一次源码、wheel、测试或性能验证准备隔离状态目录，并证明解释器、包和设备来源。用于新任务环境准备、避免 worktree/缓存串用；不替代安装指南和目标后端的硬件验收。
---

# 准备隔离验证环境

先按仓库规则同步目标分支并记录提交。安装步骤查[安装指南](../../../docs/getting-started/installation.md)，
设备前置查[平台支持](../../../docs/guides/hardware/platform-support.md)与相应硬件指南，
测试选择查[测试体系](../../../docs/development/testing/test-system.md)。不在技能中复制
SDK 版本表、某台机器的地址或第三方库依赖清单。

## 1. 为本次运行分配状态目录

从仓库根目录执行；每个并发任务使用不同 topic/run，原始结果不进入源码树：

```bash
export JITTOR_LAB_ROOT="${JITTOR_LAB_ROOT:-$(cd .. && pwd)/jittor-lab}"
run_root="$JITTOR_LAB_ROOT/_state/<topic>/<run>"
mkdir -p "$run_root/jit" "$run_root/tmp" "$run_root/xdg-cache" "$run_root/logs"
export JITTOR_HOME="$run_root/jit"
export TMPDIR="$run_root/tmp"
export XDG_CACHE_HOME="$run_root/xdg-cache"
export PYTHONDONTWRITEBYTECODE=1
```

默认保留用户 `HOME`。只有测试明确需要验证全新用户目录时才为该子进程单独指定 `HOME`，
并记录它会改变配置、凭据和工具缓存的发现方式。不要为普通运行重置 `HOME`。

首次 JIT/扩展编译串行完成；benchmark 与测试不要共享缓存。中止运行时先结束本任务的
worker 并确认退出，再处理其状态目录，不能删除仍被其它进程使用的缓存。

## 2. 选择源码或安装产物验证

| 对象 | 做法 | 必须留下的来源证据 |
| --- | --- | --- |
| 当前 checkout | 使用仓库测试入口；手工脚本可设置 `PYTHONPATH="$PWD/python"` | 提交 SHA、dirty 状态、实际 `jittor.__file__` |
| 当前提交构建的 wheel | 在 lab 内的独立环境安装 wheel，清除源码路径注入，从 checkout 外运行 | wheel 构建提交、安装包版本、实际导入位置 |

`PYTHONPATH` 不能安装独立的 `jittor-compat`；需要 Torch 前端时按安装指南安装配套包。
不要为“修正”wheel 验证而重新加源码路径。检查子进程/rank 的导入来源，不能只看父进程。

```bash
python -c 'import sys, jittor; print(sys.executable); print(jittor.__file__)'
```

上面的导入可能触发首次编译，应在指定缓存中串行执行。核对结果符合所选对象；仅导入
成功不能证明数值、CUDA、NPU 或下游应用支持。

## 3. 使用维护中的验证入口

源码的快速检查由测试 runner 管理模式与环境：

```bash
python tools/run_test_suite.py --tier core --backend cpu
```

选择 CUDA/NPU 前确认实际分配的设备、驱动和工具链；不要因为继承了 CPU-only 变量就
误报缺硬件，也不要根据 `cuda` 接口名推断芯片厂商。nox 会话有自己的隔离环境与输入
约定，先看所选会话的配置，不把手工 shell 的所有变量强灌进去。

后端计算使用真实设备并验证数值、驻留和回退；本技能只负责环境准备。
下游库的版本组合和官方用例到 [JittorEcosystem](https://github.com/JittorEcosystem)
对应仓库运行，发现核心/compat 缺陷时再在主仓保留最小复现。

## 4. 收尾

记录提交、解释器、包来源、实际设备、命令、完成数量、skip 原因和结果。验证结论写入
GitHub issue/PR，日志与数据保留在本次 lab 目录；运行结束不会把未执行的硬件项变成通过。
