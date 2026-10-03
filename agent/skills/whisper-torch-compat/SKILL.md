---
name: whisper-torch-compat
description: 原版 OpenAI Whisper 在 Jittor Torch shim 上的 CPU L0–L2 复验与双解释器对拍；明确未验证范围。
---

# 原版 OpenAI Whisper 兼容验证

## 用途

直接运行原版 `openai-whisper`，不修改下游源码，不新增私有 adapter。
本入口覆盖 `openai_whisper` 小配置模型与 `openai_whisper_log_mel` 音频预处理；
它们不是 Transformers Whisper，也不等于官方 tiny checkpoint 的完整能力验收。
三步正式任务训练与 optimizer 轨迹已完成 CPU 验证；本提交登记到严格累计 L2。
真实 checkpoint 保存互通、完整生成工作流和性能不属于本提交的支持声明。
团队登记采用协作文档的 L0–L5；此处不重新定义等级。

## 两侧环境

先读 [适配分流规范](../downstream-library-adaptation/SKILL.md)、
[环境手册](../../manuals/environment.md) 和
[统一 runbook 协议](../torch-compat-repo-runbook/SKILL.md)。

- 基线：Python `3.11.16`、NumPy `1.26.4`、Whisper 提交
  `86098128c0b4f24f0e2aa2994de830614b474227`（版本标识 `20250625`）。
- CPU oracle 固定为 `torch==2.4.1+cpu`；两侧线程数、NumPy 和 Whisper 版本必须一致。
- `SHIM_PYTHON` 指向 Jittor 环境，`REAL_TORCH_PYTHON` 指向独立 PyTorch 环境。
  两侧 CPython minor 一致，Whisper 使用同一来源包站。不要把 Whisper 依赖直接装进 shim，
  避免 pip 装入真实 torch 覆盖 shim。环境变量沿用 `_ecosystem_harness.py` 的契约。
- 以下命令在仓库根目录、Linux bash 中执行。先按环境手册安装当前工作树的 core/compat。
  环境中旧 editable 路径必须更新；不能只修改工作目录。

```bash
set -euo pipefail
: "${SHIM_PYTHON:?预先配置 Jittor 解释器绝对路径}"
: "${REAL_TORCH_PYTHON:?预先配置独立 PyTorch 解释器绝对路径}"
: "${JITTOR_ECOSYSTEM_PACKAGE_SITE:?预先配置 Whisper 包站绝对路径}"
: "${JITTOR_LAB_ROOT:?预先配置仓库外实验目录}"
export SHIM_PYTHON REAL_TORCH_PYTHON JITTOR_ECOSYSTEM_PACKAGE_SITE JITTOR_LAB_ROOT
export JITTOR_ECOSYSTEM_REFERENCE_PACKAGE_SITE="$JITTOR_ECOSYSTEM_PACKAGE_SITE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/compat/shim/resources:$PWD/python"
export JITTOR_TORCH_SHIM=1 JT_BACKEND_FALLBACK=error
export JITTOR_REQUIRE_REAL_TORCH=1 JITTOR_REQUIRE_WHISPER=1
export JITTOR_TEST_REQUIRE_EXECUTION=1
export JITTOR_ECOSYSTEM_TF32=0 JITTOR_ECOSYSTEM_CUDNN_BENCHMARK=0
export JITTOR_ECOSYSTEM_SPEED_RATIO=''
```

先用 `--list-only` 确认两项都列出；仅清单检查，不会启动模型：

```bash
"$SHIM_PYTHON" agent/skills/torch-compat-repo-runbook/scripts/verify_repo.py \
  --repo openai-whisper --list-only
```

## 在 shim 上跑

CPU 的安装、固定 oracle、Whisper 与基础算子测试入口为：

```bash
python -m nox -s whisper_cpu
```

该会话固定为 CPU，并要求缺 oracle、缺 Whisper 或发生 skip 时失败。
## 对拍

- CPU bool COO：同文件 30 项边界单测 + 31 项独立 PyTorch 精确对拍，覆盖存储共享、复制、buffer 和错误协议；仅验证 CPU bool COO，不覆盖浮点 sparse。
- 主模型：FP32，mel `(2,80,32)`、tokens `(2,8)`，检查全部 88 个参数梯度及 mel 梯度。
- log-Mel：16000 点 FP32 音频，输出 `(80,100)`，检查音频梯度。
- 使用统一 harness 的状态清单/指纹、dtype、shape、数值、退出码和 fallback 检查。
  主模型沿用前向 `2e-3` / 反向 `1e-2`；log-Mel 沿用前向 `2e-5` / 反向 `2e-3`。
  具体绝对/相对误差算法以 harness 为准，不能为迁就失败而放宽容差。
## 断点与分流

| 事项 | 归属 | 当前范围 |
| --- | --- | --- |
| 可微 STFT / FFT | native core | CPU FP32 前反向有证据 |
| bool COO 存储 | native sparse | bool 非持久元数据范围 |
| layout、Module buffer、Torch STFT 参数 | jittor.compat.torch | 公共协议，不能在 Whisper 中绕过 |
| 原版 Whisper | 下游原实现 | 没有新增 adapter；发现 C5 先同步团队严重问题文档 |

## 坑与假绿

1. 合成小模型不等于官方 tiny，更不等于全任务支持。
2. 当前 L2 只覆盖固定 tiny config 的交叉熵和三步 SGD；不等于真实 checkpoint 保存加载和全部解码模式通过。
3. CPU tests 内部固定 `use_cuda=0`；本结果不能外推为 CUDA/NPU 支持。
4. 两侧依赖来源、线程数和 fallback 策略需一致；结果标记之后非零退出仍是失败。

## 证据

维护结论统一见 [Whisper CPU L2 验证记录](../../../docs/results/2026-09-22-openai-whisper-l2-cpu.md)。
原始日志、XML、冻结列表和同步备份在 `$JITTOR_LAB_ROOT/_state/whisper/`（未版本化）。
本记录只声明 CPU L0–L2；CUDA、NPU、显存和性能均未验证。同步到新提交后必须重新运行门禁并记录结果。
