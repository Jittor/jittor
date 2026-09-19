---
name: torch-compat-repo-runbook
description: 为一个下游 Torch 生态库写"在 jittor shim 上跑 + 与原生 torch 对比"runbook 的模板，以及签收所需的四轴验收协议（支持的模型清单、精度、显存、速度）。用于新增一个库、复核已有 runbook、或判断一次对拍是否真的有证据。
---

# 下游仓库 torch-compat Runbook 模板与四轴验收

**用途**：给一个下游库建立可信的"shim 跑 / 原生 torch 跑 / 两者对拍"文档，并用
四轴证据签收。`transformers`、`diffusers`、`vllm`、`mmcv` 等各自的具体断点在它们
自己的 `*-torch-compat` skill 里；本 skill 只管**格式、协议和验收标准**。

**不覆盖**：断点本身该修 core 还是 compat 还是 adapter——那走
[`downstream-library-adaptation`](../downstream-library-adaptation/SKILL.md)。
数值 harness 细节见 [`jittor-torch-diff`](../jittor-torch-diff/SKILL.md)。

## 一、Runbook 骨架（一个仓库一个 skill）

目录 `agent/skills/<lib>-torch-compat/SKILL.md`，章节固定为：

| 章节 | 必须回答 |
| --- | --- |
| `## 用途` | 这个 runbook 能做什么，**以及明确不覆盖什么** |
| `## 两侧环境` | shim / oracle 两个解释器、库版本、离线开关、缓存隔离、oracle 断言 |
| `## 在 shim 上跑` | 可直接复制执行的命令 |
| `## 在原生 torch 上跑` | 同上 |
| `## 对拍` | 同权重同输入；精度口径；速度口径；device 规则 |
| `## 断点与分流` | 已知断点，每条标注 core / `jittor.compat.torch` / adapter |
| `## 坑与假绿` | 本库特有的坑，以及**看起来像成功**的失败 |
| `## 证据` | 结果落在哪，以及**本机实测 vs 仅报告推导**的分界 |

## 二、两个解释器

真实 PyTorch 和 jittor shim 都抢占 `torch` 这个名字，所以对拍必须**两个解释器**：
oracle 产出权重和参考值，shim 从同一份权重重算。

- **oracle 断言（看任何数字之前必须过）**：
  `assert not hasattr(torch, '_torch_compat_install_context')`。
  不过就是拿 shim 对自己，什么都证明不了。
- **环境变量契约**以
  [`_ecosystem_harness.py`](../../../compat/tests/torch/_ecosystem_harness.py)
  的模块 docstring 为准（`REAL_TORCH_PYTHON`、`JITTOR_ECOSYSTEM_PACKAGE_SITE`、
  `JITTOR_ECOSYSTEM_REFERENCE_PACKAGE_SITE`、`JITTOR_ECOSYSTEM_SPEED_RATIO`、
  `JITTOR_ECOSYSTEM_TF32`、`JITTOR_ECOSYSTEM_CUDNN_BENCHMARK` 等）。**不要另造一套。**
- **两侧下游库版本必须一致**，否则是在比两个不同的库。
- **离线固定输入**：`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，上游 checkout 钉
  commit SHA。
- 缓存隔离：每个并行任务一个 `JITTOR_HOME`（或不同 `cache_name`）；首次 JIT/扩展编译
  串行跑。

## 三、四轴验收（签收标准）

只跑通不算接入。四轴都要有数：

| 轴 | 怎么量 | 通过标准 | 记录 |
| --- | --- | --- | --- |
| **支持的模型清单** | `_ecosystem_cases.CASES` 里该库注册的 case | 每个 case 都能跑；跑不了的写明缺什么依赖 | case 名 + 依赖 |
| **精度** | 同权重、同输入、同 device，两个解释器；逐输出/逐梯度比 | 误差不超过门禁容差；**相对误差用全场最大量级做 floor** | worst abs / worst rel |
| **显存** | 子进程 device 峰值，外部采样 | 与 oracle 同量级；不出现隐式 fp32 放大 | shim vs oracle MiB |
| **速度** | 重复交错采样，**取最小值** | 记录 ratio；**PR 门禁只报告不断言墙钟** | ratio |

两条容易踩的口径：

1. **精度的相对误差别按数组自己的量级除。** 一个接近 0 的梯度会报出 ~1 的相对误差，
   而绝对误差其实只有 1e-5——把正确结果读成失败。用整个场的最大量级做分母。
2. **显存只能外部采样。** shim 的 `torch.cuda.max_memory_allocated()` 目前恒返回 0
   （本机实测），所以它不能当 shim 侧的峰值；用 `nvidia-smi` 按 pid 采样，两侧才是
   同一把尺子，代价是采样可能漏掉真峰（报告里注明是下界）。

## 四、跑四轴

[`scripts/verify_repo.py`](scripts/verify_repo.py) 把单 case 执行委托给项目自己的
[`_ecosystem_runner.py`](../../../compat/tests/torch/_ecosystem_runner.py)，因此数字和
生态门禁同源，不是私有实现：

```bash
source <lab>/env-jittor.sh
REAL_TORCH_PYTHON=<oracle>/bin/python "$VENV/bin/python" \
  agent/skills/torch-compat-repo-runbook/scripts/verify_repo.py \
  --repo transformers --device cuda --repeats 5 \
  --out "$JITTOR_LAB_ROOT/_state/<topic>/verify/transformers"
```

- 先 `--list-only` 看该库的 case 清单，不必起进程。
- 输出 `verify-report.json`，每个 case 一行：精度、显存、速度、fallback、device 一致性。
- case 名用下划线而发行版用连字符（`ms-swift` → `ms_swift_lora_llama`），脚本已归一化。

## 五、本机没有该库时

`pip` 可用的机器上，把下游依赖装到一个**独立目录**当包站，再让两侧都指过去
（`JITTOR_ECOSYSTEM_PACKAGE_SITE` / `JITTOR_ECOSYSTEM_REFERENCE_PACKAGE_SITE`）：

```bash
pip install --target <site> <lib>==<version>
```

CPython minor 版本不同时默认拒绝共享包站，确实只走稳定 ABI 才显式加
`JITTOR_ECOSYSTEM_PACKAGE_SITE_CROSS_ABI=1`。

**绝不要把下游库装进 shim 解释器本身**：pip 可能把 jittor 的 `torch` 换成真 PyTorch，
那一次安装会作废之后所有对拍。

仓库自带两条门禁，优先复用而不是重写：

- `nox -s ecosystem`：十二个下游 case 的双解释器 parity + speed；要求
  `REAL_TORCH_PYTHON`，且是 fail-closed 的。
- `nox -s optional`：真 CUDA 上的可选兼容门禁，覆盖
  `OPTIONAL_COMPAT_PACKAGES`（`torchmetrics`、`mmcv`、`mmengine`、`peft`、
  `safetensors`、`tensordict`、`flash_attn`）。

## 六、假绿清单

1. **shim 对自己**：缺 oracle 时 case 自我 skip；用 `JITTOR_REQUIRE_REAL_TORCH=1`
   把这类 skip 变成失败。
2. **"跑 CPU" 其实不是 CPU**：jittor 没有 per-tensor device，有卡时 CUDA 默认就是
   开的，必须显式关。
3. **两侧版本不同**：比的是两个库，不是两个 runtime。
4. **签名齐全的 no-op**：见 [`torch-shim-noop-audit`](../torch-shim-noop-audit/SKILL.md)。
5. **只报告不测试**：runbook 里的命令没在本机跑过，就标"未验证"，别写成既成事实。

## 七、落盘

- 可复现结论写 `docs/results/YYYY-MM-DD-<topic>.md`：Status / 日期 / Baseline commit /
  Owner / Review when，以及环境、命令、结果、边界。
- 原始日志、包站、缓存、`verify-report.json` 留在 `$JITTOR_LAB_ROOT/_state/<topic>/`，
  不进主仓库。
- 加新 runbook 时，同时在该库 skill 的 `## 证据` 里写清"本机实测"还是"报告推导"。
