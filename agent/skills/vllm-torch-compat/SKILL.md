---
name: vllm-torch-compat
description: 在 Jittor torch shim 与独立 PyTorch 两套解释器上运行和验证 vLLM 的 runbook——怎么跑 shim、怎么跑 oracle、怎么对拍数值与速度，以及断点该落 jittor 核心 / jittor.compat.torch / adapter 的哪一层。用于修改 adapters/jittor_adapters/vllm/、复验 vLLM 接入或排查 vLLM 在 shim 上的 import/执行问题。
---

# vLLM 的 shim ⇄ 原生 torch 对拍

## 用途

回答 vLLM 在 shim 上怎么跑、原生 PyTorch 上怎么跑、两者怎么比数值和速度，以及每个
断点的归属层。vLLM 是
[`downstream-library-adaptation`](../downstream-library-adaptation/SKILL.md) 分类表里
**合法需要独立 adapter 的第三行**（库私有实现细节），本 skill 按那个框架执行。

**不覆盖**：不新增 vLLM 版本支持声明（[`adapters/README.md`](../../../adapters/README.md)
明确不为当前安装版本背书）；不做 serving/训练性能验收。本机没有 vLLM 源码 checkout，
任何 engine 级结论都没有在本机复现，见「证据」。

## 两侧环境

| 角色 | 解释器 / 入口 | 说明 |
| --- | --- | --- |
| shim | `/root/jittor-lab/_state/h3/venv-jittor/bin/python`，`source /root/jittor-lab/minimax-h3/env-jittor.sh` | `import torch` 解析到 Jittor shim |
| oracle | `/root/jittor-lab/_state/h3/venv-oracle-cu129/bin/python`，`source /root/jittor-lab/minimax-h3/env-oracle-cu129.sh` | 独立二进制 PyTorch |

本机实测（2026-09-19，`importlib.metadata.version`）：`vllm 0.29.0` 两侧相同；
`transformers 5.5.3` 两侧相同；`torch 2.11.0`（shim）对 `2.13.0+cu129`（oracle）。
注意 `.../_state/h3/venv-oracle` 是另一个环境（`transformers 5.17.0`、`torch 2.13.0`），
**不是**可用的 oracle pair。

**oracle 断言（看任何数字之前必须过）**：

```bash
source /root/jittor-lab/minimax-h3/env-oracle-cu129.sh
"$VENV/bin/python" -c \
  "import torch; assert not hasattr(torch, '_torch_compat_install_context'); print(torch.__version__)"
```

本机实测输出 `2.13.0+cu129`。shim 侧同一属性为 `True`（实测），即 `import torch` 确实
解析到 shim；shim 侧必须先 `import jittor` 再 `import torch`（否则留给 shim 安装器的是
半初始化的 torch，会 fail-closed 拒绝）。本机实测 shim 的 `torch.__version__` 是后端版本
`1.3.11.0`，模拟 API level 在 `torch.__torch_version__` = `2.11.0`，且
`torch.compat_report_torch_api_version` 存在；adapter 必须走这个公开入口而不是自己写
`torch.__version__`。

离线：两套环境都设 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`。缓存：一个 device 一套
`JITTOR_HOME`；shim 的 JIT 缓存在 `env-jittor.sh` 的 `$JITTOR_HOME`。两侧不共享编译缓存。
双解释器的环境变量契约与 runner CLI 见
[`_ecosystem_harness.py`](../../../compat/tests/torch/_ecosystem_harness.py) 的 docstring 与
[`_ecosystem_runner.py`](../../../compat/tests/torch/_ecosystem_runner.py)，此处不复述。

## 在 shim 上跑

adapter 的结构与生命周期测试（不编译、不 JIT；本机实测 `9 passed in 6.57s`）：

```bash
cd /apdcephfs_private/qy/projects/zy/jittor
JITTOR_VLLM_HOST_ONLY=1 JITTOR_TORCH_SHIM=1 PYTHONPATH=adapters \
  python -m pytest -q \
    adapters/tests/vllm/test_plugin_lifecycle.py \
    adapters/tests/vllm/test_arming_ownership.py \
    adapters/tests/vllm/test_runtime_ownership.py
```

`JITTOR_TORCH_SHIM=1` 不可省：本机实测缺它会被仓库 pytest guard 直接拒绝（报
"these paths run under Torch compatibility mode"，是 ERROR 不是 FAILED）。

有真实后端时的数值/结构测试在 [`adapters/tests/vllm/`](../../../adapters/tests/vllm/)
（`test_structure.py`、`test_flash_attn.py`、`test_backend.py`、`test_layers.py`）；先读
[`conftest.py`](../../../adapters/tests/vllm/conftest.py) 的 `JITTOR_VLLM_HOST_ONLY` 开关。
结构门禁要求 adapter 只经公开入口 import jittor、不碰私有属性、不给 `jt`/`jittor`/`torch`
赋值——`test_structure.py` 里带负向用例。

在真实 vLLM checkout 上跑：本机没有源码 checkout，报告里的外置 adapter 已不在此仓/此机。
`import jittor_adapters.vllm` 本身不激活任何东西，必须经 `jittor_vllm` entry point 调
[`register`](../../../adapters/jittor_adapters/vllm/registration.py)。

## 在原生 torch 上跑

```bash
source /root/jittor-lab/minimax-h3/env-oracle-cu129.sh
"$VENV/bin/python" -c \
  "import torch; assert not hasattr(torch, '_torch_compat_install_context'); print(torch.__version__)"
```

vLLM 在 oracle 里也是 pip 安装的 0.29.0（本机实测），没有源码 checkout 可复验报告中的
Qwen3 engine 协议。双解释器 runner 的 oracle 侧由 harness 拉起，会清掉 `PYTHONPATH`、
`JITTOR_TORCH_SHIM`、`JITTOR_HOME` 等变量，不会把本 checkout pin 上去。

## 对拍

**本仓没有 `vllm` case**：`CASES`（[`_ecosystem_cases.py`](../../../compat/tests/torch/_ecosystem_cases.py)）
只有 transformers/diffusers/peft/mmcv/mmengine/ms-swift。因此通用双解释器 harness
**目前不覆盖 vLLM**；不要用生态门禁的绿灯声称 vLLM 数值通过。

- 数值：只能引报告。Qwen3-0.6B 4-token greedy 为 `[12095, 13, 576, 6722]`
  （` Paris. The capital`），CUDA 报告与 Ascend 报告两端一致。
- 速度：报告协议是「3 个独立进程、每进程 3 次 warmup + 21 次测量、取中位数」。
  CUDA 4-token 热态 `0.10924s` 对原生 transformers `0.13745s`（`0.7947x`，Jittor 快约
  20.5%）；Ascend 单请求受限协议 `0.36330s` 对 `0.38998s`。本机未复验。
- Device 规则：两侧必须同 device。Jittor 无 per-tensor device，CUDA 默认开；"CPU" 必须
  显式请求（harness 用 `jt.runtime.scope(use_cuda=0)`）并回读 `report["device"]` 断言，
  否则是在加速卡上对 CPU，数字照样"对"。
- 单次样本在此不可用；重复交错取最小值，理由见
  [`jittor-torch-diff`](../jittor-torch-diff/SKILL.md)。

## 断点与分流

| 断点 | 层 | 依据 |
| --- | --- | --- |
| `vllm._C` / `_moe_C` / `_vllm_fa2_C` 等编译扩展、`torch.ops._C` 算子 | **adapter** | 库私有实现，非 torch API；[`bootstrap.py`](../../../adapters/jittor_adapters/vllm/bootstrap.py) 做成 importable-but-empty 并用公开 Jittor primitive 补算子 |
| layer / attention patch（RMSNorm、ServingAttention 等） | **adapter** | 指向 `jt.nn` 公开 primitive，不碰私有属性 |
| vLLM 从 `torch.__version__` 推断 API level | **`jittor.compat.torch`** | 公开入口 `compat_report_torch_api_version`；adapter 不得写 `torch.__version__` |
| `torch.library.infer_schema` / `Library` / `ops`、distributed `Store`、FakeTensor 边界 | **jittor core / compat** | 缺功能或缺拼写 |
| paged attention、RMSNorm/RoPE/SwiGLU serving primitive | **jittor core** | 真缺功能（Ascend 侧） |
| flash-attn 路由 | **adapter + 环境变量契约** | `JITTOR_FLASH_ATTN_*` |

## 坑与假绿

1. `import jittor_adapters.vllm` 看着像"接好了"——它什么都不做，要经 `register`/`arm`。
2. 编译扩展是 **importable-but-empty stub**：缺一个算子不报错，只拿错结果跑到输出。必须
   按下游实际调用的算子做差集（`downstream-library-adaptation` 第 6.7 条）。
3. adapter 测试缺 `JITTOR_TORCH_SHIM=1` 会 ERROR，容易被误读成环境坏了。
4. 报告里的版本是历史 checkout（本地 dist-info `0.11.0`、Ascend `0.20.2`），本机装的是
   `0.29.0`，不要拿报告数字当 `0.29.0` 的结论。
5. `vllm.__version__` 可能来自本地 dist-info，不等于源码 commit；报告明确以 commit SHA
   为权威身份。
6. Device 假绿：GPU 机上不显式关 CUDA 的 "CPU" 对拍实际是加速卡对 CPU。

## 证据

- 报告：[`2026-08-31-vllm-ascend-jittor-bootstrap.md`](../../../refactor-wip/results/2026-08-31-vllm-ascend-jittor-bootstrap.md)（NPU）、
  [`2026-09-08-vllm-independent-distribution.md`](../../../refactor-wip/results/2026-09-08-vllm-independent-distribution.md)（adapter 抽取进仓）、
  [`2026-08-23-verl-vllm-trellis-current-baseline.md`](../../../refactor-wip/results/2026-08-23-verl-vllm-trellis-current-baseline.md)（CUDA）。
- 准入与版本矩阵：[`adapters/README.md`](../../../adapters/README.md)、
  [`_common.py`](../../../adapters/jittor_adapters/_common.py)。
- **本机已核实**：两侧 venv 的 vllm/transformers/torch 版本、oracle 断言、shim 身份、
  adapter 文件与 `jittor_vllm` entry point 存在、host-only 生命周期测试
  `9 passed in 6.57s`；本机无 in-repo `vllm` case。
- **仅报告 / 未在本机复验**：任何 vLLM engine 执行、Qwen3 token 与速度数字、MoE oracle、
  FlashAttention 路由、Ascend/NPU 结果。

## 实测（2026-09-19）

环境：仓库 HEAD `90fe0b9`；GPU 5；oracle 断言 `2.13.0+cu129`；两侧 `vllm 0.29.0`、
`transformers 5.5.3`。

**case 清单**：`verify_repo.py --repo vllm --list-only` **无任何输出**（`CASES` 无 vllm），
与 skill 所述一致；不宣称 vLLM 进了双解释器生态门禁。

**任务给的命令**（`JITTOR_VLLM_HOST_ONLY=1 JITTOR_TORCH_SHIM=1 PYTHONPATH=adapters`，
跑整个 `adapters/tests/vllm/`）：

```
27 failed, 25 passed, 1 error, 37 subtests passed in 1.45s
```

按文件：`test_backend.py` 7 failed、`test_flash_attn.py` 6 failed、`test_layers.py` 4 failed、
`test_structure.py` 10 SUBFAILED、`test_plugin_lifecycle.py` 1 error。根因分三类，都属于
**HOST_ONLY 桩的限制，不是 adapter 运行时行为**：

- `test_backend/test_layers/test_structure`：`conftest.py` 把 `jittor` 换成只有
  `jittor.compat` 的桩，于是 `compat/_aliases.py:8` 报
  `ModuleNotFoundError: No module named 'jittor._runtime'`。
- `test_flash_attn/test_structure`：桩里没有真 primitive，报
  `module 'jittor' has no attribute 'array'/'zeros'/'nn'`；另一条是
  `No module named 'vllm.vllm_flash_attn.layers'`。
- `test_plugin_lifecycle` 1 error：
  `TransactionConflict: runtime hook lost module 'vllm.vllm_flash_attn...'`。

**skill 正文那三条生命周期文件**（`test_plugin_lifecycle/test_arming_ownership/
test_runtime_ownership`）单独跑：**`9 passed in 0.67s`**（skill 记 6.57s，现为热态），
与 skill 断言一致。缺 `JITTOR_TORCH_SHIM=1` 会被仓库 guard 直接
`ERROR: these paths run under Torch compatibility mode`（本机复测）——不是 FAILED。

**另试**：不设 `JITTOR_VLLM_HOST_ONLY`（真后端）跑同一目录，source `env-jittor.sh` 后
JIT 编译超过 10 分钟仍未完成，已在有界窗口内停止，**未取到结果**。

**四轴**：支持的 case——无 ecosystem case；只跑了 adapter 的结构/生命周期测试；
精度/显存/速度——均未测（无 case、无 speed case）。

**证据状态**：三条生命周期契约机器验证通过（9 passed）；任务给的全目录 HOST_ONLY 结果主要
反映 HOST_ONLY 桩的覆盖边界。真后端数值/engine/速度数字仍**仅报告**。
