---
name: verl-torch-compat
description: 在 Jittor torch shim 与独立 PyTorch/torch_npu 上运行和验证 verl（PPO 核心算法、FSDP2 训练、权重传输）的 runbook，含断点分流、对拍协议与假绿清单。用于复验 verl 接入、定位 verl 在 shim 上的 clamp/autograd/version-identity/分布式断点，或重建 verl 对拍环境。本 skill 为报告派生，未在本机复验。
---

# verl 的 shim ⇄ 原生 torch 对拍

## 用途

回答 verl 在 shim 上怎么跑、原生 PyTorch / `torch_npu` 上怎么跑、怎么对拍数值与速度，
以及断点归 jittor 核心 / `jittor.compat.torch` / adapter 哪一层。

**不覆盖，也不夸大**：本机（当前 checkout）**没有** verl 的 in-repo case 或 harness，
`/root/jittor-lab/verl_jittor/` 也不存在，两个 venv 都没有安装 verl（均已 grep / 实测）。
因此本 skill **是报告派生的，未在本机复验**；命令与数字都应回到报告核对，不要当成
本机可跑的已验证结论。完整安装、Ray worker、rollout、1-step PPO、NPU/ROCm 均由报告
分别限定，不要跨报告外推。

## 两侧环境

| 角色 | 解释器 / 入口 | 说明 |
| --- | --- | --- |
| shim | `/root/jittor-lab/_state/h3/venv-jittor/bin/python`，`source /root/jittor-lab/minimax-h3/env-jittor.sh` | 本机有该 shim 环境，但**没有** verl 源码/安装 |
| oracle（CUDA） | 报告：真实 RTX 4090（单机最多 4 张）+ Jittor `566d8087`/`40e7df7d` | verl `3d66a3d7ca1cf783df949816ec6862d5a7af9406` |
| oracle（NPU） | 报告：Ascend 910B3 + `torch_npu`，Jittor `97dc6ce9`（源码行为 `3758c4ab`） | 单卡串行对照 |

实测：`verl` / `jittor_trellis` / `trellis` 在两个 venv 中均 **NOT INSTALLED**；
`grep` 全仓无 verl 的 harness/case（只有报告与文档提及）。
双解释器 harness 的环境变量契约见
[`_ecosystem_harness.py`](../../../compat/tests/torch/_ecosystem_harness.py) docstring，runner CLI 见
[`_ecosystem_runner.py`](../../../compat/tests/torch/_ecosystem_runner.py)；`CASES` 里没有 verl。

**oracle 断言（任何数字之前必须过，两侧同款）**：

```bash
"$ORACLE_PY" -c \
  "import torch; assert not hasattr(torch, '_torch_compat_install_context'); print(torch.__version__)"
```

缓存隔离：一个 device 一套 `JITTOR_HOME`；报告里每个 Ray worker 用独立 cache，Ray actor
设 `DISABLE_MULTIPROCESSING=1`，operator compiler 保持串行（CPU/CUDA 测试与 benchmark 不共享
编译缓存）。离线：`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`。本 lab 的 shim 环境变量见
[`vllm-omni-torch-compat`](../vllm-omni-torch-compat/SKILL.md)（同一 `env-jittor.sh`）。

## 在 shim 上跑

本机**没有可运行入口**。报告里的入口（均在验收机、未版本化）：

- `$JITTOR_LAB_ROOT/verl_jittor/scripts/run_all.sh`：14 个 stage，依次
  `py_compile import_scan ppo_config_smoke protocol_smoke protocol_extended_smoke
  protocol_v2_nested_smoke torch_functional_extended_smoke core_algos_smoke
  core_algos_extended_smoke core_algos_parity_cpu fsdp2_smoke jittor_torch_compat_ops
  core_algos_parity_cuda cuda_sanity`；`--cpu-only` 时跳过 CUDA。
- `scripts/weight_transfer_smoke.py`、`scripts/vllm_weight_apply_smoke.py`
  （设 `VERL_VLLM_MODEL` 时由 `run_all.sh` 追加）、`scripts/ray_weight_transfer_smoke.py`。
- 原生分布式 PPO 门禁：`CUDA_VISIBLE_DEVICES=2,3 python -m nox -s nccl`，
  四卡用 `JITTOR_NCCL_WORLD_SIZE=4`。

重建流程按 [`downstream-library-adaptation`](../downstream-library-adaptation/SKILL.md)：
先裸跑收集断点、按三行判据分流，**不要先写 adapter**。

## 在原生 torch 上跑

报告的原生参考是同一张 910B3 上的 `torch_npu`（NPU 报告）或真实 PyTorch CUDA（PPO 报告），
都在验收机上，本机不可复现。PPO harness 的 Hydra 配置与 Ray worker 日志保存在
`$JITTOR_LAB_ROOT/_state/verl-ppo/` 与 `_state/verl-fsdp2/` 下（报告给出各 SHA-256），
本机不存在这些目录。

## 对拍

- 数值：
  - NPU policy loss：vanilla PPO / GSPO / SAPO / GPG / geometric-mean / CISPO 的 loss 与
    `log_prob` 梯度、GRPO advantage/returns，最大绝对误差全为 `0`，输出驻留 device、
    `fallback_count=0`、未加载 `torch_npu` 之外的回退。
  - CUDA CPU/CUDA 算法矩阵：forward max `2.384e-7`；gradient max `2.794e-9`（CPU）/
    `1.863e-9`（CUDA）。
  - 1-step PPO rollout 与训练侧概率：tiny/Qwen3 各配置 Pearson `0.99998`–`0.99999`。
- 速度（只在 nightly 断言上界，PR 不 assert）：
  - NPU 微协议（`64x120`/`512x512`）六条路径只有 GPG 不慢于原生（`0.776x`/`0.774x`），
    其余 `1.25x`–`1.75x`。
  - CUDA 的 `395.99s/step`、四卡 `1086.65s/529.15s` 含首次 shape-specialized JIT，
    **只作功能门禁，不是性能**。
- Device 规则：两侧同 device；Jittor 无 per-tensor device，CUDA 默认开，"CPU" 要显式请求。
  NPU 结论必须在 NPU 侧对 `torch_npu` 复验，不能由 CPU/CUDA 外推（见「坑与假绿」）。

## 断点与分流

| 断点 | 层 |
| --- | --- |
| verl 按 `torch.__version__` 分支，DTensorSpec 注解未导入而失败 | **`jittor.compat.torch`** 的版本身份契约：Jittor 在 `torch.__version__` 保后端版本、另报 `torch.__torch_version__`；修复在 `verl/utils/fsdp_utils.py` 改为 `getattr(torch, "__torch_version__", torch.__version__)`（下游 checkout 改动，外置/未版本化，不是 core） |
| `clamp(x, lo, hi)` 在精确边界梯度为 0；reversed scalar bounds 断言失败；FP16/BF16 scalar bounds；NaN | **jittor core**（`fdae6a0f`、`3758c4ab`） |
| Adam bias-correction 除法把 float32 二阶矩提升为 float64 | **jittor core / `jittor.compat.torch`**（`5267eba3`） |
| `NanoVector.numel()`、TensorDict device | **`jittor.compat.torch`** |
| 复合 backend `cpu:gloo,cuda:nccl` 匹配；all-gather 反向；Store/c10d/FSDP2 ABC、`Module.to_empty()`、forward unshard hook、`clip_grad` 私有入口 | **jittor core（distributed）+ compat** |
| Ray 动态 NCCL bootstrap | **jittor core**，opt-in `JITTOR_TORCH_DISTRIBUTED_AUTO_INIT=1` |

## 坑与假绿

1. **metrics `.item()` 切碎惰性图**：原生三项 metrics 只 +0.060 ms，Jittor +1.103 ms；把
   metrics 去掉后 Jittor 为 `0.989x`。剩余差距不是 clamp / loss / backward 计算。
2. **冷 JIT 数字不能当性能**：`395.99s/step`、`1352.93s/step`、四卡 update 秒数都含首次
   shape-specialized JIT。
3. **replicated 模式**只让 rank 0 回传，rank 1 仍执行；旧 collect 会把 `142` 拼成 `284`。
4. **EngineCore Python spawn 恢复父 `sys.path`**，同时加载 `device-2_3` 与子进程 device cache
   的两份 `jittor_core.so` → `Op fused not found`；loader 必须先移除继承路径。子进程按 local
   rank 映射成 `device-2`/`device-3` 独立 cache。
5. **只跑 CPU 不够**：NPU 结论必须 NPU 复验；`import_scan` 只接受 Megatron、
   `transfer_queue`、TorchTitan 三种明确缺环境，其余缺失算失败。
6. **假绿**：只看 `training/global_step=1` 不够，要同时看 actor/critic grad norm 非空、rollout
   与训练概率差门禁、权重同步真的发生；否则可能是一个未被使用的参数表被更新。
7. **本 skill 的所有数字都不可复现于本机**——当复验指引读，不要当"已验证"。

## 证据

- 报告：[`2026-09-02-verl-ascend-core-algorithms.md`](../../../refactor-wip/results/2026-09-02-verl-ascend-core-algorithms.md)（NPU）、
  [`2026-08-24-verl-weight-transfer.md`](../../../refactor-wip/results/2026-08-24-verl-weight-transfer.md)（CUDA PPO/权重传输）、
  [`2026-08-23-verl-vllm-trellis-current-baseline.md`](../../../refactor-wip/results/2026-08-23-verl-vllm-trellis-current-baseline.md)（当前基线）。
- 状态索引：[`project-context.md`](../../manuals/project-context.md) 第 105–110 行。
- **本机已核实**：`CASES` 无 verl；`/root/jittor-lab/` 下无 `verl_jittor`；两个 venv 均未安装
  verl。
- **仅报告 / 未在本机复验**：全部数值、速度、命令与分布式门禁。

## 实测（2026-09-19）

仓库 HEAD `90fe0b9`。**本机环境缺席**：`find_spec('verl')` 在
`venv-jittor` 与 `venv-oracle-cu129` 都为 False；`/root/jittor-lab/verl_jittor` 不存在；
`_ecosystem_cases.py` 的 `CASES` 无 verl；`verify_repo.py --repo verl --list-only`
无输出。因此**没有可运行入口，未跑任何命令**（未安装、未训练、未起 Ray worker）。

**能核对与不能核对的**：读到并核对了 skill 引用的报告存在且确有对应数字：

- `refactor-wip/results/2026-09-02-verl-ascend-core-algorithms.md`（184 行）：六类
  policy loss/gradient 与 GRPO advantage 最大绝对误差 `0` 的表（vanilla PPO/GSPO/
  SAPO/GPG/geometric-mean/CISPO，GRPO=0），NPU 微协议 GPG `0.776x`/`0.774x`、
  其余 `1.25x-1.75x`。
- `refactor-wip/results/2026-08-24-verl-weight-transfer.md`（316 行）：CPU/CUDA 梯度
  `2.794e-9`/`1.863e-9`、forward `2.384e-7`；`395.99s/step` 只作功能门禁；
  `1086.65s/529.15s` critic/actor update。
- `refactor-wip/results/2026-08-23-verl-vllm-trellis-current-baseline.md`：CPU/CUDA 算法
  矩阵 `2.384e-7` / `2.794e-9`、`1.863e-9`。
- `agent/manuals/project-context.md` 第 105–110 行确有「910B3 上损失/梯度精确对拍、只有
  GPG 过 NPU 微性能协议」的表述。

**不能核对**：所有命令的可执行性、任何数值/速度/分布式门禁、Hydra 配置与 Ray 日志
（报告给出的 `_state/verl-ppo/`、`_state/verl-fsdp2/` 在本机不存在）。

**四轴**：全部未测——支持的 case、精度、显存、速度在本机都没有可运行载体。

**证据状态**：**报告派生**（report-derived）。报告文件存在且内容与 skill 一致这一点已核对；
数值本身未在本机复现。
