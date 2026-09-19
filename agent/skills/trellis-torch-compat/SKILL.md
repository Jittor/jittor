---
name: trellis-torch-compat
description: 在 Jittor torch shim 与独立 PyTorch 上运行和验证 TRELLIS.2 3D 生成 pipeline（含 CuMesh/FlexGEMM/nvdiffrast/o-voxel/FlashAttention 四个真实 CUDA 扩展）的 runbook，含断点分流、PLY mesh 数值对拍、重复交错速度协议与假绿清单。用于复验 TRELLIS 接入或推进其尚未通过的性能门禁。本 skill 为报告派生，未在本机复验。
---

# TRELLIS.2 的 shim ⇄ 原生 torch 对拍

## 用途

回答 TRELLIS.2 4B 在 shim 上怎么跑、原生 PyTorch 上怎么跑、怎么对拍几何输出与速度，以及
断点归 jittor 核心 / `jittor.compat.torch` / `jittor-trellis` adapter 哪一层。

**不覆盖，也不夸大**：本机没有 TRELLIS checkout、`jittor-trellis` adapter 或 harness，
两个 venv 都未安装 trellis（已实测）。本 skill **是报告派生的，未在本机复验**。
TRELLIS 的性能门禁**仍然开放**：报告里 Jittor 比真实 PyTorch 慢 `1.0926x`，不能声称达标；
也不宣称反向数值一致（它是 inference pipeline）或 NPU/ROCm 支持。

## 两侧环境

| 角色 | 解释器 / 入口 | 说明 |
| --- | --- | --- |
| shim | 验收机：Jittor `566d8087`，Python 3.11.15，CUDA 12.2.140，JIT compiler serial | RTX 4090（sm_89） |
| oracle | 真实 PyTorch `2.12.1`，同机 | TRELLIS 参考 `transformers 4.56.2` |

实测：`trellis` / `jittor_trellis` 在两个 venv 中均 **NOT INSTALLED**；
`/root/jittor-lab/` 下无 `trellis*`，`_state/` 下无 `trellis-current`；
`grep` 全仓无 trellis harness/case（只有报告与文档提及）。
版本身份（报告）：TRELLIS.2 checkout `75fbf0183001ed9876c8dbb35de6b68552ee08bd`，
`jittor-trellis` adapter `f2c23acdf2402abcf04222a4866fc87451efe959`；
vLLM `51a99565c398c8320de8131e07731c75c52eb87c`（若复跑同报告的其他库）。

**oracle 断言（任何数字之前必须过）**：

```bash
"$ORACLE_PY" -c \
  "import torch; assert not hasattr(torch, '_torch_compat_install_context'); print(torch.__version__)"
```

缓存隔离：一个 device 一套 `JITTOR_HOME`，一个进程一个 cache；报告明确每个独立进程、
extension/JIT cache 相互隔离，只租用一张可见 GPU。离线：`HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1`。双解释器 harness 契约见
[`_ecosystem_harness.py`](../../../compat/tests/torch/_ecosystem_harness.py) docstring；`CASES` 里没有 trellis。

## 在 shim 上跑

本机**没有可运行入口**。报告里的运行方式（验收机、未版本化）：

- 用当前 source checkout + adapter `src/` + 本地 checkpoint + offline 模型资产 + 隔离的
  extension/JIT cache + 一张可见 GPU。
- 四个真实 CUDA 扩展从隔离的外部状态加载：**CuMesh、FlexGEMM、nvdiffrast、o-voxel**，
  以及官方 FlashAttention 及其 packed forward adapter。
- 运行时选择：`flash_attn`、`flex_gemm`、sm_89、no managed allocator、no low-vram、
  tensorcore level 2。
- 基准与 launcher 的 SHA-256（报告）：`0dbbdf0f206ee80ff946f7d804f294521a93411ead16c58efcd03bcc2de3a77c`
  与 `2ac0260f502961d26b36aaba764741b2af303df9f1913c16bf2839d572366d76`。

重建按 [`downstream-library-adaptation`](../downstream-library-adaptation/SKILL.md)：
先裸跑收集断点。`jittor-trellis` 是第三行的 adapter，adapter 想动框架状态时应回 compat 开
公开接口，不要写私有属性。

## 在原生 torch 上跑

报告用同机真实 PyTorch `2.12.1` 跑同一 pipeline，独立进程，每进程 1 次 warmup + 3 次测量。
结果 JSON 的 SHA-256 为
`a9e9a5d8d8f575d3e0c36e21e9d98957ed5011a4994dc83ec7d6f02adecd6154`。
本机不可复现（无 checkout、无 GPU、无安装）。

## 对拍

- 数值（几何）：两侧都输出有效 PLY，用 20,000 sampled vertices per direction 比较——
  vertex 数差 `+3,368`（`0.226515%`）、face 数差 `+14,700`（`0.456760%`）、
  bounding-box extent L2 `9.9765e-5`（bbox 对角线 `7.0852e-5`）、centroid L2 `6.3595e-4`
  （`4.5164e-4`）、Jittor→Torch 最近点 mean `0.0058707`（`0.0041693`）、p95 `0.0104870`。
  参考帧张量/共享 tape 为 3,540 coordinates。
- 速度：报告用「3 个独立进程 ×（1 warmup + 3 measured），取 median-of-medians」。

  | 运行时 | 进程中位数 | median-of-medians |
  | --- | --- | ---: |
  | Jittor | `7.6291 / 7.5149 / 7.4246 s` | **`7.5149 s`** |
  | PyTorch | `6.7972 / 6.8778 / 6.8979 s` | **`6.8778 s`** |

  比值 `1.0926x`：Jittor 从本轮起点 `8.2843 s` 改善 `9.3%`，但仍未达到"不慢于参考"，
  所以性能**未接受**。project-context 记 TRELLIS.2 从约 `1.20x` 改善到 `1.093x`。
  Cold build 与每进程首次 warmup 排除。墙钟上界只在 nightly 断言，PR 不 assert。
- Device 规则：两侧同 device；Jittor 无 per-tensor device，CUDA 默认开，"CPU" 要显式请求；
  使用同一张可见 GPU 串行跑两侧。

## 断点与分流

| 断点 | 层 |
| --- | --- |
| BF16 multi-head RMSNorm：head dim ≤256 每行一个 warp | **jittor core**（保留 BF16 rounding point） |
| 单 packed self-attention kernel：合并 Q/K RMSNorm、pairwise-complex RoPE、V passthrough | **jittor core** |
| fused non-affine LayerNorm + scale/shift（精确 sparse modulated block） | **jittor core** |
| lazy-CUDA device fallback、稳定 `Var.id` cache 身份 | **`jittor-trellis` adapter**（第三行，外置/未版本化） |
| FlashAttention packed forward 路由 | **adapter + 环境变量契约**（`JITTOR_FLASH_ATTN_*`） |

被否决并已撤回、不要重做的 A/B：mixed-length Q/KV RMS、residual-plus-LayerNorm、默认
cross-KV cache、C2S topology cache；历史 cublasLt/GELU/RoPE 实验在负结果后未重跑。

## 坑与假绿

1. **性能门禁开着**：`1.0926x` 不是通过；报告定位剩余 gap 集中在 BF16 linear epilogue /
   fused elementwise（`4.2 it/s` 形状与 `7.4 it/s` texture 阶段），**不在** FlashAttention
   或 GEMM kernel 本身。不要用"pipeline 跑通"替换性能结论。
2. **只看"写出 PLY"会假绿**：两侧都写有效 mesh 也可能有结构性差异，必须做上面的几何容差
   比较（vertex/face 数、bbox、centroid、最近点）。
3. **单次计时不可用**：用重复独立进程取 median-of-medians；进程内首次样本含 JIT，排除。
4. **扩展不是同一个来源**：CuMesh/FlexGEMM/nvdiffrast/o-voxel/FlashAttention 都是真实 CUDA
   扩展，必须从隔离状态加载并记录运行时选择的 backend（`flash_attn`、`flex_gemm`、sm_89…），
   否则是在比较不同后端。
5. **adapter 未版本化**：报告里的 `jittor-trellis` 与脚本是验收机产物，SHA 只用于身份，
   不要照抄路径当本机命令。
6. **本 skill 的所有数字都不可复现于本机**——当复验指引读，不要当"已验证"。

## 证据

- 报告：[`2026-08-23-verl-vllm-trellis-current-baseline.md`](../../../refactor-wip/results/2026-08-23-verl-vllm-trellis-current-baseline.md)。
- 状态索引：[`project-context.md`](../../manuals/project-context.md)（TRELLIS.2 约 `1.20x` → `1.093x`，
  性能门禁仍开放）。
- **本机已核实**：`CASES` 无 trellis；`/root/jittor-lab/` 与 `_state/` 下无 trellis checkout；
  两个 venv 均未安装 trellis。
- **仅报告 / 未在本机复验**：全部数值、速度、命令、扩展加载与 mesh 比较。

## 实测（2026-09-19）

仓库 HEAD `90fe0b9`。**本机环境缺席**：`find_spec('trellis')`/`find_spec('jittor_trellis')`
在 `venv-jittor` 与 `venv-oracle-cu129` 都为 False；`/root/jittor-lab/` 与 `_state/` 下无
`trellis*` checkout；`CASES` 无 trellis；`verify_repo.py --repo trellis --list-only` 无输出。
因此**未跑任何命令**，也未加载四个 CUDA 扩展。

**能核对与不能核对的**：读到并核对了
`refactor-wip/results/2026-08-23-verl-vllm-trellis-current-baseline.md`（277 行），
skill 引用的数字都在：三进程中位数 Jittor `7.6291 / 7.5149 / 7.4246s`、
median-of-medians `7.5149s`，PyTorch `6.7972 / 6.8778 / 6.8979s` → `6.8778s`，
比值 `1.0926x`、从同轮起点 `8.2843s` 改善 `9.3%`；几何对照 `+3,368`（`0.226515%`）vertex、
`+14,700`（`0.456760%`）face、bbox extent L2 `9.9765e-5`、centroid L2 `6.3595e-4`、
最近点 mean `0.0058707`、p95 `0.0104870`；共享 tape `3,540` coordinates；
checkout `75fbf018...`、adapter `f2c23acd...`。
`agent/manuals/project-context.md:114` 确有「TRELLIS.2 从约 `1.20x` 改善到 `1.093x`」。
**不能核对**：任何可运行命令、运行时后端选择（`flash_attn`/`flex_gemm`/sm_89）、PLY 比较、
计时与两个 SHA-256 的产物。

**四轴**：全部未测——支持的 case、精度、显存、速度在本机都没有可运行载体；性能门禁
`1.0926x` 仍开放，本机不能声称达标或复现。

**证据状态**：**报告派生**（report-derived）。报告存在且内容与 skill 一致这一点已核对；
数值/速度本身未在本机复现。
