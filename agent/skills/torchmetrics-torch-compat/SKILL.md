---
name: torchmetrics-torch-compat
description: 在 Jittor 的 torch shim 上运行 TorchMetrics 并与原生 PyTorch 的期望值对照，用于验证分类/回归/聚合指标与所需 torch API，并说明 torchmetrics adapter 的作用与启用条件；不覆盖分布式、多 device 分布式指标和 two-interpreter 速度。
---

# TorchMetrics × Jittor torch shim

## 用途

回答「TorchMetrics 架在 `import torch` → Jittor 上能否跑、指标值对不对、需不需要 adapter」。
本仓覆盖文件：`../../../compat/tests/torch/test_torchmetrics_compat.py`；adapter：
`../../../adapters/jittor_adapters/torchmetrics.py`；报告：
`refactor-wip/results/2026-08-24-optional-compat-cuda-gate.md`。

**不覆盖**：分布式/多 device 聚合指标、`MetricCollection` 的高级状态、可视化，以及
two-interpreter 的速度对拍。

**注意**：TorchMetrics **没有进 ecosystem 的 two-interpreter harness**
（`../../../compat/tests/torch/_ecosystem_cases.py` 的 `CASES` 里没有 torchmetrics），
数值由单解释器测试内**硬编码的期望常数**充当 oracle，device 由 `nox -s optional` 的真实
CUDA 会话提供。这点与 peft/ms-swift 不同，不要照搬「同权重重算」的说法。

## 两侧环境

本机（`/root/jittor-lab`）已实测：

| 侧 | 解释器 | 关键事实 |
| --- | --- | --- |
| shim | `/root/jittor-lab/_state/h3/venv-jittor/bin/python` | py3.12.12；jittor 1.3.11.0；`import torch` → shim；**`jittor-torch-adapters` 1.3.11.0 已装（editable，指向本仓 `adapters/`）**；transformers 5.5.3 |
| 原生 torch | `/root/jittor-lab/_state/h3/venv-oracle-cu129/bin/python` | py3.12.12；torch 2.13.0+cu129；无 jittor |

先 `source /root/jittor-lab/minimax-h3/env-jittor.sh`（原生
`env-oracle-cu129.sh`）：导出 `JITTOR_TORCH_SHIM=1`、缓存收进
`$JITTOR_LAB_ROOT/_state/h3/run`、离线标志，并设
`JT_BUILD_PYTHON_CONFIG_PATH=/opt/python3.12/bin/python3.12-config`——**不设它
`import jittor` 直接 `RuntimeError: python3.12-config not found`**（实测）。

**看任何数字前先过 oracle 断言**：

```bash
/root/jittor-lab/_state/h3/venv-oracle-cu129/bin/python -c \
  "import torch; assert not hasattr(torch,'_torch_compat_install_context'), 'oracle 是 shim'; print('oracle', torch.__version__)"
# 本机实测：oracle 2.13.0+cu129
```

**本机 torchmetrics 两侧都未安装**（`PackageNotFoundError`，`find_spec('torchmetrics')`
为 False）。补上之前本 skill 的 torchmetrics 数字都是未验证。

## 在 shim 上跑

单测不依赖 `REAL_TORCH_PYTHON`，直接（缺 torchmetrics 时整类 skip）：

```bash
set -a; source /root/jittor-lab/minimax-h3/env-jittor.sh >/dev/null 2>&1; set +a
unset PYTHONPATH
cd /apdcephfs_private/qy/projects/zy/jittor
"$VENV/bin/python" -m pytest -q compat/tests/torch/test_torchmetrics_compat.py
```

覆盖四组：`classification`、`regression`、`aggregation`、以及
`test_torchmetrics_required_torch_ops`（`is_mps/is_xpu/is_meta`、`bincount`、
`dim_zero_cat`、`_safe_divide`、`trapz`）。期望值是测试里写死的常数（例如
`MeanSquaredError 0.1220000014`、`SpearmanCorrCoef 0.8999995589`）。

**环境怎么来**：`nox -s optional` 不安装包，它要求解释器里已经预置
`OPTIONAL_COMPAT_PACKAGES`（`noxfile.py:234`：torchmetrics / mmcv / mmengine / peft /
safetensors / tensordict / flash_attn），缺任何一个直接 fail-closed 报错。补齐办法
（**本机未执行**）：按 adapter 的 `SUPPORTED_VERSIONS` 装 **torchmetrics==1.7.4**，
放进 shim 解释器可导入的位置；`JITTOR_REQUIRE_OPTIONAL_DEPS=1` 让缺包变成失败而不是 skip。

```bash
# 维护者真机 CUDA 档（需 nvcc + 预置包），本机未验证
python -m nox -s optional -- compat/tests/torch/test_torchmetrics_compat.py
```

## 在原生 torch 上跑

TorchMetrics 测试本身按 shim 语义写（`forward`/`jittor.compat` 行为、adapter 断言）。
原生侧没有等价的仓库测试；要拿原生值只能在 oracle 解释器里手工跑同样的 tensor/指标，
或用 `refactor-wip/results/2026-08-24-optional-compat-cuda-gate.md` 里记录的期望常数作对照。
**不要**把 `test_torchmetrics_compat.py` 拿到 oracle 解释器跑：它依赖 shim 的 adapter 标记，
在原生 Torch 下这些断言本就不适用。

## 对拍

- **数字**：本库**不在 two-interpreter harness 里**。事实上的 oracle 是测试内硬编码常数，
  由维护者在真机跑 `nox -s optional` 时锁定；`agent/manuals/project-context.md` 记录
  「fail-closed optional CUDA base gate passes 16 TorchMetrics, MMCV/MMEngine, PEFT,
  TensorDict, and FlashAttention-adapter tests」。因此**没有**同权重-同输入的双解释器结构保证。
- **速度**：**没有** speed case，也没有 ratio。`nox -s optional` 只有单测 timeout（冷编
  会 timeout，报告把失败点归为环境吞吐限制，不是数值回归）。速度只能自行用
  `../jittor-torch-diff/SKILL.md` 的协议单独量。
- **device 规则**：Jittor 没有 per-tensor device，本机 `import jittor` 直接 `CUDA enabled`
  （实测）。`optional` session 设 `JITTOR_TEST_DEVICES=cuda`、`use_cuda=1`，跑真实 CUDA；
  要 CPU 档必须自己显式 `jt.runtime.scope(use_cuda=0)`，不能靠「不请求」。

## 断点与分流

TorchMetrics 的通用指标走 `import torch` → `jittor.compat.torch`；**只有它的私有 helper
策略需要 adapter（第三行）**：

| 断点 | 分流 |
| --- | --- |
| 指标用到的普通 torch 算子/张量方法 | `jittor.compat.torch` |
| 核心算子缺失或数值错 | jittor 核心 |
| `torchmetrics.utilities.data._bincount` / `dim_zero_cat`、`utilities.compute._safe_divide` 这些**库私有 helper** 的边界策略 | **adapter `adapters/jittor_adapters/torchmetrics.py`** |

Adapter 的契约（`adapters/README.md`）：`SUPPORTED_VERSIONS = {"1.7.4"}`，通过
`jittor.module_patches` entry point `jittor_torchmetrics = jittor_adapters.torchmetrics:register`
注册，**所在模块导入后**才应用（不改 `builtins.__import__`）；版本不匹配或缺 helper 抛
`UnsupportedAdapterVersion`；dist 名 `jittor-torch-adapters`。**没有 adapter 时通用 torch
路径仍然工作**——adapter 只补这三个私有 helper 的有界行为，不是跑 TorchMetrics 的前提。

## 坑与假绿

- **缺包整文件 skip**：测试顶部是
  `@unittest.skipIf(importlib.util.find_spec("torchmetrics") is None, ...)`。本机缺包时
  `0 passed, N skipped` 看起来和通过一样；必须配 `JITTOR_REQUIRE_OPTIONAL_DEPS=1` 才有意义。
- **adapter 标记命名**：`test_torchmetrics_compat.py` 断言模块上有
  `_jittor_fast_bincount` / `_jittor_fast_dim_zero_cat` / `_jittor_fast_safe_divide`，而当前
  `adapters/jittor_adapters/torchmetrics.py` 的 `_publish(..., marker)` 用的是
  `_jittor_orig_*` 命名。**本机没有 torchmetrics，未能执行验证**这两处是否一致；装包后
  第一次跑务必先确认 `test_torchmetrics_required_torch_ops` 真的 pass，不要假设。
- **把「装上了」当「起作用了」**：adapter 未加载时报告里是 `unavailable` 不是 applied；
  用 `jittor.compat.module_patcher.last_module_patch_report()` 看实际状态。
- **冷编 timeout ≠ 数值失败**：报告明确 fresh cache 会 timeout、热 cache 通过。
- **`_bincount` 的有界语义**：adapter 修的是 `minlength` 已知时的定长输出；
  `minlength=None` 仍走原实现，别把它当通用重写。

## 证据

- **本机已实测**：两侧均无 torchmetrics；shim venv 的 `jittor-torch-adapters` 1.3.11.0 已装
  且 `find_spec('jittor_adapters')` 为 True；oracle 断言 `oracle 2.13.0+cu129`；无
  `env-jittor.sh` 时 `import jittor` 报 `python3.12-config not found`。
- **未在本机验证**：torchmetrics 的安装、`test_torchmetrics_compat.py` 的任何 pass/fail、
  上述 `_jittor_fast_*` / `_jittor_orig_*` 命名是否真的不一致（仅读代码发现，需装包后确认）、
  `nox -s optional` 的 CUDA 结果。
- 维护者报告：`refactor-wip/results/2026-08-24-optional-compat-cuda-gate.md`
  （TorchMetrics 1.7.4；拆成四个 test timeout 后同一保留 cache `4 passed in 46.41s`；
  五模块合并 `16 passed in 259.54s`；冷 cache timeout 归为环境吞吐）；
  `agent/manuals/project-context.md`（16 TorchMetrics/MMCV/MMEngine/PEFT/TensorDict/
  FlashAttention 测试）；`adapters/README.md`。

## 实测（2026-09-19）

环境：仓库 HEAD `90fe0b9`；GPU 5；oracle 断言 `2.13.0+cu129`。包站点
`/root/jittor-lab/_state/verify-misc/site`，用 `pip install --target ... --no-deps`
锁定 `torchmetrics==1.7.4`（另补 `lightning-utilities/orjson/pyvers`，**不装 torch**）。
不锁版本会装成 `torchmetrics 1.9.0` + `torch 2.14.0`，被 adapter 的
`SUPPORTED_VERSIONS={"1.7.4"}` 拒绝——已弃用不锁版本的那次安装。

**case 清单**：`verify_repo.py --repo torchmetrics --list-only` **无任何输出**
（`_ecosystem_cases.py` 的 `CASES` 无 torchmetrics）。本库确无 ecosystem case。

**skill 里的 pytest 命令在本 venv 不可用**：
`pytest -q compat/tests/torch/test_torchmetrics_compat.py` → `4 errors in 0.58s`，全是
collection 阶段 `compat/__init__.py:3 → compat/_aliases.py:8` 的
`ImportError: attempted relative import beyond top-level package`，`collected=4 executed=0`。

**绕过 collection 的实跑**（`import jittor` 后 `runpy` 跑测试文件；包站点进
`PYTHONPATH`；`JITTOR_HOME=verify-misc/jittor-home`）：

- 第 1 次 `Ran 4 tests in 426.074s`，`FAILED (errors=3)`；加
  `DISABLE_MULTIPROCESSING=1 JT_SYNC=1` 重跑 `Ran 4 tests in 815.223s`，同为 3 error。
- `test_aggregation_metrics` **通过**（`MeanMetric`/`SumMetric` 写死期望值对上了）。
- `test_classification_metrics` 在 `torchmetrics/.../accuracy.py:79` 的 `tp.sum(...)` 报
  `RuntimeError: reduce_op.cc:291: Reduce dim out of range: requested dims [0,] for a 0-D var`；
  该行之前的断言（BinaryAccuracy/F1/AUROC、MulticlassAccuracy micro/macro、MulticlassF1Score、
  confusion matrix、`binary_accuracy`）未失败。
- `test_regression_metrics`（R2Score）与 `test_torchmetrics_required_torch_ops`
  （`torch.bincount`）都死在 JIT：nvcc 编译生成的 `setitem` 算子
  `return 512 ... overcommit issue or out of memory`，单进程编译下仍复现。

**四轴**：支持的 case——无 ecosystem case，compat 文件四组只有 aggregation 跑通；
精度——仅 aggregation 的期望值经机器验证；显存/速度——未测（无 case、无 speed case）。

**读码核对**：测试断言 `_jittor_fast_bincount/_jittor_fast_dim_zero_cat/_jittor_fast_safe_divide`，
而 `adapters/jittor_adapters/torchmetrics.py` 的 `_publish` 设的 marker 是 `_jittor_orig_*`；
skill 标为「未验证」的这处**确实不一致**（执行未走到该断言）。

**证据状态**：部分机器验证。报告/`nox -s optional` 的「4 passed」在本机**未复现**：
1 组通过、3 组 error（1 个真实 jittor reduce 错误 + 2 个 nvcc 编译失败）。未在本机跑 nox。
