---
name: tensordict-torch-compat
description: 在 Jittor 的 torch shim 上运行 TensorDict 并与 numpy/torch 期望值对照，用于验证 CUDA 上的构造、更新、index 与 lazy stack，并说明这套 index 兼容补丁在 jittor.compat.torch 中的位置；不覆盖 torchrl、分布式/多 device 和 two-interpreter 速度。
---

# TensorDict × Jittor torch shim

## 用途

回答「TensorDict 架在 `import torch` → Jittor 上能否跑、CUDA 上 `__getitem__`/lazy stack
对不对」。仓库覆盖文件：`../../../compat/tests/torch/test_tensordict_compat.py`；相关实现：
`../../../compat/torch/installers/autograd.py`（`_install_tensordict_compat`）；报告：
`refactor-wip/results/2026-08-24-optional-compat-cuda-gate.md`。

**不覆盖**：torchrl / RL 训练循环、分布式与多 device TensorDict、`TensorDictModule` 的
计算图语义，以及 two-interpreter 速度对拍。

**注意**：TensorDict **没有进 ecosystem 的 two-interpreter harness**
（`../../../compat/tests/torch/_ecosystem_cases.py` 的 `CASES` 里没有 tensordict）。
数值对照是单解释器测试里的 numpy 期望数组，device 由 `nox -s optional` 的真实 CUDA 会话提供。

## 两侧环境

本机（`/root/jittor-lab`）已实测：

| 侧 | 解释器 | 关键事实 |
| --- | --- | --- |
| shim | `/root/jittor-lab/_state/h3/venv-jittor/bin/python` | py3.12.12；jittor 1.3.11.0；`import torch` → shim；`jittor-torch-adapters` 1.3.11.0 已装；transformers 5.5.3 |
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

**本机 tensordict 两侧都未安装**（`PackageNotFoundError`，`find_spec('tensordict')`
为 False）。补上之前本 skill 的任何 tensordict 数字都是未验证。

## 在 shim 上跑

单测**硬性要求 CUDA**（`skipUnless(check_accelerator('cuda'))`），不依赖
`REAL_TORCH_PYTHON`：

```bash
set -a; source /root/jittor-lab/minimax-h3/env-jittor.sh >/dev/null 2>&1; set +a
unset PYTHONPATH
cd /apdcephfs_private/qy/projects/zy/jittor
"$VENV/bin/python" -m pytest -q compat/tests/torch/test_tensordict_compat.py
```

覆盖三组：`test_cpu_conversion_uses_device_objects`（`torch._C._nn._parse_to("cpu")`
返回 `torch.device`，`TensorDict.cpu()`）、`test_cuda_construct_update_and_lazy_stack`、
`test_jittor_tensor_indices_patch_real_tensordict`（整数/布尔/lazy 索引，断言
`TensorDictBase._jittor_index_compat` 为真）。

**环境怎么来**：`nox -s optional` 不装包，要求预置 `OPTIONAL_COMPAT_PACKAGES`
（`noxfile.py:234`），缺一个就 fail-closed。补齐办法（**本机未执行**）：装
**tensordict==0.10.0**（报告里的版本）到 shim 解释器；并设
`JITTOR_REQUIRE_OPTIONAL_DEPS=1` 让缺包变成失败而非 skip。

```bash
# 维护者真机 CUDA 档（需 nvcc + 预置包），本机未验证
python -m nox -s optional -- compat/tests/torch/test_tensordict_compat.py
```

## 在原生 torch 上跑

测试按 shim 语义写、且断言 `TensorDictBase._jittor_index_compat`，**不能**拿到 oracle
解释器跑。原生侧要做对照，只能在 oracle 解释器里手工构造同样的 `TensorDict` 并与
`test_tensordict_compat.py` 里写死的 numpy 期望数组比较：

```python
import torch
from tensordict import TensorDict
expected = torch.tensor([[1.,2.],[3.,4.],[5.,6.]])
d = TensorDict({"value": expected.cuda()}, batch_size=[3])
d["double"] = d["value"] * 2
assert torch.equal(d["value"].cpu(), expected)
```

## 对拍

- **数字**：本库**不在 two-interpreter harness 里**。事实上的 oracle 是测试内的 numpy
  期望数组（`np.testing.assert_array_equal`）与维护者在真机 `nox -s optional` 时跑出的
  结果；`agent/manuals/project-context.md` 记录 optional base gate 含 TensorDict。
  因此**没有**同权重-同输入的双解释器结构保证。
- **速度**：**没有** speed case，也没有 ratio。速度只能自行按
  `../jittor-torch-diff/SKILL.md` 的协议单独量。
- **device 规则**：Jittor 没有 per-tensor device，本机 `import jittor` 直接 `CUDA enabled`
  （实测）。该测试自己用 `jt.flag_scope(use_cuda=1)` 包住 CUDA 段；若机器没有 CUDA，整个
  文件会 skip。要 CPU 档必须显式关 `use_cuda`，不能靠「不请求」。

## 断点与分流

TensorDict 的 index 兼容补丁**住在 `jittor.compat.torch`（第二行），不是 adapter**：

| 断点 | 位置 / 分流 |
| --- | --- |
| `TensorDictBase.__getitem__`/`__getitems__`、`LazyStackedTensorDict.__getitem__` 对 jittor Var 索引的语义 | `../../../compat/torch/installers/autograd.py:_install_tensordict_compat`，安装时置 `TensorDictBase._jittor_index_compat = True`（幂等：已置位直接返回） |
| `torch._C._nn._parse_to` 返回 `torch.device` | `jittor.compat.torch` 的 `torch._C` 面 |
| 基础算子/device 语义错 | jittor 核心 |

`adapters/` 下**没有** tensordict adapter。若该补丁出现「私有属性/可变全局状态」的写法，
按 `../downstream-library-adaptation/SKILL.md` 第 0 节可重定位判据重新分流，不要直接新建 adapter。

## 坑与假绿

- **两重 skip 掩盖一切**：文件同时 `skipUnless(_HAS_TENSORDICT)` 与
  `skipUnless(check_accelerator('cuda'))`。本机缺 tensordict，`0 passed, N skipped`
  看起来和通过一样。必须配 `JITTOR_REQUIRE_OPTIONAL_DEPS=1`，并在 CUDA 机器上跑，才算验证。
- **patch 幂等标记不是 patch 生效的证据**：`_jittor_index_compat` 只是「已安装过」的守卫。
  真正的行为断言是测试里的 `selected`/`masked`/`lazy_selected` 数组比较。
- **`dim_zero_cat`/`_bincount` 属于 TorchMetrics adapter**，与本文件的 index 补丁无关，
  别混在同一个「断点」里。
- **在 CPU 上「通过」**：该测试 CUDA 段在 `flag_scope(use_cuda=1)` 内，CPU 上要么 skip，
  要么拿到未声明的 fallback；不要拿 CPU 结果声称 CUDA 支持。

## 证据

- **本机已实测**：两侧均无 tensordict；shim venv 的 `jittor-torch-adapters` 1.3.11.0 已装；
  oracle 断言 `oracle 2.13.0+cu129`；无 `env-jittor.sh` 时 `import jittor` 报
  `python3.12-config not found`；读代码确认 `_jittor_index_compat` 由
  `compat/torch/installers/autograd.py` 安装、`adapters/` 下无 tensordict adapter。
- **未在本机验证**：tensordict 的安装、`test_tensordict_compat.py` 的任何 pass/fail、
  `nox -s optional` 的 CUDA 结果。
- 维护者报告：`refactor-wip/results/2026-08-24-optional-compat-cuda-gate.md`
  （TensorDict 0.10.0；「TensorDict 与 FlashAttention 新增真实行为模块：
  `5 passed in 11.31s`，覆盖 CUDA 构造/更新/index/lazy stack」）；
  `agent/manuals/project-context.md`。

## 实测（2026-09-19）

环境：仓库 HEAD `90fe0b9`；GPU 5；oracle 断言 `2.13.0+cu129`。包站点
`/root/jittor-lab/_state/verify-misc/site` 用 `pip install --target ... --no-deps`
锁定 `tensordict==0.10.0`（另补 `pyvers/orjson`，**不装 torch**）。

**case 清单**：`verify_repo.py --repo tensordict --list-only` **无任何输出**
（`CASES` 无 tensordict）。本库确无 ecosystem case。

**skill 里的 pytest 命令在本 venv 不可用**：
`pytest -q compat/tests/torch/test_tensordict_compat.py` 与 torchmetrics 同因——collection
阶段 `compat/__init__.py:3 → compat/_aliases.py:8` 报
`ImportError: attempted relative import beyond top-level package`，0 executed。

**绕过 collection 的实跑**（`import jittor` 后 `runpy`；`PYTHONPATH` 含包站点与
`tests/`，为的是 `_helpers`）：**在 `import tensordict` 就失败**，测试一条都没执行：

```
File ".../site/tensordict/nn/distributions/truncated_normal.py", line 33
    "a": constraints.real,
AttributeError: module 'torch.distributions.constraints' has no attribute 'real'
```

根因是本机 shim 的 `torch.distributions.constraints` 是空模块：
`import jittor; import torch; import torch.distributions.constraints as c` 后
`hasattr(c,'real') == False`，`dir(c)` 连 `positive` 等一个约束都没有。tensordict 0.10.0 的
`__init__` 会导入 `tensordict.nn`，因此整个包都起不来。

**四轴**：支持的 case——无 ecosystem case，且 compat 文件因 import 失败 0 执行；
精度/显存/速度——均未测。

**证据状态**：仅安装可验证（0.10.0 已就位），**测试未能运行**。报告里的
「`5 passed in 11.31s`」在本机**未复现**；本机的阻断点在
`torch.distributions.constraints` 为空，与本 skill 正文列出的断点无关。
