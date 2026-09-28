---
name: mmcv-torch-compat
description: 在 Jittor torch shim 与独立 PyTorch 两侧跑通并对比 OpenMMLab 的 mmcv（mmcv-lite 的 mmcv.cnn 层），给出可复现命令、对拍口径与断点分流。用于验证 mmcv 在 shim 上的数值/梯度/速度，或定位 mmcv 相关断点该修 jittor 核心、jittor.compat.torch 还是 adapter 时使用。
---

# mmcv 在 torch shim 上跑 + 与原生 torch 对拍

配套仓库：[`mmengine-torch-compat`](../mmengine-torch-compat/SKILL.md)。两者是同一
OpenMMLab 栈，`mmcv_conv_module` 这个 case 同时需要 `mmcv` 与 `mmengine`，务必一起读。

## 用途

回答三件事：mmcv 怎么在 shim 上跑、怎么在原生 torch 上跑、两侧怎么对拍（数值与速度）。
维护的 mmcv 范围是 **`mmcv.cnn` 等纯 Python 模型层**。

**不覆盖**：`mmcv.ops`（NMS / RoIAlign / DeformConv / MultiScaleDeformableAttention /
CARAFE 等）——它是针对 PyTorch C++ ABI 单独编译的扩展，明确不在 Python shim 契约内。
也不覆盖 mmdetection 的完整训练、真实 OpenMMLab 数据集与 checkpoint。

## 两侧环境

| 角色 | 解释器 | 环境脚本 |
| --- | --- | --- |
| shim | `/root/jittor-lab/_state/h3/venv-jittor/bin/python` | `/root/jittor-lab/minimax-h3/env-jittor.sh` |
| 原生 torch（oracle） | `/root/jittor-lab/_state/h3/venv-oracle-cu129/bin/python` | `/root/jittor-lab/minimax-h3/env-oracle-cu129.sh` |

- shim 侧：Python 3.12.12，Jittor 1.3.11.0；`source env-jittor.sh` 后 `import jittor`
  再 `import torch`，`torch` 带 `_torch_compat_install_context` 标记（本机已实测）。
  `JITTOR_TORCH_SHIM=1`，`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，`JITTOR_HOME`
  等缓存都在 lab 的 run root 下。
- oracle 侧：Python 3.12.12，真 PyTorch 2.13.0+cu129。两侧 Python 主版本相同（都是
  3.12），所以可共享同一个下游 site。
- **本机 mmcv / mmcv-lite / mmengine 两侧都未安装**（`importlib.metadata.version` 与
  `find_spec` 均实测缺席，见「证据」）。因此**本机产不出任何 mmcv 数字**，下面命令只
  给出规范形态，必须先在环境里出现 mmcv。

任何数字之前先过 oracle 断言，`noxfile.py` 的 `ecosystem` session 起手就做：

```bash
/root/jittor-lab/_state/h3/venv-oracle-cu129/bin/python -c \
  "import torch, sys; assert not hasattr(torch, '_torch_compat_install_context'); print('oracle torch', torch.__version__)"
# 本机实测输出：oracle torch 2.13.0+cu129
```

**规范门禁怎么得到环境**：`nox -s ecosystem`（需设 `REAL_TORCH_PYTHON`）只安装
`numpy/pillow/pytest/pytest-timeout/scipy/setuptools/tqdm` 这些基础依赖，**不安装
mmcv/mmengine**；下游库由 `JITTOR_ECOSYSTEM_PACKAGE_SITE`（Jittor 侧）和
`JITTOR_ECOSYSTEM_REFERENCE_PACKAGE_SITE`（oracle 侧，ABI 不同 CPython 时必需）指定的
下游 site 提供。契约见 [`_ecosystem_harness.py`](../../../compat/tests/torch/_ecosystem_harness.py)
模块 docstring。另有一条硬件门禁 `nox -s optional`：它在预置的 CUDA python 上探测
`OPTIONAL_COMPAT_PACKAGES`（含 `mmcv`、`mmengine`）并跑
[`test_mmcv_compat.py`](../../../compat/tests/torch/test_mmcv_compat.py)。
已验证报告用的是 `mmcv-lite 2.1.0` + `mmengine 0.10.7`。

## 在 shim 上跑

runner 是唯一入口（`--help` 已实测）：

```bash
source /root/jittor-lab/minimax-h3/env-jittor.sh
cd /apdcephfs_private/qy/projects/zy/jittor
"$VENV/bin/python" compat/tests/torch/_ecosystem_runner.py \
    mmcv_conv_module /tmp/mmcv_shim.npz --runtime jittor --device cpu
```

本机实测：上面的命令在构造模型时以 `ModuleNotFoundError: No module named 'mmcv'`
退出（依赖缺席），**不是** shim 的失败。装上 `mmcv-lite`+`mmengine` 后才可能走出结果。

库自带的 import/构造契约（本机因缺库会 skip）：

```bash
"$VENV/bin/python" -m pytest compat/tests/torch/test_mmcv_compat.py -q
```

## 在原生 torch 上跑

同一个 runner，用 oracle 解释器、`--runtime torch`。harness 会清掉继承来的 Jittor
变量，oracle 侧不得看到本 checkout 与 shim：

```bash
cd /apdcephfs_private/qy/projects/zy/jittor
/root/jittor-lab/_state/h3/venv-oracle-cu129/bin/python \
    compat/tests/torch/_ecosystem_runner.py \
    mmcv_conv_module /tmp/mmcv_torch.npz --runtime torch --device cpu
```

本机同样因 `mmcv` 缺席而失败（已实测）。

## 对拍

**数值**：case 只写一次、吃 `torch` 吐 `(model, inputs)`，永远不知道自己跑在哪个
runtime 上；seed、权重传递、序列化都归 runner。torch 侧把 `named_parameters` +
`named_buffers` 存成 `<output>.weights.npz`，jittor 侧用 `--weights` 原样载入
（`--weights` 存在时 runner 会校验两边参数/缓冲一一对应），所以"同权重同输入"是结构
保证。gate 断言 forward 与**每一个参数梯度、每一个输入梯度**都一致；容差
CPU `2e-3/1e-2`、加速卡 `5e-3/2e-2`（`_ecosystem_harness.py`）。

```bash
source /root/jittor-lab/minimax-h3/env-jittor.sh
cd /apdcephfs_private/qy/projects/zy/jittor
REAL_TORCH_PYTHON=/root/jittor-lab/_state/h3/venv-oracle-cu129/bin/python \
JITTOR_ECOSYSTEM_PACKAGE_SITE=<含 mmcv-lite 2.1.0 / mmengine 0.10.7 的 site> \
JITTOR_ECOSYSTEM_REFERENCE_PACKAGE_SITE=<oracle 侧 site，可同上> \
JITTOR_TORCH_SHIM=1 "$VENV/bin/python" -m pytest \
    compat/tests/torch/test_ecosystem_parity.py -q -k 'mmcv or mmengine'
```

`JITTOR_ECOSYSTEM_PACKAGE_SITE` 必须指向真实存在的目录，否则 harness 直接抛错；两侧
下游库版本必须一致（harness 断言）。

**速度**：`_ecosystem_speed.py` 的 CASES 只有 transformers / diffusers / convnet，
**没有 mmcv 也没有 mmengine 的大尺寸 case**——即这三个仓库在规范 harness 里没有
真实尺寸速度条目。因此 mmcv 唯一能拿到的墙钟是 parity gate 的 tiny case 计时（最少
3 次取最小值），那是 Python 派发与 kernel launch 主导的，**不能当性能结论**。合理的
速度对比协议是同一 case 两侧交替多次采样、引用最小值；本仓库维护用例的 NPU 数字
（MMCV `0.927x`、MMEngine `0.796x`）来自报告的 20 次训练步取最小值协议，不是本 gate
的 3 次采样。**不要把墙钟写进 PR 门禁**：只在 nightly 由 `JITTOR_ECOSYSTEM_SPEED_RATIO`
断言（nox 默认上限 `1.07`），报告永远输出。

**device 规则**：两侧必须同 device，harness 会读回并断言 `report["device"]`。
Jittor 没有 per-tensor device，有卡机器上 CUDA 默认就是开的，所以 CPU 必须显式关
（runner 在 `--device cpu` 时进入 `jt.runtime.scope(use_cuda=0)`）；忘记关就会拿
Jittor 加速卡对 PyTorch CPU。

## 断点与分流

| 断点 | 分流 | 说明 |
| --- | --- | --- |
| `import mmcv.cnn` / `ConvModule(...)` 读 `self.conv.transposed`、`output_padding` | **jittor.compat.torch** | ACL 会把 `nn.Conv2d`/`nn.Conv` 包成不同类，installer 需给 `Conv2d`/`ConvTranspose2d` 都补上这些 torch 属性（2026-08-30 NPU 报告） |
| ACL 的 `ReLU`/`LeakyReLU` 丢掉 `inplace` 参数 | **jittor.compat.torch** | 模块与 functional 两个形态都要接受该参数 |
| mmengine 的 `InstanceData` 读 `torch.cuda.BoolTensor/LongTensor` | **jittor.compat.torch**（在 mmengine 侧触发） | shim 需发布全部十个 `torch.cuda` 类型构造器，且它们按 dtype+实际驻留做 `isinstance`，不是顶层类型的别名（见 mmengine skill） |
| `mmcv.ops` 的 NMS/RoIAlign/DeformConv 等 | **契约外 / 若必须支持才谈 adapter 或 jittor 核心** | 二进制扩展，Python shim 无法使其 ABI 兼容；`tests/models/_mmdet_ops_checks.py` 只覆盖 mmdet 用到的 torch 算子面，不覆盖这些 kernel |

## 坑与假绿

1. **缺库 = skip = 看起来通过。** `_distributions_available` 找不到 `mmcv`/`mmengine`
   时 `skipTest`。本机正是这种状态：门禁会绿，但什么都没证明。真机/nightly 用
   `JITTOR_REQUIRE_REAL_TORCH=1`（和 `JITTOR_TEST_REQUIRE_EXECUTION=1`）把这类 skip
   变成失败。
2. **有卡机器上"跑 CPU"默认不是 CPU。** 见上；只"不请求 CUDA"不等于 CPU 对比。
3. **装了完整 `mmcv` 而不是 `mmcv-lite`**：完整包会带编译 op，其 import 可能触发
   `mmcv.ops`，而该扩展不在契约内。已验证报告用的是 `mmcv-lite 2.1.0`。
4. **两侧 site 不同版本**：harness 会断言版本一致；用两份不同 wheel 对拍等于换了个参考物。
5. **tiny case 通过 ≠ 库真正用到的 API 面是真的。** 参见
   [`torch-shim-noop-audit`](../torch-shim-noop-audit/SKILL.md)。
6. **NPU 上 CUDA 与 ACL 不是一回事**：`--device npu` 在 Jittor 侧要求
   `use_cuda=1` 且 `use_acl=1`，oracle 侧要求 `torch_npu`；三者缺一就 `SystemExit`。

## 证据

- 用例与配置：[`_ecosystem_cases.py`](../../../compat/tests/torch/_ecosystem_cases.py)
  的 `_mmcv_conv_module`（`ConvModule(3,16,3,pad=1,BN,ReLU)` +
  `ConvModule(16,16,3,pad=1,GN(4),LeakyReLU)` + `nn.Conv2d(16,3,1)`，输入
  `float32 (2,3,16,16)`）。
- 分流与契约：[`downstream-library-adaptation`](../downstream-library-adaptation/SKILL.md)、
  [`jittor-torch-diff`](../jittor-torch-diff/SKILL.md)。
- 结果文档：[CI 上 CUDA typed tensor 兼容](../../../refactor-wip/results/2026-08-21-mmcv-cuda-typed-tensors.md)、
  [NPU 数值与性能](../../../refactor-wip/results/2026-08-30-mmcv-mmengine-ascend-parity.md)；
  [`project-context`](../../manuals/project-context.md) 引用 `0.927x/0.796x`。

**本机实测**（2026-09-19）：两个 venv 存在；两侧 Python 3.12.12；shim venv 的
`jittor 1.3.11.0`、`import jittor` 后 `torch` 带 shim 标记；oracle 断言通过
（`torch 2.13.0+cu129`）；`find_spec` mmcv/mmengine 两侧均为 False；
`_ecosystem_runner.py --help` 可用；runner 跑 `mmcv_conv_module --runtime jittor`
以 `No module named 'mmcv'` 结束；`test_ecosystem_speed.py` 无 mmcv 条目。

**未在本机验证**（只来自报告，换机器须重跑）：任何 mmcv/mmengine 数值或速度；
`0.927x/0.796x`、forward `3.842e-4`、最差梯度 `3.197e-4` 等均出自 Ascend 910B3 报告；
`JITTOR_ECOSYSTEM_PACKAGE_SITE` 的实际路径未在本机确认（报告记在
`$JITTOR_LAB_ROOT/_state/npu-ecosystem/20260830/`）。

## 实测（2026-09-19）

上面「本机未验证」一节的缺库状态已补齐，以下为首次实跑。GPU 4，oracle
`torch 2.13.0+cu129`、shim `jittor 1.3.11.0`，`--device cuda --repeats 5`。

**package site**：`/root/jittor-lab/_state/verify-ml/site`，`pip install --target site
peft ms-swift mmcv-lite mmengine` 装出 **mmcv-lite 2.2.0 + mmengine 0.10.7**（及 peft
0.20.0、ms-swift 4.5.3）。装的是 **mmcv-lite**，按约定避开完整 `mmcv` 的 CUDA 编译扩展
（`mmcv.ops` 本就在契约外）；pip 顺带拉入的 torch 2.14.0 / transformers 5.16.1 /
numpy 2.5.3 已从 site 删除，改用 venv 的 torch 2.13.0 / transformers 5.5.3 / numpy 2.3.5。
`mmcv` 模块由 mmcv-lite 提供，`importlib.metadata` 报告 `2.2.0`。

**命令**：

```bash
cd /apdcephfs_private/qy/projects/zy/jittor
source /root/jittor-lab/minimax-h3/env-jittor.sh && unset PYTHONPATH
export JITTOR_HOME=/root/jittor-lab/_state/verify-ml/jittor-home
export CUDA_VISIBLE_DEVICES=4
export JITTOR_ECOSYSTEM_PACKAGE_SITE=/root/jittor-lab/_state/verify-ml/site
export JITTOR_ECOSYSTEM_REFERENCE_PACKAGE_SITE=/root/jittor-lab/_state/verify-ml/site
REAL_TORCH_PYTHON=/root/jittor-lab/_state/h3/venv-oracle-cu129/bin/python \
  "$VENV/bin/python" agent/skills/torch-compat-repo-runbook/scripts/verify_repo.py \
  --repo mmcv --device cuda --repeats 5 --out /root/jittor-lab/_state/verify-ml/out/mmcv
```

**支持的 case 清单**：`--list-only` 收 1 条 `mmcv_conv_module`（requires mmcv,mmengine）；
`_ecosystem_speed.py` 无 mmcv 大尺寸 case（这里已复核）。

| case | 状态 | 精度（最差绝对 / 全场相对） | 速度 torch→jittor | 峰值显存 torch / jittor | fallback | device |
| --- | --- | --- | --- | --- | --- | --- |
| `mmcv_conv_module` | ran（10 张量，无缺失） | `1.812e-5` / `2.514e-7`（均 `grad::head.weight`，梯度容差 2e-2 内） | `2.500 → 2.495 ms`，`0.998x` | `320 KiB / 1 MiB` | 0 | 两侧 cuda |

**显存口径警告**：torch 报自身分配器真实峰值，jittor 是 `profile_memory_enable` 下 sync
后采样的 allocator 用量，不可直接比较，只并列原始值。tiny case 的 `0.998x` 由 Python
派发主导，不是 kernel 性能结论；大尺寸速度仍需按第「对拍」节另立 case。

**没跑到的**：`mmcv.ops`（NMS/RoIAlign/DeformConv 等）仍属编译扩展契约外，未测；完整
`mmcv` 未安装（改用 mmcv-lite 2.2.0）；无 mmcv/mmengine 真实尺寸速度 case。原始报告见
`/root/jittor-lab/_state/verify-ml/{mmcv.log,out/mmcv/verify-report.json}`。
