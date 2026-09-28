---
name: mmengine-torch-compat
description: 在 Jittor torch shim 与独立 PyTorch 两侧跑通并对比 MMEngine（mmengine.model 的 BaseModule/BaseModel 纯 Python 层），给出可复现命令、对拍口径与断点分流。用于验证 mmengine 在 shim 上的数值/梯度/速度，或定位 mmengine 相关断点该修 jittor 核心、jittor.compat.torch 还是 adapter 时使用。
---

# mmengine 在 torch shim 上跑 + 与原生 torch 对拍

配套仓库：[`mmcv-torch-compat`](../mmcv-torch-compat/SKILL.md)。两者同一 OpenMMLab 栈，
`mmcv_conv_module` 这个 case 需要 `mmcv` + `mmengine` 两者，务必一起读。

## 用途

回答三件事：mmengine 怎么在 shim 上跑、怎么在原生 torch 上跑、两侧怎么对拍（数值与速度）。
维护的 mmengine 范围是 **`mmengine.model`（`BaseModule` / `BaseModel`）这条纯 Python
模型层 import 与构造路径**。

**不覆盖**：MMEngine 的 runner / hook / registry 全链路训练、真实的 mmdet/mmseg 模型，
以及 `mmcv.ops`（同 mmcv skill，属编译扩展，不在 Python shim 契约内）。

## 两侧环境

| 角色 | 解释器 | 环境脚本 |
| --- | --- | --- |
| shim | `/root/jittor-lab/_state/h3/venv-jittor/bin/python` | `/root/jittor-lab/minimax-h3/env-jittor.sh` |
| 原生 torch（oracle） | `/root/jittor-lab/_state/h3/venv-oracle-cu129/bin/python` | `/root/jittor-lab/minimax-h3/env-oracle-cu129.sh` |

- shim 侧：Python 3.12.12，Jittor 1.3.11.0；`source env-jittor.sh` 后 `import jittor`
  再 `import torch`，`torch` 带 `_torch_compat_install_context` 标记（本机已实测）。
  `JITTOR_TORCH_SHIM=1`，`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，`JITTOR_HOME`
  等缓存都在 lab 的 run root 下。
- oracle 侧：Python 3.12.12，真 PyTorch 2.13.0+cu129。两侧 Python 主版本同为 3.12，
  可共享同一个下游 site；CPython 主版本不同时才需要
  `JITTOR_ECOSYSTEM_REFERENCE_PACKAGE_SITE` 或显式 `JITTOR_ECOSYSTEM_PACKAGE_SITE_CROSS_ABI=1`。
- **本机 mmengine 两侧都未安装**（`importlib.metadata.version` 与 `find_spec` 均实测
  缺席，见「证据」）。因此**本机产不出任何 mmengine 数字**。

任何数字之前先过 oracle 断言（`nox -s ecosystem` 已内置）：

```bash
/root/jittor-lab/_state/h3/venv-oracle-cu129/bin/python -c \
  "import torch, sys; assert not hasattr(torch, '_torch_compat_install_context'); print('oracle torch', torch.__version__)"
# 本机实测输出：oracle torch 2.13.0+cu129
```

**规范门禁怎么得到环境**：`nox -s ecosystem`（需设 `REAL_TORCH_PYTHON`）只安装基础
依赖，**不安装 mmengine**；下游库通过 `JITTOR_ECOSYSTEM_PACKAGE_SITE`（及 oracle 侧
的 `JITTOR_ECOSYSTEM_REFERENCE_PACKAGE_SITE`）指定的下游 site 提供，契约见
[`_ecosystem_harness.py`](../../../compat/tests/torch/_ecosystem_harness.py) 模块
docstring。硬件门禁 `nox -s optional` 在预置 CUDA python 上探测含 `mmengine` 的
`OPTIONAL_COMPAT_PACKAGES`，并跑
[`test_mmcv_compat.py`](../../../compat/tests/torch/test_mmcv_compat.py)。已验证报告
用的是 `mmengine 0.10.7`。

## 在 shim 上跑

runner 是唯一入口：

```bash
source /root/jittor-lab/minimax-h3/env-jittor.sh
cd /apdcephfs_private/qy/projects/zy/jittor
"$VENV/bin/python" compat/tests/torch/_ecosystem_runner.py \
    mmengine_base_module /tmp/mmengine_shim.npz --runtime jittor --device cpu
```

本机实测：命令在构造模型时以 `ModuleNotFoundError: No module named 'mmengine'`
退出，**不是** shim 的失败。

库自带的 import 契约（本机因缺库会 skip）：

```bash
"$VENV/bin/python" -m pytest compat/tests/torch/test_mmcv_compat.py -q
```

注意该文件里 `TestMmcvCompat` 同时 import `mmcv.cnn` 与 `mmengine.model`，缺任一都会
整体 skip；`TestCudaTypedTensorCompat` 看似通用，实际就是为 mmengine 的
`torch.cuda.*Tensor` 注解路径立的回归。

## 在原生 torch 上跑

```bash
cd /apdcephfs_private/qy/projects/zy/jittor
/root/jittor-lab/_state/h3/venv-oracle-cu129/bin/python \
    compat/tests/torch/_ecosystem_runner.py \
    mmengine_base_module /tmp/mmengine_torch.npz --runtime torch --device cpu
```

本机同样因 `mmengine` 缺席而失败（已实测）。harness 会清掉 oracle 继承到的 Jittor
变量，oracle 不得看到本 checkout 与 shim。

## 对拍

**数值**：case 只写一次、吃 `torch` 吐 `(model, inputs)`，不感知 runtime；seed、权重
传递、序列化归 runner。torch 侧存 `<output>.weights.npz`（`named_parameters` +
`named_buffers`），jittor 侧用 `--weights` 原样载入并校验一一对应，所以"同权重同输入"
是结构保证。gate 断言 forward 与每一个参数梯度、每一个输入梯度；容差 CPU
`2e-3/1e-2`、加速卡 `5e-3/2e-2`（`_ecosystem_harness.py`）。

```bash
source /root/jittor-lab/minimax-h3/env-jittor.sh
cd /apdcephfs_private/qy/projects/zy/jittor
REAL_TORCH_PYTHON=/root/jittor-lab/_state/h3/venv-oracle-cu129/bin/python \
JITTOR_ECOSYSTEM_PACKAGE_SITE=<含 mmengine 0.10.7 的 site> \
JITTOR_ECOSYSTEM_REFERENCE_PACKAGE_SITE=<oracle 侧 site，可同上> \
JITTOR_TORCH_SHIM=1 "$VENV/bin/python" -m pytest \
    compat/tests/torch/test_ecosystem_parity.py -q -k 'mmengine or mmcv'
```

`JITTOR_ECOSYSTEM_PACKAGE_SITE` 必须指向真实存在的目录，否则 harness 直接抛错；两侧
下游库版本必须一致（harness 断言）。

**速度**：`_ecosystem_speed.py` 的 CASES 只有 transformers / diffusers / convnet，
**没有 mmengine 也没有 mmcv 的大尺寸 case**——规范 harness 里没有真实尺寸速度条目。
mmengine 唯一能拿到的墙钟是 parity gate 的 tiny case 计时（最少 3 次取最小值），由
Python 派发主导，**不能当性能结论**。合理协议是同一 case 两侧交替多次采样、引用最小值；
本仓库维护用例的 NPU 数字（MMEngine `0.796x`、MMCV `0.927x`）来自报告的 20 次训练步
取最小值协议。**不要把墙钟写进 PR 门禁**：只在 nightly 由
`JITTOR_ECOSYSTEM_SPEED_RATIO` 断言（nox 默认上限 `1.07`），报告永远输出。

**device 规则**：两侧同 device，harness 读回并断言 `report["device"]`。Jittor 没有
per-tensor device，有卡机器上 CUDA 默认就是开的，所以 CPU 必须显式关（runner 在
`--device cpu` 时进入 `jt.runtime.scope(use_cuda=0)`）；忘记关就是拿加速卡对 CPU。
`--device npu` 在 Jittor 侧要求 `use_cuda=1` 且 `use_acl=1`，oracle 侧要求
`torch_npu`。

## 断点与分流

| 断点 | 分流 | 说明 |
| --- | --- | --- |
| `import mmengine.model` / `InstanceData` 读 `torch.cuda.BoolTensor`、`torch.cuda.LongTensor` | **jittor.compat.torch** | MMEngine 0.10.7 用它们构造类型注解；shim 的 CUDA installer 需发布全部十个 `torch.cuda` 类型构造器，并按 dtype+实际驻留做 `isinstance`（不是顶层类型的别名），否则 host bool/int64 会被误判成 CUDA 张量（2026-08-21 报告） |
| `ConvModule` / ACL 卷积类属性（`transposed`、`output_padding`）、ReLU/LeakyReLU 的 `inplace` | **jittor.compat.torch**（在 mmcv 侧触发） | 详见 [`mmcv-torch-compat`](../mmcv-torch-compat/SKILL.md) |
| `mmengine.model.BaseModule` / `BaseModel` 本身 | **无需分流，纯 Python** | 模型定义不依赖编译扩展；只要 torch API 面齐就能构造 |
| `mmcv.ops` 编译 kernel | **契约外** | 二进制扩展，Python shim 不承诺；需要时才谈 adapter 或 jittor 核心 |

## 坑与假绿

1. **缺库 = skip = 看起来通过。** `_distributions_available` 找不到 `mmengine` 时
   `skipTest`。本机正是这种状态：门禁会绿，但什么都没证明。真机/nightly 用
   `JITTOR_REQUIRE_REAL_TORCH=1`（和 `JITTOR_TEST_REQUIRE_EXECUTION=1`）把这类 skip
   变成失败。
2. **有卡机器上"跑 CPU"默认不是 CPU。** 只"不请求 CUDA"不等于 CPU 对比，必须显式关。
3. **`test_mmcv_compat.py` 会因缺 mmcv 而整体 skip**，即使 mmengine 已装。
4. **两侧 site 不同版本**：harness 断言版本一致；两份不同 wheel 对拍等于换参考物。
5. **tiny case 通过 ≠ API 面是真的。** `InstanceData` 注解路径能 import，不代表
   registry/hook/runner 全链路能用；见
   [`torch-shim-noop-audit`](../torch-shim-noop-audit/SKILL.md)。
6. **NPU 上 CUDA 与 ACL 不是一回事**：三个状态位缺一即 `SystemExit`。

## 证据

- 用例与配置：[`_ecosystem_cases.py`](../../../compat/tests/torch/_ecosystem_cases.py)
  的 `_mmengine_base_model`：`Head(BaseModule)`，`LayerNorm(32)` + `Linear(32,64)` +
  `Linear(64,32)`，forward = `fc2(relu(fc1(norm(x)))) + x`，输入 `float32 (2,12,32)`。
- 分流与契约：[`downstream-library-adaptation`](../downstream-library-adaptation/SKILL.md)、
  [`jittor-torch-diff`](../jittor-torch-diff/SKILL.md)。
- 结果文档：[CUDA typed tensor 兼容](../../../refactor-wip/results/2026-08-21-mmcv-cuda-typed-tensors.md)、
  [NPU 数值与性能](../../../refactor-wip/results/2026-08-30-mmcv-mmengine-ascend-parity.md)；
  [`project-context`](../../manuals/project-context.md) 引用 `0.927x/0.796x`。

**本机实测**（2026-09-19）：两个 venv 存在；两侧 Python 3.12.12；shim venv 的
`jittor 1.3.11.0`、`import jittor` 后 `torch` 带 shim 标记；oracle 断言通过
（`torch 2.13.0+cu129`）；`find_spec` mmengine/mmcv 两侧均为 False；
`_ecosystem_runner.py --help` 可用；`test_ecosystem_speed.py` 无 mmengine 条目。

**未在本机验证**（只来自报告，换机器须重跑）：任何 mmengine/mmcv 数值或速度；
`0.796x`、forward `6.525e-8`、最差梯度 `7.411e-8` 等出自 Ascend 910B3 报告；
`JITTOR_ECOSYSTEM_PACKAGE_SITE` 的实际路径未在本机确认（报告记在
`$JITTOR_LAB_ROOT/_state/npu-ecosystem/20260830/`）。

## 实测（2026-09-19）

上面「本机未验证」一节的缺库状态已补齐，以下为首次实跑。GPU 4，oracle
`torch 2.13.0+cu129`、shim `jittor 1.3.11.0`，`--device cuda --repeats 5`。

**package site**：`/root/jittor-lab/_state/verify-ml/site`，`pip install --target site
peft ms-swift mmcv-lite mmengine` 装出 **mmengine 0.10.7**（及 mmcv-lite 2.2.0、peft
0.20.0、ms-swift 4.5.3）；pip 拉入的 torch 2.14.0 / transformers 5.16.1 / numpy 2.5.3
已从 site 删除，改用 venv 的 torch 2.13.0 / transformers 5.5.3 / numpy 2.3.5。

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
  --repo mmengine --device cuda --repeats 5 --out /root/jittor-lab/_state/verify-ml/out/mmengine
```

**支持的 case 清单**：`--list-only` 收 1 条 `mmengine_base_module`（requires mmengine）；
`_ecosystem_speed.py` 无 mmengine 大尺寸 case（这里已复核）。

| case | 状态 | 精度（最差绝对 / 全场相对） | 速度 torch→jittor | 峰值显存 torch / jittor | fallback | device |
| --- | --- | --- | --- | --- | --- | --- |
| `mmengine_base_module` | ran（8 张量，无缺失） | `3.844e-6` / `3.983e-7`（均 `grad::fc1.weight`，梯度容差 2e-2 内） | `2.421 → 2.382 ms`，`0.984x` | `64.1 MiB / 1 MiB` | 0 | 两侧 cuda |

**显存口径警告**：torch 报自身分配器真实峰值（含 CUDA context），jittor 是
`profile_memory_enable` 下 sync 后采样的 allocator 用量，两者不可直接比较，只并列原始
值；`64.1 MiB / 1 MiB` 不代表 jittor 更省。tiny case 的 `0.984x` 由 Python 派发主导，
不是 kernel 性能结论。

**没跑到的**：mmengine runner/hook/registry 全链路训练与真实 mmdet/mmseg 模型未测；
`mmcv.ops` 编译扩展属契约外；无真实尺寸速度 case。`test_mmcv_compat.py` 需 mmcv 与
mmengine 同时在位，本次只经 `verify_repo.py` 跑了 ecosystem case。原始报告见
`/root/jittor-lab/_state/verify-ml/{mmengine.log,out/mmengine/verify-report.json}`。
