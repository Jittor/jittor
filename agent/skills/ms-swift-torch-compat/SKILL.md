---
name: ms-swift-torch-compat
description: 在 Jittor 的 torch shim 与原生 PyTorch/torch_npu 两侧运行并对比 ms-swift 自带 LoRA tuner，用于复现 ms_swift_lora_llama 的数值/梯度对拍与真实设备性能结论；不覆盖 PEFT 的 LoRA、全量微调、优化器与分布式训练。
---

# ms-swift × Jittor torch shim

## 用途

回答「`ms-swift` 的 `Swift`/`swift.tuners.LoRAConfig` 架在 `import torch` → Jittor 上能否跑、
和原生 torch/torch_npu 是否同数、加速卡上快不快」。上游用例：
`../../../compat/tests/torch/_ecosystem_cases.py:_ms_swift_lora_llama`（case 名
`ms_swift_lora_llama`，依赖 `transformers` + `peft` + `swift`）。

**不覆盖**：PEFT 的 LoRA（见 `../peft-torch-compat/SKILL.md`）、真实 checkpoint 全量微调、
优化器更新、分布式 ms-swift 训练。分流/定级/device 阶梯走
`../downstream-library-adaptation/SKILL.md`。

## 两侧环境

本机（`/root/jittor-lab`）已实测：

| 侧 | 解释器 | 关键事实 |
| --- | --- | --- |
| shim | `/root/jittor-lab/_state/h3/venv-jittor/bin/python` | py3.12.12；jittor 1.3.11.0；`import torch` → shim（`_torch_compat_install_context` 存在）；transformers 5.5.3、safetensors 0.8.0、numpy 2.3.5 |
| 原生 torch | `/root/jittor-lab/_state/h3/venv-oracle-cu129/bin/python` | py3.12.12；torch 2.13.0+cu129；transformers 5.5.3 |

先 `source /root/jittor-lab/minimax-h3/env-jittor.sh`（原生侧
`env-oracle-cu129.sh`）：导出 `JITTOR_TORCH_SHIM=1`、缓存收进
`$JITTOR_LAB_ROOT/_state/h3/run`、`HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1`，并设
`JT_BUILD_PYTHON_CONFIG_PATH=/opt/python3.12/bin/python3.12-config`——**不设它
`import jittor` 直接 `RuntimeError: python3.12-config not found`**（实测）。

两侧同 py3.12、ABI 相同，共享一个 package site 即可。**注意**：NPU 验收报告用的是
py3.9(jittor) 对 py3.10(torch_npu)，ABI 不同，必须分别给
`JITTOR_ECOSYSTEM_PACKAGE_SITE` 和 `JITTOR_ECOSYSTEM_REFERENCE_PACKAGE_SITE`，harness 会
只断言依赖**版本**相同、允许 origin 不同。

**看任何数字前先过 oracle 断言**：

```bash
/root/jittor-lab/_state/h3/venv-oracle-cu129/bin/python -c \
  "import torch; assert not hasattr(torch,'_torch_compat_install_context'), 'oracle 是 shim'; print('oracle', torch.__version__)"
# 本机实测：oracle 2.13.0+cu129
```

**本机 `swift` / `ms-swift` / `peft` 两侧都未安装**（`PackageNotFoundError`，
`find_spec('swift')` 为 False）。补上之前本 skill 的 ms-swift 数字都是未验证。

## 在 shim 上跑

ms-swift 的 case 依赖 `swift` 与 `peft` 两个包，都放进
`JITTOR_ECOSYSTEM_PACKAGE_SITE` 指向的 site（契约见
`../../../compat/tests/torch/_ecosystem_harness.py` 模块 docstring）。本机两侧同 py3.12，
一个共享 site 即可。构建办法（**本机未执行**）：建空目录 `<site>`，`pip install --target <site>
ms-swift==4.5.2 peft==0.17.1`（版本取自 NPU 报告），并保证两侧 transformers 版本一致。
**不要**把 `ms-swift`/`peft` 装进 oracle venv 去覆盖 torch。

跑单条 case（pytest 起在 shim 解释器）：

```bash
set -a; source /root/jittor-lab/minimax-h3/env-jittor.sh >/dev/null 2>&1; set +a
unset PYTHONPATH
cd /apdcephfs_private/qy/projects/zy/jittor
export REAL_TORCH_PYTHON=/root/jittor-lab/_state/h3/venv-oracle-cu129/bin/python
export JITTOR_ECOSYSTEM_PACKAGE_SITE=<含 swift/peft 的 site>
"$VENV/bin/python" -m pytest -q \
  "compat/tests/torch/test_ecosystem_parity.py::EcosystemParity::test_ms_swift_lora_llama"
```

`ms_swift_lora_llama` 在 CPU / CUDA / NPU 三档都有：`EcosystemParity`（CPU）、
`EcosystemParityCUDA`（继承）与 `EcosystemParityNPU` 都收了它（NPU 档另有 diffusers、
mmcv、mmengine 三个用例）。NPU 档跑法（需先 source CANN，shim 侧 python 与 oracle
torch_npu 解释器按各自主机环境给出；下面是报告里的维护命令，本机不是 Ascend 环境）：

```bash
REAL_TORCH_PYTHON=/path/to/torch-npu/python JITTOR_TORCH_SHIM=1 \
JITTOR_ECOSYSTEM_PACKAGE_SITE=/path/to/python39/site-packages \
JITTOR_ECOSYSTEM_REFERENCE_PACKAGE_SITE=/path/to/python310/site-packages \
python -m pytest -q \
  "compat/tests/torch/test_ecosystem_parity.py::EcosystemParityNPU::test_ms_swift_lora_llama"
```

（该维护命令来自 `refactor-wip/results/2026-08-31-ms-swift-ascend-parity-performance.md`，
报告里写的是旧路径 `tests/compat/...`，本仓当前路径是 `compat/tests/...`。）

**或手动两步**（`../../../compat/tests/torch/_ecosystem_runner.py`）：

```bash
cd compat/tests/torch
env -u PYTHONPATH -u JITTOR_TORCH_SHIM \
  /root/jittor-lab/_state/h3/venv-oracle-cu129/bin/python \
  _ecosystem_runner.py ms_swift_lora_llama /tmp/torch.npz --runtime torch --device cpu
source /root/jittor-lab/minimax-h3/env-jittor.sh
JITTOR_ECOSYSTEM_PACKAGE_SITE=<site> "$VENV/bin/python" \
  _ecosystem_runner.py ms_swift_lora_llama /tmp/jittor.npz --runtime jittor --device cpu \
  --weights /tmp/torch.weights.npz
```

runner CLI（`--help` 实测）：`python _ecosystem_runner.py <case> <output.npz> [--weights W]
[--repeats N] [--seed N] [--runtime {torch,jittor}] [--device {cpu,cuda,npu}]`。
`--runtime` 默认 `torch`，shim 解释器里**必须显式 `--runtime jittor`**。

## 在原生 torch 上跑

原生侧入口是 harness 的 oracle 半边（上面的 `--runtime torch`，NPU 档是 `torch_npu` 解释器）。
仓库里没有独立的 ms-swift 单测文件；契约就是 `_ms_swift_lora_llama` 这个 case 本身。

## 对拍

- **数字**：同权重同输入由 harness 结构保证；比较前向、每个参数梯度、每个输入梯度。
  容差 CPU `2e-3 / 1e-2`，CUDA/NPU `5e-3 / 2e-2`。
  **关键细节**：ms-swift 的 tuner 只把 adapter 放进 `state_dict`，若按 `state_dict` 传权重，
  两侧 backbone 会各自随机初始化、对拍无意义；runner 因此枚举
  `named_parameters()+named_buffers()`（`_ecosystem_runner.py` 的 `transferable()`），
  任何一侧不匹配即 `SystemExit`。这是 ms-swift 与 peft 用例的实质差别。
- **速度**：harness 取 `min(durations)`、`JITTOR_ECOSYSTEM_REPEATS>=10`（speed 模块），
  **只在设 `JITTOR_ECOSYSTEM_SPEED_RATIO` 时才断言墙钟**。`_ecosystem_speed.CASES` 里
  **没有 ms-swift 的 large case**，报告里的 50 步取最小值是独立于 harness 的协议。
- **device 规则**：两侧必须同 device。Jittor 没有 per-tensor device，本机 `import jittor`
  直接 `CUDA enabled`（实测）；CPU 档必须显式 `jt.runtime.scope(use_cuda=0)`（runner 的
  `--device cpu` 已处理），否则就是 Jittor 在加速卡上对原生 CPU。NPU 档还额外断言
  `has_acl/use_acl/use_cuda` 与 `fallback_count == 0`、`fallback_policy == "error"`。

## 断点与分流

ms-swift 是纯 `import torch` 消费者，**adapter（第三行）为空**。已记录的断点：

| 现象 | 分流 |
| --- | --- |
| LoRA 训练步比 torch_npu 慢 ~18% | 不在 tuner；是 Transformers Llama 选了 SDPA，而 ACL adapter 拒绝 causal/masked training，退回 matmul/softmax，而 torch_npu 用 CANN fused attention → **jittor 核心**（ACL FlashAttentionScoreV2 接入） |
| `TypeError` 且真 PyTorch 下同样报 | 下游版本组合问题，不是 jittor 缺陷 |
| `state_dict` 只有 adapter | runner 侧已按 `named_parameters` 枚举，不是断点 |

文档明确：`ms-swift 4.5.2` 需 `peft < 0.20`；`ms-swift 4.5.2 + peft 0.19` 在真 PyTorch 下
也报同样的 `TypeError`（`docs/compatibility/torch.md`）。**遇到该错先怀疑版本，不要当 jittor bug。**

## 坑与假绿

- **名字相同不是同一个 tuner**：`swift.tuners.LoRAConfig` 与 PEFT 的 `LoraConfig` 行为不同，
  不要把 peft 的结论搬过来。case 注释原文即「ms-swift's own LoRA tuner, which is not PEFT's」。
- **权重只传 adapter 的假绿**：用 `model.state_dict()` 传权重会让两侧 backbone 独立初始化，
  数值仍然「一致」可能只是都碰巧接近随机初值；runner 的 `transferable()` 是绕开它的原因。
- **`peft` 版本上限**：不满足 `peft < 0.20` 时真 PyTorch 也会失败，别误判。
- **在 tiny case 上读速度**：hidden 64 / seq 8 的 ratio 说明不了 kernel；NPU 性能结论要用
  报告的 50 步取最小协议。
- **skip 看起来像通过**：本机 `swift` 缺失时整类 `skipTest("missing ...")`。验收用
  `JITTOR_REQUIRE_REAL_TORCH=1`（`nox -s ecosystem` 会设）把「没有 oracle」变成失败。

## 证据

- **本机已实测**：两个 venv 中 `swift`/`ms-swift`/`peft` 缺失；oracle 断言
  `oracle 2.13.0+cu129`；无 `env-jittor.sh` 时 `import jittor` 报 `python3.12-config not found`；
  `test_ecosystem_parity.py` 收集 28 项，`ms_swift_lora_llama` 在 CPU/CUDA/NPU 三档各有一项。
- **未在本机验证**：ms-swift 的安装、`ms_swift_lora_llama` 的任何数值/速度、上述 NPU 命令
  （本机不是 Ascend 环境）。命令按 harness/runner 与报告的真实接口给出，但**没有跑过**。
- 维护者报告：`refactor-wip/results/2026-08-31-ms-swift-ascend-parity-performance.md`
  （Ascend 910B3，ms-swift 4.5.2 / PEFT 0.17.1 / Transformers 4.57.6；前向归一化误差
  `1.980e-7`、最差梯度归一化误差 `4.268e-7`；50 步取最小 `torch_npu 14.089 ms` 对
  Jittor ACL compressed causal mask `13.654 ms`，`0.969x`；优化前 `1.182x`）；
  `docs/compatibility/torch.md`；`agent/manuals/project-context.md`。

## 实测（2026-09-19）

GPU 4，oracle `torch 2.13.0+cu129`、shim `jittor 1.3.11.0`，`--device cuda --repeats 5`。

**package site**（两侧同 py3.12 共用）：

- `/root/jittor-lab/_state/verify-ml/site`：按本文件给出的构建命令装出 **ms-swift 4.5.3 +
  peft 0.20.0**（及 mmcv-lite 2.2.0、mmengine 0.10.7）；pip 拉入的 torch 2.14.0 /
  transformers 5.16.1 / numpy 2.5.3 已从 site 删除，改用 venv 的对应版本。
- `/root/jittor-lab/_state/verify-ml/site-peft17`：`pip install --target site-peft17
  transformers==4.56.2 peft==0.17.1 ms-swift==4.5.2`（4.56.2 是 `jittor_adapters` 允许的
  版本），torch 已删除。

**实际跑通的命令**（site-peft17；原命令的 ms-swift 4.5.3+peft 0.20.0 在 shim 上失败，见下）：

```bash
cd /apdcephfs_private/qy/projects/zy/jittor
source /root/jittor-lab/minimax-h3/env-jittor.sh && unset PYTHONPATH
export JITTOR_HOME=/root/jittor-lab/_state/verify-ml/jittor-home
export CUDA_VISIBLE_DEVICES=4
export JITTOR_ECOSYSTEM_PACKAGE_SITE=/root/jittor-lab/_state/verify-ml/site-peft17
export JITTOR_ECOSYSTEM_REFERENCE_PACKAGE_SITE=/root/jittor-lab/_state/verify-ml/site-peft17
REAL_TORCH_PYTHON=/root/jittor-lab/_state/h3/venv-oracle-cu129/bin/python \
  "$VENV/bin/python" agent/skills/torch-compat-repo-runbook/scripts/verify_repo.py \
  --repo ms-swift --device cuda --repeats 5 --out /root/jittor-lab/_state/verify-ml/out2/ms-swift
```

**支持的 case 清单**：`--list-only` 收 1 条 `ms_swift_lora_llama`（requires
transformers,peft,swift；仓库名里的 `-` 由工具归一化）；`_ecosystem_speed.py` 无大尺寸 case。

| case | 状态 | 精度（最差绝对 / 全场相对） | 速度 torch→jittor | 峰值显存 torch / jittor | fallback | device |
| --- | --- | --- | --- | --- | --- | --- |
| `ms_swift_lora_llama` | ran（30 张量，无缺失） | `1.209e-2` / `2.745e-4`（均 `grad::...embed_tokens.weight`，梯度容差 2e-2 内） | `5.77 → 11.84 ms`，`2.05x` | `65.0 MiB / 4.0 MiB` | 0 | 两侧 cuda |

**显存口径警告**：torch 报自身分配器真实峰值，jittor 是 `profile_memory_enable` 下 sync
后采样的 allocator 用量，不可直接比较，只并列原始值。tiny case（hidden 64 / seq 8）的
`2.05x` 由 Python 派发主导，不是 kernel 性能结论。

**没跑到的及原因**：原命令解析出的 **ms-swift 4.5.3 + peft 0.20.0** 在 shim 上构造 tuner
时 `TypeError: Linear.__init__() missing 1 required positional argument: 'config'`
（`swift/tuners/lora_layers.py` 与 peft 0.20 的 `Linear` 签名不符）——即第 135 行所述
ms-swift/peft 版本组合问题，与 jittor 无关；换成报告用的 **ms-swift 4.5.2 + peft 0.17.1 +
transformers 4.56.2** 即跑通。原始报告见
`/root/jittor-lab/_state/verify-ml/{ms-swift.unpinned-peft0.20.log,ms-swift.pinned.log}`。
