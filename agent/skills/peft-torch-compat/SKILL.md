---
name: peft-torch-compat
description: 在 Jittor 的 torch shim 与原生 PyTorch 两侧运行并对比 PEFT(LoRA)，用于验证 peft_lora_llama 用例的前向/全部梯度一致性、排查 LoRA 冻结与 adapter 存取；不覆盖 ms-swift 自带 tuner、真实 checkpoint 微调和分布式训练。
---

# peft × Jittor torch shim

## 用途

回答「PEFT 架在 `import torch` → Jittor 上能否跑、和真 PyTorch 是否同数、快不快」。
上游用例：`../../../compat/tests/torch/_ecosystem_cases.py:_peft_lora_llama`（case 名
`peft_lora_llama`）；单库契约：`../../../compat/tests/torch/test_peft.py`。

**不覆盖**：ms-swift 的 `Swift` / `swift.tuners.LoRAConfig`（名字相同但不是 PEFT，见
`../ms-swift-torch-compat/SKILL.md`）；真实 checkpoint 全量微调、分布式/FSDP、量化。
分流判据、定级、device 阶梯、adapter 准入走
`../downstream-library-adaptation/SKILL.md`，本文件不重复协议，只给 peft 的可执行入口。

## 两侧环境

本机（`/root/jittor-lab`）已实测：

| 侧 | 解释器 | 关键事实（`importlib.metadata` / import 探针） |
| --- | --- | --- |
| shim | `/root/jittor-lab/_state/h3/venv-jittor/bin/python` | py3.12.12；`jittor` 1.3.11.0；`import torch` → shim（`hasattr(torch,'_torch_compat_install_context')` 为 True）；transformers 5.5.3、safetensors 0.8.0、accelerate 1.15.0、numpy 2.3.5、`jittor-torch-adapters` 1.3.11.0 |
| 原生 torch | `/root/jittor-lab/_state/h3/venv-oracle-cu129/bin/python` | py3.12.12；torch 2.13.0+cu129；transformers 5.5.3、numpy 2.3.5；无 jittor |

先 `source /root/jittor-lab/minimax-h3/env-jittor.sh`（原生侧对应
`env-oracle-cu129.sh`）：

- 导出 `JITTOR_TORCH_SHIM=1`，并把 `JITTOR_HOME`/`TMPDIR`/`XDG_CACHE_HOME` 收进
  `$JITTOR_LAB_ROOT/_state/h3/run`，JIT 缓存隔离；
- 设 `JT_BUILD_PYTHON_CONFIG_PATH=/opt/python3.12/bin/python3.12-config`——**不设它
  `import jittor` 直接 `RuntimeError: python3.12-config not found`**（实测）；
- 导出 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`。

两侧同为 py3.12，ABI 相同，可以共用同一个下游 package site
（`_ecosystem_harness.REFERENCE_SHARES_PACKAGE_SITE` 成立），只在 ABI 不同时才需要
`JITTOR_ECOSYSTEM_REFERENCE_PACKAGE_SITE`。

**看任何数字前先过 oracle 断言**（`noxfile.py` 的 `ecosystem` session 起手也做同一件事）：

```bash
/root/jittor-lab/_state/h3/venv-oracle-cu129/bin/python -c \
  "import torch; assert not hasattr(torch,'_torch_compat_install_context'), 'oracle 是 shim'; print('oracle', torch.__version__)"
# 本机实测：oracle 2.13.0+cu129
```

shim 身份同法可查（实测 `torch True`，即 torch 名、对象属 jittor 前端）：

```bash
source /root/jittor-lab/minimax-h3/env-jittor.sh
"$VENV/bin/python" -c "import jittor, torch; print(torch.__name__, hasattr(torch,'_torch_compat_install_context'))"
```

**本机 peft 未安装**：shim 与 oracle 两侧 `importlib.metadata.version('peft')` 都是
`PackageNotFoundError`，`find_spec('peft')` 为 False。补上之前，任何 peft 数字都是未验证。

## 在 shim 上跑

环境契约是 ecosystem 门禁把下游库放在 `JITTOR_ECOSYSTEM_PACKAGE_SITE` 指向的
site-packages 里（以 `../../../compat/tests/torch/_ecosystem_harness.py` 模块 docstring
为准）。本机两侧同 py3.12，一个共享 site 即可。构建办法（**本机未执行**，按契约给出）：
建空目录 `<site>`，`pip install --target <site> peft==0.17.1`（0.17.1 是
`refactor-wip/results/2026-08-24-optional-compat-cuda-gate.md` 验证过的版本），并保证两侧
transformers 等依赖版本一致（harness 会断言依赖版本相同）。**不要**把新库 `pip install`
进 oracle venv 去覆盖 torch，那会作废整个对拍（见 downstream-library-adaptation 第 3 节）。

跑单条 case（pytest 必须在 shim 解释器里起）：

```bash
set -a; source /root/jittor-lab/minimax-h3/env-jittor.sh >/dev/null 2>&1; set +a
unset PYTHONPATH                      # env 脚本把 PYTHONPATH 指向 diffusers checkout，对拍不需要
cd /apdcephfs_private/qy/projects/zy/jittor
export REAL_TORCH_PYTHON=/root/jittor-lab/_state/h3/venv-oracle-cu129/bin/python
export JITTOR_ECOSYSTEM_PACKAGE_SITE=<含 peft 的 site>
"$VENV/bin/python" -m pytest -q \
  "compat/tests/torch/test_ecosystem_parity.py::EcosystemParity::test_peft_lora_llama"
```

CUDA 档换 `EcosystemParityCUDA::test_peft_lora_llama`。**NPU 档没有 peft 用例**
（`test_ecosystem_parity.py:EcosystemParityNPU` 只收 diffusers/mmcv/mmengine/ms-swift）。
本机 `--collect-only` 实测：该文件收集 28 项，含上述两个 peft 节点；但冷启动 import
jittor + CUDA 就花掉约 5m10s，单条 case 不会秒回。

单库契约（不依赖 `REAL_TORCH_PYTHON`，单解释器，覆盖 LoRA 冻结与梯度语义、200 步拟合、
adapter `save_pretrained → PeftModel.from_pretrained` roundtrip）：

```bash
source /root/jittor-lab/minimax-h3/env-jittor.sh; unset PYTHONPATH
"$VENV/bin/python" -m pytest -q compat/tests/torch/test_peft.py     # 期望 3 passed
```

**或用 runner CLI 手动分两步**（`../../../compat/tests/torch/_ecosystem_runner.py`，CLI 实测）：
`python _ecosystem_runner.py <case> <output.npz> [--weights W] [--repeats N] [--seed N] [--runtime {torch,jittor}] [--device {cpu,cuda,npu}]`

```bash
cd compat/tests/torch
# 1) oracle 造权重 + 参考值，写 <output>.weights.npz
env -u PYTHONPATH -u JITTOR_TORCH_SHIM \
  /root/jittor-lab/_state/h3/venv-oracle-cu129/bin/python \
  _ecosystem_runner.py peft_lora_llama /tmp/torch.npz --runtime torch --device cpu
# 2) shim 载同一权重复算
source /root/jittor-lab/minimax-h3/env-jittor.sh
JITTOR_ECOSYSTEM_PACKAGE_SITE=<含 peft 的 site> "$VENV/bin/python" \
  _ecosystem_runner.py peft_lora_llama /tmp/jittor.npz --runtime jittor --device cpu \
  --weights /tmp/torch.weights.npz
```

`--runtime` 默认是 `torch`；在 shim 解释器里**必须显式 `--runtime jittor`**，否则会去要
真 PyTorch。`--runtime jittor` 由 runner 自己设 `JITTOR_TORCH_SHIM=1` 并先 `import jittor`。

## 在原生 torch 上跑

原生侧没有对应的 PEFT 单测（`test_peft.py` 本身 `import jittor`）。原生侧的入口就是
harness 的 oracle 半边，即上面第 1 步 `--runtime torch`；或在 `test_ecosystem_parity.py`
里由 `REAL_TORCH_PYTHON` 自动调起同一命令图。

## 对拍

- **数字**：harness 结构保证「同权重同输入」——oracle 先跑并落盘全部
  `named_parameters()+named_buffers()`，shim 侧 `copy_` 同一组值再跑；比较前向输出
  `__output__`、每个参数梯度和每个输入梯度（`EcosystemComparison._compare`）。
  容差：CPU `forward 2e-3 / backward 1e-2`，CUDA 放大到 `5e-3 / 2e-2`，并按全场最大量级取
  floor 防「数学上为 0 的梯度被噪声放大成假 bug」。缺任一梯度即失败。
- **速度**：harness 报告 `min(durations)`（干扰只会让样本更慢），parity 默认 `--repeats 3`；
  speed 模块要求 `JITTOR_ECOSYSTEM_REPEATS >= 10`。**只在设了
  `JITTOR_ECOSYSTEM_SPEED_RATIO` 时才断言墙钟**，PR 门禁不断言（`noxfile.py` 的
  `ECOSYSTEM_SPEED_RATIO` 只在 nightly）。重复采样、取最小、报告而非门禁。
- **peft 没有 speed case**：`../../../compat/tests/torch/_ecosystem_speed.py` 的 `CASES`
  只有 transformers/diffusers/convnet，没有 peft 的 large 配置。真实尺寸性能必须先在
  `_ecosystem_speed.CASES` 加一个 case，或用 runner 自定义 `--repeats` 单独量，不要在
  parity 的 tiny case 上读速度（那里量的是 dispatch 开销）。
- **device 规则**：两侧必须同 device。Jittor 没有 per-tensor device，本机 `import jittor`
  日志直接打印 `CUDA enabled`（实测），所以「不请求 CUDA」的 CPU 对拍实际是 Jittor 在 GPU
  上对 PyTorch CPU。harness 用 `jt.runtime.scope(use_cuda=0)` 显式请求 CPU，并读回
  `report['device']` 断言两侧都在 `cpu`；runner 只认 `--device cpu/cuda/npu` 三档。

## 断点与分流

peft 是纯 `import torch` 消费者，**第三行（adapter）为空**；修复只落前两行：

| 现象 | 分流 |
| --- | --- |
| 基础算子/自动微分不对 | jittor 核心（`python/jittor/`） |
| `torch` 拼写、签名、属性缺失 | `jittor.compat.torch` |
| LoRA 冻结依赖 `requires_grad` 语义 | compat.torch 的 autograd/参数语义 |
| adapter 存取的 `safetensors` 读写 | compat.torch 或 jittor 核心，视缺什么而定 |

`test_peft.py` 的 `_is_active_jittor_frontend` 是本仓对「shim 是否真的在生效」的判据：
用 `jittor.compat.torch.publication.namespace_owner` + `jittor.compat.shim.runtime.activation_status`
判断，而不是看模块名。

## 坑与假绿

- **用 `torch.__name__` 判定 shim**：历史事故（optional CUDA 报告）里，测试要求
  `torch.__name__ == "torch"`，导致已装 PEFT 时 3 个测试全部静默 skip。现在按对象/命名空间
  归属判断；`test_peft.py` 的 `_is_active_jittor_frontend` 是正确写法。
- **`3 skipped` 看起来和通过一样**：缺 peft 时整个类 `skipUnless(_HAS)`。在验收机上用
  `JITTOR_REQUIRE_OPTIONAL_DEPS=1`（`nox -s optional` 会设）把「没装」变成显式失败。
- **LoRA B=0 初始化的假红**：标准 LoRA 第一步 `lora_A` 梯度**恰好为 0**、`lora_B` 非 0。
  写「所有梯度非零」的检查会误报；契约测试用的是这个精确语义。
- **在 tiny parity case 上读速度**：batch 2 / hidden 64 的步长被 Python dispatch 主导，
  ratio 说明不了 kernel 性能。
- **把 peft 的 LoRA 当成 ms-swift 的**：两者 API/行为不同，不要复用结论。

## 证据

- **本机已实测**：两个 venv 的包清单（peft/mmengine/mmcv/swift/ms-swift/torchmetrics/tensordict
  全部缺失，只有 transformers/vllm/jittor/torch/numpy 等）；oracle 断言输出
  `oracle 2.13.0+cu129`；shim 身份探针；无 `env-jittor.sh` 时 `import jittor` 报
  `python3.12-config not found`；`test_ecosystem_parity.py` 收集 28 项、含两个 peft 节点、
  冷启动约 5m10s。
- **未在本机验证**：peft 的安装、`peft_lora_llama` 的任何数值/速度、`test_peft.py` 的
  3 passed——因为本机两侧都没有 peft。上述命令按 harness/runner 真实接口给出，但**没有跑过**。
- 维护者报告：`refactor-wip/results/2026-08-24-optional-compat-cuda-gate.md`（PEFT 0.17.1，
  修复后 `3 passed in 126.50s`）；`docs/compatibility/torch.md`（peft 覆盖行）；
  `agent/manuals/project-context.md`。原始 runtime 状态/日志按仓库规则放
  `$JITTOR_LAB_ROOT/_state/<topic>/`，不入库。

## 实测（2026-09-19）

GPU 4，oracle `torch 2.13.0+cu129`、shim `jittor 1.3.11.0`，`--device cuda --repeats 5`。

**package site**（两侧同 py3.12 共用）：

- `/root/jittor-lab/_state/verify-ml/site`：按本文件给出的构建命令 `pip install --target site
  peft ms-swift mmcv-lite mmengine` 装出 **peft 0.20.0**、ms-swift 4.5.3、mmcv-lite 2.2.0、
  mmengine 0.10.7。pip 顺带拉入的 torch 2.14.0 / transformers 5.16.1 / numpy 2.5.3 已从
  site 删除，改用 venv 的 torch 2.13.0 / transformers 5.5.3 / numpy 2.3.5。
- `/root/jittor-lab/_state/verify-ml/site-peft17`：`pip install --target site-peft17
  transformers==4.56.2 peft==0.17.1 ms-swift==4.5.2`（4.56.2 是 `jittor_adapters` 允许的
  版本之一；transformers 走 site 覆盖 venv 的 5.5.3），torch 已删除。

**实际跑通的命令**（site-peft17；原命令的 peft 0.20.0 在 shim 上起不来，见下）：

```bash
cd /apdcephfs_private/qy/projects/zy/jittor
source /root/jittor-lab/minimax-h3/env-jittor.sh && unset PYTHONPATH
export JITTOR_HOME=/root/jittor-lab/_state/verify-ml/jittor-home
export CUDA_VISIBLE_DEVICES=4
export JITTOR_ECOSYSTEM_PACKAGE_SITE=/root/jittor-lab/_state/verify-ml/site-peft17
export JITTOR_ECOSYSTEM_REFERENCE_PACKAGE_SITE=/root/jittor-lab/_state/verify-ml/site-peft17
REAL_TORCH_PYTHON=/root/jittor-lab/_state/h3/venv-oracle-cu129/bin/python \
  "$VENV/bin/python" agent/skills/torch-compat-repo-runbook/scripts/verify_repo.py \
  --repo peft --device cuda --repeats 5 --out /root/jittor-lab/_state/verify-ml/out2/peft
```

**支持的 case 清单**：`--list-only` 收 1 条 `peft_lora_llama`（requires
transformers,peft）；`_ecosystem_speed.py` 无 peft 大尺寸 case。

| case | 状态 | 精度（最差绝对 / 全场相对） | 速度 torch→jittor | 峰值显存 torch / jittor | fallback | device |
| --- | --- | --- | --- | --- | --- | --- |
| `peft_lora_llama` | ran（30 张量，无缺失） | `1.209e-2` / `2.745e-4`（均 `grad::...embed_tokens.weight`，梯度容差 2e-2 内） | `5.80 → 11.82 ms`，`2.04x` | `65.0 MiB / 4.0 MiB` | 0 | 两侧 cuda |

**显存口径警告**：torch 报自身分配器真实峰值，jittor 是 `profile_memory_enable` 下 sync
后采样的 allocator 用量，两者不可直接比较；这里只并列原始值，不主张持平。速度是
tiny case，由 Python 派发主导，`2.04x` 不是 kernel 性能结论。

**没跑到的及原因**：`site` 上的 **peft 0.20.0**（原命令解析结果）在 shim 上 import 即
`ModuleNotFoundError: No module named 'torch.distributions.wishart'`（新依赖
`tuners/lora/monteclora.py`），case 失败；**peft 0.17.1** 又要求 transformers 4.x
（`from transformers import HybridCache`），与本机 venv 的 transformers 5.5.3 不兼容。
只有把它降到 adapter 允许的 `4.56.2` 才跑通。原始报告见
`/root/jittor-lab/_state/verify-ml/{peft.unpinned-peft0.20.log,peft.pinned.log}`。
