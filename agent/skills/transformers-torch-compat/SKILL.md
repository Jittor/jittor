---
name: transformers-torch-compat
description: 在 Jittor torch shim 与独立 PyTorch 两侧跑通并对比 HuggingFace Transformers，给出 shim/oracle 解释器、双解释器 runner 用法、同权重数值对拍与速度口径、断点分流与假绿清单。用于验证 GPT-2/Llama/BERT/ViT/T5/Whisper/Qwen3 在 shim 上的数值和速度，或定位 transformers 相关断点该修 jittor 核心、jittor.compat.torch 还是 adapter 时使用。
---

# Transformers 在 torch shim 上跑 + 与原生 torch 对拍

## 用途

回答三件事：Transformers 怎么在 shim 上跑、怎么在原生 torch 上跑、两侧怎么对拍
（数值与速度）。覆盖 `_ecosystem_cases.py` 的 6 个 tiny case（`transformers_gpt2`、
`_llama`、`_bert`、`_vit`、`_t5`、`_whisper`）与 `_ecosystem_speed.py` 的 5 个真实尺寸
case（`large_transformers_llama`、`_gpt2`、`_qwen3`、`_bert`、`_vit`）。

**不覆盖**：`transformers` 的完整训练/微调、真实 checkpoint 与数据集、generation 采样；
任何 `_C` 编译扩展（flash-attn、bitsandbytes、torchvision.ops）；vLLM 等推理引擎；
NPU 专项（条目存在但本机未跑）。深度性能分析（SDPA 后端、算子热点、allocator）走
[`jittor-transformers-perf`](../jittor-transformers-perf/SKILL.md)，**注意它文档里的解释器
路径**（`/home/zy/miniconda3/envs/jt311/bin/python`、`/home/zy/rt_venv/bin/python`）**来自旧机器，
不描述当前 lab**；当前 lab 以本文件「两侧环境」为准。

## 两侧环境

| 角色 | 解释器 | 环境脚本 |
| --- | --- | --- |
| shim | `/root/jittor-lab/_state/h3/venv-jittor/bin/python` | `/root/jittor-lab/minimax-h3/env-jittor.sh` |
| 原生 torch（oracle） | `/root/jittor-lab/_state/h3/venv-oracle-cu129/bin/python` | `/root/jittor-lab/minimax-h3/env-oracle-cu129.sh` |

- shim 侧：Python 3.12.12，Jittor 1.3.11.0；`source env-jittor.sh` 后必须
  **先 `import jittor` 再 `import torch`**，`torch` 才带 `_torch_compat_install_context`
  标记（本机已实测）。env 设 `JITTOR_TORCH_SHIM=1`、`HF_HUB_OFFLINE=1`、
  `TRANSFORMERS_OFFLINE=1`，`JITTOR_HOME`/`TMPDIR`/`XDG_CACHE_HOME` 都在
  `$RUN_ROOT`（`/root/jittor-lab/_state/h3/run`）下。
- oracle 侧：Python 3.12.12，真 PyTorch **2.13.0+cu129**。
- **Transformers 两侧都是 5.5.3**（本机实测 `importlib.metadata.version` 与
  `transformers.__version__`）。两侧 Python 主版本相同（3.12）→ ABI 相同 → harness
  默认让两侧**共用同一份下游 site**（shim venv 的 `site-packages`，即
  `JITTOR_ECOSYSTEM_PACKAGE_SITE` 的推导值）；本机已验证 oracle 能直接 import 该 site 的
  `transformers 5.5.3`。这也意味着**两侧同版本是数字可比的前提**，换机器先用下面的命令
  各测一次，不一致就别看数字。
- `env-jittor.sh` 把 `$JITTOR_LAB_ROOT/diffusers-main/src` 放进 `PYTHONPATH`。对纯
  transformers case **不影响数字**（这些 case 不 import diffusers）；harness 在起 oracle
  子进程时会把继承的 `PYTHONPATH` 清空（`_ecosystem_harness.py:254`），所以不会把
  shim 侧的 diffusers 泄漏进 oracle。
- `env-jittor.sh` **不把 venv 放进 `PATH`**，`python` 仍是系统解释器；必须显式用
  `"$VENV/bin/python"`（`VENV` 由脚本导出）。

任何数字之前先过 oracle 断言（`noxfile.py` 的 `ecosystem` session 起手就做同一件事）：

```bash
/root/jittor-lab/_state/h3/venv-oracle-cu129/bin/python -c \
  "import torch; assert not hasattr(torch,'_torch_compat_install_context'); print('oracle torch', torch.__version__)"
# 本机实测输出：oracle torch 2.13.0+cu129
```

反向确认 shim 一侧（证明 `torch` 不是真 torch）：

```bash
source /root/jittor-lab/minimax-h3/env-jittor.sh
"$VENV/bin/python" -c "import jittor, torch; assert hasattr(torch,'_torch_compat_install_context'); print('shim jittor', jittor.__version__)"
# 本机实测输出：shim jittor 1.3.11.0
```

Transformer 专用 adapter 只做一件事：把 `transformers.utils.import_utils.is_torch_npu_available`
钉成返回 `False`，避免原生 `torch_npu` 探测进入独立 Jittor 前端
（[`adapters/jittor_adapters/transformers.py`](../../../adapters/jittor_adapters/transformers.py)，
`SUPPORTED_VERSIONS = {4.56.2, 5.5.3}`）。本机实测该 guard 已生效
（`_jittor_transformers_npu_guard is True`，`is_torch_npu_available() is False`）。

## 在 shim 上跑

唯一入口是 runner（`--help` 本机两个解释器都实测可用）：

```bash
source /root/jittor-lab/minimax-h3/env-jittor.sh
cd /apdcephfs_private/qy/projects/zy/jittor
mkdir -p "$JITTOR_LAB_ROOT/_state/h3/ecosystem"
"$VENV/bin/python" compat/tests/torch/_ecosystem_runner.py \
    transformers_llama "$JITTOR_LAB_ROOT/_state/h3/ecosystem/llama_shim.npz" \
    --runtime jittor --device cpu
```

runner 在 `--runtime jittor` 时自己设 `JITTOR_TORCH_SHIM=1`、先 `import jittor`，并在
`jt.runtime.scope(backend_fallback="error")` + `forbid_backend_fallbacks()` 内跑；输出
一行 `ECOSYSTEM_RESULT {...}` JSON（含 `seconds`、`device`、`fallback_count`、
`dependencies`、`tf32`）。**注意**：直接这样调用 runner 时 `import jittor` 解析到
**deployed site-packages 的 jittor 拷贝**，不是你刚改的 checkout；要测仓库改动，走下面的
pytest（`conftest` 会把 checkout 钉上 `PYTHONPATH`），或先把改动拷进 deployed 树。

整套 shim gate（pytest 用 shim venv 跑，`PYTHON=sys.executable`）：

```bash
source /root/jittor-lab/minimax-h3/env-jittor.sh
cd /apdcephfs_private/qy/projects/zy/jittor
REAL_TORCH_PYTHON=/root/jittor-lab/_state/h3/venv-oracle-cu129/bin/python \
JITTOR_REQUIRE_REAL_TORCH=1 JITTOR_TEST_REQUIRE_EXECUTION=1 JITTOR_TORCH_SHIM=1 \
"$VENV/bin/python" -m pytest compat/tests/torch/test_ecosystem_parity.py -q -k transformers
```

`-k transformers` 选中 6 个 tiny case；跑速度那半再加 `JITTOR_ECOSYSTEM_LARGE=1`、
`JITTOR_ECOSYSTEM_REPEATS=10`（`test_ecosystem_speed.py` 会拒绝小于 10 的值）。

## 在原生 torch 上跑

同一个 runner、同一份 case，用 oracle 解释器与 `--runtime torch`。人工双解释器流程里
oracle 也要清掉 Jittor 变量并零 `PYTHONPATH`，否则可能 import 到 deployed facade：

```bash
cd /apdcephfs_private/qy/projects/zy/jittor
O=/root/jittor-lab/_state/h3/venv-oracle-cu129/bin/python
PYTHONPATH= JITTOR_TORCH_SHIM= \
  "$O" compat/tests/torch/_ecosystem_runner.py \
    transformers_llama "$JITTOR_LAB_ROOT/_state/h3/ecosystem/llama_torch.npz" \
    --runtime torch --device cpu
```

不带 `--weights` 时 runner 会把模型参数与 buffer 存成 `<output>.weights.npz`，这就是
oracle 侧的标准产物。harness 自己调用 oracle 时会清 `PYTHONPATH` 并删除
`JITTOR_SOURCE_ROOT`/`JITTOR_HOME`/`JITTOR_TORCH_SHIM`/`JT_BACKEND` 等（见
`_ecosystem_harness.py:254-260`）。

## 对拍

**数值**：case 只写一次、吃 `torch` 吐 `(model, inputs)`，永远不知道自己跑在哪个 runtime；
seed、权重传递、序列化都归 runner。torch 侧产出 `*.weights.npz`，jittor 侧用
`--weights` 原样载入（存在 `--weights` 时 runner 会校验两边参数/缓冲一一对应），所以
「同权重同输入」是结构保证。gate 断言 forward 与**每个参数梯度、每个输入梯度**都一致；
容差 CPU `2e-3/1e-2`、加速卡 `5e-3/2e-2`（`_ecosystem_harness.py`）。手动两段：

```bash
source /root/jittor-lab/minimax-h3/env-jittor.sh
cd /apdcephfs_private/qy/projects/zy/jittor
D="$JITTOR_LAB_ROOT/_state/h3/ecosystem"
O=/root/jittor-lab/_state/h3/venv-oracle-cu129/bin/python
topk=transformers_bert
PYTHONPATH= "$O" compat/tests/torch/_ecosystem_runner.py "$topk" "$D/torch.npz" --runtime torch --device cpu
"$VENV/bin/python" compat/tests/torch/_ecosystem_runner.py "$topk" "$D/jittor.npz" --runtime jittor --device cpu --weights "$D/torch.weights.npz"
```

对比两个 npz 的 `__output__`、`grad::*`、`ingrad::*` 用**全场最大量级**做 scale
（`_divergence` 的 floor 逻辑），不要用单参数相对误差——数学上为 0 的梯度会被噪声放大成假 bug。

**速度**：`--repeats N`，runner 报 N 次里的**最小值**（干扰只会让样本变慢）。
`test_ecosystem_speed.py` 强制 `N>=10`。同一 case 两侧交替、多次重跑再引用最小值，单次
样本在这类共享机器上不可信。**墙钟永远只报告，不在 PR 门禁断言**；只有 nightly 用
`JITTOR_ECOSYSTEM_SPEED_RATIO`（nox 默认上限 `1.07`）断言。真实尺寸速度 case 是
llama/gpt2/qwen3/bert/vit；注意 **qwen3 只有速度 case，没有 tiny 数值 case**，t5/whisper
只有 tiny 数值 case、没有大尺寸速度 case。

**device 规则**：两侧必须同 device，harness 读回并断言 `report["device"]`。Jittor 没有
per-tensor device，有卡机器上 CUDA 默认就是开的（本机 Jittor 日志实测 `CUDA enabled`，
8 卡 sm_90），所以 CPU 必须显式关：runner 的 `--device cpu`（也是默认值）会进入
`jt.runtime.scope(use_cuda=0)`。绕开 runner 直接跑模型时不会自动关，会拿 Jittor 加速卡对
PyTorch CPU。

## 断点与分流

先按 [`downstream-library-adaptation`](../downstream-library-adaptation/SKILL.md) 的三行判据
分流；transformers 的默认去处是第二行，**默认不需要 adapter**。

| 断点 | 分流 | 依据 |
| --- | --- | --- |
| 原生 `torch_npu` 探测进入独立 Jittor 前端 | **adapter** | 库私有实现细节，已由 `adapters/jittor_adapters/transformers.py` 的 NPU guard 处理；版本门 `SUPPORTED_VERSIONS={4.56.2,5.5.3}`，失败 fail-closed |
| `param.grad` 曝光为 `None`（计算对、只暴露错） | **jittor.compat.torch** | 已在 compat 修（见 [`jittor-torch-diff`](../jittor-torch-diff/SKILL.md)），不是核心缺能力 |
| 拼写/签名与 torch 不同的 API | **jittor.compat.torch** | compat 只做适配；出现第二套实现即放错位置 |
| 能力本身 jittor 没有（例：NPU bool 归约，公共 `jt.all`/`Tensor.all`/`jt.any` 是 Transformers 需要的路径） | **jittor 核心** | [`known-issues` KI-BACKEND-001](../../manuals/known-issues.md) |
| `*._C` 编译扩展（flash-attn / bitsandbytes / torchvision.ops） | **契约外** | Python shim 无法使其 ABI 兼容 |

本机**没有**独立列出 5.5.3 版本下全部 transformers 断点；上表是分流判据与已入仓的实例，
不是完整缺口清单。断点先复现、再分流，不先写适配代码。

## 坑与假绿

1. **`import torch` 单独用不了。** shim venv 里直接 `python -c "import torch"` 抛
   `ModuleNotFoundError`；必须先 `import jittor`（runner 内部就是这顺序）。只 source
   `env-jittor.sh` 不设 `JITTOR_TORCH_SHIM` 也不够。
2. **`env-jittor.sh` 不激活 venv。** `python` 是系统解释器、没有 torch；用 `"$VENV/bin/python"`。
3. **有卡机器上「跑 CPU」默认不是 CPU。** 只「不请求 CUDA」不等于 CPU 对拍，必须显式关；
   见「对拍」的 device 规则。
4. **缺 oracle = skip = 看起来通过。** `REAL_TORCH_PYTHON` 未设时 parity/speed 全部
   `skipTest`，缺 oracle 的 nightly 会为它唯一存在的理由报绿。用
   `JITTOR_REQUIRE_REAL_TORCH=1` + `JITTOR_TEST_REQUIRE_EXECUTION=1` 把这类 skip 变失败。
5. **COPY-DEPLOY。** deployed jittor 是 `site-packages/jittor` 的**真实拷贝**（非 symlink），
   本机实测其 `__init__.py` 与仓库 `python/jittor/__init__.py` 已不同。改仓库 `python/jittor/`
   不会影响直接调用 runner 时 `import jittor` 的结果，除非把改动拷进 deployed 树或走会钉
   checkout 的 pytest。
6. **并行 JIT 抢同一缓存。** `env-jittor.sh` 把 `JITTOR_HOME` 固定在 `$RUN_ROOT`，本机观测到
   该缓存锁被另一个进程（`probe_decode_flag_matrix.py`）持有。并行任务必须另设
   `JITTOR_HOME` 或 `cache_name`，否则会在 `jittor.lock` 上串行等待。
7. **两侧版本不一致就换了参考物。** harness 断言两侧 `transformers` 版本（及共享 site 时的
   依赖 origin）相等；本机两侧当前都是 5.5.3，**未出现**版本不一致。换机器先核对。
8. **tiny case 通过 ≠ 库真正用到的 API 面是真的。** 见
   [`torch-shim-noop-audit`](../torch-shim-noop-audit/SKILL.md)。

## 证据

- 用例与契约：[`_ecosystem_cases.py`](../../../compat/tests/torch/_ecosystem_cases.py)、
  [`_ecosystem_speed.py`](../../../compat/tests/torch/_ecosystem_speed.py)、
  [`_ecosystem_harness.py`](../../../compat/tests/torch/_ecosystem_harness.py)（环境变量契约见
  其模块 docstring）、[`_ecosystem_runner.py`](../../../compat/tests/torch/_ecosystem_runner.py)。
- 门禁：[`test_ecosystem_parity.py`](../../../compat/tests/torch/test_ecosystem_parity.py)、
  [`test_ecosystem_speed.py`](../../../compat/tests/torch/test_ecosystem_speed.py)、
  [`noxfile.py`](../../../noxfile.py) 的 `ecosystem` session。
- Adapter：[`adapters/jittor_adapters/transformers.py`](../../../adapters/jittor_adapters/transformers.py)。
- 对拍/分流方法：[`jittor-torch-diff`](../jittor-torch-diff/SKILL.md)、
  [`downstream-library-adaptation`](../downstream-library-adaptation/SKILL.md)；
  更深的性能方法：[`jittor-transformers-perf`](../jittor-transformers-perf/SKILL.md)。

**本机实测**（2026-09-19，仓库 HEAD `90fe0b9`，分支 `2.0-refactor`）：
两个 venv 存在（Python 均 3.12.12）；shim venv `jittor 1.3.11.0`、`import jittor` 后 `torch`
带 shim 标记；oracle 断言通过（`torch 2.13.0+cu129`）；两侧 `transformers` 均 5.5.3，
oracle 能 import shim site 的 `transformers 5.5.3`；shim 侧 `import transformers` 成功；
adapter NPU guard 生效；runner `--help` 两个解释器均可用；shim venv 有 `pytest 9.1.1`、
两侧 `numpy 2.3.5`；deployed jittor 为拷贝且与仓库 `__init__.py` 不同；shim venv 裸
`import torch` 报 `ModuleNotFoundError`。

**未在本机验证**：任何 transformers 的数值或速度数字；整套 `test_ecosystem_parity/speed`
是否在当前 lab 一次跑通；CUDA `--device cuda` 与 NPU 路径；`JITTOR_ECOSYSTEM_PACKAGE_SITE`
显式路径（本机走的是从已装 transformers 推导的默认值）。

## 实测（2026-09-19）

用四轴工具 `agent/skills/torch-compat-repo-runbook/scripts/verify_repo.py` 实跑。工具让 oracle 与
shim 吃**同一份权重、同一批输入**，每 case 报 min-over-repeats 秒数、worst abs / worst rel
（相对**全场**最大幅值）、两侧 peak 字节、fallback 数、device 一致性。环境：GPU 2（H20），
`--device cuda --repeats 5 --seed 0`，shim/oracle 用上文两侧解释器；
`JITTOR_HOME=/root/jittor-lab/_state/verify-tf/jittor-home`（独立缓存，避免与其他 agent 抢锁）。

命令（同一工具分两档跑，`--cases` 只决定派发哪些 case，不改任何执行路径）：

```bash
source /root/jittor-lab/minimax-h3/env-jittor.sh
export JITTOR_HOME=/root/jittor-lab/_state/verify-tf/jittor-home
mkdir -p "$JITTOR_HOME"
export CUDA_VISIBLE_DEVICES=2
REAL_TORCH_PYTHON=/root/jittor-lab/_state/h3/venv-oracle-cu129/bin/python \
"$VENV/bin/python" agent/skills/torch-compat-repo-runbook/scripts/verify_repo.py \
  --repo transformers \
  --cases transformers_bert,transformers_gpt2,transformers_llama,transformers_t5,transformers_vit,transformers_whisper \
  --device cuda --repeats 5 --out /root/jittor-lab/_state/verify-tf/out
# large 速度档把 --cases 换成：
#   large_transformers_bert,large_transformers_gpt2,large_transformers_llama,large_transformers_qwen3,large_transformers_vit
```

注册 case 共 11 个：6 个 tiny parity + 5 个 large 速度。实跑成 10 个（tiny 5/6、large 5/5），
全部 `fallback_count=0`、`device_agreement=true`。worst 误差都落在梯度（`grad::*`）而不是输出。

### tiny parity 档（CUDA，repeats=5）

| case | 状态 | worst abs | worst rel（全场） | torch s | jittor s | jittor/torch | torch peak | jittor peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| transformers_bert | ran | 5.82e-4 | 2.48e-6 | 0.00300 | 0.00946 | 3.15 | 64.8 MiB | 3.0 MiB |
| transformers_gpt2 | ran | 2.12e-2 | 3.61e-4 | 0.00309 | 0.00950 | 3.07 | 65.2 MiB | 4.0 MiB |
| transformers_llama | ran | 1.21e-2 | 2.75e-4 | 0.00561 | 0.00958 | 1.71 | 65.0 MiB | 4.0 MiB |
| transformers_vit | ran | 1.96e-2 | 1.31e-4 | 0.00294 | 0.00944 | 3.21 | 65.2 MiB | 4.0 MiB |
| transformers_whisper | ran | 5.39e-2 | 2.52e-4 | 0.00867 | 0.01929 | 2.23 | 66.4 MiB | 8.0 MiB |
| transformers_t5 | **failed** | — | — | — | — | — | — | — |

worst rel 全部 ≤ 3.6e-4，远低于加速卡容差 `5e-3/2e-2`。

### large 速度档（CUDA，repeats=5；门禁只测速度、不断言这档数值）

| case | 状态 | worst abs | worst rel（全场） | torch s | jittor s | jittor/torch | torch peak | jittor peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| large_transformers_bert | ran | 2.90 | 3.71e-4 | 0.0614 | 0.0701 | 1.14 | 2129.5 MiB | 5870.0 MiB |
| large_transformers_gpt2 | ran | 10.61 | 8.18e-4 | 0.0882 | 0.0908 | 1.03 | 4163.2 MiB | 9186.2 MiB |
| large_transformers_llama | ran | 9.07 | 8.33e-4 | 0.0928 | 0.1010 | 1.09 | 3899.6 MiB | 9512.0 MiB |
| large_transformers_qwen3 | ran | 15.87 | 1.51e-3 | 0.0926 | 0.1105 | 1.19 | 4536.0 MiB | 9401.0 MiB |
| large_transformers_vit | ran | 0.31 | 7.51e-4 | 0.0530 | 0.0619 | 1.17 | 1712.7 MiB | 5038.0 MiB |

（absolute 误差看着大只是因为该档张量幅值大；全场相对误差最大 1.51e-3，仍在容差内。整档
墙钟：parity 15m39s、large 7m56s，含首次 JIT 编译；表内是 min-over-repeats。）

**显存列是旧口径，且旧结论是错的（2026-09-19 复核）**：上表两列由当时的工具产出——
oracle 报 `torch.cuda.max_memory_allocated`（**活跃**字节，单卡），jittor 报
`jt.get_mem_info().total_cuda_used`（**活跃+缓存空闲**，且 `mem_info.cc` 对**所有**设备求和）。
那是 reserved 对 allocated，不是同一个量纲，工具已改为两侧都报 live 与 pool 两个数
（`verify_repo.py` 的 `_MEMORY_WRAPPER`，jittor 侧用 `device_memory_used` /
`device_memory_reserved`）。**不要**继续引用上表两列相除的倍数；下表数字待用新工具重测。

但是：当时写的「两个数不可直接比、所以不能读成 jittor 真占更多显存」**结论也是错的**。
`large_transformers_bert` 上按同口径实测（GPU 2，同权重同输入，`repeats=1`，仓库 checkout）：

| 口径 | torch | jittor | 比 |
| --- | --- | --- | --- |
| live（`max_memory_allocated` / `device_memory_used`） | 2129.5 MiB | 6131.2 MiB | **2.88x** |
| pool（`max_memory_reserved` / `device_memory_reserved`） | 2386.0 MiB | 6588.0 MiB | **2.76x** |
| 旧工具那一对（allocated / `total_cuda_used`） | 2129.5 MiB | 6588.0 MiB | 3.09x |

jittor 的缓存空闲只有 `6588.0 - 6131.2 = 456.8 MiB`（占 pool 的 7%），所以这**不是**分配器
攒着不放的假象：jittor 在这个 case 上真实持有的活跃显存约为 torch 的 **2.9 倍**。把口径换成
like-for-like 只把 3.09x 挪到 2.88x，没有消掉它。`fuse_op_limit` 0 与 16 两档数字完全一致
（6588.0 / 6131.2），所以 §47 的融合宽度上限不是这里的杠杆。

数字随 jittor 版本变：deployed 那份（2026-09-11 拷贝，`total_cuda_used=5870.0 MiB`）比仓库
checkout 低 12%，但量级相同。**显存这条应当被当作一个待查的 jittor 问题**，不是口径噪声。


### 未跑与失败

- **`transformers_t5`（CUDA）：oracle 先崩，shim 侧没执行到。** 真实错误是真 torch 2.13.0+cu129
  走 apex：
  `File "/opt/python3.12/lib/python3.12/site-packages/apex/normalization/fused_layer_norm.py", line 254, in fused_rms_norm_affine_fwd ... RuntimeError: input must be contiguous`
  （transformers 5.5.3 的 `T5LayerNorm` 在 CUDA 上命中 apex fused RMSNorm，拒绝非连续输入）。
  与 shim 无关：手工只跑 shim（`--runtime jittor --device cuda`）该 case 成功，
  `seconds=0.01938`、`fallback_count=0`、`device=cuda`。工具先跑 oracle 再跑 shim，oracle 一崩整
  case 就被记为 failed，因此没有 t5 的 CUDA 对拍数字。
- 补一个 t5 **CPU 交叉核对**（同一工具，`--device cpu --no-memory`）：ran，
  worst abs 2.10e-5、worst rel 5.74e-7、torch 0.0204 s、jittor 0.0821 s、ratio 4.02、fallback 0。
  说明 t5 在 shim 上语义/数值正常，卡的只是 CUDA 上的 oracle 环境（apex）。
- 其余 10 个注册 case 全部跑成，无跳过、无 fallback。large 档没有 `t5`/`whisper`（本就没有对应
  large case）、tiny 档没有 `qwen3`（qwen3 只有速度 case），这是 case 注册表本身的范围，不是失败。
