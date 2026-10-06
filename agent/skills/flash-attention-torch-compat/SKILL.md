---
name: flash-attention-torch-compat
description: 在 jittor 的 `import torch` shim 上运行官方 flash-attention（含其在 SDPA 上的接线与 stub 假象），并说明在原生 PyTorch 上运行/对拍 flash-attn 需要什么、环境里常缺什么。用于验证 flash-attn 兼容层、判断当前注意力到底走没走 fused kernel、以及规划两解释器对拍时使用。
---

# flash-attention ⇄ jittor torch shim

## 用途

回答：flash-attn 在 shim 上怎么跑、怎么**证明当前注意力真的走了官方 fused 扩展而不是
静默 composite**、在原生 torch 上跑/对拍需要什么、断点该修哪一层。

**不覆盖**：通用 SDPA 语义（见
[`jittor-torch-diff`](../jittor-torch-diff/SKILL.md)）；vLLM 服务化；上游 flash-attention
的训练脚本。也不覆盖 Diffusers/Transformers 侧的注意力性能（那是各自 skill 的事）。

## 两侧环境

| | shim 侧 | 原生 torch 侧 |
|---|---|---|
| 解释器 | `<jittor-python>`（部署了 shim） | `<real-torch-python>`（独立真 PyTorch） |
| 激活 | 本地、不入库的 env 脚本（设下面的 opt-in 变量与隔离缓存） | 同左，但不部署 shim |
| `import flash_attn` 解析到 | venv 内**部署 stub** `<jittor-python> 的 site-packages/flash_attn/__init__.py`（实测） | oracle base 前缀 site-packages 里的**真上游** `flash_attn/__init__.py`，`__version__="2.8.1"`（实测） |

shim 侧 env 脚本为 fused 路径设的 opt-in（实测读出）：

```text
JITTOR_FLASH_ATTN_JITTOR_SRC=$JITTOR_LAB_ROOT/flash-attention
JITTOR_FLASH_ATTN_HEAD_DIMS=64,128
JITTOR_FLASH_ATTN_DTYPES=bf16,fp16
# JITTOR_FLASH_ATTN_JITTOR_REQUIRED 未设（形状/dtype 不覆盖时退回 composite，不 abort）
# JITTOR_FLASH_ATTN_CAST_FLOAT32 未设（避免把 float32 注意力偷偷改走 bf16）
```

源码 checkout：`$JITTOR_LAB_ROOT/flash-attention`，HEAD `5231d95fe13733fb534c01895f7ea88c6a6c7793`，
remote `https://github.com/Dao-AILab/flash-attention.git`（实测）。

- **oracle 断言**（任何数字可信前必须过）：`assert not hasattr(torch,
  '_torch_compat_install_context')`。实测原生侧 `False`（通过），shim 侧 `True`。
- **实测环境的原生侧跑不了真 flash-attn**：`import flash_attn` 在 oracle venv 里失败——
  `ImportError: .../flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so:
  undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib`，即该扩展按旧
  ABI 编译，与 oracle 的 `torch 2.13.0+cu129` 不匹配。要用原生侧必须先按此 torch 重编/重装
  匹配的 flash-attn；在那之前**任何"shim 对真 flash-attn"的对拍都不成立**。
- 缓存隔离与 copy-deploy 规则同其它 skill：`JITTOR_HOME`/`XDG_CACHE_HOME` 隔离；改仓库
  `python/jittor/` 或 `compat/` 不生效，必须拷进 deployed site-packages。

## 在 shim 上跑

```bash
# shim 侧环境已激活，当前目录为仓库根

# A. 维护门禁：`nox -s optional` 先在 math fallback 下跑 deployed adapter 契约
#    （含 compat/tests/torch/test_flash_attn_compat.py），再用 native-required 阶段
#    要求真扩展加载、不许回退 math（phase 之间互相不掩盖，见 noxfile.py）。
export nvcc_path="$(command -v nvcc)"
export JITTOR_FLASH_ATTN_JITTOR_SRC="$JITTOR_LAB_ROOT/flash-attention"
python -m nox -s optional

# B. 只读探针：当前 SDPA 到底走了什么（见"坑与假绿"）
<jittor-python> - <<'PY'
import torch, jittor as jt
import torch.nn.functional as F
from jittor.compat import diagnostics
jt.flags.use_cuda = 1
q = k = v = jt.randn((2, 512, 4, 64)).float16()
with torch.no_grad():
    F.scaled_dot_product_attention(q, k, v)
    jt.sync_all(True)
print(diagnostics.sdpa_flash_stats(jt))   # {"hits":..,"misses":{..},"casts":{..},"backend":..}
PY
```

`compat/tests/torch/test_flash_attn_compat.py` 是本仓库对 flash-attn 兼容路径的**自身契约**
（读它、按它验收）：dense/packed/varlen 前反向对独立 numpy SDPA 参考在 `atol=2e-5` 内一致、
var-len 不跨段泄漏、`dropout_p=1.0` 返回全零；`TestPackedEntryDeviceGuard` 断言生成的每个
packed 入口都绑定输入 tensor 的 device（旧 bug 是 `device_guard{0}`，让 TP2 rank 1 撞
`cudaErrorIllegalAddress`）。

在部署 venv 里直接 `python -m pytest compat/tests/torch/test_flash_attn_compat.py` 会撞上与
[`diffusers-torch-compat`](../diffusers-torch-compat/SKILL.md) 相同的 `compat` 包 pytest 收集
问题；所以这里走 `nox -s optional`，不要裸跑 pytest 当门禁。

实测：上面的探针在 `(2,128,4,64)` fp16、CUDA、**训练态**下回报
`hits=0, misses={'short_training_math': 1}`——路由确实进了
`_try_flash_scaled_dot_product_attention`，但训练态的 `score_elements` 低于
`JITTOR_FLASH_ATTN_TRAINING_MIN_SCORES`（默认 `1<<24`）而按设计退回 composite。
`(2,512,4,64)` fp16 + `no_grad` 的 fused-hit 探针第一次在 150 s 内未出结果（期间有另一 JIT
进程持锁）；fused hit 后来在 2026-09-19 观测到，见文末。

## 在原生 torch 上跑

```bash
# 原生侧环境；上游自带测试在 checkout 内（<flash-attention checkout>/tests/test_flash_attn.py 等）
<real-torch-python> -c "import flash_attn; print(flash_attn.__version__)"
```

**实测这条就失败**（见上，ABI undefined symbol）。要跑通得先在与 oracle 同 ABI 的
环境里从 `$JITTOR_LAB_ROOT/flash-attention` 源码构建/安装匹配的 flash-attn；在完成前，本
skill 不声称跑过任何原生 flash-attn 用例。

## 对拍

- **没有注册的生态 case。** `_ecosystem_cases.py` 的 `CASES`（含合并后的 speed cases）里
  没有 flash-attn 条目——所以**当前不存在可复用的 flash-attn 两解释器对拍**。要建，按
  [`downstream-library-adaptation`](../downstream-library-adaptation/SKILL.md) 第 4 节把 case
  注册进 `CASES`（同权重、同输入、两 interpreter），这是**计划项，尚未做**。
- **数值**：仓库现在用另一种更弱但真实的外部参考——`test_flash_attn_compat.py` 在**同一个
  shim 进程内**把 shim 的输出和独立 `numpy.einsum` softmax 参考比（`atol=2e-5`）。它证明了
  数值正确，但**不是**与真 flash-attn/PyTorch 的对拍，不要混淆这两件事。
- **速度**：重复、交替采样、报最小值；墙钟永不在 PR 门禁里断言。实测环境没有可用的原生 flash-attn
  基线，故**无法给出任何对拍速度**。已知的 shim-侧代价引用：
  `docs/results/2026-09-14-vllm-omni-h3-enablement.md` 与 `agent/manuals/environment.md` 记载的
  MiniMax-H3 视频 VAE 解码 `(1,32,1797,64)` fp16 上，naming the source 把 decode 从
  10.13 s 降到 8.30 s；那是该 workload 的结论，不是 flash-attn 通用性能。
- **Device 规则**：两侧同 device；"CPU" 在 Jittor 侧必须显式关 CUDA，否则 `not_cuda` 会让你
  误判 fused 路径不可用。

## 断点与分流

| 东西 | 分流 | 依据 |
|---|---|---|
| 官方扩展的加载/构建/能力校验（`flashattn_jittor`、`load_backend_for`、capability miss） | `jittor.compat.shim.backends.flash_attention` | `compat/shim/backends/flash_attention/{__init__,official_build,official_codegen,adapter,packed}.py` |
| SDPA 里 fused → composite 的路由与 `sdpa_flash_stats` | `jittor.compat.torch` | `compat/torch/installers/nn/attention.py:_try_flash_scaled_dot_product_attention` |
| `import flash_attn` 的公开 API facade | 部署 stub（compat 资源），不是 adapter | `compat/shim/resources/stubs/flash_attn/` |
| `torch.ops.aten._scaled_dot_product_flash_attention` / `_scaled_dot_product_efficient_attention` / `reshape(.default)` | vLLM adapter（注册这几个名字，两个注意力算子被调用时明确拒绝） | `adapters/jittor_adapters/vllm/aten_ops.py` |
| 其余 aten 名字（`clone`、`copy_`、`view`、`permute`、`mm`、`_scaled_mm` 等）未注册 | jittor 核心/compat（未建通用 aten bridge） | `agent/manuals/known-issues.md` KI-COMPAT-002（Open） |
| 生成入口的 CUDA device guard | compat（已修，有回归测试） | `test_flash_attn_compat.py::TestPackedEntryDeviceGuard` |
| 原生 flash-attn 的 ABI 不匹配 | 环境问题，不是 jittor | 实测 undefined symbol |

**flash-attn 本身没有 `adapters/` 参与**：它走的是 shim 内置桥接（compat 与部署 stub），按三行判据
不需要新建外置 adapter；上表的 aten 名字由 vLLM adapter 注册，是 vLLM 的事。

## 坑与假绿

1. **`import flash_attn` 成功 ≠ 官方扩展在跑。** venv 里部署的是 Jittor 后端的 stub
   （`__version__="2.7.4.post1"`，`_jittor_flash_attn_stub=True`，与仓库
   `compat/shim/resources/stubs/flash_attn/` 内容一致）。它只是 API facade，默认分段落
   composite，不是 fused kernel。
2. **`flash_attn.flashattn_jittor_backend()` 返回 `"math"` 不是证据。** 这个字符串来自 stub
   包，看的是顶层 `flash_attn` 能否找到 `flash_attn_jittor_cuda`，与 SDPA 实际路径是两回事。
   唯一可信的是 `diagnostics.sdpa_flash_stats(jt)` 的 `hits/misses/backend`。
3. **没设 `JITTOR_FLASH_ATTN_JITTOR_SRC` 时只有 `no_backend`。** 没有源 checkout 时扩展
   从不加载，`misses={'no_backend': n}`，composite 静默接管且数值正确——最容易漏。
   其它 miss 是真实守卫：`mask`、`heads`、`head_dim`（只支持 32/64/96/128/192/256）、`dtype`、
   `no_training_backend`、`short_square_math`、`short_training_math`。
4. **fused 分支拒绝任何 `attn_mask`；训练还要 `_flashattn_jittor_training`。** 训练态小形状
   会按 `JITTOR_FLASH_ATTN_TRAINING_MIN_SCORES` 退回 composite（实测）。
5. **扩展构建是串行的、且 digest 含 shim 头文件**：不要中途重启，也不要对同一 digest 目录起
   第二个 builder；改 `c10/cuda/*.h` 之类后必须让 `|shim_hdrs=<sha256>` 进入 digest
   （`compat/shim/backends/flash_attention/official_build.py`），否则陈旧 `.so` 会被继续
   使用。官方扩展落盘在
   `$XDG_CACHE_HOME/jittor/torch-shim/<tag>/torch_extensions/flashattn_jittor/official_flash_attn*`
   ——实测这些 `.so` 存在。
6. **不要把 oracle 侧 base 前缀的真 flash-attn 当成可用**：实测环境里它在 `import` 即失败；
   shim venv 因自身 site-packages 优先而解析到 stub，两个"flash_attn"不是同一个东西。

## 证据

**实测**

- shim 侧 env 脚本的三个 opt-in 变量取值，以及 `JITTOR_FLASH_ATTN_JITTOR_REQUIRED` /
  `_CAST_FLOAT32` 未设。
- shim 侧 `flash_attn` origin 在 `<jittor-python>` 的 site-packages，`__version__="2.7.4.post1"`、
  `_jittor_flash_attn_stub=True`；与仓库 stub 一致。
- oracle 侧 `flash_attn` origin 在 oracle base 前缀的 site-packages，`__version__="2.8.1"`，`import`
  因 `undefined symbol` 失败。
- 官方扩展构建产物 `.so` 存在于 XDG cache 的 `torch_extensions/flashattn_jittor/...`。
- SDPA 探针在 CUDA 训练态回报 `hits=0, misses={'short_training_math': 1}`。
- flash-attention checkout HEAD 与 remote。

**未验证 / 仅计划**

- 第一次探针**未观测到 fused hit**（150 s 超时）；2026-09-19 补测命中，见下节。
- **没有**任何 shim-vs-原生 flash-attn 的两解释器数值或速度对拍；flash-attn 也不在
  `_ecosystem_cases.py: CASES` 中。原生侧 flash-attn 在实测环境不可用。
- 未跑 `nox -s optional`、未跑上游 checkout 的 `tests/test_flash_attn.py`。

## 实测（2026-09-19）

本次把上一节的"未观测到 fused hit"补上了：在单卡 CUDA 上首次观测到
**fused hit**，`backend="flashattn_jittor_official:$JITTOR_LAB_ROOT/flash-attention"`（backend 字符串带的是源码 checkout 路径）。

### 1) shim 解析到哪个 flash_attn —— 是 stub

`import flash_attn` 在 shim 侧解析到 venv 内**部署 stub**：

```bash
SP="$(<jittor-python> -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
head -42 "$SP/flash_attn/__init__.py"
# -> __version__ = "2.7.4.post1"
# -> _jittor_flash_attn_stub = True
```

运行态确认（同一脚本同时打印）：`flash_attn.__file__` 是上述 `site-packages/flash_attn/__init__.py`，
`__version__=2.7.4.post1`，`_jittor_flash_attn_stub=True` → **确为 stub**，不是官方扩展本身。
但 stub 会把 `flash_attn_func` 等入口转发给 `jittor.compat.shim.backends.flash_attention`
加载的**官方扩展**（见下，`backend()` 返回 `flashattn_jittor_official:...`，
`is_flashattn_jittor_available()=True`，`last_error=None`）。

### 2) 原生侧真扩展不可导入

```bash
# 原生侧环境
<real-torch-python> - <<'PY'
import importlib.util
print(importlib.util.find_spec("flash_attn").origin)
try:
    import flash_attn; print("OK", flash_attn.__version__)
except Exception as e:
    print("FAIL", type(e).__name__); print(str(e)[:300])
PY
```

结果：origin 在 oracle base 前缀的 `site-packages/flash_attn/__init__.py`，
`import` **失败**：`ImportError: .../flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so:
undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib`（旧 ABI 对不上 oracle
的 `torch 2.13.0+cu129`）。结论同原文：**实测环境不存在可用的原生 flash-attn 基线**，因此没有
任何 shim-vs-原生 的速度/数值对拍，也不得声称 fused 加速比。

### 3) `F.scaled_dot_product_attention` 实际派发到哪

```bash
# 仓库根，shim 侧环境已激活
export JITTOR_HOME="$JITTOR_LAB_ROOT/_state/verify-diff/<run>/jittor-home"
export CUDA_VISIBLE_DEVICES=<gpu>
<jittor-python> - <<'PY'
import json, jittor as jt, torch
import torch.nn.functional as F
from jittor.compat import diagnostics
jt.flags.use_cuda = 1
R = {"hits": 0, "misses": {}, "casts": {}, "backend": None}
def reset(): diagnostics.set_sdpa_flash_stats(dict(R, misses={}, casts={}), jt)
# A 推理 fp16，(2,512,4,64)
reset(); q = jt.randn((2,512,4,64)).float16()
with torch.no_grad():
    F.scaled_dot_product_attention(q, q, q); jt.sync_all(True)
print("A", json.dumps(diagnostics.sdpa_flash_stats(jt)))
# B 训练态小形状 fp16
reset(); q2 = jt.randn((2,128,4,64)).float16()
F.scaled_dot_product_attention(q2, q2, q2); jt.sync_all(True)
print("B", json.dumps(diagnostics.sdpa_flash_stats(jt)))
# C 推理 fp32
reset(); q3 = jt.randn((2,512,4,64)).float32()
with torch.no_grad():
    F.scaled_dot_product_attention(q3, q3, q3); jt.sync_all(True)
print("C", json.dumps(diagnostics.sdpa_flash_stats(jt)))
PY
```

实测（计数器即 `compat/diagnostics.py` 的 `sdpa_flash_stats`，由
`compat/torch/installers/nn/attention.py:_try_flash_scaled_dot_product_attention` 记录）：

| 探针 | shape / dtype / 模式 | hits | misses | backend |
|---|---|---|---|---|
| A 推理 | (2,512,4,64) fp16, `no_grad` | **1** | `{}` | `flashattn_jittor_official:<flash-attention checkout>` |
| B 训练 | (2,128,4,64) fp16 | 0 | `{"short_training_math":1}` | null |
| C 推理 | (2,512,4,64) fp32 | 0 | `{"dtype":1}` | null |

- **compat 层 flash 路径被命中了**（A）：`JITTOR_FLASH_ATTN_JITTOR_SRC` 生效，官方扩展
  加载成功，`_try_flash_scaled_dot_product_attention` 返回 fused 结果。这直接推翻了原文
  "尚未观测到 fused hit"，也使坑 1（"import 成功 ≠ 扩展在跑"）有了正面对照：本例中
  扩展确实在跑，证据是 `backend` 字段不是 `None`。
- B 属设计守卫：训练态 `score_elements` 低于 `JITTOR_FLASH_ATTN_TRAINING_MIN_SCORES`
  （`1<<24`），按设计退回 composite。
- C 属设计守卫：`JITTOR_FLASH_ATTN_CAST_FLOAT32` 未设，fp32 直接 `dtype` miss，不被偷偷
  转 bf16/fp16。
- 首次运行触发了官方扩展**从源码重建**（digest 变了），耗时约 30 min（多个 nvcc 进程并发，
  同机另有其它 agent 编译），期间无输出，看起来像卡住；构建完成后探针秒级返回。这印证坑 5：
  扩展构建是串行且 digest 相关，不要中途重启。

### 四个轴 / 未运行

- flash-attention **没有注册的生态 case**（`_ecosystem_cases.py` / `_ecosystem_speed.py`
  里都没有），因此本次**没有** precision / peak-memory / speed 四轴表；`verify_repo.py
  --repo flash-attention` 找不到 case，属预期。
- 本次**未测量**任何 fused-kernel 加速比（原生基线不可用，见 2），不声称速度结论。
- **未运行**：`nox -s optional`、`test_flash_attn_compat.py`（部署 venv 的 `compat` 包
  pytest 收集问题，见原文）、上游 checkout 的 `tests/test_flash_attn.py`、任何两解释器对拍。
