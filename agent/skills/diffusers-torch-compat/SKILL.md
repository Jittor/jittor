---
name: diffusers-torch-compat
description: 把 diffusers 分别跑在 jittor 的 `import torch` shim 与原生 PyTorch 两个解释器上，并用同一套权重/输入对拍数值与墙钟。用于验证 diffusers 在 shim 上的数值兼容、定位断点、以及做一次可信的两解释器对拍时使用。
---

# diffusers ⇄ jittor torch shim

## 用途

回答四件事：diffusers 怎么在 shim 上跑、怎么在原生 torch 上跑、怎么对拍（数值 + 速度）、
出断点后往哪分流。真正的对拍契约与判据见
[`downstream-library-adaptation`](../downstream-library-adaptation/SKILL.md) 与
[`jittor-torch-diff`](../jittor-torch-diff/SKILL.md)，本 skill 只写 diffusers 的落地细节。

**不覆盖**：transformers / peft / mmcv 等其它库；vLLM、TRELLIS 服务化；长训练与大 benchmark
的启动；NPU 实机（本机没有）。不要求也不允许改 diffusers checkout 或 jittor 源码；本 skill
只做"怎么跑、怎么读结果"。

## 两侧环境

| | shim 侧 | 原生 torch 侧 |
|---|---|---|
| venv | `/root/jittor-lab/_state/h3/venv-jittor` | `/root/jittor-lab/_state/h3/venv-oracle-cu129` |
| 入口 | `source /root/jittor-lab/minimax-h3/env-jittor.sh` | `source /root/jittor-lab/minimax-h3/env-oracle-cu129.sh` |
| Python | 3.12.12 | 3.12.12 |
| torch | jittor shim，`torch.__version__ == "1.3.11.0"`，`hasattr(torch,'_torch_compat_install_context') is True`（实测） | `torch 2.13.0+cu129`，有 `torch._C`，`torch.cuda.is_available() is True`（实测） |

- **diffusers 没有 pip 安装在任何一侧**（实测 `"$VENV/bin/pip" show diffusers` → not found）。
  两侧都由 env 脚本的 `PYTHONPATH` 指向 lab checkout
  `/root/jittor-lab/diffusers-main/src`（HEAD `a71e62e0d226c284b86abf518791a5ffbba064bf`，
  remote `huggingface/diffusers`），实测版本 `0.41.0.dev0`，origin 两侧一致。
  用别的 checkout 时必须同时核对 remote 和 commit。
- 两侧同为 CPython 3.12，ABI 相同，可以共用一个 package site；harness 的
  `_reference_shares_this_abi()` 会自行判断。
- 离线：`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`（env 脚本已设，runner 也会 setdefault）。
- 缓存隔离：`JITTOR_HOME` / `TMPDIR` / `XDG_CACHE_HOME` 都在 `$JITTOR_LAB_ROOT/_state/h3/run`
  下；一个 device 一套 `JITTOR_HOME`。本机实测会与别的 JIT 进程抢构建锁，并发前先确认。
- **oracle 断言**（任何数字可信前必须过）：
  `assert not hasattr(torch, '_torch_compat_install_context')`。实测原生侧为 `False`（通过）；
  shim 侧为 `True`（若在 shim 上跑这条断言就会红，这正是它要抓的"拿 shim 当 oracle"）。
  `noxfile.py` 的 `ecosystem` session 在跑用例前就执行这条。

**PYTHONPATH 陷阱**：diffusers 只存在于 `PYTHONPATH`。读 `_ecosystem_harness._run()` 可知它
会给 oracle 子进程 `PYTHONPATH = ""`，因此走 harness 时 oracle 侧会因找不到 diffusers 而
skip；此时需显式设 `JITTOR_ECOSYSTEM_PACKAGE_SITE`（或
`JITTOR_ECOSYSTEM_REFERENCE_PACKAGE_SITE`）指向 `/root/jittor-lab/diffusers-main/src`
（这条是代码推导，本机未跑通 harness 路径；本机实际验证的是下面的 runner CLI，
它在 env 脚本的 `PYTHONPATH` 下能找到 diffusers）。

## 在 shim 上跑

先 `source /root/jittor-lab/minimax-h3/env-jittor.sh`，`cd` 到仓库根。单解释器 smoke 只证明
"能跑"，数值结论必须走下面的对拍。

```bash
# 单解释器 smoke（可直接 unittest 运行，不需要 pytest 的包收集）
# 本机直接跑会 2 error / 1 failure：它没关 CUDA，见"坑与假绿"第 2 条
PYTHONPATH="$PWD/tests:/root/jittor-lab/diffusers-main/src" \
  "$VENV/bin/python" compat/tests/torch/test_diffusers.py

# 生态 runner：与 oracle 共用同一权重/输入契约，对拍用它
"$VENV/bin/python" compat/tests/torch/_ecosystem_runner.py \
  diffusers_unet2d /tmp/jittor_diffusers.npz --runtime jittor --device cpu --repeats 10
```

- runner 是自足的：内部 `sys.path.insert` 会挂上 `tests/`；case 名取
  `compat/tests/torch/_ecosystem_cases.py` 的 `CASES`（diffusers 有 `diffusers_unet2d`、
  `diffusers_dit`），速度大用例在 `_ecosystem_speed.py` 的 `large_diffusers_unet2d`。
- `--device cpu` 会走 `jt.runtime.scope(use_cuda=0)`。**必须显式给**：Jittor 没有 per-tensor
  device，本机默认 `use_cuda=1`，不给就会在加速卡上跑。
- 实测（本机，CPU）：`diffusers_unet2d` 输出 146 个张量、loss `12.708709716796875`、
  `seconds=0.9557`、`device="cpu"`、`fallback_policy="error"`、`fallback_count=0`、
  `backend={has_acl:false,use_acl:false,use_cuda:false}`，diffusers `0.41.0.dev0`。
- 直接 `python -m pytest compat/tests/torch/test_ecosystem_parity.py` 在本 lab venv 里会
  在 setup 阶段 `ImportError: attempted relative import beyond top-level package`
  （`compat/__init__.py`）；`--import-mode=importlib` 又会丢 `_ecosystem_harness`。维护入口是
  `python -m nox -s ecosystem`（并设 `REAL_TORCH_PYTHON`），不要在 lab venv 里裸跑 pytest
  并把结果当门禁。

## 在原生 torch 上跑

```bash
source /root/jittor-lab/minimax-h3/env-oracle-cu129.sh
/root/jittor-lab/_state/h3/venv-oracle-cu129/bin/python \
  compat/tests/torch/_ecosystem_runner.py \
  diffusers_unet2d /tmp/oracle_diffusers.npz --runtime torch --device cpu --repeats 10
```

实测（本机，CPU）：146 个张量、loss `12.708718299865723`、`seconds=0.4688`、`device="cpu"`、
diffusers origin 与 shim 侧相同。`--device cuda` 可用（`torch.cuda.is_available()` 为真），
但本 skill 未在本机复验 CUDA 行。

## 对拍

**数值**：同一份权重 + 同一份输入，两个解释器各跑一次。`_ecosystem_runner.py` 的结构保证
了这一点——不带 `--weights` 的 oracle 跑会把权重存成 `<output>.weights.npz`，shim 侧用
`--weights` 载入完全相同的权重；种子固定（`--seed`），输入由同一个 `RandomState` 生成。
比较的是 forward 输出、**全部**参数梯度和输入梯度（本机 146 张量 = 1 输出 + 145 梯度）。

```bash
# 1) oracle 生成权重与参考值
source /root/jittor-lab/minimax-h3/env-oracle-cu129.sh
/root/jittor-lab/_state/h3/venv-oracle-cu129/bin/python \
  compat/tests/torch/_ecosystem_runner.py \
  diffusers_unet2d /tmp/oracle.npz --runtime torch --device cpu --repeats 10

# 2) shim 用同一权重重算
source /root/jittor-lab/minimax-h3/env-jittor.sh
/root/jittor-lab/_state/h3/venv-jittor/bin/python \
  compat/tests/torch/_ecosystem_runner.py \
  diffusers_unet2d /tmp/jittor.npz --weights /tmp/oracle.weights.npz \
  --runtime jittor --device cpu --repeats 10

# 3) 用 harness 的净量级归一化比较（_divergence / _comparison_floor 的等价实现）
python3 - <<'PY'
import numpy as np
ref, cand = np.load('/tmp/oracle.npz'), np.load('/tmp/jittor.npz')
assert not (set(ref.files) - set(cand.files)), "缺张量"
def div(a, b, floor):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(float(np.abs(b).max()), floor, 1e-6))
def floor(keys): return 1e-3 * max(float(np.abs(ref[k]).max()) for k in keys)
grads = [k for k in ref.files if k != '__output__']
print('forward', div(cand['__output__'], ref['__output__'], floor(['__output__'])))
print('worst_grad', max((div(cand[k], ref[k], floor(grads)), k) for k in grads))
PY
```

实测（本机，CPU，`diffusers_unet2d`）：forward 归一化偏差 `8.885e-07`，最差梯度
`4.555e-06`（`grad::mid_block.resnets.1.norm1.bias`），均在容差内。这是本机唯一被完整复验的
diffusers 两解释器数值结论。

**速度**：重复、交替采样，报最小值（干扰只会让单次变慢，最小值是对的统计量）；大尺寸用例
`JITTOR_ECOSYSTEM_REPEATS` 不低于 10。**墙钟永不在 PR 门禁里断言**：`JITTOR_ECOSYSTEM_SPEED_RATIO`
只在 nightly 生效（`noxfile.py:ECOSYSTEM_SPEED_RATIO = "1.07"`），平时只报告。本机单次 CPU
读数 oracle `0.4688 s` / shim `0.9557 s`（≈2.04x，tiny case 受 dispatch 主导），只作说明，
不得当作性能结论。

**Device 规则**：两侧必须同 device，harness 会把实际 device 读回来断言。Jittor 无 per-tensor
device、CUDA 默认开，所以 "CPU" 必须显式请求（runner 的 `--device cpu`）。

## 断点与分流

diffusers 走的是纯 `import torch` 消费者路线，**没有 adapter**。历史/当前断点按三行判据：

| 东西 | 分流 | 依据 |
|---|---|---|
| UNet/DiT/VAE/DDIM 前向+反向 | `jittor.compat.torch`（零适配代码，注册成生态 case 即可） | 实测 `_ecosystem_cases.py` 的 `diffusers_unet2d`/`diffusers_dit` |
| `torch.linalg.inv_ex`（kornia import 期需要，返回含 `.info` 的结构） | jittor 核心 | `compat/tests/torch/test_torch_compat_diffusers_video.py` 的注释：加速卡上它是需要 CuPy 的 numpy-code op |
| 旧报告里的配置边界：`attention_head_dim=8` + 4 通道触发 `invariant_uniform` 的 `ZeroDivisionError`；默认 `norm_num_groups=32` 触发 GroupNorm 整除断言 | jittor 核心（初始化/GroupNorm 语义） | `refactor-wip/results/2026-09-09-ecosystem-vllm-diffusers.md` |
| GroupNorm / SiLU / 最近邻 upsample / 训练 SDPA 的 ACL 原生路径 | jittor 核心 ACL 后端 + compat 路由 | `refactor-wip/results/2026-08-30-diffusers-ascend-parity-performance.md` |
| torchvision / torchaudio / torchdata / flash_attn | 部署 stub（`jittor.compat.shim` 资源），不是 adapter | `compat/shim/resources/stubs/` |

已接受的结论引用（非本机复验）：`refactor-wip/results/2026-08-23-ecosystem-parity-performance.md`
（12 个生态用例 CPU/CUDA 全过，真实规模 Diffusers UNet 在双方 cuDNN autotune 下中位数比约
`0.93x`）；`refactor-wip/results/2026-08-30-diffusers-ascend-parity-performance.md`（910B3 上
forward + 145 梯度通过、零 fallback，`0.964x`）；`agent/manuals/project-context.md` 的中文结论
"Diffusers correctness and maintained float32 performance are accepted at `0.964x` native
`torch_npu`"。这些各自绑定当时的 device、版本和 baseline commit，不能外推。

## 坑与假绿

1. **单解释器 smoke 不是对拍。** `test_diffusers.py` 只断言"能跑、梯度非 None、eval 两次一致"，
   不跟 PyTorch 比。
2. **不显式关 CUDA，"CPU" 就不是 CPU。** 本机默认 `use_cuda=1`；实测直接跑 `test_diffusers.py`
   （它设 `torch.tensor(...)` 不带 device）会在加速卡默认下混到后端不匹配，出现
   `dispatch_context ... Expected all tensor inputs on the same backend and device`，以及
   eval 两次不一致的失败。必须显式 `use_cuda=0`（runner 的 `--device cpu` 会做）。
3. **diffusers 只在 PYTHONPATH 里。** 忘了给 `PYTHONPATH`，或只依赖 env 脚本而不设
   `JITTOR_ECOSYSTEM_PACKAGE_SITE`，oracle 会静默 skip 掉——看起来像通过。
4. **shim 的 `torch` 没有 `__file__`。** 直接 `torch.__file__` 抛
   `AttributeError: __file__. Did you mean: '__all__'?`；用 `getattr(..., default)` 或
   `torch.__name__`/`torch.__version__` 判断。harness 已用 `getattr(torch, "__file__", "")` 兜住。
5. **copy-deploy**：改仓库 `python/jittor/` 不会影响已部署的解释器，必须先拷到
   `/root/jittor-lab/_state/h3/venv-jittor/lib/python3.12/site-packages/jittor/` 才对运行生效。
6. **Jittor decode/懒求值与 eval 非确定性**：`eval()` 在 Jittor 里还会停掉参数梯度（runner
   会 `start_grad()` 恢复），且未 sync 的两次前向不一定逐位一致；逐位校验不能当门禁。

## 证据

**本机实测（本次会话）**

- 环境：两个 venv 的 Python/torch 版本、oracle 断言、diffusers 两个 venv 都未 pip 安装且由
  `PYTHONPATH=/root/jittor-lab/diffusers-main/src` 提供（版本 `0.41.0.dev0`）。
- `diffusers_unet2d` CPU 两解释器对拍：146 张量 / 145 梯度；forward 归一化偏差 `8.885e-07`、
  最差梯度 `4.555e-06`；shim `fallback_count=0`、`fallback_policy="error"`、`use_cuda=false`；
  单次墙钟 oracle `0.4688 s` / shim `0.9557 s`。
- 直接 pytest 在本 lab venv 会因 `compat` 包导入方式失败（见上），故本 skill 用 runner CLI。
- 直接跑 `test_diffusers.py`（未关 CUDA）失败 2 error / 1 failure，作为"必须显式选 device"的
  证据保留。

**仅引用、未在本机复验**：CUDA/NPU 数值与性能、`large_diffusers_unet2d` 速度门禁、
`refactor-wip/results/` 里三份报告的结论（各自环境见报告）。本机无 Ascend 卡，NPU 一列是引用，
不是本机结论。

## 实测（2026-09-19）

本机在 CUDA（`CUDA_VISIBLE_DEVICES=3`）上把 `verify_repo.py` 完整跑了一遍，三个已注册 case
全部 `status="ran"`。oracle 侧先自证可用：`import torch` 为 `2.13.0+cu129`、
`torch.cuda.is_available() is True`、`import diffusers` 成功且
`__file__=/root/jittor-lab/diffusers-main/src/diffusers/__init__.py`（`0.41.0.dev0`），
与 skill 侧同一份 `PYTHONPATH` checkout，故 diffusers 数字可信。runner 侧
`_import_torch("torch")` 会 `pop JITTOR_TORCH_SHIM` 并断言 `hasattr(torch,"_C")`，oracle
不是 shim。

命令（仓库根，独立 JIT home，避免与默认缓存抢构建锁）：

```bash
source /root/jittor-lab/minimax-h3/env-jittor.sh
export JITTOR_HOME=/root/jittor-lab/_state/verify-diff/jittor-home
mkdir -p "$JITTOR_HOME"
export CUDA_VISIBLE_DEVICES=3
REAL_TORCH_PYTHON=/root/jittor-lab/_state/h3/venv-oracle-cu129/bin/python \
"$VENV/bin/python" agent/skills/torch-compat-repo-runbook/scripts/verify_repo.py \
  --repo diffusers --device cuda --repeats 5 \
  --out /root/jittor-lab/_state/verify-diff/out
```

**支持的 case**：`--list-only` 列出 `diffusers_dit`、`diffusers_unet2d`、
`large_diffusers_unet2d`（都在 `_ecosystem_cases.py` / `_ecosystem_speed.py` 的 `CASES`，
`requires=diffusers`）。三个全部运行成功，无缺张量、无 fallback、两侧 device 一致。

| case | 张量 | worst abs | worst rel (vs 全字段) | torch s | jittor s | 速度比 | torch peak | jittor peak |
|---|---|---|---|---|---|---|---|---|
| diffusers_dit | 46 | 1.22e-3 (`grad::transformer_blocks.0.norm1.linear.weight`) | 4.93e-5 | 0.00577 | 0.01667 | 2.89x | 67.9 MB | 3.0 MB |
| diffusers_unet2d | 146 | 1.61e-2 (`grad::up_blocks.1.resnets.1.conv1.weight`) | 3.06e-4 | 0.01258 | 0.03037 | 2.41x | 71.0 MB | 12.6 MB |
| large_diffusers_unet2d | 272 | 1.94e-1 (`grad::conv_out.weight`) | 6.77e-4 | 0.07051 | 0.09177 | 1.30x | 1.10 GiB | 3.78 GiB |

- 精度：三个 case 的 `worst_rel_err_vs_field` 均 ≤ 6.8e-4，`jittor_fallback_count=0`、
  `device_agreement=true`。worst abs 落在梯度上；绝对值大但相对全字段量级仍小（这正是
  `verify_repo.py` 用全字段归一化的原因）。数字来自 shim 与真实 PyTorch 的同权重/同输入
  对拍（`--weights` 复用 oracle 落盘的 `.weights.npz`）。
- 速度：`--repeats 5`，报最小值。tiny case（dit）受 dispatch 主导，比值 2.89x 只作说明；
  `large_diffusers_unet2d` 是唯一非 dispatch 主导的读数，1.30x。墙钟不在 PR 门禁里断言。
- **显存列是旧口径（2026-09-19 复核）**：上表 `torch peak` 是
  `torch.cuda.max_memory_allocated`（**活跃**字节，单卡），`jittor peak` 当时取的是
  `jt.get_mem_info().total_cuda_used`（**活跃+缓存空闲**，且对所有设备求和）——reserved 对
  allocated，不是同一量纲。工具已改为两侧都报 live 与 pool 两个数（jittor 侧用
  `device_memory_used` / `device_memory_reserved`），**上表两列相除的倍数不要继续引用**。
  在 transformers 的同口径复核里，换成 like-for-like 只把倍数挪动约 10%、**没有**消掉差距
  （live 2.88x、pool 2.76x），所以「口径不同」不足以解释 `large_diffusers_unet2d` 的
  1.10 GiB -> 3.78 GiB；但 diffusers 这三个 case 尚未按新口径重测，在重测之前两侧都
  不下结论（既不说 jittor 占更多，也不说 parity）。
- 本次全程耗时 ~23.5 min，主要花在首次 JIT 内核编译与官方 flash 扩展构建（fresh
  `JITTOR_HOME`），非 case 本身。

**没有运行/未复验**：CPU 行（本次只跑 CUDA）、NPU/ACL（本机无卡）、`nox -s ecosystem`
门禁、`compat/tests/torch/*.py` 的 pytest（lab venv 无法导入 `compat` 包，见上，非本次范围）。
