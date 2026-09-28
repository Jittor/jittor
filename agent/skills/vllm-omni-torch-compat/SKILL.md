---
name: vllm-omni-torch-compat
description: 在 Jittor torch shim 与原生 PyTorch 上运行/验证 vLLM-Omni 的 MiniMax-H3 推理（离线 Omni engine、vllm-omni serve 与 diffusers parity 三条路径）的 runbook，含真实 lab 脚本、copy-deploy/JIT 缓存/残留进程三个环境陷阱、重复交错速度协议与断点分流。用于当前 H3/vLLM-Omni 联调、复跑 parity、或排查 H3 在 shim 上的失败。
---

# vLLM-Omni（MiniMax-H3）的 shim ⇄ 原生 torch 对拍

## 用途

回答：vLLM-Omni 的 MiniMax-H3 `fl2va` 在 shim 上怎么跑、原生 torch 上怎么跑、怎么对拍
数值与速度，以及每个断点归 jittor 核心 / `jittor.compat.torch` / adapter / shim 部署
哪一层。这是团队当前焦点（vLLM-Omni checkout 为本报告锁定 HEAD）。

**不覆盖**：不宣称本机跑过（本 skill 只读取脚本与报告，未启动任何 server/长跑）；TP2 的
画面噪声与「非注意力 device 侧 gap」仍是开放项，不要据此接受性能；不覆盖 vLLM-Omni 的
训练（它是 inference runtime）。

## 两侧环境

| 角色 | 解释器 / 入口 | 说明 |
| --- | --- | --- |
| shim | `/root/jittor-lab/_state/h3/venv-jittor/bin/python`，`source /root/jittor-lab/minimax-h3/env-jittor.sh` | `JITTOR_TORCH_SHIM=1`；`JITTOR_HOME=/root/jittor-lab/_state/h3/run/jittor-home` |
| oracle | `/root/jittor-lab/_state/h3/venv-oracle-cu129/bin/python`，`source /root/jittor-lab/minimax-h3/env-oracle-cu129.sh` | 独立 PyTorch，cu129 |

本机实测（2026-09-19）：`vllm 0.29.0` 两侧相同；`transformers 5.5.3` 两侧相同；
`torch 2.11.0`（shim）对 `2.13.0+cu129`（oracle）。
`.../venv-oracle`（`transformers 5.17.0`、`torch 2.13.0`）不是可用 pair——报告指出它会
破坏 lab 的 diffusers，必须用 `venv-oracle-cu129`。

上游 checkout：`/root/jittor-lab/vllm-omni`，remote
`https://github.com/vllm-project/vllm-omni.git`，HEAD 本机实测
`446c2b5474dabffccf9a0034836eeab77c62b4b8`。

shim 侧 `env-jittor.sh` 已经：把 flash-attn 经
`JITTOR_FLASH_ATTN_JITTOR_SRC=/root/jittor-lab/flash-attention` 命名为
`HEAD_DIMS=64,128`、`DTYPES=bf16,fp16`（故意**不设** `..._REQUIRED` 与
`..._CAST_FLOAT32`，未覆盖形态落回组合路径）；把 diffusers main 放进 `PYTHONPATH`；设
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`；把 `/usr/local/openmpi/bin` 提前（避开坏 mpicc）。

**oracle 断言（看任何数字之前必须过）**：

```bash
source /root/jittor-lab/minimax-h3/env-oracle-cu129.sh
"$VENV/bin/python" -c \
  "import torch; assert not hasattr(torch, '_torch_compat_install_context'); print(torch.__version__)"
```

本机实测输出 `2.13.0+cu129`。shim 侧该属性为 `True`（实测）；shim 侧必须先 `import jittor`
再 `import torch`，shim 的 `torch.__version__` 是后端版本 `1.3.11.0`、`torch.__torch_version__`
才是模拟 API level `2.11.0`（均实测）。双解释器 harness 契约见
[`_ecosystem_harness.py`](../../../compat/tests/torch/_ecosystem_harness.py) 的 docstring，
本 lab 不走 `_ecosystem_runner.py`。

## 在 shim 上跑

所有脚本都 `cd /root/jittor-lab/minimax-h3` 后 source `env-jittor.sh`，并设
`use_cuda=1`、`JITTOR_TORCH_SKIP_EXT_BUILD=1`（只 import vllm-omni，不建 CUDA 扩展）、
`VLLM_ENABLE_V1_MULTIPROCESSING=0`、`PYTHONPATH=/root/jittor-lab/vllm-omni`。

| 目的 | 命令 |
| --- | --- |
| 只构造 engine（`vllmomni_try.py`） | `./run-vllmomni.sh` |
| 端到端 generate（`vllmomni_generate.py`） | `./run-vllmomni-gen.sh` |
| probe / stats / 关 guard | `./run-vllmomni-probe.sh`、`./run-vllmomni-stats.sh`、`./run-vllmomni-noguard.sh`、`./run-vllmomni-noguard-stats.sh` |
| diffusers parity（单请求） | `./run-jittor-parity.sh`、`./run-jittor-parity-fp32vae.sh` |

起 server（默认单卡 layerwise offload `dit`+`text_encoder`，`ATTN=TORCH_SDPA`）：

```bash
cd /root/jittor-lab/minimax-h3
./serve-vllmomni.sh                                  # 单卡，lab 验证过的 profile
TP=2 NUM_GPUS=2 GPU=0,1 ./serve-vllmomni.sh          # TP2
```

多 rank 时脚本自己设置 `JITTOR_TORCH_DISTRIBUTED_AUTO_INIT=1`、
`JT_BUILD_NCCL_{INCLUDE,LIB}_PATH`、`JITTOR_TORCH_KEEP_TMPDIR=1`，并清
`/tmp/jittor-nccl-*.bin*`；`ATTN=FLASH_ATTN` 时另设
`PYTHONPATH=/root/jittor-lab/minimax-h3/shim-extra:...`（顶层 `flash_attn_interface`
别名，见坑 8）。停服务必须用 `./stop-vllmomni.sh <PORT>`。

## 在原生 torch 上跑

diffusers 路径；与 shim 注入同一份 latents，保证两侧同输入：

```bash
cd /root/jittor-lab/minimax-h3
./run-oracle-cu129.sh            # infer_h3.py，512x512x124 6 步
```

脚本里 `--dump-tensors` 与 `--latents-file` 指向
`/root/jittor-lab/_state/h3/runs/tensors/` 的固定张量。`--attn-backend` 默认 `flash`。
vLLM-Omni server 的原生对拍属于 vllm-omni 自身栈，本机未做。

## 对拍

**数值**

- tiny checkpoint（`hf-internal-testing/tiny-minimax-h3-modular-pipe`，t2va，124 帧，2 步，
  注入 noise）：视频帧 max abs diff `1`（uint8）/ mean `0.046`；soundtrack 未对齐
  （corr≈0.05、RMS 比 0.53）——音频 VAE 路径结构性差异，**不要拿视频结论覆盖音频**。
- VAE decode 用 `/root/jittor-lab/minimax-h3/compare_vae_outputs.py` 做容差比较
  （`max|d|`、`mean|d|`、`rel`），按运行间噪声 floor 判；decode 不是 bit-reproducible，
  单次 checksum 不能当门禁。

**速度**

- 用 `run-repeat-interleaved.sh [rounds] [samples]`（或 `run-lt-ab-interleaved.sh [rounds]`）：
  逐轮交替跑 shim/torch，取每轮样本的**最小值**（`probe_decode_base_repeat.py` 打印
  `[rep] n=...`）。同一 build 的背靠背 pair 在这里读到过 6.68–22.59 s，不可用。
- 端到端可比量是 `generate_seconds`，**不用** `load_seconds`（shim 预加载 ~385–392 s，
  oracle 懒加载 ~2.13 s，语义不同）。
- 区间数字（报告）：512x512x124 6 步 warm `generate_seconds` shim `92.34 s` 对 torch
  `75.94 s`（1.22x）；cold `253.69 s` 是 JIT 编译，不能当速度。分相：dit 1.11x、
  text_encoder 0.37x（更快）、vae.video 2.31x、vae.audio 20.1x。vllm-omni server 侧
  8 步 832x480：TP1 `309.2 s` 对 TP2 `180.2 s`。
- 墙钟上界只在 nightly 断言，PR 不 assert；通用门禁见
  [`noxfile.py`](../../../noxfile.py) 的 `ecosystem` session（其 `JITTOR_ECOSYSTEM_SPEED_RATIO`
  不含 H3）。

**Device 规则**：两侧同 device；Jittor 无 per-tensor device 且 CUDA 默认开，"CPU" 要显式
请求。`--vae-dtype float16` 曾是绕开 autocast 缺陷的 workaround，autocast 修好后不再是必需。

## 断点与分流

主报告按「一缺陷一节」组织，下面的层归属来自各节的根因与验证：

| 断点 | 层 |
| --- | --- |
| `TCPStore` 多地址 `localhost` rendezvous | jittor core（分布式） |
| `with torch.device("cpu"):` 不移动分配默认 | `jittor.compat.torch` |
| op-type 表按运行时 flag 而非编译目标选 | jittor core |
| `nn.Conv3d` 拒绝 `padding_mode` | `jittor.compat.torch` |
| `torch.get_device_module` 缺失 | `jittor.compat.torch` |
| `tensor.data = other` 改元素而非替换 storage | `jittor.compat.torch` |
| size-changing rebind 在懒记录 view 上 abort | jittor core |
| `as_strided` / `empty_strided` 缺失 | `jittor.compat.torch` |
| 跨 device `copy_` 把目标搬到源 device | jittor core |
| core helper 在 ambient placement 上分配 | jittor core |
| 混合 fp32/fp16 积落回 outer product | jittor core |
| `linspace` 落不到 `end`（≥4 步全挂） | jittor core |
| 非 256x256 丢场景：device Var 停在 host | jittor core |
| `torchaudio.functional.melscale_fbanks` 子模块解析 | `jittor.compat.torch`（stub 缺 `__path__`） |
| autocast 下 conv/cuBLAS matmul/`linear` 只改结果 dtype 不 cast 操作数 | jittor core（amp register） |
| `flash_attn_interface` 顶层名解析到未桥接的 libtorch 扩展 | shim 部署 / 环境 alias（`shim-extra`），非模型 |
| TP2 batch node 被 planner 释放（`fused_op.cc:89`） | jittor core |
| Generator 没有 stream → TP2 噪声 | `jittor.compat.torch`（已修） |
| 部署 core 缺整套 stream 一致性 | 部署（stale `.so`，见坑 1、2） |

MiniMax-H3 按报告是**消费未修改**：断点尽量落 jittor core / compat，不要为它写模型侧
补丁。vLLM 侧唯一合法的 adapter 是仓内
[`adapters/jittor_adapters/vllm/`](../../../adapters/jittor_adapters/vllm/)（见
[`vllm-torch-compat`](../vllm-torch-compat/SKILL.md)）。

## 坑与假绿

1. **copy-deploy trap**：改 `python/jittor/` 对运行**没有影响**，必须先复制到
   `/root/jittor-lab/_state/h3/venv-jittor/lib/python3.12/site-packages/jittor/`（本机确认
   该路径存在）。deployed tree 是自洽快照，逐文件部署再重跑。
2. **JIT 缓存**：`$JITTOR_HOME` = `/root/jittor-lab/_state/h3/run/jittor-home`（本机存在）。
   冷 cache 首请求 332.2 s、首次 512 请求 300.9 s；一个源码改动让受影响算子 kernel 全部
   失效。速度只取 warm；冷数字不是吞吐。
3. **残留进程**：只按 `--port` 字符串 kill 会留下 DiffusionWorker 子进程占住 MASTER_PORT，
   下一次 rank1 与孤儿 store rendezvous，报 "NCCL store rendezvous timeout"。用
   `/root/jittor-lab/minimax-h3/stop-vllmomni.sh` 走进程树。
4. **NCCL rendezvous 文件**：`/tmp/jittor-nccl-*.bin*` 按 `MASTER_ADDR-MASTER_PORT` 命名，
   同端口重跑会继承上一轮 unique id 并挂起；serve/stop 都会清。
5. **TMPDIR 太长**：shim 把 TMPDIR 换成 `<runtime>/tmp`，ipc socket 超过 `sun_path`(107)；
   多 rank 需 `JITTOR_TORCH_KEEP_TMPDIR=1`。
6. **假绿检查**：报告靠 `grep -c best_algo_idx`（应 0）、`video decode failed`（0）、
   `fallback_count=0`、`cpu_compile_count=0`、`torch_npu`/`vllm_ascend` 未加载、四 token
   一致来判成功；只看「写出了 mp4」会漏掉占位符 VAE。
7. **"跑完"不等于正确**：TP2 曾完成但画面是彩色噪声，根因是 shim 的 Generator 没有 stream；
   单卡正常不能推断 TP2 正确。
8. **attention 后端不要混读**：`TORCH_SDPA` 是 shim 实际跑通的请求后端；`FLASH_ATTN` 必须
   有 `shim-extra` 顶层别名，且报告里它使 TP2 请求无法完成。
9. `probe_decode_base_both.py` 的单次输出（6.68–22.59 s）是噪声，不是回归。
10. `load_seconds` 两侧语义不同，不可比。

## 证据

- 主报告：[`docs/results/2026-09-14-vllm-omni-h3-enablement.md`](../../../docs/results/2026-09-14-vllm-omni-h3-enablement.md)（34+ 节）。
- [`2026-09-12-minimax-h3-torch-compat.md`](../../../docs/results/2026-09-12-minimax-h3-torch-compat.md)、
  [`2026-09-14-autocast-conv-mixed-dtype.md`](../../../docs/results/2026-09-14-autocast-conv-mixed-dtype.md)。
- lab：`/root/jittor-lab/minimax-h3/`（run/env 脚本 + 176 个 `probe_*.py`）；原始日志、权重、
  FlashAttention build、JIT cache 未版本化于 `/root/jittor-lab/_state/h3/`。
- **本机已核实**：两个 env 脚本内容、两侧 venv 版本、vllm-omni HEAD、oracle 断言、shim 身份、
  deployed site 与 `$JITTOR_HOME` 路径存在、脚本清单与参数。
- **仅报告 / 未在本机复验**：所有 serving/TP2、数值与速度数字；本机未启动任何 server 或长跑。

## 实测（2026-09-19）

环境：仓库 HEAD `90fe0b9`；GPU 5；oracle 断言 `2.13.0+cu129`；两侧 `vllm 0.29.0`、
`transformers 5.5.3`。

**已核实的身份/版本**：`git -C /root/jittor-lab/vllm-omni rev-parse HEAD` =
`446c2b5474dabffccf9a0034836eeab77c62b4b8`，与 skill 一致。但该 checkout **不是干净消费**：
`git status` 显示 `M vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py`——
skill 正文「MiniMax-H3 按报告是消费未修改」在**本 lab checkout 不成立**。

**有界的唯一实跑**（不建 CUDA 扩展、不建 engine、不启动 server；`source env-jittor.sh`，
`JITTOR_HOME=verify-misc/jittor-home`，`CUDA_VISIBLE_DEVICES=5`）：

```
"$VENV/bin/python" probe_vae_bare.py \
  --model /root/jittor-lab/_state/h3/models/tiny-h3 \
  --latents /root/jittor-lab/_state/verify-misc/tiny_latent17.npy \
  --dtype float32 --device cuda --warmup 0 --trials 1
```

输出：

```
[bare] dtype=float32 force_cast= decoder=torch.float32 latent=(1, 4, 17, 8, 8)
[bare] trial 0 167.518s out (1, 3, 56, 128, 128)
```

即 shim 上 H3 video VAE（diffusers `AutoencoderKLMiniMaxH3`）decode 在 float32 下能完成，
latent `(1,4,17,8,8)` → 输出 `(1,3,56,128,128)`。第一次用 T=1 的 latent 时
`_decode` 的 chunk 列表为空，`torch.cat(decoded_chunks)` 报 `IndexError: list index out of
range`（chunking 需要至少 `tokens_chunk_size` 帧），T=17 后成功。`167.518s` 含首次 JIT 编译，
**不是速度数字**。

**四轴**：支持的 case——本 skill 不走 `_ecosystem_runner`，只做了上面一条 VAE decode；
精度/显存——未测（未对 oracle 同输入比较）；速度——未测（唯一时间含冷编译）。

**证据状态**：整体仍**报告派生**。仅「vllm-omni HEAD、两侧版本、oracle 身份」与
「shim 能完成一次小型 H3 VAE decode（dtype/shape）」是机器验证；报告中全部 serving/TP2、
数值、速度结论本机未复验。本机未启动 server 或长跑。
