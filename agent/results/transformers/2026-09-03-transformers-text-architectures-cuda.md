# Transformers 4.56.2 三类文本架构 CUDA 兼容验证

- Status: tiny FP32 encoder-only, decoder-only, and encoder-decoder workflows accepted
- Last reviewed: 2026-09-03
- Source baseline: `e27b6b80557b836c1ed8d5c70fb57fd8e1a113a5` plus this report's Torch-compat follow-up
- Owner: Torch compatibility and Transformers maintainers
- Review when: Transformers generation/cache APIs, decoder-start handling, in-place
  arithmetic, serialization, CUDA, or the model implementations below change

## 结论

Transformers 4.56.2 的三类文本架构已在真实 NVIDIA A800 CUDA 上完成独立
PyTorch 2.6.0+cu124 对拍。Encoder-only 以 BERT、RoBERTa、DeBERTa-v2 和 MPNet
覆盖绝对/相对位置、mask 与分类类任务头；Decoder-only 以 GPT-2 和 Llama 覆盖传统
learned position、RoPE/GQA、causal LM、DynamicCache 和生成；Encoder-decoder 以 T5
覆盖 labels shift、cross-attention、EncoderDecoderCache 和 beam generation。

所有用例都使用两端相同权重、固定输入和独立进程。BERT、RoBERTa、GPT-2、Llama 与
T5 主锚点已验收前向结构和数值、真实任务 loss、全部命名参数梯度以及 state_dict/
safetensors/`save_pretrained` round-trip；DeBERTa-v2/MPNet 补充前向、loss 和全部梯度
覆盖。GPT-2、Llama 和 T5 还覆盖 cache prefill/decode、cached-vs-full logits、greedy
和 beam generation。最终结果没有 CPU fallback，也没有修改 Transformers 源码。

本轮发现并修复两个共享 Torch-compat 缺陷：Jittor 的物理 rank-1 scalar 使
Transformers 误判 T5 decoder-start token；原生 `Var.__imul__` 在 CUDA 上以 float32
极大负数乘动态图 bool 条件时会产生 NaN，污染 cached causal mask。前者在
Transformers bridge 内保留配置标量的意图，后者只旁路同 shape、浮点、非显式 Torch
leaf 的 bool mask 乘法；高层回归锁定真实 CUDA batch-2 generation 和 cache 增长。

## 环境与方法

| 项目 | 配置 |
| --- | --- |
| Jittor baseline | `e27b6b80557b836c1ed8d5c70fb57fd8e1a113a5` plus working change |
| Transformers | `4.56.2` |
| PyTorch oracle | `2.6.0+cu124`, real site-packages |
| Device | NVIDIA A800 80GB PCIe, compute capability 8.0 |
| Jittor CUDA | CUDA 12.2.140, cuDNN 8, `sm_80` |
| Dtype | FP32 parameters/activations, int64 token ids |
| JIT isolation | separate `HOME`, `JITTOR_HOME`, `TMPDIR`, runtime root, and cache name |

前向数组使用

$$
e_{\mathrm{rel},\infty} =
\frac{\max_i |x_i-y_i|}{\max_i |y_i|+10^{-8}}
$$

比较，门槛为 $10^{-4}$。梯度先检查参数键全集、shape、finite 和缺失项，再以全网梯度
尺度归一，门槛为 $10^{-5}$。token、labels shift、state/safetensors keys 和 cache
signature 使用精确比较。比较器遇到任一非有限数组直接返回失败。

## Encoder-only

| 模型 | 覆盖 | 最大前向误差 | 最大梯度误差 | 结果 |
| --- | --- | ---: | ---: | --- |
| BERT | base manual fallback/SDPA、MLM、sequence/token classification、QA、双向 round-trip、AdamW | $2.581\times10^{-7}$ | $3.106\times10^{-7}$ | PASS |
| RoBERTa | base manual fallback/SDPA、MLM、sequence/token classification、QA、双向 round-trip、AdamW | $2.732\times10^{-7}$ | $2.671\times10^{-7}$ | PASS |
| DeBERTa-v2 | eager relative attention、sequence classification、task loss、50/50 gradients | $1.804\times10^{-7}$ | $4.450\times10^{-8}$ | PASS |
| MPNet | eager relative attention、sequence classification、task loss、41/41 gradients | $2.061\times10^{-7}$ | $1.342\times10^{-7}$ | PASS |

BERT MaskedLM 的 42/42 独立参数均有梯度，state dict 保留 44 个公开键及 tied aliases；
最大 PyTorch-to-Jittor round-trip 和单步 AdamW 误差分别为
$2.361\times10^{-7}$、$2.345\times10^{-7}$。RoBERTa 对应 round-trip 和 AdamW
最大误差为 $2.549\times10^{-7}$、$3.679\times10^{-6}$。

DeBERTa-v2 和 MPNet 强制 SDPA 时，真实 PyTorch 在 `AutoModel.from_config` 给出与
Jittor 相同的“不支持 scaled dot product attention”错误；因此最终按 Transformers
4.56.2 的能力声明使用 eager attention，这不是后端 fallback 或 Jittor 缺陷。

## Decoder-only

| 模型 | Loss 相对误差 | 最大前向误差 | 梯度覆盖/最大误差 | Cache 最大误差 | 结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| GPT-2 | $9.848\times10^{-8}$ | $2.696\times10^{-7}$ | 28/28, $1.640\times10^{-7}$ | $2.464\times10^{-7}$ | PASS |
| Llama | $9.829\times10^{-8}$ | $2.649\times10^{-7}$ | 21/21, $2.379\times10^{-7}$ | $3.622\times10^{-7}$ | PASS |

GPT-2 的两层 cache 从 `[2,4,4,16]` 增长到 `[2,4,5,16]`；Llama 的两层 GQA
cache 从 `[2,2,4,16]` 增长到 `[2,2,5,16]`。两端 cached-vs-full 最大误差分别为
$1.995\times10^{-7}$ 和 $1.613\times10^{-7}$。两种模型的 greedy/3-beam token
逐元素一致，PyTorch `save_pretrained` 后由 Jittor 加载的 forward/cache/generation
均通过。

GPT-2 `state_dict` 有 29 个公开键；`lm_head.weight` 与
`transformer.wte.weight` tied，safetensors 正确去重为 28 个独立键。Llama 的 21 个
state/safetensors tensors 两端逐元素一致。

## Encoder-decoder

最终 T5 使用 batch 2、encoder 长度 6、decoder teacher-forcing 长度 5、一层四头
配置。Jittor/PyTorch loss 均为 `4.871007919311523`，labels 自动 shift 精确等于
`[[0,20,21,1,0],[0,22,23,24,25]]`。最大前向误差为
$4.513\times10^{-7}$，26/26 参数均有梯度，最大梯度误差为
$4.466\times10^{-7}$。

cache 在 step 前深快照，避免 `EncoderDecoderCache` 原地增长覆盖 prefill 证据：

| 阶段 | Self K/V | Cross K/V |
| --- | --- | --- |
| Prefill | `[2,4,3,8]` | `[2,4,6,8]` |
| One-token decode | `[2,4,4,8]` | `[2,4,6,8]` |

跨框架 cache 最大误差为 $2.335\times10^{-7}$；cached-vs-full 在 Jittor/PyTorch
均为 $8.787\times10^{-8}$。greedy 与 3-beam 的四步 scores 全部有限，最大误差为
$2.702\times10^{-7}$，序列逐元素一致。

T5 有 26 个独立命名参数和 29 个 state keys。`encoder.embed_tokens.weight`、
`decoder.embed_tokens.weight`、`lm_head.weight` 均与 `shared.weight` tied；两端
safetensors 正确去重为 26 个键。strict state clone 误差为零，PyTorch
`save_pretrained` 到 Jittor 的 missing/unexpected/mismatched keys 均为空，forward、
cache 和 generation round-trip 全部通过。

## 修复

### Decoder-start 标量意图

Jittor 不能物理存储 0-D Var，因此 `torch.tensor(0)` 仍是 shape `[1]`。Transformers
将 `decoder_start_token_id=0` 转成 tensor 后按 `ndim` 判断它是 batch vector，batch
size 2 因长度只有 1 而报错。直接把所有一元素 Var 伪装为逻辑 0-D 会让 `shape`、
索引、算术、归约和 `stack/cat` 语义互相矛盾，因此本轮没有改变通用 Tensor 元数据。

Transformers 可选安装器改为在 `_prepare_special_tokens` 中记录 Python/NumPy 配置
标量的来源，并仅在 encoder-decoder generation 入口按 batch size 展开该 token。
显式 list/batch vector 仍走上游原逻辑，Jittor core 的 0-D 表示差异继续登记为已知
问题。这一局部适配解除 batch generation 阻断，也不对公共 scalar 语义作过度承诺。

### 动态布尔 mask 的原地乘法

最小复现为 float32 `finfo.min` tensor 原地乘以 `arange > cache_position` 产生的全
False bool 图。原生 C++ `Var.__imul__` 在 CPU 得到有限零值，但真实 CUDA 得到 NaN；
相同的 `self * other` 后 `assign` 在两端均有限。T5 单 token decoder 恰好走该因果
mask 分支，NaN 随后传播到 cached logits 和 generation scores。

Torch 兼容层只在 RHS 为 bool Var、两端 shape 完全一致、lhs 为浮点且不是显式注册的
Torch leaf 时使用已有 `_ip(self, self * other)`；其他 `__imul__` 调用仍委托原生
slot。修复不插入 host 同步，并保持该 Tensor 的 Python identity；它不声明 Jittor
物化 view 具有 PyTorch storage alias 语义。修复后低层 CPU/CUDA 最小例、T5 cache 与
生成均有限且对拍通过。

## 仓库回归

- [`test_torch_compat_ops.py`](../../../tests/compat/torch/test_torch_compat_ops.py)
  在 CPU/CUDA 覆盖同 shape float32 动态图 bool `__imul__` 的 finite、identity、
  shape 和 dtype。
- [`test_install_context.py`](../../../tests/compat/torch/test_install_context.py)
  覆盖 Transformers generation patch 的 scalar/list 区分与重复安装幂等性。
- [`test_torch_hf_models.py`](../../../tests/compat/torch/test_torch_hf_models.py)
  修正 shim 检测，新增显式 CUDA T5 batch-2 默认 decoder-start generation、合法 EOS
  `-inf` 与非 EOS score、prefill `2 -> 3` cache 增长和 cached-vs-full logits 回归，
  并断言模型、输入、输出和 cache 的 CUDA residency。
- Torch op suite 为 `29 passed`；serialization/state_dict suite 为 `28 passed`；
  HF alias 与真实 CUDA device suites 分别为 `3 passed`、`4 passed`；定向 T5 为
  `1 passed`。
- 文档/仓库布局门禁通过；`tests/structure` 为 `234 passed`。

## 产物

原始脚本、模型、日志和 NPZ 只保存在 `$JITTOR_LAB_ROOT`：

- `transformers_compat/bert_compat_cuda4_after_9eb696d9_20260902/`
- `transformers_compat/roberta_compat_cuda4_20260902/`
- `transformers_compat/encoder_quick_cuda1_20260903_v3/`
- `transformers_compat/gpt2_compat_cuda1_20260903_v2/`
- `transformers_compat/llama_compat_cuda1_20260903_v1/`
- `transformers_compat/t5_compat_cuda7_registry_bridge_20260903/`

各目录的 `comparison.json`、`meta_*.json`、`outputs/grads/cache/generation_*.npz`
和运行日志构成完整复现证据；环境与命令记录在
`transformers_compat/LOCAL_WORK_LOG.md`。

## 未覆盖范围

本结论只接受 tiny FP32、当前模型配置和真实 A800 CUDA 下的三类文本架构兼容，不能
外推为所有 Transformers 模型、公开 checkpoint 或所有后端均已验收。公开 tokenizer/
checkpoint、FP16/BF16、长序列、left padding、sampling/top-k/top-p、dropout 随机流、
多步优化、Trainer/Accelerate、ROCm/NPU 和性能需要独立结果。DeBERTa-v2/MPNet 本轮
未扩展到全部任务头与正式 round-trip；Encoder-only 的序列化结论由 BERT/RoBERTa
锚点承担。模型版本、cache class 或 generation internals 变化时应重新执行严格三阶段
对拍，不能只运行 import/forward smoke。
