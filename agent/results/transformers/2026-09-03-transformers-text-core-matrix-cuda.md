# Transformers 4.56.2 文本核心模型矩阵 CUDA 扩展验证

- Status: 17-model tiny FP32 text core matrix accepted on CUDA
- Last reviewed: 2026-09-03
- Source baseline: `eb4fdedf` plus this report's acceptance-level follow-up
- Owner: Torch compatibility and Transformers maintainers
- Review when: Transformers model/cache/generation/serialization APIs, MoE routing,
  sliding-window attention, or the covered Jittor compatibility paths change

## 结论

工作日志指定的 Transformers 4.56.2 文本核心矩阵已经全部执行，而不是把 GPT-2 与
Llama 的锚点结果外推到其余 Decoder-only 模型。最终范围为 Encoder-only 4 个、
Decoder-only 10 个、Encoder-decoder 3 个，共 17 个模型：

| 架构 | 已验证模型 | 结果 |
| --- | --- | --- |
| Encoder-only | BERT、RoBERTa、DeBERTa-v2、MPNet | 4/4 PASS |
| Decoder-only | GPT-2、GPT-NeoX、Llama、Qwen2、Qwen3、Mistral、Gemma2、Phi-3、Falcon、Mixtral | 10/10 PASS |
| Encoder-decoder | T5、BART、Pegasus | 3/3 PASS |

首阶段只选 GPT-2 与 Llama 是为了先关闭 learned position 与 RoPE/GQA 两个主锚点，
并不意味着其余模型的实现路径完全相同。扩展结果证明这些模型需要独立覆盖：Falcon
使用 MQA，Qwen3 有独立 `head_dim`/`cache_position` 路径，Mistral 和 Gemma2 有不同
滑动窗口 cache，Phi-3 使用 fused QKV 和 partial RoPE，Mixtral 还引入 router、auxiliary
loss 和稀疏 experts。普通 dense decoder 不能替代 MoE 验证。

所有新增模型均在真实 NVIDIA A800 CUDA 上，以真实 PyTorch 2.6.0+cu124 为独立
oracle，并使用 Transformers 4.56.2、相同配置、权重和固定输入。前向输出、任务 loss、
所有实际参与计算的命名参数梯度、state/safetensors、双向 `save_pretrained` round-trip
均经过 fail-closed 比较；所有生成模型还验证 cache prefill/decode、cached-vs-full
logits、greedy 与 3-beam generation。模型参数、输入、loss、输出、梯度、cache 和生成
结果均断言位于目标 CUDA device，不以导入成功或 CPU fallback 作为通过。

## 验收层级

按工作日志中 L0-L5 的严格累计定义，本轮不能标记为“完整 L3”或“完整 L4”。当前正式
状态是：**17/17 模型完成 L0，能力子门禁覆盖到 L4-partial，M2 文本模型实现矩阵完成**。

| 层级 | 当前状态 | 已通过证据 | 尚缺条件 |
| --- | --- | --- | --- |
| L0 | COMPLETE | 17/17 公共 Config/AutoClass、tiny config、参数清单与两端独立构造 | 无 |
| L1 | PARTIAL | 17/17 相同权重前向、输出结构/mask 与真实 CUDA 数值对拍，无 CPU fallback | 尚未对 17 个模型逐一完成 CPU 对拍 |
| L2 | PARTIAL | 17/17 task loss 与全部实际参与计算的 trainable gradients 对拍 | train/eval、dropout、zero-grad 和 optimizer step 尚未在整个矩阵逐项补齐 |
| L3 | PARTIAL | 主锚点及本轮 10 个新增模型完成 state/safetensors/双向 `save_pretrained` round-trip | DeBERTa-v2/MPNet 正式 round-trip 及矩阵级 tokenizer 尚未完成 |
| L4 | PARTIAL | 10 个 Decoder-only 与 3 个 Encoder-decoder 均完成 prefill/decode、cache、greedy/beam | 未使用真实公开 checkpoint/tokenizer，sampling 未形成矩阵门禁 |
| L5 | NOT RUN | 无性能验收声明 | 需同 device/dtype/shape 的稳态吞吐、显存与热点证据 |

因此这里的“17/17 PASS”只表示已声明的 tiny FP32 CUDA 数值与公共模型工作流通过，不能
替换 L3 的 tokenizer 条件或 L4 的真实 checkpoint 条件。后续要把整体等级提升到完整
L3/L4，必须补齐上表缺口后重新更新状态，不能只复用当前 generation 结果。

## 环境与判据

| 项目 | 配置 |
| --- | --- |
| Jittor | `1.3.11.0`, baseline `26c5fb13` plus working change |
| Transformers | `4.56.2` |
| PyTorch oracle | `2.6.0+cu124`, real site-packages |
| Device | NVIDIA A800 80GB PCIe, compute capability 8.0 |
| Jittor CUDA | CUDA 12.2.140, cuDNN 8, `sm_80` |
| Dtype | FP32 parameters/activations, int64 token ids |
| Attention | eager，另按模型验证滑动窗口、GQA/MQA 和 softcap 路径 |
| Isolation | 每组独立 `HOME`、`JITTOR_HOME`、`TMPDIR`、runtime root 与 `cache_name` |

前向、cache 和 generation 的有限值采用

$$
e_{\mathrm{rel},\infty}=
\frac{\max_i |x_i-y_i|}{\max_i |y_i|+10^{-8}}
$$

比较，门槛为 $10^{-4}$。梯度先精确比较键集合、shape、缺失集合、finite 与 device，
再以全网梯度尺度归一，门槛为 $10^{-5}$。state 门槛为 $10^{-7}$；token、cache
signature、序列化 keys、missing/unexpected/mismatched keys 和非有限 mask 使用精确比较。

## Decoder-only 扩展

下表列出首阶段 GPT-2/Llama 之外的 8 个模型；两项锚点数据及 Encoder-only 四模型
见[三类文本架构首阶段报告](2026-09-03-transformers-text-architectures-cuda.md)。

| 模型 | 独立实现特征 | 梯度 | 前向最大误差 | 梯度最大误差 | Cache 最大误差 | Generation 最大误差 | 结果 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| GPT-NeoX | rotary embedding、parallel residual | 28/28 | $2.267\times10^{-7}$ | $2.259\times10^{-7}$ | $2.859\times10^{-7}$ | token exact | PASS |
| Qwen2 | RoPE、GQA、RMSNorm/SwiGLU | 27/27 | $2.599\times10^{-7}$ | $2.518\times10^{-7}$ | $1.823\times10^{-7}$ | token exact | PASS |
| Qwen3 | `head_dim=12`、显式 position/cache position | 25/25 | $2.416\times10^{-7}$ | $4.213\times10^{-7}$ | $2.562\times10^{-7}$ | token exact | PASS |
| Mistral | `sliding_window=3`、滚动 cache | 21/21 | $2.853\times10^{-7}$ | $1.712\times10^{-7}$ | $2.502\times10^{-7}$ | token exact | PASS |
| Gemma2 | sliding/full 交替、attention/final softcap | 24/24 | $5.566\times10^{-7}$ | $2.523\times10^{-7}$ | $4.496\times10^{-7}$ | $4.328\times10^{-7}$ | PASS |
| Phi-3 | fused QKV、`partial_rotary_factor=0.5` | 15/15 | $1.801\times10^{-7}$ | $2.053\times10^{-7}$ | $1.882\times10^{-7}$ | $2.342\times10^{-7}$ | PASS |
| Falcon | MQA、parallel attention | 16/16 | $3.207\times10^{-7}$ | $3.269\times10^{-7}$ | $3.296\times10^{-7}$ | token exact | PASS |
| Mixtral | 2 层、2 experts、top-2 sparse MoE | 29/29 | $1.558\times10^{-7}$ | $3.025\times10^{-7}$ | $1.736\times10^{-7}$ | $1.424\times10^{-7}$ | PASS |

GPT-NeoX 的两层 cache 从 `[2,4,4,16]` 增长至 `[2,4,5,16]`；Falcon 的 MQA
cache 从 `[2,1,4,16]` 增长至 `[2,1,5,16]`。Qwen2 为
`[2,2,4,16] -> [2,2,5,16]`，Qwen3 的非默认 head dimension 路径为
`[2,2,4,12] -> [2,2,5,12]`。Phi-3 与 Mixtral 的两层 cache 均从逻辑长度 6
增长到 7。

Mistral 的 `DynamicSlidingWindowLayer` 累计长度从 4 增至 5，但 `window=3` 下物理
K/V backing 按上游设计维持 `[2,2,2,16]`。Gemma2 的 sliding 层 backing 维持
`[2,2,3,8]`，full 层从 `[2,2,6,8]` 增至 `[2,2,7,8]`；这两项均同时比较
逻辑长度、cache class、逐层 K/V shape 和数组值，未把“存储 shape 不增长”误判为失败。

## Mixtral MoE

Mixtral 训练阶段显式开启 `output_router_logits` 和 auxiliary router loss。Jittor 与
PyTorch 的 causal LM loss 均为 `4.588456630706787`，aux loss 均为 `2.0`，两个
router logits 均为 `[16,2]`。29/29 命名参数梯度全部位于 CUDA、有限且非零，其中
2/2 gate 权重和 12/12 expert `w1/w2/w3` 权重全部具有非零梯度；因此该结果实际执行
了 MoE 路由和 expert 计算，而不是只让共享 attention/embedding 参与 loss。

Transformers 4.56.2 的 Mixtral 在 cached 单 token decode 与
`output_router_logits=True` 组合下，真实 PyTorch 也会在 load-balancing loss 中发生
attention-mask 行数不匹配。最终合法契约按上游公共 API 拆分：训练验证 router/aux，
cache/generate 显式关闭 router 输出。这个上游组合限制保存在 v2 PyTorch 日志中，
没有用宽泛异常捕获或 Jittor 特判掩盖。

## Encoder-decoder 扩展

| 模型 | 梯度 | Loss | 前向最大误差 | 梯度最大误差 | Cache 最大误差 | Generation 最大误差 | 结果 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BART | 49/49 | `4.1584444046` | $1.621\times10^{-7}$ | $1.363\times10^{-7}$ | $2.106\times10^{-7}$ | $2.093\times10^{-7}$ | PASS |
| Pegasus | 47/47 实际梯度 | `4.2190656662` | $1.828\times10^{-7}$ | $1.188\times10^{-7}$ | $2.262\times10^{-7}$ | $2.317\times10^{-7}$ | PASS |

BART 与 Pegasus 均使用 batch 2、encoder 长度 6、teacher-forcing decoder 长度 5；
`EncoderDecoderCache` 的 self K/V 从长度 3 增至 4，cross K/V 保持长度 6。两端
greedy/3-beam 序列逐元素一致，scores 的有限值和允许出现的 EOS `-inf` mask 精确一致。

Pegasus 有一项上游共享限制需要显式说明：Transformers 4.56.2 从
`save_pretrained` 重新加载后，encoder/decoder 的两个 `embed_positions.weight` 被标记为
`requires_grad=True`，但 forward 使用的正弦位置表与这两个参数断开，因此真实 PyTorch
和 Jittor 的 `.grad` 均为 `None`。直接 `from_config` 的真实 PyTorch 模型原本把它们
冻结；`low_cpu_mem_usage=True/False` 均能复现 round-trip 后的状态变化。比较器要求两端
缺失集合精确等于这两个键，并对其余 47/47 梯度完整比较，未将 Jittor 独有的缺失项
豁免为通过。

## 序列化与 round-trip

8 个新增 Decoder-only 和 BART/Pegasus 都执行真实 PyTorch source checkpoint 到
Jittor strict load，再由 Jittor `save_pretrained` 到真实 PyTorch strict reload。
state/safetensors keys、tied-weight 去重、missing/unexpected/mismatched keys 均核验；
reload 后再次执行 forward、cache 和 generation。所有模型双向 round-trip 通过，
Jittor 保存后由 PyTorch 复验的 Mixtral forward/cache/generation 误差均为零。

本轮没有从 Hugging Face Hub 下载公开大 checkpoint，也没有删除用户可能复用的模型
缓存。各用例使用合法 tiny config 自建权重，以便覆盖模型实现、梯度、cache 与序列化
控制流；生成的 `model.safetensors`、`save_pretrained` 目录和全部 NPZ/JSON/log 仍保留
在 `$JITTOR_LAB_ROOT/transformers_compat/`，可供评测后统一清理。

## 修复与仓库回归

扩展矩阵没有暴露新的共享算子错误，但复核首阶段 T5 修复时，把动态图 bool mask 的
`__imul__` 旁路进一步收窄为“仅将同 shape bool RHS cast 到 lhs float dtype，再调用
原生 in-place slot”。这样保留原生 dispatch、shape 与 autograd 路径；低层 CPU/CUDA
最小复现和完整 T5 strict CUDA 流程均重新通过。

仓库测试补入 Falcon 与 Mixtral tiny 配置，修复 Falcon 曾因缺配置而被循环静默跳过的
伪覆盖；通用梯度 loss 纳入 `pooler_output`，使 BERT/ViT pooler 参数确实参与反传。
新增 Mixtral CUDA 定向测试锁定 router/aux、全部参数梯度、1 个 gate 与 6 个 expert
矩阵非零梯度、cache `4 -> 5`、cached-vs-full 以及 2-beam generate。该测试在物理
GPU9、全新独立串行 JIT cache 上为 `Ran 1 test in 803.132s, OK`；九模型完整梯度循环
随后在同一 cache 上为 `Ran 1 test in 600.882s, OK`。T5 当前源码严格复验 11/11
checks PASS，前向、26 项梯度、cache 最大误差分别为 $2.289\times10^{-7}$、
$5.805\times10^{-7}$、$2.763\times10^{-7}$。仓库布局与文档治理通过；CPU-only
独立暖 cache 的 `tests/structure` 为 `232 passed, 2 skipped in 29.51s`。

## 产物

关键实验目录均相对于 `$JITTOR_LAB_ROOT/transformers_compat/`：

- `gpt_neox_compat_gpu1_20260903_v1/`
- `falcon_compat_gpu1_20260903_v1/`
- `dense_decoder_qwen2_cuda2_20260903/`
- `dense_decoder_qwen3_cuda2_20260903/`
- `dense_decoder_mistral_cuda2_20260903/`
- `ecd_gemma2_cuda3_20260903_v2/`
- `ecd_phi3_cuda3_20260903_v2/`
- `ecd_mixtral_cuda3_20260903_v3/`
- `bart_compat_cuda4_20260903_v2/`
- `pegasus_compat_cuda4_20260903_v3/`
- `t5_compat_cuda6_native_bool_cast_20260903/`

每个最终目录的 `comparison.json` 为汇总结论，NPZ、metadata、state/safetensors 和运行
日志为逐项证据。未入库 harness 的 SHA-256 为：

| Harness | SHA-256 |
| --- | --- |
| `decoder_ext_compat.py` | `1598af4def07da78b0dc2c593605782cc8dc0f4f244009d449a721f6146e7bff` |
| `run_decoder_ext_compat.sh` | `4cb214a64e6f8e6c123d9aab03cc50e58960f3db7fdee26ac776e1390ab0e137` |
| `dense_decoder_compat.py` | `4b91851d0f4d6a8d65949b842e17eae1662c83e0d4061b2cde5f85f04e7e7285` |
| `decoder_ext_ecd_compat.py` | `552f813b43c809d6fde18cbbee1f463d07cd0eef6739331f7ebd8d2efdacb6a2` |
| `t5_compat.py` | `e191da8761dbe9d8071e7a51da2a689a000469b2869d73dd0afee7c8b3196d14` |

## 未覆盖范围

本结论接受的是当前 17 个 tiny FP32 config 在真实 A800 CUDA 上的模型实现兼容，不等价
于公开大 checkpoint、tokenizer 端到端质量或全部 Transformers 模型均已验收。真实
Hub checkpoint、FP16/BF16、量化、长上下文、left padding、sampling、Trainer/
Accelerate、PEFT、多卡、ROCm/NPU 与性能仍是独立 L4 范围；后续若下载真实 checkpoint，
应继续保留到用户验收后再清理。已有 tiny round-trip checkpoints 本轮全部保留。
