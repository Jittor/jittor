# 用 PyTorch API 写 Jittor

Jittor 提供一层 Torch 兼容前端，让按 PyTorch 写的代码——包括 **transformers**、
**PEFT**、**accelerate**、**LlamaFactory**——**不改一行**跑在 Jittor 上，同时支持
**NVIDIA GPU（CUDA/NCCL）** 与 **华为昇腾 NPU（ACL/HCCL）**。

兼容进程必须**显式激活**：

```python
from jittor.compat.shim import activate
activate()
import torch
import torch.nn as nn
```

Torch 风格的 API 归属于 `jittor.compat.torch` 包。它只在选择了 Torch 入口时才安装：
显式调用 `jittor.compat.shim.activate()`、`JITTOR_TORCH_SHIM=1`、已部署的 `import
torch` 包，或历史路径 `import jittor.torch_compat`。

- 普通的 `import jittor as jt` 保持原生 Jittor 行为，**不会占用 `torch` 命名空间**。
- `import jittor as torch` 只是一个本地 Python 别名，**不会激活兼容层**。

默认的 `activate()` 与部署包使用**独立的** Tensor、Parameter、Module 和 NN 类型。
原生前端与 Torch 前端共享 Var/Op 运行时，但 **`torch is not jittor`**。历史上的
别名式安装仍可用（`activate(independent_namespace=False)`，或设了 `JITTOR_TORCH_SHIM=1`
但没设 `JITTOR_TORCH_INDEPENDENT=1`）；**在创建任何应用状态之前选定一种模式**。

这种分离在原生与 Torch 契约不同的 API 上是可观察的。典型例子：原生的 `Var.data`
仍是共享的 NumPy 视图，而 Torch 模式返回一个 detach 的 Tensor 别名，对它的原地写入
会更新持有者，且**不把 CUDA/NPU 数据搬经主机**。

把 shim 部署到当前环境：

```bash
jittor-torch-shim            # 安装到当前环境
jittor-torch-shim --check    # 校验已部署的文件
```

## 快速开始

```python
from jittor.compat.shim import activate
activate()
import torch

lin = torch.nn.Linear(8, 8)
opt = torch.optim.AdamW(lin.parameters(), lr=1e-3)
x = torch.randn(16, 8)
for _ in range(10):
    opt.zero_grad()
    loss = (lin(x) ** 2).mean()
    loss.backward()        # 由当前激活的优化器接管
    opt.step()
```

设备会自动启用：import 时只要存在加速器（昇腾 `has_acl` 或 NVIDIA `has_cuda`），
Jittor 就把 `flags.use_cuda` 置 1。

> **务必确认它真的在设备上跑**：`npu-smi` / `nvidia-smi` 应显示该进程占用 GB 级显存。
> 只有约 100 MB 说明算子跑在 CPU 上（慢约 1000 倍）。

## 数值正确性

与**真实 PyTorch**用相同权重和输入对拍（`tests/backends/npu/manual/xcheck/`）：

- GPT-2 前向 + 反向与 torch 相差约 `1e-7`（CUDA）/ `1e-5`（昇腾）；
- 真实 Qwen3-0.6B（经 transformers）产生**完全相同的 top-5 下一 token 预测**，
  logits 相差约 `1e-4`；
- jittor-ACL 与 jittor-CUDA 之间**逐位一致**。

## 混合精度（bf16 / fp16）

```python
scaler = torch.cuda.amp.GradScaler()                 # 函数式动态 loss scaler
with torch.autocast("cuda", dtype=torch.bfloat16):   # 既是上下文管理器也是装饰器
    loss = model(x, labels=y).loss
scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
```

- **昇腾上首选 bf16**（cube 单元原生支持：2048³ 矩阵乘约 0.9 ms，fp32 约 11 ms），
  loss 曲线与 fp32 一致；
- fp16 配合 GradScaler 可用（loss 缩放 + inf/nan 跳步）。

## 多卡 DDP —— 不需要 mpirun

一个 torchrun 风格的启动器在两种后端上都能用（NVIDIA 走 NCCL，昇腾走 HCCL），
通过环境变量/文件 rendezvous，**不依赖 MPI**：

```bash
# 2 张 GPU / NPU，数据并行：
python -m jittor.distributed.launch -n 2 -- python train.py
```

训练脚本里用常规的数据并行 API（`jt.rank`、`jt.world_size`、
`var.mpi_all_reduce("mean")`、`module.mpi_param_broadcast`）。已验证：两种后端上
**双卡梯度 == 单卡梯度**。

## 迁移 torch checkpoint 与 safetensors

```python
sd = torch.load("model.pt")                   # 直接读真实 torch 的 .pt（zip + storages）
from safetensors.torch import load_file       # 纯 Python，产出 jittor Var
weights = load_file("model.safetensors")
```

两者都能直接加载真实 torch 保存的文件（含 bf16 在内的全部 dtype），**不需要装真 torch**。

## 跑 transformers / LlamaFactory

环境要求：

- Python 3.10+（transformers 需要 `types.UnionType`）；
- `HF_DEACTIVATE_ASYNC_LOAD=1`——单线程权重实体化（工作线程没有设备上下文）；
- 离线运行设 `HF_HUB_OFFLINE=1`、`DISABLE_VERSION_CHECK=1`；
- 安装 `trl` 要加 `--no-deps`，否则它会拉进真 torch 把 shim 覆盖掉。

> **不要让 pip 把真实的 `torch` 装进 shim 环境**——它会覆盖 `torch/__init__.py`。
> 真 torch 请放在单独的环境里（例如作为对拍参考）。

## 已验证的模型覆盖

以下模型都跑通了 `import torch` → jittor 这条链，并与**真实 PyTorch 2.12 逐层对拍**
（相同权重与输入；前向比 `last_hidden_state`，反向比每个参数的 `jt.grad` 与 torch
的 `.grad`，按网络规模归一）。约 30 个 `transformers` 架构与 CNN/diffusers 栈的
**前向和反向**都吻合到约 `1e-6`：

- **解码器 LLM**：gpt2、llama、qwen2/qwen3、mistral、gemma/gemma2、phi/phi3、opt、
  bloom、gpt_neox、gptj、gpt_neo、stablelm、starcoder2、mpt、**falcon**（multi-query）、
  **mixtral**（MoE）。
  另有（前向干净，多数也验过反向）：cohere、gemma3、granite、olmo/olmo2、persimmon、
  qwen2_moe、qwen3_moe、glm/glm4、gpt_bigcode、biogpt、ctrl、xglm、codegen、
  **longformer**（滑动窗口）、**roformer**（旋转位置）、**phimoe**、**dbrx**、
  **nemotron**。
- **编码器**：bert、roberta、electra、distilbert、albert、**deberta/deberta-v2**、
  mpnet、xlm、flaubert、camembert、ernie、fnet、layoutlm、mobilebert、nystromformer、
  mra、yoso、**convbert**（span-conv）、megatron-bert、rembert、luke、markuplm、**canine**。
- **编码器-解码器**：t5、bart、mbart、pegasus、**pegasus_x**（块局部）、m2m_100、
  marian、blenderbot/-small、mvp、plbart、umt5、nllb-moe、fsmt、led、big_bird。
- **音频**：wav2vec2、**hubert**、**wavlm**（经 `F.multi_head_attention_forward`）、
  sew、unispeech/-sat、data2vec-audio。
- **视觉**：vit、deit、swin、convnext、**beit**、data2vec-vision、dpt、segformer、
  **levit**（hardswish），以及 `jittor.models` 的 resnet/vgg/… 和原生 ViT。
- **diffusers 生成**：`UNet2DModel` 前向 `1.1e-6` / 反向 `1.5e-6`，DDIM 去噪循环
  `3e-5`，`AutoencoderKL` 编码+解码 `1.4e-6`——也就是说 jittor **真的在生成**，
  且数值与 torch 吻合。用构造函数 / `from_config` 建模型；加载**预训练** checkpoint
  见下文。

端到端训练是可用的：`transformers.Trainer` 能微调（loss 下降，`grad_norm` /
`clip_grad_norm_` 生效），CNN 能训练（每个卷积权重都在更新）。端到端**推理**同样可用：
`model.generate()` 支持**贪心**（带 KV cache 的解码与从头重算逐位一致，说明 cache 是对的）、
**beam search**、**采样**（temperature/top-k/top-p）和**批量**生成。

回归套件覆盖约 30 个架构（`tests/compat/torch/test_torch_hf_models.py`，含
`generate()` 的贪心/beam/采样测试）与 diffusers 生成路径
（`tests/compat/torch/test_diffusers.py`）。

`jittor.models` 提供经典 CNN 以及现代 **Vision Transformer**（`vit_b_16`/`vit_b_32`/
`vit_l_16`）。LLM 与扩散模型直接来自 `transformers` / `diffusers`。

## 下游库状态

`tests/compat/torch/test_ecosystem_parity.py` 把每个用例**跑两遍**——一次在 `torch`
是真实 PyTorch 的解释器里，一次在本解释器里——从相同权重出发，比较前向输出以及
**每一个参数梯度和输入梯度**，CPU 与 CUDA 都比。另有一个真实 NPU 的作用域类覆盖
MMCV ConvModule 与 MMEngine BaseModule 对 `torch_npu` 的对拍，含 fail-closed 的
CPU 回退检测。用 `REAL_TORCH_PYTHON` 指向真 PyTorch 解释器，可选用
`JITTOR_ECOSYSTEM_SPEED_RATIO` 约束墙钟比值。

| 库 | 对拍门禁覆盖 | 说明 |
| --- | --- | --- |
| `transformers` | gpt2、llama、bert、vit、t5、whisper | 编码器、解码器、编码器-解码器与音频路径 |
| `diffusers` | `UNet2DModel`、`DiTTransformer2DModel` | 卷积与隐空间 transformer 骨干 |
| `peft` | llama 上的 LoRA | |
| `ms-swift` | 它自己的 LoRA tuner（llama） | 需要 `peft < 0.20`；ms-swift 4.5.2 配 peft 0.19 在真 PyTorch 下也报同样的 `TypeError` |
| `mmcv` / `mmengine` | `mmcv.cnn.ConvModule`、`mmengine.model.BaseModule` | CPU、CUDA、NPU；仅限纯 Python 层，见下 |

有两条边界属于**架构性的**，不是尚未完成的工作：

- **编译过的 PyTorch 扩展。** `mmcv.ops` 以及任何自带 `torch` 链接 `.so` 的包，都是
  针对 PyTorch 的 C++ ABI 编译的。**Python 层的兼容层无法加载它们**；这些算子需要
  Jittor 自己的实现——`jittor.models` 和原生算子面就是为要紧的那些场景提供的。
- **自己掌管设备的运行时。** vLLM 内嵌自定义 CUDA kernel 和它自己的显存与调度层，
  而不是调用 `torch.*`，所以光靠 shim 带不动它。它通过外部适配器
  （`vllm_jittor_ops`）运行，由适配器针对 Jittor 提供那些 kernel：vLLM V1 能加载
  Qwen3-0.6B、建立 KV cache、完成真实 CUDA 上的贪心解码，token 与真实
  PyTorch/Transformers **完全一致**，四 token 暖启动生成 `0.109s` 对 `0.137s`。
  verl 走同一条路——它的 import、协议、FSDP2 和 PPO 门禁都通过，含四卡 FSDP2。

TRELLIS.2 4B 在同一个外部适配器上以四个真实 CUDA 扩展完成了对齐的端到端流水，但
暖态流水中位数是 `7.515s` 对真实 PyTorch 的 `6.878s`——约 `1.09x`，**性能尚未验收**。

TRELLIS 这类项目粘合代码放在通过 entry point 注册的可选集成发行物里
（`jittor-trellis` 等），**不在主线 Jittor 中**，理由见
[仓库布局](../development/repository-layout.md)。

## 复数与 FFT

```python
c = torch.complex(re, im)                 # -> 原生 complex64 Var
torch.view_as_complex(x); torch.view_as_real(c)
torch.polar(abs, angle); torch.real(c); torch.imag(c); torch.conj(c)
torch.fft.fft(x); torch.fft.ifft(x); torch.fft.rfft(x); torch.fft.irfft(r, n=N)
torch.fft.fft2(x2); torch.fft.fftn(x, dim=(-2,-1))   # norm='backward'|'forward'|'ortho'
```

`complex64` 是 `jittor_core` 的一等 dtype，上述公开 API 接受并返回原生复数 `Var`。
部分线性代数与 FFT 算法内部仍在用旧的 `jittor.nn.ComplexNumber` 实部/虚部表示，
再把结果转回原生张量。该桥接是实现细节，限制已登记，见
[复数 dtype](../notes/complex-dtype.md)。

## Lightning 风格训练

```python
import jittor.lightning as pl          # 或：import pytorch_lightning as pl（已做别名）

class Lit(pl.LightningModule):
    def training_step(self, batch, idx): ...; return loss
    def configure_optimizers(self): return torch.optim.Adam(self.parameters(), lr=1e-3)

pl.Trainer(max_epochs=5, gradient_clip_val=1.0,
           callbacks=[pl.ModelCheckpoint(monitor="val_loss"),
                      pl.EarlyStopping(monitor="val_loss", patience=3)]).fit(model, train_loader)
```

核心训练/验证循环已实现（epoch、梯度累积、裁剪、lr 调度、`self.log`、
`ModelCheckpoint`/`EarlyStopping` 回调）。**DDP 策略、精度插件和完整的 logger 生态
尚未覆盖。**

## 报错

算子失败会给出真实原因（算子类型、输入形状/dtype、`[Reason]`），而不是旧的
"Wrong inputs arguments / help(jt.sync)" 噪声。不支持的 dtype（例如昇腾上的 float64）
抛出干净的 Python 异常而不是直接中止。异步算子失败时设 `JT_SYNC=1` 精确定位。

## torch API 覆盖（对真实 PyTorch 2.12 验证）

下列每一项都在 **CPU 与 CUDA 上**与真实 PyTorch 用相同输入/权重**逐位（或到约 1e-6）
对拍**过，并锁进回归套件（`test_torch_compat.py`、`test_torch_compat_linalg.py`、
`test_torch_compat_distributions.py`、`test_torch_hf_models.py`、`test_peft.py`、
`test_diffusers.py`）。

- **注意力 / transformer**：`F.scaled_dot_product_attention`（普通/因果/bool mask/
  scale/GQA，前向+反向）、`F.multi_head_attention_forward`、`nn.MultiheadAttention`、
  `nn.TransformerEncoderLayer`/`TransformerEncoder`/`TransformerDecoderLayer`/
  `TransformerDecoder`/`Transformer`（pre/post-norm、`generate_square_subsequent_mask`）。
- **循环网络**：`nn.LSTM`/`GRU`/`RNN`（含 Cell）——前向与 torch 逐位一致（门顺序
  i/f/g/o），`batch_first` 输出正确为 `(batch, seq, hidden)`，支持双向、`h_n`/`c_n`。
- **归一化 / 激活**：`F.rms_norm`（Llama/Qwen）、`group_norm`/`batch_norm`/
  `instance_norm`/`layer_norm`、`silu`/`mish`/`hardswish`/`hardsigmoid`/`glu`/`elu`/
  `selu`/`celu`/`softplus`/`tanhshrink`/`softmin`/`threshold`。
- **损失**：`cross_entropy`（含 `label_smoothing`/`weight`/`ignore_index`）、`kl_div`
  （蒸馏、`batchmean`）、`ctc_loss`（语音识别的 CTC 前向 DP）、`F.logsigmoid`
  （DPO/RLHF）、`binary_cross_entropy`(`_with_logits`)、`huber_loss`、
  `cosine_embedding_loss`、`margin_ranking_loss`、`triplet_margin_loss`、
  `gaussian_nll_loss`、`poisson_nll_loss`、mse/l1/smooth_l1，以及对应的 `nn.*Loss` 类。
- **`torch.distributions`**：`Categorical`（logits=softmax，可导——修掉了一个会让 PPO
  失效的静默 sigmoid 缺陷）、`Normal`、`Bernoulli`、`Exponential`、`Uniform`、
  `Geometric`、`Independent`、`OneHotCategorical`、`kl_divergence`、`Distribution` 基类。
- **`torch.linalg`**：`svd`（`full_matrices`、具名 `(U,S,Vh)`）、`svdvals`、`inv`、
  `solve`、`cholesky`、`det`/`slogdet`、`eigh`/`eigvalsh`/`eigvals`、`qr`、`pinv`、
  `matrix_rank`、`multi_dot`、`lstsq`、`norm`/`matrix_norm`（CUDA 的 svd/eigh 需要 `cupy`）。
- **`torch.func`**：`functional_call`、`grad`/`grad_and_value`、`vmap`、`jacrev`、
  `stack_module_state`（LoRA / 元学习 / 集成）。
- **算子与方法**：`einsum`、`take_along_dim`（含广播）、`roll`（含负维与展平）、
  `cumprod`（符号感知）、`index_fill_`、`index_put_`（重复下标累加）、`movedim`/
  `tensor_split`/`take`、`cdist`、`bucketize`、`searchsorted`、`pixel_shuffle`/
  `pixel_unshuffle`、`gumbel_softmax`、`interpolate`、`grid_sample`、`all`/`any`（`axis=`）。
- **`nn.utils`**：`weight_norm`/`spectral_norm`（真实的重参数化）、`clip_grad_*`、
  `rnn.pad_sequence`；`torch.optim.lr_scheduler`（LambdaLR/Linear/Cosine/Step/MultiStep/
  Exponential/Polynomial/Constant/Sequential/ReduceLROnPlateau）。

## 状态与限制

**两种后端上都已完成并验证**：约 75 个 transformers（解码器/编码器/编码器-解码器/
音频/视觉/MoE）加 CNN 加 diffusers 生成模型的前向/反向/训练精度一致性、设备分派、
bf16 与混合精度、无需 mpirun 的 DDP、梯度检查点、checkpoint/safetensors 迁移、
`model.save()`/`load()`、真实的 `torch.cuda` 显存报告、复数与 `torch.fft.*`、
`F.multi_head_attention_forward`、`torch.func`（functorch 系列，与真 torch 逐位一致）、
`nn.utils.weight_norm`/`spectral_norm`（真实重参数化：`weight` → `weight_g`/`weight_v`
在前向前重算，σ 用幂迭代；与真 torch 和 `np.linalg.svd` 对拍过）、
`nn.utils.rnn.pad_sequence`、torch 兼容的 `torch.optim.lr_scheduler`（在
`import jittor as torch` 与已部署 shim **两条路径上单一实现**；HF 的
`get_*_schedule_with_warmup` 包装 LambdaLR 并产生与 torch 完全相同的曲线）、
Lightning 风格训练核心，以及清晰的报错。算子面通过算子级差分对拍
（`op_parity.py`：约 84 个算子对真 torch，另有反向对拍）在**昇腾和 CUDA 上都**验证过。

**用 `from_pretrained` 加载预训练 checkpoint**——包括 accelerate 快路径（diffusers，
以及 transformers 默认开启的 `low_cpu_mem_usage=True`）——现已可用且能**精确**还原
权重。accelerate 在 `init_empty_weights()` + `no_init_weights()` 下构造模型，再用
`set_module_tensor_to_device` 逐个填参数；jittor 没有 `meta` 设备，靠两处修正让这条
路径正确：(1) `torch.nn.init` 被保护，使 `no_init_weights()` 不会把 jittor 自己的
构造初始化清空；(2) `Module._parameters` / `_buffers` 改为写穿，使 accelerate 的
`module._parameters[name] = value` 真正生效并保持参数/缓冲的分类。已验证：diffusers
`UNet2DModel` 的 save → `from_pretrained` → 前向往返在 meta 与普通两条路径上都吻合到
`0.0`，transformers `BertModel` 往返同样吻合到 `0.0`。

**NumPy 2.x 与 Python 3.13** 是维护中的兼容路径。Jittor 为数组拷贝选择带版本的
NumPy C-API 入口，避免构造 legacy dtype descriptor，并从 Jittor dtype 推导传输尺寸。
Python 3.13 的 wheel 门禁在 NumPy 2.x 下运行，覆盖 CPU 自检、非 C 连续数组传输和
Python 变量追踪。包接受 3 以下的所有 NumPy 版本。

**进行中（更底层）**：`complex128` 与消除剩余内部复数桥接的原生 kernel、PP/TP、
显存管理器调优、CUDA 11/13 的 wheel 家族（对齐的 CUDA 12.2 栈——cuDNN 8.9.7 或更新
——已通过 `jittor[cuda12]` 提供）、Lightning 的剩余接口（DDP/精度/logger），以及
triton/tilelang 自定义算子支持。
