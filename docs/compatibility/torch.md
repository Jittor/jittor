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

`activate()` 与部署包使用**独立的** Tensor、Parameter、Module 和 NN 类型。
原生前端与 Torch 前端共享 Var/Op 运行时，但 **`torch is not jittor`**。

**把 Torch API 装到原生 `jittor` 模块上的别名模式已经移除**，它不再是一种可选形态：
`activate(independent_namespace=False)`、`JITTOR_TORCH_INDEPENDENT=0` 和直接调用
`compat.torch.install(jittor)` 都会明确报错而不是退回旧行为（`compat/shim/preflight.py`
的 `require_independent_frontend()`，以及部署的 `torch/__init__.py` 开头那条检查）。
保留下来的关键字只用于给出迁移错误。曾经用来选模式的 `JITTOR_TORCH_INDEPENDENT`
现在**不需要设**；把它设成假值会让 `import torch` 直接失败。

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

## 设备怎么命名

**加速器一律以 `cuda` 这个设备名对外暴露，昇腾也是。** 这不是笔误：昇腾构建上
`torch.cuda.is_available()` 返回 `True`（`compat/torch/installers/cuda/api.py` 的
`is_available()` 同时看 `has_cuda` 与 `has_acl`），`torch.cuda.get_device_name()`
返回 `Ascend910B/NPU`，`.cuda()` / `.to("cuda")` 把张量放到昇腾卡上。这样写死
`cuda` 的上游代码不用改一行就能跑。

`npu` 的拼写同时可用：`.npu()`、`.to("npu")`、`"npu:1"` 这类设备串都会解析到 ACL
后端（`compat/torch/types.py`、`compat/torch/frontend.py`、
`compat/torch/installers/nn/module_methods.py`）。

**但 `torch.npu` 是一个有意保持惰性的命名空间：`torch.npu.is_available()` 永远返回
`False`。** `torch.npu`、`torch.xpu`、`torch.mps`、`torch.mtia` 都是同一批占位模块
（`compat/torch/installers/cuda/bindings.py`）。这是刻意的：很多库用
`torch.npu.is_available()` 作为开关去走 `torch_npu` 专属路径——而 `torch_npu` 是编译
过的 PyTorch 扩展，这里没有它。答 `False` 让那些分支保持关闭，计算走通用的 `cuda`
路径。同理，可选的 Transformers 适配器把 `is_torch_npu_available()` 也改成返回
`False`（`adapters/jittor_adapters/transformers.py`）。所以：**判断有没有加速器请问
`torch.cuda.is_available()`，不要问 `torch.npu.is_available()`。**

`torch.__file__` 指向实际被导入的那个 `torch/__init__.py`——已部署的那份入口文件，
或者（未经部署入口激活时）它所复制自的 shim 源文件。独立命名空间自己没有文件，
`compat/shim/runtime.py` 的 `_adopt_entry_file()` 把入口的路径接到它上面，这样
`os.path.dirname(torch.__file__)` 这种常见写法才能拿到一个真实目录。

## 数值正确性

与**真实 PyTorch**用相同权重和输入对拍（`tests/backends/acl/manual/xcheck/`，
手工运行，需要真实昇腾设备）：

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

`torch.amp` 的整个表面都在：`autocast`、`GradScaler`、`custom_fwd`/`custom_bwd`、
`is_autocast_available`，以及 `autocast_mode`/`grad_scaler` 两个子模块；
`torch.cuda.amp` 与 `torch.cpu.amp` 是把设备写死的子类（`issubclass` 成立）。
per-device 的状态函数（`is_autocast_enabled`/`set_autocast_enabled`/
`get_autocast_dtype`/`set_autocast_dtype`/`clear_autocast_cache`/
`autocast_increment_nesting` 及其旧拼写）都是真实读写，不是空实现。

已知差异（`fidelity_of("torch.autocast").detail` 里逐条写着）：

- autocast 落到 jittor 的全局 amp 寄存器，而不是 torch 的逐算子白名单。matmul /
  linear / 卷积会降精度、归约保持 float32（与 torch 一致），但 torch 不动的
  fall-through 类（float32 `add`）在这里也会降精度；
- 一个寄存器服务所有 device type，CPU 区域与 CUDA 区域不像 torch 的 dispatch key
  那样互相独立；
- `is_autocast_available` 只对本次构建能跑的后端（cpu/cuda，装了 ACL 时还有 npu）
  回答真，torch 会对 xpu/mps/xla/ipu/mtia 也回答真——在这里那等于允许
  `torch.autocast("xla")` 悄悄改掉当前后端的精度，所以直接拒绝。

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

`torch.load` 的安全规则：

- **`weights_only` 默认为 `True`**，与 torch ≥ 2.6 一致；格式探测本身也走受限的
  Unpickler，所以一个恶意的普通 `.pt` pickle 在探测阶段就不会被执行。
- 标准的 torch zip checkpoint 在受限模式下读取：按 storage offset、shape 与 stride 重建，
  负 stride 和越出存储的描述在构造之前报错，未知的存储 dtype 被拒绝而不是当成 float32。
  读取会把值物化出来：共享存储的往返标识、非连续的物理布局都不保留。
- 旧版（非 zip）torch 格式、原生 URL 加载与原生 `.pkl` 回退**没有**可注入受限 Unpickler
  的接口，因此必须显式传 `weights_only=False`，并且只用于可信输入。
- safetensors：NumPy 请求仍然返回 NumPy 数组；torch 请求保留宽整数、BF16 与所请求的设备；
  float8 显式不支持；编码不支持的 dtype 在打开输出文件之前就报错。

## 跑 transformers / LlamaFactory

环境要求：

- Python 3.10+（transformers 需要 `types.UnionType`）；
- `HF_DEACTIVATE_ASYNC_LOAD=1`——单线程权重实体化（工作线程没有设备上下文）；
- 离线运行设 `HF_HUB_OFFLINE=1`、`DISABLE_VERSION_CHECK=1`；
- 安装 `trl` 要加 `--no-deps`，否则它会拉进真 torch 把 shim 覆盖掉。

> **不要让 pip 把真实的 `torch` 装进 shim 环境**——它会覆盖 `torch/__init__.py`。
> 真 torch 请放在单独的环境里（例如作为对拍参考）。

## 已验证的模型覆盖

以下模型都跑通了 `import torch` → jittor 这条链，并与**独立安装的真实 PyTorch
逐层对拍**（相同权重与输入；前向比 `last_hidden_state`，反向比每个参数的 `jt.grad`
与 torch 的 `.grad`，按网络规模归一）。约 30 个 `transformers` 架构与 CNN/diffusers
栈的**前向和反向**都吻合到约 `1e-6`。

> **对拍用的是哪个 torch。** 兼容层声明的 Torch API 级别是 `2.11.0`
> （`torch.__torch_version__` / `torch.version.__version__`；`torch.__version__`
> 报告的是 Jittor 自己的版本）。作为参照物的真实 PyTorch 版本**按报告而异**：
> nightly 生态门禁用 CPU 版 `2.7.1`（`.github/workflows/ecosystem.yml`），
> [真实模型差距表](../results/2026-09-24-torch-compat-real-models.md)用
> `2.11.0+cu128`，[下游库台账](../performance/library-standing.md)里较新的几次测量
> 用 `2.13.0+cu129`。引用某个数字时连它的参照版本一起引用。

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

回归套件覆盖约 30 个架构（`compat/tests/torch/test_torch_hf_models.py`，含
`generate()` 的贪心/beam/采样测试）与 diffusers 生成路径
（`compat/tests/torch/test_diffusers.py`）。

`jittor.models` 提供经典 CNN 以及现代 **Vision Transformer**（`vit_b_16`/`vit_b_32`/
`vit_l_16`）。LLM 与扩散模型直接来自 `transformers` / `diffusers`。

## 下游库状态

`compat/tests/torch/test_ecosystem_parity.py` 把每个用例**跑两遍**——一次在 `torch`
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
| `torchmetrics` | 分类与回归指标（`compat/tests/torch/test_torchmetrics_compat.py`） | 1.7.4，经 `jittor_adapters.torchmetrics` 适配 |

有两条边界属于**架构性的**，不是尚未完成的工作：

- **编译过的 PyTorch 扩展。** `mmcv.ops` 以及任何自带 `torch` 链接 `.so` 的包，都是
  针对 PyTorch 的 C++ ABI 编译的。**Python 层的兼容层无法加载它们**；这些算子需要
  Jittor 自己的实现——`jittor.models` 和原生算子面就是为要紧的那些场景提供的。
- **自己掌管设备的运行时。** vLLM 内嵌自定义 CUDA kernel 和它自己的显存与调度层，
  而不是调用 `torch.*`，所以光靠 shim 带不动它。它通过可选适配器
  `jittor_adapters.vllm`（随 `jittor-torch-adapters` 发行物分发，entry point 名
  `jittor_vllm`）运行，由适配器针对 Jittor 提供那些 kernel：vLLM V1 能加载
  Qwen3-0.6B、建立 KV cache、完成真实 CUDA 上的贪心解码，token 与真实
  PyTorch/Transformers **完全一致**，四 token 暖启动生成 `0.109s` 对 `0.137s`。
  verl 走同一条路——它的 import、协议、FSDP2 和 PPO 门禁都通过，含四卡 FSDP2。

TRELLIS.2 4B 曾在一个**不在本仓库**的外部适配器上以四个真实 CUDA 扩展完成对齐的
端到端流水，暖态流水中位数是 `7.515s` 对真实 PyTorch 的 `6.878s`——约 `1.09x`，
**性能尚未验收**；该测量是一次性结论，见[下游库台账](../performance/library-standing.md)。

项目专属的粘合代码都在通过 entry point 注册的可选适配器里，**不在主线 Jittor 中**，
理由见[仓库布局](../development/repository-layout.md)。本仓库维护的一份是
[`adapters/`](https://github.com/Jittor/jittor/blob/master/adapters/README.md)
（发行物 `jittor-torch-adapters`，Python 包 `jittor_adapters`），它有三个 entry
point：`jittor_transformers`（Transformers 4.56.2 / 5.5.3：让它自带的 `torch_npu`
探测返回假）、`jittor_torchmetrics`（TorchMetrics 1.7.4）、`jittor_vllm`。版本是显式
的：未识别的版本会抛 `UnsupportedAdapterVersion` 而不是碰运气。TRELLIS 与 Gaussian
Splatting 的粘合代码不在本仓库，也没有随本发布线发布。

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

## 报错

算子失败会给出真实原因（算子类型、输入形状/dtype、`[Reason]`），而不是旧的
"Wrong inputs arguments / help(jt.sync)" 噪声。不支持的 dtype（例如昇腾上的 float64）
抛出干净的 Python 异常而不是直接中止。异步算子失败时设 `JT_SYNC=1` 精确定位。

## torch API 覆盖（对真实 PyTorch 验证）

下列每一项都在 **CPU 与 CUDA 上**与真实 PyTorch 用相同输入/权重**逐位（或到约 1e-6）
对拍**过，并锁进回归套件（`compat/tests/torch/` 下的 `test_torch_compat.py`、
`test_torch_compat_linalg.py`、`test_torch_compat_distributions.py`、
`test_torch_hf_models.py`、`test_peft.py`、`test_diffusers.py`）。参照版本见上面
「已验证的模型覆盖」里的说明。

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

**两种后端上都已完成并验证**：上面列出的那批 transformers（解码器/编码器/
编码器-解码器/音频/视觉/MoE，其中 33 个架构锁在
`compat/tests/torch/test_torch_hf_models.py` 的回归套件里，其余来自手工对拍）加 CNN
加 diffusers 生成模型的前向/反向/训练精度一致性、设备分派、
bf16 与混合精度、无需 mpirun 的 DDP、梯度检查点、checkpoint/safetensors 迁移、
`model.save()`/`load()`、真实的 `torch.cuda` 显存报告、复数与 `torch.fft.*`、
`F.multi_head_attention_forward`、`torch.func`（functorch 系列，与真 torch 逐位一致）、
`nn.utils.weight_norm`/`spectral_norm`（真实重参数化：`weight` → `weight_g`/`weight_v`
在前向前重算，σ 用幂迭代；与真 torch 和 `np.linalg.svd` 对拍过）、
`nn.utils.rnn.pad_sequence`、torch 兼容的 `torch.optim.lr_scheduler`（`activate()`
与已部署 shim **两条路径上单一实现**；HF 的 `get_*_schedule_with_warmup` 包装
LambdaLR 并产生与 torch 完全相同的曲线），以及清晰的报错。算子面通过算子级差分对拍
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

**优化器与调度器的已知限制。** Adam/AdamW 调用共享的原生 `adam_update`，SGD、RMSprop 与
Adan 委托各自的原生 step，没有第二套更新公式。`torch.optim.LBFGS` 未实现（构造时报
`NotImplementedError`，保真度登记为 `unimplemented`）；`SGD` 还不接受 `foreach=`/`fused=`
（见已知问题 `KI-COMPAT-008`）。调度器公式保持原样，但**不承诺**完整的 resume 语义，也不
填充被忽略的可选参数；`AveragedModel` 仍以原生 Module 为基类，接受 `use_buffers` 但不做
buffer 平均。这些 API 在保真度报告里登记为 `approximate`。

**NumPy 2.x 与 Python 3.13** 是维护中的兼容路径。Jittor 为数组拷贝选择带版本的
NumPy C-API 入口，避免构造 legacy dtype descriptor，并从 Jittor dtype 推导传输尺寸。
Python 3.13 的 wheel 门禁在 NumPy 2.x 下运行，覆盖 CPU 自检、非 C 连续数组传输和
Python 变量追踪。包接受 3 以下的所有 NumPy 版本。

**不提供的**：`pytorch_lightning` / `lightning`。曾经有一份自研的 Lightning 风格
训练循环（`jittor.lightning`），已于 2026-07 删除（`5d49afc3`）；现在没有 Lightning
兼容层，也没有 `pytorch_lightning` 别名。请直接用 `transformers.Trainer`，或自己写
训练循环。

**进行中（更底层）**：`complex128` 与消除剩余内部复数桥接的原生 kernel、PP/TP、
显存管理器调优、CUDA 11/13 的 wheel 家族（对齐的 CUDA 12.2 栈——cuDNN 8.9.7 或更新
——已通过 `jittor[cuda12]` 提供），以及 triton/tilelang 自定义算子支持。

用户可见的限制与它们在问题总账里的编号集中在[已知限制](../guides/known-limitations.md)；
各后端与 Torch 版本的验证位置见[平台支持](../guides/platform-support.md)。
