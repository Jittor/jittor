---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# 把真实的生态库跑在 Jittor 上

[用 PyTorch API 写 Jittor](torch_compat.md) 讲的是**你自己**的代码改成 `import torch` 之后
怎么跑。[把已有 PyTorch 脚本迁到 Jittor](torch_compat_migration.md) 讲的是迁移时要改哪几处。

这一篇是第三种情况，也是最省事的一种：**代码不是你的，是别人的**——`transformers`、
`diffusers`、`peft`、`mmcv` 这些库。你没有改它们的打算，也不该改。

好消息是这类库几乎都是同一副骨架：

```{code-block} python
import torch
import torch.nn as nn

class SomeModel(nn.Module): ...
```

它们**只通过 `torch` 这个名字**接触框架。所以关键只有一句话：

> **让 `import torch` 拿到 Jittor 的实现，库的源码一行都不用动。**

这一篇会：

* 用一个**当场可跑的小实验**证明这句话——写一个「第三方库」，它的源码里写死了
  `import torch`，然后看它在 Jittor 上直接跑起来；
* 说清楚**为什么必须先激活再 import**，以及顺序错了会看到什么；
* 给出主流库（transformers / diffusers / PEFT / mm-swift / mmcv）各自真正需要注意的点，
  全部来自实测；
* 讲清两条**架构性的**边界——有些东西不是「还没支持」，而是这条路走不通；
* 给出验证「这个库在 Jittor 上算得对不对」的方法。

```{code-cell} ipython3
import numpy as np
import jittor as jt

jt.flags.use_cuda = 0
```

## 1. 一个「第三方库」

下面这段源码**原样**就是一个典型第三方库的样子。注意它做的事：模块级 `import torch`，
用 `nn.Module` 定义模型，用 `torch.optim` 训练，用 `F.cross_entropy` 算损失。

把它当作「从 pip 装进来的、你无法修改的包」。

```{code-cell} ipython3
LIB_SOURCE = '''
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyNet(nn.Module):
    def __init__(self, in_dim=4, out_dim=2):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


def make_optimizer(model, lr=0.1):
    return torch.optim.SGD(model.parameters(), lr=lr)


def train_step(model, optimizer, x, y):
    optimizer.zero_grad()
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()
    return float(loss)
'''
print("库的源码 %d 字符，里面写着 import torch" % len(LIB_SOURCE))
```

## 2. 先确认 `import torch` 现在指向谁

激活做的事是**替换 `torch` 这个名字指向的模块**。所以第一件该确认的事是：在你手上这个
环境里，`import torch` 现在到底拿到什么。

```{code-cell} ipython3
import jittor
import torch as torch_now

is_shim = getattr(torch_now, "__file__", None) is None

print("torch.__version__  :", torch_now.__version__)
print("torch.__file__     :", getattr(torch_now, "__file__", None))
print("jittor.__version__ :", jittor.__version__)
if is_shim:
    print("\n-> 这是 Jittor 实现的 torch：shim 已经装成环境里的 torch")
else:
    print("\n-> 这是真的 PyTorch：库会跑在 PyTorch 上，不是 Jittor")
```

判据是 `__file__`：shim 的 `torch` 在激活时**整个模块对象被换掉了**，所以没有
`__file__`；真的 PyTorch 有一个指向它自己包内 `__init__.py` 的路径。

两种结果对应两种部署方式，**没有哪一种天然更对**，但不确认就有风险：

| `import torch` 拿到 | 怎么来的 | 后果 |
| --- | --- | --- |
| Jittor 的 shim | shim 被装成了环境里的 `torch`，或设了 `JITTOR_TORCH_SHIM=1` | 整个环境都跑在 Jittor 上，库不用改代码 |
| 真的 PyTorch | 环境里装了真 torch，且没有激活 | 库跑在 PyTorch 上——**你以为在测 Jittor，其实没有** |

危险的是第二行。库不会报错，它只是老老实实地跑在 PyTorch 上。所以「要不要显式激活」
不是一个风格问题：**不确认的话，实验到底跑在哪套框架上是不确定的。**

## 3. 激活，然后同一个库原样跑起来

显式激活一次。如果上面已经指向 shim，再调一次也没有副作用（激活是幂等的）。

```{code-cell} ipython3
from jittor.compat.shim import activate
activate()
```

现在把**同一段源码**执行一遍。

```{code-cell} ipython3
import types

lib = types.ModuleType("third_party_lib")
exec(compile(LIB_SOURCE, "<third_party_lib>", "exec"), lib.__dict__)

print("库拿到的 torch 版本:", lib.torch.__version__)

model = lib.TinyNet()
optimizer = lib.make_optimizer(model)

# 关键证据：库创建的参数到底是什么类型
param = model.fc.weight
print("\n库创建出来的参数类型:", type(param).__module__ + "." + type(param).__name__)
print("它是 Jittor 的 Var 吗:", isinstance(param, jt.Var))
```

最后一行是整件事的核心。库的源码里写的是 `nn.Linear`，但它拿到的 `nn` 来自 Jittor，
所以**库在不知不觉中造出了 Jittor 的参数**——不只是「长得像」，而是**就是**
`jittor.Var`。再看它能不能真的训练。

```{code-cell} ipython3
rng = np.random.RandomState(0)
x = lib.torch.tensor(rng.randn(32, 4).astype("float32"))
y = lib.torch.tensor(rng.randint(0, 2, 32))

before = model.fc.weight.numpy().copy()
for step in range(60):
    loss = lib.train_step(model, optimizer, x, y)
after = model.fc.weight.numpy()

print("60 步之后损失: %.4f" % loss)
print("权重更新了吗:", not bool(np.allclose(before, after)))
print("权重变化量:", float(np.abs(after - before).max()))
```

**没有改库的一行代码**，它的 `forward` / `zero_grad` / `backward` / `step` 全部生效了。
这就是生态库能被带起来的原因：它们调用的是 `torch` 这个名字，而不是某个具体的实现。

## 4. 顺序是唯一真正的要求

激活做的事是**替换 `torch` 这个名字所指的东西**。这意味着：

```{code-block} text
正确： activate()  ->  import 库         库看到 Jittor
错误： import 库   ->  activate()        库已经把真 torch 记在自己的模块里了
```

第二种情况下，之后再激活也救不回来——库模块已经完成了 `import torch`，它内部的全局名
早就指向别处了。所以激活要放在**尽可能早**的位置：入口脚本的第一行，或者干脆用环境变量
让整个进程生效。

```{code-block} bash
# 整进程生效：不用改任何代码，库自然拿到 Jittor
JITTOR_TORCH_SHIM=1 python your_train_script.py
```

这是接入生态库**最省事**的方式：写一个包装脚本，或者直接在命令行前加环境变量。

## 5. 各个库需要注意什么

下面是实测过的结论。**没有一条需要改库的源码**，都是环境或调用方式上的注意点。

### 5.1 通用环境要求

```{code-block} bash
export HF_DEACTIVATE_ASYNC_LOAD=1   # 单线程权重实体化（工作线程没有设备上下文）
export HF_HUB_OFFLINE=1             # 离线运行
export DISABLE_VERSION_CHECK=1      # 跳过版本检查
```

**最重要的一条：不要让 pip 把真 `torch` 装进 shim 环境。** 它会覆盖 `torch/__init__.py`，
把整个兼容层顶掉。真 PyTorch 如果想要，放在**单独的环境**里当对拍参考。

`trl` 这类带 `torch` 依赖的包要加 `--no-deps` 安装，否则会把真 torch 拉进来。

### 5.2 transformers

```{code-block} python
from jittor.compat.shim import activate
activate()
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
out = model.generate(**tokenizer("你好", return_tensors="pt"), max_new_tokens=16)
```

已验证：约 30 个架构的前向与反向都与真实 PyTorch 吻合到约 `1e-6`；`Trainer` 能微调
（loss 下降、`clip_grad_norm_` 生效）；`generate()` 支持贪心、beam search、
temperature/top-k/top-p 采样和批量生成，其中**贪心解码与「每步从头重算」逐位一致**——
这一条是 KV cache 正确性的硬证据。

### 5.3 diffusers

```{code-block} python
from diffusers import UNet2DModel, DDIMScheduler
# 用构造函数 / from_config 建模型；加载预训练 checkpoint 见下面的边界说明
unet = UNet2DModel(sample_size=32, in_channels=3, out_channels=3,
                   layers_per_block=2, block_out_channels=(32, 64))
```

`UNet2DModel` 前向 `1.1e-6` / 反向 `1.5e-6`，DDIM 去噪循环 `3e-5`，`AutoencoderKL`
编码+解码 `1.4e-6`——**也就是说 Jittor 真的在生成**，且数值与 PyTorch 吻合。

### 5.4 PEFT / ms-swift

LoRA 这两个都在门禁里有对拍用例（llama 上的 LoRA）。ms-swift 需要 `peft < 0.20`；
它自己那个版本的组合在**真 PyTorch 下也报同样的 `TypeError`**——那是上游的兼容问题，
不是 Jittor 引入的。

### 5.5 mmcv / mmengine

`mmcv.cnn.ConvModule` 和 `mmengine.model.BaseModule` 在 CPU / CUDA / NPU 上都对拍过。
但**只有纯 Python 层可以**，原因见下一节。

## 6. 生态库是并列的：一个库一个仓库

前面几节讲的是「怎么把别人的库带起来」。这一节讲**这些库现在放在哪**——因为从
2026 年 10 月起，主仓库不再携带任何 adapter：`adapters/` 目录已经整体迁出，主仓库里只留
`jittor` 本体与 `jittor.compat.torch` 这一层共享前端。

每个下游库因此有自己的仓库（都在 [JittorEcosystem](https://github.com/JittorEcosystem) 组织下），
负责三件主仓库不该承担的事：

1. **pin**：接受哪些上游版本（`compatibility.json`），以及对应的 `jittor-compat` 版本；
2. **adapter**：需要补的补丁（入口点 `jittor.module_patches`），不需要补的就不带；
3. **验证记录**：官方测试的子集、示例，以及「哪些在什么设备上真的跑过」。

它们**布局是同一套**：`README.md` + `README.en.md`、`AGENTS.md`、`compatibility.json`、
`docs/`、`examples/`、`tests/`、`support/`。所以读一个仓库的经验可以平移到下一个——
**flash-attention 和 MiniMax-H3 不是特例，它们和 transformers、vllm 是并列的**。

| 库 | 仓库 | 钉住的上游 | 备注 |
| --- | --- | --- | --- |
| transformers | [jittor-transformers](https://github.com/JittorEcosystem/jittor-transformers) | 5.9.0（准入 4.56.2 / 5.5.3 / 5.9.0 / 5.13.0） | `AutoModel` / `AutoTokenizer`；官方测试子集 |
| vllm | [jittor-vllm](https://github.com/JittorEcosystem/jittor-vllm) | 2.0.0.dev6 | 推理服务适配；含部署与批处理流程 |
| torchmetrics | [jittor-torchmetrics](https://github.com/JittorEcosystem/jittor-torchmetrics) | 1.7.4 | 指标库；bounded helper 的补丁 |
| deepspeed | [jittor-deepspeed](https://github.com/JittorEcosystem/jittor-deepspeed) | 0.19.7 | `deepspeed.initialize` 与 Engine |
| ms-swift | [jittor-ms-swift](https://github.com/JittorEcosystem/jittor-ms-swift) | 4.5.2 | `swift sft` / `swift rlhf` |
| verl | [jittor-verl](https://github.com/JittorEcosystem/jittor-verl) | 0.9.1 | RL 训练栈 |
| trl | [jittor-trl](https://github.com/JittorEcosystem/jittor-trl) | 0.29.1 | 官方 trainer API |
| tensordict | [jittor-tensordict](https://github.com/JittorEcosystem/jittor-tensordict) | 0.10.0 | `TensorDict` / `tensorclass` |
| datasets | [jittor-datasets](https://github.com/JittorEcosystem/jittor-datasets) | 5.0.1 | 数据加载 |
| tokenizers | [jittor-tokenizers](https://github.com/JittorEcosystem/jittor-tokenizers) | 0.22.2 | 分词器 |
| safetensors | [jittor-safetensors](https://github.com/JittorEcosystem/jittor-safetensors) | 0.8.0 | 序列化格式 |
| accelerate | [jittor-accelerate](https://github.com/JittorEcosystem/jittor-accelerate) | 1.10.1 | 无需 adapter：官方测试与示例验收 |
| ray | [jittor-ray](https://github.com/JittorEcosystem/jittor-ray) | 2.48.0 | 无需 adapter：同上 |
| flash-attention | [jittor-flash-attn](https://github.com/JittorEcosystem/jittor-flash-attn) | 2.7.4.post1（commit `5231d95fe1`） | 官方 CUDA kernel 的桥；**首次使用要编译**，部署时预热 |
| MiniMax-H3 | [jittor-minimax-h3](https://github.com/JittorEcosystem/jittor-minimax-h3) | vLLM-Omni `v0.29.0rc1-58-g446c2b54` / vLLM 0.29.0 | 服务启动参数与客户端协议；需要 8×H100 级显存 |

不在这个清单里的库（diffusers、PEFT、mmcv/mmengine 等）按前面几节的通用路径走：
它们的接口落在 `jittor.compat.torch` 的公开 API 上，不需要额外的补丁仓库；这也意味着
**没有版本 pin 的保证**——跟着 `jittor-compat` 的版本走。

### 装一个生态库

```{code-block} bash
# 适配类：装的是它自己的包，入口点会自动注册到 jittor
python -m pip install "jittor-transformers @ git+https://github.com/JittorEcosystem/jittor-transformers"

# 验证类（accelerate / ray）：装的是官方库本身，仓库提供的是测试与示例
python -m pip install accelerate==1.10.1
```

装完先看它声明的 pin 与你的环境是否对得上——每个仓库的 `compatibility.json` 都写了
上游版本、`jittor-compat` 版本，以及哪些结论来自真实设备执行、哪些还没有。

### 出问题该改哪一层

这是接入生态库时最容易走错的一步。判断只有三行：

1. 换个库也错的语义问题（张量、求导、设备、精度）→ `jittor` 本体或 `jittor.compat.torch`；
2. 只有这个库错、且错在它**依赖的私有 helper** 上 → 该库自己的生态仓库；
3. 只有在这个库的某个模型上错、换个模型就好 → 先缩成最小复现，再按 1 还是 2 分。

不要为了绕过某个库的问题去改它的源码：那是它下一次升级就会丢掉的补丁，而且会把
「兼容层没做到」伪装成「已经支持」。

## 7. 两条走不通的路

这两条属于**架构性的**，不是「还没做」。分清楚这一点能省很多时间。

### 6.1 编译过的 PyTorch 扩展

`mmcv.ops` 以及任何自带链接 `torch` 的 `.so` 的包，都是针对 **PyTorch 的 C++ ABI**
编译的。Python 层的兼容层替换的是 `torch` 这个名字，**碰不到已经编译好的二进制**。

这类算子需要 Jittor 自己的实现——`jittor.models` 和原生算子面就是为这些场景准备的。
判断方法很简单：**看这个包有没有编译产物**。

```{code-block} text
纯 Python（能用）        : transformers、diffusers、peft、mmengine、mmcv.cnn
带 torch 链接的 .so（不能用）: mmcv.ops、flash-attn、apex、xformers
```

### 6.2 自己掌管设备的运行时

vLLM 内嵌自定义 CUDA kernel，还有自己的显存管理和调度层，**它不调用 `torch.*` 去干活**。
所以光靠替换 `torch` 这个名字带不动它。

这类库走的是**外部适配器**的路：由适配器针对 Jittor 提供那些 kernel。
已验证 vLLM V1 能加载 Qwen3-0.6B、建立 KV cache、在真实 CUDA 上完成贪心解码，
产出 token 与真实 PyTorch/Transformers **完全一致**。

结论：**先看库的形态，再决定走哪条路。**

| 库的形态 | 能不能直接用兼容层 |
| --- | --- |
| 纯 Python，只用 `torch.*` | 能，激活即可 |
| 带编译好的 torch 扩展 | 不能，需要 Jittor 侧实现 |
| 自带 kernel + 设备管理 | 不能，需要外部适配器 |

## 8. 怎么知道它算得对

「跑起来了」和「算对了」是两件事。生态库的正确性验证用的是一个**双跑对拍**的方法：
同一个用例跑两遍——一次在 `torch` 是真 PyTorch 的解释器里，一次在 shim 解释器里——
从**相同的权重**出发，比较：

1. 前向输出；
2. **每一个参数的梯度**；
3. **输入梯度**；
4. CPU 与 CUDA 都要比。

比较梯度而不是只看输出，是因为有些错误（比如梯度算错、某条分支没接进计算图）前向完全
看不出来，只在训练时会静默地学坏。

在 Jittor 上自己验证一个库时，可以按这四个轴走：

| 轴 | 问题 | 怎么量 |
| --- | --- | --- |
| 精度 | 数值对吗 | 与真 PyTorch 比前向 + 每个参数的梯度 |
| 显存 | 峰值得住吗 | 比**活跃**显存，不要比池子大小 |
| 速度 | 慢多少 | 交错多次取最小值，别信单次采样 |
| 模型表 | 覆盖够吗 | 逐架构跑一遍，记录哪些通过 |

## 9. 检查清单

```{code-block} text
1. 激活要最早           —— 在库 import torch 之前；或直接用 JITTOR_TORCH_SHIM=1
2. 别让 pip 装进真 torch —— 它会覆盖 shim；对拍用的真 torch 放独立环境
3. 先判断库的形态        —— 纯 Python 能用；带 .so 或自带 kernel 不能
4. 梯度也要对拍          —— 只比前向会漏掉静默的学习错误
5. 显存比活跃值          —— 比池子大小会得出错误结论
6. 速度交错多次取最小     —— 单次采样在这个栈上噪声极大
7. 上游的错先在上游确认   —— 真 PyTorch 下同样报的错不是 Jittor 的问题
```

第 3 条是最能省时间的一条。看到一个库跑不起来，先花三十秒确认它有没有编译产物；
如果有，那么问题不在你的环境配置上，继续调是浪费时间。

## 接下来

* [用 PyTorch API 写 Jittor](torch_compat.md)：激活、张量、训练的完整入门。
* [把已有 PyTorch 脚本迁到 Jittor](torch_compat_migration.md)：迁移时必须改的几处。
* [混合精度](mixed_precision.md)：精度策略在两种前端下的写法。
* 各库的完整实测结论与门禁状态，见维护者文档
  [用 PyTorch API 写 Jittor](https://github.com/Jittor/jittor/blob/2.0-refactor/docs/compatibility/torch.md)。
* 回到 [教程总目录](README.md)。
