# torch_compat — 真实模型上 Torch 兼容层对 PyTorch 的速度

回答一个问题：**把 `import torch` 换成 Jittor 的 Torch 兼容层，同一份模型代码在真实
任务上慢多少（或快多少）。**

和仓库里其他测速设施的分工：

| 设施 | 回答什么 | 规模 |
| --- | --- | --- |
| `benchmarks/`（ASV） | 某个算子/小模型跨提交有没有回归 | 算子、tiny Llama |
| `compat/tests/torch/test_ecosystem_speed.py` | nightly 门禁：单步前向+反向不超过阈值 | 8 层随机配置，只测一步 |
| **本目录** | 经典工作负载端到端的差距：推理、生成、采样、带优化器的训练 | 已发布模型的真实配置 |

手工运行，不被 pytest 收集，不进 CI。

当前的差距表与它的测量条件见
[`docs/results/2026-09-24-torch-compat-real-models.md`](../../docs/results/2026-09-24-torch-compat-real-models.md)。

## 工作负载

全部写在 [`workloads.py`](workloads.py)，是**普通的 PyTorch / Transformers / Diffusers
代码**，两个运行时跑同一份源码，里面不允许按运行时分支——需要 Jittor 专用写法才能跑，
说明兼容层有 bug，应该修兼容层，而不是在这里绕开。

| 名称 | 模式 | 默认 dtype | 内容 |
| --- | --- | --- | --- |
| `qwen3_prefill` | 推理 | bf16 | Qwen3-0.6B，batch 1 × 2048 token 的 prompt 前向，只取最后位置 logits |
| `qwen3_decode` | 推理 | bf16 | Qwen3-0.6B，`generate()` 贪心解码，prompt 128，定长生成 128 token，KV cache |
| `qwen3_decode_static` | 推理 | bf16 | 同上，静态 KV cache：`reduce-overhead` 针对的形状稳定解码 |
| `qwen3_train` | 训练 | fp32 | Qwen3-0.6B，batch 4 × 512，因果 LM loss、反向、AdamW |
| `sd15_sample` | 推理 | fp16 | SD1.5 UNet，512×512，DDIM 20 步，CFG 7.5（每步 batch 2） |
| `sd15_vae_decode` | 推理 | fp16 | SD1.5 VAE，64×64 latent 解码成 512×512 图像 |
| `sd15_unet_train` | 训练 | fp32 | SD1.5 UNet 微调步：DDPM 加噪、ε-MSE、反向、AdamW，batch 2 |
| `ddpm_unet_train` | 训练 | fp32 | diffusers `train_unconditional` 的像素空间 DDPM UNet，64×64，batch 16 |
| `resnet50_infer` | 推理 | fp16 | ResNet-50，batch 64，224×224 |
| `resnet50_infer_b1` | 推理 | fp16 | ResNet-50，batch 1：延迟受逐算子主机开销支配的区间 |
| `resnet50_train` | 训练 | fp32 | ResNet-50，batch 64，SGD momentum |
| `vit_b16_train` | 训练 | fp32 | ViT-B/16 分类，batch 64，AdamW |
| `bert_base_train` | 训练 | fp32 | BERT-base 序列分类微调，batch 32 × 128，AdamW，dropout 0.1 |
| `bert_base_infer` | 推理 | fp32 | BERT-base 序列分类，batch 1 × 128：在线推理 |

权重按已发布的模型配置初始化，但数值**由 NumPy 按固定种子生成**（`deterministic_init`：
矩阵与卷积核取 `N(0, 1/fan_in)`，1-D 的 weight 取 1、bias 取 0），输入同样来自 NumPy：
两侧从同一组数出发，逐步的 loss（推理为输出均值）因此可以对比，而不只是时间。
套件离线可复现，工作量也不变——一步的 FLOPs、形状和 kernel 与权重取值无关。生成类
任务禁用 EOS、强制定长，理由相同。每个任务持有一个 4 个 batch 的池，**每步换一个**：
训练循环每步都见新数据，被捕获重放的一步也必须跟着换（固定输入会掩盖输入拷贝的错误）。
用框架自己的随机数（dropout）的任务标为 `stochastic`，不对比数值。fp32 任务在两侧都开
TF32（`--no-tf32` 关闭）。

`--compile none,reduce-overhead`（或 `max-autotune`）让每个任务再以 `torch.compile` 跑
一遍，两侧同一份代码：训练任务编译整个 `run`（前向、反向、优化器），SD 采样编译 UNet，
静态 cache 解码用 Transformers 自带的 `generate` 编译配置；动态 cache 解码没有形状稳定
的编译形式，编译行记为 SKIPPED。每个编译步前调用
`torch.compiler.cudagraph_mark_step_begin()`（PyTorch 对 CUDA graph 模式的要求）；编译
模式至少预热 6 步，捕获与录制不进计时。

## 环境

需要两个解释器：

- **参考侧**：装有独立二进制 PyTorch 的环境（不是 Jittor 的 shim）。
- **Jittor 侧**：装好本仓库和 `jittor-torch`（`compat/`）的环境，worker 在里面直接
  `import torch`——用户的用法。单元测试用的进程内开关 `JITTOR_TORCH_SHIM=1` 不行：它
  装上了 `torch` 模块，却没有发布 `torch` 的发行元数据，Transformers/Diffusers 会据此判定
  没装 PyTorch、拒绝建模型。runner 用 `PYTHONPATH` 钉住**当前 checkout**。

两侧的 `transformers`、`diffusers` 版本必须相同，结果里会逐行记录双方版本。两个环境
不能共享 site-packages：PyTorch 自带一整套 `nvidia-*` wheel，与 `jittor[cuda12]` 钉住的
CUDA 12.2 组件冲突。参考侧优先选与 Jittor 同一 CUDA 大版本的构建（例如 `+cu128`），
比的才是框架，而不是 cuBLAS/cuDNN 的大版本。

```bash
# 参考侧（示例版本：与 shim 声明的 torch API 版本一致）
conda create -n torch-ref python=3.11
conda run -n torch-ref pip install "torch==2.11.0+cu128" \
    --index-url https://download.pytorch.org/whl/cu128
conda run -n torch-ref pip install transformers==4.56.2 diffusers==0.35.1 numpy==1.26.4 nvidia-ml-py

# Jittor 侧：core 用 compat 模式可编辑安装，再装 jittor-torch（见 agent/manuals/environment.md）
pip install -e ".[cuda12]" --group dev --config-settings editable_mode=compat
pip install --no-deps --no-build-isolation -e compat
pip install transformers==4.56.2 diffusers==0.35.1 nvidia-ml-py
```

## 运行

```bash
python bench/torch_compat/run.py --list
python bench/torch_compat/run.py --size tiny --torch-python $REF/bin/python   # harness 自检，几分钟
python bench/torch_compat/run.py --torch-python $REF/bin/python --gpu 1       # 全部
python bench/torch_compat/run.py --workloads qwen3,sd15_sample --torch-python $REF/bin/python
```

`--torch-python` 也可以用环境变量 `REAL_TORCH_PYTHON`。`--workloads` 接受名字或家族前缀
（`qwen3` 选中三个 Qwen3 任务）。`--runtimes jittor` 只跑一侧，适合改完代码后反复测。
`--batch N` 把选中任务的 batch 在**两侧同时**改成 N——一侧装不下时用它拿到可比的数字，
原规格的 OOM 本身仍是要报告的结果。

每个（工作负载，运行时）都在**全新进程**里跑，显存和 JIT 状态不会串到下一次测量。结果写在
`$JITTOR_LAB_ROOT/_state/bench-torch-compat/<时间戳>/`：`results.json`、`results.md` 和
每次测量的完整日志 `logs/`。Jittor 编译缓存放在同一目录下的 `cache-<size>/`，**不与单元
测试或 ASV 共享**。

```bash
python bench/torch_compat/report.py results.json            # 重新出表
python bench/torch_compat/report.py old.json new.json       # 两次运行对比：Jittor 有没有变快
```

## 怎么读

**比值 = Jittor 耗时 / PyTorch 耗时（每步中位数），小于 1 表示 Jittor 更快。** 与
[`docs/performance/library-standing.md`](../../docs/performance/library-standing.md) 口径一致。

- **时间**：计时步的中位数。预热至少跑工作负载声明的步数，然后一直跑到相邻两步相差不到
  10%（最多再加 20 步）——Jittor 的后几步仍可能碰到首次出现的形状而编译，固定步数的预热
  会让计时区间横跨两个状态。每步结束时两侧都等设备完成：PyTorch 是
  `torch.cuda.synchronize()`；Jittor 先 `jt.sync` 这一步返回的张量、再 `jt.sync_all`，
  让优化器更新之类的惰性工作也在计时区间内完成。
- **jittor 1st step**：第一步的墙钟，主要是 JIT 编译。单列出来，不进比值——它是冷启动
  成本，用户确实会付，但与稳态速度是两个问题。
- **host**：一步里 `step()` 返回前所占的比例。PyTorch eager 是分发与发射，Jittor 是构图
  （执行在同步时）；比例高说明这一步受主机限制。真正的设备忙时要用 `jt.profile`。
- **agree**：两侧逐步数值的最大差，以参考序列的均方根为尺度；训练只比前 3 步（之后两条
  正确的轨迹也会因训练本身的混沌分开）。大于 5e-2 标 `!`。编译对 eager 的同一列在第二张
  表里。
- **compiled**：编译后实际发生了什么——Jittor 的重放与设备图发射次数或拒绝原因，
  PyTorch 的 graph break 数。静默退回 eager 的编译步在时间列里看不出来，在这里看得出。
- **steady**：计时结束时进程占用的显存（NVML）；编译模式下录制会一直持有一步的工作集。
- **mem**：NVML 读到的**进程实际占用的显存峰值**（后台线程每 20 ms 采样），两侧同一把
  尺子，含 CUDA 上下文与分配器缓存；失败（例如 OOM）的行也记录它撞到了多高。各运行时
  自己的 `torch.cuda.max_memory_allocated()` 记在结果 JSON 的 `peak_memory_bytes`，
  只作参考：它不含上下文与缓存，两侧分配器的记账口径也不同。Jittor 的这个值现在由内存池
  在每次分配时记录（此前是调用时采样，22 GB 的训练步只报 0.1–1.3 GB）。没有 NVML
  （`pip install nvidia-ml-py`）时才退回它。
- **状态**：`ERROR`/`OOM`/`CRASH`/`TIMEOUT` 是真实结果，不要删掉。默认禁止 Jittor 的 CPU
  fallback（`backend_fallback="error"`）：CUDA 测量里混进 CPU 回退，数字就不再是同一件
  事；`--allow-fallback` 只用于调查。

测量纪律（与 [`bench/README.md`](../README.md)、
[`docs/performance/benchmarking.md`](../../docs/performance/benchmarking.md) 相同）：

- 选一张空闲的卡（`--gpu`），不要和单元测试或另一个测速抢同一张卡；
- 只把同一台机器、同一 `--size`、同一 dtype 的数字放在一起比；
- 贴出结论时带上提交、GPU 和双方版本——`results.md` 的表头已经写好，直接引用它。
