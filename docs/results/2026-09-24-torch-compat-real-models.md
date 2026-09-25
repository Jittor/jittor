# Torch 兼容层在真实模型上对 PyTorch：差距表、显存与性能修复

- 状态：第三版（显存、融合注意力、fused AdamW 与主机开销修复后）。套件
  （[`bench/torch_compat/`](../../bench/torch_compat/README.md)）可复用。修复与本报告一同
  提交；表中「修后」来自提交前 `f1e3dfa9` 加这些修改的工作区（结果文件标 dirty），与提交
  内容一致。
- 日期：2026-09-24
- 基线提交：`f1e3dfa9`
- 硬件：RTX 4090 24 GB（驱动 580.178.04），AMD EPYC 9554
- 对照：PyTorch 2.11.0+cu128（cuDNN 9.19），Transformers 4.56.2，Diffusers 0.35.1，
  两侧 NumPy 1.26.4；Jittor 侧 nvcc 12.2.140 + `jittor[cuda12]` 组件（含
  `nvidia-cudnn-frontend` 1.30），shim 声明的 torch API 为 2.11.0
- Owner：Jittor 核心维护者
- 复查触发：执行器释放时机、反向构图、图重放、SDPA 分派、优化器、placement/类型提升
  变化；换卡或换对照版本

## 测的是什么

同一份普通 PyTorch 代码（HF Transformers / Diffusers / 手写 ResNet-50），在两个
解释器里各跑一遍：一个的 `torch` 是独立的二进制 PyTorch，一个的 `torch` 是装了
`jittor-torch` 之后的 Jittor。**比值 = Jittor 耗时 / PyTorch 耗时**，小于 1 表示
Jittor 更快，与[下游库性能台账](../performance/library-standing.md)口径一致。

权重按已发布配置随机初始化（参数量与官方一致：Qwen3-0.6B 5.96 亿、SD1.5 UNet 8.60
亿、VAE 0.84 亿），生成类任务禁 EOS、定长。fp32 任务两侧都开 TF32。每个（任务，
运行时）独立进程；时间是计时步的中位数；显存是 NVML 读到的**进程实际占用峰值**
（两侧同一把尺子，含 CUDA 上下文与分配器缓存）。

```bash
python bench/torch_compat/run.py --torch-python $TORCH_REF/bin/python --gpu 1
python bench/torch_compat/run.py --runtimes jittor --workloads qwen3_train,vit_b16_train
```

## 结果

默认规格。PyTorch 与「Jittor 修前」来自 `full-2`，「Jittor 修后」来自 `round2`，测量期间
机器上没有其它负载。

| 任务 | 规格 | PyTorch | Jittor 修前 | Jittor 修后 | 比值 | 显存 PT / JT 修前 → 修后 |
| --- | --- | --- | --- | --- | --- | --- |
| `qwen3_prefill` | bf16，1×2048 | 36.2 ms | 52.6 ms | 40.4 ms | 1.12x | 1.7 / 3.3 → 1.6 GB |
| `qwen3_decode` | bf16，prompt 128，生成 128 | 2.15 s | 5.88 s | 4.73 s | **2.20x** | 1.6 / 6.0 → 1.6 GB |
| `qwen3_train` | fp32，4×512，AdamW | 267.6 ms | OOM | 237.2 ms | 0.89x | 20.0 / >22 → 17.4 GB |
| `sd15_sample` | fp16，512²，DDIM 20 步，CFG | 438 ms | 1588 ms | 1211 ms | **2.76x** | 2.4 / 5.7 → 3.1 GB |
| `sd15_vae_decode` | fp16，64²→512² | 34.0 ms | 36.0 ms | 37.5 ms | 1.10x | 1.7 / 7.0 → 2.4 GB |
| `sd15_unet_train` | fp32，batch 2，AdamW | 270.6 ms | OOM | 273.3 ms | 1.01x | 17.1 / >22 → 15.9 GB |
| `ddpm_unet_train` | fp32，64²，batch 16，AdamW | 66.6 ms | 199.9 ms | 104.0 ms | **1.56x** | 4.4 / 6.9 → 4.4 GB |
| `resnet50_infer` | fp16，batch 64 | 14.3 ms | 12.7 ms | 12.3 ms | 0.86x | 1.1 / 1.7 → 1.2 GB |
| `resnet50_train` | fp32，batch 64，SGD | 91.5 ms | 98.5 ms | 98.3 ms | 1.07x | 6.5 / 10.4 → 7.1 GB |
| `vit_b16_train` | fp32，batch 64，AdamW | 159.2 ms | OOM | 182.2 ms | 1.14x | 9.0 / >22 → 9.4 GB |
| `bert_base_train` | fp32，32×128，AdamW | 54.8 ms | 84.8 ms | 62.9 ms | 1.15x | 4.5 / 10.0 → 5.7 GB |

11 个任务全部跑通，比值几何平均 **1.26x**（修前 8 个能跑的任务是 1.69x）。
Jittor 的首步（冷编译）在 30 秒到 7 分钟之间，缓存热了之后 1.1–11.5 s；它不进比值，
但用户确实会付。

## 读法

- **显存**：修前三个训练任务在 24 GB 卡上装不下；修后全部装下，各任务进程峰值在
  PyTorch 的 0.87–1.45 倍，三个训练任务（`qwen3_train`、`sd15_unet_train`、
  `ddpm_unet_train`）不高于 PyTorch。
- **持平或更快**：`qwen3_train` 0.89x、`resnet50_infer` 0.86x、`sd15_unet_train` 1.01x。
- **余下的速度缺口都在主机侧**：`sd15_sample` 20 步的设备时间只有约 350 ms（比 PyTorch
  整个 438 ms 还少），墙钟 1211 ms；`qwen3_decode` 同理。差距来自每步在 Python 里重建
  图（torch 前端每个算子数微秒）和执行器逐算子的规划与发射。见下文「主机开销」。

## 显存：差距从哪来、怎么修的

每一条都先用最小复现量出差距，修复带回归测试，并确认测试修前失败、修后通过。

| 现象 | 根因 | 修复位置 | 回归测试 |
| --- | --- | --- | --- |
| 反向峰值约等于所有中间量之和：8 层 Qwen3 一步 13.07 GB 对 PyTorch 9.17 GB，16 层堆叠的反向多占 23.8 个激活 | `ceae1910` 为修 loader 竞态加的 `batch_hold` 把 batch 里每个 var 钉到整个 batch 结束；一次反向就是一个 batch | `src/core/executor.cc`（`schedule_hold_release`）、`exec_plan.h`、`exec_runner.cc`：按每个 var 在 batch 内最后一次使用的段位置释放 hold | `tests/backends/cuda/test_batch_releases_memory.py`（23.8 → 4.0 个激活） |
| 默认 `auto_flush_ops` 下反向峰值是惰性执行的 5.5 倍（2534 对 460 MB） | 构造反向时 Python `Function.grad` 建的 op 触发自动 flush，反向被切成 16 个 batch，前一段的梯度在后一段开始前就被迫落地 | `src/core/grad.cc`、`src/runtime/submission_pipeline.h`：`GradConstructionScope` 期间不自动 flush | `tests/backends/cuda/test_auto_flush_graph_split.py::TestNoFlushWhileBuildingTheBackward`、`compat/tests/torch/test_backward_peak_memory.py` |
| 交叉熵峰值是 logits 的 4.25 倍，Qwen3 的 15 万词表上尤其大 | 组合实现经 one-hot 与自动微分保留多份 `[N, C]` 中间量 | `python/jittor/nn/functional/loss.py`：`_CrossEntropyRows` 闭式反向，用 reindex 取目标列 | `tests/nn/test_cross_entropy_memory.py`（≤ 2.5 倍，数值对 NumPy） |
| SD1.5 UNet 训练 OOM：64² 分辨率的一次注意力前向后保留 522 MB，PyTorch 5 MB | 数学路径 SDPA 为反向保留 `[..., Lq, Lk]` 的 softmax，自动微分再保留 scores | `python/jittor/nn/functional/attention.py`：需要梯度且无 dropout 时走 `_MemoryEfficientAttention`，只保存 q/k/v/输出，反向重算概率 | `tests/nn/test_sdpa_memory_efficient.py`（闭式数值、推理路径不变、保留量） |
| SD1.5 VAE 解码进程峰值 7.0 GB，PyTorch 1.7 GB；解码本身峰值只有 0.5 GB | 自动图重放（`auto_graph_replay`）用 `keep_graph` 让整张图保持可重跑，同时让每个中间量的显存常驻：第二次调用起多占 6.2 GB | `src/core/exec_runner.cc`：`keep_graph=2` 保留节点、按 batch 内最后一次使用归还中间量显存，下次运行重新分配（存储视图恢复对父缓冲的别名）；`python/jittor/_runtime/graph_replay.py` 的捕获与重放用它。只有录成 CUDA graph 需要固定地址、保留全部缓冲，自动策略只对总分配 ≤ `auto_graph_replay_retain_bytes`（64 MiB）的图录制；`src/mem/`：池新增累计分配计数 | `tests/core/test_graph_replay_retention.py`（修前常驻与峰值都是 128 MiB；逐次换输入校验、含存储视图） |
| SD1.5 采样比 PyTorch 多 0.9 GB 激活 | 推理（无梯度）注意力一次性构造整个 `[..., Lq, Lk]` 分数矩阵和 softmax，64² 分辨率下每次 1 GiB | `python/jittor/nn/functional/attention.py`：分数超过 384 MiB 时按 query 分块（非因果），并把 Kᵀ 作为 cuBLAS 标志传入 | `tests/nn/test_sdpa_memory_efficient.py::TestInferenceAttentionInBlocks` 等 |
| `torch.cuda.max_memory_allocated` 在 22 GB 的训练步上只报 0.1–1.3 GB | 峰值只在 Python 调用时采样，batch 内部的峰值看不到 | `src/mem/allocator/sfrl_allocator.*`、`src/mem/mem_info.*`：按设备记录每次分配的峰值；`compat/torch/installers/cuda/api.py` 读它 | `compat/tests/torch/test_backward_peak_memory.py::TestMaxMemoryAllocatedSeesInsideABatch` |

记忆高效注意力的取舍：

- **重算的代价**：它比保留概率的组合实现多一次 QKᵀ 和一次 softmax。
  - 4096 token 的单次注意力前后向是 16.1 对 12.2 ms。
  - 整模型上 `vit_b16_train` 慢约 10%（179.7 对 162.6 ms），换来少 1 GB。
  - `sd15_unet_train` 则是从装不下到能跑。
- **两个实现细节影响明显**：
  - 转置要作为 cuBLAS 标志传给 batched matmul，不能对中间量调 `.transpose()`。后者会
    整块复制一个 `Lq×Lk` 矩阵，4096 token 时每次 2.5 ms。
  - softmax 要用 max/exp/sum 组合，不能用融合 softmax kernel。后者对融合不透明，
    输入和输出都会落地，SD1.5 UNet 的峰值会从 20.2 GB 升到 21.1 GB 并再次 OOM。

推理侧的取舍：

- **图重放**：VAE 解码修前 36.0 ms / 7.0 GB（重放、全部常驻），只关重放是 45.5 ms；
  重放而中间量按最后使用归还后是 37.5 ms / 2.4 GB。d512 L8 的 decode 步一次只分配
  0.3–3 MiB，仍录成 CUDA graph，每步 0.3–0.4 ms 不变。
- **分块大小**（无融合 kernel 时的数学路径）：设备侧分块不更慢，但会影响流水化的整步。
  在 20 步采样循环里，每次调用分 2–3 块与不分块持平（1528–1534 对 1535 ms），分 4 块和
  8 块则慢到 1649–1685 ms。原因没有分离出来（自动 flush 的切点变了），所以块取得大：
  256 MiB 以内不分，1 GiB 的分数分三块。

## 融合注意力

数学路径把 `[..., Lq, Lk]` 分数矩阵和 softmax 写进显存再读出；PyTorch 在 fp16/bf16 上用
FlashAttention、在 fp32 上用 memory-efficient kernel，都按块在片上算、从不落地分数。
`nn.scaled_dot_product_attention` 现在先试 `nn.fused_attention`：

| kernel | 覆盖 | 4096 token、8 头（SD1.5 64²）| 回归测试 |
| --- | --- | --- | --- |
| cuDNN 融合 SDPA（`backends/cuda/kernels/nn/cudnn_attention_cuda.py` + `cudnn_sdpa.cc`） | fp16/bf16，前向与反向，因果；bool/浮点掩码仅推理（整行被掩的行给 0，与组合实现一致） | 推理 2.92 → 0.53 ms、+401 → +20 MiB；训练一步 15.3 → 2.2 ms、+2079 → +50 MiB | `tests/backends/cuda/test_cudnn_fused_attention.py` |
| fp32 memory-efficient（`fused_attention_f32_cuda.py`，FlashAttention-2 算法的 SIMT 实现） | fp32，前向与反向，无掩码或因果，head_dim ≤ 128 | 推理 6.0 → 2.3 ms、+802 → +40 MiB；训练一步 25.4 → 10.0 ms、+4147 → +110 MiB | `tests/backends/cuda/test_fused_attention_f32.py` |

cuDNN 前端是只含头文件的 C++17，Jittor 的 JIT 是 C++14，所以图构造代码单独编译成带 C 接口的
小库，`jt.code` 算子只调用它；需要 `nvidia-cudnn-frontend`（已加入 `jittor[cuda12]`），
没有时 kernel 拒绝、走原路径。整模型：`sd15_unet_train` 426 → 377 ms、20.2 → 17.0 GB
（再加上下文的 fused AdamW 为 273 ms、15.9 GB）；`qwen3_prefill`（bool 因果掩码）
52.5 → 43.2 ms（再加上主机开销修复为 40.4 ms）。

## 主机开销

修后的差距主要在主机侧：`ddpm_unet_train` 一步设备时间约 40 ms，`sd15_sample` 每个 UNet
步设备约 17 ms 而墙钟约 60 ms。本轮修了三处：

| 现象 | 根因 | 修复 | 回归测试 |
| --- | --- | --- | --- |
| AdamW 一步的主机时间超过整个前向 | torch 前端对每个参数构造约十个图节点，diffusers UNet 450 个参数张量即 4500 个；PyTorch 默认走 foreach 多张量 kernel | `src/ops/composite/fused_adamw_op.cc` 的 CUDA 实现（每次发射最多 36 个张量的参数表）+ `backends/cuda/kernels/optim/fused_adamw_cuda.py`；torch 前端 AdamW 在 `fused`/`foreach` 未显式关闭时默认使用 | `tests/backends/cuda/test_fused_adamw_cuda.py`、`compat/tests/torch/test_adamw_fused_default.py`（与逐参数路径相差 < 1e-5） |
| 每建一个算子都要遍历 Python 栈 | 异步错误定位记录每个 op 的 Python 行号：逐帧按名字取 `f_globals`、解码模块名、计算行号，占训练步主机时间 5–8% | `src/bindings/pybind/py_var_tracer.cc`：按 code 对象缓存「是否 Jittor/torch 内部」，按 (code, 指令偏移) 缓存 origin | `tests/backends/cuda/test_async_error_location.py::test_launch_history_survives_graph_release` |
| torch 前端的 `a * 2.0` 18 µs（原生 4 µs） | 每个二元算子把两侧 dtype 转成 torch dtype 对象再转回，标量还要走完整的 `result_type` | `compat/torch/installers/tensor/method_api.py`：直接读原生 dtype 名；浮点标量配浮点张量、整数标量配非 bool 张量时结果 dtype 即张量 dtype | `compat/tests/torch/test_torch_compat_promotion.py` |

效果：`ddpm_unet_train` 211 → 104 ms，`bert_base_train` 84 → 63 ms，`sd15_sample`
1438 → 1211 ms，`qwen3_decode` 5451 → 4735 ms；torch 前端的 `a + b` 7.8 → 5.8 µs，
`Linear` 35 → 28 µs。余下的主机时间分散在 torch 前端的 Python 包装、执行器的规划与发射、
cuDNN 卷积计划的主机侧执行，没有单一热点。

## 让它们跑起来修掉的缺陷

套件第一次在 Jittor 侧运行时 11 个任务只有 1 个跑通。下面每一条都先有最小复现，
修复带回归测试：

| 现象 | 根因 | 修复位置 | 回归测试 |
| --- | --- | --- | --- |
| conda 的 nvcc 下首次主机编译找不到 `crt/host_config.h` | 只搜 `<cuda_home>/include`，conda 的头文件在 `targets/<target>/include` | `python/jittor/build/compiler.py` | `tests/build/test_cuda_library_search_dirs.py` |
| 任何 CUDA kernel 都编不出 `npp.h` | 为 8 个整数常量无条件 include 可选组件 NPP | `src/type/cuda_limits.h` | （由任何 CUDA kernel 编译覆盖） |
| fp16/bf16 `MaxPool2d` 在 CUDA 上 nvcc 报 `min` 歧义 | half 的 `jittor::min/max` 遮住全局整数重载 | `src/type/fp16_compute.h` | `tests/type/test_half_precision_parity.py::TestHalfPooling` |
| 所有 HF/diffusers 模型在 GPU 上 forward 死于 `setitem` 混设备 | `7b0f188f` 让每次 module call 都进入「默认设备=CPU」的环境 placement，forward 里 op 为结果分配的缓冲也落到 CPU | `compat/torch/frontend.py`、`nn_frontend.py` | `compat/tests/torch/test_default_device_is_cpu.py::TestTheDefaultIsOnlyForConstructors` |
| Qwen3 `generate()` 在 CUDA 上 `TypeError: enable_gqa`；`test_torch_compat_attention.py` 在 HEAD 上 9 条失败 | `31fc3b2b` 在 CUDA 注册的 flash SDPA 也会应答 Torch 前端的 ACL 快捷钩子：签名缺 `enable_gqa`，且绕过前端自己的 flash 门槛（短训练走数学路径、后端能力检查与缓存、GQA 紧凑头） | `backends/cuda/kernels/nn/flash_attention_cuda.py`：接受 `enable_gqa`，自身拒绝短训练，并标记 `torch_frontend_loads_directly`；`compat/torch/installers/nn/attention.py` 据此不走钩子；阈值统一由 `compat/shim/backends/flash_attention.training_min_scores()` 给出 | `tests/nn/test_sdpa_cuda_flash_kernel.py`、`compat/tests/torch/test_torch_compat_attention.py`（9 → 0 失败） |
| fp16 SD UNet 静默变 fp32，随后 SDPA 报 q/k/v dtype 不一致 | `half / 1.0` 与 `half * 0-dim fp32` 按两个张量提升；torch 里 Python 标量与 0 维张量只从更高类别参与提升 | `compat/torch/installers/core.py`、`installers/tensor/method_api.py` | `compat/tests/torch/test_torch_compat_promotion.py` |

测试侧的相应调整：
- **测试基础设施：**
  - `tools/run_test_suite.py` 的 CPU 会话只清空旧名 `nvcc_path`，不清规范名
    `JT_BUILD_NVCC_PATH`。于是在导出后者的 shell 里，CPU 门禁会构建 CUDA，再被自己的
    就绪探针拒绝（`tests/structure/test_gate_scope.py`）。
  - `f1e3dfa9` 新增的结果报告没有重新生成 `MANIFEST.in`，结构测试
    `test_manifest_covers_runtime_trees_without_cache_payloads` 在 HEAD 上就是红的。
- **按新的默认设备语义修改的测试：**
  - `7b0f188f` 遗留的 `test_independent_frontend.py` 里两处 CUDA 断言。
  - `test_torch_compat_attention.py` 的两条 `MultiheadAttention` 用例。它们的「cuda」
    一轮原来整段 forward 被压到 CPU 上跑，现在和 torch 一样先把模块和输入 `.to(dev)`，
    真正在 CUDA 上执行。

验证：
- CPU/CUDA `--tier core` 全绿（CUDA：native 128 passed，torch 215 passed）。
- `tests/structure` + `compat/tests/structure` 1494 passed、0 failed。
- CUDA 上 `tests/backends/cuda`、`tests/core`、`tests/nn`、`tests/optim`：1163 passed、
  12 failed，12 条在 `f1e3dfa9` 干净 worktree 上同样失败（见下节）。
- `compat/tests/torch/test_torch_compat_attention.py` 在 CUDA 上 0 失败（HEAD 上 9 条）。

## 顺带发现、未修的问题

这些在 `f1e3dfa9` 干净 worktree 上同样复现，不是本轮引入的：

- **进程内 `JITTOR_TORCH_SHIM=1` 不发布 `torch` 的发行元数据**：Transformers/Diffusers
  据此判定没有 PyTorch。所以测试套件里的 `test_torch_hf_models.py`、`test_diffusers.py`
  在装了这些库的环境里 10 条全部 `ImportError`（没装时它们被跳过，于是从未暴露）。
  用户路径（直接 `import torch`）没有这个问题，本套件走的就是用户路径。
- **CUDA 构建、零可见 GPU 时 `torch.save` 崩溃**（`backend_streams.cc: Invalid stream
  device 0`）：`CUDA_VISIBLE_DEVICES=""` 下复现。
- smoke 层 torch 会话在本机另有 5 条 HEAD 上即失败：两条 `trapz` 用例调用 NumPy 2 才有的
  `numpy.trapezoid`（dev 组钉的是 1.26.4）；autocast conv2d 带 bias 返回 float32；
  Generator 的 pickle 身份断言；C++ 扩展在无 GPU 会话里无法确定 CUDA 架构。
- `Tensor.true_divide` 缺失。
- CUDA 上在 HEAD 即失败的 12 条（本轮逐条在干净 worktree 复核）：
  - `test_async_error_location.py` 两条：注入的非法地址写不报或不带 launch 历史。
  - `test_cublas_test_op.py` 三条、`test_cudnn_op.py` 两条、`test_cuda_op_capabilities.py`
    两条、`test_backend_teardown.py` 一条。
  - `tests/optim/test_optim_core.py::...test_low_precision_parameter_and_state_dtypes_are_preserved`
    （执行器拓扑序不变量 `queue.size() == roots.size()` 失败）。
  - `tests/nn/test_conv_parity.py::...test_the_generic_paths_do_reproduce_bit_for_bit`。
- torch 前端优化器测试在 HEAD 即失败的 2 条：`test_torch_compat_optim.py` 的 Adan 步
  混设备、`normal_` 初始化后不在 CUDA 上。
- 显存上仍可继续压的点：
  - 没有融合 kernel 覆盖的训练注意力（带掩码、head_dim > 128、CPU）仍用重算概率的
    Function，其反向同时持有约 4 个 `Lq×Lk` 的 fp32 临时量。
  - 权重绑定的词嵌入梯度保留 3 份词表大小的副本。
  - SD 推理进程峰值里分配器缓存与上下文的部分（`sd15_sample` 3.5 GB 中活跃只有 2.0 GB）。
  - 因果注意力的推理不分块（掩码按整段 query 布局）；长序列因果 prefill 仍会构造完整分数矩阵。

## 边界

- 每个配置只测一轮；修后数字在空闲机器上测得。同一代码的 `sd15_sample` 在不同进程间测到
  过约 ±5% 的波动，比值的最后一位不可靠。主机开销敏感的任务对同机负载很敏感：一轮与单元
  测试并发的测量把 `ddpm_unet_train` 测成 426 ms，单独复测是 212 ms。
- 随机权重：速度与权重取值无关，但本报告不含任何数值对拍；正确性由各自的单元测试
  与生态对拍门禁负责。
- 原始结果 JSON 与日志未版本化，位于 `$JITTOR_LAB_ROOT/_state/bench-torch-compat/`。
  `round2/` 在；`full-1/`、`full-2/` 及中间各轮在机器重启时随内存盘丢失，表中 PyTorch 与
  「修前」列取自本报告第一版记录的数字。
