---
name: jittor-transformers-perf
description: Reproducible CUDA and Ascend performance analysis for Transformers, Diffusers, and vision inference on Jittor's Torch frontend versus real PyTorch. Use for operator, Transformer-block, HF-model, full diffusion generation, SAM masks, SDPA, gradient-management, or optimizer-step benchmarks where Jittor lazy execution and backend/version isolation must be handled correctly.
---

# Jittor Transformers 性能基准

本 skill 的可复用源码位于 `scripts/`，运行状态与结果默认写到
`$JITTOR_LAB_ROOT/jittor_transformers_perf/`（未设置时为仓库同级的 `jittor-lab/`）。
目标是得到能解释、能复现的
Jittor vs 真 PyTorch 数据，而不是把 JIT、H2D 或 lazy graph 漏执行当成性能结论。

## 环境

- Jittor：`<jittor-python>`（Python 3.11，跑仓库源码 `python/jittor`）。
- PyTorch oracle：`<real-torch-python>`，独立的真 PyTorch。运行前必须断言 `torch.__file__`
  不含 Jittor shim。
- 所有命令通过 `agent/skills/jittor-transformers-perf/scripts/run_perf_env.sh`，
  并为并行任务设置唯一 `cache_name` 与 `CUDA_VISIBLE_DEVICES=<gpu>`。缓存、日志和
  结果必须留在 lab 工作区，不得写入 Jittor 主仓库。需要指定 CUDA toolkit 时设
  `JTCUDA=<cuda-root>`（含 `bin/nvcc` 与 `lib64`）；不设则由 Jittor 自行发现 nvcc，设错直接退出。

## 不可省略的计时规则

1. 预分配多个不同输入 slot；计时前触碰所有 slot，排除首次 H2D residency。
2. 计时内保留每次输出；训练还要保留每次梯度，最后统一同步。只保留最后一个输出会让
   Jittor lazy executor 漏执行前面的迭代，产生假快。
3. 首次 JIT/extension 编译不计入稳态；同时记录 cold build 时必须单独标注。
4. SDPA 必须区分 PyTorch `default`、forced `math` 和 forced `flash`。Jittor math
   fallback 不能直接与 PyTorch default fused kernel 归因成“softmax 慢”。
5. 训练结果必须检查每个目标梯度 finite 且非零。显式 cuBLAS op 若缺反向，会产生看似
   很快的静默零梯度。
6. HF 对比必须使用同一 Transformers 版本。`benchmark_hf_tiny_models.py` 从
   `JITTOR_PERF_RT_SITE`（真 PyTorch/torchvision 的 site-packages）固定 torch，再从
   `JITTOR_PERF_JT_SITE`（装 Transformers 4.56.2 的 site-packages）加载 Transformers；
   两个变量都没有默认值，未设置时脚本直接退出。
7. allocator 指标只能作为 harness 工作集方向性数据；严格峰值需另用 NVML/进程级采样。
8. SDPA 同时报告 `--sync-mode per_call` 延迟与 `queued` 吞吐。前者每步同步，后者保留
   全部输出/梯度后统一同步；二者不可互相替代。per-call 结果还分别记录 graph/build
   与 sync 中位数，用于区分 Python/grad 构图和设备执行开销。

## 工具路由

- 原生与独立 Torch frontend 的 NPU 矩阵乘：`benchmark_npu_frontends.py`（见下节）
- 通用算子初始基线：`benchmark_transformer_bottlenecks.py`
- forward/backward 算子与 Transformer block：`benchmark_training_hotspots.py`
- 同版本 Tiny Llama/BERT/ViT：`benchmark_hf_tiny_models.py`
- GELU/softmax kernel 与 build/profile：`probe_softmax_gelu.py`
- 大词表、LayerNorm 边界正确性：`audit_kernel_changes_gpu.py`
- clip/scaler 消融：`probe_clip_grad_norm.py`、`benchmark_grad_management.py`
- 完整 SGD/AdamW step：`benchmark_optimizer_step.py`
- SDPA layout/Flash 物化：`probe_sdpa_layout_materialization.py`
- Ascend Qwen3 整模推理：`benchmark_qwen3_ascend.py`
- Ascend Qwen3 两进程结果对拍：`compare_qwen3_ascend.py`
- Ascend SDXL / Wan 完整生成与图像、视频对拍：`benchmark_diffusion_ascend.py`
- Ascend SAM2 真实图像、点提示与掩码对拍：`benchmark_sam2_ascend.py`
- Ascend arg-reduce 的 value / index 不变量：`probe_acl_arg_reduce.py`
- norm / normalize 的 NaN、Inf、溢出与 epsilon 对拍：`probe_acl_vector_norm.py`

### 生成与视觉效果验收

两个入口均要求先完成同版本、同 checkpoint、同输入的小模型 CPU 对拍，再在真实
NPU 验收。CPU fixture 只证明对应路径的兼容性；不能用其时间替代完整预训练模型
的速度，也不能由一次真实图像成功推断整套视觉任务质量。当前实验使用
Diffusers 0.35.1 / Transformers 4.56.2；比较进程必须固定相同版本。首次 JIT 串行，
原生 Torch 与 Jittor 分进程、分缓存顺序运行，设备由外部
`ASCEND_RT_VISIBLE_DEVICES` 指定，运行前保留驱动、CANN 和设备负载快照。

工具不自动下载模型或图片。显式传入 `--model <local-checkpoint>`，或
`--models-manifest <json>`；manifest 按 `sdxl`、`wan`、`sam2` 映射到
`{"path": "...", "repo": "...", "sha": "..."}`，相对路径相对 manifest 所在目录。
`JITTOR_LAB_ROOT` 必须设置，输出和 SAM 共享输入必须在该 lab 下且位于 checkout
外。提交元数据默认取脚本所在 checkout；复制脚本运行时用 `--source-root` 指定。

```bash
SCRIPT_ROOT=agent/skills/jittor-transformers-perf/scripts
perf_run="$JITTOR_LAB_ROOT/_state/generative-vision/<run>"
JITTOR_TORCH_SHIM=0 <real-torch-python> "$SCRIPT_ROOT/benchmark_diffusion_ascend.py" \
  --backend torch --case sdxl --models-manifest "$MODEL_MANIFEST" \
  --height 1024 --width 1024 --steps 20 --guidance 5 --warmups 3 --samples 5 \
  --output "$perf_run/sdxl-torch.json"
JITTOR_TORCH_SHIM=1 JITTOR_HOME="$perf_run/jittor-cache" cache_name=diffusion \
  <jittor-python> "$SCRIPT_ROOT/benchmark_diffusion_ascend.py" \
  --backend jittor --case sdxl --models-manifest "$MODEL_MANIFEST" \
  --height 1024 --width 1024 --steps 20 --guidance 5 --warmups 3 --samples 5 \
  --output "$perf_run/sdxl-jittor.json"
python "$SCRIPT_ROOT/benchmark_diffusion_ascend.py" \
  --compare "$perf_run/sdxl-torch.json" "$perf_run/sdxl-jittor.json" \
  --output "$perf_run/sdxl-parity.json"
```

Diffusion 入口使用同一 NumPy 初始噪声和实际 pipeline，保存最终 latent、像素、PNG，
Wan 另存逐帧 NumPy 和 GIF。测速两侧可加 `--timing total`，只在整次生成前后同步；
callback 只检查步数、保留最后 latent，不安装内部 forward hook，也不报告分段耗时。
默认 `--timing stages` 的 hook 会绕过 Jittor 自动重放，因此分段诊断不能代替
无内部 hook 的总耗时比较。Jittor 可显式设置 `--auto-graph-replay-input-mib 8`；
这只调整输入准入门槛，保留现有 recording 内存预算，不能仅凭此参数宣称启用 device graph。
默认单次无预热明确标为 cold；稳态须显式指定完整生成
预热和样本数，不能把首次 JIT 算入稳态。计时包括文本编码、完整去噪循环、VAE 解码
与 NumPy 后处理，排除加载和文件写入；`stages` 的公开 hook 与每步 callback 均同步，阶段数值
属于带同步插桩的完整生成耗时。默认 SDXL 为 1024²、20 Euler 步、CFG 5；
Wan 为 320×512×17 帧、20 FlowMatchEuler 步、CFG 5，较低分辨率结果不能标为
704×1280。单 prompt / seed 的数值误差与 PSNR 衡量实现间一致性，没有真值质量评分。
`--scheduler euler` 为默认值，Wan 显式将 checkpoint 的 `flow_shift` 映射到
FlowMatchEuler 的 `shift`，两侧记录实际类与配置。`--scheduler native` 保留 checkpoint
调度器；例如 UniPC 的线性求解可能触发原生 torch_npu CPU 回退，必须单列限制，
不能与完整 NPU 的 Euler 耗时混比。

310P 路径使用 FP16 主体，保留 Wan 官方 FP32 timestep / norm / scale-shift 模块，
RoPE 显式 FP32；Wan VAE 为 FP32，SDXL 按官方 `force_upcast` 解码。
`--pack` 只快照 FP16 exact Linear，保留 FP32 例外并排除会升精度的 VAE；
必须用独立原生结果验证快照前后数值。`--vae-tiling` 单列配置；
Wan 可用 `--vae-tile-size` 指定内部 sample tile，默认 256、stride 192，
128 对应 stride 96。工具记录实际 sample / latent / 输出 tile 几何及当前 shape 是否
触发 tiling；Wan2.2 的 patch 空间可能使输出 tile 大于该参数，不能只凭开关判断生效。
公开 tiling 路径仍须检查实际输出通道数、尺寸和帧数。
固定的 Diffusers 0.35.1 中，Wan2.2 `patch_size=2` 的 tiled decode 绕过
`unpatchify`，入口明确拒绝该组合。显存受限时可先独立准备原生条件：

```bash
JITTOR_TORCH_SHIM=0 <real-torch-python> "$SCRIPT_ROOT/benchmark_diffusion_ascend.py" \
  --backend torch --case wan --models-manifest "$MODEL_MANIFEST" \
  --prepare-prompt-embeddings --output "$perf_run/wan-prompts.npz"
```

此模式只在 NPU 加载 FP32 UMT5 与 tokenizer，通过公开 `WanPipeline.encode_prompt`
保存正负条件及 `.npz.json` 元数据，不加载去噪器或 VAE。之后两侧生成均加
`--prompt-embeddings "$perf_run/wan-prompts.npz"`，不启用 tiling，pipeline 显式不加载
文本编码器和 tokenizer。工具核对 prompt、checkpoint、archive SHA 和 finite，并将
共享条件预先传到设备。此时报告只覆盖 scheduler / 去噪 / VAE 解码与像素后处理，
明确排除文本编码、archive 加载和初始条件传输；不能称为端到端文生视频耗时，也
不能证明 Jittor 文本编码器已经验收。两侧对拍必须使用同一个条件 archive。

`--text-device cpu` 仅供 Wan 显式选择 CPU 文本编码，编码和传输仍计入每次总耗时，
该混合配置不能称为全 NPU 推理。两侧都检查 NPU 驻留，Jittor 使用零回退守卫。

SAM 先用 Pillow/NumPy 生成一次共享输入，显式传入原图像素坐标的正点提示；预处理
按 checkpoint 的 square resize / rescale / normalization 设置，使用一致的 Pillow
bilinear，而非可选 torchvision fast processor。之后两进程读取同一 NPZ：

```bash
python "$SCRIPT_ROOT/benchmark_sam2_ascend.py" --prepare \
  --models-manifest "$MODEL_MANIFEST" --image "$SAM_IMAGE" --point 266 216 \
  --inputs "$perf_run/sam-inputs.npz"
JITTOR_TORCH_SHIM=0 <real-torch-python> "$SCRIPT_ROOT/benchmark_sam2_ascend.py" \
  --backend torch --device cpu --dtype float32 --models-manifest "$MODEL_MANIFEST" \
  --inputs "$perf_run/sam-inputs.npz" --warmups 3 --samples 5 \
  --output "$perf_run/sam-torch-cpu.json"
JITTOR_TORCH_SHIM=1 JITTOR_HOME="$perf_run/sam-jittor-cache" cache_name=sam_effects \
  <jittor-python> "$SCRIPT_ROOT/benchmark_sam2_ascend.py" \
  --backend jittor --device npu --dtype float32 --models-manifest "$MODEL_MANIFEST" \
  --inputs "$perf_run/sam-inputs.npz" --warmups 3 --samples 5 \
  --output "$perf_run/sam-jittor-npu.json"
python "$SCRIPT_ROOT/benchmark_sam2_ascend.py" \
  --compare "$perf_run/sam-torch-cpu.json" "$perf_run/sam-jittor-npu.json" \
  --output "$perf_run/sam-effect-parity.json"
```

当前 310P 实测原生 Torch 的 FP16 MaxPool、FP32 非连续 GELU 路径不可用，因此示例
使用独立 Torch CPU FP32 参考。当前 Jittor NPU 仅图像编码器已通过；完整掩码解码仍
被 ConvTranspose 路径阻塞，上述完整模型命令用于复现与后续验收，尚不能产出成功的
完整掩码对拍。图像编码器可用下文组件工具的 `--case sam2 --dtype float32` 单独验证。
跨设备输出参考不能称为原生 NPU oracle，不能将两者耗时相除形成速度比。
SAM 计时覆盖图像编码器、点提示编码器和多掩码解码器；CPU 预处理、掩码插值和叠加图
写入均在计时外，cold 单列。`--compare <torch.json> <jittor.json> --output <lab.json>`
保存 logits / 分数误差、逐候选二值掩码一致性 IoU 与并排图。模型预测 IoU 是置信度，
实现间掩码 IoU 是对拍指标；二者均不是对标注真值的质量评估。

### 独立前端 NPU 最小基准

`benchmark_transformer_bottlenecks.py` 仍使用历史 `import jittor as torch` 与命名空间
替换，不能证明独立 Torch frontend。新的 `benchmark_npu_frontends.py` 每次只运行一个
模式，compat 显式调用 `activate(independent_namespace=True)`，检查独立 Tensor 身份。
它不通过旧 `run_perf_env.sh` 激活环境；先按昇腾指南加载 CANN，在相同已分配设备上
用以下两个独立进程顺序执行（外部设置 `ASCEND_RT_VISIBLE_DEVICES`）：

```bash
export JITTOR_LAB_ROOT="${JITTOR_LAB_ROOT:-$(cd .. && pwd)/jittor-lab}"
perf_run="$JITTOR_LAB_ROOT/_state/ascend-validation/matmul-$(date +%Y%m%d-%H%M%S)"
perf_script=agent/skills/jittor-transformers-perf/scripts/benchmark_npu_frontends.py
perf_python="${JITTOR_CI_PYTHON:-python3}"
JITTOR_HOME="$perf_run/native-cache" cache_name=npu_matmul_native \
  "$perf_python" "$perf_script" --mode native --output "$perf_run/native.json"
JITTOR_HOME="$perf_run/compat-cache" cache_name=npu_matmul_compat \
  "$perf_python" "$perf_script" --mode compat --output "$perf_run/compat.json"
```

解释器需已安装当前核心和独立 `jittor-torch` 兼容发行版；源码 `PYTHONPATH`
本身不会安装独立包。`jittor.compat.shim` 是安装后的公开入口，不应改为 native alias。
用与测试相同的已安装解释器设置 `JITTOR_CI_PYTHON`；例如 task venv 的 `bin/python`。
先在对应的隔离缓存中只验证导入与身份，再启动计时：

```bash
JITTOR_TORCH_SHIM=0 JITTOR_HOME="$perf_run/compat-cache" \
  "$perf_python" -c 'import jittor as jt; from jittor.compat.shim import activate; activate(independent_namespace=True); import torch; assert torch is not jt and torch.Tensor is not jt.Var; print(torch.__file__, torch.Tensor.__module__)'
```

模板默认 float32、256×256 矩阵、3 组输入、
10 次预热和 50 次计时；用 `--m --k --n --slots --warmup --repeats --seed` 修改。
每组输入先与 NumPy float64 累积的矩阵乘对拍，检查设备驻留，预热后逐次同步计时，
最后再对拍。整个计算处于零回退守卫内；任何 ACL、数值或驻留失败都不输出成功报告。
记录每次延迟、median、p95、回退计数、设备快照、Python/包来源、提交和 dirty 补丁
摘要；原始 JSON 只允许写到 lab 下且在 checkout 外。保存运行前驱动/固件/CANN 版本
快照，并按 [NPU 记录模板](../../../docs/development/npu-validation-templates.md) 填写报告。

本入口测量单次前向的 Python 构图到同步完成延迟，不含 D2H、首次编译或训练。
保持各组 shape、精度和设备负载一致，多轮交错重跑后再形成比较结论；微基准结果
不能外推到整个模型。此入口的代码审查/语法检查不代表真实 NPU 性能已经验证。

Ascend 整模对拍必须为 Jittor 和原生 `torch_npu` 使用独立 Python 进程、相同
Transformers 版本和相同本地 checkpoint。Jittor 解释器安装独立 `jittor-torch`，或在
已有 Torch 发行元数据的测试环境设置 `JITTOR_TORCH_SHIM=1`；工具检查 shim 标记，
不再要求 `torch is jittor`。加载权重后显式迁移到 NPU；两侧都通过
`ASCEND_RT_VISIBLE_DEVICES` 选择已分配设备。首次 JIT 单列，稳态 prefill 和
generation 在每次样本后同步；单 token 与多 token decode 都要测，不能用一次
generation 成功推断 KV-cache 后续步可用。Jittor 每次 prefill 和 generation 都在
`forbid_backend_fallbacks()` 中执行并同步；原生策略设为 `error`，退出时核对尝试计数，
即使内部吞掉回退异常也会失败，不再依赖日志文字。计时结束后另外各执行一次验证。可用
`--logits-output` 保存末 token logits，再在进程外比较 argmax、Top-K 和全量误差。
Jittor 侧可用 `--profile-output /path/to/report.json` 在所有计时结束后额外执行一次
generation，并保存聚合 profiler row；该次运行不进入稳态样本。profile 不能在
原生 PyTorch backend 使用。`--pipeline-ops` 必须通过成对 A/B 决定，旧图上的最佳
阈值不能直接作为新算子组合的默认值。两侧都可用
`--python-profile-output /path/to/profile.prof` 在计时后额外执行一次 generation，
保存可由 `python -m pstats` 读取的 cProfile 数据；profiler 会放大墙钟时间，只用它
比较调用结构和热点，不把该次执行写入性能样本。

`benchmark_qwen3_ascend.py` 保留默认短问答、eager attention、动态缓存与原有 JSON
字段。310P 使用 `--dtype float16 --allow-hf32`：后者在两侧显式允许 RoPE FP32
矩阵乘降精度，并记录实际设置。`--context-tokens 512` 或 `2048` 使用原 310P
报告的固定基础设施段落及同一重复次数算法，最终长度记录为 `prompt_tokens`，
不是强行截断到请求长度。`--fixed-length --new-tokens 32` 强制生成 32 个 token。

原生 `torch_npu` 必须记录算子编译模式；310P 上默认模式与预编译二进制核的
耗时可能不同。用 `--torch-jit-compile off` 显式选择二进制核，或 `on` 选择即时
编译；默认 `auto` 保留环境设置。JSON 的 `torch_jit_compile` 记录实际状态。
attention 的 `eager` 与 `sdpa` 应分别实测，采用可用的更快配置作为比较基线。
比较权重预转换时，原生侧使用 `--torch-pack-linear`：对 exact `torch.nn.Linear`
的权重创建 format 29 快照并单列耗时。仅重绑 Linear 的 Parameter，保持共享源
权重的 Embedding 原有布局；计时仍调用原生 Linear。JSON 记录实际转换层数。

Jittor FP16 可加 `--freeze-linear`，在计时前将 Linear 显式替换为 ACL NZ
权重快照；JSON 分别记录转换耗时、转换层数和转换前后的 Parameter 数量。
这项优化只适用于不再更新权重的无梯度推理，普通 checkpoint 应在转换前加载。
`--validate-prompts` 此时引用的是同一个已转换模型，只能验证缓存和重放一致性；
权重转换正确性需要独立参考，例如比较两侧 JSON 中每个验证 prompt 的完整 token
序列以及 logits，不能把同模型自比较当作跨框架对拍。

`--cache-implementation static --compile` 使用 Transformers 的
`CompileConfig(fullgraph=False, dynamic=False, mode="reduce-overhead")`，只编译
decode；也可指定 `--compile max-autotune`。显式 compile 只接受 static cache。
每次预热都执行完整生成，compile 模式至少预热 6 次；两侧每个计时样本前后同步。
结果中的 `compile_observed`/`compile_observed_final` 给出真实 `stats`、拒绝原因与
device graph 拒绝原因，不能仅凭 `compile` 参数宣称发生重放。默认自动重放开关也
记录为 `auto_graph_replay`；原生 Torch 不提供这些 Jittor 计数时相应字段为 null。

原有 `prefill_*` 仍测直接 `model(...)` 前向（含全部位置 logits），不能与历史报告的
`generate(1)` 耗时混用。多 token 模式额外报告 `generation_prefill_*`，并用
`(generate(N) - generate(1)) / (实际生成长度 - 1)` 计算 `decode_seconds_per_token`
及 `decode_tokens_per_second`。每次输出都在丢弃前同步；token 读回在计时外。

`--validate-prompts` 在计时后对主 prompt 和两个附加问题比较优化配置与同模型
dynamic cache、关闭显式 compile 的参考生成，要求逐 token 一致；也可重复传入
`--validation-prompt "问题"` 指定问题。该检查守住缓存/重放一致性，跨框架正确性仍需
比较两侧 JSON 的 token ids 和 `--logits-output`。`--output` 将完整 JSON 保存到 lab；
所有验证和 profiling 完成后才检查 Jittor 回退计数并报告成功，原生 Torch 没有对应
回退计数器，`fallback_count_available=false` 明确标识这一点。

两侧都保存 validation 和 logits 后，用独立 NumPy 进程比较，避免加载任一框架：

```bash
python agent/skills/jittor-transformers-perf/scripts/compare_qwen3_ascend.py \
  "$JITTOR_RESULT" "$TORCH_RESULT" --output "$PARITY_RESULT"
```

比较器要求 checkpoint、版本、精度、长度等相同，所有预热/计时/验证 token 序列
逐位一致，且 Jittor 有真实的零回退计数。logits 要求 finite、argmax 一致和相对
L2 误差不超过 1%；可显式设置 `--max-relative-l2`，结果会记录实际阈值与误差。
attention、权重预转换和 compile 配置可不同，报告时仍需分别列出。

```bash
perf_script=agent/skills/jittor-transformers-perf/scripts/benchmark_qwen3_ascend.py
python "$perf_script" --backend jittor --model "$QWEN_MODEL" \
  --dtype float16 --allow-hf32 --context-tokens 512 --new-tokens 32 \
  --fixed-length --attn-implementation eager --cache-implementation static \
  --compile --warmups 6 --samples 5 --validate-prompts \
  --output "$JITTOR_LAB_ROOT/_state/qwen3-310p/jittor-static-512.json"
```

切换到原生环境、`--backend torch` 后使用相同参数。原生 `torch_npu` 是否提供可用的
compile backend 需在该环境验证；工具不吞编译异常。与未编译基线比较时明确标注双方
模式，不把编译模式的优势归因于 eager 算子。

`benchmark_training_hotspots.py` 的 SDPA 模式支持可配置 batch/head/sequence/head-dim、
causal 与同步口径。Jittor `math` 显式绕过 native loader，`flash` 要求且校验 official
backend，`direct` 使用预物化 BSHD 输入，`default` 验证生产 dispatch。非 required、
无 dropout 的训练默认在 `B*H*Lq*Lk < 2^24` 时选择 math；可用
`JITTOR_FLASH_ATTN_TRAINING_MIN_SCORES` 调整，设为 0 则禁用该阈值。

## 常用命令

```bash
SCRIPT_ROOT=agent/skills/jittor-transformers-perf/scripts
JITTOR_PERF_JT_SITE=<site-packages with transformers 4.56.2> \
JITTOR_PERF_RT_SITE=<real torch site-packages> \
CUDA_VISIBLE_DEVICES=<gpu> cache_name=hf_tiny \
  "$SCRIPT_ROOT/run_perf_env.sh" \
  <jittor-python> \
  "$SCRIPT_ROOT/benchmark_hf_tiny_models.py" \
  --backend jittor --model bert --phase forward --repeats 20
```

```bash
SCRIPT_ROOT=agent/skills/jittor-transformers-perf/scripts
JITTOR_FLASH_ATTN_JITTOR_SRC=/path/to/flash-attention \
JITTOR_FLASH_ATTN_HEAD_DIMS=64 JITTOR_FLASH_ATTN_DTYPES=fp16 \
CUDA_VISIBLE_DEVICES=<gpu> cache_name=sdpa_train \
  "$SCRIPT_ROOT/run_perf_env.sh" \
  <jittor-python> \
  "$SCRIPT_ROOT/benchmark_training_hotspots.py" \
  --backend jittor --case sdpa --phase fwd_bwd --dtype float16 \
  --sdpa-backend default --sync-mode per_call \
  --batch 4 --heads 12 --length 1024 --head-dim 64 --repeats 12
```

```bash
SCRIPT_ROOT=agent/skills/jittor-transformers-perf/scripts
CUDA_VISIBLE_DEVICES=<gpu> cache_name=softmax_boundary \
  "$SCRIPT_ROOT/run_perf_env.sh" \
  <jittor-python> \
  "$SCRIPT_ROOT/audit_kernel_changes_gpu.py" \
  --task softmax-case --length 50257 --rows 2 --dtype float32
```

详细结果查 `docs/results/index.md`（昇腾 Qwen3 训练见
`docs/results/2026-08-30-qwen3-ascend-training.md`；早期按算子的 transformers 报告已退役，
Git 历史 `e3c369acb` 可查）；数值精度对拍仍使用
`agent/skills/jittor-torch-diff/`。

### Ascend 生成与视觉模型组件

`benchmark_generative_vision_ascend.py` 与 `_generative_vision_cases.py` 使用同一套
离线 checkpoint 和 NumPy 输入，支持 Wan2.2 denoiser、SDXL UNet、SigLIP2 vision tower、
DINOv2、SAM2 image model，以及四个完整预测模型：ConvNeXtV2 分类、SwinV2 分类、
Depth Anything V2 深度估计、RT-DETRv2 检测。后四者由 `_stock_vision_cases.py`
提供官方模型定义，包含预测 head，不替换 forward 或安装 adapter。先用 `--tiny --prepare --backend torch --device cpu`
保存 CPU fixture，再由 Jittor 加载同一目录并对拍；完整尺寸测速必须另传真实 `--model`。
两侧独立进程、相同 Transformers/Diffusers 版本，NPU 首次编译串行，至少 3 次预热与
5 次逐次同步样本。运行前加载 CANN、选择设备，并设置外部 `JITTOR_LAB_ROOT`。

```bash
python agent/skills/jittor-transformers-perf/scripts/benchmark_generative_vision_ascend.py \
  --backend jittor --device npu --case wan --model "$WAN_CHECKPOINT" \
  --attention sdpa --warmups 3 --samples 5 \
  --output "$JITTOR_LAB_ROOT/_state/model-bench/jittor-wan.json"
python agent/skills/jittor-transformers-perf/scripts/compare_generative_vision_ascend.py \
  "$TORCH_RESULT" "$JITTOR_RESULT" --max-relative-l2 0.02 \
  --output "$JITTOR_LAB_ROOT/_state/model-bench/parity.json"
```

比较器核对模型、规格、版本、参数数量和零回退，再计算逐字段误差。默认 Wan 组件
为 320×512×17 的一次条件前向（800 self-attention tokens），`--profile large` 为
704×1280×17（4400 tokens）；包含逐 token 时间条件和官方 FP32 islands。
SDXL 为 1024² 的单次条件前向，均不能当作完整 CFG 采样耗时。SAM2 为单图和
一个正点提示，不能外推视频记忆模块。只有通过输出对拍的数据才能作为性能结论。
组件比较器无论输入顺序如何均核对每个 Jittor 报告的零回退计数。组件报告与比较 JSON
必须写入外部 lab；`--prepare` 创建的 fixture 目录也必须在 lab 下且位于 checkout 外。

组件报告在计时后读取 `graph_replay` 元数据，记录自动重放次数、设备图是否启用、
拒绝原因及保留内存估计；读取不执行张量计算。算子 profile 需要至少 5 次完整预热，
避免把第三次重放时的一次录图尝试误当成稳态重复执行。原生 Torch 的设备图字段为
`null`，不能据此推断它执行了 Jittor 的录图策略。

310P 的无梯度 FP16、等 head 数 noncausal attention 在没有用户 mask 时可使用
PFA 的原始 Q/K 长度，包括矩形 cross-attention，无需构造 padding 或 mask。
有 mask / causal 路径保留各自验证范围；不能简单取消 SDK 对带 mask 矩形输入的
限制。旧 causal padding 的只读零缓冲区按设备、dtype、shape 缓存并设内存上限，
不省略实际输入复制、裁剪或因果语义。


新增四个视觉 case 名为 `convnextv2`、`swinv2`、`depth_anything_v2`、`rtdetr_v2`。
它们的完整随机输入尺寸依次为 224、256、518、640；随机输入只能验收数值与速度，
不能说明语义效果。Transformers 4.56.2 的 ConvNeXtV2、SwinV2、RT-DETRv2 仅接受
stock eager；Depth Anything 可用 `--attention sdpa` 选择其 DINOv2 backbone，深度 head
保持 eager。本工具默认显式选择 eager，不代表上游库自动选择的默认后端。
两侧选择已测较快且不同的 eager / SDPA 配置时，对拍须显式加
`--allow-attention-difference`；比较器仅允许 attention 元数据差异，其他规格与全部
输出仍检查。不能用较慢原生配置作唯一加速分母。Tiny RT-DETR 使用公开 `initializer_range=0.1`，避免小网络初始化时所有
anchor 分数退化为相同 bias 而只测到未约定的 top-k tie 顺序；这不改变预训练配置。

默认直接调用 stock 模型，后端的算子分派与自动重放不要求修改模型推理代码。
`--pack` 是额外显式的 FP16 Linear 权重快照预处理，需单列配置、层数与转换时间，
不能将其收益归为默认自动捕获。分别运行 raw 与 `--pack` 并核对同一 checkpoint 的
全部输出；`--pack` 仅适用于 NPU，无适用层时转换数可能为零。脚本定义 case 不代表其所有 dtype / device
已通过验收；每个模型仍须按 CPU fixture → 完整设备前向 → 真图像效果逐级记录。
报告另存实际 HF32 / 重放策略、harness 与 case helper 的 SHA256；计时中的
`no_grad` 状态从作用域内部读取，不将作用域退出后的状态当作计时条件。

310P 的无梯度连续 FP16 / FP32 推理已接入转置卷积与双线性 resize；FP32 卷积仍需
显式允许该卡的降精度 cube 策略。新 ACL 入口按 SoC 验证范围选择，不以 310P 对拍
代替 910B 验收。`interpolate(antialias=True)` 当前明确拒绝，不能把参数被接受当作
抗混叠数学已实现。

310P 无梯度推理另有 GridSampler2D（bilinear / zeros / align_corners=False）、
单轴一维整数 GatherV2，以及连续二维 FP16/FP32 最后轴的 largest / sorted TopK。
这些由公共算子入口选择，不要求改模型 forward；不满足条件时保留原有路径。
GatherV2 只验收合法索引，既有 ACL advanced indexing 未正确拒绝越界输入的限制见
问题总账 KI-BACKEND-024。TopK 的算子提速不能作为检测整模数值通过的替代证据。

真实图像先在独立原生 CPU 环境，通过 `_generative_vision_cases.preprocess_image`
调用 checkpoint 的官方 slow image processor 一次，保存 NumPy 输入及处理器元数据；
两侧以 `--input-npz "$SHARED_INPUT"` 读取同一个 lab archive。输入 archive 的 SHA256
写入报告与规格，预处理不计入 forward。Depth Anything 保留长宽比，其实际尺寸可能
不同于 518²。所有 principal 输出保存在报告旁的 `.json.npz`：分类 logits、未着色深度、
以及未做阈值筛选的检测 logits / boxes。效果评估应在计时后使用同一个原生 CPU 官方
postprocessor；深度可视化共用色标，检测阈值固定。单图对拍不等同于标注集准确率。

```bash
python agent/skills/jittor-transformers-perf/scripts/benchmark_generative_vision_ascend.py \
  --backend jittor --device npu --case depth_anything_v2 --model "$DEPTH_CHECKPOINT" \
  --input-npz "$SHARED_INPUT" --attention sdpa --warmups 5 --samples 7 \
  --output "$JITTOR_LAB_ROOT/_state/model-bench/depth-jittor-raw.json"
```

用独立原生 CPU 环境运行 `compare_stock_vision_effects.py`，读取已保存输出，生成
分类 top-5、共用色标的深度图或固定阈值检测对照，以及 JSON。检测匹配只表示两种
实现的同类预测框一致程度，不是有标注数据集的 AP；完整 logits / boxes 的有序数值
对拍仍须单独通过。效果比较要求两侧使用同一 attention 配置及共享预处理输入。

```bash
python agent/skills/jittor-transformers-perf/scripts/compare_stock_vision_effects.py \
  --left "$TORCH_IMAGE_RESULT" --right "$JITTOR_IMAGE_RESULT" \
  --image "$SOURCE_IMAGE" --input-metadata "$SHARED_INPUT_METADATA" \
  --source-root "$JITTOR_SOURCE_ROOT" \
  --output "$JITTOR_LAB_ROOT/_state/model-bench/image-agreement.json"
```

### Arg-reduction 的 value / index 不变量

`probe_acl_arg_reduce.py` 用无模型有限输入，检查极值是否正确、是否等于返回 index
指向的原值，以及有限值并列时是否选首个位置。默认逐行偏移的 `[1,8400,80]`
可复现 310P / CANN 9.1.0 的原始 FP16 `aclnnMaxDim` 行边界错误；重复相同行容易
掩盖它。先独立 CPU，再串行执行原生 NPU、Jittor 公共路径和原始 SDK 路径：

```bash
<real-torch-cpu-python> agent/skills/jittor-transformers-perf/scripts/probe_acl_arg_reduce.py \
  --backend torch --device cpu --require-exact \
  --output "$JITTOR_LAB_ROOT/_state/arg-reduce/cpu.json"
<real-torch-npu-python> agent/skills/jittor-transformers-perf/scripts/probe_acl_arg_reduce.py \
  --backend torch --device npu --output "$JITTOR_LAB_ROOT/_state/arg-reduce/torch-npu.json"
JITTOR_TORCH_SHIM=0 <jittor-python> agent/skills/jittor-transformers-perf/scripts/probe_acl_arg_reduce.py \
  --backend jittor --device npu --route public --require-exact \
  --output "$JITTOR_LAB_ROOT/_state/arg-reduce/jittor-public.json"
JITTOR_TORCH_SHIM=0 <jittor-python> agent/skills/jittor-transformers-perf/scripts/probe_acl_arg_reduce.py \
  --backend jittor --device npu --route sdk \
  --output "$JITTOR_LAB_ROOT/_state/arg-reduce/jittor-sdk.json"
```

原始 SDK 的错误是诊断证据，不算公共行为通过：JSON 明确记录 `passed`，
`--require-exact` 额外要求失败时非零退出。`--op min`、`--rows`、`--columns` 可收窄
域；SDK route 只允许原生 Jittor NPU，输出必须在外部 lab。CPU Jittor 必须显式设置
`JT_BACKEND=cpu`。分别保留每条命令的输入 / 输出 NPZ、JSON 和 fallback 增量。
此探针不覆盖 NaN 或混合正负零的索引策略，不能由普通有限值通过推断这些边界支持。

### Vector norm 的精度与特殊值

310P 无梯度 FP16 / FP32 的最后单轴 L2 norm 可走标准 `LinalgVectorNorm` owner，
native vector_norm、Torch norm 和 normalize 共享查询入口。strided 输入显式复制为
连续输入，必须把复制时间计入。多轴 GRN 本轮未获稳定整模收益，保留旧路径；
CPU、梯度及未准入轴也保留各自原有实现。canonical vector_norm 的 dim=None 会
先展平再按最后轴查询，不能把 provider 直接拒绝 None 解释成所有公共 None 都拒绝。
标准生产开 / 关对照没有确认稳定提速。保留该路径的收益是修正真范数仍可表示时
FP16 中间平方的溢出 / 下溢；不得将 LAB 原型约 2.2% 的收益当作生产速度结论。

`probe_acl_vector_norm.py` 比较独立 CPU 与 NPU 的公开 norm / normalize，保存原始
数组、非有限值 mask 和有限误差；专门包含半精度中间平方溢出、真范数溢出、NaN、
Inf、零向量与 epsilon 临界值。已知 310P 特殊值限制可能使 `reference_matched=false`，
不能把与原生 NPU 一致或诊断退出零当成 CPU 数学全部通过。

```bash
<real-torch-cpu-python> agent/skills/jittor-transformers-perf/scripts/probe_acl_vector_norm.py \
  --backend torch --device cpu --output "$JITTOR_LAB_ROOT/_state/vector-norm/cpu.json"
<real-torch-npu-python> agent/skills/jittor-transformers-perf/scripts/probe_acl_vector_norm.py \
  --backend torch --device npu --reference "$JITTOR_LAB_ROOT/_state/vector-norm/cpu.json" \
  --output "$JITTOR_LAB_ROOT/_state/vector-norm/torch-npu.json"
JITTOR_TORCH_SHIM=1 <jittor-python> agent/skills/jittor-transformers-perf/scripts/probe_acl_vector_norm.py \
  --backend jittor --device npu --reference "$JITTOR_LAB_ROOT/_state/vector-norm/cpu.json" \
  --output "$JITTOR_LAB_ROOT/_state/vector-norm/jittor-npu.json"
```

`--require-match` 使参考 mask 或有限误差不匹配时非零退出。每个 dtype 和边界分别
报告；常规有限输入、极值保留与梯度准入不能用一个宽松容差混成“支持”。
