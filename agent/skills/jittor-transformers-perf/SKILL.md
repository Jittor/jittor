---
name: jittor-transformers-perf
description: Reproducible CUDA and Ascend performance analysis for Transformers on import-jittor-as-torch versus real PyTorch. Use for operator, Transformer-block, HF-model, SDPA, gradient-management, or optimizer-step benchmarks where Jittor lazy execution and backend/version isolation must be handled correctly.
---

# Jittor Transformers 性能基准

本 skill 的可复用源码位于 `scripts/`，运行状态与结果默认写到
`${JITTOR_LAB_ROOT:-$(cd .. && pwd)/jittor-lab}/jittor_transformers_perf/`。
目标是得到能解释、能复现的
Jittor vs 真 PyTorch 数据，而不是把 JIT、H2D 或 lazy graph 漏执行当成性能结论。

## 环境

- Jittor：显式设置的 `JITTOR_CI_PYTHON`，仓库源码 `python/jittor`。
- PyTorch oracle：显式设置的 `REAL_TORCH_PYTHON`。运行前必须断言 `torch.__file__`
  不含 Jittor shim。
- 所有命令通过 `agent/skills/jittor-transformers-perf/scripts/run_perf_env.sh`，
  并为并行任务设置唯一 `cache_name` 与 `CUDA_VISIBLE_DEVICES=<gpu>`。缓存、日志和
  结果必须留在 lab 工作区，不得写入 Jittor 主仓库。需要指定 CUDA toolkit 时设
  `JTCUDA=<cuda-root>`（含 `bin/nvcc` 与 `lib64`）；不设则由 Jittor 自行发现 nvcc，设错直接退出。

## 不可省略的计时规则

1. 预分配多个不同输入 slot；计时前触碰所有 slot，排除首次 H2D residency。
2. 计时内保留每次输出；训练还要保留每次梯度，最后统一同步。只保留最后一个输出会让
   Jittor lazy executor 漏执行前面的迭代，产生假快。
   这条用于裸算子 queued 微基准。下述 Trainer 阶段同步协议让正常 optimizer 消费
   梯度并逐步强同步；计时路径不得为观测额外 clone、D2H 或归档完整梯度。
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
Transformers 版本和相同本地 checkpoint。运行 Jittor 一侧前设置
`JITTOR_TORCH_SHIM=1`，并在加载权重后显式迁移到 NPU；两侧都通过
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

当前 `benchmark_qwen3_ascend.py` 验证独立 Torch frontend 的模块/Tensor 身份，
公开设备使用 `npu:0`，逐张量核实原生 `acl` 后端与逻辑卡号；模型所有参数/buffer、
输入和每层KV-cache都纳入驻留检查。`--validation-output /lab/run/full.npz`
保存完整 prefill logits、输入/mask 与 cached/uncached greedy token IDs；
`--logits-output` 仍只保存末 token，用途必须区分。两侧都在计时后检查 cache on/off
一致及完整 logits finite，跨 runtime 数值对拍仍需独立比较 NPZ，生成文本相同不能代替它。
`--report-output /lab/run/report.json` 写结构化结果，所有输出与 JITTOR_HOME 必须在
显式 `JITTOR_LAB_ROOT` 下且 checkout 外；已有输出不会覆盖。支持FP32/FP16/BF16，
需要外部固定相同checkpoint、精度、attention和设备负载再配对，工具本身不声明模型已通过。

整模入口还提供 `--batch-size`、`--sequence-length`（0保留原prompt长度）和
`--input-slots`（默认3）。所有slot保持相同batch/context形状，使用checkpoint tokenizer的
有效token确定性轮转并按需重复到指定长度，属于功能/性能输入，不代表自然语言质量评测。
计时前逐slot预分配、同步并执行prefill/generation；稳态循环轮转slot，分别记录每次slot编号。
两侧固定highest FP32 matmul并显式禁HF32、禁环境autocast，JSON读回精度设置。
生成固定执行 `new_tokens` 步，逐batch保留token IDs；以所有batch的实际生成位置数计算吞吐。
计时后每个slot都检查完整logits finite及cache on/off exact，NPZ首slot使用原key，其余
以 `_slot_0001` 等后缀保存完整logits/输入/mask/生成ID。跨runtime比较必须覆盖全部slot，
不能用单条生成文本替代。该工具的host检查不构成NPU速度或数值通过。

`compare_qwen3_ascend.py` 在独立CPU进程读取两份report/NPZ，left必须为Jittor、right
必须为原生Torch参考。它核对checkpoint路径/Transformers版本/attention/dtype/形状和严格
精度、驻留与零回退报告，按 `--atol/--rtol` 显式提供的事前预算检查所有slot完整logits，
所有batch的输入、mask和cache on/off greedy IDs必须exact；漏slot、截断logits、NaN、
dtype漂移或错误元数据直接失败。调用者仍需另用SHA清单固定共享checkpoint；比较器本身
不生成训练或速度结论。输出仅允许新建到lab/checkout外。

回退计数只在Jittor侧由core实际测量并要求0；原生torch_npu没有此工具可核验的计数器，
JSON明确使用null/unavailable，仍检查外部张量实际NPU驻留，不把未测量伪装成原生零回退。

显存JSON分别报告allocator live/reserved/peak与设备free/total。Jittor池统计可能不含
CANN直接workspace分配，不能据两侧allocator计数宣称总HBM占用更低；free/total还包含
其他进程。严格进程峰值需另用驱动采样并记录采样间隔，采样峰值也不能称作精确分配峰值。

`benchmark_training_hotspots.py` 的 SDPA 模式支持可配置 batch/head/sequence/head-dim、
causal 与同步口径。Jittor `math` 显式绕过 native loader，`flash` 要求且校验 official
backend，`direct` 使用预物化 BSHD 输入，`default` 验证生产 dispatch。非 required、
无 dropout 的训练默认在 `B*H*Lq*Lk < 2^24` 时选择 math；可用
`JITTOR_FLASH_ATTN_TRAINING_MIN_SCORES` 调整，设为 0 则禁用该阈值。

## 常用命令

```bash
JITTOR_LAB_ROOT=${JITTOR_LAB_ROOT:-$(cd .. && pwd)/jittor-lab}
SCRIPT_ROOT=agent/skills/jittor-transformers-perf/scripts
CUDA_VISIBLE_DEVICES="${ASSIGNED_CUDA_MASK:?external allocation required}" cache_name=hf_tiny_owned \
  "$SCRIPT_ROOT/run_perf_env.sh" \
  "${JITTOR_CI_PYTHON:?set the task interpreter}" \
  "$SCRIPT_ROOT/benchmark_hf_tiny_models.py" \
  --backend jittor --model bert --phase forward --repeats 20
```

```bash
SCRIPT_ROOT=agent/skills/jittor-transformers-perf/scripts
JITTOR_FLASH_ATTN_JITTOR_SRC=/path/to/flash-attention \
JITTOR_FLASH_ATTN_HEAD_DIMS=64 JITTOR_FLASH_ATTN_DTYPES=fp16 \
CUDA_VISIBLE_DEVICES="${ASSIGNED_CUDA_MASK:?external allocation required}" cache_name=sdpa_train_owned \
  "$SCRIPT_ROOT/run_perf_env.sh" \
  "${JITTOR_CI_PYTHON:?set the task interpreter}" \
  "$SCRIPT_ROOT/benchmark_training_hotspots.py" \
  --backend jittor --case sdpa --phase fwd_bwd --dtype float16 \
  --sdpa-backend default --sync-mode per_call \
  --batch 4 --heads 12 --length 1024 --head-dim 64 --repeats 12
```

```bash
JITTOR_LAB_ROOT=${JITTOR_LAB_ROOT:-$(cd .. && pwd)/jittor-lab}
SCRIPT_ROOT=agent/skills/jittor-transformers-perf/scripts
CUDA_VISIBLE_DEVICES="${ASSIGNED_CUDA_MASK:?external allocation required}" cache_name=softmax_boundary_owned \
  "$SCRIPT_ROOT/run_perf_env.sh" \
  "${JITTOR_CI_PYTHON:?set the task interpreter}" \
  "$SCRIPT_ROOT/audit_kernel_changes_gpu.py" \
  --task softmax-case --length 50257 --rows 2 --dtype float32
```

详细结果查 `docs/results/index.md`（昇腾 Qwen3 训练见
`docs/results/2026-08-30-qwen3-ascend-training.md`；早期按算子的 transformers 报告已退役，
Git 历史 `e3c369acb` 可查）；数值精度对拍仍使用
`agent/skills/jittor-torch-diff/`。


### Ascend 逐样本交替与独立 cached decode

`benchmark_qwen3_ascend.py --control-stdin` 在每次已同步的设备调用前输出
`BENCHMARK_CONTROL_READY` JSON，并等待唯一匹配的 `{"go": N}`。使用同目录
`paired_benchmark_control.py --spec <lab/spec.json> --output <lab/new-run> --timeout 5400`
可以管理两个独立常驻解释器，逐次核对call/phase/slot/cache一致，以ABBA顺序放行；
前一个worker到达下一同步边界后才允许另一个执行。准入等待单独记录并从host延迟扣除，
首步/JIT、各slot预热与稳态分开。每个稳态phase至少10组成对样本，不能用连续跑完A的10次
再跑B的10次代替。controlled模式不允许同时profile，另起进程诊断。

spec的physical_device明确物理卡号，workers必须分别含torch和jittor，每项为argv数组、cwd、env字典及可选unset_env数组。
controller在两个启动边界重新查询npu-smi：第一套前必须为空，第二套前只允许PID/启动tick
确认的第一套worker，并要求剩余HBM至少覆盖其已用显存再加4 GiB。这只是保守启动门槛，
不能保证整个workload峰值；两边env必须只暴露这张卡。
argv指向各自解释器与相同benchmark源码，分别使用--backend torch/jittor，双方启用
--control-stdin及相同checkpoint/dtype/attention/batch/context/slots/warmups/samples。
输出路径、HOME、SDK cache与短TMP互相独立；native只保留CANN官方PYTHONPATH，不含repo/shim。
父进程不代替设备租用/空闲检查、源码和环境freeze、HF32/线程配置及共享checkpoint SHA。
调度失败只终止PID/启动tick确认的自身进程组；被中断的Jittor cache在审核并归档前不得复用。

两套模型常驻同一张已分配NPU，设备执行严格串行，启动第二套前先让第一套停在READY边界。
这要求显存容纳两套模型、各自KV/workspace及余量；不能把co-resident时的device_free当单模型
峰值，也不能直接外推独占整卡吞吐。单模型显存对比另用独立进程协议；本协议的allocator
计数保持各runtime自身口径，报告共享驻留条件。parent只证明调度并始终输出
performance_accepted=false；两边正常退出后仍须用compare_qwen3_ascend.py验收全量输出，
只有功能合格且每phase样本数达标，才形成性能结论。

启用--cached-decode时要求new-tokens>=2，每个decode样本另建同一输入的KV prefill，
并在计时外计算首个greedy token及预分配mask，再执行new_tokens-1次缓存token forward。
保留每次输出直到同步；decode计时含forward/argmax/host构图与同步，排除prefill/首token/
mask分配/D2H/准入等待。first_decode单列，decode_samples才是稳态，generate_samples仍是
含prefill的完整generation，二者不能互换。验证每slot完整cached-decode logits，手工token
轨迹必须与public generate和cache off逐整数exact；比较器拒绝缺失/截断/非finite/dtype
漂移及步数或vocab不一致。此入口不测训练step；带逐梯度存档、eval和checkpoint的release
workflow总耗时只作诊断，不能当训练吞吐。


## 持久双 worker 的 Trainer 更新协议

`benchmark_hf_trainer_steps.py` 使用两个显式解释器和本地冻结初始化 NPZ，运行小型
BERT+PEFT 分类 fixture。它不加载真实 checkpoint，也不下载模型。attention 明确为
`eager`，dropout 为零、数据顺序固定，正常 Transformers Trainer 负责 backward、
梯度累积、裁剪、AdamW 和 constant/LambdaLR。这个入口的口径是
**synchronized training phases**：双方在 update 起止与 postclip/preoptimizer
边界使用相同强同步；不能把它描述成无额外 barrier 的普通 Trainer。

两个持久 worker 各先独立执行14次功能影子更新，归档每个微批次及每个 postclip 的完整
目标梯度。计时轨迹从同一个 initial model、全新 optimizer/scheduler 与相同 RNG 重置，
必须对影子逐数组 strict0；跨 runtime 比较完整模型、optimizer、scheduler schema和值、
全部梯度及损失。`_trainer_step_protocol.py` 拒绝额外/缺失状态键与梯度，不取键交集冒充
完整比较。超参数和 step 计数严格相等；跨 runtime 的浮点数学使用冻结数值预算。
CPU/NPU RNG 的 opaque byte 编码仅同 runtime strict0，跨 runtime 单列格式差异，不能据此
宣称随机训练恢复或 checkpoint 互操作已通过。

控制器先串行初始化两个 worker，再给每次更新独占计算许可：4个 warm pairs 覆盖全部8种
微批次形状，随后10个 steady pairs，每个 runtime 总计14次更新、10个计时样本。
前一个 update 的 end 强同步完成后才交棒，final Trainer 损失读回与状态归档等双方全部
更新结束后再串行执行。计时路径没有完整梯度 clone/D2H；完整梯度由功能影子承担，计时
路径通过完整最终状态 strict0取得资格。输入签名、步号、原始时长、进程/source与事件流
必须保留。旧 `AdmissionSequence` 的20-cohort账本是另一份协议，不与这个10-pair口径合并。

`ReadbackGuard` 按张量实际 device 拒绝加速器上的 cpu/numpy/item/tolist/bool/float/int，
同时允许 Native Adam 的 CPU step.item。正式NPU入口先在实际 NPU all-one mask 上调用
Transformers SDPA helper，必须在 tensor bool 原方法前拒绝读回；host stub不能代替这项
实卡负向门禁。双方的功能和计时都使用明确 eager 配置，原 SDPA 功能结果单独保留。
`logging_nan_inf_filter=False`，logging/eval/save/checkpoint均不进入计时区间。
两个模型/上下文同时驻留，不能用此协议宣称独立单模型HBM优势。

下面的路径、解释器、缓存和设备均由调用者配置；先按环境指南准备只读 dev、纯 CPU
oracle与实际NPU oracle，不能把 `torch_npu` 装进纯CPU参考环境。所有输出与HOME/TMP/HF/
JITTOR_HOME/SDK缓存必须在lab且checkout外，外部先分配资源，脚本不自行获得租约。
`--environment-json` 是当前任务冻结的 `{torch: {...}, jittor: {...}}` 环境表，
`--adapter-site` 可指向经核验的正规distribution overlay；它们不写入共享环境。

```bash
script="$TREE/agent/skills/jittor-transformers-perf/scripts/benchmark_hf_trainer_steps.py"
"$ORACLE_CPU_PYTHON" "$script" --create-initial --device cpu \
  --repo-root "$TREE" --lab-root "$LAB" --initial "$INITIAL_NPZ"
"$CONTROL_PYTHON" "$script" --device cpu --repo-root "$TREE" --lab-root "$LAB" \
  --run-dir "$UNIQUE_RUN" --initial "$INITIAL_NPZ" --source-manifest "$SOURCE_MANIFEST" \
  --torch-python "$ORACLE_CPU_PYTHON" --jittor-python "$DEV_PYTHON" \
  --jittor-home "$OWN_CPU_CACHE" --cache-name "$OWN_CACHE_NAME" \
  --adapter-site "$ADAPTER_SITE" --environment-json "$ENVIRONMENT_JSON"
```

NPU运行需另行授权的单物理mask与 `--lease-id`，并使用含真实torch_npu/匹配SDK依赖的
独立oracle、CANN前缀与专有ACL缓存；控制器不把CPU验证转换成NPU准入。
原生 `acl` 后端、真实materialized location、逻辑设备、物理PID、fallback0和线程数都要核实。
当前工具的原始时长不自动形成性能主张；native/vendor编译证据必须另行核对，缺少覆盖
或计时中新增编译就拒绝稳态结论。运行结果与版本边界以冻结报告为准。

无框架导入的host合同可直接运行
`tests/structure/skills/test_transformers_trainer_step_protocol.py`；它覆盖独占交棒、完整形状
warmup、同步边界、提前finalize拒绝、读回负向、完整状态schema及梯度缺失拒绝。


The persistent controller records each owned worker's Linux PID, start ticks,
session and process group. Connection/authentication has a separate bounded
`--connect-timeout` (60 seconds by default); the longer `--timeout` covers runtime
initialization and compilation. Failure cleanup signals only a still-matching
owned session, waits for its workers, and saves the cleanup receipt. Host tests
exercise no connection, early crash, and refusal to signal a mismatched identity.

Record the actual determinism API state rather than infer it from a setter call.
The native Torch functional oracle can explicitly enable deterministic algorithms
before its first NPU operation. The current Jittor compatibility API setter is
UNIMPLEMENTED and its getter returns False; preserve that boundary. These two
states are not a shared deterministic policy and do not admit a performance
claim. CPU qualification proves functional/control contracts only; a later NPU
protocol requires an explicit reviewed policy and resource assignment.
