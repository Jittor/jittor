# 昇腾 310P：视觉模型复测与 FP16 数值修复

- 状态：生产速度未取得突破；修复 FP16 极值与单轴范数数值，ConvNeXt / Swin 完整输出和真图像对拍通过；RT-DETR 原顺序仍未验收
- 复查日期：2026-10-30（America/Los_Angeles）
- 实现提交：`02b021cba3c22f7907ae283610d82fe6dd6fb4de`；最终 16 份前向报告均核验该 SHA、空 dirty diff、未 packing，Jittor fallback 增量为 0
- 同步范围：起点 `f283ba92d`，已整合上游 `4543b0f2b1ebda3382c67670dd3f12c105ad3f7e`；两次各 45 秒 fetch 超时，未声称取得更新远端
- 范围：Ascend 310P3 / CANN 9.1.0 的公共算子正确性、两个完整分类模型复测及 RT-DETR 诊断；不含整模训练或标注集准确率
- 维护者：ACL 后端与 Torch 兼容层维护者
- 复查触发：CANN / 驱动、dtype / 分派、模型快照、Transformers 或 replay 策略变化

## 最终性能与效果

ConvNeXtV2、SwinV2 仍慢于独立原生 Torch。标准生产范数入口开 / 关的受控 A/B 未确认稳定提速；保留该入口的依据是有限 FP16 输入的真实精度修复。早期 LAB 候选收益不计最终生产加速。

| 模型 / 输入，batch=1 | Torch | Jittor | 完整输出 |
| --- | ---: | ---: | --- |
| ConvNeXtV2 Tiny / 224² | 20.002 ms | 22.010 ms | PASS |
| SwinV2 Tiny / 256² | 11.803 ms | 33.242 ms | PASS |
| Depth Anything V2 Small / 518²，引用 10/29 | eager 44.671 ms | 显式 SDPA 27.050 ms | 上轮 PASS、快 1.65×；本轮未重测 |
| RT-DETRv2 R18vd / 640² | 排名排除 | 排名排除 | 原顺序仍 FAIL |

前两行是 raw FP16 / eager / no packing，两轮各 7 个样本、共 **14 样本池化中位数**，不是两个中位数平均；每进程 5 warmups、两个不同随机输入 slot。Conv 两轮原生为 20.129 / 19.979 ms，Jittor 为 22.124 / 21.846 ms；Swin 原生为 11.880 / 11.796 ms，Jittor 为 33.421 / 33.197 ms。次轮同时反转模型与后端顺序，首次 JIT、加载、预处理及输出后处理不计稳态。

本轮只重测受影响的 Conv / Swin。Depth 的完整配置与上轮生成模型结果保留在[10/29 四模型报告](2026-10-29-stock-vision-ascend310p.md)及其引用中；不能把引用值当成本轮新测收益，也不以较慢的原生 Depth SDPA 构造领先倍率。

相同共享猫图 archive、独立 CPU 官方后处理的结果如下。两图已视觉检查，原图 / 原生 / Jittor 三栏和全部 top-5 文字完整，无缺失或裁切。

| 真图像 | logits 相对 L2 | top-1 | top-5 |
| --- | ---: | --- | --- |
| ConvNeXtV2 | 0.002758931 | 双方 Egyptian cat | 顺序相同，重合 5/5 |
| SwinV2 | 0.003420380 | 双方 Egyptian cat | 顺序相同，重合 5/5 |

随机与图片共 10 份完整数组对拍均满足既有相对 L2≤0.02、shape / finite / 全字段检查，包括后述 4 份生产 A/B。没有重排输出或放宽阈值。猫图 SHA256 为 `f311fbd22e8a0562a237f0985abe185b2dba82435acebd9b452f1dbdd4387dda`；单图实现一致性不是 ImageNet accuracy、COCO mAP 或真值深度质量。Depth / RT 的真图像效果本轮未重测。

## 标准生产 A/B 没有确认速度收益

同一实现提交下，只开 / 关 canonical norm query，保持 stock 模型和原输出；报告 `operator_ab` 记录策略、命中数和诊断 wrapper hash。query 关闭为旧表达式，开启为标准生产入口，命中数分别 0 / 72。

| Swin 生产 query 对照 | 第 0 轮 | 第 1 轮 | 14 样本池化中位数 |
| --- | ---: | ---: | ---: |
| 关闭（旧表达式） | 33.566 ms | 33.333 ms | 33.431574 ms |
| 开启（生产） | 33.615 ms | 33.164 ms | 33.471810 ms |

生产路径约慢 **0.12%**，处于本次测量波动量级，不能称稳定改善；两轮方向也不一致。全部原顺序 logits 对原生均 PASS。早期 operator-substitution LAB prototype 为 33.451848→32.714397 ms（约 2.20%），随后单次公共阶段为 33.008215 ms；它们不是最终生产收益，不挑选较快的一轮替代完整对照。

Conv 的 device graph 活跃；Swin 保留 executor replay，但 whole device graph 因 stock forward 的 CPU `ternary[4,64,64]` mask 拒绝录制。未使用不安全的 CPU 常量折叠或强行录制。零 fallback 不代表模型没有显式 CPU 工作。

## 保留范数入口是为了精度

一个 canonical `_try_vector_norm` query 由 native `linalg.vector_norm`、`nn.normalize(p=2)` 和 Torch norm 签名适配器共享。新标准 ACL runner 只准入 **Ascend310P / no_grad / real FP16 或 FP32 / 正尺寸 / p=2 / canonical 单一最后轴**；非连续输入显式 contiguous，复制成本计入测量。其他 SoC、dtype、阶数、轴或梯度保留旧表达式，未新增模型 forward、checkpoint 或 adapter 修改。

在双独立 oracle（Torch CPU 与原生 NPU）验证下，下列 **真实有限值结果**得到修复：

| FP16 输入 / 操作 | CPU / 原生 NPU / 新路径 | 旧 Jittor |
| --- | --- | --- |
| `[1000,1000,1000]` 的 norm | 1732，结果可由 FP16 表示 | 255.875，中间平方溢出导致错误 |
| 同一输入 default normalize | 每项约 0.577 | 每项约 3.908 |
| `[1e-5,2e-5,3e-5]` 的 norm | 约 3.743×10⁻⁵ | 0 |
| 同一小值 default normalize | 约 `[0.2676,0.535,0.801]` | 饱和值 65504 |

因此该功能作为精度修复保留，同时报告其未证明稳定生产加速。原生 NPU 一致不自动等于 CPU 正确：真 norm 超过半精度范围以及 NaN / Inf 等限制仍在后文单列。

原 14 个 Swin normalize、40 个 GRN SDK case 的所测有限输出与原生 NPU byte-exact，dtype 不变、fallback=0。早期默认 transpose 已物化；后来开启 `transpose_storage_view=1` 重测四个真实 view stage × FP16 / FP32，仍与原生一致。FP16 真 view 微算子含复制，旧约 404–417 μs、新约 398–423 μs，无稳定微算子胜出。

native `vector_norm(dim=None)` 先 flatten 到 rank1，再归约 axis0，因此能进入该最后轴实现；provider 直接拒绝 `None`，直接 query 的 compat `torch.norm(dim=None)` / `F.normalize(dim=None)` 保留旧 fallback。native `jt.norm` 的 square-sum epsilon floor 未改。GRN `(1,2)` 多轴不进入新生产路径。

## FP16 MaxDim / MinDim 的独立错误与修复

无模型输入 `x[0,r,c]=(r % 97)/16+c/128`，FP16 `[1,8400,80]`，每行极值唯一。原始 SDK 在 1056–1067、2112–2123、……、7392–7403 共 **84 行**返回错误值。max 的第 1056 行真实最大值与索引 79 对应值是 5.9921875，SDK 返回 5.3671875。

独立 Torch CPU / NumPy 正确；从同一 NumPy 重新创建 contiguous 原生 NPU tensor 仍失败；原生 Jittor 直接 `ArgReduceOpRunner`→SDK 得到同样错误。这排除了模型 hook 快照、读回时机、存储污染及 TopK tie policy 作为该局部错误的解释。

更新后的 portable probe 对 max/min 都运行同一五种输入：CPU 5/5 通过；原生 Torch NPU 与 raw Jittor SDK 一致，shifted FP16 为 84 错值 / 0 错索引，random FP16 为 83 错值 / 28 错索引；ascending、descending、FP32 控制正确。公共 Jittor max/min 各 5/5 `passed=true`，`--require-exact` 退出 0。原始 SDK 的 `passed=false` 保留为诊断失败，不算公共通过。

既有 `arg_reduce_acl` provider 对 **310P FP16 max/min** 用 FP32 求索引，再 Gather **原 FP16 输入**的值，按输入轴 geometry reshape 并恢复原输出 shape。有限 FP16 值可被 FP32 精确表示；保留 rank1、keepdims、索引 / 值 dtype 与梯度，不增加 no-grad 或模型几何特判。其他 SoC / dtype 不变。

不把 FP32 结果直接 cast 回 FP16：该设备 Cast 实验将 Inf / NaN 饱和为 65504，Gather 原输入避免这一新增错误。已测 native / compat × max / min 的混合正负零、多 NaN 和 Inf，修复前后 value / zero sign / indices 不变；其既有 first-index 缺口并未修复。

两 slot、5 warmups、每 slot 15 轮交错顺序、保留输出且同步的整调用微测，包括 cast + Gather：`[1,16,7]` 为 **0.0856→0.1423 ms**，`[1,8400,80]` 为 **0.1038→0.2570 ms**。这是正确性开销，不是加速；不能直接外推 RT 整模增量。

## RT-DETR 仍未通过

旧原生 FP16 trace 的 encoder scores 与 observed max/topk input 每 slot 有 84/8400 行差异，最大绝对差 2.361328125 / 2.861328125，现已独立隔离上述 SDK 错误。Jittor compat `.max(...).values` 原本使用单独正确的 ReduceMax；修复索引后，完整 stock RT-DETR 的 encoder scores、max 输入、TopK IDs、原 logits / boxes 与旧 Jittor trace **逐位相同**，fallback=0。

对保留的原生 FP16 trace，两个 slot 原顺序 logits 相对 L2 为 **0.0543528 / 0.0480597**，boxes 为 **0.2026479 / 0.1640688**，仍未通过 2% 阈值。半精度参考的局部 SDK 错误不能称为全部差异的唯一原因：此前 FP32 猫图 logits / boxes 也失败（0.050188 / 0.12894），并列 / 近似分数对候选集合与顺序的敏感性仍在。没有 ID 对齐、输出重排或阈值放宽；不报告合格 RT 速度。

## 拒绝的性能候选

| 候选 | 可审核结果 | 决策 |
| --- | --- | --- |
| 专用 ConvDepthwise2d | 输出 byte-exact；整模首轮约快 3.1%，反向顺序仅 0.28%，微算子无稳定信号 | 不新增分派 |
| FP16 Im2col+Baddbmm | 普通 stage3 / stage4 约 0.56 / 0.27 ms；通用 columns 有差异，同一 NaN 输入相对旧 Conv 改变 791 值、最大绝对差 79984 | 不采纳 |
| GatherV2+where 替代 Im2col | 全部索引 / mask / 传输 / Gather / where / 收缩 / 同步约 25.86 ms，旧 stage3 约 1.1 ms | 不采纳 |
| FP32 Im2col+乘 / sum | stage3 17.79 ms，旧约 1.17 ms；普通 columns 766/3687936 不符 CPU，zero-init 仍不修复 | 不采纳；根因未隔离，不断言 SDK 绝对缺陷 |
| GRN `(1,2)` 多轴范数 | LAB 整模 21.920203→22.440176 ms，变慢 | 保留旧多轴表达式 |

旧 GRN 极端有限值精度缺口也保留：所测 FP16 100× 大值 / 1e-4× 小值 norm 对原生误差 10.2035% / 69.3944%，SDK 可改善，但不因此声称生产多轴或整模获得支持。

## 门禁与未修限制

| 会话 | 实际结果 |
| --- | --- |
| CPU core | Native 98 pass / 44 skip / 1 xfail；Torch 229 pass / 1 skip，exit 0 |
| Torch 全结构 + 10 compat CPU | 1591 pass / 4 skip |
| native recorder / package identity | 7 pass |
| HalfCPU 单独选择 | 26 pass / 32 deselected |
| native ACL / half 混合 | 501 pass / 0 skip，包含 26 CPU，不能称 501 项 NPU |
| compat 混合 | 128 pass / 0 skip，包含 CPU |
| VectorNorm 定向（与最终会话重叠） | 30 真 NPU / 3 compat CPU / 69 host；typed owner / 注册 / launcher ABI canary 通过 |
| 静态 | 源码阶段 layout 497 Markdown，manifest / diff 与 18 Python AST 全通过 |

首次 native 结构组合用 `-m not npu` 把 Half 文件 58 项全部 deselect，其他 7 项 pass，但 execution gate 正确拒绝缺失的必要执行；改为单独 `-k cpu` 后 26 项通过，首个 attempt 日志保留。这是选择错误，不是数学失败。Core xfail 为既有 swap 条件，能力 / opt-in 等 skip 不称硬件验证。

以下已知限制仍开启，见 [KI-BACKEND-025 / 026 / 027](../../agent/manuals/known-issues.md)：

- 多 NaN 与混合正负零 arg-index 不满足 CPU first-index，修复前后相同。原始 CANN MaxDim / MinDim 及 torch_npu 本身未被本补丁修复。
- 既有 int32→int64 arg-reduction→int32 路径将 `2147483647` 返回为 `-2147483648`，索引正确。新 FP16 guard 不涉及整数；强制原分支的独立进程同样失败。具体 input cast / SDK 值输出 / 还原 cast 尚未逐段隔离。该测试不在最终 501 项指定集合中，40 pass / 1 fail 的初次额外会话和独立 1 fail 控制保留，测试未放宽。
- FP16 norm 遇 NaN、±Inf 或真实结果超过 65504，CPU 为 NaN/Inf，新 SDK 为 65504；旧 Jittor 在这些样例也已错误。FP16 零向量 default eps=1e-12 normalize：CPU NaN，原生及旧/新 Jittor 65504；显式 eps=1e-4 的零向量均为零。
- FP32 vector norm 的 NaN / Inf 对 CPU 正确，但 normalize `[NaN,1,2]` 旧/新 Jittor 为 `[NaN,1/eps,2/eps]`，CPU / 原生全 NaN。责任单一 owner 未隔离，不声称 clamp 根因；不据这些样例断言硬件永远不能支持特殊值。80 字段特殊值探针保留 `reference_matched=false`，不算 CPU 全域通过。

优化在公共 op / dtype / axis / grad / SoC 条件满足时自动分派，无需改 Wan2.2 或这些模型的推理代码；Torch 程序须安装并激活 shim。显式 attention、packing 与 replay 参数仍是运行配置。新 norm 入口无梯度，ArgReduce 修复保留既有梯度；不外推完整训练、BF16、910B、CUDA 或 ROCm 验收。

## 环境、复现与证据

同上轮固定快照：Conv `facebook/convnextv2-tiny-1k-224` revision `f4db009e63145e02b3c075aa64d90ce41bcca4b1`；Swin `microsoft/swinv2-tiny-patch4-window8-256` revision `40213dad8563a5a916d434a1291443c0fa6358f0`。双方使用同一 Swin 原样转换 safetensors（245 张量不变）。环境为 310P3 44 GB / CANN 9.1.0 / driver 26.1.0.b050，独立 Torch 2.12.0+cpu / torch_npu 2.12.0、Transformers 4.56.2、Diffusers 0.35.1；保留后台服务，未取得物理独占。

原生 jit_compile=False、matmul / conv HF32=True；Jittor no_grad=1、AMP 关闭、acl_allow_hf32=1、replay 输入 4 MiB / retention 256 MiB。seed=20261027，解释器 / 编译缓存隔离、单设备串行；计时期间无本任务其他重作业。原生 fallback counter 不适用，不能称其通过 Jittor counter。

按[性能 skill](../../agent/skills/jittor-transformers-perf/SKILL.md)设置独立解释器、CANN 与 lab 路径。主表复现入口：

```bash
perf_scripts="$JITTOR_SOURCE_ROOT/agent/skills/jittor-transformers-perf/scripts"
vision_run="$JITTOR_LAB_ROOT/_state/vision-follow-up"
# case_name: convnextv2 / swinv2；checkpoint 为对应固定本地快照
"$TORCH_NPU_PYTHON" "$perf_scripts/benchmark_generative_vision_ascend.py" \
  --backend torch --device npu --case "$case_name" --model "$checkpoint" \
  --dtype float16 --attention eager --seed 20261027 --warmups 5 --samples 7 --slots 2 \
  --source-root "$JITTOR_SOURCE_ROOT" --output "$vision_run/torch.json"
JITTOR_TORCH_SHIM=1 "$JITTOR_PYTHON" "$perf_scripts/benchmark_generative_vision_ascend.py" \
  --backend jittor --device npu --case "$case_name" --model "$checkpoint" \
  --dtype float16 --attention eager --seed 20261027 --warmups 5 --samples 7 --slots 2 \
  --source-root "$JITTOR_SOURCE_ROOT" --output "$vision_run/jittor.json"
"$TORCH_CPU_PYTHON" "$perf_scripts/compare_generative_vision_ascend.py" \
  "$vision_run/torch.json" "$vision_run/jittor.json" --max-relative-l2 0.02 \
  --output "$vision_run/parity.json"
```

反转后端 / 模型顺序再测一轮，保留不同文件。真图像按上轮共享预处理入口生成 archive / sidecar，双方加相同 `--input-npz`，由 `compare_stock_vision_effects.py` 独立 CPU 后处理。极值探针 `probe_acl_arg_reduce.py` 的 public / sdk 路由分开，CPU-first，同输入各跑 max/min；`--require-exact` 对原始 SDK 的已知错误应失败。范数探针 `probe_acl_vector_norm.py` 先存独立 CPU reference，再对 NPU；默认记录差异，`--require-match` 才以非零退出，普通诊断退出不等于数学通过。

未版本化证据根 `$JITTOR_LAB_ROOT/_state/generative-vision-310p/iteration-2026-10-30/`：

- `final/`：`baseline.json`、`commands.json`、`summary.json`、16 份前向 / 数组、10 份 parity、2 份 effects / PNG 与 `visual-qa.json`；A/B 报告含 `operator_ab`，wrapper 在外部 lab，不替换模型。
- `final-gates/`：命令 / 日志及保留的错误 selection attempt。
- `root-proofs/`：五输入 max/min CPU / 原生 NPU / raw SDK / public 控制与数组。
- `rtdetr/`：直接 SDK、特殊值前后、成本、原整数失败、same-output trace、`rtdetr-fixed-comparison.json`。
- `vector-norm/`：有限精度 / 特殊值、true-view、LAB A/B、公共阶段与 source freeze；`convnext/`：全部拒绝候选及 `IM2COL_HANDOFF.md`、FP32 zero-init 失败日志。

原始模型、缓存、图片和实验日志不进入仓库。报告保留范围与失败，不把 LAB prototype 或 SDK 相同错误当成最终生产收益或数学正确。
