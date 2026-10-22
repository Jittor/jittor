# 生成与视觉模型：Ascend 310P3 推理优化

- Status: 新版组件、total-only 生成、数值与门禁实测完成；SDXL 默认设备录制仍有资源限制，Wan 诊断视频视觉质量失败
- Last reviewed: 2026-10-22（America/Los_Angeles）
- Final implementation: `0e0c28bc0e193103e34760a279e932b0b5e1fcbd`
- Synced baseline: `1f213177d76671bc59e7816707641914aad5baec`；已整合远端 `6878e585cee4fd987bb821ee1a24a75e5f333a8d`
- Previous verified stage: `f9dde035b27fc85580333603976c9043216b4aa6`；历史数据保留原提交 / dirty patch
- Owner: ACL backend and Torch compatibility maintainers
- Review when: CANN / 驱动、Torch / torch_npu、checkpoint、下游版本、精度、replay policy 或资源预算改变

## 新版组件结果

两轮反转运行顺序后，DINOv2、Wan 单次去噪与 SigLIP2 分别加速约 **1.91×、
1.24×、1.06×**。SDXL 单次条件 UNet 仅约 1% 小幅领先；完整出图在已测资源配置下
基本持平，不能宣称所有生成流程都超过原生。Wan 可调配置的生成耗时更短，
但两侧视频均有严重伪影，速度提升不代表可用视频质量。

| batch=1 完整预训练组件 | 原生中位数 | Jittor 中位数 | 原生 / Jittor | 两轮分别的加速比 |
| --- | ---: | ---: | ---: | --- |
| Wan2.2-TI2V-5B，一次条件 DiT 前向 | 1.49814 s | 1.21174 s | **1.236×** | 1.239 / 1.233 |
| SDXL base，一次条件 UNet 前向 | 0.64178 s | 0.63499 s | **1.011×** | 1.008 / 1.012 |
| SigLIP2 base vision | 7.03044 ms | 6.66336 ms | **1.055×** | 1.039 / 1.064 |
| DINOv2 base vision | 86.39834 ms | 45.33810 ms | **1.906×** | 1.910 / 1.900 |

原生采用本轮较快已测配置：Wan packed SDPA，其余 packed eager；Jittor 均 packed
SDPA。每个独立进程预热 5 次、计时 7 次，两个固定 NumPy 输入 slot；第二轮反转
每对 runtime 的顺序及 case 顺序。主表合并每侧 **14 个同步时延后取中位数**。
Wan / SigLIP2 / DINOv2 耗时降低 19.1% / 5.2% / 47.5%；SDXL 仅低 1.1%，
两轮约 0.8–1.2%，接近持平，不外推为普遍领先。

16 份 runtime 报告全部记录最终提交及空 diff 摘要
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。
每轮四对、共八份完整输出对拍全部 PASS，Jittor fallback 均为 0：

| 组件，跨两轮 / 输出 / slot | 最大相对 L2 | 最小余弦相似度 |
| --- | ---: | ---: |
| Wan | 0.00420069 | 0.99999118 |
| SDXL | 0.00032094 | 0.99999995 |
| SigLIP2 | 0.00750592 | 0.99997183 |
| DINOv2 | 0.00861022 | 0.99996293 |

组件阈值为相对 L2 ≤ 0.02，同时检查 shape、finite 与实际设备驻留。计时排除模型
加载、首次传输、JIT、权重预转换和 D2H 保存。Wan 为 320×512、17 帧对应的
`[1,48,5,20,32]` latent、800 self-attention token；SDXL 为 1024² 对应的
`[1,4,128,128]` latent；二者仅一次条件前向，不含 CFG 另一分支、文本、调度或 VAE。
SigLIP2 / DINOv2 分别为 224² / 518² 合成输入，两轮复用同两组输入。

## 真实生成：五个成功配置与一个默认失败

以下均来自最终 clean 提交，使用已提交 portable tool 的 `--timing total`：
不挂内部 forward hooks，仅起止同步；相同 prompt、seed `20261020`、初始 NumPy
噪声、20 Euler 步、CFG=5，每侧 **1 次完整预热 + 1 次计时**。共有五个成功进程，
下表两条 Wan 对比复用同一原生样本；这些不是多样本尾延迟或稳健生成排名。

| 生成任务 | 原生 | Jittor | Jittor 输入 / retention 预算 | 本次观察 |
| --- | ---: | ---: | --- | --- |
| SDXL 1024² 完整出图 | 26.83742 s | 27.04501 s | 4 MiB / **1 MiB scoped** | 慢 0.77%，基本持平 |
| Wan 320×512×17，从预编码条件开始 | 64.25871 s | 54.26700 s | 默认 4 MiB / 256 MiB | **1.184×**，耗时低 15.55% |
| 同一 Wan，调高输入预算 | 64.25871 s | 49.12329 s | **8 MiB** / 256 MiB | **1.308×**，耗时低 23.55% |

SDXL 包含文本编码、latent 准备、去噪、VAE 和 NumPy 后处理，排除加载与产物写盘。
Wan 使用同一个原生 UMT5 预编码 archive；计时含 scheduler / 去噪、VAE 与后处理，
**排除文本编码、archive 加载和初始条件传输**，不能称端到端文生视频时间。
所有成功输出 shape 正确、全部 finite，Jittor fallback=0；原生无同一回退计数器，
JSON 记录 null，不写成测得 0。

SDXL 初始噪声逐元素相同，图像 PSNR **38.183 dB**、像素相对 L2 **0.0419456**、
MAE 0.00523272，最终 latent 相对 L2 **0.0233136**。直接查看新配对 PNG，主体与
构图接近；不将完整生成套用组件 2% 阈值，也不称严格数值 PASS。

Wan 两配置的初始 latent、最终 latent 与 decoded 数组均逐字节相同，调 flag 的收益
没有改变输出。相对原生，像素 PSNR **23.418 dB**、相对 L2 **0.136817**、
MAE 0.0397984，最终 latent 相对 L2 **0.141551**。新 contact sheet 第 0 / 8 / 16
帧两侧均严重高饱和、彩色扭曲，**视觉质量仍失败**。这是低分辨率、20 步、FP16
主体的诊断配置，没有 GT / FID 或数据集质量验收，不宣传为可用质量视频。

### 默认 SDXL total-only 的已知限制

同一 0e0、1024²、CFG=5、输入 4 MiB / retention 256 MiB 默认配置，在首轮 warmup
的 FP32 GroupNorm `[1,256,512,512]` 处报
`207007 ACL_ERROR_RT_NO_EVENT_RESOURCE`，没有有效 total 时延。原 failed JSON 与
日志完整保留，不用旧 instrumented 时间替代，也不以单条件组件 graph active 宣称
该完整路径成功。

上表成功的 SDXL 是独立进程外围 scope **收紧 retention 为 1 MiB** 的 workaround，
使用同一 public tool、数学和精度，保留 executor replay。两个 CLIP / UNet 的
设备录制分别因估计 6.6 / 8.2 / 175.6 MiB > 1 MiB 被拒绝；它不是默认 256 MiB
配置成功，也没有放宽资源预算。

该问题独立登记为 [KI-BACKEND-023](../../agent/manuals/known-issues.md#ki-backend-023-sdxl-automatic-device-recording-can-exhaust-310p-event-resources)，
owner 为 ACL / graph replay 维护者。SDK 报告 event 资源不足，具体 recording 交互
尚未隔离；不据 workaround 成功断言根因。退出条件：默认配置完成预热与计时、shape /
finite / 配对输出及零回退检查，并有真实 NPU 资源边界回归，或在捕获前安全拒绝
该 device graph 而保留 executor 路径。CANN / 驱动或 graph 资源管理变化时复查；
不将其归为硬件缺少某种 dtype，也不凭 workaround 成功宣称默认问题已修复。

## 六项优化与保真边界

**1. Native tanh GELU。** 将 `approximate='tanh'` 的展开数学接入对应 CANN native
实现，保持同一近似与精度。FP16 / FP32 前向和定向梯度有真实 NPU 验证；代表 Wan
算子从约 12.199 ms 降至 2.049 ms，完整模型收益以组件表为准，不使用微基准倍率
代替整模时间。GELU-only 并未让 SDXL / DINOv2 明显更快。

**2. Stock RMSNorm。** 新 ACL 推理路径限定 FP16、无梯度、最后单轴、连续输入、
D≥16。当前 SDK / 310P 的小宽度多行输入实测 **49 / 336 条错误**：42 shape、
2 slot、mapped / code / poisoned code / fresh-descriptor 四种方式，错误分别
13 / 12 / 11 / 13 条，均为 D<16 且多行。底层 runner 对该子集明确拒绝，公共
路径保留已有数学实现；拒绝测试不计为 SDK 数值支持，不能从大宽度成功外推。

**3. 精确除以 `+1.0`。** 对 ACL FP16 / FP32 连续浮点非原地真除法采用独立复制，
保留值、输出独立性、梯度、signed zero 和 subnormal；strided 等走原分支。
原生 torch_npu 在该设备的 FP16 `/ +1.0` 将 NaN / +Inf 饱和到 +65504、-Inf 到
-65504，而独立 Torch CPU 与原生 clone 保留特殊值。Jittor 遵循 CPU 数学结果，
不复刻饱和错误，不能声称所有特殊值与原生 NPU bit-exact。
36 个 fixture 为 18 CPU / 18 ACL，且 ACL 组含显式 CPU 拒绝边界，不称 36 项 NPU 数学。

**4. 只读 padding 常量缓存。** 只缓存 shape / dtype / device 对应的零填充尾部，
不缓存输入或 attention 输出；直接 / replay 两输入逐字节一致。上限 128 项、
64 MiB、单项 8 MiB，含驱逐 / 驻留恢复边界；与下一项有测试重叠，不加总节点。

**5. 无 mask、无 padding 的 raw PFA。** 对已验证 noncausal 输入使用 SDK 允许的
`mask=null` 与原始实际 shape，mode=0、scale 不变，无新属性；旧 causal 四输入
约束保留。旧非空 mask 的矩形 / 未对齐调用明确拒绝，不忽略调用者 mask 或回退 CPU。
8 shape、两 slot、48 条独立 CPU FP32 对拍和 16 条 constant-V / backing-poison
检查通过。当前组件结果不外推训练、任意 dtype / head dimension 或 mask 支持。

**6. 等待稳定参数后自动 replay。** Stateless signature 加入参数 / buffer holder
identity 与 current Var pointer；已有 capture 时复用 `cap.params`，无 capture 时
收集 `_module_vars`。stateful 路径、显式 GraphReplay API 和 retention 契约保持。
VAE 每轮 force-up/downcast 原先导致反复 capture；同一 VAE-only 裸流程的 decode
从 4.83957 s 降至 dirty 修复探针的 0.94763 s，保持 auto replay=1，因参数不稳定
seen 停在 1、不 capture；与 noReplay arm 三组输出 byte-exact。
新稳定性检查有少量 host 开销，最新 SigLIP2 已包含该代价，不沿用 f9 更快数值。

VAE-only 探针为 5 warm / 5 samples，hooks / op profile 另跑。旧 profile 的
F32 Conv 158、GroupNorm 133、两组 BMM 各 18 次，修复后为 30 / 25 / 各 3，
不能把重复执行计数当单次 decoder 热点或断言 D=512 attention 是主因。
`vae-parity.json` 严格 passed=false 保留：unscale latent 非 byte-equal（相对 L2
0.00006924）；decoded 相对 L2 0.000914、PSNR 69.789 dB 仅作诊断吻合。

LayerNorm 辅助统计量候选**未采用，无生产改动**：主 y 的 24 组对拍 byte-equal，
但代表耗时约 ±1%、无稳定收益；FP32 stats 的 rstd 未过独立 1e-4 oracle，
422 / 1024 项不符，不能写成新增 FP32 stats 支持。

## 阶段来源与归因

iter0–5 基于同步提交 `1f213177d76671bc59e7816707641914aad5baec`，下表仅列代表
已测值与 dirty hash 摘要；完整 hash、输入、采样和数组在逐阶段 JSON。
未测阶段不推算，iter4 没有单独保存对拍，不反向归用 iter5 的 PASS。

| 阶段 | 变更 / 代表 Jittor 时间 | source / patch 摘要 |
| --- | --- | --- |
| iter0 | Wan 2.01209 s；SDXL 0.75974 s；Sig 11.41501 ms；DINO 52.30873 ms | 基线 clean，空 diff |
| iter1 | tanh GELU：Wan 1.68751 s；Sig 8.26520 ms | dirty `8bfb5b17…` |
| iter2 | RMSNorm：Wan 1.30702 s | dirty `1aae2761…` |
| iter3 | unit division：SDXL 0.71818 s | dirty `5fd0ebc7…` |
| iter4 | padding 缓存：Wan 1.29397 s；Sig 7.09113 ms | dirty `ef9b028c…` |
| iter5 | raw PFA：Wan 1.21355 s；SDXL 0.63418 s；Sig 6.35479 ms；DINO 44.64009 ms | dirty `29b16429…` |
| f9 归档 | 默认组件 14 样本：Wan 1.21024 s；SDXL 0.63293 s；Sig 6.36184 ms；DINO 44.46393 ms | clean `f9dde035…` |
| 稳定性探针 | VAE-only decode 0.94763 s，流程 1.01502 s | f9 + dirty `c404106a…`，不改记为 clean |

f9 旧完整生成使用内部 hooks / 逐步同步的 instrumented path，去噪器 pre-hook 会
绕过兼容层外层 auto replay；只保留以下两行用于定位，不与新版 total-only 混排名：

| f9 instrumented 样本 | 原生 / Jittor 合计 | 原生 / Jittor VAE + 后处理 |
| --- | --- | --- |
| SDXL 1024²、20 Euler、CFG=5 | 26.86660 / 30.80546 s | 1.00766 / 4.04702 s |
| Wan 预编码条件、320×512×17 | 64.57587 / 54.22255 s | 3.97288 / 5.66706 s |

早期 warmup=3 的逐算子 profile 含录制尝试，部分 Count 约两倍；profile 后改为
至少 5 次预热。嵌套 hooks 是 inclusive，不能与裸样本相加，或当作稳态前向计数。

## 环境、复现与验证

单张 Ascend 310P3、44 GB；驱动 26.1.0.b050 / CANN 9.1.0。原有后台服务约占
2.7 GB 显存且保留，未取得物理独占。原生 / Jittor 独立解释器和缓存，单卡顺序运行，
首次 JIT 串行，接受性能样本时无并行 CPU 编译或其他本任务 NPU 作业。
原生 PyTorch 2.12.0+cpu + torch_npu 2.12.0；Transformers 4.56.2、Diffusers 0.35.1。
FP16 主体，保留 Wan 官方 FP32 部分与 FP32 RoPE、SDXL 官方 force-upcast，HF32
策略相同。Checkpoint 沿用[前轮报告](2026-10-21-generative-vision-ascend310p-performance.md)
固定快照，前轮核验 17 个 safetensors，完整摘要见 `weight_hashes.json`。

原生设 jit_compile=False，harness 未显式调用 torch.compile 或创建原生 device graph；
Jittor 保留自动 replay。组件默认输入 4 MiB / retention 256 MiB；Sig / DINO / SDXL
两轮均 device_graph_active=true，Wan seen=0。Wan 递归输入共 4,504,704 B 超过
4 MiB，更新 seen 前返回；外层深度守卫仍在，不能推断子模块已有部分 auto replay。
8 MiB total-only Wan 的 transformer seen=80、captured=1、replayed=79、graph=0；
device graph 因估计 1782.1 MiB > 256 MiB 被拒绝，收益来自 executor replay。
同提交的额外 Wan 组件实验先测 8 MiB、再测 4 MiB，各 6 次预热、7 次计时、
2 个 slot，分别为 1.10174 / 1.21345 s，独立原生对拍通过；不混入默认主表的
14 样本。两个输入门槛实验都保持 256 MiB recording 预算。

复用[性能 skill](../../agent/skills/jittor-transformers-perf/SKILL.md) 的
`benchmark_generative_vision_ascend.py` / `compare_generative_vision_ascend.py`，
两侧同 checkpoint、pack、device=npu、warmups=5、samples=7、slots=2；原生 attention
取 eager（Wan 为 sdpa），Jittor 为 sdpa，按上述两轮顺序运行并保存全部输出。
完整生成使用 `benchmark_diffusion_ascend.py --timing total`，指定相同 prompt / seed /
steps / guidance；Wan 增加 `--prompt-embeddings`，输入预算用
`--auto-graph-replay-input-mib 4` 或 `8`，不修改生产默认值。
SDXL workaround 在已激活 Jittor frontend 的同一进程以
`jt.runtime.scope(auto_graph_replay_retain_bytes=1 << 20)` 包围相同 public tool，
实际 policy、tool / wrapper 来源记录于 JSON / 日志；不能只传输入预算便宣称已收紧 retention。

| 验证会话 | 实际结果与范围 |
| --- | --- |
| NN / ACL 原生专项 | 96 passed、0 skipped，19.73 s；含 GELU、PFA、mapped RMS 与拒绝边界 |
| 兼容层专项 | 86 passed、0 skipped，19.10 s；stock RMS 50 + unit division 36（混合 CPU / ACL fixture） |
| 浮点分类独立设备探针 | FP16 / FP32 × 五种 predicate 共 10 组合 PASS，fallback=0；不计为 pytest 节点 |
| 新 policy 全 native CPU graph 模块 | 31 passed、31 accelerator skipped，30.14 s；旧 policy 的两项新回归确实失败 |
| 新 policy CPU core | Native 98 passed、41 skipped、1 xfailed，2.53 s；Torch 224 passed、1 skipped，7.45 s；exit=0 |
| 新 policy 选定真实 ACL graph | 7 passed、0 skipped，4.00 s，fallback=0；覆盖新两项及旧 keyword / param / buffer / stateful |
| 严格全量结构 | Torch 1537 passed、4 skipped，136.35 s；Native dtype recorder 5 passed、0 skipped，0.77 s；require_execution=1 |
| 代码阶段静态检查 | 布局 300 active Markdown、links 172、manifest、diff 检查通过；f9 的 16 个、后续 3 个 Python 文件通过 Python 3.7 AST |
| 本报告文档检查 | 布局 301 active Markdown、默认 172 文件及本报告 / 索引 / 问题总账 3 文件的链接检查、manifest 与 diff 检查通过 |

各会话有重叠，不简单相加；拒绝边界也不计为数学支持。新 core Native 的 41 项
skip 为 32 accelerator / 7 opt-in / 2 insufficient-devices，另有 1 expected failure；
Torch core 的 1 项为 accelerator。结构 4 项为 2 个 resolver 位置声明和 2 个 CUDA
不可用，Native recorder 是静态检查。f9 还完成 core + float-class 扩展 105 pass /
48 skip / 1 xfail；它不与新版纯 core 98 的计数混用。
预算报告另修复默认 configured workers 与 cgroup capped runtime workers 混淆，
默认配置仍为 4，预算 / worker 设置未调；28 项回归已包含在全量结构，不重复相加。

原始证据均未版本化，位于 `$JITTOR_LAB_ROOT/_state/generative-vision-310p/`：

- `bench/iter-stable-final-r*.json(.npz)`、逐轮 parity、`iteration-2026-10-22/stable-final/balanced-summary.json`：新版 16 报告、8 对拍与 pooled 14 样本。
- `iteration-2026-10-22/stable-final/total-*.json(.npz)`、三份 `parity-total-*.json`、`generation-summary.json`、新 side-by-side / `wan-total-only-contact-sheet.png`：五个成功 arm、保留的默认 SDXL failure、PNG / GIF / 帧与质量检查。
- `bench/jittor-iter*.json(.npz)`、`bench/iter-final-r*`、旧 effects：dirty ablation 与 f9 clean 归档，保留原 source / patch，不覆盖。
- `iteration-2026-10-22/{gelu-tanh-*,rms-stock-*,rms-small-width-*,divide-one/,pfa-padding-cache.json,pfa-rect-unpadded-*}`：单算子、49 / 336 SDK 异常、特殊值差异与独立参考。
- 同目录 `vae-*.json(.npz)`、`wan-replay-limit-*`、`layernorm-optional/`，以及 `stable-final/stable-wan-replay-limit-*` 和 `logs/iter-*` / `replay-stability-*`：因果探针、严格 parity=false、未采用候选与全部门禁日志。

本报告限于上述单卡、固定 checkpoint / 精度 / shape 和推理模式，不外推训练、
并发服务、多卡、CUDA / ROCm、未测模型或数据集级生成质量；前轮 SAM 的 NPU
验收仍仅图像编码器，CPU 猫图掩码不改称为 NPU 完整分割。
