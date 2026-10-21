# Wan2.2、SDXL 与视觉模型：Ascend 310P3 推理验证

- Status: 本报告范围的组件数值 / 性能、实际生成与代码门禁完成；Wan 两侧均有严重伪影，最终视频差异显著，不记为质量通过；SAM NPU 仅覆盖图像编码器
- Last reviewed: 2026-10-21（America/Los_Angeles）
- Source baseline: `2e031215f6c466419dc2a16f2154e0bbff667bed`，已整合远端 `897d9cc61d9f0d9ddc6e51d02da5289cf56e085e`
- Final implementation: `a1a07b84eef858839c5ea838a4e1388330870edd`
- Measurement provenance: 主表四组件在最终 clean commit 复测；历史优化阶段与实际生成仍以各 JSON 的原基线 / `dirty_sha256` 为来源，不能改记为 clean commit 的生成实测
- Owner: ACL backend and Torch compatibility maintainers
- Review when: CANN、驱动、Torch / torch_npu、Transformers / Diffusers、checkpoint、精度策略、attention 派发或 packed Linear 实现改变

## 已验证结果与计时范围

最终 clean commit 上，完整 DINOv2-base vision encoder 从历史 packed 基线
**105.10 ms 降至 51.59 ms**，加速 **2.04 倍**；相对本次最快已测原生 torch_npu
的 86.36 ms，加速 **1.67 倍**、耗时降低 **40.3%**。SDXL 单次条件 UNet 前向
为 **0.75983 s**，比原生的 0.64172 s 长 **18.4%**。Wan 优化后为 **2.00651 s**，
相对修复正确性后的 packed 基线 5.78645 s 加速 **2.88 倍**、耗时降低约 **65.3%**，
仍比最快已测原生的 1.49627 s 长 **34.1%**。四个完整组件
均已与各自最快已测原生配置完成数组对拍，Jittor fallback 尝试计数均为 0。

实际生成另行验证：SDXL 配对图片 PSNR 为 38.85 dB；Wan 视频为 21.58 dB，
两侧都有严重过饱和、彩色纹理与形变，细节差异明显。Wan 完整执行仅通过 shape、
finite 和 Jittor 零回退检查，不能把单次 DiT 组件对拍通过等同于最终视频质量通过。

以下主表前四行为 **最终 clean commit 的重新采样**；Wan 已加入 RoPE 优化，
四模型均已接入 PFA 派发。
“Jittor / 原生”大于 1 表示 Jittor 耗时更长。两侧均 packed，Jittor 使用 SDPA
派发；原生各行选择本次
实测中较快的 attention 配置，不能用较慢的原生 SDPA 配置宣称领先。

| 完整预训练组件，batch=1 | 输入与范围 | 原生 attention | 原生中位数 | Jittor 历史 PFA 接入前 | Jittor 最终 clean 中位数 | 最终 Jittor / 原生 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Wan2.2-TI2V-5B DiT，4.9998 B 参数 | 320×512、17 帧对应 latent；一次条件前向，800 self-attention token | SDPA | 1.49627 s | 2.58278 s | 2.00651 s | 1.341 |
| SDXL base UNet，2.5675 B 参数 | 1024² 对应 `[1,4,128,128]` latent；一次条件前向 | eager | 0.64172 s | 1.18187 s | 0.75983 s | 1.184 |
| SigLIP2 base vision，92.884 M 参数 | 224² 图像，vision encoder | eager | 7.05662 ms | 10.60919 ms | 10.85594 ms | 1.538 |
| DINOv2 base，86.580 M 参数 | 518² 图像，vision encoder | eager | 86.35812 ms | 105.09653 ms | 51.58603 ms | 0.597 |
| SAM2.1 Hiera small | 仅 1024² 图像编码器、三个 image embeddings；不含提示 / mask decoder | 原生 NPU 尚无有效结果，见后文 | — | — | 422.18065 ms（独立 FP32 轮） | — |

Wan 此前通过 chunk views 将 5.786 s 降至 **2.583 s**，加速 **2.24 倍**、
耗时降低 **55.4%**；该结果仍然有效。只新增 PFA 时 Wan 中位数为 2.775 s，
比此前长 7.4%；历史 SigLIP2 PFA 轮为 11.08697 ms，比此前长 4.5%。
这些历史差异尚未以交错重测区分回归与运行波动，不能宣称 PFA 对所有模型都有
完整组件收益。最终 clean 轮 SDXL / DINOv2 相对各自历史 packed 基线耗时降低
35.7% / 50.9%；SigLIP2 最终轮仍比历史基线长 2.3%。
历史 Wan RoPE 轮从 2.775 s 降至 2.00454 s，完整组件加速 1.38 倍；
最终 clean 轮 2.00651 s，约为原始正确 packed 基线的 2.88 倍加速。

最终 clean 结果的 p95 为 Wan **2.00981 s**、SDXL **0.76015 s**、
SigLIP2 **10.98346 ms**、DINOv2 **51.65238 ms**。

组件表使用固定 NumPy seed 生成的合成输入；真实图片 / 视频效果另节报告。
前四行 accepted packed 结果预热 5 次、计时 7 次，交替使用两个不同输入 slot，
每次调用同步完成，另存 cold 时间；模型加载、输入初次传输、首次 JIT、权重预转换、
D2H 保存均不在稳态计时内。组件参数数目在 packed 转换前统计。
Wan / SDXL 行只有一次条件去噪前向，不包含 CFG 的另一分支、20 步调度、文本编码、
VAE 或文件保存，不能作为完整视频 / 图片生成时间。
SAM 图像编码器为独立 FP32 验证，预热 1 次、计时 3 次、单个输入 slot，
p95 为 422.89713 ms；与 CPU 参考只比较输出，不计算跨设备速度比。

为解释原生配置选择，同样 packed 的原生 SDPA 中位数分别为：SigLIP2
9.39244 ms、DINOv2 148.44644 ms、SDXL 1.71656 s。Jittor packed eager
分别为 11.43993 ms、127.75051 ms、1.28306 s；这些替代配置均未优于主表选择。
Wan 目前仅比较已验证的 SDPA 路径。

## 正确性与环境

| 项目 | 配置 |
| --- | --- |
| NPU | 单张 Ascend 310P3，44 GB |
| 设备共享情况 | 原有空闲后台服务约占 2.7 GB 显存，全程保留；顺序运行本任务的 NPU 进程，未取得物理独占 |
| 驱动 / CANN | 26.1.0.b050 / 9.1.0 |
| Jittor | 本树 2.0.0，独立 Torch frontend |
| 原生参考 | 独立 PyTorch `2.12.0+cpu` + torch_npu 2.12.0；不经过 Jittor shim |
| Transformers / Diffusers | 两侧同为 4.56.2 / 0.35.1 |
| 执行方式 | `eval()` / `no_grad()`；原生 `jit_compile=False`，未调用 `torch.compile`；逐调用同步 |
| 推理精度 | FP16 主体，保留 checkpoint 指定的 FP32 部分；RoPE 显式 FP32；两侧允许 HF32 matmul |
| packed 策略 | 两侧显式快照 FP16 exact Linear；保留共享引用，保留 FP32 层，排除直接读取 `out_proj.weight` 的 MHA 叶子 |
| 设备与回退验证 | Jittor `use_cuda=1,use_acl=1`，输出实际驻留 NPU，`backend_fallback=error` 与计数守卫均启用 |

先在独立 CPU 环境使用相同 tiny checkpoint、相同输入完成五模型前向对拍；
最大相对 L2 约 `8.73e-7`，用于定位兼容路径，CPU fixture 的速度不参与主表。
随后加载全部预训练参数，在真实 NPU 比较两个不同输入 slot 的所有指定输出，
检查 shape、finite、相对 L2 和余弦相似度。以下为最终 clean 轮完整输出与主表
原生最快已测配置直接对拍：Wan 原生 SDPA，另外三个模型原生 eager；阈值统一为
相对 L2 ≤ 0.02：

| 模型 | 跨全部输出与 slot 的最大相对 L2 | 最小余弦相似度 | Jittor fallback 尝试次数 |
| --- | ---: | ---: | ---: |
| Wan2.2-5B，chunk + PFA + RoPE | 0.00418715 | 0.99999124 | 0 |
| SDXL UNet | 0.00031373 | 0.99999995 | 0 |
| SigLIP2 vision | 0.00734299 | 0.99997304 | 0 |
| DINOv2 base | 0.00861022 | 0.99996293 | 0 |

最终四组件 JSON 的提交均为 `a1a07b84eef858839c5ea838a4e1388330870edd`，
工作树 clean，`dirty_sha256` 均为
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
（空 diff）。上表误差来自该轮新保存的数组与独立原生最快配置直接对拍。

此前 chunk-only Wan 的同 SDPA 对拍最大相对 L2 为 0.00413593，原始证据独立保留。
历史 PFA 四模型轮的 `dirty_sha256` 为
`e1574b556c5ced967904c43e9fe1557cec5b5ca045cdfb69004433328302f4f6`；
历史 Wan RoPE 轮为
`137b9a490ef84a777a6b0f03f2f67dd39ef64822cee3ac06cd08253cd56afe9a`。
原生 Torch 不提供同一回退计数器，
其 JSON 的 `fallback_count=null`；不能将其写成测得 0。

全部 checkpoint 固定到以下快照；17 个 safetensors 的 SHA256 均与下载缓存 blob
摘要核对通过，完整文件摘要留在未版本化 `weight_hashes.json`。

| 模型仓库 | 固定快照 SHA |
| --- | --- |
| `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | `b8fff7315c768468a5333511427288870b2e9635` |
| `stabilityai/stable-diffusion-xl-base-1.0` | `462165984030d82259a11f4367a4eed129e94a7b` |
| `google/siglip2-base-patch16-224` | `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` |
| `facebook/dinov2-base` | `f9e44c814b77203eaa57a6bdbbd535f21ede1415` |
| `facebook/sam2.1-hiera-small` | `ee5bba1d82bb8749febdf90f45e84b687142ba03` |

DINOv3 权重访问受模型仓库权限限制，本次没有获得 checkpoint；没有绕过权限，
也没有把 DINOv2 结果标为 DINOv3 或外推到其他前沿模型。

## 修复与优化的实际依据

Wan 的首轮完整前向曾输出错误结果：相对 L2 约 2.805、余弦相似度约 0.013。
该次 `5.368 s` 记录保留作失败证据，排除在性能比较之外。
逐层追踪后，排队 GEMM、chunk 索引和 host 内存反复分配的最小复现确认：
单元素、非 scalar 的 ArrayOp host 源数据在异步传输完成前失效。ACL 执行器现按
`num == 1` 保留源数据生命周期，复现从 5/5 错误变为 5/5 精确正确；
随后四模型完整输出才通过上述对拍。

另修复了 Torch frontend 构造器未遵循默认 dtype、LayerNorm 禁用 affine 时应保留
`None`、Conv2d / Pool2d 列表几何参数，以及 SAM 触发的 CPU channels-last pooling
路径。Wan patch embedding 的 Conv3D 原先没有 ACL native 实现，退到 generic
reindex / reduce 并触发被禁止的后端回退；新增 NCDHW CANN convolution 路径后，
完整 `[1,48,5,20,32]` 输入、`[3072,48,1,2,2]` 卷积核的真实 NPU 探针通过，
fallback 为 0。这些属于运行正确性的前提，不以原先失败路径计算加速比。

310P 的 Conv3D SDK 分支仅在无梯度、输入 / 权重均停止梯度，或仅求 bias 梯度等
已验证边界使用；**若输入或权重需要梯度，明确抛出 `NotImplementedError`**。
32 项 Conv3D 回归中的 10 项验证的是这一拒绝行为，不是 310P 数值梯度通过；
另有无梯度、输入 / 权重均停止梯度、仅 bias 梯度三个边界测试，不能据前向性能
宣称 310P Conv3D 训练支持。

Wan 的 profile 中，单元素 chunk 分支使用 advanced indexing，产生大量 Gather
拷贝；普通 slice 已能表达相同区间且保持视图。修复这一通用分支后，六路 chunk、
squeeze、一次 affine modulation、FP16 cast 与同步的同形状微基准从 **168.525 ms
降至 1.379 ms**，所有分块共享输入存储。完整 Wan 前向随后从 5.786 s 降至
2.583 s，输出误差不变；**约 122 倍的微基准收益不能当成整模加速比**。

SDXL / DINOv2 profile 的 score elementwise 与 softmax 合计约占设备算子累计时间
38.5% / 41.1%。这些为逐算子同步的诊断数据，profile wall time 不进入性能表。
原先 PFA 派发仅覆盖 causal attention；新增 noncausal 候选先在真实 310P
使用完整 attention window、补齐 key mask，与独立原生 CPU FP32 eager 对拍。
生产入口新增范围限定为 FP16、无梯度、无显式 mask、Q / K 等长且等 head 数、
head dimension 为 64 或 128 的 attention；保留原有 causal / GQA 规则。
Q / K 长度不同、带 mask 和训练不因本次扩展进入新的 PFA 分支。
诊断形状的 attention 调用结果如下，预热 2 次、计时 3 次：

| attention 代表形状 | math 中位数 | PFA 中位数 | 单调用加速比 |
| --- | ---: | ---: | ---: |
| SigLIP2，S=196、D=64 | 0.61150 ms | 0.22192 ms | 2.76 |
| DINOv2，S=1370、D=64 | 7.55431 ms | 2.70463 ms | 2.79 |
| SDXL，S=4096、D=64 | 40.81943 ms | 16.40259 ms | 2.49 |
| Wan，S=800、D=128 | 4.30365 ms | 2.99268 ms | 1.44 |

上述 PFA 对拍均 finite、相对 L2 小于 0.0011、fallback 为 0；另验证 SDXL CFG
batch=2 与 batch=128 的代表形状。生产入口接入后的四模型结果见主表；
PFA 已包含在最终 NPU 回归通过范围内，并收录于最终实现提交。

Wan RoPE profile 的 120 次 slice assignment 曾展开完整坐标网格并进入 IndexPut。
符合条件的单轴正步长 slice 现在复用已有 IndexCopy 路径：仅限 310P、无梯度、
FP16 / FP32 同 dtype、精确 source shape、连续 base / source；多轴、整数索引、
广播和训练等原分支保持原语义。真实设备检查输出正确、输入 base 不变与结果独立，
随后在独立测试缓存运行 **76 项 NPU 回归，全部通过**，无跳过；完整 Wan 历史
RoPE 轮为 2.00454 s，最终 clean 轮为 2.00651 s，均与原生数组对拍通过，
误差与 PFA 轮相同。

实际 Wan 重复解码还暴露了通用 graph replay 正确性问题：首次 eager 预热会修改
传入的 Python cache / index 容器，后续捕获若再次用已修改的容器执行模块，就重复
累加索引并越界。运行时现在识别这类可变参数的 host 状态变化，返回已算出的正确
eager 结果并拒绝对该调用再次捕获；只读 Tensor 容器仍可进入重放。
该语义的 **4 项 CPU 回归通过**；同一已加载 VAE 在真实 NPU 连续两次解码分别为
6.33688 / 5.50321 s，输出逐元素相同，fallback 为 0。该探针只覆盖重复 VAE
解码，不代替完整 DiT / 调度 / VAE 视频生成结果。

## 实际图片、视频与分割效果

本节生成与 SAM 效果来自原基线 `2e031215f6c466419dc2a16f2154e0bbff667bed`
及各 JSON 对应的 dirty patch；它们保留原始来源，不记为最终 clean commit 的生成
复测。最终 clean 复测范围为上文四个组件。

SDXL 已完成两侧实际 1024²、20 Euler 步、CFG=5 图片生成，prompt 为溪边苔石上的
小熊猫、seed `20261020`；两侧初始 NumPy 噪声逐元素一致，使用同一 checkpoint、
调度器与 VAE 精度策略。原生采用较快的 packed eager，Jittor 为 packed SDPA / PFA，
输出像素与 latent 均为正确 shape、全部 finite，Jittor fallback 为 0。

| 带同步插桩的完整 SDXL 生成阶段 | 原生 torch_npu | Jittor |
| --- | ---: | ---: |
| 文本编码与准备 | 0.06851 s | 0.78789 s |
| 完整去噪循环 | 25.75206 s | 31.10096 s |
| VAE 解码与 NumPy 后处理 | 1.00134 s | 3.90121 s |
| 完整生成 | **26.82191 s** | **35.79007 s** |

两侧均完整生成预热 **1 次、计时 1 次**；本轮 Jittor 完整生成耗时比原生长
**33.4%**。这些是一次预热后的诊断样本，包含公开 hook / 每步 callback 的同步开销，
排除模型加载与产物写盘；不能当作 7 样本稳态分布，也不能从单样本 p95 推断尾延迟。

像素 PSNR **38.8528 dB**、相对 L2 **0.0388329**、平均绝对误差 **0.00487545**
（像素范围 `[0,1]`）；最终 latent 相对 L2 **0.0185146**。并排图左为原生、右为
Jittor，人工查看外观接近。对拍显式允许所选 eager / SDPA 实现不同；调度器配置
比较只规范化 `_use_default_values` 这类无序元数据列表的顺序，没有改变运行时配置。
单 prompt / seed 的实现间数值一致性不等于有标注的数据集质量评估。

此前原生 SDPA 的 **81.193 s** 记录属于无预热、单样本的 cold preflight，生成时
另有 CPU 验证任务；保留作先前产物证据，不参与上述最终配置的速度比较。

Wan 首次同驻留 UMT5、DiT、VAE 的全 NPU 生成在 VAE 解码时显存不足。
原生 UniPC 还出现线性求解 CPU 回退，故失败运行不记为全 NPU 成功或正式速度。
固定 Diffusers 0.35.1 的 Wan2.2 `patch_size=2` tiled decode 跳过 unpatchify，
本次工具明确拒绝该组合，未用修改后的 VAE 伪造成功结果。

为测可复现的真实视频生成，原生 FP32 UMT5 已在独立 NPU 进程完成正 / 负文本条件
编码，耗时 **1.892 s**，保存共享 `wan-prompts.npz` 与摘要。两侧随后使用同一
archive，仅加载 DiT 与 FP32 VAE，使用 20 步 FlowMatchEuler、shift=5、CFG=5，
输出目标为 320×512、17 帧。修复 graph replay 的可变参数问题后，两侧均实际完成
**1 次完整生成预热、1 次计时**，保存 `[1,17,320,512,3]` 视频数组、GIF 和关键帧；
初始噪声逐元素相同、输出 shape 正确且全部 finite，Jittor fallback 为 0。

| 从共享预编码条件开始的 Wan 生成阶段 | 原生 torch_npu | Jittor |
| --- | ---: | ---: |
| 准备 | 0.00904 s | 0.01704 s |
| 完整去噪循环 | 60.04517 s | 80.70927 s |
| VAE 解码与 NumPy 后处理 | 3.95118 s | 5.64228 s |
| 生成阶段合计 | **64.00539 s** | **86.36858 s** |

Jittor 该样本比原生耗时长 **34.9%**。这同样是一次预热后的同步插桩诊断样本，
不代表多样本延迟分布。两侧最终 latent 相对 L2 **0.182447**、余弦相似度
**0.983451**；像素相对 L2 **0.169155**、MAE **0.0499367**、PSNR **21.5755 dB**、
余弦相似度 **0.985662**。单次去噪器约 0.0042 的误差没有保证多步输出仍接近，
最终视频差异必须独立报告，不能用组件对拍的 0.02 阈值宣布视频数值一致。

人工检查第 0 / 8 / 16 帧左右对照：两侧均出现严重过饱和、彩色纹理与形变，
构图大致相似，细节差异明显。**当前 320×512、17 帧、20 Euler 步、FP16 主体
是低分辨率诊断配置；该结果不能称为正常质量通过、视觉接近或推荐配置的质量验证。**
没有人工标注真值、FID 或数据集级视频质量评估，也尚未定位两侧伪影及差异的全部
原因；不把它们直接归因于某一个算子或单一优化。

首次 Jittor 完整生成曾在第二次 VAE 解码发生 `feat_cache` 越界，该次失败排除在
速度比较外。nearest-exact、Tensor 字符串比较与 graph replay 修复后的正式重测
采用源补丁摘要 `90e52b2456b956070898934e6cbf1454e4082946a566a1abf5ca9f5cf4a06c61`，
不以单次 VAE 成功代替上述完整重复生成验证。
该模式的计时包括调度 / 去噪、VAE 与 NumPy 后处理，排除文本编码、archive 读取、
初始条件传输、模型加载和文件保存；即使成功也只能称为“从预编码条件开始生成”，
不能称为端到端文生视频，也不能证明 Jittor UMT5 已验收。

SAM2.1 已使用真实猫图、中心正点提示生成三个候选掩码；独立原生 **CPU FP32**
参考一次前向为 1.217 s（无预热、单样本，cold 另为 1.251 s），保存完整 logits、
置信度、二值掩码与叠加图。最高模型预测 IoU 为 0.9747，属于模型置信度，并非
对人工标注测得的 IoU。原生 NPU FP16 在 MaxPool 不支持该 dtype 处失败，FP32
重试在 GELU memory_format 不支持处失败，未修改原生算子实现隐藏这些限制。
Jittor 在补齐 bicubic 路径后，完整预训练 **图像编码器**已通过 NPU FP32 前向；
三个公开 image embeddings 与独立原生 CPU 同输入参考的最大相对 L2 为
**0.00246298**、最小余弦相似度 **0.99999699**，fallback 为 0。该组件使用固定
合成输入，不能将它标为猫图 NPU 掩码效果验证。完整 NPU mask decoder 仍不支持，
因此本次 SAM 的 NPU 结论仅覆盖图像编码器，CPU 猫图掩码单独作为效果参考；
没有 NPU 完整掩码结果或同设备速度比。

## 复现与证据

按 [310P 环境指南](../guides/ascend-310p.md) 准备独立解释器、相同版本下游包及上表
固定 checkpoint；由外部选择可用设备。测试与 benchmark 分配不同缓存，首次 JIT
与单卡工作顺序运行。复用
[性能 skill](../../agent/skills/jittor-transformers-perf/SKILL.md) 的现有入口，原始结果
必须位于 checkout 外的 `$JITTOR_LAB_ROOT`。以下变量由运行环境提供：

```bash
SCRIPT_ROOT=agent/skills/jittor-transformers-perf/scripts
RUN="$JITTOR_LAB_ROOT/_state/generative-vision-310p/reproduce"
mkdir -p "$RUN"

JITTOR_TORCH_SHIM=0 "$TORCH_PYTHON" "$SCRIPT_ROOT/benchmark_generative_vision_ascend.py" \
  --backend torch --device npu --case dinov2 --model "$DINOV2_MODEL" \
  --dtype float16 --attention eager --pack --warmups 5 --samples 7 --slots 2 \
  --output "$RUN/torch-dinov2.json"

JITTOR_TORCH_SHIM=1 JITTOR_HOME="$RUN/jittor-bench-cache" \
  cache_name=generative_vision backend_fallback=error \
  "$JITTOR_PYTHON" "$SCRIPT_ROOT/benchmark_generative_vision_ascend.py" \
  --backend jittor --device npu --case dinov2 --model "$DINOV2_MODEL" \
  --dtype float16 --attention sdpa --pack --warmups 5 --samples 7 --slots 2 \
  --output "$RUN/jittor-dinov2.json"

"$NUMPY_PYTHON" "$SCRIPT_ROOT/compare_generative_vision_ascend.py" \
  "$RUN/torch-dinov2.json" "$RUN/jittor-dinov2.json" \
  --max-relative-l2 0.02 --output "$RUN/parity-dinov2.json"
```

`--case` 可改为 `siglip2`、`sdxl` 或 `wan`，并传对应 checkpoint；Wan 原生使用
`--attention sdpa`。tiny CPU fixture 验证须先完成，但不能作为预训练速度测量。
完整生成使用 `benchmark_diffusion_ascend.py`，SAM 使用
`benchmark_sam2_ascend.py`，参数与阶段边界见 skill。复现完整生成时应分别报告
cold、完整生成预热 / 样本数和实测值，不能从单次去噪前向相乘推算。

本次未版本化证据位于 `$JITTOR_LAB_ROOT/_state/generative-vision-310p/`：

- `bench/torch-packed-{eager,sdpa}-*.json(.npz)`、`jittor-clean-final-sdpa-*.json(.npz)`：主表原生最快已测配置与最终 clean 组件的原始采样、版本、提交及空 diff 摘要。
- `bench/parity-clean-final-*.json`：最终 clean 四组件对各自最快已测原生配置的全部输出对拍。
- `bench/jittor-pfa-packed-sdpa-*.json(.npz)`、`jittor-rope-pfa-packed-sdpa-wan.json(.npz)`、`parity-pfa-best-*.json`、`parity-rope-pfa-wan.json`：历史 PFA / RoPE 轮的独立采样、dirty patch 与对拍证据。
- `bench/jittor-packed-sdpa-*.json(.npz)`、`jittor-chunk-packed-sdpa-wan.json(.npz)`、`parity-packed-*.json`、`parity-chunk-wan.json`：PFA 接入前与 chunk-only Wan 的独立证据。
- `profile/{wan,sdxl,siglip2,dinov2}-packed.json`、`profile/chunk-{vs-slice,after}.json`：逐算子诊断与 chunk 最小复现。
- `pfa-noncausal/` 与 `logs/pfa-noncausal-full-window-marked.log`：独立原生 FP32 attention 参考与 NPU 微基准。
- `bench/rope-assignment-*.json`、`logs/rope-slice-tests.log`：RoPE slice / IndexCopy 最小复现与 76 项真实 NPU 回归。
- `effects/{torch,jittor}-sdxl-final.json(.npz)`、`parity-sdxl-final.json` 与对应 PNG：最终配置的一次预热后完整生成样本、latent / 像素对拍及并排图。
- `effects/torch-sdxl-preflight.json(.npz)` 及其 `artifacts/`：原生 cold 图片，不能当作正式稳态基线。
- `effects/{torch,jittor}-wan-final.json(.npz)`、`parity-wan-final.json` 与 GIF / 帧：两侧一次预热后的预编码条件生成，包含执行验证和显著的输出数值差异，不作质量通过结论。
- `effects/wan-final-contact-sheet.png`：第 0 / 8 / 16 帧左右比较，左为原生、右为 Jittor，两侧伪影与细节差异可见。
- `effects/vae-twice-{before,after}.json`、`logs/vae-twice-{before,after}.log`、`logs/graph-replay-mutable-cpu.log`：重复 VAE 解码最小复现、修复后的两次精确输出与 4 项 CPU 回归。
- `effects/torch-cpu-sam2.json(.npz)` 及其 `artifacts/`：CPU 猫图掩码参考。
- `bench/{torch-cpu-sam2-image,jittor-sam2-image-after-bicubic}.json(.npz)`、`sam2-image-parity-cpu-reference.json`：SAM 仅图像编码器的跨设备数组对拍，不能用来计算 NPU 同设备速度比。
- `effects/wan-prompts.npz(.json)`、`model_paths.json`、`weight_hashes.json`：共享文本条件与固定 checkpoint 身份。

## 验证边界与最终门禁

最终 CPU 定向兼容回归 **117 passed，4 deselected，0 skipped，73.03 s**，
覆盖构造器 dtype / alias、LayerNorm、列表几何参数、NHWC pooling、chunk views、
nearest-exact、Tensor 字符串比较与 graph replay 可变参数。4 个 deselected 为
另在 NPU 会话执行的 nearest-exact 设备 case，不能当作 CPU 已执行节点。
此前 48 项为分批专项记录；graph replay 4 项已包含在最终 117 项中。
不同会话的重叠节点不另行相加。
五模型 CPU fixture 与四模型完整 NPU 数组对拍通过。真实 NPU 已复现并验证 Conv3D native 路径、
单元素 host 生命周期以及 chunk 性能 / 视图语义；RoPE 单轴 slice assignment
优化的独立 NPU 回归 **76 passed，0 skipped，282.42 s**，完整 Wan DiT 再对拍通过。
SDXL 两侧完整生成完成且配对指标已记录，采样范围为各 1 次预热 / 1 次计时；SAM
仅图像编码器 NPU / CPU 参考对拍通过，完整掩码仍不在 NPU 验收范围。
Wan 重复生成问题已修复，完整 1 次预热 / 1 次计时执行完成，Jittor fallback 为 0；
最终视频两侧均有严重伪影，且数值差异显著，**不作视频质量通过结论**。Graph replay 可变参数专项
**4 passed，0 skipped，3.06 s**，不与其他 CPU 结果重复相加。

最终门禁结果如下；跳过项不计为功能通过，不引用其他任务的历史门禁替代本次验收。

| 最终验收项 | 结果 | 证据与范围 |
| --- | --- | --- |
| RoPE slice / IndexCopy 真实 NPU | 76 passed，0 skipped | `logs/rope-slice-tests.log` |
| CPU 定向兼容回归 | 117 passed，4 deselected，0 skipped，73.03 s | `logs/final-cpu-regressions.log`；包含 graph replay 4 项，不重复计数 |
| 同载 Wan VAE 两次真实 NPU 解码 | 输出逐元素相同，fallback=0 | `effects/vae-twice-after.json` |
| 最终 NPU Conv3D、constant lifetime、causal / noncausal PFA、bicubic、RoPE、indexing | 200 passed，1 skipped，0 failed，11.55 s | `logs/final-npu-regressions-rerun.log`；201 节点，包含 RoPE 76 项，不再相加 |
| nearest-exact 独立 NPU | 4 passed，15 deselected，0 skipped，3.56 s | `logs/final-nearest-npu.log`；15 个 CPU case 在此会话未选择 |
| Native CPU `core` + 整个 `tests/core/test_graph_replay.py` 扩展 | exit=0；127 passed，70 skipped，1 xfailed，148.27 s | `logs/final-core-native.log`；该计数包含 graph replay 扩展，不标为纯 core |
| Torch CPU `core` | exit=0；224 passed，1 skipped，149.73 s | `logs/final-core-torch.log`；225 节点，唯一跳过为 accelerator 条件，不计为该功能已验证 |
| Torch 全量 `tests/structure` 会话 | exit=0；1537 passed，4 skipped，136.62 s | `logs/final-structure-rerun.log`；1541 节点，2 个 resolver 名字允许位置的声明跳过、2 个 CUDA 不可用 |
| Native 结构 dtype recorder | 5 passed，0 skipped，0.77 s | `logs/final-structure-native-dtype.log`；补齐 Torch 会话未收集的 5 项静态 recorder 检查，不代表新 NPU 数学执行 |
| 仓库布局、manifest、文档 / 可复用 harness 检查 | 全部通过 | 报告落地后布局 300 个 active Markdown、链接 172、manifest `--check`；源码阶段 37 个 Python 文件的 Python 3.7 AST 与最终 `git diff --check` 通过 |
| 最终 Wan 完整视频执行与效果记录 | 执行完成，Jittor fallback=0；不作质量通过结论 | 同 archive / 噪声，视频数值差异显著，两侧均有严重伪影；数组 / GIF / 对照图已保存 |
| 最终实现与 clean 组件采样来源 | `a1a07b84eef858839c5ea838a4e1388330870edd` | 四组件在 clean 工作树重新采样并对拍；历史优化阶段和实际生成保留其原基线 / dirty patch |

NPU 组唯一跳过项为 `test_acl_indexing.py` 的 rfft：310P 缺少所需 FP32 cube 路径。
跳过不计为功能支持；Conv3D 的上述 10 个明确拒绝梯度 case 也不计为数值训练验收。
Native CPU 扩展门禁的 70 项跳过为 accelerator 61、opt-in 7、insufficient-devices 2，
均不计为已执行支持。唯一 xfailed 是已标注 `expectedFailure` 的
`TestCore.test_swap`（`KI-TEST-001`），不写成普通通过。
注册 manifest 另补登记已有的 `test_aclop::test_dropout_grad`，本次只验证结构契约；
没有在该 NPU 数学测试上给出新验收结果。
nearest-exact 当前验证 size 指定与正整数 scale_factor；非整数 scale_factor 明确拒绝，
可改用 size，不能从 VAE 的整数 2× 上采样外推任意比例支持。

范围限于单张 310P3、batch=1、上述固定 checkpoint / 输入 / 精度及推理阶段；
Wan 原生分辨率更大配置、其他前沿模型、训练、并发服务、CUDA、ROCm、多卡、
数据集级视觉质量和严格显存峰值均未在本次验收，不能据本表外推。
