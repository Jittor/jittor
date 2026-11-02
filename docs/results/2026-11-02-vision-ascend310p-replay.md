# 昇腾 310P：CPU 前缀重算与视觉模型设备图复测

- 状态：同提交控制中 Swin 延迟下降 32.64%，但仍慢于原生 Torch；分类 / 深度完整输出与同图效果通过，RT-DETR 原序仍失败
- 复查日期：2026-11-02（America/Los_Angeles）
- 实现提交：`013bc35bdeb3d460c269c85dc9781e841731764e`；42 份最终前向报告均核验该干净 SHA、harness 指纹、未 packing，Jittor fallback 增量为 0
- 同步：起点 `d5bd7f551d1864f58e751e3faf5f9dc6fa4e64e6` 已整合上游 `4543b0f2b1ebda3382c67670dd3f12c105ad3f7e`；本轮 45 秒 fetch 超时，不声称取得更新远端
- 范围：Ascend 310P3 / CANN 9.1.0 的 stock ConvNeXtV2、SwinV2、Depth Anything V2 推理及公共重放；不含整模训练或标注集准确率
- 维护者：运行时、ACL 后端与 Torch 兼容层维护者
- 复查触发：重放 / placement / liveness、CANN / 驱动、dtype、模型快照或 Transformers 变化

## 默认调度下的最终速度

主表使用双方相同的**默认 CPU 调度、4 threads、父进程允许 CPU 0–19、不加 taskset 限制**。每进程 15 次完整预热、15 次同步采样、两个不同输入槽；两轮反转模型 / 后端顺序，报告全部 30 样本的池化中位数。双方使用同一固定权重、FP16、batch=1、未 packing；JIT、加载、预处理及输出后处理不计稳态，每次结果均保留并同步。

| 模型 / 随机输入 | 原生 Torch | Jittor | 完整原顺序输出 |
| --- | ---: | ---: | --- |
| ConvNeXtV2 Tiny / 224²，双方 eager | 20.013794 ms | 22.496115 ms | PASS |
| SwinV2 Tiny / 256²，双方 eager | 11.954828 ms | 22.398027 ms | PASS |
| Depth Anything V2 Small / 518² | eager 44.742496 ms | 显式 SDPA 27.203338 ms | PASS，已测配置快 1.644743× |

Conv 仍约慢 12.4%；Swin 耗时仍约为 Torch 的 1.87 倍。Depth 的优势来自此前已验证的显式 SDPA 配置，本轮重新测量并确认，不称为新代码获得的加速。主表 Depth 比较显式允许 attention 配置差异，其他规格和完整输出仍检查；不以更慢的原生 SDPA 图片耗时构造排名。

Conv 两轮原生 / Jittor 中位数分别为 20.167 / 19.906、22.469 / 22.549 ms；Swin 为 12.161 / 11.565、22.871 / 22.137 ms；Depth 为 44.435 / 44.823、27.216 / 27.167 ms。10/30 的 Swin 33.242 ms 保留为[历史结果](2026-10-30-vision-ascend310p-optimization.md)，本轮代码收益以同提交开关控制归因。其他生成模型的既有范围见[四模型报告](2026-10-29-stock-vision-ascend310p.md)及其引用，本轮未重测。

## 同提交开关控制

在同一干净实现、默认调度和上述计时规则下，只切换公共 ne 方法与 CPU 前缀录图准入，保持 stock 模型、权重、输入和所有输出。历史 ne 从固定 `d5bd7f551` 提取，避免把当前函数误当关闭组；函数 / wrapper SHA、政策及命中数写入 `operator_ab`。共享正确性修复始终保留，因此 `old_all` 是当前源码上的政策关闭组，不是假装重新运行整个历史版本。

| 默认调度的 Swin 政策 | 第 0 轮 | 第 1 轮 | 30 样本池化中位数 |
| --- | ---: | ---: | ---: |
| 历史 ne + 前缀录图关闭 | 33.455691 ms | 33.241483 ms | 33.443680 ms |
| 当前 ne + 前缀录图开启 | 22.791812 ms | 22.114030 ms | 22.529030 ms |

延迟下降 **32.63591%**，两轮方向一致。两个 slot 的完整 logits 在各政策间逐 byte 相同，且对独立原生 NPU 完整对拍通过。控制用 wrapper 的 22.529030 ms 与主表直接生产入口的 22.398027 ms 属于不同协议，不混用分母。政策中的 Linear query 命中为 0；本轮没有保留新增 Linear 分派，也不能把这项控制收益归给它。当前 ne 方法组包含 self-ne 简化、标量 placement 和结果 frontend 修复，不将整个方法组的最终收益全部归到一行代码。

另以四个不同物理 P 核 `0,2,4,6` 固定亲和性测得以下结果，作为独立资源预算。它让原生 Swin 变慢，故不作为唯一性能分母，也不与默认调度的行相除。

| 固定四 P 核，其他协议相同 | 原生 Torch | Jittor |
| --- | ---: | ---: |
| ConvNeXtV2，eager | 21.555060 ms | 22.362533 ms |
| SwinV2，eager | 15.035254 ms | 24.215964 ms |
| Depth，原生 eager / Jittor SDPA | 46.926215 ms | 27.023503 ms |

固定四 P 核的同提交 Swin 控制为：历史 ne / 前缀关闭 33.430506 ms，仅 ne 方法组开启 24.238347 ms，当前生产 23.746110 ms；整体延迟下降 28.96874%，完整输出仍通过。更早未固定亲和性的 23.6285→22.1355 ms 前缀单项实验只保留为阶段证据，不替代最终控制。

## 保留的三组公共层修复

1. **减少比较子图并保留设备语义。** 同一个真实 Tensor 的 operator / method self-ne 复用 canonical `isnan`：NaN 为真，±Inf 与正负零为假，避免减法、绝对值及重复特殊值分类。结果保留输入的 frontend 类型和 placement。新 Python / NumPy 标量在 Tensor 输入的 construction scope 中创建，沿用原 dtype 推断；修复前 stock Swin 的 CPU mask `!= 0` 会在外层 ACL scope 建立 int32 标量再 D2H，修复后比较留在 CPU。已存在 Tensor 操作数不被重新放置，complex、不同 holder 和独立函数入口不进入 self 捷径。
2. **每次重算 CPU 前缀，录制设备后缀。** 仅接纳封闭、显式放在 CPU 的前缀，通过显式 H2D 复制进入设备后缀。CPU 运算、等待和 H2D 每次执行并计入完整前向；不缓存 mask，不按 `_is_constant` 猜测纯度。随机、外部代码和设备读回依赖仍拒绝，普通 executor 保留。Swin 最终报告确认前缀设备图活跃，两个主表进程各有 29 次设备图重放。
3. **保证重放状态与存储有效。** 私有 CPU 输入复制保留 construction placement；kept fused 图的内部条件也标为 kept，避免物理写入将条件保存成旧值。重写 CPU 暂存前等待上次异步消费，前缀重算后再次检查设备地址，迁移则销毁旧图再走普通执行；低层 launch 也检查。CPU 中间存储 / H2D 目标计入预算，原生 pins 参与 `graph_check` 根计数，避免第二个 holder 覆盖原 owner。动态分支、排队、迁移、释放与 graph-check 回归覆盖这些契约。

这些入口由 Jittor 根据公共算子、placement 和重放条件自动选择，无需改 Wan2.2 或视觉模型推理代码；Torch 程序需激活 shim。保持 no_grad、AMP=0、HF32 opt-in，以及原 4 MiB 输入 / 256 MiB 保留预算。Swin 报告的**前缀保留预算上界为 18,394,432 bytes**，包括 CPU 中间存储和 H2D 设备目标，别名可能重复计数；这是准入计数，不是实测 CPU 占用或峰值，另列的 `estimated_retained_bytes` 也不改称实测峰值。不因录图跳过前缀工作。10/30 的 p=2 norm / arg-reduction 精度修复仍是基线，见[对应报告](2026-10-30-vision-ascend310p-optimization.md)。本轮真实硬件为 310P3，不外推其他 SoC、CUDA、ROCm、NPU BF16 或完整训练验收。

## 同一真实图像的效果

双方读取同一不可变猫图输入及 hash-bound 官方预处理元数据，独立 CPU 官方后处理；分类 / 深度完整数组均通过既有 relative-L2≤0.02，不重排或放宽门槛。猫图 SHA256：`f311fbd22e8a0562a237f0985abe185b2dba82435acebd9b452f1dbdd4387dda`。

| 真图像 | 完整输出相对 L2 | 效果 |
| --- | ---: | --- |
| ConvNeXtV2 logits | 0.002758931 | 双方 Egyptian cat，top-5 顺序相同、重合 5/5 |
| SwinV2 logits | 0.003420380 | 双方 Egyptian cat，top-5 顺序相同、重合 5/5 |
| Depth predicted depth `[1,420,518]` | 0.001378379 | max abs 0.0625；恢复原图 432×532 后相对 L2 0.001338019 |

Depth 图片双方均显式 SDPA，保持效果规格一致；此处图片时间不进入主表排名。深度图使用同一 relative-depth 色域，不是米。三张并排图已视觉检查，原图 / 原生 / Jittor 面板、caption 与分类 top-5 均完整无裁切，见未版本化 `final/visual-qa.json`。单图实现一致性不是 ImageNet accuracy、COCO mAP 或带真值的深度质量。

RT-DETR 新诊断仍失败：两个 slot 的原顺序 logits 相对 L2 为 0.054352806 / 0.048059742，boxes 为 0.202647855 / 0.164068814。保留完整数组，不做 ID 对齐或重排，不加入速度排名。

## 门禁与边界

| 验证 | 实际结果 |
| --- | --- |
| CPU core | 327 pass / 45 skip / 1 既有 xfail |
| 最终 Torch 结构会话 | 1646 pass / 4 skip；另 native recorder / package identity 7 pass |
| CPU replay / rematerialization | 49 pass / 44 skip；26 pass / 13 skip |
| Half CPU 单独选择 | 26 pass / 32 deselected |
| 原生 ACL / CPU 混合 | 515 pass / 0 skip，包含 CPU，不能称 515 项全 NPU |
| compat 混合 | 318 pass / 0 skip，包含 ne 的 190 项混合矩阵 |
| 公共 self-ne 探针 | 独立 CPU reference、Jittor CPU / NPU 各 10 字段 exact；原生 NPU 报告 `passed=false`，`--require-exact` 退出 1 |
| 公共 Linear 探针 | Jittor CPU / NPU 与原生 NPU 各 28 case 对独立 CPU 通过；没有新增 Linear route |

探针先同步 lazy 输出再检查驻留，不把未执行的 `location=none` 当作设备失败。原生 NPU 的 Inf 自比较错误是诊断证据，不作 CPU 数学通过；同样，skip 不作硬件验收。

Linear / Addmm 新分派已撤，仅保留可复用探针。CPU scalar-ne 的 raw native 比较和 eq+not 各有 14/192 字段不符 CPU Torch；显式 `result_type` 与 typed scalar 后，CPU及 ambient ACL 的 CPU 计算 210/210 exact，但完整 Swin 两轮 +2.89% / −8.38%、合并约 −1.12%，方向不一致，未采纳。不是 NPU BF16 计算证明。

公开 CPU scalar eq/ne 的五个边界案例、六种拼写仍有真实差异，已记录 [KI-DTYPE-005](../../agent/manuals/known-issues.md#ki-dtype-005-cpu-scalar-comparisons-differ-at-half-and-int64-boundaries)；self-ne 与 placement 修复不解决所有标量签名。既有 KI-BACKEND-025 / 026 / 027 的特殊索引、整数值和 norm 限制仍开启。Swin 还需 CPU FP16 mask 运算、H2D 和设备执行；Conv 卷积数学本轮未改，不能把调度变化归为卷积优化。

## 复现与证据

环境：310P3 44 GB、CANN 9.1.0、driver 26.1.0.b050，独立 Torch 2.12.0+cpu / torch_npu 2.12.0，Transformers 4.56.2、Diffusers 0.35.1。保留后台服务，未宣称物理独占。固定快照为 Conv `f4db009e63145e02b3c075aa64d90ce41bcca4b1`、Swin `40213dad8563a5a916d434a1291443c0fa6358f0`（原样转换的 245 张量 safetensors）、Depth `5426e4f0f36572d16453bbda7a8389317b1bef99`，模型 ID 见[10/30 报告](2026-10-30-vision-ascend310p-optimization.md)与[四模型报告](2026-10-29-stock-vision-ascend310p.md)。

按[性能 skill](../../agent/skills/jittor-transformers-perf/SKILL.md)设置独立解释器、CANN 和外部 lab，双方以标准 `benchmark_generative_vision_ascend.py` 传入同一 `--case`、`--model`、`--source-root`，使用 `--dtype float16 --seed 20261027 --warmups 15 --samples 15 --slots 2`。主表保持默认 affinity，线程数 4；Conv / Swin 双方 `--attention eager`，Depth 为原生 eager / Jittor sdpa，比较时仅后者加 `--allow-attention-difference`。比较器保留 `--max-relative-l2 0.02`。第二轮反转执行顺序并保存不同输出；真图传相同 `--input-npz`，Depth 图片双方 sdpa。

未版本化证据根 `$JITTOR_LAB_ROOT/_state/generative-vision-310p/iteration-2026-11-02/`：`final/summary.json` 汇总 42 份前向、默认 / 固定核协议、全部数组与政策；`final/default-{baseline,commands}.json`、`final/{baseline,commands}.json` 记录两个资源预算；`final/effects-*` 与 `visual-qa.json` 保存三图效果 / QA；`final-gates/`、`final-probes/` 与 `final-source-freeze.json` 保存门禁、探针、命令和源码指纹。`final-probes/verification.json` 显式索引最终 28 case 对拍、首次 lazy 驻留 guard 失败、随后成功重跑及预期负例，原始 `commands.json` 的首次 exit 1 保留，不宣称所有尝试均成功。`cpu-ne-scalar/` 保留拒绝候选，早期诊断不替代最终生产表。原始模型、缓存、图片和日志不进入仓库。
