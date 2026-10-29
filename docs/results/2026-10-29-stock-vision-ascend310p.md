# 昇腾 310P：四个完整视觉模型的兼容与性能

- 状态：三模型完整输出与真图像对拍通过；RT-DETRv2 可执行但 ordered 输出未验收，排除合格排名
- 复查日期：2026-10-29（America/Los_Angeles）
- 实现提交：`56ee4e59e3ed634af27fe328c9a19194b802847e`；最终 28 份前向报告均从 clean 工作树测量
- 同步范围：已整合上游 `4543b0f2b1ebda3382c67670dd3f12c105ad3f7e`；恢复后的三次 fetch 因网络失败，未声称取得更新远端
- 范围：单张 Ascend 310P3，完整预训练分类、深度与检测前向；不含训练或标注集准确率
- 维护者：ACL 后端与 Torch 兼容层维护者
- 复查触发：CANN / 驱动、算子分派、dtype 规则、模型快照、Transformers 或 replay 策略变化

## 最终性能

Depth Anything V2 显式选择 SDPA 后，比本轮已测较快的原生 eager **快 1.65×**；
ConvNeXtV2、SwinV2 仍慢于原生 Torch。四者均保留 stock forward 和任务 head。

| 模型 / 输入（batch=1） | Torch eager | Jittor eager（工具默认） | Jittor SDPA（显式选择） | 全数组对拍 |
| --- | --- | --- | --- | --- |
| ConvNeXtV2 Tiny / 224² | 20.050 ms | 22.391 ms | 无对应分支 | PASS |
| SwinV2 Tiny / 256² | 11.712 ms | 33.333 ms | 无对应分支 | PASS |
| Depth Anything V2 Small / 518² | 44.671 ms | 67.931 ms | **27.050 ms** | eager / SDPA 均 PASS |
| RT-DETRv2 R18vd / 640² | 排名排除 | 排名排除 | 无对应分支 | 两轮 ordered FAIL |

时间为两轮各 7 样本、共 **14 样本池化中位数**，不是两个中位数的平均；FP16、未 packing。
原生 Depth SDPA 为 77.273 ms；同 attention 对拍也通过，但不使用该较慢分母宣称 2.86×。
RT-DETR 诊断中位数为原生 41.553 / Jittor 75.751 ms，输出未通过，`qualified_speedup=null`。
这里只比较本轮测过的 eager / SDPA 与独立 packing 阶段，未穷举 Torch compile 或自定义 kernel。

## 自动优化与模型代码边界

无需修改所覆盖的 Wan2.2 或新视觉模型 forward，也未新增模型 adapter。
Torch 生态程序仍须安装并激活 `jittor.compat.torch` shim，选择 NPU、支持的 dtype 与 no-grad 推理。
原生 Jittor 与 shim 共享适用的后端路径；任意表达式并不保证被识别成融合算子。

| 机制 | 自动或显式 |
| --- | --- |
| GELU / RMSNorm / attention / unit division、公共语义修复和本轮新算子 | 满足 op、shape、dtype、grad、SoC 条件时自动分派 |
| executor / device graph replay | 默认自动尝试；预算或资源可拒绝设备录制，保留适用执行路径 |
| `freeze_linear_layers_` / `--pack` | 显式冻结、预打包 FP16 Linear；不是默认透明优化，不适用于可训练权重 |
| 前轮 Wan 8 MiB 输入门槛、SDXL 1 MiB recording 预算 | 显式运行参数，未改生产默认值或模型 forward；本轮四模型不使用它们 |
| Depth `--attention sdpa` | 显式公共模型配置；工具默认强制 eager，不代表 HF 自动选择的默认后端 |

冲突整合保留上游 910B 实现。310P raw PFA、小宽度 RMSNorm 拒绝及本轮
ConvTranspose2d、bilinear resize、grid sampling、GatherV2、TopK 新入口有 SoC / 推理条件保护。
frontend 参数 / dtype、normalize、argsort 构造和 canonical module 分派属于共享修改。
ConvTranspose2d module 新增分派查询也可能触达已有 CUDA handler：CPU 契约已测，CUDA 实际执行未验；
910B、ROCm 实机未验；新 SDK 推理路径不准入梯度，不以 310P 结果代替其他硬件验收，范围见[待硬件验证](../../agent/manuals/deferred-hardware.md)。

## 已验证的修复与瓶颈

- 修正 all / any 的 dim / keepdim；exp / pow 保持 frontend 低精度输出 dtype；normalize 用 scalar clamp
  避免 Python epsilon 将 FP16 norm 提升为 FP32。native exp / pow 白名单精度策略不变，含梯度 / active-epsilon 对拍。
- 接入 310P 无梯度 ConvTranspose2d 与 bilinear resize，使完整 Depth head 可执行。
  `interpolate(antialias=True)` 从被忽略改为明确拒绝；未实现 antialias。
- argsort 构造补传 stable 参数；ACL `stable=True` 仍不支持。grid sampling 接入
  FP16 / FP32、4D、bilinear / zeros / align_corners=False 子集；不声称 raw SDK zero-copy。
- Swin 单一非空一维整数索引、其他维完整 slice 时，用 GatherV2 替代全坐标 advanced Index。
  无输入值或模型表缓存；梯度、910B、bool / 多索引 / partial slice 等保留原路径。
- 通用 TopK 在 310P 无梯度、连续二维 FP16 / FP32、最后轴、正 k、largest / sorted=True 时
  用部分选择替代完整 sort+slice。22 同输入组合 values / indices 与原生 NPU 逐字节一致；旧 compat 索引本来也一致，不能称 tie 修复。

Swin dirty 阶段受控整模 A/B 为 **46.033 → 33.564 ms（1.37×）**，30 样本池化、两输入输出 byte-exact。
最终 clean 为 33.333 ms，仍未超过 Torch。stock 模型显式构造 CPU attention mask，
whole device graph 因 host ternary 拒绝录制，executor replay 保留；fallback=0 不代表所有操作都在 NPU。
ConvNeXt profile 的主要成本为卷积；profile 逐算子同步的总墙钟不作为模型速度。
TopK 公共路径微基准比旧路径快 1.36–1.70×，22 组合倍率中位数 1.58×，不外推整模收益。
独立 packing 阶段无稳定收益；原生 ConvNeXt packed 因 SDK 布局不支持失败，未混入主表。

RT-DETR 中间值观察到大量并列分数、候选集合与顺序敏感，完整差异根因尚未完全隔离。
随机 logits 相对 L2 0.048–0.054、boxes 0.164–0.203；TopK 接入后猫图 FP32 仍失败（0.050188 / 0.12894）。
候选 ID 对齐只作诊断；未隔离的原生中间值差异不能据以断言厂商 bug。
非法索引见 [KI-BACKEND-024](../../agent/manuals/known-issues.md#ki-backend-024-acl-advanced-indexing-does-not-reject-out-of-bounds-integer-indices)：
旧 Index 与 GatherV2 同样错误返回，CPU 应抛错；非法用例不计支持。

## 固定权重与环境

| 官方模型 ID | 固定 revision |
| --- | --- |
| `facebook/convnextv2-tiny-1k-224` | `f4db009e63145e02b3c075aa64d90ce41bcca4b1` |
| `microsoft/swinv2-tiny-patch4-window8-256` | `40213dad8563a5a916d434a1291443c0fa6358f0` |
| `depth-anything/Depth-Anything-V2-Small-hf` | `5426e4f0f36572d16453bbda7a8389317b1bef99` |
| `PekingU/rtdetr_v2_r18vd` | `5650961749fa93567c0d46fc7f43ea4f9e914107` |

三份 safetensors 与 Swin 官方 bin 的完整 hash 见外部 `model_paths.json`。
Swin 经独立 Torch `weights_only=True` 读取、原样转存 safetensors，245 张量不变，双方使用同一份转换权重。
Ascend 310P3 44 GB、CANN 9.1.0、驱动 26.1.0.b050；原生 PyTorch 2.12.0+cpu / torch_npu 2.12.0、
Transformers 4.56.2、Diffusers 0.35.1。后台服务保留约 2.7 GB，未取得物理独占。
原生 jit_compile=False、matmul / conv HF32=True；Jittor acl_allow_hf32=1、AMP 关闭、自动 replay 输入 4 MiB / retention 256 MiB。

独立解释器与缓存，单卡顺序运行；首次 JIT、profile、oracle 与计时分离，接受样本时无本任务并行编译或 NPU 作业。
seed=20261027、两个随机输入 slot、每轮 cold 单记、5 warmups / 7 逐次同步 samples。
首轮 ConvNeXt → Swin → Depth → RT-DETR、原生 → Jittor，次轮模型 / 后端均反转；Depth 附加配置也反转。
load、packing、cold、预处理、输出 CPU 后处理不计热 forward。ConvNeXt / Depth 与 RT-DETR 两轮设备图启用，Swin 拒绝原因如上。
全部 28 份报告的源码 SHA / 空 diff 的 SHA256 已核验；Jittor 在禁止 fallback 作用域执行并核验设备驻留、计数增量=0。
原生 counter 不适用（null），不能说原生也通过 Jittor fallback counter。
随机完整输出阈值相对 L2≤0.02，并检查 shape / finite / 全字段；tiny CPU 为≤1e-4，不能替代完整模型验收。

## 真图像效果

同一张 432×532 猫图 SHA256 为 `f311fbd22e8a0562a237f0985abe185b2dba82435acebd9b452f1dbdd4387dda`。
独立 CPU 生成共享 NPZ，双方直接读相同 archive；四份主测 archive 未变。sidecar / archive / spec / report
绑定图片与输入 hash、shape、checkpoint；renderer 对错误 archive / checkpoint / shape 的三个负控均拒绝。
两侧同 attention（Depth 为 SDPA），独立原生 CPU FP32 官方后处理；四图视觉检查无缺失或裁切。

| 真图像结果 | 数值与可视化 |
| --- | --- |
| ConvNeXtV2 | logits 相对 L2 0.00275893；两侧 top-1 Egyptian cat，top-5 顺序相同、重合 5/5 |
| SwinV2 | logits 相对 L2 0.00377682；两侧 top-1 Egyptian cat，top-5 顺序相同、重合 5/5 |
| Depth | 保持长宽比 420×518，raw 相对 L2 0.00137838；官方 resize 到 432×532 后 0.00133802，共用色标视觉接近 |
| RT-DETRv2 | ordered logits / boxes 相对 L2 **0.07662588 / 0.21533035，FAIL**；阈值 0.3 两侧各 1 猫框，无未匹配，IoU 0.99899117，score 差 0.00102407 |

检测后处理接近不能替代完整有序数组验收。单图实现一致性不是 ImageNet accuracy、COCO mAP、
深度真值精度或米制距离；Depth 非方形效果耗时不与 518² 随机主表混用。

## 验证与失败范围

| 会话 / 证据 | 实际结果 |
| --- | --- |
| 四模型 tiny CPU | 四项通过；RT-DETR 原 fixture 84 分数全并列，改同一公开 config 的 initializer_range 0.01→0.1 生成非退化 fixture；不改预训练权重或输出排序 |
| all / any | 独立 CPU 177 契约：115 不同→0；定向 CPU 41 / NPU 41 passed |
| dtype / normalize | 纯 CPU 42 passed；混合 71=42 CPU+29 真设备；八个独立 Torch / JT×CPU / NPU×FP16 / FP32 进程完成 |
| ConvTranspose2d / bilinear | SDK 14 / 32 组合；公共会话 84=81 ACL suite+3 CPU；antialias 4=2 CPU+2 ACL；拒绝边界不计数学支持 |
| Grid sampling / TopK | Grid 16 SDK、26 NPU / 7 CPU / 39 host；TopK SDK / 公共各22同输入，回归 native NPU50 / CPU17、compat CPU3 / NPU会话5、host39，均0skip，不加总重叠 |
| CPU core | Native 98 passed / 44 skipped / 1 xfailed；Torch 229 passed / 1 skipped；两侧 exit=0 |
| 最终结构及附加 CPU | Torch 全结构+7 compat CPU：1586 passed / 4 skipped；native-only recorder / graph / capture / dispatch：80 passed / 66 GPU能力 skipped，不称 CUDA 验证 |
| 最终 NPU / compat | Native 真 NPU 411 passed / 0 skipped；compat 混合125 passed / 0 skipped，包含 CPU，不称125项NPU |
| 工具边界 / 静态 | strict FP32 guard 4 CPU、CLI 4 边界、renderer 3 负控通过，producer 数组未变；源码布局495 active Markdown / diff通过 |

core xfail 为已有 KI-TEST-001 swap；core skip 为能力 / opt-in / 设备数等既有条件，结构4项为既有声明条件。
首轮结构 5 failed / 1574 passed / 4 skipped 已保留：修导入循环、source-owner 清单和重复测试名，未放宽 ratchet。
首轮 NPU 409 passed / 1 failed 源于旧测试误认 numpy readback 会迁移源；修后验证驻留 / 指针保持及显式 CPU 迁移恢复，未改生产行为迎合测试。
修前 Swin dtype、Depth 转置卷积 / resize、RT-DETR argsort / grid 的失败均归档；旧 dirty 时间不改记 clean。
[KI-BACKEND-019](../../agent/manuals/known-issues.md#ki-backend-019-acl-argsort-constructor-repair-still-needs-the-original-910b-and-dtype-acceptance) 的原910B / float64 门禁仍未复验。

## 可复现入口与外部证据

按[性能 skill](../../agent/skills/jittor-transformers-perf/SKILL.md)及[310P 指南](../guides/ascend-310p.md)
安装独立核心 / compat，加载 CANN、外部选择同一设备；`TORCH_NPU_PYTHON` 为独立原生解释器，
`TORCH_CPU_PYTHON` 为不导入 torch_npu 的 CPU oracle，`JITTOR_PYTHON` 已安装 compat；仅 PYTHONPATH 不会安装独立包。
以下变量均指本地 checkout / 外部 lab / 固定快照；同一矩阵所有进程串行、测试缓存另设。

```bash
perf_scripts="$JITTOR_SOURCE_ROOT/agent/skills/jittor-transformers-perf/scripts"
vision_run="$JITTOR_LAB_ROOT/_state/vision-reproduction"
mkdir -p "$vision_run"
# case_name: convnextv2 / swinv2 / depth_anything_v2 / rtdetr_v2
"$TORCH_NPU_PYTHON" "$perf_scripts/benchmark_generative_vision_ascend.py" \
  --backend torch --device npu --case "$case_name" --model "$checkpoint" \
  --dtype float16 --attention eager --seed 20261027 --warmups 5 --samples 7 --slots 2 \
  --source-root "$JITTOR_SOURCE_ROOT" --output "$vision_run/torch.json"
JITTOR_TORCH_SHIM=1 JITTOR_HOME="$vision_run/benchmark-cache" backend_fallback=error \
  "$JITTOR_PYTHON" "$perf_scripts/benchmark_generative_vision_ascend.py" \
  --backend jittor --device npu --case "$case_name" --model "$checkpoint" \
  --dtype float16 --attention eager --seed 20261027 --warmups 5 --samples 7 --slots 2 \
  --source-root "$JITTOR_SOURCE_ROOT" --output "$vision_run/jittor.json"
"$TORCH_CPU_PYTHON" "$perf_scripts/compare_generative_vision_ascend.py" \
  "$vision_run/torch.json" "$vision_run/jittor.json" --max-relative-l2 0.02 \
  --output "$vision_run/parity.json"
```

交换后端 / 模型顺序再测一轮，保留各轮不同文件名。Depth 两侧各补测 `--attention sdpa`；
原生 eager 对 Jittor SDPA 的比较额外加 `--allow-attention-difference`，仅豁免 attention 元数据差异。
图片先用以下官方预处理入口生成 archive / sidecar，再在**双方相同 attention** 的前向命令中加
`--input-npz "$shared_input"`，保存为 `torch-image.json` / `jittor-image.json`，其余计时选项相同。

```bash
"$TORCH_CPU_PYTHON" - "$perf_scripts" "$case_name" "$checkpoint" "$source_image" "$shared_input" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from _generative_vision_cases import preprocess_image
metadata = preprocess_image(*sys.argv[2:])
Path(sys.argv[5]).with_suffix('.json').write_text(json.dumps(metadata, indent=2))
PY
"$TORCH_CPU_PYTHON" "$perf_scripts/compare_stock_vision_effects.py" \
  --left "$vision_run/torch-image.json" --right "$vision_run/jittor-image.json" \
  --image "$source_image" --input-metadata "${shared_input%.npz}.json" --threshold 0.3 \
  --source-root "$JITTOR_SOURCE_ROOT" --output "$vision_run/effects.json"
```

未版本化证据根为 `$JITTOR_LAB_ROOT/_state/generative-vision-310p/iteration-2026-10-27/`：
`model_paths.json`、`inputs/`、`final-oct29/{baseline,commands,summary,visual-qa}.json`，
`r{0,1}-*.json` / `.json.npz`、`image-*.json` / `.json.npz`、四份 `effects-<case>.json.png` 及对应日志。
SDK 补充证据在相邻 `iteration-2026-10-29/`。原始数据、图片、模型和缓存不进入仓库；缺失时用上述入口重建。
