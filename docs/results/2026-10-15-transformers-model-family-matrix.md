# Transformers 模型族 CPU/CUDA 验证矩阵

- 状态：已测量；九个 tiny 场景达到 L3，完整 Transformers 支持仍未完成。
- 日期：2026-10-15，Asia/Shanghai。
- 测量基线：`8e388d108eb83c1c438936709f29855f47ba2a69`，运行前工作树干净。
- 维护者：Jittor Torch compatibility maintainers。
- 复查条件：模型 builder、兼容层、训练 probe、Transformers、Torch、驱动或精度设置变化。

## 场景与分母

两侧均使用 Transformers 4.56.2，干净上游源码提交
`cd74917ffc3e8f84e4a886052c5ab32b7ac623cc`。从该版本注册表重新提取的分母为
374 个模型族、46 类任务。CPU 和 CUDA 各验证 **8/374 个模型族、2/46 类任务**；
分别仍有 366 个模型族和 44 类任务未经本轮验证。NPU、ROCm、Corex 本轮为零验证。
两个 Llama 场景只贡献一个模型族。按场景抽样：原有六项 core、新增三项 tail，
每项各跑 CPU/CUDA，因此是 12 个 core 格子、6 个 tail 格子，broad 为零。

所有场景均为固定随机权重的小配置，使用真实上游模型执行前向与反向；随后固定原始
权重，分别以 seed 7、19 改变输入，确认两侧输出确实改变、请求梯度完整且数值有限。
另执行学习率 0.001 的三步 SGD，对拍每步 loss、参数梯度、浮点输入梯度和参数轨迹。
训练输入 seed 为 1/2/3；训练保留原模型的 frozen 划分。下表张量数为每侧归档数组数，
训练误差为训练比较器的最大 field-scaled 相对误差。

| 场景 | 训练参数 / 冻结参数 | 基础数组 | 三步数组 | 训练误差 CPU / CUDA |
| --- | --- | --- | --- | --- |
| bert | 39 / 0 | 40 | 240 | 2.18e-07 / 8.92e-07 |
| clip_multimodal | 78 / 0 | 80 | 477 | 6.62e-07 / 0.00114 |
| gpt2 | 28 / 0 | 29 | 174 | 2.38e-07 / 0.000168 |
| llama | 21 / 0 | 22 | 132 | 2.09e-07 / 0.000161 |
| llama_rope_2048 | 12 / 0 | 13 | 78 | 1.79e-07 / 0.000115 |
| mixtral_moe | 41 / 0 | 42 | 252 | 1.19e-07 / 8.45e-05 |
| t5 | 47 / 0 | 48 | 288 | 3.36e-07 / 0.000355 |
| vit | 40 / 0 | 42 | 249 | 5.17e-07 / 6.12e-05 |
| whisper | 88 / 1 | 91 | 540 | 3.83e-07 / 0.000336 |

- GPT-2、Llama、Mixtral 和 RoPE Llama 对应 `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES`。
  BERT、ViT、T5、Whisper、CLIP 对应 `MODEL_MAPPING_NAMES`。
- BERT 与 ViT 同时检查 hidden states 和 pooler 输出，因此 pooler 参数参与反向。
- CLIP 使用真实文本与图像双塔，文本形状 `(2,8)`、图像 `(2,3,16,16)`，检查图文 logits、
  全部参数和 pixel 梯度。hidden 32、两层、四头、projection 16、patch 8。
- Mixtral 使用两层、hidden 32、intermediate 64、四专家 top-2，输入 `(2,32)`；
  每层每个专家的 w1/w2/w3 和路由器梯度都被要求存在。本轮不认证 router auxiliary loss。
- RoPE Llama 实际输入长度为 2048；原 max position 512、linear factor 4，单层 hidden 32。
  这不证明 32K 上下文、generation 或完整尺寸模型的可用性。
- T5/Whisper 是基础 encoder-decoder 模型，无 LM head/generation 验证。
  Whisper 基础对拍重新开启全部参数梯度；训练则保留一个冻结的位置 embedding，
  所以基础与训练的梯度参数分母不同。浮点输入梯度包括 Whisper features、ViT/CLIP pixels。

## 平台和验收

Python 3.11.16、NumPy 1.26.4、Jittor 2.0.0；独立二进制 oracle 为
PyTorch 2.11.0+cu128，已检查原生 `_C` 扩展、shim 标记为 false。
Linux 6.1.188/glibc 2.34，AMD EPYC 9554，CUDA 使用单张 RTX 4090，驱动 580.178.04。
Candidate 为 CUDA toolkit 12.2.140、GCC 11.5；oracle 为 CUDA 12.8、cuDNN 91900。
Candidate 的实际 cuDNN API/header 为 92400，安装元数据却声明 9.26.0.51；
报告保留这个差异，外部产物绑定了实际库的哈希，并未把元数据当成加载库版本。

CPU 禁用 CUDA/ACL，TF32 关闭、matmul precision highest；CUDA 开启 TF32 matmul/cuDNN、
precision high，cuDNN benchmark 关闭。实际 Jittor auto graph replay 为 1，retain budget
268435456 bytes，CUDA kernel math 为 strict。两侧线程数为 2，运行条件逐阶段一致。
设备映射、完整 wheel 来源、二进制路径、模型完整配置与全部参数名保存在外部平台产物中。

基础 `verify_repo.py` 使用既有默认 atol 0.002 / rtol 0.01；训练及扰动比较沿用
CPU 0.002/0.01、CUDA 0.005/0.02。各阶段逐数组先按自身阈值验收，scorecard 摘要的上界
取各阶段阈值最大值，并保存 `stage_tolerances`；不能拿摘要上界解释更严格基础阶段。
CUDA T5 的 seed 19 扰动最大 scaled relative 为 0.011524663，符合预设 0.02，
不符合基础阶段的 0.01，故明确保留两种口径。没有为本轮结果放宽阈值。
这些 field-scaled 指标与共享 pytest 的逐张量 divergence 指标不同。

实际 pytest 生态门禁另行跑过 CPU 9 passed / CUDA 9 passed，均零 skip，
CUDA 最少实际执行数要求为 9。独立训练和扰动进程均启用 fallback=error，实际计数为零。
这证明所测场景零 fallback；由于未做 L4 真实尺寸性能验收，本轮最高级别仍为 L3。
共享机器、单次诊断计时和缺少完整进程峰值采样均不满足速度结论的条件。

## 复现与证据

使用 [统一 runbook](../../agent/skills/torch-compat-repo-runbook/SKILL.md) 的独立解释器、
同权重流程与 `training_probe.py` 三步训练命令。先串行预热，单测与 benchmark 使用
独立缓存，CPU/CUDA 和并行任务也使用独立缓存。基础命令为：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
JITTOR_REQUIRE_REAL_TORCH=1 JITTOR_REQUIRE_TRANSFORMERS=1 \
JITTOR_ECOSYSTEM_CACHE_ROOT="$RUN_ROOT/benchmark-cache" \
<jittor-python> agent/skills/torch-compat-repo-runbook/scripts/verify_repo.py \
  --repo transformers --cases <case> --device <cpu-or-cuda> \
  --oracle-python <real-torch-python> --no-memory --repeats 1 --out "$RUN_ROOT/results"
```

CPU 显式设置 `CUDA_VISIBLE_DEVICES='' JT_USE_CUDA=0 use_cuda=0 nvcc_path=''`；CUDA 使用
已分配设备映射并显式设置编译工具链，不能只用默认设备推断。九项 case 全名均为
`transformers_` 加表中场景。扰动复用原始 `.ref.weights.npz`，不重新生成权重。
每次训练和扰动的完整双解释器命令已写入相应 comparison JSON。

未版本化原始证据位于 `$JITTOR_LAB_ROOT/_state/transformers-matrix/final-8e388d108/`。
`source-manifest.json` 固定六份执行源码 SHA256；`independent-artifact-audit.json`
逐项核对 18 个场景、468 个产物绑定、完整参数梯度集合、输入扰动非空操作、训练参数分区、
实际 device/precision、依赖版本及零 fallback。`scorecard-evidence-v2.json` 绑定原始测量，
审计器接受 18 行、拒绝零行。负向检查同步改写 CUDA 平台 JSON、标签及其 SHA，但保留实际
测量，九个 CUDA 行全部因 precision 不符被拒；九个 CPU 行继续通过。

关键产物 SHA256：

- `independent-artifact-audit.json`：`bfd36e44cc7a34fddb2d07eebb73a9f7faf068c01490ea27acbdf77936e21304`。
- `scorecard-evidence-v2.json`：`3827caa84e8b37e75ca2226482748a5be5a3ab47ac5fc07bbe355b039024f13c`。
- `scorecard-v2.json`：`5239ab929958cfff3d22079295ce39b1409b2845334d7263c817796e2b45b87b`。
- `negative-precision-scorecard.json`：`5638335206d146066659389339cbe6a45e47f1bc860081aa9a527ad6f5a33003`。

## 结论边界

L3 仅指表中固定配置、三步 SGD 数值轨迹；不能推广为 Trainer、AdamW、任务收敛、真实
checkpoint、bf16、多卡训练或整族支持。32K Longformer 的验证是独立后续任务，不计入本表。
历史 [真实尺寸模型报告](2026-09-24-torch-compat-real-models.md) 和
[H3 报告](2026-09-12-minimax-h3-torch-compat.md) 保留各自版本、基线和已知问题，
不能合并成本提交的新测量或当前家族覆盖分子。
