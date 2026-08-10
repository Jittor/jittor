# Transformers 性能深挖工作记录（2026-07-10 至 2026-07-11）

## 状态

- ✅ 已完成：CLIP/梯度管理、GELU、softmax、LayerNorm、SDPA、Transformer block、
  Tiny Llama/BERT/ViT 和 optimizer step 的 CUDA 分析。
- ✅ 已完成：采用低风险优化并完成 CPU/CUDA、源码 shim/部署 shim 回归。
- 🟡 未完成：当前主机 `cscg104` 无 `npu-smi`/CANN，NPU/910B 未复验。
- 🟡 后续工程：CUDA/NPU fused SDPA training、multi-tensor optimizer、fused LayerNorm、
  cublasLt bias/activation epilogue、真实 CLIP 端到端 profile。

## 环境与口径

- 项目目录：`${JITTOR_LAB_ROOT}/jittor_transformers_perf/`；runtime/cache/log/results 均在项目内。
- Jittor：conda `jt311`；真 PyTorch oracle：`/home/zy/rt_venv` 的
  `torch 2.12.1+cu130`；GPU 为 RTX 4090。
- 基准保留计时区间内所有 slot 的输出/梯度并在末尾同步，排除 Jittor lazy 图只执行
  最后一个输出的假快。训练数据必须检查全部目标梯度 finite 且非零。
- HF 整模两端均强制加载 jt311 的 Transformers `4.56.2`，不直接比较 4.56.2 与 5.12.1。

## ✅ CLIP 与梯度管理

- 旧 device-only 原型对每个参数分别 reduction，再归约标量。512 个梯度张量时比原始
  `.item()` 路径慢 43.5%，根因是数百个小 reduction launch，而不是“device-only”本身。
- 正式实现改为一次 flat reduction + device coefficient。真实发生裁剪时，128/512 张量
  比旧 host-branch 快 22.7%/20.6%；相对 per-gradient 方案快 38.4%/44.7%。
- no-clip 且 512 张量时仍慢 9%-14.5%，因为 device coefficient 为 1 时仍逐张量写回。
  下一步应把 coefficient 融进 optimizer，而不是恢复 D2H 分支。
- GradScaler 从每梯度一次 `.item()` 改为一次 flat finite reduction + 一次 host 决策。
  512 张量、262K/16.8M 元素由约 `12.80/14.16 ms` 降到 `9.09/8.05 ms`；复测有波动，
  仍稳定优于旧路径。
- 真实 optimizer step 显示管理占分段 SGD 的 38.6%-40.6%，占 AdamW 的 22.7%-23.9%；
  但 AdamW 512 张量自身 update 已约 31.5 ms。固定总元素、张量数 128→512 时 SGD/AdamW
  无管理 step 变慢 4.25x/3.87x，P0 应是 multi-tensor/fused optimizer。

## ✅ 算子优化

- GELU exact 的 float32 输入此前被 Python float 常数提升到 float64 divide。改用 dtype
  常数，低精度显式 fp32 compute 后 cast 回输入 dtype；forward 从约 2.69x 差距缩至
  `0.0698/0.0565 ms = 1.24x`。
- softmax 每次动态定义相同 `jt.Function`，有约 20-25 us Python/build 开销。按
  `(length, log)` 缓存 Function class 后，1024 维 forward 为
  `0.0636/0.0546 ms = 1.16x`。
- 大词表原先在 10000→10001 从 1 launch 退化为通用 5-launch 图。新增 register/streaming
  单 kernel forward/backward，覆盖 10001、50257、128256。
- 边界审计后让 10000 使用 500 threads，forward+backward kernel 约快 31%；对
  `length>49152 && ILP==1` 仅将 backward 切到 streaming，50257/65535 分别再快
  3.6%/17.4%，保留 50000/65536 的 ILP8 register 优势。
- LayerNorm 缓存按 dims/eps 生成的 Function class，并用 `rsqrt`；fwd+bwd 从
  `0.3066 ms` 降到 `0.2605 ms`，约快 15%。no-grad CUDA fast path 显式排除 ACL，
  防止 NPU 误选 CUDA `jt.code`。
- 显式 2D cuBLAS matmul 原本没有 `grad()`，可能产生静默零梯度。补齐四种转置组合的
  dA/dB cuBLAS 图；之后 Transformer block 13 组梯度全部 finite/nonzero。

## ✅ SDPA / Flash 与正确性

- fp32 inference 的 Jittor math SDPA `0.2066 ms`，PyTorch default fused `0.0421 ms`，
  约 4.9x；同为 forced math 的 Transformer block forward 差距约 1.5x。训练中 Jittor
  图融合可抵消部分 eager launch，但当前 CUDA 仍没有默认可用的 fused SDPA backward。
- 外部 flash-attn 稳态 adapter 的 `F.sdpa` 约 0.1968 ms，direct BSHD 约 0.1180 ms，
  PyTorch flash 约 0.0273 ms。单独删除 clone 只快 3.6%，lazy packed 场景反而慢 7.1%；
  上游直接产出 BSHD/packed 布局可降低约 47%，因此优化点是 QKV/layout 契约。
- 修复 math SDPA 与 MHA dropout 被忽略的问题。
- 修复 fully-masked query row：bool/additive mask 现在 forward 输出零，随机上游梯度也为零，
  不再返回 V 均值或 NaN。
- 修复 native SDPA bool mask 使用未定义变量，以及 `[B,H,L,S]` 广播 mask 对二维 bias
  做四维 setitem 的崩溃；2D/4D bool/additive CPU/CUDA 均有回归。

## ✅ Block 与 HF 整模

- 合成 Transformer block（B2/S128/H768/12 heads，fp32 TF32-on）：forward Jittor
  `0.7006 ms`，PyTorch default `0.2607 ms`（2.69x），PyTorch forced math `0.4667 ms`
  （1.50x）；fwd+bwd Jittor `1.5985 ms`，PyTorch default `1.8279 ms`（0.87x）。
- TF32-off、同为 math 的 fwd+bwd 为 `1.9077/1.6914 ms = 1.13x`。输出 rel-L2
  `8.50e-6`，13 组梯度最差 rel-L2 `3.66e-4`，全部 finite/nonzero。
- 同版本 Transformers 4.56.2 tiny forward：Llama `1.9222/2.0645 ms = 0.93x`，
  BERT `1.4989/1.0696 ms = 1.40x`，ViT `1.7640/0.9561 ms = 1.85x`。
- Tiny Llama fwd+bwd `3.2109/9.7066 ms = 0.33x`，20 组梯度均 finite/nonzero；
  不含 optimizer/clip/scaler，且整模两端是独立初始化，不能外推为大模型训练快 3 倍或
  作为数值对拍结论。

## ✅ 验证与产物

- 核心组合回归：47 tests 中仅既有 packed-split 测试开关写反；修正测试后该项独立通过。
- `test_torch_compat_norm`：24/24 OK；attention mask targeted：2/2 OK；packed split：1/1 OK。
- 大词表 float32 边界 10000/50257/65535/65536 正反向通过；50257 fp16/bf16 通过
  dtype 对应误差口径。部署到项目内 site-packages 的 `import torch` clip 路径通过。
- 详细报告：`agent/results/transformers/`。
- 复用 skill：`agent/skills/jittor-transformers-perf/SKILL.md`。
