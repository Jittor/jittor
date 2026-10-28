# Qwen3-0.6B：310P3 FP16 推理性能

- Status: 单卡、batch=1、动态 KV cache 的设备性能与对拍通过，范围见下
- Last reviewed: 2026-10-12
- Source baseline: `aad488adca252aeec92edd2edf21548cf3c321f9`；基于 `3cfaa92e9`（含远端 `7a18abf29`）的优化实现
- Owner: ACL backend and Torch compatibility maintainers
- Review when: CANN、驱动、Transformers、Torch、权重布局、SDPA 或图重放机制改变

## 结果与口径

在相同 checkpoint、精度和设备上，Jittor 的 502-token decode 比原生
torch_npu 快 **6.4%**，1998-token decode 快 **25.6%**。生成 32 token
的完整耗时分别减少 **5.6%**、**32.0%**。两侧均显式预转换 Linear 权重，
每轮生成前后同步，所有预热、计时和验证输出都参与 token 一致性检查。

| 输入 token | 阶段 | Jittor | 原生 torch_npu |
| --- | --- | --- | --- |
| 502 | prefill：generate(1) | 0.07014 s | 0.07051 s |
| 502 | decode | **42.70 token/s** | 40.13 token/s |
| 502 | generate(32) 全程 | **0.79607 s** | 0.84300 s |
| 1998 | prefill：generate(1) | **0.31810 s** | 0.64120 s |
| 1998 | decode | **38.45 token/s** | 30.61 token/s |
| 1998 | generate(32) 全程 | **1.12434 s** | 1.65381 s |

主表为完整生成预热 5 次、计时 7 次的中位数。decode 定义为
`31 / (median(generate(32)) - median(generate(1)))`，不能与包含 prefill
的 `32 / total_time` 混用。工具另存的直接 `model(...)` prefill 包含全部
位置 logits，不用于上表。首次 JIT、模型加载及权重转换不计入稳态。

两侧顺序运行，主表采用 RMSNorm 辅助统计修复后的正式复测。
短输入 prefill 基本持平，不把不足 1% 的差异视为稳定收益。

原生侧选择二进制算子模式 `jit_compile=False`、eager attention 和 NZ 权重。
另测同样预转换权重的原生 SDPA：502/1998-token decode 分别为
39.48/16.17 token/s，完整生成 0.902/3.060 s，因此使用更快的 eager 作为基线。
Jittor 使用 SDPA、显式 `freeze_linear_layers_`；两侧 cache 都为 dynamic，
显式 `compile=none`。Jittor 保留默认 `auto_graph_replay=1`，不以这个开关
或 `compile` 参数替代真实重放统计。

## 环境与正确性

| 项目 | 配置 |
| --- | --- |
| NPU | Ascend 310P3，44 GB，单卡 |
| 驱动 / CANN | 26.1.0.b050 / 9.1.0 |
| Python / NumPy | 3.12.3 / 1.26.4 |
| Jittor | 本树 2.0.0，独立 Torch frontend |
| 原生参考 | PyTorch 2.12.0+cpu + torch_npu 2.12.0，独立解释器 |
| Transformers | 两侧均 4.56.2 |
| 权重 | Qwen/Qwen3-0.6B，快照 `c1899de289a04d12100db370d81485cdf75e47ca` |
| 推理 | FP16、batch=1、greedy、强制 32 个新 token |
| RoPE FP32 矩阵乘 | 两侧显式允许 HF32，Jittor 为 `acl_allow_hf32=1` |

每个长 prompt 加上算术、法国首都两个短 prompt，生成结果与独立原生参考
逐 token 一致；重复验证也一致。末位置 logits 的相对 L2 误差在 502/1998
token 时分别为 **0.003048 / 0.003917**，最大绝对误差为
**0.05859 / 0.07422**，argmax 相同，全部 finite。

Jittor 全部计算处于 `backend_fallback=error` 和回退计数守卫内，验证和
profiling 完成后计数仍为 **0**。原生 Torch 没有对应计数器，JSON 明确记录
`fallback_count_available=false`。

## 改动为何有效

未打包推理的 CANN trace 中，TransData 分别占 Jittor、Torch 设备算子累计
时间的约 44%、51%，大词表输出权重也在每个 token 重复转换。显式快照在设备上
一次性把 197 个 Linear 的 FP16 权重整理为 NZ 布局，直接作为矩阵乘输入。
Jittor 转换约 0.38–0.39 s，原生参考约 0.63 s，均单独计时。

源模型有 596,049,920 个 Parameter 元素。Jittor 转换后将线性权重注册为
buffer，Parameter 统计变为 155,648,000；数值与模型的数学权重保持一致。
两侧都为原本与 Embedding 共享的 lm_head 创建独立快照，因此实际存储会增加。
使用约束与 checkpoint 格式见 [310P 指南](../guides/ascend-310p.md)。

310P 的 prefill 接入 PromptFlashAttentionV3，设备侧完成 padding、KV 扩头
和输出裁剪；decode 使用 IncreFlashAttentionV4。前者与该设备缺失的
FlashAttentionScore 是不同接口，支持条件按
[CANN PFA V3 文档](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-transformer/aclnnPromptFlashAttentionV3.md)
及真实设备探针对拍限定。

Packed Linear 的 ND 形状处理、无梯度 Concat 与 FP16 RMSNorm 改为原生节点；基础切片、
cat 和 RMSNorm 的前端派发减少重复 Python 验证，同时保留 registry 覆盖、
AMP、视图写回及训练分流。cProfile 用于定位这些主机派发开销，profile 耗时
不用于性能表。

RMSNorm 原生入口仅接受 FP16 输入和 FP16 权重，返回归一化结果；CANN 要求的
辅助输出保存在私有缓冲区。FP32、BF16 和训练使用 Function 路径，不把本次
FP16 推理加速外推为这些路径的优化。

以下正确性修复也属于验收条件：

- 大图触发内存池 trim 时，ACL 原始释放接口会立即释放仍被排队任务使用的
  内存，1998-token prefill 曾复现非法地址。实际归还 SDK allocation 前现在
  等待设备；正常池内复用不增加同步。Pinned allocation 等待已使用它的设备。
  [CANN aclrtFree 文档](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_0100.html)
  明确说明该 API 不隐式同步。
- 图重放增加 registered buffer 的 Var 变更检查。加载新的 packed buffer 后
  会重新捕获，避免继续输出旧权重结果；共享 holder 去重，更新型状态仍可重放。
- 310P 上重复调用 CANN RMSNorm 时，小张量的辅助 `rstd` 曾出现部分零值，
  直接读取设备内存确认主输出正确、辅助值错误。310P 训练在设备上独立计算
  FP32 `rstd` 供反向使用；推理不向调用者暴露 CANN 的辅助结果。

## 复现

先按 [310P 环境指南](../guides/ascend-310p.md) 安装两个独立解释器、相同版本的
Transformers，并准备同一份本地 checkpoint。为测试、benchmark 和不同后端
分配独立缓存，单卡任务顺序执行。原始数据放在 lab，以下变量由运行环境提供：

```bash
perf_script=agent/skills/jittor-transformers-perf/scripts/benchmark_qwen3_ascend.py
JITTOR_TORCH_SHIM=1 PYTHONPATH=python backend_fallback=error \
  "$JITTOR_PYTHON" "$perf_script" --backend jittor --model "$QWEN_MODEL" \
  --dtype float16 --allow-hf32 --context-tokens 512 --new-tokens 32 \
  --fixed-length --warmups 5 --samples 7 --attn-implementation sdpa \
  --freeze-linear --validate-prompts \
  --logits-output "$RUN/jittor-512-logits.npy" --output "$RUN/jittor-512.json"

"$TORCH_PYTHON" "$perf_script" --backend torch --model "$QWEN_MODEL" \
  --dtype float16 --allow-hf32 --context-tokens 512 --new-tokens 32 \
  --fixed-length --warmups 5 --samples 7 --attn-implementation eager \
  --torch-jit-compile off --torch-pack-linear --validate-prompts \
  --logits-output "$RUN/torch-512-logits.npy" --output "$RUN/torch-512.json"

"$NUMPY_PYTHON" agent/skills/jittor-transformers-perf/scripts/compare_qwen3_ascend.py \
  "$RUN/jittor-512.json" "$RUN/torch-512.json" --output "$RUN/parity-512.json"
```

把 `512` 改为 `2048` 并使用不同输出名复现长输入。实际 prompt 长度由 JSON
的 `prompt_tokens` 给出，本 checkpoint 上分别为 502、1998。

本次未版本化证据位于 `$JITTOR_LAB_ROOT/_state/310p-perf/`：`bench/` 下的
`jittor-accepted-*`、`torch-final-*` 和 `parity-accepted-*` JSON/NumPy 文件；
CANN 原始 trace 与 CSV 位于
`profile-static-api/`、`profile-torch-api/`。完整命令入口见
[性能 skill](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/skills/jittor-transformers-perf/SKILL.md)。

## 验证范围

定向验收包含真实 NPU 的打包布局/数值/独立快照、ND 与空输入、PFA 全量输出、
Concat/RMSNorm、私有缓冲区生命周期、异步 device/pinned 释放，以及 buffer 更新后
的图重放。共享前端快速路径 CPU 回归 53 项通过，11 项按加速后端条件跳过。
最终 RMSNorm NPU 回归 48 项通过，包含 AMP、形状变化、重放、缓冲区释放、
推理后训练及此前失败序列的解析梯度检查。

提交门禁：CPU `core` 原生会话 98 passed / 41 skipped / 1 xfailed，Torch
会话 224 passed / 1 skipped。完整 `tests/structure` 按 `process_modes.py`
拆分，Torch 会话 1409 passed / 4 skipped，原生 dtype recorder 5 passed；
两会话严格执行覆盖检查均退出 0。4 项跳过来自加速器条件与显式声明，不作为
已验证功能计数。仓库布局与生成的 manifest 检查通过。

本报告限于该模型、设备、精度、长度和生成方式，不外推 sampling、多请求、训练
或其他模型。CPU 与 310P 已实际执行；本次没有 CUDA 或多张 ACL 卡用于硬件验证。
