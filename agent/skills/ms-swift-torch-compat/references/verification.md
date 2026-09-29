# ms-swift CUDA 验收合同

本合同只判定真实 CUDA 证据。CPU 是共享的算子/兼容性前置；CUDA 是本任务目标设备。未运行、依赖缺失、无公开入口或性能前置未满足都保持 `not-run`/`blocked`，不写成通过。

## L0-L5

| 等级 | 必须证据 |
| --- | --- |
| L0 | 两侧身份断言；真实 ms-swift import；模型、tuner、训练/推理组件构造；版本、配置、设备和 fallback 合同成立 |
| L1 | 同权重同输入的 native CUDA vs Jittor CUDA 前向；结构、shape、dtype、有限值和推理 greedy token 一致 |
| L2 | 全部可训练参数梯度、适用输入梯度、loss、更新量、参数和优化器状态；至少三步固定数据轨迹对拍 |
| L3 | 不中断与全新进程 checkpoint 恢复各自一致；跨两侧比较模型、buffers、optimizer/scheduler、RNG、global step、下一批和继续训练轨迹 |
| L4 | 真实公开 ms-swift CLI/launcher；训练和适用推理入口；子进程继承 CUDA、离线配置和 `fallback_policy=error` |
| L5 | 前置等级通过后，真实目标尺寸预热后至少 10 次；同步测量训练或 prefill/decode；报告 native/candidate ratio、吞吐、显存和零 fallback |

每个适用等级都检查：参数/输入/输出/梯度真实 CUDA 驻留；candidate fallback 策略为 error 且完整窗口计数为零；键集合、shape、dtype、NaN/Inf、数值容差预先固定。不能用只比较最终 loss、只传 adapter 权重、旧进程内存或 tiny timing 替代完整证据。

## 数值与范围

先执行 CPU native-vs-shim，排除 compat 语义问题；再执行 CUDA native-vs-shim。CUDA 的默认生态容差沿用 harness 的 `atol=5e-3, rtol=2e-2`，但 BF16 RMSNorm 等敏感算子必须另报最大绝对误差、全场 scale、相对 L2 和首个分歧层，不得看到结果后放宽门槛。若 native PyTorch 自身也失败，标记为上游/版本问题；若只 shim 失败，按三行判据修核心、compat 或 adapter。

## ms-swift 覆盖清单

以本地 ms-swift checkout 的 `tests/`、`.github/workflows/citest.yaml`、`examples/` 和公开 `swift` CLI 为清单来源。先列出依赖/模型/数据/服务前置，再逐项执行：通用工具与模板、tiny Transformers/LoRA 构造、训练 SFT/分类/embedding/生成式任务、checkpoint/resume、推理入口和 CUDA 多步训练。需要 NPU、TP/多机、vLLM/sglang、联网模型或缺失可选依赖的项目单独标明 `not-applicable` 或 `blocked`，不伪造 CUDA 通过。

## 分流与修复

能力缺失修 Jittor core/CUDA kernel；Torch API 拼写或状态语义修 `jittor.compat.torch`；仅 ms-swift 私有且可迁移的 glue 才进 adapter；runner 误报修 harness。不得修改 ms-swift 源码绕过失败。每个修复保留首次失败日志、最小回归和受影响等级的重新运行记录。
