# Qwen3-0.6B 在 Ascend 910B3 上训练：正确性与性能

- 状态：FP32 前向/loss/反向接受（约 `1.07x-1.12x` 原生 `torch_npu`）；BF16 单步
  前向逐元素对齐接受；**BF16 精确路径性能门禁与跨框架训练轨迹仍开放**
- 日期：2026-08-30（BF16 部分至 2026-09-02）
- 基线提交：FP32 与切片后续为 `d02a72ed` 加本报告的连续切片改动；BF16 语义对齐
  为 `8ab4d2b5`
- 验证范围：一张 Ascend 910B3，驱动 25.5.1 / CANN 9.0.0；Qwen3-0.6B（596,049,920
  参数）；FP32 用 Transformers 4.56.2，BF16 用 5.5.3；eager 注意力、序列长度 8、
  2 次预热、5 次同步计时取各阶段中位数。未覆盖：FP16、Qwen3-8B 训练、多卡、长期
  轨迹、checkpoint 恢复、采样与量化
- 维护者：Torch 兼容层与昇腾后端维护者
- 复查条件：CANN、Transformers 的 Qwen3 模块、embedding/RMSNorm/RoPE/切片的 ACL
  lowering、dtype、优化器或计时协议变化

## 结论

FP32 下完整前向、causal-LM loss 与反向在 NPU 上跑通，两次稳态运行均
`cpu_compile_count=0`、`fallback_count=0`，loss `3.2688751221`，抽查的 embedding、
首层 attention 与末层 MLP 梯度存在且有限。最初跑通时是原生的 `1.87x`；主要开销不是
GEMM，而是训练图里的设备回退与组合算子：label 的 constant pad 走 reindex 出现 CPU
路径，RMSNorm/RoPE 反向未接 CANN，embedding 反向对 `[151936, 1024]` 权重走通用
IndexPut。接入受限 ACL concat、CANN RMSNorm/RoPE/embedding 前反向之后：

| 实现 | Forward | Backward | 合计 | 对 `torch_npu` |
| --- | ---: | ---: | ---: | ---: |
| 原生 `torch_npu` | 87.348 ms | 103.230 ms | 190.579 ms | 1.000x |
| Jittor，仅修 constant pad | 161.586 ms | 195.253 ms | 356.839 ms | 1.872x |
| Jittor，全部接入（两次运行） | 109.9–114.0 ms | 94.5–100.1 ms | 204.4–214.1 ms | 1.073x–1.123x |

FP32 计时不含 `optimizer.step()`。新算子路径都用独立 NumPy 参考验证了输出与梯度：
`aclnnRmsNormGrad`、`aclnnRotaryPositionEmbeddingGrad`（q/k/cos/sin）、
`aclnnEmbeddingDenseBackward`（重复 token 累加、`padding_idx`）。

## BF16：语义对齐

三处最先出现的 BF16 语义差异已修：Python 浮点标量与 BF16 张量运算按
`torch.result_type` 保持 BF16；RMSNorm 保留 PyTorch「FP32 归一化、BF16 舍入、BF16
affine」的顺序；BF16 SiLU 改用 `aclnnSwish`（旧 `aclnnSilu` 在 `5.9375` 等边界值上
偏离）。修后同一 checkpoint 与输入下，29 份 hidden state 与最终 logits 与
`torch_npu` 逐元素一致，初始 loss `3.377089977` 对 `3.377089262`；反向误差逐层累积到
`hidden_grad_0` 时相对 L2 为 `0.022717`。**这接受单步前向与有限梯度，不等于长期训练
轨迹已对齐。**

显式 `fused=True` 时 AdamW 走 CANN `aclnnApplyAdamWV2`，两步 BF16 参数、一二阶矩与
`torch.optim.AdamW(fused=True)` 逐元素一致；默认 `fused=None` 保持 foreach 式舍入
语义，不拿一次舍入的结果冒充默认路径。

## BF16：性能（仍开放）

Transformers 5.5.3、BF16 eager、显式 fused AdamW、同卡紧邻执行：

| 路径 | Forward | Backward | AdamW | Full step | 对原生 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原生 `torch_npu` | 81.120 ms | 111.297 ms | 19.006 ms | 211.424 ms | 1.000x |
| Jittor `d02a72ed` | 120.805 ms | 106.521 ms | 25.275 ms | 252.601 ms | 1.195x |
| Jittor，加连续切片反向（三进程中位数） | 120.817 ms | 79.881 ms | 24.139 ms | 224.837 ms | 1.063x |

两处切片改动：完整切片（`begin=0, end=shape, step=1`）返回共享存储的 clone，反向
不再做 memset 加 `StridedSliceAssignV2`；仅末维连续子区间的切片，反向用一次 CANN Cat
把 `dout` 与缓存的零块拼回原形。profile 中 112 次 `StridedSliceAssignV2` 全部消失，
backward device time `88.666 → 68.203 ms`；61 个 hidden/logits/梯度数组与已验收的
精确快照逐位相同。剩余差距主要在 forward 与优化器周边的 Python 状态整理（930 个参数）。

被否决或撤回的尝试：直接用 CANN RoPE 能到 `0.988x`，但 logits 相对 L2 `0.04270`、
最差 hidden 梯度 `0.23168`，改变了 BF16 舍入轨迹，不作默认；用 memset+2D memcpy
做半切片反向使 full step 退化到 `272.504 ms`；用非连续 ACL tensor view 做 rotate-half
在最小用例触发 vendor `libnnopbase` 段错误。三者均已完全撤回，没有用回退掩盖。

## 复现

训练探针未进仓库（仓库内的 `tests/backends/acl/manual/run_qwen3_transformers.py`
只做推理）。协议：兼容层用完整部署的 shim site，而不是只设环境变量。

```bash
python -m jittor.compat.shim.deploy --target "$RUN_ROOT/shim-site"
export PYTHONPATH="$RUN_ROOT/shim-site:$JITTOR_SOURCE/python"
source "$CANN_SET_ENV"; export ASCEND_RT_VISIBLE_DEVICES=<allocated-device>
# Transformers 5.5.3 的异步加权重线程没有 ACL 线程上下文
export HF_DEACTIVATE_ASYNC_LOAD=1
JITTOR_TORCH_SHIM=1 python <training-probe> --backend jittor --model "$QWEN3_MODEL" \
  --sequence-length 8 --dtype bfloat16 --optimizer adamw --fused --warmups 2 --repeats 5
python <training-probe> --backend torch --model "$QWEN3_MODEL" \
  --sequence-length 8 --dtype bfloat16 --optimizer adamw --fused --warmups 2 --repeats 5
```

两侧用同一 checkpoint、token ids、dtype、注意力实现与同步边界；计时不含加载。
Transformers 4.56.2 的 Qwen3 内联组合 RoPE、不调用 `jt.nn.rotary_emb`，FP32 的融合
RoPE 路径需要探针显式替换 `modeling_qwen3.apply_rotary_pos_emb`。native 与 Torch 兼容
的回归文件须分进程运行。

## 仍开放

- BF16 精确路径 full step 仍比原生慢约 6.3%（性能门禁未过）；
- 跨框架 BF16 训练轨迹、FP32 optimizer 与 checkpoint 恢复未验收；
- embedding 快路径不覆盖 `scale_grad_by_freq=True`、`max_norm` 与 sparse；
- 数字只代表单卡、序列长度 8 的同步 eager 协议。

当前树（`e3c369acb`）上这些路径仍在（`backends/acl/kernels/native/` 下的 AdamW、
embedding、norms、SiLU runner 与 `getitem_op.py` 的切片 lowering），本报告之后没有新的
NPU 实测；问题总账中没有单独条目，本报告就是该性能门禁的唯一记录。
