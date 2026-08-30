# Qwen3-0.6B Ascend FP32 前向与反向优化

- Status: Accepted for FP32 eager forward, causal-LM loss, and backward on one Ascend 910B3
- Last reviewed: 2026-08-30
- Source baseline: `b6ce20cdf` plus the changes in this report's commit
- Owner: Torch compatibility and ACL backend maintainers
- Review when: CANN, Transformers Qwen3 modules, embedding/RMSNorm/RoPE lowering,
  dtype, checkpoint, sequence shape, optimizer, or timing protocol changes

## 结论

本结论是独立的真实 NPU 验证，不来自 Transformers、Diffusers、ms-swift、
MMCV/MMEngine、verl 或 vLLM 的 CPU/CUDA 维护用例。Transformers 4.56.2 的
Qwen3-0.6B 在一张 910B3 上完成 FP32 eager 前向、causal-LM loss 和反向；最终两次
稳态运行均为 `cpu_compile_count=0`、`fallback_count=0`，loss 为
`3.2688751221`，选取的 embedding、首层 attention 和末层 MLP 梯度均存在且有限。

最初跑通后的主要耗时不是单一 GEMM，而是训练图中的设备回退和组合算子提交：

- label constant pad 使用 reindex，导致 NPU 训练图出现 CPU 路径；
- RMSNorm 和 RoPE 训练时未接入 CANN backward；
- embedding backward 对 `[151936, 1024]` 权重做通用 IndexPut accumulate。

本次为 constant pad 增加受限 ACL concat 路径，为 RMSNorm、RoPE 和 embedding
接入 CANN 前向/反向，并用独立 NumPy 参考验证输出和梯度。最终 Jittor 的
前向+反向中位数为 `204.44-214.08 ms`，同协议 `torch_npu` 为 `190.58 ms`，
即约 `1.07x-1.12x`；最初跑通时为 `356.84 ms`、约 `1.87x`。

## 验证口径

| 项目 | 配置 |
| --- | --- |
| Device | one Ascend 910B3 |
| Driver / CANN | 25.5.1 / 9.0.0 |
| Model | local Qwen3-0.6B, 596,049,920 parameters |
| Transformers | 4.56.2 |
| Dtype / attention | float32 / eager |
| Input | fixed tokenizer output, sequence length 8 |
| Timing | 2 warmups, 5 repeats, synchronized phase medians |
| Measured step | `zero_grad` + forward/loss + `loss.backward()` |

计时不包含模型加载，也不包含 `optimizer.step()`。Jittor 和 `torch_npu` 使用相同
checkpoint、token ids、dtype、attention 实现、warmup/repeat 数量和同步边界。

Transformers 4.56.2 的 Qwen3 模块默认内联组合 RoPE，不会调用
`jt.nn.rotary_emb`。因此带融合 RoPE 的性能运行在未版本化 probe 中显式替换
`modeling_qwen3.apply_rotary_pos_emb`，只把 q/k/cos/sin 转发给该公共 Jittor
入口；没有该补丁时模型仍可正确运行，但不会命中本报告的 RoPE 融合性能路径。

## 性能结果

| 实现 | Forward median | Backward median | Combined | vs `torch_npu` |
| --- | ---: | ---: | ---: | ---: |
| Native `torch_npu` | 87.348 ms | 103.230 ms | 190.579 ms | 1.000x |
| Jittor, constant-pad device fix | 161.586 ms | 195.253 ms | 356.839 ms | 1.872x |
| Jittor, + RMSNorm backward | 141.387 ms | 143.817 ms | 285.204 ms | 1.496x |
| Jittor, + explicitly patched RoPE, run A | 116.284 ms | 114.489 ms | 230.773 ms | 1.211x |
| Jittor, + explicitly patched RoPE, run B | 110.476 ms | 108.771 ms | 219.247 ms | 1.150x |
| Jittor, + embedding backward, run A | 109.921 ms | 94.521 ms | 204.443 ms | 1.073x |
| Jittor, + embedding backward, run B | 113.954 ms | 100.126 ms | 214.080 ms | 1.123x |

两个最终重复结果的中心约为 `1.10x`。最终 backward 已与 native 相当，剩余差距
主要在 forward；本报告不把短序列的测量波动解释成 backward 稳定快于 native。

## 代码与正确性

- constant pad 仅在 ACL、非负整数 padding width 和 constant mode 下接管；
  int64 label pad 与 float32 backward 均验证无 CPU 路径；
- `aclnnRmsNormGrad` 对 x 和 gamma 的梯度与独立 NumPy 公式一致；
- `aclnnRotaryPositionEmbeddingGrad` 对 q、k、cos 和 sin 的梯度与独立 NumPy
  公式一致；
- `aclnnEmbeddingDenseBackward` 验证重复 token 累加、`padding_idx=3` 冻结和
  `padding_idx=None`；快路径当前只接管 FP32、`scale_grad_by_freq=False`、
  `max_norm=None` 和 `sparse=False`；
- 模型最终两次运行均无 ACL fallback 或 CPU-compiled operation；`lm_head.weight`
  未单独出现在 `named_parameters()` 是 tied weights 去重，不是缺失梯度错误。

回归结果：

- CPU pad/embedding 定向 OpInfo：`42 passed, 3 skipped, 660 deselected`；
- real-NPU 新增算子定向集合：`7 passed, 29 deselected`；
- real-NPU native ACL 文件：`32 passed`；
- real-NPU Torch-compat ACL 文件：`4 passed`。

native 与 Torch-compat 文件必须按仓库规则放在不同进程运行；同进程导入兼容层会
改变 native 返回类型，不能作为有效门禁方式。

## 复现与产物

probe 和原始日志未版本化，位于：

```text
$JITTOR_LAB_ROOT/qwen3-npu-training/probe_qwen3_training.py
$JITTOR_LAB_ROOT/_state/qwen3-npu-training/20260830/logs/
```

在隔离的 Jittor cache、已 source CANN 且只暴露已分配设备后，核心调用为：

```bash
JITTOR_TORCH_SHIM=1 python probe_qwen3_training.py \
  --backend jittor --model "$QWEN3_MODEL" \
  --sequence-length 8 --warmups 2 --repeats 5 --jittor-fused-rope

python probe_qwen3_training.py \
  --backend torch --model "$QWEN3_MODEL" \
  --sequence-length 8 --warmups 2 --repeats 5
```

关键未版本化产物的 SHA-256：

```text
c27c0d6ef45856e76506b7ad77c150c310c46e1287025f82b8656c2f9d231c2c  probe_qwen3_training.py
f7d9d668baf3747e519f50f9891dd6b1645c095f83eebad02203b8a24b1dabbd  torch-fp32-train-steady.log
b5c4146ec7908efb8307bf15c68420e2c663479a68ec22956d4ad70dbdfad4fb  jittor-fp32-train-steady.log
66b956de542e03d9c8eb12c09eed9a16f403a2fb994cf67dd932e3c84d095a51  jittor-fp32-train-rmsnorm-steady.log
82affecf2224e252512a2576cb8f9e148f8f7ef2feb9ccb2b52ced5f08c00923  jittor-fp32-train-rmsnorm-rope-patched-steady.log
93334fdd9d49bf822f2bd076b96a09a00b55341eec34c7e8ea441802b4d651c9  jittor-fp32-train-rmsnorm-rope-patched-steady-repeat.log
bc8d79594b1be3892259b0f965d22ddcf594dd6f93070fa6fd14a1d74f1a6c0c  jittor-fp32-train-rmsnorm-rope-embedding-steady.log
cc5e82527e7dc00cbc3b8b83d7e10d1402b1f6b8712639995974d81ac9afa9a3  jittor-fp32-train-rmsnorm-rope-embedding-steady-repeat.log
```

## 未覆盖边界

- 没有验证 optimizer update、参数轨迹或 checkpoint restore，因此不能称为完整训练；
- 没有验证 BF16/FP16 训练、Qwen3-8B 训练、多卡训练、采样或量化；
- Transformers 默认 Qwen3 RoPE 尚未自动路由到融合入口；
- embedding 快路径未声明 `scale_grad_by_freq=True`、`max_norm` 或 sparse 支持；
- 当前性能数字只代表该单卡、FP32、sequence length 8 的同步 eager 协议。
