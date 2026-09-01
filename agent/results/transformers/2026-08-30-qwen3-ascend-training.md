# Qwen3-0.6B Ascend 训练验证与优化

- Status: FP32 forward/backward accepted; BF16 one-step and short-step performance accepted, cross-framework trajectory open
- Last reviewed: 2026-09-01
- Source baseline: `406f56e1`
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

## BF16 optimizer follow-up

Transformers 5.5.3 的同一 Qwen3-0.6B checkpoint 随后在 BF16 下完成完整
forward、causal-LM loss、backward 和一次 AdamW update。596,049,920 个参数、梯度和
optimizer state 均在真实 910B3 上执行，日志为 `fallback_count=0`、
`cpu_compile_count=0`。Jittor Torch shim 还显式让 Transformers 的
`is_torch_npu_available()` 返回 false，进程没有导入真实 `torch_npu` 二进制扩展。
Transformers 5.5.3 的异步权重线程没有 ACL thread context，因此复现命令使用其公开
`HF_DEACTIVATE_ASYNC_LOAD=1` 开关关闭异步加载。

该结果只接受“单步可执行”，不接受 BF16 数值轨迹或性能。固定输入下，Jittor/native
初始 hidden state 完全一致，但差异逐层累积：最终 hidden/logits 的相对 L2 分别为
`0.03573/0.03491`，cosine 为 `0.999362/0.999401`；初始 loss 分别为
`3.313017/3.377089`。七次更新后的 loss 轨迹也不同，因此不能把 BF16 结果表述成
与 `torch_npu` 精确对齐。

同一短序列、2 次 warmup、5 次同步计时的中位数如下：

| 实现 | Forward | Backward | AdamW | Full step |
| --- | ---: | ---: | ---: | ---: |
| Native `torch_npu` | 84.93 ms | 109.48 ms | 30.23 ms | 224.64 ms |
| Jittor | 130.58 ms | 179.36 ms | 285.61 ms | 595.55 ms |

默认 `fused=None` 下，Jittor full step 为 native 的 `2.65x`，其中 AdamW 为
`9.45x`。参数和一、二阶 moment 在更新后保持 BF16，避免了早先 state scalar 将
更新图提升为 FP32；默认路径的舍入语义保持不变。

`532c250b` 进一步接入 CANN 9 `aclnnApplyAdamWV2`。独立 raw ACL 两步探针与
`torch.optim.AdamW(fused=True)` 的 BF16 参数、moment 和 variance 逐元素一致；原生
Jittor 与 Torch-compatible 两步回归同样通过且零 CPU compile/fallback。该能力只在
显式 `fused=True` 时启用，不把 CANN 一次舍入的结果冒充默认 foreach-like 语义。

相同模型、输入和同步计时的显式 fused 结果为：

| 实现 | Forward | Backward | AdamW | Full step |
| --- | ---: | ---: | ---: | ---: |
| Native `torch_npu`, fused | 86.83 ms | 114.64 ms | 20.22 ms | 221.69 ms |
| Jittor, fused TensorList | 134.03 ms | 185.93 ms | 27.33 ms | 347.29 ms |

新的 TensorList mapped op 将 Jittor AdamW 从 `285.61ms` 降到 `27.33ms`，提升
`10.45x`；full step 降低 `41.7%`。但 AdamW 仍为 native 的 `1.35x`，full step
仍为 `1.57x`，因此性能门禁继续开放。七次更新后的 Jittor/native loss 分别为
`3.8173/26.8827`；两侧模型 BF16 梯度本就不同，该数据不能替代固定梯度的 optimizer
逐元素对拍，也不能证明长期训练轨迹一致。

## BF16 training fusion follow-up

`406f56e1` 将 CANN 9 的 embedding、RMSNorm 和 RoPE 训练入口扩展到 BF16。
三条路径分别通过独立 NumPy 前向/梯度参考和真实 NPU 测试，完整 native ACL 文件为
`41 passed`，日志中没有 CPU compile 或 fallback。

整模隔离显示 RMSNorm 是可保持当前 Jittor 数值轨迹的安全优化：仅替换
`Qwen3RMSNorm.forward` 后，保存的全部 hidden state 和 logits 与当前默认路径逐元素
相同，两个 snapshot 的 SHA-256 也相同；七次更新后的 loss、选取梯度和参数更新样本
同样一致。RoPE 融合会改变 BF16 舍入轨迹：当前默认和 RMS-only 的 logits 相对
`torch_npu` L2 为 `0.04388`，加入 RoPE 后为 `0.04807`。因此 RMSNorm 可由外置版本
适配器默认启用，RoPE 只保留显式性能实验，不在核心里自动替换 Transformers 模块。

在一张空闲 910B3 上紧邻执行相同短序列、显式 fused AdamW、2 次 warmup 和 5 次
同步计时，结果如下：

| 实现 | Forward | Backward | AdamW | Full step | vs native |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native `torch_npu`, fused | 102.67 ms | 112.66 ms | 26.20 ms | 241.53 ms | 1.000x |
| Jittor, default Transformers | 110.46 ms | 106.09 ms | 24.65 ms | 241.20 ms | 0.999x |
| Jittor, + RMSNorm adapter | 110.93 ms | 85.94 ms | 24.69 ms | 221.56 ms | 0.917x |
| Jittor, + RMSNorm and RoPE adapters | 91.41 ms | 83.28 ms | 24.57 ms | 199.27 ms | 0.825x |

默认 Transformers 路径的完整 step 已与 native 持平；数值不变的 RMSNorm 适配进一步
快 `8.3%`。最后一行虽快 `17.5%`，但因 RoPE 改变轨迹不作为默认正确性结论。所有
Jittor 行均为 `fallback_count=0`、`cpu_compile_count=0`。

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
- FP32/BF16 `aclnnRmsNormGrad` 对 x 和 gamma 的梯度与独立 NumPy 公式一致；
- FP32/BF16 `aclnnRotaryPositionEmbeddingGrad` 对 q、k、cos 和 sin 的梯度与独立 NumPy
  公式一致；
- `aclnnEmbeddingDenseBackward` 验证重复 token 累加、`padding_idx=3` 冻结和
  `padding_idx=None`；快路径当前接管 FP32/BF16、`scale_grad_by_freq=False`、
  `max_norm=None` 和 `sparse=False`；
- 模型最终两次运行均无 ACL fallback 或 CPU-compiled operation；`lm_head.weight`
  未单独出现在 `named_parameters()` 是 tied weights 去重，不是缺失梯度错误。

回归结果：

- CPU pad/embedding 定向 OpInfo：`42 passed, 3 skipped, 660 deselected`；
- real-NPU 新增算子定向集合：`7 passed, 29 deselected`；
- real-NPU native ACL 文件：`32 passed`；
- real-NPU Torch-compat ACL 文件：`4 passed`。
- BF16 follow-up CPU optimizer：`39 passed, 1 skipped`；real-NPU native fused
  AdamW 定向 `1 passed`，完整 Torch-compat ACL 文件 `16 passed`。
- BF16 embedding/RMSNorm/RoPE follow-up：real-NPU 定向 `3 passed`，完整 native
  ACL 文件 `41 passed`；结构测试 `232 passed, 2 skipped`。

native 与 Torch-compat 文件必须按仓库规则放在不同进程运行；同进程导入兼容层会
改变 native 返回类型，不能作为有效门禁方式。

## 复现与产物

probe 和原始日志未版本化，位于：

```text
$JITTOR_LAB_ROOT/qwen3-npu-training/probe_qwen3_training.py
$JITTOR_LAB_ROOT/_state/qwen3-npu-training/20260830/logs/
```

Transformers 5.5.3 环境使用完整部署的 shim site；仅设置环境变量而不部署 bundled
`torchvision`/`flash_attn` stub 不是本报告的复现协议：

```bash
python -m jittor.compat.shim.deploy --target "$RUN_ROOT/shim-site"
export PYTHONPATH="$RUN_ROOT/shim-site:$JITTOR_SOURCE/python"
```

在隔离的 Jittor cache、已 source CANN 且只暴露已分配设备后，核心调用为：

```bash
JITTOR_TORCH_SHIM=1 python probe_qwen3_training.py \
  --backend jittor --model "$QWEN3_MODEL" \
  --sequence-length 8 --warmups 2 --repeats 5 --jittor-fused-rope

python probe_qwen3_training.py \
  --backend torch --model "$QWEN3_MODEL" \
  --sequence-length 8 --warmups 2 --repeats 5

HF_DEACTIVATE_ASYNC_LOAD=1 JITTOR_TORCH_SHIM=1 \
python probe_qwen3_training.py \
  --backend jittor --model "$QWEN3_MODEL" \
  --sequence-length 8 --dtype bfloat16 --optimizer adamw \
  --warmups 2 --repeats 5

HF_DEACTIVATE_ASYNC_LOAD=1 python probe_qwen3_training.py \
  --backend torch --model "$QWEN3_MODEL" \
  --sequence-length 8 --dtype bfloat16 --optimizer adamw \
  --warmups 2 --repeats 5

# 对两侧同时追加 --fused，复现显式 CANN fused 协议。
```

关键未版本化产物的 SHA-256：

```text
c27c0d6ef45856e76506b7ad77c150c310c46e1287025f82b8656c2f9d231c2c  probe_qwen3_training.py (FP32 evidence snapshot)
f7d9d668baf3747e519f50f9891dd6b1645c095f83eebad02203b8a24b1dabbd  torch-fp32-train-steady.log
b5c4146ec7908efb8307bf15c68420e2c663479a68ec22956d4ad70dbdfad4fb  jittor-fp32-train-steady.log
66b956de542e03d9c8eb12c09eed9a16f403a2fb994cf67dd932e3c84d095a51  jittor-fp32-train-rmsnorm-steady.log
82affecf2224e252512a2576cb8f9e148f8f7ef2feb9ccb2b52ced5f08c00923  jittor-fp32-train-rmsnorm-rope-patched-steady.log
93334fdd9d49bf822f2bd076b96a09a00b55341eec34c7e8ea441802b4d651c9  jittor-fp32-train-rmsnorm-rope-patched-steady-repeat.log
bc8d79594b1be3892259b0f965d22ddcf594dd6f93070fa6fd14a1d74f1a6c0c  jittor-fp32-train-rmsnorm-rope-embedding-steady.log
cc5e82527e7dc00cbc3b8b83d7e10d1402b1f6b8712639995974d81ac9afa9a3  jittor-fp32-train-rmsnorm-rope-embedding-steady-repeat.log
052eb594c9ecca1027054921b9cfd05d1361ac9a556e16745c59f2483fae907c  probe_qwen3_training.py (current BF16/fused follow-up)
ba5c4ba854a18a8111a72f9fbd1ec32407bcb1396eeae6eeb07110daf91ee3c0  jittor-bf16-snapshot.npz
b5374886777d14d1115d3f9d28bcebf8e082320478663752448f73d8371bc9aa  torch-bf16-snapshot.npz
4afc795a30a0de985a8ee81c73e724a4f5fac069b2ee5f3427192830838672ee  probe_apply_adamw_v2.cc
f5d7717a60d29631c7c749c9e3bc3fd11e2a80cecf4c96c361d56ae287768a51  probe_torch_adamw.py
9a17a9a1b99c889f15b5a7dab964976f5532f7698147e76788df01dc29a08ed5  probe_qwen3_training.py (BF16 fusion isolation)
0a66cf06aeef7f48cb03cc92547f430753c16cce7a456415999748052c787a7c  current default / RMS-only snapshot
e55e235f2fe4e9b21e06307672cea3904b9d4b1add51140c316a176db380d3c8  RMSNorm + RoPE snapshot
311298ee7d69cb62a08b6ebea9c429b71665b2b7eeae1b992cece4e3b2cbf8ad  fused forward profile
dab14d76b32dad33d91323d5a109f4a2dcecc908b5fbb06b75aecbf4e8bb09a9  fused backward profile
```

## 未覆盖边界

- FP32 optimizer、完整 checkpoint restore 和长期参数轨迹仍未验证；BF16 显式 fused
  接受固定梯度两步精确对拍和短序列性能，整模跨框架训练轨迹仍未验收；
- 没有验证 FP16、Qwen3-8B 训练、多卡训练、采样或量化；
- 数值不变的 RMSNorm 仍需由外置 Transformers 版本适配器接入；RoPE 因改变 BF16
  舍入轨迹不应成为默认路由；
- embedding 快路径未声明 `scale_grad_by_freq=True`、`max_norm` 或 sparse 支持；
- 性能数字只代表该单卡、sequence length 8 的对应同步 eager 协议；FP32 不含
  optimizer，BF16 follow-up 包含 AdamW。
