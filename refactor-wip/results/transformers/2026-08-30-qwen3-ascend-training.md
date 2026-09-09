# Qwen3-0.6B Ascend 训练验证与优化

- Status: FP32 forward/backward accepted; BF16 one-step forward parity accepted,
  exact-path performance and cross-framework trajectory open
- Last reviewed: 2026-09-02
- Source baseline: `d02a72ed` plus this report's contiguous-slice follow-up; semantic baseline `8ab4d2b5`
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

后续复查撤回了旧版报告中“显式 RMSNorm adapter 快 `8.3%`”的归因。Torch 兼容层的
标准 RMSNorm hook 已经自动接管 `*RMSNorm` 模块，显式替换
`Qwen3RMSNorm.forward` 没有改变计算图；当时 `221.56 ms` 与 `241.20 ms` 的差异是
短序列运行波动，不能作为 adapter 收益；这些旧测量不再作为当前结论。

## BF16 semantic parity follow-up

`8ab4d2b5` 修复了三处最先出现的 BF16 语义差异：

- Python 浮点标量与 BF16 tensor 运算按 `torch.result_type` 保持 BF16，不再提升到
  FP32；
- RMSNorm 使用 BF16 单位权重完成归一化，再单独乘原始 BF16 weight，保留 PyTorch
  “FP32 归一化、BF16 舍入、BF16 affine”的顺序；
- BF16 SiLU 改用 CANN 9 `aclnnSwish`。独立 raw CANN 探针与 `torch_npu` 的
  BF16/FP32 前向、反向在固定边界值上逐元素一致，包括旧 `aclnnSilu` 会偏离的
  `5.9375`。

同一 checkpoint 和输入下，Jittor 的 29 份 hidden state 与最终 logits 均与
`torch_npu` 逐元素一致。`hidden_grad_28` 逐元素一致，误差向前累积到
`hidden_grad_0` 时相对 L2 为 `0.022717`；logits gradient 只差一个约
`9.09e-13` 的 FP32 元素。Jittor/native 初始 loss 为
`3.377089977/3.377089262`。这接受 BF16 单步前向和有限梯度，不等同于长期训练轨迹
已经验收。

在一张空闲 910B3 上紧邻执行 Transformers 5.5.3、BF16 eager、显式 fused AdamW、
2 次 warmup 和 5 次同步计时，结果如下：

| 实现 | Forward | Backward | AdamW | Full step | vs native |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native `torch_npu`, fused | 86.06 ms | 111.50 ms | 19.82 ms | 217.39 ms | 1.000x |
| Jittor, exact default | 116.06 ms | 91.35 ms | 25.13 ms | 232.54 ms | 1.070x |
| Jittor, direct CANN RoPE | 101.78 ms | 87.92 ms | 25.06 ms | 214.76 ms | 0.988x |
| Jittor, exact Roll-RoPE experiment | 115.66 ms | 93.76 ms | 24.50 ms | 233.92 ms | 1.076x |

直接 CANN RoPE 达到性能目标，但 logits 相对 L2 为 `0.04270`，最差 retained hidden
gradient 相对 L2 为 `0.23168`，因此拒绝作为默认路径。基于 `aclnnRoll` 的表达式与
默认 snapshot SHA-256 完全相同，但没有性能收益。当前精确路径仍比相邻 native 慢
约 `7.0%`，性能门禁保持开放。所有 Jittor 行均为 `fallback_count=0`、
`cpu_compile_count=0`。

## Full-slice identity follow-up

2026-09-02 在当前 `2.0` 基线上复查 exact backward profile，发现 ACL basic slice
会把完整切片也降为 `SliceV2`，其反向再执行
`aclrtMemsetAsync + aclnnStridedSliceAssignV2`。Qwen3 图中 57 次完整切片反向累计
约 `10.742 ms`，虽然它们在数学上只是恒等映射。

`basic_slice_acl` 现在只在所有维度均满足 `begin=0`、`end=shape`、`step=1` 时返回
独立的 Jittor clone。CloneOp 共享输入存储并具有恒等梯度，因此不引入设备拷贝，也
保持返回 Var 与输入不是同一 Python 对象。任一维度为真子区间、负区间或非单位步长
时仍走原来的 CANN SliceV2 路径。

同一张空闲 910B3、Transformers 5.5.3、BF16 eager、显式 fused AdamW、序列长度 8、
2 次 warmup 和 5 次同步采样的紧邻结果如下：

| Path | Forward | Backward | AdamW | Full step | vs native |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native `torch_npu` | `85.718 ms` | `111.016 ms` | `22.260 ms` | `218.995 ms` | `1.000x` |
| Jittor before follow-up | `118.577 ms` | `114.249 ms` | `25.086 ms` | `257.912 ms` | `1.178x` |
| Jittor full-slice run A | `120.222 ms` | `106.535 ms` | `25.095 ms` | `251.852 ms` | `1.150x` |
| Jittor full-slice run B | `118.437 ms` | `102.245 ms` | `24.988 ms` | `245.670 ms` | `1.122x` |

两个候选进程的中心为 `248.761 ms`、约 `1.136x` native；相对紧邻原始 Jittor
改善约 `3.5%`。当前 native 与此前 `217.39 ms` 基线一致，而当前 Jittor 原始路径
比 2026-09-01 的 `232.54 ms` 慢；因此本节只接受候选相对紧邻原始路径的改善，不把
不同日期的漂移归因于本修改，也仍不接受 exact-path 性能目标。

device profile 与实现预期一致：

| Phase | Before | After | Launches before/after |
| --- | ---: | ---: | ---: |
| Forward | `49.480 ms` | `50.699 ms` | `1266 / 1205` |
| Backward | `98.228 ms` | `88.666 ms` | `1666 / 1609` |

完整切片 backward 从 57 次降到 0；剩余 112 次真半切片仍使用
`StridedSliceAssignV2`，累计约 `21.371 ms`。尝试将这些连续末维半切片改为
`aclrtMemsetAsync + aclrtMemcpy2dAsync` 后，backward 退化到 `125.428 ms`、full step
退化到 `272.504 ms`，该实现已完全撤回。

候选重新生成的 61 个 hidden/logits/gradient 数组与已验收 exact snapshot 全部
逐元素一致，两个 NPZ 的 SHA-256 均为
`e9165acd269c11086dfb81effa90ecb7f49e89e2ae368a7e83880c324b7ce807`；整模和定向
用例均为零 CPU compile/fallback。

### Contiguous slice-gradient follow-up

完整切片化简后，profile 仍有 112 次 RoPE 半切片反向，每次执行 memset 加
`StridedSliceAssignV2`，合计 `21.371 ms`。新的受限 lowering 只接管 FP16、BF16 和
FP32、非空、单位步长、前置维全部取满、仅末维为连续子区间的 basic slice：forward
仍使用 SliceV2；backward 将 `dout` 与左右零块通过一次 CANN Cat 拼回原形状。其他
dtype、跨维子区间和非单位步长继续使用原 slice-scatter。

零块由最多 32 项的 LRU 按 `(device_id, shape, dtype)` 缓存并标记 stop-grad。它们作为
SliceV2 CodeOp 的显式输入进入依赖图，因此没有未跟踪的异步初始化；Qwen3 的 56 个
Q/K rotate-half 调用只保留两种 shape 的 BF16 零模板。

扩大 indexing 回归时还复现了 ACL 构建中的 CPU scope 问题：中央 `warp()` 和 bool
mask 规范化使用了不同的 backend 判断。现在统一以 `use_acl && use_cuda` 判断 ACL
device execution；CPU bool mask 的 `nonzero [N, rank]` 被转换为逐维坐标 tuple，而
不是错误 flatten 为 `N*rank` 个行索引。同 rank、低 rank 和 `masked_select` 三项从
错误 shape 恢复为 Torch 结果。

在空闲的同一张 910B3 上紧邻执行，配置仍为 Transformers 5.5.3、BF16 eager、显式
fused AdamW、序列长度 8、2 次 warmup 和 5 次同步采样：

| Path | Forward | Backward | AdamW | Full step | vs native |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native `torch_npu` | `81.120 ms` | `111.297 ms` | `19.006 ms` | `211.424 ms` | `1.000x` |
| Jittor `d02a72ed` | `120.805 ms` | `106.521 ms` | `25.275 ms` | `252.601 ms` | `1.195x` |
| Jittor candidate A | `123.489 ms` | `79.202 ms` | `24.502 ms` | `227.192 ms` | `1.075x` |
| Jittor candidate B | `118.797 ms` | `77.915 ms` | `24.575 ms` | `221.287 ms` | `1.047x` |
| Jittor candidate C | `120.817 ms` | `79.881 ms` | `24.139 ms` | `224.837 ms` | `1.063x` |

候选 full-step 三进程中位数为 `224.837 ms`，相对同卡 `d02a72ed` 改善约 `11.0%`，
相对 native 为 `1.063x`。该优化已稳定收窄差距，但尚未满足“不慢于原生”的性能
门禁。

最终 profile 中 112 次 `StridedSliceAssignV2` 全部消失，112 次
`contiguous_slice_grad` Cat 合计 `3.398 ms`。相对只含完整切片化简的 profile，forward
从 `50.699 ms` 降至 `48.209 ms`，backward 从 `88.666 ms` 降至 `68.203 ms`；launch
数分别保持 `1205/1609`，收益来自更低成本的 lowering 而非省略同步。optimizer
profile 只有一个 `fused_adamw`，device time 为 `11.500 ms`，但整段墙钟约
`24.5-25.3 ms`；剩余开销包含 930 个参数的 Python 状态整理、Var 重绑定和 ACL
descriptor 调度，尚未优化。

候选 exact snapshot 的 61 个数组继续逐元素一致，NPZ SHA-256 仍为
`e9165acd269c11086dfb81effa90ecb7f49e89e2ae368a7e83880c324b7ce807`。尝试用非连续
ACL tensor view 直接实现 exact rotate-half 时在最小用例触发 vendor `libnnopbase`
段错误，已完全撤回；不以 fallback 隐藏该失败。

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
- BF16 semantic follow-up：完整 real-NPU Torch-compat ACL 文件 `19 passed`；CPU
  dtype 文件 `25 passed, 2 skipped`；干净结构测试 `232 passed, 2 skipped`，仓库布局
  门禁通过。
- slice follow-up：real-NPU ACL indexing `7 passed`（内部 29 组 indexing 子用例全部
  零误差），完整 Torch indexing CPU/NPU `26 passed`，完整 native ACL `46 passed`；
  完整 Torch-compat ACL `21 passed`；FP16/BF16/FP32 连续末维首段、中段、末段梯度及
  缓存复用通过，61 个整模 snapshot 数组逐元素一致。

native 与 Torch-compat 文件必须按仓库规则放在不同进程运行；同进程导入兼容层会
改变 native 返回类型，不能作为有效门禁方式。

## 复现与产物

probe 和原始日志未版本化，位于：

```text
$JITTOR_LAB_ROOT/qwen3-npu-training/probe_qwen3_training.py
$JITTOR_LAB_ROOT/_state/qwen3-npu-training/20260830/logs/
$JITTOR_LAB_ROOT/_state/qwen3-bf16/20260901/
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
2a11c8e649173a9bbaef6fc19efcd41bc3b3852cd907af84643e6a4d909ece6a  probe_qwen3_training.py (BF16 semantic follow-up)
32e90b3c36dbfb3c9db995a2078873bc59f98ca42316256b008b819c745442af  probe_aclnn_swish.cc
e9165acd269c11086dfb81effa90ecb7f49e89e2ae368a7e83880c324b7ce807  Jittor exact/roll-RoPE snapshot
948b4b9c40334039bdb8a249c68b4d4088cde83e091f1645045bb815cfd9024f  torch_npu BF16 gradient snapshot
f2ee46d87ea1fbe99eac0f4c63dabd788c092d71fa626bd3753b2f7bcf4c4578  exact forward profile
04bbf90839ed37c312d26138ec617ce9a66c2618460c93728fdb82ada85ec8c7  exact backward profile
570d96d1ece8657eab81f882c2be1b671f45d152b6060afe60652632a726769e  current native adjacent log
ae15933b56e35fbd3226d5cf80ab6b1ca11d9ed43fd28f2edddfd87a1dd83186  current Jittor pre-follow-up log
d7f229e5b682855597dd476fa572a1018d8da74bb740d6179bb03cc6c61db52f  full-slice run A log
e28583a6f35803f2f7db874d855d0b488816549da4073bbecdd28105f2a77388  full-slice run B log
6aaf0ae8819e8abe3322eae6e07a9c8d3617328d3fa1340abf4049308e6852a1  full-slice forward profile
4faebde56b4b3151385349f2e4674d11a55c7c613253ee16ced77d06fabe3bcc  full-slice backward profile
a3cbe9d78af7eafdb6124bb4dfc7a785de76eee562ed3fa423cbd8b26776efdc  rejected D2D half-slice log
e9165acd269c11086dfb81effa90ecb7f49e89e2ae368a7e83880c324b7ce807  full-slice exact snapshot
3c8e4ceb369392f126c4722d5011e535fe8e213a8b2352bd7b9a19e04b2b57e6  NPU1 native adjacent log
53fce9bfb652e4ee905a6d876b9d66db610cbe3095673c701a6acb9ec841b4f6  NPU1 d02a72ed adjacent log
a068acad7a1733c79b6709779286baedf7b37744972d724a8a3bcfbc7fadbc2b  contiguous-slice run A log
3a436f3d6b47fa6a5b4122a7358d157f543551975915e154199f7bb99c1d9de2  contiguous-slice run B log
8293648dc9a2155b7e7f092e7ef91f6f5f4b1d0ebd6c7e6267aa05e4a6368f4e  contiguous-slice run C log
c4cc2904e3de2d3d8fa4ebb8ee963f04b911f24d60735bccf6565e77195e8868  contiguous-slice forward profile
5a703f2af4ad00726c1553e0d974d4fd0aacf8c25dca10e60ba946fd18382704  contiguous-slice backward profile
eac08bf9f73673df844f0fac0440bc0e1b8fbc79f66a9129d85782cf983c2fe2  fused AdamW optimizer profile
e9165acd269c11086dfb81effa90ecb7f49e89e2ae368a7e83880c324b7ce807  contiguous-slice exact snapshot
```

## 未覆盖边界

- FP32 optimizer、完整 checkpoint restore 和长期参数轨迹仍未验证；BF16 显式 fused
  接受固定梯度两步精确对拍和短序列性能，整模跨框架训练轨迹仍未验收；
- 没有验证 FP16、Qwen3-8B 训练、多卡训练、采样或量化；
- 精确 BF16 full step 的历史相邻结果慢约 `7.0%`，当前 slice follow-up 的同卡三进程
  中位数约慢 `6.3%`；直接 CANN RoPE 因改变 BF16 舍入轨迹不应成为默认路由；
- embedding 快路径未声明 `scale_grad_by_freq=True`、`max_norm` 或 sparse 支持；
- 性能数字只代表该单卡、sequence length 8 的对应同步 eager 协议；FP32 不含
  optimizer，BF16 follow-up 包含 AdamW。
