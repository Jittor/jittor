# verl 核心算法 Ascend 数值、梯度与性能复验

- Status: core-algorithm correctness accepted on a real Ascend 910B3; performance and end-to-end PPO open
- Date: 2026-09-02
- Jittor baseline: `3758c4ab`
- verl source: `3d66a3d7ca1cf783df949816ec6862d5a7af9406`
- Owner: Jittor Torch compatibility and verl adapter maintainers
- Review when: verl policy-loss formulas, Jittor clamp/autograd semantics, CANN, or the NPU adapter changes

## Scope

本轮从报告锁定的上游提交恢复 verl，直接加载该提交中的
`verl/trainer/ppo/core_algos.py`、`verl/utils/torch_functional.py` 和 groupwise
实现。探针只为重量级 worker config 和未参与计算的可选依赖提供最小类型外壳；
policy loss、loss aggregation、masked helpers 和 GRPO groupwise 计算均来自上游源码。

同一组固定输入分别在原生 `torch_npu` 和 Jittor Torch shim 进程执行。覆盖：

- vanilla PPO、GSPO、SAPO、GPG、geometric-mean 和 CISPO policy loss；
- 每个 policy loss 对 `log_prob` 的完整梯度；
- GRPO outcome advantage 和 returns；
- scalar/tensor clamp、精确边界和 clip 后 minimum 的反向。

本报告不证明 verl 完整安装、Ray worker、FSDP2、权重传输、rollout 或 1-step PPO
已经在 NPU 上通过。此前这些闭环仍只有 CPU/CUDA 证据。

## Reproduced issue

修复前，Jittor `clamp(x, -0.15, 0.25)` 在两个精确边界的输入梯度为 0：

```text
input:       [-0.16, -0.15, -0.14, 0.24, 0.25, 0.26]
torch_npu:   [ 0.00,  1.00,  1.00, 1.00, 1.00, 0.00]
Jittor NPU:  [ 0.00,  0.00,  1.00, 1.00, 0.00, 0.00]
```

verl geometric-mean loss 的一个 `negative_approx_kl` 恰好等于 clip 上界，因此
对应梯度在 Jittor 中为 0、原生为 `0.0051804874`。其他 loss、梯度和 GRPO 输出
已经一致。

`fdae6a0f` 将 clamp 表达为包含边界的 ternary 选择，并保留 NaN。它同时修正：

- `x == min/max` 时输入梯度为 1；
- reversed scalar bounds 按 PyTorch 返回全 `max`，而不是断言失败；
- tensor bounds 的 input/min/max 梯度路由；
- FP16/BF16 Python scalar bounds 的同 dtype 计算；
- integer tensor 配 Python 浮点 bound 时提升到 float32。

`3758c4ab` 进一步让 ACL float32、有序双 Python scalar bound 的 forward 调用
`aclnnClampTensor`；tensor bounds、反向 bounds、其他 dtype 继续走组合实现。自定义
backward 保持边界梯度为 1、NaN 梯度为 0，未把 CANN forward 直接当作完整语义。

## Correctness result

固定 `4 x 6` 输入同时包含正负 advantage、ragged mask、上下 clip 和精确 clip 边界。
修复后的两端结果为：

| Path | Loss max abs | Gradient/output max abs |
| --- | ---: | ---: |
| vanilla PPO | `0` | `0` |
| GSPO | `0` | `0` |
| SAPO | `0` | `0` |
| GPG | `0` | `0` |
| geometric-mean | `0` | `0` |
| CISPO | `0` | `0` |
| GRPO advantage/returns | n/a | `0` |

Jittor 结果全部在 `device`，捕获窗口内
`fallback_count=0`、`cpu_compile_count=0`，进程没有加载 `torch_npu`。原生参考
进程实际加载 `torch_npu`，两端使用同一张 910B3 的串行运行。

## Performance result

计时包含上游函数内部 metrics `.item()`、loss forward、对 `log_prob` 的 backward
和设备同步；输入 tensor 在计时前驻留 NPU。每条路径先暖身，再取独立进程内样本
中位数。数值门禁在计时前单独通过。

| Shape | Path | torch_npu | Jittor | Jittor/native |
| --- | --- | ---: | ---: | ---: |
| `64 x 120` | vanilla | `2.809 ms` | `3.507 ms` | `1.248x` |
| `64 x 120` | GSPO | `2.777 ms` | `4.591 ms` | `1.653x` |
| `64 x 120` | SAPO | `2.556 ms` | `3.433 ms` | `1.343x` |
| `64 x 120` | GPG | `0.835 ms` | `0.648 ms` | `0.776x` |
| `64 x 120` | geometric-mean | `2.361 ms` | `3.787 ms` | `1.604x` |
| `64 x 120` | CISPO | `1.694 ms` | `2.528 ms` | `1.492x` |
| `512 x 512` | vanilla | `2.679 ms` | `3.692 ms` | `1.378x` |
| `512 x 512` | GSPO | `2.707 ms` | `4.728 ms` | `1.746x` |
| `512 x 512` | SAPO | `2.511 ms` | `3.403 ms` | `1.356x` |
| `512 x 512` | GPG | `0.846 ms` | `0.655 ms` | `0.774x` |
| `512 x 512` | geometric-mean | `2.397 ms` | `3.823 ms` | `1.595x` |
| `512 x 512` | CISPO | `1.698 ms` | `2.561 ms` | `1.508x` |

GPG 达到不慢于原生；原生 CANN Clamp 将其余五条差距从约 `2.32x-2.62x` 缩小到
`1.25x-1.75x`，但仍未通过性能目标。单独诊断表明 metrics 同步只解释部分差距，
不能用模型训练耗时稀释结果。因此本轮仍不接受 verl NPU 或完整 PPO 性能。

## Validation

- CPU focused clamp boundary/NaN/Torch regression: `3 passed`；此前完整 edge cases
  `24 passed`、Torch compatibility ops `26 passed`。
- Real-NPU focused clamp regression: `5 passed`；完整 Torch compatibility ops
  `27 passed`。最终代码的完整 edge cases 为 `27 passed`，final Torch clamp focused
  为 `2 passed`，clamp OpInfo CPU/NPU 为 `4 passed, 904 deselected`。
- 原生 CANN Clamp forward/backward 定向用例通过；完整 ACL 为 `46 passed`。
- verl Jittor NPU 与 `torch_npu` 比较器：六类 loss/gradient 与 GRPO 全部最大绝对
  误差 `0`，零 fallback，输出驻留 device。
- Python compile、changed-line Ruff 和 `git diff --check`：通过。

## External artifacts

下列文件位于 `$JITTOR_LAB_ROOT`，未版本化，不进入 Jittor 主仓库：

| Artifact | SHA-256 |
| --- | --- |
| `verl-npu/probes/core_algos_parity.py` | `fc3b06f1e574df54d6d7709d33e761f00076622841f125c977ac782e07731d45` |
| `_state/verl-npu/current/native.json` | `c029a3c281355f90c88cff7eb16e56b35fc04b60ecc111aefc114a6a9af0ce01` |
| `_state/verl-npu/current/native-benchmark.json` | `27342011552fba92bdfef35507dda8d10947d13f11413c5a413067969531cab2` |
| `_state/verl-npu/current/jittor-benchmark-cann-clamp.json` | `c31526b5ed8657e930b3d7aec66180931cf3d7b47b2c9d2f71fa4e267d5e5092` |
| `_state/verl-npu/current/native-benchmark-512x512.json` | `673673a081450f52393fbef5fd150f2219aee07d2d95cb198038290b11539803` |
| `_state/verl-npu/current/jittor-benchmark-cann-clamp-512x512.json` | `79e0f657165f1580ac7d931cd1395dfba5bfab21ad31a94ba24e42cabb0a413b` |

## Open work

- 降低 policy-loss metrics 和 backward 的同步/调度开销，使全部六条维护协议不慢于
  原生 `torch_npu`。
- 恢复或重建可维护的 verl NPU adapter，跑通真实 worker import、TensorDict batch、
  actor/critic forward-backward、optimizer、weight transfer 和 1-step PPO。
- NPU FSDP2/HCCL、Ray 多进程、vLLM rollout 和 Qwen3 模型规模仍需分别验收；在此
  之前不能把 CPU/CUDA 的完整 PPO 结论外推到 NPU。
