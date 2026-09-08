# verl 核心算法 Ascend 数值、梯度与性能复验

- Status: core-algorithm correctness accepted on a real Ascend 910B3; performance and end-to-end PPO open
- Date: 2026-09-02
- Jittor baseline: `97dc6ce9` (source behavior from `3758c4ab`)
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
`1.25x-1.75x`，但仍未通过性能目标。下节的进一步诊断将差距定位到 policy
metrics 触发的惰性图分段；不能用模型训练耗时稀释结果。因此本轮仍不接受 verl
NPU 或完整 PPO 性能。

### Metrics and lazy-graph diagnosis

后续在 `512 x 512` 上对六条路径执行 Jittor device profile。profile 本身有明显
插桩开销，下面的 device time 不能与墙钟 benchmark 直接比较，但图规模和相对热点
可用于定位：

| Path | Profiled device time | Launches |
| --- | ---: | ---: |
| vanilla | `6.301 ms` | 33 |
| GSPO | `5.145 ms` | 34 |
| SAPO | `2.424 ms` | 30 |
| GPG | `0.659 ms` | 8 |
| geometric-mean | `3.245 ms` | 26 |
| CISPO | `1.802 ms` | 27 |

vanilla 的前三个大融合图分别约为 `1.924/1.825/1.024 ms`；它们由 metrics
`.item()`、loss 和 backward 分阶段触发。GPG 没有三项内联 metrics，因此只有 8 次
launch，也已经通过性能门禁。

为隔离该差异，实验 worktree 只把 vanilla 返回的三项 metrics 替换为空字典，loss、
梯度、输入和同步协议保持不变。双端 21 个样本中位数如下：

| Variant | torch_npu | Jittor | Jittor/native |
| --- | ---: | ---: | ---: |
| upstream metrics | `2.679 ms` | `3.692 ms` | `1.378x` |
| metrics removed | `2.619 ms` | `2.589 ms` | `0.989x` |

去掉 metrics 后 Jittor 已略快于原生，且 loss/gradient 仍精确一致、fallback 为 0。
原生三项 metrics 只增加约 `0.060 ms`，Jittor 增加约 `1.103 ms`。因此剩余 vanilla
差距不是 clamp 或 loss/backward 计算，而是惰性图在逐项 `.item()` 处被拆成多个
前向执行阶段。

以下实验均保持数值和梯度一致，但未达到性能目标，已经完全撤回：

| Experiment | Jittor vanilla | Result |
| --- | ---: | --- |
| 四个共享中间量 `stop_fuse` | `3.558 ms` | 仍为原生 `1.328x` |
| ACL `.item()` 改为全图 `sync_all` | `3.744 ms` | 退化且样本出现双峰 |
| metrics 与 loss 一次性取回 | `3.731 ms` | 退化 |
| 先堆叠逐 token metrics 再共同归约 | `3.268 ms` | 最好但仍为原生 `1.220x` |
| 分组 ACL CodeOp masked mean | `3.319 ms` | 比纯 Jittor 分组归约更慢 |

这些结果排除了扩大同步范围、简单切断融合和仅合并归约。下一步需要让 policy-loss
forward、metrics 和 backward 共享一次专用融合计算，或在 verl NPU adapter 中延迟并
统一提取 metrics；不能把全局 `.item()` 行为改成 `sync_all`。

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
| `verl-npu/probes/core_algos_parity.py` | `186da95cf350e842bc6a41a38e54b2249f4c4021fba5524b2b7802775ba34425` |
| `_state/verl-npu/current/native.json` | `c029a3c281355f90c88cff7eb16e56b35fc04b60ecc111aefc114a6a9af0ce01` |
| `_state/verl-npu/current/native-benchmark.json` | `27342011552fba92bdfef35507dda8d10947d13f11413c5a413067969531cab2` |
| `_state/verl-npu/current/jittor-benchmark-cann-clamp.json` | `c31526b5ed8657e930b3d7aec66180931cf3d7b47b2c9d2f71fa4e267d5e5092` |
| `_state/verl-npu/current/native-benchmark-512x512.json` | `673673a081450f52393fbef5fd150f2219aee07d2d95cb198038290b11539803` |
| `_state/verl-npu/current/jittor-benchmark-cann-clamp-512x512.json` | `79e0f657165f1580ac7d931cd1395dfba5bfab21ad31a94ba24e42cabb0a413b` |
| `_state/verl-npu/current/policy-profiles.json` | `966aa6dc16b9336a8a433f8ac0845c3ec9ad22b3943bafa01d379f39039e1427` |
| `_state/verl-npu/current/native-vanilla-no-metrics-512x512.json` | `0bb43b8dc164f6e862a5d4ca9446803135a5960ad8bcceb1025a449bdbad55f0` |
| `_state/verl-npu/current/jittor-vanilla-no-metrics-512x512.json` | `a0f6f64bfffdf5ab87e16b3fcd0379fb9128d903e6a917ffe8835b4ecbc47e2b` |
| `_state/verl-npu/current/jittor-stop-fuse-aggressive.json` | `8ce333a1ebe61d846c66299dc240e4c1bbeba584afd158984047c5271007deb1` |
| `_state/verl-npu/current/jittor-item-sync-all-512x512.json` | `589c54b2a7e9509a4cabf6f0983fad8ccbd972fbc0061d523211cb82c88024c8` |
| `_state/verl-npu/current/jittor-vanilla-batch-metrics-loss-512x512.json` | `9183a7c34036942a8b65db5d18eb6863aa8725cb421268323347de6fc1780e18` |
| `_state/verl-npu/current/jittor-vanilla-grouped-token-metrics-512x512.json` | `d80f6ed1644e84f328bc93c7177b2e47f6b4790b78cf2262889b32d1ba3b6f44` |
| `_state/verl-npu/current/jittor-vanilla-grouped-acl-metrics-512x512.json` | `7835d675ca69b7f1da5938b02543af42164ec916a5b716c8fd7d25f331d11fe5` |

## Open work

- 为 policy-loss forward/metrics/backward 实现专用融合或在 adapter 中统一延迟提取
  metrics，使全部六条维护协议不慢于原生 `torch_npu`。
- 恢复或重建可维护的 verl NPU adapter，跑通真实 worker import、TensorDict batch、
  actor/critic forward-backward、optimizer、weight transfer 和 1-step PPO。
- NPU FSDP2/HCCL、Ray 多进程、vLLM rollout 和 Qwen3 模型规模仍需分别验收；在此
  之前不能把 CPU/CUDA 的完整 PPO 结论外推到 NPU。
