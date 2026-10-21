# verl 核心算法在 Ascend 910B3 上的数值、梯度与性能

- 状态：六类 policy loss、其梯度与 GRPO advantage 在真实 910B3 上与原生
  `torch_npu` 逐位一致，接受；**性能（五条慢 `1.25x-1.75x`）与 NPU 端到端 PPO 仍开放**
- 日期：2026-09-02
- 基线提交：Jittor `97dc6ce9`（行为来自 `3758c4ab`）；verl 源码
  `3d66a3d7ca1cf783df949816ec6862d5a7af9406`
- 验证范围：一张 Ascend 910B3、CANN 9.0.0；原生参考进程加载 `torch_npu`，Jittor
  进程不加载，两端串行跑在同一张卡上；直接加载该 verl 提交的
  `verl/trainer/ppo/core_algos.py`、`verl/utils/torch_functional.py` 与 groupwise 实现，
  只为未参与计算的 worker config 与可选依赖提供最小外壳。未覆盖：verl 完整安装、Ray
  worker、FSDP2、权重传输、rollout、1-step PPO（这些此前只有 CPU/CUDA 证据）
- 维护者：Torch 兼容层与 verl adapter 维护者
- 复查条件：verl 的 policy-loss 公式、Jittor clamp/autograd 语义、CANN 或 NPU adapter
  变化

## 结论

覆盖 vanilla PPO、GSPO、SAPO、GPG、geometric-mean 与 CISPO 六类 policy loss，各自
对 `log_prob` 的完整梯度，GRPO outcome advantage 与 returns，以及 clamp 在标量/张量
边界、精确边界与 clip 后 minimum 上的反向。固定 `4 x 6` 输入同时含正负 advantage、
ragged mask、上下 clip 与精确 clip 边界：

| 路径 | loss 最大绝对误差 | 梯度/输出最大绝对误差 |
| --- | ---: | ---: |
| vanilla PPO、GSPO、SAPO、GPG、geometric-mean、CISPO | `0` | `0` |
| GRPO advantage/returns | — | `0` |

Jittor 结果全部驻留设备，捕获窗口内 `fallback_count=0`、`cpu_compile_count=0`。

## 先复现、再修的缺陷：clamp 精确边界的梯度

修复前 `clamp(x, -0.15, 0.25)` 在两个精确边界处的输入梯度为 0：

```text
input:       [-0.16, -0.15, -0.14, 0.24, 0.25, 0.26]
torch_npu:   [ 0.00,  1.00,  1.00, 1.00, 1.00, 0.00]
Jittor NPU:  [ 0.00,  0.00,  1.00, 1.00, 0.00, 0.00]
```

geometric-mean loss 有一个 `negative_approx_kl` 恰好落在 clip 上界，于是该梯度在
Jittor 为 0、原生为 `0.0051804874`。`fdae6a0f` 把 clamp 表达成包含边界的三元选择并
保留 NaN：`x == min/max` 时梯度为 1；反向的标量边界按 PyTorch 返回全 `max`；张量边界
的 input/min/max 梯度分别路由；FP16/BF16 的 Python 标量边界保持同 dtype；整数张量配
浮点边界提升到 float32。`3758c4ab` 让 ACL float32、有序双标量边界的前向调用
`aclnnClampTensor`，自定义反向仍保持「边界梯度 1、NaN 梯度 0」，没有把 CANN 前向直接
当成完整语义。

## 性能（仍开放）

计时含上游函数内部 metrics 的 `.item()`、loss 前向、对 `log_prob` 的反向与设备同步；
输入计时前已驻留 NPU，数值门禁在计时前单独通过。

| 形状 | 路径 | torch_npu | Jittor | Jittor/原生 |
| --- | --- | ---: | ---: | ---: |
| `512 x 512` | vanilla | 2.679 ms | 3.692 ms | 1.378x |
| `512 x 512` | GSPO | 2.707 ms | 4.728 ms | 1.746x |
| `512 x 512` | SAPO | 2.511 ms | 3.403 ms | 1.356x |
| `512 x 512` | GPG | 0.846 ms | 0.655 ms | 0.774x |
| `512 x 512` | geometric-mean | 2.397 ms | 3.823 ms | 1.595x |
| `512 x 512` | CISPO | 1.698 ms | 2.561 ms | 1.508x |

`64 x 120` 的比值与之几乎相同（`0.776x-1.653x`）。原生 CANN Clamp 已把五条慢路径从
约 `2.3x-2.6x` 收到上表。

**归因：差距在 metrics 触发的惰性图分段，不在计算本身。** vanilla 的 device profile
有三张大融合图，分别由 metrics `.item()`、loss 与反向分阶段触发；GPG 没有内联 metrics，
只有 8 次 launch，也已不慢于原生。只把 vanilla 返回的三项 metrics 换成空字典（loss、
梯度、输入与同步协议不变），21 个样本中位数为原生 `2.619 ms`、Jittor `2.589 ms`
（`0.989x`）：三项 metrics 在原生侧只多 `0.060 ms`，在 Jittor 侧多 `1.103 ms`。

试过且已撤回（数值都一致，但没有达到目标）：对四个共享中间量 `stop_fuse`
（`1.328x`）；ACL `.item()` 改成全图 `sync_all`（退化且样本双峰）；metrics 与 loss
一次性取回（退化）；先堆叠逐 token metrics 再共同归约（最好，仍 `1.220x`）；分组 ACL
CodeOp masked mean（更慢）。下一步只能是让 policy-loss 前向、metrics 与反向共享一次
专用融合计算，或在 verl NPU adapter 里统一延迟提取 metrics；不能把全局 `.item()` 改成
`sync_all`。

## 复现

比较器探针（`core_algos_parity.py`）未进仓库。协议：先 `source "$CANN_SET_ENV"` 并只
暴露 `ASCEND_RT_VISIBLE_DEVICES=<allocated-device>`；在原生 `torch_npu` 环境与
`JITTOR_TORCH_SHIM=1` 的 Jittor 环境各起一个进程，加载上述 verl 提交的源码，喂同一组
固定输入，比较 loss、梯度与 GRPO 输出的最大绝对误差；计时先暖身，再取进程内样本
中位数，并记录回退计数与 CPU 编译计数。

主仓侧的维护回归：clamp 的 CPU 边界/NaN 用例与 Torch 兼容用例
（`compat/tests/torch/test_torch_compat_ops.py`），以及 NPU 上的
`tests/backends/acl/test_acl.py`（含 CANN Clamp 前反向）。

## 仍开放

- 让全部六条协议不慢于原生（见上面的归因与下一步）；
- 恢复或重建可维护的 verl NPU adapter，跑通真实 worker import、TensorDict batch、
  actor/critic 前反向、optimizer、权重传输与 1-step PPO；
- NPU 上的 FSDP2/HCCL、Ray 多进程、vLLM rollout 与 Qwen3 规模需分别验收，在此之前
  不能把 CPU/CUDA 上完整 PPO 的结论外推到 NPU。

`e3c369acb` 上核对：clamp 的修复与 CANN Clamp runner
（`backends/acl/kernels/native/clamp_op_acl.cc`）仍在，之后没有新的 NPU 复测；问题
总账中没有单独条目，本报告是该性能缺口与 NPU PPO 门禁的唯一记录。
