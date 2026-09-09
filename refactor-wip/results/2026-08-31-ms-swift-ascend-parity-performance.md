# ms-swift LoRA Ascend 数值、梯度与性能

- Status: maintained-case correctness and performance accepted on one real
  Ascend 910B3
- Last reviewed: 2026-08-31
- Baseline: `9d3e0452` plus the changes described in this report
- Owner: Torch compatibility and ACL backend maintainers
- Review when: ms-swift/Transformers attention changes, CANN FlashAttention
  changes, or the ecosystem timing protocol changes

## 结论

`ms-swift 4.5.2` 的 LoRA Llama 维护用例现在使用同一组权重分别在 Jittor ACL
与独立 `torch_npu` 进程中执行，并对拍前向和全部可训练参数梯度。Jittor
结果报告真实 NPU、`has_acl/use_acl/use_cuda=true`，捕获窗口内没有
`fallback cpu` 或 `compile cpu`。

初始 Jittor 训练步比 `torch_npu` 慢 `18.2%`。根因不在 ms-swift LoRA tuner：
Transformers 的 Llama 选择了 SDPA，但 ACL adapter 拒绝所有 causal/masked
training 请求，Jittor 因而退回普通 matmul/softmax attention，而 `torch_npu`
使用 CANN fused attention。验证 CANN forward/backward 后，float32、相同 query/KV
head 数、`dropout=0` 的 causal 和 additive-mask training 接入
`FlashAttentionScoreV2`；causal 请求同时使用 CANN sparse mode 2 的固定
`2048 x 2048` 压缩 mask。

最终 `50` 次训练步取最小值：

| Runtime | Step | Ratio |
| --- | ---: | ---: |
| native `torch_npu` | 14.089 ms | 1.000x |
| Jittor ACL, dense causal mask | 14.708 ms | 1.044x |
| Jittor ACL, compressed causal mask | 13.654 ms | 0.969x |

优化前同一用例的 `20` 次结果为 `torch_npu 14.002 ms`、Jittor ACL
`16.556 ms`，比例 `1.182x`。去掉 Swift tuner 的同配置 Transformers Llama
基线为 `11.441 ms` 对 `14.649 ms`，比例 `1.280x`，进一步确认共享 attention
路径才是主要瓶颈。

最终 30 个对拍张量中，前向归一化误差为 `1.980e-7`、最大绝对误差
`1.192e-7`；最差梯度归一化误差为 `4.268e-7`、最大绝对误差
`1.192e-7`。

## 环境与协议

- Device: one Ascend 910B3
- CANN: 9.0.0
- Jittor: Python 3.9.25, Jittor 1.3.11.0
- Oracle: Python 3.10.20, PyTorch 2.10.0, torch_npu 2.10.0
- Downstream: Transformers 4.57.6, PEFT 0.17.1, ms-swift 4.5.2
- Model: two Llama layers, hidden size 64, two attention/KV heads, sequence
  length 8, LoRA rank 4 on `q_proj` and `v_proj`
- Timing: four preallocated device-resident input slots, four warm-up steps,
  one complete forward/loss/backward synchronization per measured step
- JIT: first compilation serialized; unittest and benchmark use distinct
  `cache_name` values

Python 3.9 and 3.10 cannot safely share a dependency site containing
minor-version-specific extension modules. The ecosystem harness therefore
accepts `JITTOR_ECOSYSTEM_REFERENCE_PACKAGE_SITE` for an independent oracle
site and asserts matching dependency versions while allowing origins to differ.
Both isolated sites contain the exact versions listed above.

Runtime state, package sites, NPZ snapshots and profiles are unversioned under
`$JITTOR_LAB_ROOT/_state/npu-ms-swift/`.

## Maintained command

After sourcing CANN, the NPU parity case is selected with:

```bash
REAL_TORCH_PYTHON=/path/to/torch-npu/python \
JITTOR_TORCH_SHIM=1 \
JITTOR_ECOSYSTEM_PACKAGE_SITE=/path/to/python39/site-packages \
JITTOR_ECOSYSTEM_REFERENCE_PACKAGE_SITE=/path/to/python310/site-packages \
python -m pytest -q \
  tests/compat/torch/test_ecosystem_parity.py::EcosystemParityNPU::test_ms_swift_lora_llama
```

The focused ACL regression separately forces causal and additive-mask
forward/backward through the native adapter, compares against the CPU canonical
implementation, and rejects CPU fallback/compilation.

Verification results:

```text
focused causal/additive ACL regression: 1 passed
complete ACL Torch-compat plus harness unit tests: 23 passed, 1 skipped
ms-swift independent-site NPU parity: 1 passed in 132.08s
```

The maintained parity run compared 29 gradients, reported zero CPU paths and
printed `0.96x` from its short three-repeat sample. The longer 50-repeat result
above is the performance conclusion. The only combined-test skip requires a
real CUDA backend and is inapplicable to the ACL environment.

## Boundaries

- This proves a small deterministic ms-swift LoRA training graph, not a full
  checkpoint fine-tuning run.
- The fused training subset remains float32, equal query/KV head count and
  `dropout=0`, with a non-trainable attention mask. BF16 training, GQA training,
  trainable masks and attention dropout use the existing non-fused behavior and
  require separate acceptance.
- Qwen3 inference and full-model float32 forward/backward have separate real-NPU
  reports; optimizer update and distributed ms-swift training are not claimed
  here.
