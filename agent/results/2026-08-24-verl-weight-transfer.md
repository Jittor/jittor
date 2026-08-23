# verl Jittor 权重传输真实 CUDA 门禁

- Status: Adapter transport gate accepted on real CUDA
- Last reviewed: 2026-08-24
- Jittor baseline: `bc2229f9`
- verl source: `3d66a3d7ca1cf783df949816ec6862d5a7af9406` plus existing adapter edits
- Owner: verl/vLLM external-adapter maintainers
- Review when: verl bucket protocol, adapter transport, or rollout weight loader changes

## 结论

外部 `verl_jittor` adapter 的 numpy-over-ZMQ weight transport 已在真实 RTX 4090
上完成 sender/receiver 往返，不再只有 import 或函数替换证据。lazy import hook 对
canonical `verl.workers.rollout.vllm_rollout.bucketed_weight_transfer` 模块生效，实际
`BucketedWeightSender.async_send_weights` 与
`BucketedWeightReceiver.receive_weights` 均运行 adapter 实现。

探针发送 fp32、fp16、bf16 三个 CUDA 权重。sender 按 adapter 契约转为连续 fp32
host array，经 ZMQ REQ/REP 逐项传输；receiver 在 CUDA 上重建 Jittor tensor。名称、
shape 和逐元素值全部一致，`rotary_emb.inv_freq` 按协议被过滤。

## 验证

- 独立 cache 首次完成 Jittor core、CUDA extern 与 MKL 初始化。
- 一次性最小复现输出：`VERL_WEIGHT_TRANSFER_OK 3 True`。
- 可复用 harness：
  `$JITTOR_LAB_ROOT/verl_jittor/scripts/weight_transfer_smoke.py`。
- harness 热 cache：`verl weight transfer smoke OK weights=3`，约 8 秒。
- `$JITTOR_LAB_ROOT/verl_jittor/scripts/run_all.sh` 已把该脚本加入默认 CUDA gate，
  `--cpu-only` 时显式跳过；shell syntax 与 Python compile 检查通过。

## 隔离方法

verl 的 `vllm_rollout/__init__.py` 会同时加载完整 vLLM server 和无条件 NPU utility，
把 `vllm_ascend` 等与 transport 无关的可选依赖引入最小测试。harness 注册同名父
package 的真实 `__path__`，但不执行其 `__init__`，随后按 canonical module name
导入真实 `bucketed_weight_transfer.py`。因此 adapter 的 meta-path lazy hook 仍被
验证，同时没有用假的 sender/receiver 实现替代被测代码。

## 边界

该门禁证明 transport payload 与 handshake 正确，但尚未证明真实 vLLM EngineCore
的 `on_bucket_received/load_weights` 已应用权重，也没有覆盖 Ray 跨进程、FSDP actor
导出或完整 PPO step。下一阶段仍需在真实 rollout engine 中比较更新前后参数摘要和
生成/logprob 变化，并完成训练、rollout、reward、actor/critic update 闭环。
