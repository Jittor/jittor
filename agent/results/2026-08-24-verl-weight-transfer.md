# verl Jittor 权重传输真实 CUDA 门禁

- Status: Adapter transport gate accepted on real CUDA
- Last reviewed: 2026-08-24
- Jittor baseline: `867664a5`
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

同一 transport 随后连接到真实 vLLM V1 UniProc EngineCore 内的 Qwen3-0.6B。
receiver callback 调用模型原生 `load_weights`，将
`model.layers.0.input_layernorm.weight` 增加 `0.125`；EngineCore 模型参数读回
`delta=0.125000`，随后通过同一 loader 恢复原值并逐元素验证。

## 验证

- 独立 cache 首次完成 Jittor core、CUDA extern 与 MKL 初始化。
- 一次性最小复现输出：`VERL_WEIGHT_TRANSFER_OK 3 True`。
- 可复用 harness：
  `$JITTOR_LAB_ROOT/verl_jittor/scripts/weight_transfer_smoke.py`。
- harness 热 cache：`verl weight transfer smoke OK weights=3`，约 8 秒。
- `$JITTOR_LAB_ROOT/verl_jittor/scripts/run_all.sh` 已把该脚本加入默认 CUDA gate，
  `--cpu-only` 时显式跳过；shell syntax 与 Python compile 检查通过。
- 真实模型应用 harness：
  `$JITTOR_LAB_ROOT/verl_jittor/scripts/vllm_weight_apply_smoke.py`；设置
  `VERL_VLLM_MODEL` 时由 `run_all.sh` 追加执行。
- Qwen3-0.6B V1/UniProc：模型加载约 7 秒、engine warmup 约 2 秒，输出
  `verl vLLM weight apply smoke OK ... loaded=1 delta=0.125000`。

## 隔离方法

verl 的 `vllm_rollout/__init__.py` 会同时加载完整 vLLM server 和无条件 NPU utility，
把 `vllm_ascend` 等与 transport 无关的可选依赖引入最小测试。harness 注册同名父
package 的真实 `__path__`，但不执行其 `__init__`，随后按 canonical module name
导入真实 `bucketed_weight_transfer.py`。因此 adapter 的 meta-path lazy hook 仍被
验证，同时没有用假的 sender/receiver 实现替代被测代码。

## 边界

该门禁证明 transport payload、handshake 和 UniProc EngineCore 模型应用正确，但
没有覆盖 Ray 跨进程、FSDP actor 导出或完整 PPO step。下一阶段仍需比较权重更新前后
生成/logprob 变化，并完成训练、rollout、reward、actor/critic update 闭环。
