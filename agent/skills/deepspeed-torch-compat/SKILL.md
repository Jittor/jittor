---
name: deepspeed-torch-compat
description: DeepSpeed 0.17.6 显式 adapter 的导入、构造和独立 PyTorch L0 对拍入口。
---

# DeepSpeed 接入入口

本阶段提交显式 `jittor_adapters.deepspeed` 接入和可复现的 L0 门禁。
`activate(device="cpu")` 用于导入、配置及模型构造；CPU Engine 明确拒绝。
NPU 路径须在真实 HCCL 环境与同一提交上单独验证，未验证前不得声明支持等级。
ZeRO 训练、权重恢复、真实模型生成、速度和显存均不属于本阶段验收结论。

## 运行顺序

1. 阅读 `adapters/jittor_adapters/deepspeed/README.md` 的固定版本、源码哈希和拒绝边界。
2. 在仓外安装 DeepSpeed 0.17.6 和 `requirements/deepspeed-l0.txt`，将真 PyTorch 与 Jittor 放在独立解释器。先断言 oracle `torch` 没有 `_torch_compat_install_context`。
3. 用独立缓存运行 `adapters/tests/test_deepspeed_offline.py -v`，检查生命周期、源码和失效边界。
4. 用 `REAL_TORCH_PYTHON=<oracle>`、`JITTOR_DEEPSPEED_WHEEL=<固定 wheel>` 运行 `python -m nox -s deepspeed_l0`。检查 `comparison.json` 的 `status=passed`，并记录设备、依赖、提交 SHA 和原始日志路径。
5. 对 NPU 另行申请真实设备并重复 L0，确认原生 HCCL、无 host fallback。不能拿 CPU 或 mock 测试替代。

`scripts/l0_probe.py` 记录模型参数、buffer、配置及错误契约，并拒绝把 shim 当作真 PyTorch oracle。所有实验产物留在仓外；跳过的测试不得计为通过。
