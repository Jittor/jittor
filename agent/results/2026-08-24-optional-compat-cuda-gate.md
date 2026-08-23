# 可选依赖 fail-closed CUDA 门禁

- Status: Selected optional packages accepted on real CUDA
- Last reviewed: 2026-08-24
- Commits: `566eae8e`, `2cf096d5`
- Owner: Torch compatibility and test-infrastructure maintainers
- Review when: optional package versions, Torch shim identity, or nox hardware
  environment contracts change

## 结论

项目现在提供 `python -m nox -s optional`，在预配置 CUDA 环境中 fail-closed
验证 TorchMetrics、mmcv-lite、MMEngine、PEFT 和 Safetensors。session 在 pytest 前
检查全部包，显式启用 Jittor Torch shim 与离线模式，并在真实 CUDA 上运行三个兼容
模块；缺少依赖时不再以 skip 充当通过。

PEFT 测试原先还要求 shim 的 `torch.__name__ == "torch"`，但部署契约是
`torch is jittor`，模块名保留 `jittor`，因此已安装 PEFT 时三个测试仍全部误跳过。
判断现在使用对象身份；`JITTOR_REQUIRE_OPTIONAL_DEPS=1` 下任何导入异常直接失败。

## 环境

- Python 3.11.15，Jittor 1.3.11.0，真实 RTX 4090，CUDA toolkit 12.2.140。
- TorchMetrics 1.7.4、MMEngine 0.10.7、PEFT 0.17.1、Safetensors 0.8.0；
  mmcv-lite 可从当前预配置环境导入。
- `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、
  `JITTOR_TORCH_SHIM=1`、`use_cuda=1`、`use_parallel_op_compiler=0`。

## 验证

- 修复前 `tests/compat/torch/test_peft.py`：`3 skipped`，实际 PEFT 导入探针成功。
- 修复后 PEFT：`3 passed in 126.50s`，覆盖 LoRA 冻结与梯度、200 步拟合、
  Safetensors adapter 保存/加载。
- TorchMetrics：`2 passed in 655.93s`，覆盖分类、回归、聚合和 required ops。
- mmcv-lite/MMEngine：`3 passed in 16.62s`，含 CUDA typed tensor 真实设备执行。
- 三模块同一 shim 会话：`8 passed, 1 warning in 17.83s`（warm cache）。
- `nox -s optional -- tests/compat/torch/test_mmcv_compat.py`：依赖预检通过，
  `3 passed in 550.05s`，session 成功完成冷 cache 编排。
- 布局检查通过；`tests/structure`：`218 passed`；`noxfile.py` Ruff 检查通过。

首次 cold-cache 组合运行在 20 分钟保护下因主机另一个长期满核进程而超时，没有失败
traceback；相同隔离 cache 补齐编译后，上述模块和组合门禁全部通过。原始 cache 与运行
状态均在 `$JITTOR_LAB_ROOT/_state/`，未进入仓库。

## 边界

本门禁只声明上述五个有真实兼容用例的包。当前环境虽可发现 TensorDict 和
FlashAttention adapter，但尚无对应的真实 package 行为模块；完整 ecosystem
forward/backward 与性能仍由独立 same-version harness 维护。NPU/ROCm 也未因本次
CUDA 结果获得任何通过结论。
