# 可选依赖 fail-closed CUDA 门禁

- Status: Selected optional packages accepted on real CUDA
- Last reviewed: 2026-08-24
- Commits: `566eae8e`, `2cf096d5`, `c2e340f8`, `90e00edd`
- Owner: Torch compatibility and test-infrastructure maintainers
- Review when: optional package versions, Torch shim identity, or nox hardware
  environment contracts change

## 结论

项目现在提供 `python -m nox -s optional`，在预配置 CUDA 环境中 fail-closed
验证 TorchMetrics、mmcv-lite、MMEngine、PEFT、Safetensors、TensorDict 和部署的
FlashAttention adapter。session 在 pytest 前检查全部包，显式启用 Jittor Torch
shim 与离线模式，并在真实 CUDA 上运行五个兼容模块；缺少依赖时不再以 skip
充当通过。

PEFT 测试原先还要求 shim 的 `torch.__name__ == "torch"`，但部署契约是
`torch is jittor`，模块名保留 `jittor`，因此已安装 PEFT 时三个测试仍全部误跳过。
判断现在使用对象身份；`JITTOR_REQUIRE_OPTIONAL_DEPS=1` 下任何导入异常直接失败。

FlashAttention math fallback 原先接收 `dropout_p` 却没有传给 canonical SDPA，
会静默返回未 dropout 的结果。dense、packed 和 varlen fallback 现在都传递该参数，
并由真实部署 adapter 的 `dropout_p=1` 零输出回归锁定。

## 环境

- Python 3.11.15，Jittor 1.3.11.0，真实 RTX 4090，CUDA toolkit 12.2.140。
- TorchMetrics 1.7.4、MMEngine 0.10.7、PEFT 0.17.1、Safetensors 0.8.0；
  TensorDict 0.10.0、FlashAttention adapter 2.7.4.post1；mmcv-lite 可从当前
  预配置环境导入。
- `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、
  `JITTOR_TORCH_SHIM=1`、`use_cuda=1`、`use_parallel_op_compiler=0`。

## 验证

- 修复前 `tests/compat/torch/test_peft.py`：`3 skipped`，实际 PEFT 导入探针成功。
- 修复后 PEFT：`3 passed in 126.50s`，覆盖 LoRA 冻结与梯度、200 步拟合、
  Safetensors adapter 保存/加载。
- TorchMetrics：`2 passed in 655.93s`，覆盖分类、回归、聚合和 required ops。
- mmcv-lite/MMEngine：`3 passed in 16.62s`，含 CUDA typed tensor 真实设备执行。
- 三模块同一 shim 会话：`8 passed, 1 warning in 17.83s`（warm cache）。
- TensorDict 与 FlashAttention 新增真实行为模块：`5 passed in 11.31s`，覆盖 CUDA
  构造/更新/index/lazy stack，以及 dense/packed/varlen attention、梯度和 dropout。
- 五模块加既有 loader/stub 契约同一 shim 会话：`16 passed, 1 warning in 28.37s`。
- `nox -s optional -- tests/compat/torch/test_mmcv_compat.py`：依赖预检通过，
  `3 passed in 550.05s`，session 成功完成冷 cache 编排。
- 布局检查通过；`tests/structure`：`218 passed`；`noxfile.py` Ruff 检查通过。
- 相对 `HEAD` 构建前后两个 797-member wheel，差分只有已批准的 FlashAttention
  stub 和派生 `RECORD` 两项内容变化；无成员新增或删除，wheel 内容审计通过。

首次 cold-cache 组合运行在 20 分钟保护下因主机另一个长期满核进程而超时，没有失败
traceback；相同隔离 cache 补齐编译后，上述模块和组合门禁全部通过。原始 cache 与运行
状态均在 `$JITTOR_LAB_ROOT/_state/`，未进入仓库。

## 边界

FlashAttention 结果声明的是部署 adapter 的 CUDA math fallback；本机没有配置
`JITTOR_FLASH_ATTN_JITTOR_SRC`，因此不声明 native fused backend 已验证。完整
ecosystem forward/backward 与性能仍由独立 same-version harness 维护。NPU/ROCm
也未因本次 CUDA 结果获得任何通过结论。
