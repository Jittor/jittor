# MMCV/MMEngine Ascend 数值与梯度对拍

- Status: correctness and maintained-case performance accepted on one real
  Ascend 910B3
- Last reviewed: 2026-08-31
- Baseline: `cb3453a2` plus the changes described in this report
- Owner: Torch compatibility and ACL backend maintainers
- Review when: MMCV/MMEngine, ACL convolution/normalization/activation, the
  ecosystem runner, or the NPU performance protocol changes

## 结论

`mmcv-lite 2.1.0` 的 `mmcv.cnn.ConvModule` 和 `MMEngine 0.10.7` 的
`BaseModule` 现在使用同一组权重分别在 Jittor ACL 与独立 `torch_npu` 进程中执行。
两个用例均通过前向、输入梯度和全部参数梯度对拍；Jittor 捕获窗口内没有
`fallback cpu` 或 `compile cpu`。本结论只覆盖 OpenMMLab 的纯 Python 模型层，
不包含依赖 PyTorch C++ ABI 的 `mmcv.ops`。

优化后的 `20` 次训练步取最小值显示两个维护用例均不慢于原生 `torch_npu`：

| Case | torch_npu | Jittor ACL | Ratio |
| --- | ---: | ---: | ---: |
| MMCV ConvModule | 1.697 ms | 1.574 ms | 0.927x |
| MMEngine BaseModule | 1.442 ms | 1.147 ms | 0.796x |

MMCV 前向归一化误差为 `3.842e-4`，最差参数梯度为 GroupNorm weight
的 `3.197e-4`；MMEngine 前向和最差梯度分别为 `6.525e-8` 和
`7.411e-8`。两边各使用同一组权重、输入种子和损失权重。

## 环境与隔离

- Device: one Ascend 910B3
- CANN: 9.0.0
- Jittor interpreter: Python 3.9.25, Jittor 1.3.11.0
- Oracle interpreter: Python 3.10.20, PyTorch 2.10.0, torch_npu 2.10.0
- Downstream: mmcv-lite 2.1.0, MMEngine 0.10.7
- JIT: serial first compilation with an isolated `JITTOR_HOME` and `cache_name`
- Runtime state and packages: unversioned under
  `$JITTOR_LAB_ROOT/_state/npu-ecosystem/20260830/`

The downstream site contains pure Python packages plus an ABI3 OpenCV wheel.
Because the two interpreters use different CPython minor versions, the run sets
`JITTOR_ECOSYSTEM_PACKAGE_SITE_CROSS_ABI=1` explicitly. Both result reports
assert the exact same dependency versions and origins; this override must not be
used for an arbitrary site containing CPython-minor-specific extensions.

## Findings and fixes

The initial import-only compatibility gate passed, but constructing the real
MMCV model exposed two NPU-only wrapper defects:

1. ACL wraps `nn.Conv2d` and `nn.Conv` into distinct classes. The Torch
   compatibility installer added `transposed` and `output_padding` only to
   `nn.Conv`, so MMCV failed while reading `self.conv.transposed`. The installer
   now handles `Conv2d` and `ConvTranspose2d` explicitly.
2. ACL's replacement `ReLU` and `LeakyReLU` constructors did not retain Torch's
   `inplace` argument. The ACL module and functional forms now accept the
   argument, expose the module attribute, and retain Jittor's graph-equivalent
   execution semantics.

The ecosystem harness also gained an explicit `npu` device. It now distinguishes
ACL from CUDA, imports and checks `torch_npu`, moves real-Torch tensors and models
to NPU, synchronizes `torch.npu`, reports Jittor ACL flags, and fails on captured
CPU fallback or CPU compilation. Its shim preflight follows the same
Jittor-first import order as the subprocess runner.

本轮性能复查从优化前的 `10` 次取最小值开始：MMCV 为 `1.529x`，MMEngine
为 `1.100x`。分析确认小模型主要损耗不是单个大核，而是 Python `jt.Function`
前反向边界、冗余临时变量和重复同步。修复包括：

1. LayerNorm、推理态 BatchNorm、无 bias Conv2d、ReLU 和 LeakyReLU 改为
   联合前反向 CodeOp，保留原生 ACL kernel 和自动微分输入顺序。
2. BatchNorm 激活张量使用 NCHW descriptor，参数、统计量和保存值使用 ND，
   避免 `aclnnBatchNorm` 在错误 storage format 上拒绝执行。
3. 无 bias convolution backward 只申请 input/weight 两个梯度，并正确传递
   bias shape 与释放 ACL 属性对象；functional `conv2d` 同步接入 ACL wrapper。
4. 计时器用一次 targeted `jt.sync(..., device_sync=True)` 强制完整前反向，
   不再紧接一次遍历全部活跃 Var 的 `sync_all()`。PyTorch 仍执行一次设备同步。

## Verification

The first cold-cache package/import gate completed serially:

```text
tests/compat/torch/test_mmcv_compat.py (before construction regression)
3 passed in 1600.78s
```

Focused real-NPU regression after the fixes:

```text
ACL ReLU/LeakyReLU inplace forward/backward: 1 passed
ACL BatchNorm eval forward/backward: 1 passed
ACL LayerNorm forward/backward: 1 passed
ACL functional no-bias Conv2d forward/backward: 1 passed
MMCV import and ConvModule construction: 4 passed
MMEngine Jittor ACL vs torch_npu parity: 1 passed
MMCV Jittor ACL vs torch_npu parity: 1 passed
```

修改后的完整 ACL Torch-compat 文件结果为：

```text
13 passed in 302.22s
```

The final combined maintained command selected the ACL Torch-compat file, MMCV
compatibility, ecosystem device selection, and the NPU parity class:

```text
18 passed, 1 skipped in 238.40s
```

The only skip is the device-selection regression that requires a real CUDA
backend. Every NPU parity result asserts `device=npu`, Jittor
`has_acl/use_acl/use_cuda=true`, zero CPU fallback, zero CPU compilation, equal
downstream dependency origins, and equal TF32-disabled policy.

本轮改动的 Python 文件同时通过 Python 3.12.14 `py_compile`。最终性能采样的
两个 Jittor 结果均报告真实 NPU、ACL 三个状态位为真、`fallback_count=0`、
`cpu_compile_count=0`；全部原始 NPZ、profile 和运行状态保存在未版本化的
`$JITTOR_LAB_ROOT/_state/npu-ecosystem/20260830/`。

## Boundaries

- `mmcv.ops` remains outside the Python Torch-shim contract.
- The tiny maintenance cases prove correctness and meet the no-slower target for
  these exact shapes, but do not replace a realistic OpenMMLab training benchmark.
- Training-mode BatchNorm is not routed through this inference-only fast path;
  it retains the existing stable Jittor implementation.
- Diffusers now has a separate real-NPU correctness result; see the
  [Diffusers Ascend report](2026-08-30-diffusers-ascend-parity-performance.md).
  ms-swift, verl, and vLLM NPU coverage remains separate work.
