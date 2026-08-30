# MMCV/MMEngine Ascend 数值与梯度对拍

- Status: correctness accepted on one real Ascend 910B3; performance not accepted
- Last reviewed: 2026-08-30
- Baseline: `52b7fc68` plus the changes described in this report
- Owner: Torch compatibility and ACL backend maintainers
- Review when: MMCV/MMEngine, ACL convolution/normalization/activation, the
  ecosystem runner, or the NPU performance protocol changes

## 结论

`mmcv-lite 2.1.0` 的 `mmcv.cnn.ConvModule` 和 `MMEngine 0.10.7` 的
`BaseModule` 现在使用同一组权重分别在 Jittor ACL 与独立 `torch_npu` 进程中执行。
两个用例均通过前向、输入梯度和全部参数梯度对拍；Jittor 捕获窗口内没有
`fallback cpu` 或 `compile cpu`。本结论只覆盖 OpenMMLab 的纯 Python 模型层，
不包含依赖 PyTorch C++ ABI 的 `mmcv.ops`。

维护模型的绝对耗时很小，本轮 `min-of-3` 快照仍显示 Jittor 较慢，因此不接受
NPU 性能结论：

| Case | torch_npu | Jittor ACL | Ratio |
| --- | ---: | ---: | ---: |
| MMCV ConvModule | 1.8 ms | 3.6 ms | 1.97x |
| MMEngine BaseModule | 1.5 ms | 1.8 ms | 1.17x |

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

## Verification

The first cold-cache package/import gate completed serially:

```text
tests/compat/torch/test_mmcv_compat.py (before construction regression)
3 passed in 1600.78s
```

Focused real-NPU regression after the fixes:

```text
ACL ReLU/LeakyReLU inplace forward/backward: 1 passed
MMCV import and ConvModule construction: 4 passed
MMEngine Jittor ACL vs torch_npu parity: 1 passed
MMCV Jittor ACL vs torch_npu parity: 1 passed
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

## Boundaries

- `mmcv.ops` remains outside the Python Torch-shim contract.
- The tiny maintenance cases prove correctness and expose latency, but do not
  replace a realistic OpenMMLab training benchmark.
- The observed NPU ratios do not meet the broader no-slower target. A stable,
  realistic shape protocol and ACL kernel profile are required before accepting
  or optimizing that target.
- Diffusers now has a separate real-NPU correctness result; see the
  [Diffusers Ascend report](2026-08-30-diffusers-ascend-parity-performance.md).
  ms-swift, verl, and vLLM NPU coverage remains separate work.
