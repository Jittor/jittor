# SGD CUDA 低精度更新边界

- 状态：已验证有界的 CPU/CUDA 单测契约；CPU 低精度一致性仍未关闭。
- 复查日期：2026-10-16。
- 维护者：原生优化器与 Torch 兼容性维护者。
- 实现交付：`c5ed1212851dda005a7c64f1fc512974e1ed1e77`。
- 原始 CPU 单测源码：`5740cb4c761950fd0d1458ad79810a5b2bcf3228`。
- 原始 CUDA 单测源码：`2bc8ee53594fee93aa4aea935266a8a162ee4ea9`。
- 复查条件：SGD 算术、派发、动量状态、FSDP 更新或精度策略发生变化。

Torch CUDA FP16/BF16 SGD 使用 FP32 中间值执行缩放加法，并只做一次目标类型转换。此前的
原生表达式在相加之前就对缩放乘积做了舍入，从而改变小幅参数更新。Torch 前端现在请求
一个显式的原生 `torch_math` 策略。原生 SGD 默认 `False`；CPU、FP32 和既有的融合内核
选择保持其原有算术。

既有的共享 SGD 所有者实现权重衰减、参数更新和 Nesterov 缩放加法。动量乘法保留其单独的
类型转换，与实际 `mul_` 后接 `add_` 的边界一致。FSDP2 将同样的策略转发给该所有者。没有
adapter 实现优化器数学，也未新增主机更新或后端回退。

这三个生产文件与原始被测源码逐字节一致。加上六个既有的 `TestSGD` 用例，CPU 单测通过
**11 / 7 个 CUDA-only skip**；CUDA 配置通过 **17 / 1 个 CPU-only skip**。仅新增的精度
文件就贡献 CPU **5 / 7 skip** 和 CUDA **11 / 1 skip**；后者的张量位置被显式检查。
基准数据覆盖 FP16/BF16/FP32、普通 SGD、带权重衰减与 dampening 的动量，以及三步的
Nesterov，包括每个动量缓冲区。CPU 低精度单测断言既有的原生表达式得以保留；它们不声明
与 Torch CPU 的低精度一致性。

独立的只读验证重新哈希了全部 58 个原始证据文件，并将 90 个已发布参数/动量数组与原始
真实 Torch `2.13.0+cu130` 的 NPZ 数据比较。新的真实 CPU 随后 CUDA 重放又逐位比较了
144 个完整数组（包括梯度和动量）与那些原始记录。实际 oracle `_C` SHA256 为
`819a187ea3b1f4d781200fe73fba3264689835d2285f7e6d10b986257a802dce`。
这些是正确性检查，不含性能声明。

```bash
JITTOR_TORCH_SHIM=1 PYTHONPATH=python \
python -m pytest -q compat/tests/torch/test_torch_compat_sgd_precision.py \
  compat/tests/torch/test_torch_compat_optim.py::TestSGD
```

在该 Torch 版本中，CPU 的向量块与标量尾部具有不同的低精度舍入。该差异仍未解决。新的
FSDP2 单测仅证明单次更新的策略转发；它不是分布式动量，也不是全新检查点/续训证据。通用
的 saved-value/version 检查和可微 `copy_` 存储身份也仍是未关闭的独立契约。本结果不授予
完整的优化器 ABI、其他加速器或整模型支持。

原始证据未纳入版本控制，位于 `$JITTOR_LAB_ROOT/_state/omni-training-profile/20261016/`；
独立复核位于 `$JITTOR_LAB_ROOT/_state/sgd-policy-independent-review/20261016/`。
后者的 21 文件封存 SHA256 为
`275f438c901ed80573bc9363f794779421ccf33c5c2994660b32aa003dd9f934`。
主整合命令、精确源码映射和后续验证位于 `$JITTOR_LAB_ROOT/_state/sgd-main-integration/20261016/`。

在 `46391b45a61a298a246630b6e9f1e896cec9208a` 加上精确的五文件交付之上，整合通过 CPU
相邻契约 **50 / 40 个 CUDA skip**、合并 CUDA 配置 **95 / 1 个 CPU skip**、原生 CUDA
优化器契约 **46 / 0 skip**、核心 **322 / 42 skip / 1 个既有 xfail**，以及结构
**1628 / 6 skip**。两个 CUDA 命令守卫都记录前后回退为零。这些计数描述各自的命令，包括
CUDA 可见环境中的 CPU 对照。一个仅含遗留项的补充命令通过了六项断言，但因声明无加速器
用例而退出 1；该失败门禁保留在原始证据中。最终合并命令包含显式的设备契约，并通过未改动
的加速器门禁。
