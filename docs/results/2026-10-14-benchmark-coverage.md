# Benchmark 覆盖与比较协议补齐

- 状态：已实现并完成下述验证；仍有未覆盖设备与场景，不代表全套性能验收。
- 日期：2026-10-14。
- 基线：`56b5271471ca9d872f287ea54f9d135875bafa59`，加本报告所在提交。
- 范围：ASV 卷积、Accelerate CPU 训练、真实模型比较协议、目录清单与 CI 选择器。
- 维护者：Jittor 维护者。
- 复查条件：ASV/nox 入口、精度配置、比较协议、后端执行或 CI 选择器发生变化。

## 改动与覆盖

新增 `convolution.Conv2dBenchmarks`，覆盖普通 3×3、步长为 2 的 1×1、depthwise
3×3 三种卷积，分别测前向与前向加反向。输入、权重、bias 的 VJP 与前向结果在
实际计时 shape 上先对照独立 NumPy float64 参考，再进行 FP32 计时。CUDA 路径
固定并恢复精度配置，检查实际张量位置以及零 fallback。

模型 harness 现在对齐 CPU 线程配置，记录实际设备、精度、依赖和 fallback。
条件缺失或不一致时保留原始时间，但不生成可比较的速度比。成功标记之后的崩溃与
超时也会使该次运行失败。`--list --json` 无须导入模型即可查看全部 14 项 workload。

`benchmarks/catalog.py` 汇总六类入口，区分声明能力与已有验证；本次发现 20 个 ASV
方法、14 个模型 workload、3 个 ACL 模型、7 个生态速度 case、4 个图构建 case 和
17 个 profiling 工具。这些是清单条目数，不是全组合的硬件验证数量。

CPU PR 基准加入卷积，定期基准再加入 reductions；CUDA 定期基准补入卷积、
reductions、transfer。新增可选 `nox -s benchmark_accelerate` 入口，比较同一 Jittor
训练在直接调用与 Accelerate 包装下的成本。修复 ASV 配置相对路径和 CPU-only
Accelerate 初始化；安装、编译或执行错误不再被当成缺设备而跳过。

## 真实执行与回归结果

| 验证 | 结果与边界 |
| --- | --- |
| 卷积：Jittor/PyTorch × CPU/CUDA × 3 种卷积 × 前向/前反向 | 24 个组合通过；前向及输入、权重、bias 梯度对照 NumPy；RTX 4090 上实际 CUDA 执行，零 fallback |
| 模型 harness：CPU tiny ResNet50 inference | 两个独立解释器均成功；实际 CPU、FP32、2 线程、相同 NumPy 1.26.4；Jittor fallback 为 0；标量 mean 序列在 2e-6 内一致。仅验证 harness，不等同逐元素/梯度对拍或性能结论 |
| Accelerate：CPU SGD/AdamW × direct/full | 4 个组合实际训练通过，参数有限、实际 CPU、零 fallback；对应 direct/full 参数最大绝对差分别为 7.45e-9、1.95e-7，在 FP32 `rtol=1e-5, atol=1e-6` 预算内 |
| CPU core | 322 passed、42 skipped、1 xfailed（native 与 Torch 模式合计） |
| 全部结构测试 | 1429 passed、4 skipped、1 failed；唯一失败是基线已有的 DropoutACL 梯度清单过期项 |
| 新增定向测试 | ASV 入口 10、模型协议 11、卷积 6、目录清单 6 项通过 |
| 静态门禁 | ASV 实际配置 check、清单无 site-packages 枚举、布局、文档链接、维护范围 Ruff 检查通过 |

CPU 卷积与 Accelerate 使用通用 CPU kernel；本次没有给出 CPU 优化库性能或跨框架
速度比结论。模型 smoke 的计时样本不足以发布性能比较。结构测试的唯一失败项为
`test_backend_grad_contract.py::test_the_manifest_is_exactly_the_gradients_in_the_tree`，
指向清单中已过期的 `DropoutACL` 项；该后端源码不在本次改动范围内。

## 复现与产物

从仓库根目录执行，实际计时需使用已配置的解释器、设备和独立编译缓存；完整参数
见[基准说明](../performance/benchmarking.md)与[覆盖矩阵](../../benchmarks/README.md)。

```bash
python -S benchmarks/catalog.py --json
python -m asv --config benchmarks/asv/asv.conf.json check
nox -s benchmark -- --bench '^convolution\.' --quick
nox -s benchmark_cuda -- --bench '^convolution\.' --quick
nox -s benchmark_accelerate -- --quick
python tools/run_test_suite.py --tier core --backend cpu --serial-compile
JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q tests/structure
```

`--quick` 适合执行冒烟，不用于发布稳定速度比。原始日志、缓存与 JSON 均在外置
实验目录，**未版本化**：卷积位于
`$JITTOR_LAB_ROOT/_state/benchmark-asv/asv-{cpu,cuda}/logs/`；模型 JSON 位于
`$JITTOR_LAB_ROOT/_state/benchmark-completion/harness/results-cpu/results.json`；
整合检查与 Accelerate 日志位于
`$JITTOR_LAB_ROOT/_state/benchmark-completion/integration/logs/`。

## 仍未验收的范围

本次没有重跑完整模型 CUDA/NPU harness，也没有验证 Accelerate CUDA。NPU 的
FP32 比较协议仍依赖声明，Jittor cube 精度配置缺少公开读回，修改相关配置后需要
真实 NPU 重验。ROCm/Corex 定期基线、NPU ASV/CI、多卡通信与分布式训练、数据
输入流水线、冷启动及系统性 shape/dtype 扫描仍有缺口；持续覆盖状态以覆盖矩阵为准。
