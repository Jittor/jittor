# 检查点加载会保留独立的模型存储

- 状态：KI-COMPAT-006 已关闭，限于经验证的 CPU/CUDA 复制式加载契约。
- 复查日期：2026-10-16。
- 基线：`9fdc2b57f1f5deefb585e6145e5ed97fdc3b0ed9`，加上本报告的两处生产改动。
- 维护者：Torch 兼容性与原生优化器维护者。
- 复查条件：模块加载、跨步复制、存储别名或融合 AdamW 发生变化。

默认的 `load_state_dict(assign=False)` 现在与处理连续参数时一样，通过既有的公开原生
`copy_` 所有者复制进非连续参数。目标保留其 Python 对象、分配、跨步、dtype 和解绑别名。
此前的非连续路径会重新绑定源存储：随后对源做一次融合 CUDA AdamW 更新，会静默地一并
改动目标。FP32 与 BF16 在 CPU 上都能复现该分配/跨步缺陷；FP32 在 CUDA 上复现了这次
静默的跨模型更新。CPU 上那个会重新绑定的优化器恰好掩盖了该更新。

BF16 的 CUDA 复现还暴露了一个原生融合 AdamW 的编译缺陷。外层命名空间中的半精度重载
遮蔽了双精度 `pow` 重载。把这两处既有调用限定为 `::pow` 即可修复 FP16/BF16 的 CUDA
编译，且不改变偏置校正公式及其精度。

修复前，维护版加载回归在 CPU 上通过两个用例、失败两个跨步用例；四个 CUDA 用例被跳过。
维护版原生 CUDA 回归通过 FP32，但对 FP16/BF16 编译失败。原始失败记录予以保留。修复后：

| 命令范围 | 结果 |
| --- | --- |
| 三个模块加载文件，CPU | 14 passed，4 个 CUDA skip |
| 同样三个文件，CUDA 可见 | 18 passed |
| 新增加载测试未改动地在真实 Torch 中运行，CPU | 4 passed，4 个 CUDA skip |
| 新增加载测试未改动地在真实 Torch 中运行，CUDA 可见 | 8 passed |
| 原生融合 AdamW、优化器核心与 RMSprop 契约，CUDA | 49 passed |

CUDA 分配检查使用真实的驱动指针属性。加载回归执行真实反向传播和原始融合 AdamW
内核，计入它的两个参数条目；它检查只有源模型发生变化。整调用候选守卫在 CPU 与 CUDA
上均报告零回退。真实参照是独立的 Torch 2.11 CPU 与 CUDA 安装，不含 Jittor shim。

原生融合测试覆盖 FP32/FP16/BF16、两个参数和三个步骤，将全部 54 个参数/动量数组与
新建的真实 Torch CUDA 基准数据逐一比对。它使用 `beta1=beta2=0`、零权重衰减与
`lr=eps=.125`，从而把编译与存储写入同不相关的低精度舍入差异隔离开。低精度数组逐位
一致；FP32 使用 `rtol=1e-6, atol=1e-7`。每份输出都保留其输入 GPU 存储指针。主机快照
先复制，以免迁移原始分配。

```bash
JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q \
  compat/tests/torch/test_module_load_independence.py \
  compat/tests/torch/test_module_load_grad_mode.py \
  compat/tests/torch/test_module_load_storage.py

# In a real CUDA-enabled native environment:
JITTOR_TORCH_SHIM=0 PYTHONPATH=python python -m pytest -q \
  tests/optim/test_fused_adamw_device_contract.py \
  tests/optim/test_optim_core.py tests/optim/test_rmsprop_device_contract.py
```

本结论不确立通用的 saved-value/version 语义、完整的优化器 ABI、其他加速器或整模型
支持。`assign=True` 保留其独立的赋值行为。原始命令、失败尝试、精确源码哈希、真实基准
来源、加载的核心副本与编译后的 CUDA 内核快照未纳入版本控制，位于
`$JITTOR_LAB_ROOT/_state/module-load-independence/20261016/`。
两次早期 CPU 启动尝试在收集前即被拒绝，因为它们继承了要求 CUDA 的门禁；最终 CPU
命令显式选择 CPU。

整合核心通过 **322 / 42 skip / 1 个既有 xfail**；合并的原生与兼容性结构门禁通过
**1628 / 6 skip**。仓库布局、生成的 manifest 与新增测试的 Ruff 检查均通过。这些检查
不扩大设备覆盖范围。
