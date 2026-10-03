# OpenAI Whisper：CPU L2 兼容验证记录

- **状态**：正式支持等级 L2；L3–L5 不在本提交声明范围
- **日期**：2026-09-22
- **负责人**：赵佳祥
- **Jittor 基线**：`origin/2.0-refactor@e9234ca8` + 本提交候选
- **Whisper**：`openai-whisper==20250625`，上游提交 `86098128c0b4f24f0e2aa2994de830614b474227`
- **独立 oracle**：PyTorch `2.4.1+cpu`，Python `3.11.16`，NumPy `1.26.4`
- **范围**：CPU、FP32、固定 tiny config；不声明 CUDA/NPU、真实 checkpoint 保存互通、完整解码或性能

## 结论

本提交在不修改 OpenAI Whisper 下游源码、不增加 Whisper 私有 adapter 的前提下，
补齐原版 Whisper 达到严格累计 L2 所需的公共能力：bool COO 非持久 buffer、可微
STFT/log-Mel、模型状态传递、完整参数及输入梯度检查，以及三步交叉熵/SGD 训练轨迹。

| 等级 | 结果 | 验证内容 |
| --- | --- | --- |
| L0 导入与构造 | 通过 | 原版 `import whisper`；固定 `ModelDimensions` 构造；88 个参数及 `alignment_heads` buffer 的名称、shape、dtype/layout |
| L1 前向语义 | 通过 | 同权重模型前向；原版 `log_mel_spectrogram`；shape、dtype、数值和错误退出；禁止 backend fallback |
| L2 训练语义 | 通过 | 三步交叉熵；SGD `lr=0.005, momentum=0.9`；每步 88 个参数梯度、mel 输入梯度、参数值和更新量；`train→eval→train`；`zero_grad(set_to_none=True/False)` |
| L3 权重与任务 | 未声明 | 公共双向 checkpoint 安全 round-trip 不属于本提交 |
| L4 端到端能力 | 未声明 | 真实 checkpoint 的完整 greedy/beam/sample/cache/词级时间戳不属于本提交 |
| L5 性能 | 未验证 | 未执行成对速度、峰值显存与真实规模性能验收 |

OpenAI Whisper 模型没有 Dropout 模块，因此 L2 对模式切换做传播和输出一致性检查，
并明确记录 dropout 不适用；没有伪造随机 dropout 验收。

## 实现归属

| 能力 | 实现位置 | 原因 |
| --- | --- | --- |
| dense bool → COO、`to_dense` | `python/jittor/sparse/coo.py` | 真实存储与转换属于 native sparse 能力 |
| `torch.sparse_coo`、layout、identity、复制协议 | `compat/torch/sparse_frontend.py` 及 installers | PyTorch 公共协议属于兼容层 |
| 非持久 sparse buffer | `python/jittor/_core/module.py` 与 Module installer | Whisper 构造需要 `alignment_heads` 的注册、枚举和迁移 |
| 可微 STFT | `python/jittor/fft/spectral.py` | 波形和窗口必须保留在计算图内，不能转 NumPy |
| `torch.stft` 参数协议 | `compat/torch/installers/numerical/signal.py` | 兼容 PyTorch 的参数、默认值和返回结构 |
| 独立 oracle 与状态传递 | ecosystem harness/runner | 保证参考端是真 PyTorch，并完整比较参数、buffer、输出和梯度 |
| 三步训练 | `test_whisper_training.py` 与 `acceptance_probe.py` | 持续验证 task loss、全部梯度、optimizer 和模式行为 |

## 精度口径

前向与梯度使用组级尺度归一化最大偏差：

```text
divergence = max(abs(actual - expected)) /
             max(max(abs(expected)), 1e-3 * group_peak, 1e-6)
```

- 小配置模型：前向 `< 2e-3`，梯度 `< 1e-2`。
- log-Mel：前向 `< 2e-5`，梯度 `< 2e-3`。
- 三步训练：loss/logits `< 2e-3`；参数值 `< 2e-5`；梯度、输入梯度及更新量 `< 1e-2`。
- bool COO 的坐标、值、shape、dtype、layout、身份和错误类别采用精确比较。

## 本次复验

当前最新基线使用全新的隔离缓存和两个解释器运行。最终提交前必须全部满足：

| 门禁 | 结果 |
| --- | --- |
| oracle 身份与版本 | 通过：PyTorch 2.4.1+cpu，`shim=false` |
| Jittor shim 导入 | 通过：当前工作树 Jittor `1.3.11.0`，`shim=true` |
| sparse + STFT + 模型前向/log-Mel | 通过：`75 + 2` passed，0 skipped |
| 三步训练 L2 | 通过：`1` passed，0 skipped；三步、每步 88 个参数梯度 |
| 结构、manifest、语法与 `git diff --check` | 通过：结构 `72` passed；30 个 Python 文件语法通过；布局、manifest、skill validator 和 diff check 通过 |
首次模型门禁在进入数值比较前拒绝运行：oracle 的 `runtime_threads=2`，shim 报告 8。
按 nox 契约把 `OMP_NUM_THREADS`、`MKL_NUM_THREADS`、`OPENBLAS_NUM_THREADS` 统一为
本机逻辑 CPU 数 8 后，原断言和容差不变，模型与 log-Mel 两项通过。该失败作为环境
基线校验证据保留，不能写成数值修复。

## 复现

```bash
export REAL_TORCH_PYTHON=<oracle>/bin/python
export JITTOR_ECOSYSTEM_PACKAGE_SITE=<oracle>/lib/python3.11/site-packages
export JITTOR_REQUIRE_REAL_TORCH=1
export JITTOR_REQUIRE_WHISPER=1
export JITTOR_TEST_DEVICES=cpu
export JT_BACKEND=cpu
export JITTOR_TORCH_FALLBACK=error
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

python -m pytest -q -s --confcutdir=compat/tests \
  compat/tests/torch/test_torch_sparse_metadata.py \
  compat/tests/torch/test_torch_stft.py \
  compat/tests/torch/test_ecosystem_parity.py::OpenAIWhisperParity \
  compat/tests/torch/test_whisper_training.py
```

`REAL_TORCH_PYTHON` 必须先通过：

```python
import torch
assert hasattr(torch, "_C")
assert not hasattr(torch, "_torch_compat_install_context")
```

原始 JSON、NPZ、日志、虚拟环境和 JIT 缓存位于仓库外实验目录，不进入版本控制。