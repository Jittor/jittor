# `test_example` 可用性验证工作记录

状态：🔴 有问题

## 目标

验证当前 `2.0` 分支上的以下入口是否可用：

```bash
python -m jittor.test.test_example
```

本次只验证并诊断，不修改 Jittor 源码。

## 验证环境

- 时间：2026-07-12 17:40（Asia/Shanghai）
- 分支：`2.0`
- HEAD：`540a2fe74b2d838f752b127431baa748bc952a00`
- Python：3.11.15，`/home/zy/miniconda3/envs/jt311/bin/python`
- Jittor：1.3.11.0，源码来自 `/home/zy/projects/jittor/python/jittor`
- 设备：NVIDIA GeForce RTX 4090，物理 GPU 1
- CUDA/cuDNN：项目已有的 `cuda12.2_cudnn8_linux` 工具链
- 独立缓存：`${JITTOR_LAB_ROOT}/_state/test-example/test_example_20260712_default/`
- 本地完整日志：`agent/worklogs/2026-07-12-test-example.log`（未版本化）

当前非交互 shell 的 `PATH` 中没有 `python`。验证时把 `jt311/bin` 放到 `PATH` 首位，实际测试命令主体仍为 `python -m jittor.test.test_example`。

## 执行方式

```bash
export PATH=/home/zy/.cache/jittor/jtcuda/cuda12.2_cudnn8_linux/bin:/home/zy/miniconda3/envs/jt311/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/home/zy/.cache/jittor/jtcuda/cuda12.2_cudnn8_linux/lib64
export JTCUDA=/home/zy/.cache/jittor/jtcuda/cuda12.2_cudnn8_linux
export CUDA_HOME="$JTCUDA"
export nvcc_path="$JTCUDA/bin/nvcc"
export PYTHONPATH=/home/zy/projects/jittor/python
export JITTOR_LAB_ROOT=${JITTOR_LAB_ROOT:-/home/zy/projects/jittor-lab}
export JITTOR_HOME="$JITTOR_LAB_ROOT/_state/test-example/test_example_20260712_default"
export CUDA_VISIBLE_DEVICES=1
export use_cuda=1
python -m jittor.test.test_example
```

首次用空 `JITTOR_HOME` 预检时，Jittor 发现了系统 `/usr/local/cuda`，但该目录没有 `cudnn.h`，因此导入失败。显式配置项目已有的 `JTCUDA` 后，当前仓库源码、CUDA 和独立缓存均正常加载，正式测试得以执行。

## 验收标准

- 进程退出码为 0。
- `unittest` 输出 `Ran 1 test` 和 `OK`。
- 1000 个训练步骤完成。
- `jt.liveness_info()` 稳定，不触发内存泄漏断言。
- 最终 loss 命中源码列出的结果之一，绝对误差小于 `1e-6`。

## 实际结果

正式测试退出码为 1：

```text
TypeError: 'dtype' object is not callable
Ran 1 test in 15.779s
FAILED (errors=1)
```

异常发生在 `python/jittor/test/test_example.py:54` 的第一个 batch 数据转换：

```python
yield jt.float32(x), jt.float32(y)
```

测试没有进入训练循环，因此 1000 步训练、liveness 稳定性和最终 loss 均未被验证。

## 根因

`python/jittor/__init__.py` 在导入末尾调用 `torch_compat.install()`。该安装过程通过 `_make_dtypes(g)` 无条件把顶层 `jt.float32` 设置为 PyTorch 风格的 dtype 对象，覆盖了 Jittor 旧接口中的可调用转换符号。

最小探针结果：

```text
FLOAT32_REPR=torch.float32
FLOAT32_TYPE=jittor.torch_compat.dtype
FLOAT32_CALLABLE=False
EXPLICIT_ARRAY_DTYPE=float32
EXPLICIT_ARRAY_VALUE=[1.25]
```

`git blame` 显示：

- `test_example.py` 的 `jt.float32(x)` 调用自 2020 年起存在。
- 顶层自动安装 torch compatibility layer 及 dtype 覆盖来自提交 `bf44c474d`（2026-06-21）。
- 仓库中还有多项旧测试和文档继续使用 `jt.float32(...)`，因此问题不只影响 `test_example`。

## 结论与后续

当前 HEAD 下，`python -m jittor.test.test_example` **不可用**。这是已复现的 API 兼容回归，不是 CUDA 编译失败或训练数值偏差。

当前可工作的显式构造形式是：

```python
x = jt.array(data, dtype="float32")
```

后续修复需要同时保持两类语义：PyTorch 生态需要 `jt.float32` 作为 dtype 对象，Jittor 旧 API 和现有测试又依赖 `jt.float32(data)` 可调用。修复后应重新运行本测试，并覆盖仓库内其他 `jt.float32(...)` 调用点。本次按约定只完成验证和诊断，没有修改源码。
