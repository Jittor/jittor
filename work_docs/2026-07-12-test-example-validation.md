# `test_example` 可用性验证

状态：🔴 当前不可用

## 验证对象

```bash
python -m jittor.test.test_example
```

验证环境为 `2.0` 分支 HEAD `540a2fe7`、Python 3.11.15、Jittor 1.3.11.0、CUDA 12.2 + cuDNN 8 和 RTX 4090。测试使用物理 GPU 1，并为本次运行配置独立的项目内 `JITTOR_HOME`。

## 结果

命令退出码为 1：

```text
TypeError: 'dtype' object is not callable
Ran 1 test in 15.779s
FAILED (errors=1)
```

异常发生在 `python/jittor/test/test_example.py:54` 的 `jt.float32(x)`。错误位于第一个 batch 数据转换，测试没有进入 1000 步训练，因此 liveness 稳定性和最终 loss 均未验证。

## 根因

`python/jittor/__init__.py` 在导入末尾调用 `torch_compat.install()`；安装过程中的 `_make_dtypes(g)` 会把顶层 `jt.float32` 覆盖为 PyTorch 风格的 `jittor.torch_compat.dtype` 对象。最小探针确认：

```text
FLOAT32_REPR=torch.float32
FLOAT32_TYPE=jittor.torch_compat.dtype
FLOAT32_CALLABLE=False
```

显式 `jt.array(data, dtype="float32")` 可以正常创建 float32 Var。该问题属于 Jittor 旧可调用 dtype API 与 torch dtype 对象语义之间的兼容回归，不是 CUDA 编译或训练精度问题。仓库内还有其他测试使用 `jt.float32(...)`，修复时需要统一考虑。

## 环境说明

当前非交互 shell 的 `PATH` 中没有 `python`，需使用 `/home/zy/miniconda3/envs/jt311/bin/python`。空 `JITTOR_HOME` 还需显式配置已有的 `/home/zy/.cache/jittor/jtcuda/cuda12.2_cudnn8_linux`，否则会选到缺少 `cudnn.h` 的系统 `/usr/local/cuda`。

本次没有新增依赖，也没有修改 Jittor 源码。详细工作记录位于 `agent/workdocs/2026-07-12-test-example-validation.md`，中文交付文档位于 `/home/zy/projects/doc/2026-07-12-test-example-validation.md`，原始日志位于 `agent/worklogs/2026-07-12-test-example.log`。
