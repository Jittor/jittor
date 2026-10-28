---
name: datatrove-torch-compat
description: 用同一组离线输入在 Jittor Torch shim 与独立 PyTorch 上复验 DataTrove CPU 数据集和本地 JSONL 管线。
---

# DataTrove CPU compatibility

## 用途

固定 DataTrove `v0.10.0`，上游
`https://github.com/huggingface/datatrove.git`，提交
`7024aecca2f9ffb7b7cf0d02c0c823b8b24cf664`。

维护入口为
[`compat/tests/torch/test_datatrove_compat.py`](../../../compat/tests/torch/test_datatrove_compat.py)，
范围仅为以下三个 CPU 场景：

1. `DatatroveFileDataset` 的 uint16 token 文件、int64 CPU Tensor 与 EOS position 重建；
2. JSONL reader → LambdaFilter → writer 的本地管线；
3. `LocalPipelineExecutor(tasks=4, workers=2, start_method="forkserver")`，核对输出 ID、
   过滤统计和每个 rank 的 completion 文件。

## 两侧环境

两个独立解释器使用同一个固定 DataTrove checkout，以及 `orjson`、`regex`。
先核对安装后的 DataTrove 版本及源码身份，再解释结果。
oracle 使用官方 PyTorch；Jittor 使用当前源码及 shim 的构建依赖。

- oracle 必须有真实 `torch._C`，且不存在 `_torch_compat_install_context`；
- Jittor 分支断言独立 Torch frontend 已激活，`torch.Tensor is not jittor.Var`；
- 显式 `JT_USE_CUDA=0`，测试断言 `jittor.flags.use_cuda == 0`；
- `JITTOR_REQUIRE_DATATROVE=1` 将缺依赖变成错误，不能把全 skip 当作通过。

运行状态与原始日志放在 `$JITTOR_LAB_ROOT/_state/datatrove/<run>/`，不进入仓库。

## 在 shim 上跑

在当前 checkout 根目录执行；每个首次编译使用独立缓存并串行运行：

```bash
JITTOR_REQUIRE_DATATROVE=1 JITTOR_TORCH_SHIM=1 \
PYTHONPATH="$PWD/compat/shim/resources:$PWD/python" \
JT_USE_CUDA=0 JT_BUILD_NVCC_PATH='' JT_BUILD_USE_NCCL=0 JT_BUILD_USE_CUTT=0 \
JT_USE_PARALLEL_OP_COMPILER=0 DISABLE_MULTIPROCESSING=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
JITTOR_HOME="$JITTOR_LAB_ROOT/_state/datatrove/<run>/jittor-home" \
<jittor-python> compat/tests/torch/test_datatrove_compat.py -v
```

## 在原生 torch 上跑

使用干净环境，不继承 `JITTOR_TORCH_SHIM` 或 shim 的 `PYTHONPATH`：

```bash
JITTOR_REQUIRE_DATATROVE=1 JITTOR_DATATROVE_RUNTIME=oracle \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
<torch-python> compat/tests/torch/test_datatrove_compat.py -v
```

## 对拍

两侧执行同一文件、同一固定输入与预期值，整数 token、positions 和文档 ID 必须
精确相等，dataset 输出还必须为 int64 CPU Tensor。多 worker 检查全部四个 rank
完成标记以及 total=8、forwarded=4、dropped=4；不是只看进程退出。

这三个数据处理场景没有模型参数或训练梯度，不能用模型训练的等级替代这些断言。

## 断点与分流

普通 JSONL 管线直接运行；dataset 使用现有 `jittor.compat.torch`。
不需要 DataTrove 专属 adapter，不修改下游源码或框架状态。

首次失败先在独立 oracle 复现；只有 shim 出错时才归入兼容层。
Jittor 导入前的编译错误归构建前置，不能算 DataTrove 数值验证失败或通过。

## 坑与假绿

1. JSONL 路径自身不调用 torch，不能代替 dataset 的框架兼容验证。
2. 导入成功和测试文件存在不能代替实际执行；三个测试均有输出断言。
3. Python 3.13 / `multiprocess 0.70.19` 的 Manager 子进程会触及已变更的 tempfile
   私有 API。测试父进程预建 multiprocess 临时目录后恢复 `tempfile.tempdir`；
   两侧使用同一前置，不在生产 shim 中补丁第三方库。

## 证据

服务器上同一固定输入的两侧实测结果见
[`CPU 适配报告`](../../../docs/results/2026-10-24-datatrove-cpu-compatibility.md)。
普通 pytest 门禁发现该测试文件；没有依赖时显式报告环境 skip，验收环境使用
`JITTOR_REQUIRE_DATATROVE=1` 强制实际执行。
