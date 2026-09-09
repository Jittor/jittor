# 教程

可执行教程以 MyST Markdown 维护在 `examples/notebooks` 下。`tutorials` 这个 nox
会话会用 Jupytext 在仓库外生成 notebook，并执行离线 CPU 冒烟教程。

## 快速开始

- [模型定义与训练](https://github.com/Jittor/jittor/blob/master/examples/notebooks/example.md)
- [算子与变量](https://github.com/Jittor/jittor/blob/master/examples/notebooks/basics.md)
- [元算子](https://github.com/Jittor/jittor/blob/master/examples/notebooks/meta_op.md)
- [自定义 C++ 与 CUDA 算子](https://github.com/Jittor/jittor/blob/master/examples/notebooks/custom_op.md)
- [性能分析器](https://github.com/Jittor/jittor/blob/master/examples/notebooks/profiler.md)

## 模型与技术

- [残差网络训练](https://github.com/Jittor/jittor/blob/master/examples/notebooks/resnet_training.md)
- [从零实现 Transformer](https://github.com/Jittor/jittor/blob/master/examples/notebooks/transformer.md)
- [从零实现去噪扩散模型](https://github.com/Jittor/jittor/blob/master/examples/notebooks/diffusion.md)
- [LoRA 参数高效微调](https://github.com/Jittor/jittor/blob/master/examples/notebooks/lora.md)

## 在本地打开 notebook

```bash
python -m pip install -r requirements/examples.txt
STATE="${JITTOR_LAB_ROOT:-../jittor-lab}/_state/notebooks"
mkdir -p "$STATE"
cp -R examples/notebooks/. "$STATE/"
find "$STATE" -type f -name '*.md' ! -name README.md -print0 \
  | xargs -0 -n1 python -m jupytext --to ipynb
python -m notebook --ServerApp.root_dir="$STATE"
```

notebook 由 Markdown 源生成，**生成物不进仓库**，一律写在 `$JITTOR_LAB_ROOT/_state`
之下。可复现性与离线 CPU 检查用：

```bash
python -m nox -s tutorials
```
