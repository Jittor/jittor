# 开始使用 Jittor

[文档首页](../index.md) · [English](index.en.md)

第一次使用，依次阅读[安装](installation.md)和[快速开始](quickstart.md)。这条路径只用 CPU，不需要模型权重或数据集下载。

| 你想做什么 | 接下来读 |
| --- | --- |
| 学习张量、求导与训练 | [完整入门教程](https://github.com/Jittor/jittor/blob/2.0-refactor/examples/notebooks/getting_started.md) |
| 已有 PyTorch 脚本 | [Torch 兼容使用指南](../compatibility/torch.md) |
| 按课程继续学习 | [教程目录](https://github.com/Jittor/jittor/blob/2.0-refactor/examples/notebooks/README.md) |
| 查询函数参数 | [API 参考](../api/index.md) |

## 运行完整教程

仓库的 `examples/notebooks` 保存可转换为 notebook 的 Markdown 教程。从仓库根目录运行：

```bash
python -m pip install -r requirements/examples.txt
STATE="${JITTOR_LAB_ROOT:-../jittor-lab}/_state/notebooks"
mkdir -p "$STATE"
cp -R examples/notebooks/. "$STATE/"
find "$STATE" -type f -name '*.md' ! -name README.md -print0 \
  | xargs -0 -n1 python -m jupytext --to ipynb
python -m notebook --ServerApp.root_dir="$STATE"
```

先完成安装，再运行这些单元格。教程目录注明了前置知识与设备要求；包含数据集下载的章节需要网络。生成的 notebook 和运行结果写在 `STATE` 中，便于保留自己的练习。

<!-- sphinx-only
```{toctree}
:hidden:

installation
quickstart
```
-->