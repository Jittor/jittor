# 教程

可执行教程以 MyST Markdown 维护在 `examples/notebooks` 下。

**学习路径的唯一权威来源是
[`examples/notebooks/README.md`](https://github.com/Jittor/jittor/blob/master/examples/notebooks/README.md)**，
那里有按顺序编排的完整课程，注明每篇教什么、需要什么前置、是否需要显卡。

这里不再重复一份清单。此前本页、根 README 和 notebooks 目录各维护一份，三份互不
一致且没有一份完整——`tests/integration/test_notebooks.py` 现在会检查权威路径列全了
所有教程，一篇存在却没被编排的教程会让门禁变红。

## 从哪里开始

**唯一的入口**是
[从零开始：装好、跑通、训练第一个模型](https://github.com/Jittor/jittor/blob/master/examples/notebooks/getting_started.md)，
它从安装讲到训练，全程 CPU 可跑完。之后按背景选一条主线：

* **原生 Jittor**：接着读
  [算子与 Var](https://github.com/Jittor/jittor/blob/master/examples/notebooks/basics.md)、
  [设备与驻留](https://github.com/Jittor/jittor/blob/master/examples/notebooks/device_placement.md)、
  [模型定义与训练](https://github.com/Jittor/jittor/blob/master/examples/notebooks/example.md)。
* **PyTorch 兼容**：手上已有 `torch` 代码时，从
  [用 PyTorch API 写 Jittor](https://github.com/Jittor/jittor/blob/master/examples/notebooks/torch_compat.md)
  开始，再读
  [把已有 PyTorch 脚本迁到 Jittor](https://github.com/Jittor/jittor/blob/master/examples/notebooks/torch_compat_migration.md)。

「设备与驻留」排在训练之前是有意的：Jittor 用一个全局标志移动整张计算图，而不是
每个张量自带设备；不先弄清这个差异，后面的性能问题会被归因到错误的地方。

## 在本地打开

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
