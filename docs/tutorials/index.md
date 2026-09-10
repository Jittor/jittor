# 教程

可执行教程以 MyST Markdown 维护在 `examples/notebooks` 下。

**学习路径的唯一权威来源是
[`examples/notebooks/README.md`](https://github.com/Jittor/jittor/blob/master/examples/notebooks/README.md)**，
那里有按顺序编排的完整课程，注明每篇教什么、需要什么前置、是否需要显卡。

这里不再重复一份清单。此前本页、根 README 和 notebooks 目录各维护一份，三份互不
一致且没有一份完整——`tests/integration/test_notebooks.py` 现在会检查权威路径列全了
所有教程，一篇存在却没被编排的教程会让门禁变红。

## 两条入门线

* **零基础**：从「60 分钟入门」四篇开始，安装到 MNIST。
* **有其它框架经验**：走主线，它只讲 Jittor 与它们的差异——先读
  [算子与 Var](https://github.com/Jittor/jittor/blob/master/examples/notebooks/basics.md)，
  再读
  [设备与驻留](https://github.com/Jittor/jittor/blob/master/examples/notebooks/device_placement.md)，
  然后是
  [模型定义与训练](https://github.com/Jittor/jittor/blob/master/examples/notebooks/example.md)。

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
