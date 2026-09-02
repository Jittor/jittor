---
name: jittor-grad-silence-probe
description: 判断 Jittor 的反向是不是把梯度静默吞掉了。给出"零梯度是真零还是没算"的区分手法、第二次 backward 静默归零的复现、缺失梯度告警的去重陷阱，以及改这类语义前必须先量一遍的爆炸半径。
---

# 判断梯度是"真的零"还是"被静默吞掉"

Jittor 反向里最贵的一类 bug 是**梯度没算出来，但接口返回零**。零是合法值，所以
`assert grad is not None` 与 `assert grad.shape == x.shape` 都通过，训练照常收敛到错的地方。
本 skill 是把这类静默区分出来的固定手法。

## 1. 零梯度的三种来源，必须先分开

| 现象 | 来源 | 怎么确认 |
| --- | --- | --- |
| 目标根本不在反向闭包里 | `grad()` 末尾 `materialize_grads` 用 `make_number(0)` 补零 | 日志里有 `doesn't have gradient. It will be set to zero` |
| 某个算子没有反向 | `Op::grad` 返回 nullptr（floor/round/ceil、mod、floor_divide、位运算、以及没实现反向的算子） | 日志里有 `Grad of <op> return zeros` |
| 前向图已被上一次 backward 摧毁 | `retain_graph=false` 时对整条闭包调 `set_stop_grad()`，而 `stop_grad` 是**永久**的 | 第一次反向对、第二次全零；`x.is_stop_grad()` 变成 True |

**判据**：把同一个前向图跑两次反向，比较两次结果。

```python
loss = build_forward(x)
g1 = jt.grad(loss, x, retain_graph=False)
loss2 = build_forward(x)          # 重新建图
g2 = jt.grad(loss2, x, retain_graph=False)
# g1 == g2 才正常；g2 全零而 g1 不是，说明第一次反向摧毁了共享的前向段
```

共享前向段的写法（一个 encoder 两个 head、GAN 不 detach）最容易踩：第一次 backward
会释放**没有被 Python 变量持有**的中间 var，第二次走到那里就断了。被 Python 持有的
var（`hold_vars`）是豁免的——所以"参数不受影响、中间量受影响"，现象非常像随机。

## 2. 告警不等于没被吞

`grad.cc` 的缺失梯度告警一度以 **var 的名字**为键做进程级去重，而绝大多数 var 名字是空串：
第一条告警之后所有缺失梯度**完全无声**。所以：

- **不要用"日志里没有 warning"当作"没有缺失梯度"的证据。**
- 量化的做法是数值对拍：把同一个模型在真 PyTorch 上跑一遍（开发环境里的 `torch` 是
  shim，对拍必须用装了真 torch 的解释器），逐参数比 `grad`，而不是只看 loss 曲线。
- 想让缺失梯度必须可见时，把去重键换成 `(op, 输入下标)` 或干脆不去重，或直接改成报错。

## 3. 改这类语义前先量爆炸半径

把"静默补零"改成"报错"，会打破一切依赖静默的既有用法（模型里有不参与当前 loss 的参数、
优化器对全部参数求梯度、多任务共享 trunk 等）。**先跑一遍再决定默认值**：

```bash
JITTOR_TEST_DEVICES=cpu nvcc_path="" pytest tests/core tests/nn tests/optim tests/models -q
```

- 红的不多 → 默认报错，给一个 flag 让用户显式降级为警告。
- 面太大 → 保留补零，但**去掉去重**让每一次缺失都报出来，另加 flag 让用户升级为报错。

**本仓库量过的结果**（2.0 整改 6.C07）：默认报错会红 16 条，分布在
`test_grad.py`、`test_function.py`、`test_rootcause_semantics.py`、`test_setitem.py`、
`test_reindex_op.py`、`test_misc_issue.py`。其中两条是**有文档的语义**，不是疏忽：

- `x.requires_grad = False` 之后 `jt.grad(x**2, x) == 0`；
- `jt.Function.grad` 对某个输入返回 `None` 时该输入的梯度是 0。

所以缺失梯度**不能**默认报错，最终选的是"每次都警告 + `jt.flags.missing_grad_error=1`
升级为报错"。要改这个默认值，先把上面两条语义一起改掉。

无论选哪个，把"选了哪个、为什么、量到了多少红"写进提交说明——这是后来人唯一能复查的依据。

## 4. 复现脚本要起子进程

反向里的空指针解引用（`make_grad` 对 nullptr 的 `dx` 写 `loop_options`、amp level 3 的
回转 cast 读 `grad->ns`）**崩的是进程**。放在同一个 pytest 进程里，回归表现为整个
session 猝死而不是一条断言失败。这类用例一律 `subprocess.run` 起子进程断言退出码，
并且显式传 `PYTHONPATH=<worktree>/python`（jittor 是 editable 安装，裸 `python -c`
导入的是 `.pth` 指向的那棵树）。
