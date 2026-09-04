---
name: jittor-grad-silence-probe
description: 判断 Jittor 的反向是不是把梯度静默吞掉了，以及怎么问「反向图的形状」。给出"零梯度是真零还是没算"的区分手法、第二次 backward 静默归零的复现、缺失梯度告警的去重陷阱、改这类语义前必须先量一遍的爆炸半径；另有 is_leaf/grad_fn 这类图查询必须复用的四条边过滤器、和真 torch 对 autograd 语义时怎么先把 requires_grad 的差异摘出去、以及怎么用 tflag_count 证明一个查询没有做图遍历（从而不需要缓存）。
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

## 5. 反向图的形状：`is_leaf` / `grad_fn` 只有一条正确的问法

「梯度能不能进到这个 var」这件事，内核里只有一个地方真的在算：`grad()` 的
`bfs_backward` 的那组过滤器。任何新查询都必须复用**同一组**，否则「查询说它是叶子」
与「`jt.grad` 真的走不进去」会分叉，而分叉的方向是静默的。

四条过滤器，缺一条就答错。每条都有一个具体的、今天就能触发的反例：

| 过滤器 | 为什么必须有 | 漏掉它的症状 |
| --- | --- | --- |
| var 自己的 requires_grad：`!is_stop_grad() && !flag(_requires_grad_disabled)`，**两个标志都要读** | `stop_grad` 是永久的，`requires_grad_(False)` 是可逆的，两者都让梯度进不来 | 只读 `_stop_grad`：`x.requires_grad = False` 之后仍然报「非叶子」，而且把它打开再关上答案不回来 |
| 生产者算子**自己**的 `stop_grad` | **`detach()` 把 stop_grad 标在 clone 算子上，不标在它产出的 var 上**（`ops/clone_op.cc`）。所以 `x.detach().requires_grad` 在 Jittor 里是 `True` | 只看 var 的标志：detach 出来的 var 被报成非叶子 |
| 入边的 `index < 0`，即控制依赖边（`VarHolder::_add_dependency` 打的标记） | `make_grad` 对负下标直接返回 nullptr，这条边只排执行顺序 | 加一条 `_add_dependency` 就把叶子变成非叶子 |
| 冻结的 requires-grad-disabled 边（`is_requires_grad_disabled_edge`） | `Op::init` 在算子构造时快照了当时被禁用的输入边；之后把那个 var 的 requires_grad 打开，这条边仍然不导梯度 | 把 requires_grad 往回打开就能凭空穿过一条冻结的边 |

**判据**：「不是叶子」必须等价于「`jt.grad` 真的会走进那个算子」。所以每个用例同时断言
两件事——查询的答案，以及 `jt.grad(loss, [v])` 有没有拿到非零梯度。只断言查询自己，
证明的只是查询和自己一致。

本仓库的实现是 `jittor::backward_grad_fn`（`src/grad.h`）；用例在
`tests/core/test_backward_leaf_query.py` 与 `src/tests/test_backward_leaf.cc`，一条一条
对应上表。

## 6. 和真 torch 对 autograd 语义：先把 requires_grad 的差异摘出去

Jittor 与 torch 在 autograd 上的差异**几乎全部集中在 `requires_grad` 的默认值与传播**，
而不在图的形状上。所以对拍要比的是三元组，不是单个属性：

    (requires_grad, is_leaf, grad_fn is None)

已经量过的差异（`is_leaf`/`grad_fn` 两侧一致，只有 `requires_grad` 不一致）：

| 构造 | torch 的 requires_grad | Jittor 的 | 为什么 |
| --- | --- | --- | --- |
| 新建一个 float 张量 | `False` | `True` | Jittor 的 float var 默认可导（`var.cc` 只对非 float 与 `no_grad` 设 `_stop_grad`） |
| `t.detach()` | `False` | `True` | detach 停的是算子（见上一节） |
| 对一个 `stop_grad` 的 var 做算子 | `False` | native 策略下 `True` | `stop_outputs_when_inputs_stopped` 默认关；`EXPLICIT_REQUIRES_GRAD` 策略才打开 |

**判据**：先按三元组分类，只在 `requires_grad` 一致的用例上追究 `is_leaf`/`grad_fn`；
`requires_grad` 本身不一致的用例单独列成一张「归属别的任务」的表，并写清是哪一条。
**不要把 requires_grad 的差异改成 `is_leaf` 的补丁**——「`is_leaf` 恒 `True`」那种修法
就是这么来的。

真 torch 一侧跑在另一个装了 binary PyTorch 的解释器里（开发环境里的 `torch` 是 shim）。
用 `REAL_TORCH_PYTHON` 指向它，并且**把它答出来的表按提交冻结在用例里**：只在环境里
有真 torch 时重新推导一遍并断言冻结值没变。只 skip 的对拍在门禁上是一条恒绿的空用例。

## 7. 怎么证明「这个查询没有做图遍历」

属性访问触发的查询（`is_leaf`、`grad_fn`、`requires_grad`），"会不会每次都全图遍历"
不能靠读代码保证。这棵树里每一次图遍历都要取一个 `TraversalEpoch`，而每个 epoch 都会
推进全局 `tflag_count`（`misc/traversal_epoch.h`）。于是有一条不依赖计时的判据：

```cpp
int64 before = tflag_count;
for (int i=0; i<64; i++) CHECK(!is_backward_leaf(deep_chain_tail));
CHECKop(tflag_count,==,before);      // 计数器没动 == 没开过遍历
```

比计时稳：计时随机器负载翻，这一条要么成立要么不成立。配套再加一条——在一个**还没结束
的** `TraversalEpoch` 里做同样的查询，断言外层的每个标记都还在、`epoch.displaced` 是空的。
属性访问会落在任何时刻，包括别人正在走图的时候（`MemoryProfiler::check()` 就是从
`run_sync` 的算子循环里走全图的）。

反过来也成立：**只有能通过这条判据的查询才不需要缓存。** 一旦需要缓存，下一个问题必然是
「什么时候失效」，而陈旧标记（`5.03` 的转置隐藏标记）是这棵树已经付过一次代价的坑。
