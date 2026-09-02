---
name: optimizer-semantics-diffing
description: 验证 jittor 优化器语义（梯度累积等价性、全局梯度裁剪只施加一次、偏差修正按 param group 步数、zero_grad 清缓冲）的对拍口径与陷阱。改 optim/base.py、optim/algorithms/*、optim/legacy_schedulers.py 或写这些地方的回归用例前先读。
---

# 优化器语义怎么测才算数

优化器的错都是「静默算错」：不抛异常、loss 还在降，只是降得不对。下面每条都写清
**判据**（什么样的断言才能真的抓住它）和 **反例**（看着合理但抓不住的写法）。

## 0. 前置：别测错一棵树

- `pytest` 会用当前 worktree 的 `python/`（`tests/conftest.py` 决定，可用 `JITTOR_SOURCE_ROOT` 覆盖）。
- **手写的 `python -c`、脚本、子进程都不会**：`jittor` 是 editable 安装，`.pth` 指向主树。
  必须显式 `PYTHONPATH=<worktree>/python`，并且第一次验证前先自检：
  `python -c "import jittor,os;print(os.path.dirname(jittor.__file__))"` 必须打印 worktree 路径。
- 每个并行 agent 用自己的 `JITTOR_HOME` 与 `TMPDIR`；**同一个 JITTOR_HOME 下不要并发跑
  两个 jittor 进程**。纯 Python 改动加 `JITTOR_TEST_DEVICES=cpu nvcc_path=""` 快很多。

## 1. 梯度累积等价性（偏差修正是否用错步数）

`Optimizer.backward` 每调用一次就 `self.n_step += 1`，而 `base.py` 的 docstring 正在推荐
「累积 k 次 backward 再 step 一次」的写法。任何拿 `self.n_step` 做偏差修正的优化器
（Adam / AdamW / Adan）在这种写法下指数就偏了 k 倍。

**判据**：用对参数线性的 loss（`loss = (g_target * p).sum()`，梯度恒等于 `g_target`），
让两个优化器从同一初值出发：

```python
opt_ref.step(linear_loss(ref, g1 + g2))          # 一次 backward
opt_acc.backward(linear_loss(acc, g1))           # 累积两次
opt_acc.backward(linear_loss(acc, g2))
opt_acc.step()
assert acc == ref                                 # 必须逐元素相等
```

线性 loss 让「两个半梯度之和 == 整批梯度」在数值上精确成立，差异只可能来自步数。
Adam 第一步用 n=1 与 n=2 的 step_size 差约 26%，rtol=1e-5 抓得住。
再跑一个多步版本（连续 4 个 step 都累积），确认误差不是只在第一步。

**修法**：步数存进 param group（`pg["n_step"]`），由 `Optimizer._advance_step_count(pg)`
在 `step()` 里推进。存在 group 里有两个好处：跟着 `state_dict` 走；中途 `add_param_group`
的新组从第 1 步开始（补一个用例：训练 3 步后加一组，新组的首步必须等于全新优化器的首步）。

**反例**：只断言 loss 在降、或只跑一步不比较绝对值——都过。

## 2. 全局梯度裁剪只施加一次

`Optimizer.clip_grad_norm` 是**全局**的：它把所有 param group 的梯度拼起来算范数再统一缩放。
所以在 param group 循环里调用它 = 裁剪被施加 N 次，而且先被遍历的组是拿没裁过的梯度更新的。

**反例（抓不住）**：比较「分成 3 组」与「合成 1 组」之后的参数值。
Adam 系的更新 `m/sqrt(v)` 对梯度整体缩放**是不变的**（分子分母同比例），
多裁几次只是把梯度整体又缩小一点，参数更新几乎不变。用参数值对拍会得出「没问题」的错误结论。

**判据**：探针包住 `clip_grad_norm`，在 step 过程中观察：

```python
real = opt.clip_grad_norm
def spy(*a, **k):
    record["calls"] += 1
    out = real(*a, **k)
    record["norm_after"] = global_grad_norm(opt)   # 在 step 里读，不是 step 之后
    return out
opt.clip_grad_norm = spy      # 实例属性会遮蔽绑定方法
```

断言 `calls == 1`，且 `norm_after == max_grad_norm`。
把 `max_grad_norm` 取到和 `clip_grad_norm` 内部的 `1e-6` 稳定项同量级（例如 `1e-6`），
第二次裁剪就会把范数再砍一半，3 组时只剩 1/3，差异非常显眼；取 1.0 则第二次几乎是恒等，
测不出来。

## 3. step 之后不要再去读 `pg["grads"]`

`post_step` 每步都调 `zero_grad`，而 `zero_grad` 要真正把缓冲清零（否则
`opt.step(loss)` 后再 `opt.step()` 会把上一步的梯度再施加一次）。
**所以任何「step 之后读梯度」的断言都不成立**，要么在 step 过程中用探针读（见上），
要么在 `backward()` 与 `step()` 之间读。

`zero_grad` 清缓冲的性能顾虑可以打消：jittor 是惰性图，正常循环里这些 0 会被下一次
`backward` 覆盖、根本不会执行。量过：4096×4096 参数 + 每步 `sync_all` 的循环，
改前改后都落在 0.017~0.034 s/步的噪声区间里（这台机器上多 agent 并行，单次计时不可信，
至少跑 3 轮取最小值，并且改前改后各跑两轮交叉验证）。

## 4. lr scheduler：共享 lr 还是每组 lr

jittor 的 param group **默认没有 `"lr"` 键**，所有组回退到 `optimizer.lr` 这一个共享值。
在循环里「读 `optimizer.lr` → 改 `optimizer.lr`」会让 N 个组把它降成 `lr * factor**N`。

**判据**：同一个 scheduler，组数取 1/2/3 都必须得到同一个 lr。
构造时要**显式把优化器塞进 param group 的 `"lr"` 键删掉**：

```python
opt = jt.optim.SGD(params, 1.0)
for pg in opt.param_groups:
    pg.pop("lr", None)
opt.lr = 1.0
```

因为 torch 模式下装上的优化器会给每组塞一个 `"lr"`，不删就走到另一条分支，用例静默变成
「测了个别的东西」。

## 4b. 两套 scheduler 并存：找到那个「真正被用的 lr」

`jt.lr_scheduler.*`（旧式）与 `jt.optim.LRScheduler`（新式，LambdaLR 的基类）是两套。
**新式的构造函数会把 `"lr"` 塞进每个 param group 且从不移除**，而旧式的 `update_lr`
分支在「这个 group 有没有自己的 lr 键」上。于是「用过一次 LambdaLR」会永久改变旧式
scheduler 走哪条分支——行为取决于历史。

**唯一算数的观测量**是 `Optimizer.step` 实际读的那个值：

```python
def effective_lr(opt):
    return [float(pg.get("lr", opt.lr)) for pg in opt.param_groups]   # base.py 就是这么读的
```

**三个抓不住它的写法**（都试过，都是绿的）：

1. **比较「有没有先跑过 LambdaLR」两条轨迹的 effective lr**——用 `lambda e: 1.0` 时
   `pg["lr"]` 与 `opt.lr` 相等，两个存储各乘同一个 gamma，结果一模一样。
2. **比较 effective lr 的衰减比值**（`lr_k / lr_0` 是不是 `gamma**k`）——两个存储被乘
   的是同一个 gamma，比值恒对，即使两者的绝对值已经差了一倍。
3. **只改组数（1/2/3）比较轨迹**——只能抓到「一次衰减被施加 N 次」那类（第 4 节），
   抓不到两存储漂移。

**抓得住的判据**：让两个存储从**不同的基数**出发，再断言
`opt.lr == effective_lr(opt)`。

```python
opt = SGD([{"params":[p]} for p in ...], 1.0)
for pg in opt.param_groups: pg.pop("lr", None)     # 回到 jittor 默认布局
opt.lr = 1.0
jt.optim.LambdaLR(opt, lambda e: 0.5)              # 只构造，就已经埋下 pg["lr"]=0.5
sched = jt.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
for _ in range(3): sched.step()
# 改前：opt.lr == 0.25，而 effective == 0.125（训练用的是后者，打日志的人读的是前者）
assert all(abs(float(opt.lr) - lr) < 1e-9 for lr in effective_lr(opt))
```

`lambda e: 0.5` 是关键：identity lambda 会让两个存储始终相等，测不出来。

**另一条独立的判据**（针对同名方法语义不一致）：`update_lr()` 必须恰好施加一次
`get_lr()`。

```python
predicted = sched.get_lr()
assert predicted == sched.get_lr()      # 纯查询，问两次不能改 lr
sched.update_lr()
assert effective_lr(opt) == predicted
```

**注意配的是 `update_lr` 而不是 `step`**：一部分 scheduler 在 `step()` 里**先**
`last_epoch += 1` 再算，所以从外面调 `get_lr()` 拿到的是上一轮的值——torch 也是这样，
拿 `step()` 去配会得到一条假失败。

## 5. 不要在同一个 pytest 进程里混跑原生用例与 torch 兼容用例

`tests/conftest.py` 的 `pytest_ignore_collect` 会在**宽选择**（`pytest tests/`）时
把 torch 模式路径整个跳掉，两种语义各跑各的 session。
手动写 `pytest tests/compat/... tests/optim` 会强制进 torch 模式，
`jt.optim.*` 被整体换掉（`pg["grads"]` 变成 `_torch_grad`、每组自带 `lr`），
于是得到一堆与你的改动无关的失败。要么单独跑一个目录，要么用 `tools/run_test_suite.py`。

## 6. 真 PyTorch 对拍

本机没有与 jittor 同一个 Python 小版本的真 torch（jittor 环境 3.11、torch oracle 3.12），
`REAL_TORCH_SITE` 的进程内加载走不通。做法：
用 torch oracle 的解释器**另起一个子进程**导出参考数值，把公式（不是路径）固化进用例，
用 numpy 复现同一套公式做断言，并在提交说明里写清对拍用的 torch 版本。
