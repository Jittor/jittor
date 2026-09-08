---
name: static-launch-equivalence
description: 在没有目标硬件的机器上证明「去样板前后生成的调用等价」。做任何把 N 处重复的设备调用尾巴收进共享 helper 的重构（ACL 的 executeOp→BaseOpRunner::launch、cuDNN/cuBLAS 的 checkX 尾巴、HCCL 宏）都用它；也用于回答「这次重构到底改了行为没有」，避免把「删了样板」和「顺手改了失败语义」混成一个绿色提交。
---

# 用 token 流证明去样板没改行为

去样板的重构有一个特有的失效模式：**样板里那 N 份拷贝并不完全一样**。合并的时候，
其中三五处的差异会被静默吃掉，而 diff 看着就是「N 处重复变成一处」，reviewer 没法逐处比。
没有目标硬件时更糟——跑不了，字符串合同又只能证明源码长什么样，不能证明设备会看到什么。

办法是把每个调用点归约成**设备会观察到的东西**的有序 token 流，然后拿改前的树和改后的树
逐点对 diff。token 流相同 ⇒ 等价；不同 ⇒ 差异被强制显示成具体调用点的具体 token，逐条解释。

参考实现：`tools/build/acl_launch_program.py`（ACL 的 `executeOp`）。用法是两个树当参数：

```bash
git worktree add --detach $TMPDIR/before <去样板之前的提交>
python tools/build/acl_launch_program.py $TMPDIR/before .   # 退出码非 0 即有差异
```

实现归仓库的 `tools/build/`，本 skill 保留使用方法。`agent/` 只维护操作手册与
skills，不放另一份公共仓库检查脚本。

## 怎么选 token

只留**设备会区分**的东西，其它一概丢掉。ACL 那份留了五类：

| token | 含义 |
| --- | --- |
| `query <fn>` | workspace 尺寸查询调了哪个入口 |
| `queryfail <fatal\|return\|throw\|unchecked>` | 查询失败怎么处理 |
| `exec <fn>` | 真正发出去的 execute 入口 |
| `execfail <…>` | execute 失败怎么处理 |
| `sync <0\|1>` | 之后有没有同步 |

丢掉的：描述符构造、RAII、workspace 记账、日志文本、代码顺序里的空行。这些恰好是共享 helper
接手之后**每个调用点都一样**的部分，留着只会产生噪声 diff。

**判断标准是「设备能不能区分」，不是「源码像不像」。** 举两个真实的坑：

- 三目运算 `ret = c ? aclnnA(...) : aclnnB(...)` 和 `Launcher f = c ? aclnnA : aclnnB; launch(ret, f)`
  是同一件事。不把它折叠成一个 `exec aclnnA|aclnnB` 事件，71 个 owner 里会凭空冒出两条假差异。
- 函数体末尾一个 `syncRun();` 是 switch 里**每个 case** 的尾巴。按「这个 exec 之后是否存在
  syncRun」判定，而不是按「这一行紧接着谁」，否则整个 switch 的同步策略会全判错。

## 必须做的一步：让「短路」赢

样板里最常见的是这种形状：

```cpp
ret = fooGetWorkspaceSize(...);
CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("failed\n"); return);   // ← 打印后 return
launch(ret, foo, true);                                          // ← launch 里也会检查
```

第一版提取器看到 `launch(...)` 就把 `queryfail` 记成 `fatal`，于是删掉那行 `CHECK_RET` 显示成
**零差异**——而它明明把「留下未初始化的输出继续跑」改成了「抛」。
**调用点上会 `return`/`throw` 的处理器先执行，它才是有效策略。** 提取器必须让它覆盖掉
helper 的策略，否则这个工具在最该报警的地方沉默。

## 怎么确认提取器真的在起作用

跑出「零差异」不算数，先做反向对照。至少这三个，每个都必须让 diff 非空：

1. 把某个调用点的 launcher 换成另一个真实入口（`exec` 应当变）。
2. 把某个 `launch(ret, f, true)` 改成 `false`（`sync` 应当变）。
3. 把删掉的 `CHECK_RET(..., return)` 放回去一处（`queryfail` 应当变——这条就是上面那节的判据）。

## 配套的门禁写成什么形状

等价性证明是一次性的审计，门禁要防的是**样板长回来**。两条：

- **写成不变量，不要写成计数。** 「共有 65 处调用 `checkRet`」这种合同在第一个合法重构之后就
  作废；本仓真发生过，然后红穿约四十个提交没人看见。要断言的是「没有任何调用点自己发 execute
  调用 / 自己分配 workspace / 自己处理查询失败」。
- **让门禁自证扫到了东西。** 断言「每个扫描根各自非空」，不要断言「总数 > N」。半途搬迁的目录
  结构下（本仓 ACL 同时存在 `python/jittor/extern/acl` 与 `backends/acl`），只扫到一侧的门禁
  看起来和健康的门禁一模一样。参考 `tests/_helpers/acl_launch_tails.py` 的 `populated_roots`。

## 这个方法证明不了什么

它比较的是**调用序列**，不是**实参**。传错张量个数、顺序、类型，token 流照样相同。
所以它和桩 SDK 的 TU 检查（`agent/skills/acl-host-syntax-check`）是互补的两层，
两层都过也**不是硬件验证**——设备侧留什么写进 `agent/manuals/deferred-hardware.md`。
