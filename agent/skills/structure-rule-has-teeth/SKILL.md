---
name: structure-rule-has-teeth
description: 把 tests/structure 里的「精确清单」断言改写成「规则」之后（任务 0.19 要求的那类改写），用反例证明新规则不是空的。凡是让一条结构测试从红变绿的改动都该用它——包括放宽白名单、删除字节/行数冻结、把断言从字符串拼写换成 AST 形状。
---

# 让结构测试从红变绿有两种方式，其中一种是作弊

任务 0.19 要求把 `tests/structure` 的精确清单改成规则。这类改写每一次都让一条红的测试变绿，
而**「规则写对了」和「规则被写空了」在结果上一模一样**：两者都是绿的。

所以改写的验收不是「它绿了」，而是**「造反例它还会红」**。少了这一步，一次改写就等于悄悄
删掉一道门禁——而且比直接删更糟，因为看板上它还记着「已合并」。

## 判据

每条改写后的规则，至少造两个反例：

1. **原本要抓的东西**——把那条测试当初为之存在的违规重新造出来（例如：facade 里定义了函数、
   子进程没钉 `PYTHONPATH`、清单漏了一个打包资源）。
2. **改写新引入的边界**——你的规则比旧断言宽在哪里，就在那个位置造一个。例如把
   「`assertIn("from .runtime import enable", source)`」换成「`__all__` 的每个名字都来自再导出」
   之后，反例是「`__all__` 里加一个既没 import 也没 alias 的名字」。

两个反例都必须报红。有一个不报，规则在那个方向上是空的。

## 做法

反例要**改真实文件、跑真实 nodeid、然后还原**，不要用 mock 或临时目录——被测的正是真实文件的形状。
用 `git checkout --` 还原，不要用 `git stash`（stash 栈是所有 worktree 共用的）。

```bash
cd "$JITTOR_LAB_ROOT/refactor/<分区>"

probe () {  # probe <名字> <nodeid>
  local out
  out=$(JITTOR_HOME="$JITTOR_LAB_ROOT/refactor/_home/<分区>" \
        TMPDIR="$JITTOR_LAB_ROOT/refactor/_tmp/<分区>" \
        JITTOR_TORCH_SHIM=1 JITTOR_TEST_DEVICES=cpu nvcc_path="" \
        python -m pytest "$2" -q 2>&1 | grep -cE "^1 failed|1 failed,")
  [ "$out" -ge 1 ] && echo "  [有牙] $1" || echo "  [!! 空规则] $1"
}

# 每个反例：改 → probe → 立刻还原
echo 'def defined_here(): return 1' >> <被规则约束的文件>
probe "facade 里有定义" "tests/structure/<文件>::<类>::<方法>"
git checkout -- <被规则约束的文件>

# ...其余反例同构

git status --short   # 必须只剩你自己改的测试文件
```

最后那句 `git status --short` 是这个流程唯一容易漏的一步：反例改的是**产品文件**，忘了还原就会
被下一次 `git commit <路径>` 带走，或者更糟——被别人的 rebase 当成你的改动。

### 先提交，再造反例

`git checkout -- <文件>` 还原到的是 **HEAD**，不是"你造反例之前的样子"。所以如果反例要改的文件
**同时也带着你自己未提交的改动**，还原会把你的改动一起删掉，而且悄无声息——脚本照常打印
"[有牙]"，`git status` 只是少了一个文件名。

顺序因此是硬性的：**先把真实改动提交，再跑反例脚本**。已经发生过一次：`__init__.py` 上的
改动（把两个 helper 搬出根）在第三个反例还原时被抹掉，5 个反例全部报告有牙，然后才发现产品
改动没了。

如果确实必须在提交前跑，就不要用 `git checkout --`：先 `cp <文件> $TMPDIR/<文件>.keep`，
反例后 `cp` 回来。**不要用 `git stash`**（栈是所有 worktree 共用的）。

## 提交说明里写什么

把两件事都写进去，否则复查的人无法判断这次改写是不是放水：

- 旧断言**为什么**失效（哪个任务改变了被断言的形状，例如「7.04 把三条激活入口收敛成
  `activate()`」）；
- 反例清单与结果（「造 6 个反例，全部报红」）。

## 一个信号：豁免清单在增长

发现被改写的测试带着一份「例外路径」集合，且集合里不止一两项，那它几乎肯定是精确清单而不是规则——
**每次合法编辑加一条豁免**就是它的运行方式。`test_torch_shim_structure` 的字节 manifest 攒到 5 条
豁免时，36 个文件里已有 7 个哈希漂移，而仍在报红的两个是一份文档和一个在改的 `.cu`。
遇到这种，先问「这条测试当初要防的是什么」，再只保留那件事。

## 豁免清单什么时候反而是对的：分类闭集

上一节说豁免清单在增长是坏味道，但有一种清单是**目的本身**：当一个问题的形状是
「全树有 N 处写同一种全进程状态，其中一部分必须进事务、一部分有正当理由不进」，
那么「哪些还没做」这个问题本身就必须被钉住。这种情况下 grep 是不够的——
**grep 说的是「什么匹配上了」，不是「还剩什么」**。7.05 的看板连着十一波记着
「其余 installer 的写入口仍待」，每一波都是一次新的 grep、每一波结论都不一样。

区别在三条，缺一条就退化成上一节那种攒豁免的清单：

1. **闭集**：扫描器发现的每一项都必须在表里，不在就红。表里有而树里已经没有的，
   也必须红（陈旧豁免正是闭集失效的方式）。两个方向都要造反例。
2. **每一项带类别，而且类别数量很少**。类别不是「允许」，是**为什么不属于这本账**：
   `ledger` / `runtime`（调用方在安装之后自己要的）/ `pre-ledger`（事务还不存在，
   例如 core import 之前的 preflight）/ `deployed-payload`（另一个进程）/ `pending`。
3. **`pending` 单独一份，写的是障碍而不是文件名**。做完一项，这份清单在同一个
   diff 里变短——于是「还剩什么」不再依赖看板上谁记得更新。

```python
PENDING = {
    "…/external_backend.py":
        "source-root import 用整表快照恢复 sys.path/sys.modules，会丢弃并发写者的"
        "条目而不是报告；需要 owner-aware 条目或子进程隔离",
}

def test_pending_names_exactly_the_unfinished_files():
    assert {p for (p, _o, _k), c in CLASSIFIED.items() if c == "pending"} == set(PENDING)
```

键要选**「所在的那个 def」而不是行号**：函数内部挪代码不该让表抖动，而把一处写入挪到
*另一个* 函数里恰好是值得重新审的那种变化。

扫描器本身也会误报，而误报会被当成「又发现一处」写进表里。真实踩到过：把
`insert`/`append` 也算成模块表写入，于是 `modules.insert(0, self)`——一个装子模块的
普通 list——被报成 `sys.modules` 写入。`sys.modules` 是 dict，没有 `insert`；
**按 owner 的真实类型分别给动词**（dict 一套、list 一套），不要一张动词表打天下。
