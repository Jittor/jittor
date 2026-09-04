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
