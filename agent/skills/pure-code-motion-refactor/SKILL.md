---
name: pure-code-motion-refactor
description: 把一个大函数拆成几段、或把一段代码搬到另一个翻译单元时，怎么证明「行为逐字不变」。包含归一化文本比对（比测试强的证据）、让工作树在长测试跑着时仍可编辑的快照法、「基线失败集合逐条相同」而不是「全绿」的判据，以及搬代码会静默掏空按文件名点名的结构门禁这个坑。改 executor.cc / grad.cc / op_compiler.cc 这类大文件之前读。
---

# 证明一次纯搬运没有改变行为

整改计划里有一批任务是「把 X 拆成 A 和 B」（3.01 的 Planner/Runner、3.20、5.02、11.04）。
这类改动的验收不是「测试绿了」——**被拆的代码往往正是测试覆盖不到的那部分**，而且
一个大函数的分支组合远多于任何测试集。真正的判据是**逐字不变**，而那是可以直接证明的。

    10|## 1. 先取一棵改动前的树，别只靠 git

```bash
git archive HEAD | tar -x -C $SC/pristine      # 整棵树 20MB 级
```

这棵树有三个用途，缺一不可：

- **文本比对的左边**（下一节）；
- **基线测试的跑法**（第 3 节）——它有自己的 `JITTOR_HOME`，跟你的工作树不抢缓存；
    20|- 出现「这条红是不是我造成的」时的对照组。

**不要用 `git stash`**（栈是全仓库共用的，见 `git-worktree-shared-state`），也不要靠
「等会儿 `git checkout --` 回去」——你会在中途需要同时看到两个版本。

### 例外：CUDA 基线不能用新目录，必须复用预热的 `JITTOR_HOME`

上面那棵副本树跑 CPU 基线没问题，**跑 CUDA 基线会失败**，而且失败信息指向别处：

```
Check failed ret(256) == 0(0) Run cmd failed: "nvcc" ".../nan_checker.cu" ...
WARN third-party archives 3 still to download: dnnl_lnx_2.2.0_cpu_gomp.tgz, cub-1.11.0.tgz, cutt-1.2.zip
```

**同一条 nvcc 命令单独手跑是成功的**，所以很容易被当成自己改坏了或机器坏了。真正的原因是
一次**冷** CUDA 构建要拉三个三方包，而这台机器没有网络出口。能带 CUDA 构建的只有别人
预热过的那个 `JITTOR_HOME`，而缓存目录名里带源码内容哈希——副本树换个路径就是冷的。

所以 CUDA 的 A/B 要这样做：**把工作树临时切回基线提交，复用同一个 `JITTOR_HOME` 跑，
再切回来**，并且把还原放进 `trap`：

```bash
restore() { for f in $NEW; do cp "$ASIDE/$(basename $f)" "$f"; done
            git checkout "$MINE" -- $OLD; }
trap restore EXIT
for f in $NEW; do cp "$f" "$ASIDE/"; rm -f "$f"; done   # 新增的文件先挪走
git checkout "$BASE" -- $OLD
<跑 CUDA 门禁>
```

`trap` 不是讲究，是必需：这一段跑三十分钟，中间任何异常退出都会把工作树**留在基线状态**，
而你的改动只存在于 `$ASIDE` 里。代价是这段时间不能改代码（第 3 节的快照法在这里用不了，
因为快照树的缓存是冷的）。

## 2. 归一化文本比对：搬运的判据

搬一段代码时，你会顺手做几件事：把局部变量换成结构体字段、把 `this->x` 换成 `exe.x`、
补几行注释。**除此之外的任何差异都是 bug**，而这一点可以用一个二十行的脚本证明：

    30|```python
def norm(s):
    s = re.sub(r'//[^\n]*', '', s)      # 注释不算行为
    return re.sub(r'\s+', ' ', s).strip()

o = norm(region(orig, "起始标记", "结束标记"))
n = norm(region(new,  "起始标记", "结束标记"))
for a, b in RENAMES:                     # 把「打算做的改名」施加到*原文*上
    o = o.replace(norm(a), norm(b))
assert o == n
    40|```

**关键在最后那步的方向**：把改名施加到**旧代码**上，然后要求与新代码逐字相等。反过来
（把新代码改回去再比）会让你不知不觉把真实差异也「改回去」。`RENAMES` 那张表就是这次
改动的完整声明——表里没有的差异，脚本会打出来。

失败时用 `difflib.unified_diff(o.split(' '), n.split(' '))` 按词打印，比按行清楚得多：
搬运造成的差异通常是一个词（少了个 `weak_sync &&`），按行 diff 会把整段贴出来。

实测价值：3.01 拆 `run_sync` 时，这个脚本在第一轮就抓到 BFS 里漏掉的 `weak_sync ||`
    50|——一个只在 `weak_sync=false`（`submit_pending` 的自动冲刷路径）下才走到的分支，
CPU 门禁 900 条用例一条都不会红。

**这一步在编译之前做。** 它比一轮重编快一千倍，而且它证明的东西比测试强。

## 3. 长测试跑着的时候怎么继续改代码：快照，不要等

`jittor-core-cpp-edit-loop` §5b 说「别在别人跑着测试的时候动 `python/jittor/src/**`」，
因为跑着的测试会起子进程重编核心。但拆大函数时**你自己就是那个跑测试的人**，一轮
`tests/core` 二十分钟，串行等下去这类任务做不完。

    60|做法是**把要验证的状态快照出去跑，工作树留给自己改**：

```bash
git ls-files -z | tar --null -T - -cf - | tar -x -C $SC/s2
git ls-files --others --exclude-standard | while read -r f; do
  mkdir -p "$SC/s2/$(dirname "$f")"; cp "$f" "$SC/s2/$f"; done   # 新增的文件
setsid $SC/gates.sh s2 $SC/s2 $SC/s2_home $SC/s2_tmp
```

**第二条命令不能省。** 拆函数几乎总会新增 `.cc`/`.h`，它们还没进索引，
    70|`git ls-files` 看不见；漏掉的表现是快照里编译单元少一个、链接报未定义符号，
而你会以为是自己的改动写错了。

代价是每个快照一次冷编（CPU-only 约 40 秒）加约 350MB。收益是**每一步中间提交都能有
自己的一轮完整门禁**，而不是最后攒一个大的。

## 4. 判据是「失败集合逐条相同」，不是「全绿」

这棵树的 `tests/core` 有二十来条既存失败（多数属「存活 Var 与内存记账」一簇）。
拿「全绿」当判据的话这类任务永远交不了差，拿「failed 数相同」当判据又会漏掉
「修好一条、弄坏另一条」。**正确的判据是集合相等**：
    80|
```bash
diff <(grep "^FAILED" base-core.log | sort) <(grep "^FAILED" after-core.log | sort)
```

跑基线时 `-rf` 不能省，否则日志里只有计数没有名字。

顺带：`tests/structure` 属于 **torch shim** 那套门禁（`JITTOR_TORCH_SHIM=1`），
原生模式下 conftest 会整片丢掉它，`pytest tests/structure` 安静地收集到零个用例并
以 `EXIT=5` 退出。**把它当成「结构门禁通过了」是这条路上最容易犯的错**——
判据见 `jittor-refactor-gates` §3.1。
    90|
## 5. 搬代码会静默掏空按文件名点名的门禁

结构门禁里有一类断言是**按文件名列清单**的。`tests/structure/test_shared_backend_consumers.py`
拿一张 `CONSUMERS` 表逐个 `g++ -fsyntax-only -DHAS_ACCELERATOR`，证明这些文件不依赖任何
厂商 SDK 头。表里有 `executor.cc`。

把 `executor.cc` 里所有碰 `migrate_to_gpu` / `sync_devices` / `event_queue` 的代码搬到
`exec_runner.cc` 之后，**这条门禁仍然通过，但它守护的代码已经不在它的清单里了**——
`executor.cc` 剩下的部分本来就不碰这些东西。测试没有变红，它只是变得没有意义
（`jittor-refactor-gates` §10 讲的就是这类失效）。
   100|
所以：**搬代码之前先 grep 谁按文件名点了它的名**。

```bash
grep -rn "executor\.cc\|executor\.h" tests/structure
```

命中的每一处都要问「这条断言想守护的是这个文件，还是这个文件里的那段代码」。
答案是后者的，把新文件加进清单，**和搬运放在同一个提交里**。

## 6. 中间提交的证据链要写清楚，不要含糊
   110|
拆成 N 个提交时，通常不会给每一个中间状态都跑一遍完整门禁（一轮半小时）。
这没关系，但**提交说明里要写清这一条是被哪一轮覆盖的**：

- 只加注释和头文件契约的那条：说明「本提交不含可执行代码改动」，并给出 `git diff --stat`；
- 搬运那条：给出第 2 节的比对结果（「归一化后与原文逐字相同，改名表见下」）**加**它自己那轮门禁；
- 若干条合起来跑了一轮的：说明是哪几条一起跑的，别写成每条都跑过。

含糊其辞的代价是下一个人重跑一遍；写清楚的代价是三行字。
