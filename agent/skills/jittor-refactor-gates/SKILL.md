---
name: jittor-refactor-gates
description: 在多个 agent 并行往同一分支推提交时，运行并判读 Jittor 的三套回归门禁（原生 CPU / CPU torch shim / CUDA nox），给出可信的红绿判断并把失败归因到具体提交。包含三套门禁的基线用例数、只看 EXIT=0 会漏掉的静默空跑、以及区分「真回归」与「缓存损坏 / 磁盘满 / 并发」的判据。
---

# 跑门禁与判读结果

门禁 agent 的产出只有一个：**一句可信的「这个提交是红还是绿」**。这件事难在假失败比真失败多——
缓存损坏、磁盘满、被 kill 的编译进程，表现和真回归一模一样。本 skill 是把这三类噪声排掉的固定流程。

## 1. 三套门禁是什么，基线是多少

三套都是「全量 `tests/` 的一个划分」，合起来覆盖一次。**必须核对用例数，不是只看 EXIT=0。**

下表是**分支起点**的数。整改期间每个 agent 都在加测试（「修静默算错必须先写一个修前失败的测试」），
所以 passed 会一路往上涨——**基线数是下界，不是等号**。判据是：

- `failed` 不超过起点已知失败清单（见第 6 节）——这是红绿的唯一判据；
- `passed` **不下降**。涨了是正常的（有人加了测试）；跌了说明用例被删了或被静默过滤掉了，要查。

（实测：起点 775 passed 的原生门禁，第一轮并行推进后是 822 passed / 816 skipped / 0 failed，
收集总数从 1540 涨到 1638。这是绿。）

| 门禁 | 范围 | 基线 | 耗时（全核 / 16 核实测） |
| --- | --- | --- | --- |
| 原生 CPU | `tests` 减 `compat/torch`、`ops/test_ops.py`、`core/test_regression.py`、`structure`、`backends/triton/test_triton_torch_compat.py` | **775 passed, 765 skipped** | 9 分 / 约 50 分 |
| CPU torch shim | 上面被减掉的那些，加 `compat/vllm` | **1595 passed, 285 skipped** | 3 分 / 约 8 分 |
| CUDA nox | `nox -s cuda`，6 次 pytest 调用 | **574 passed, 9 skipped, 0 failed** | 2 小时 / 更久 |

原生与 torch 的差别只有两个环境变量：`JITTOR_TORCH_SHIM=0/1`。两套的测试集**互斥且互补**——
把一个用例从一套挪到另一套，两边的基线数会同时变，只对其中一套是判不出来的。

CUDA 那套的 6 次调用各自的基线，缺一不可：

| 调用 | 基线 |
| --- | --- |
| `tests/backends/cuda` | 112 passed, 1 skipped |
| `tests/backends/parity/test_dtype_coverage.py` | 6 passed |
| `tests/backends/parity/test_device_parity.py` | 227 passed（约 1 小时 50 分，占整套的 80%） |
| `tests/compat/torch/test_torch_compat_cuda_tf32.py` | 2 passed |
| `tests/ops/test_ops.py` | 227 passed |
| `tests/models/test_network_training_parity.py` | **8 skipped，零 passed** |

最后一条现在就是全 skip，这是**已接受的基线**，不要当成红。但它同时是一个哨兵：
如果哪天它变成 `0 collected`，说明有人把设备过滤或收集规则改坏了，那是真回归。

## 2. 怎么起

三套各自一个脚本，各自 `flock` 防重入、各自 `timeout`、结束写哨兵文件。共同要素：

```bash
export JITTOR_HOME=<自己名下>/native      # 三套三个目录，不共用
export TMPDIR=<自己名下>/tmp-native
export nvcc_path="" JITTOR_TEST_DEVICES=cpu REAL_TORCH_SITE="" JITTOR_TORCH_SHIM=0
export use_parallel_op_compiler=0 gdb_path=""
timeout 4200 taskset -c <自己的核段> <jt311>/bin/python -m pytest tests \
  --ignore=... -p no:cacheprovider --timeout=900 -q -rf > native.log 2>&1
echo "EXIT=$?" >> native.log
echo done > native.done
```

torch 那套把 `JITTOR_TORCH_SHIM` 改成 `1`，测试集换成互补的那半。

CUDA 那套走 `nox -s cuda`，有一个坑：**nox 的 `_session_env` 不认你导出的 `JITTOR_HOME`**，
它按 `JITTOR_LAB_ROOT` 自己造一个（还会 `rmtree` 重建）。所以隔离靠的是

```bash
export JITTOR_LAB_ROOT=<自己名下>/lab     # 不设就会和别人抢同一个 JITTOR_HOME
export JITTOR_CI_PYTHON=<jt311>/bin/python
export nvcc_path=/usr/local/cuda/bin/nvcc PATH=/usr/local/cuda/bin:$PATH
```

`nox.options.envdir` 也挂在 `JITTOR_LAB_ROOT` 下，所以设了它顺带把 venv 也隔离了。

## 3. 报「通过」之前必须过的三道检查

**EXIT=0 不等于通过。** 按顺序：

1. **用例数对得上**。设备过滤配置错一个字符就静默收集到零个用例，`0 passed` 照样 EXIT=0。
   passed 只许涨不许跌；passed 不变而 skipped 掉了一大截，说明用例被删了而不是被修好了。
   两个数都要记进看板，只记 passed 看不出后一种。
2. **哨兵文件在**。没有 `.done` 说明脚本没跑完（被 timeout 杀了、或者被别的东西打断），
   这时候日志里可能已经有一段漂亮的 passed，但那只是中途快照。**超时按「未完成」报，不要按通过报。**
3. **日志尾部有 pytest 的汇总行**。段错误会让进程死在半路，日志停在一串点上，没有汇总行。

## 4. 假失败的判据

真回归的形状是**集中**的：同一个文件、同一族算子、和某个提交改的东西对得上。
下面几类的形状是**散布**的，看着吓人但和代码无关。

### 磁盘满

- 判据：**跑之前先 `df -h`**。低于 30G 就不要跑，报警。
- 表现：散布在毫不相干的算子上的失败，外加一个段错误。和真回归、和缓存损坏都一模一样，
  事后从日志区分不出来，所以只能事前查。
- 十来个 agent 各建各的 JITTOR_HOME，一个 JITTOR_HOME 就是 5–10G，掉得很快。

### 缓存损坏

- 成因：**`kill -9` 了正在编译的 Jittor 进程**。留下半截 `.so`，下一轮在毫不相干的算子上
  大面积报梯度不符（实测 165 failed / 62 passed，全是幻觉）。
- 判据：失败数大到不成比例（三位数），且失败集中在「梯度不符」这一种断言上。
- 处理：**把整个 `JITTOR_HOME` 删掉重跑**。不要试图定位——里面没有真信息。
- 预防：要杀门禁就先删 `JITTOR_HOME` 再重跑，不要 `kill -9` 了就直接再跑一遍。

### 并发污染

- 成因：两个进程共用一个 `JITTOR_HOME` 或 `TMPDIR`。
- 判据：同一个提交跑两遍结果不一样；或者失败的用例和别的 agent 正在改的模块对得上，
  但你的 worktree 里那个文件根本没变。
- 处理：确认三套门禁各有各的 `JITTOR_HOME`，CUDA 那套确认 `JITTOR_LAB_ROOT` 也是自己的。

**独立的 `JITTOR_HOME` 还不够：两个 jittor 进程同时冷启动会互相污染 cuda key。** 实测两棵 worktree
各自 `JITTOR_HOME`、同时起，`compiler.py:1062` 算出的 cuda key 变成

```
cu12.2.140_..._sm_0902_232043.337919_48_89_Create_[i_file..../jittor.lock_lock_lock.py85]
```

——**另一个进程的日志行被当成算力值读了进来**，接着 `make_cache_dir` 拿这个含 `/` 的字符串建目录，
报 `FileNotFoundError`，两棵树双双 `1 error in 0.36s`。这看着像"两个提交都挂了"，其实一次都没跑。

判据：日志里 `cuda key:` 那行不是干净的 `sm_89`。**做 A/B 对照一律串行**，别为省时间并排跑。

### 串号：提交里混进了别人的改动

- 成因：`git stash` 的栈存在公共的 `.git` 里，**所有 worktree 共用一个栈，worktree 不隔离它**。
  A 分区 `stash`、B 分区 `stash pop`，B 就把 A 的 WIP pop 进了自己的树，一起提交上来。
  （真实发生过：两个分区的 WIP 被完整对调。）
- 判据：**一个提交改的文件和它提交说明里的 `[编号]` 对不上**——出现了与该任务毫无关系的文件。
- 处理：这不是逻辑回归，是串号。通知**两个**当事 agent（改动的来源方和提交方），不要去读代码找 bug。
- 预防（对所有 agent）：不要用 `git stash`；真要临时搁置就 `git diff > 自己目录下的 .patch` 再
  `git checkout --`。门禁 agent 自己只对**自己的** worktree 做 `git reset --hard`，
  绝不对别人的 worktree 做任何 git 写操作。

### 还有一类：跑挂了不动

比失败更常见也更难发现——一个门禁卡住几小时，没有输出也没有退出。
所以每套都要有 `timeout`，并且**只信哨兵文件，不信日志里最后那行 passed**。

## 5. 等长任务不要用 pgrep

`pgrep -f gate-native` 会匹配到**你自己那个正在等待的 shell**，也会匹配到 heredoc 里的脚本文本，
于是永远「还在跑」。用哨兵文件：

```bash
until [ -f "$G/fast.done" ]; do sleep 20; done
```

要判断「是否正在跑」而不是「是否跑完」，用锁本身，不要用进程名：

```bash
if flock -n 9 9>$G/native.lock; then echo "没在跑"; else echo "在跑"; fi
```

## 6. 起点已知失败：报回归之前先减掉它

**分支起点本身不是全绿的。** 有些失败是从起点继承来的，把它算成回归就会把责任安到无辜的提交上。
所以「红」的定义不是 `failed > 0`，而是 **`failed` 比起点多**。

维护一份起点已知失败清单，每次报告前先减掉：

| 用例 | 症状 | 状态 |
| --- | --- | --- |
| `tests/compat/torch/test_torch_compat.py` | 段错误 | 分支起点就存在，非任何 agent 引入 |
| `tests/compiler/test_atomic_tuner.py::test_atomic_tuner` | 第 4 项抓到 0 条 `to loop -1` 日志 | 起点就存在；根因 `032ecfe1` 全归约快路径绕开 JIT，是过期断言 |

对照起点的办法（`git stash` 已禁用，别用它切来切去）：另开一个只读的 worktree 钉在起点上，
在里面跑同一条用例：

```bash
git worktree add /自己目录/baseline origin/2.0     # 起点，只读，不在里面提交
# 用它自己的 JITTOR_HOME 跑那一条失败用例，对比结论
```

这个起点树只用来对照，跑完留着，下次还用得上——重建一次要重编十分钟。

## 7. 比 bisect 便宜的手法：rebase 前后对拍

`git bisect` 是最后手段，一轮几十分钟。绝大多数归因用不着它：

**执行 agent 自己就能做**——rebase 前跑一遍受影响目录，rebase 后再跑一遍，对比 failed 数。
差出来的就是别人在这段区间里引入的；没差出来就是自己引入的。这比 bisect 便宜一个数量级，
而且不需要门禁 agent 介入。发现红之后，先问对应分区的 agent 做这个对拍，再考虑自己 bisect。

**门禁 agent 这边的等价物**：同一套门禁，把上一次绿的提交和现在这个提交各跑一遍**失败的那几条用例**
（不是整套）。两个提交号之间通常只有几个提交，`git log --oneline` 一眼就能看出嫌疑人。

## 8. 红了怎么归因

并行推提交的前提下，红了不要求别人停推，代价由你承担：

1. 先看失败用例涉及哪些文件，`git log --oneline <上一次绿的提交>..HEAD -- <那些文件>`。
   提交说明以 `[编号]` 开头，编号直接对应看板上的任务和负责人，多数情况一步到位。
2. **锁定嫌疑提交后，第一件事是 `git show --stat` 核对它改的文件与 `[编号]` 是否相符。**
   不符就先怀疑串号（见上一节），不要怀疑逻辑——去读那段代码是白费功夫。
3. 一步定不了再 `git bisect`，`git bisect run` 的脚本只跑失败的那几个用例，不要跑整套。
4. **定位到就停手，不要自己修**。把责任写进看板那一行，通知对应的 agent。

判成红之前先回到第 3 节那三道检查、第 4 节那几类假失败、第 6 节的起点已知失败清单——
**在并行环境里，第一次看到的红有很大概率不是回归。**同一个提交复现两次再下结论。

## 9. 被点名的提交不等于有罪：先看 pass 顺序再跑对照

实例：`test_atomic_tuner` 抓不到 `atomictuner:` 日志，嫌疑指向 `9eb696d9`（新增 WarpReducePass，
把 `atomicAdd(...);` 改写成复合块）。**读一眼 `pass_manager.cc` 就能排除**：WarpReducePass 挂在
AtomicTunerPass **之后**，原子调优早已打完日志才轮到它改写。跑 `9eb696d9^` 对照，失败逐字一致，坐实。

真正的原因在别处：`032ecfe1` 给 CUDA 全量归约装了 `nn/backends/full_reduce_cuda.py` 的 cub 两级折叠
快路径，`x.sum()` 从此不进融合算子 JIT，AtomicTunerPass 连一条 atomic 语句都看不到。

三条可复用的手法：

1. **先读顺序，再跑对照。**一次 `pass_manager.cc` 的阅读能省掉两次冷编译。
2. **看用例是几项挂的。**这个用例前三项（reindex_reduce 的 add/max/min）一直是通过的，只有第 4 项
   （全归约）挂——"整个用例红了"和"用例里某一项红了"是完全不同的线索，前者才指向 pass，
   后者指向那一项碰到的那条代码路径。
3. **JIT 缓存目录就是证据。**`$JITTOR_HOME/.../jit/` 下的文件名带算子键。该用例跑完那里只有
   `reindex_reduce` 和 `code__IN_SIZE_1...` 两族，没有 `reduce`——全归约被 code op 接管了，
   一眼可见，不用去读被混淆的 `data.cc`。

顺带一条：`opt/pass/` 下 `atomic_tuner_pass.cc`、`parallel_pass.cc`、`shared_reduce_pass.cc`
**git 里根本没有**（2020 年 `8f316a2e` 删的），实现藏在 `python/jittor/utils/data.gz` 解压出的
`data.cc` 里，编译时 `-include vdp`。所以这三个 pass 的日志 `__FILE__` 是 `data.cc`——
`log_vprefix` 要写 `data=100` 才抓得到，写 `atomic=100` 抓不到。这就是任务 1.01 要还原的东西。
