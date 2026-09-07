# 布局收尾交接：4.15 与它带出的三条，以及一个挡住冷构建的前置

写给接手布局链的人。2026-09-07 由协调者执行，做到哪、卡在哪、哪些坑已经付过代价，
下面按「你第一件事该做什么」排。

## 0. 你第一件事：解掉 preflight 对冷构建的拒绝

**现状：本机上任何冷构建都会被 preflight 拒掉，与本次搬动无关，但会挡住你的验证。**

```
FAIL network   cannot reach cg.cs.tsinghua.edu.cn: [Errno 101] Network is unreachable
WARN third-party archives 1 still to download: cub-1.11.0.tgz
```

机制：`jittor_utils/preflight.py:check_third_party` **无条件**把 MKL、CUB、CUTT 三个
归档列为 wanted（`:200-206`），不看这次构建是否用得到；`check_network` 又以
`needed=third_party.status != "ok"` 触发（`:308`），于是「有任何归档缺失」加「那个主机
不可达」等于硬失败。本机 `cub-1.11.0.tgz` 从未下载过（全盘找不到），而
`cg.cs.tsinghua.edu.cn` 不可达——**注意 `github.com` 是通的**，所以不是全局断网，是那个
主机的问题。

**为什么以前没人撞到**：`51d0439f` 给核心编译加了构建戳，戳一致时 `build_core()` 整步
跳过，**preflight 根本不会跑**。所有人的缓存都是热的，所以这条前置一直没被执行过。
我这次搬了 360+ 个文件使戳失效，触发真实构建，它才第一次运行。

**这与 `9.01` 的验收「离线只读环境能 import」矛盾**——那条是在热缓存下验的。所以
`9.01` 虽已标已合并，它的离线那一半实际上没有被冷构建证明过。**这是一条应当登记的
新发现**，我没有改 `9.01` 的状态（它的另一半「热缓存 import < 1 s」是实测达成的）。

三条可走的路，按我建议的顺序：

1. **让 `check_third_party` 只要这次构建真需要的东西**。cub 确实要下载
   （`install_cub` 从 `manifest.CUB` 取 tgz 解压，`-I` 指向解压出来的那份），
   但**只有 CUDA 路径用得到它**；`backends/cuda/libraries/cub/include/` 下只有一个
   `cub_test.h`（从 cub 抠出来的片段），不是内置的完整库。问题因此不在「已内置还要」，
   而在**对 CPU-only 构建也无条件要**：`nvcc_path=""` 的构建从头到尾不碰 cub，却被它
   拦住。判据应当跟着「这次构建选了哪个后端」走。这条最正确，也一并解决离线安装。

   （**更正**：本文档第一版写的是「树里已内置 cub 头，所以要归档是错的」。那是错的，
   我把 `libraries/cub/include` 里那一个 `cub_test.h` 当成了完整库；实测
   `find -name "*.cuh"` 在该目录下零命中。结论方向不变，理由换成上面这条。）
2. 用文档给的 `JITTOR_OFFLINE_PATH` 指向一个装着三个归档的目录。**本机没有 cub 归档**，
   要先在能联网的机器上 `nox -s prefetch`。
3. 只在 CUDA 配置下验证（CUDA 侧的库都在缓存里）。**不推荐**：CPU-only 那一路就不会被
   任何冷构建覆盖，而它正是出问题的那一路。

## 1. 已经做完并推上 `2.0-refactor` 的

| 提交 | 内容 |
| --- | --- |
| `592433ef7` | corex provider 进 `backends/corex/__init__.py`，entry point 改指 `jittor.backends.corex` |
| `d78947796` | **整棵死的 `extern/rocm` 删除**，`9.12` 的字节钉定换成规则 `test_no_unexplained_binaries.py` |
| `ae5205f91` | mkl 14 个 oneDNN 算子进 `backends/cpu/libraries/mkl`，新建 `backends/cpu/__init__.py` |
| `2bdaf9ddc` | **`python/jittor/src` 的 360 个文件搬到顶层 `src/`**，新增 `core_root()` |
| `09cb78270` | **`2.23`+`3.24`+`1.05` 三刀 src 内部重组**（含 `backends/` 侧的 include 重写） |

前三刀每刀都跑了 `tests/structure`（15 failed **逐条同基线**、0 error）与 CUDA 冒烟。
后两刀只做了静态 include 检查——原因见第 0 节。

`src/` 现在的形状，根下**零个**散落 `.cc/.h`：

```
src/{bindings,codegen,core,debug,mem,ops,runtime,tests,third_party,type,utils}
    bindings/{pyjt,pybind}          2.23
    codegen/{opt,op_compiler,jit_key,jit_compiler}   3.24
    core/                            1.05（35 个原根文件，含 common.h）
    ops/                             七个元算子 + op_utils/op_register
    ops/composite/                   50 个具体算子
    runtime/{init,lock,profiler}     2.23
```

## 2. 还没做的

**`4.15` 剩最后一刀：`extern/acl` 的 92 个（102 减 hccl 10）。**

⚠ **这一刀有一个会静默出错的陷阱，别分刀做。** `python/jittor/extern/acl/acl_compiler.py:51`
用 `acl_compiler_home = os.path.dirname(__file__)` 再
`glob(acl_compiler_home + "/**/*.cc", recursive=True)` **递归找自己的源码**。如果照 corex
的做法只把 provider 搬到 `backends/acl/__init__.py` 而把 `aclops/`（84 个）留在老位置，
`dirname(__file__)` 变成 `backends/acl/`，glob 会**编出一套完全不同的源码集合而不报任何
错**——编译能过、测试可能大半还绿，但编进去的是错的东西。**provider 与源码必须同一刀。**

搬完 acl 之后 `extern/` 只剩 `cuda/nccl`（15）与 `mpi`（10），**那两组归 `8.19`**
（它明确要三个通信后端进 `backends/comm/`，另含 `extern/acl/hccl`）。`extern/` 清空后记得
删掉 `compiler.py:1724` 的 `-I{jittor_path}/extern`，以及 `MANIFEST.in:8` 那行
`recursive-include python/jittor/extern *`。

**其余布局条目**：`0.20`（文档树，要等所有 agent 停写看板）、`5.26`（Python 包内部）、
`9.19`（`tools/` 职责）、`8.19`、`10.23`（卡在 `7.18`，`7.18` 卡在 `7.12`——我实测
`torch is jittor` 还是 True，模块身份那层没开始）。

**`4.15` 的两条验收按字面都达不到**，已记在看板：`python/jittor/` 下无 `.cc/.cu` 还需
`7.18`（compat 的 15 个）与 `5.26`（`math_util` 的 1 个）；`grep cuda_src == 0` 不可能，
因为它是 `jt.code()` 的公开参数名，24 处在 `__init__.pyi`、25 处在 `code_op.{cc,h}`。

## 3. 工具：别再为每刀付一次全量重建

`agent/scripts/check_core_includes.py`，**0.4 秒**，解析核心里每条 quoted include 并对着
`-I` 根求解。它不是编译器（模型不了每个后端自带的 `-I`），所以在能编过的树上仍报约 280
处盲点——**所以只能用差分模式**：

```bash
python3 agent/scripts/check_core_includes.py --baseline /tmp/inc_baseline.txt .   # 已知良好时记
python3 agent/scripts/check_core_includes.py --check    /tmp/inc_baseline.txt .   # 搬完只看新增
```

对比一下成本：`src` 那一刀我付了**四次全量重建**（178 个 TU、每次 5–9 分钟）才把四处
路径耦合逐个照出来，因为构建在第一个错误处就停、后面的问题被挡着。有了这个脚本，
后三刀的 include 正确性是秒级确认的。

**但它覆盖不到两类东西**，这两类只有真实构建会暴露，是我付重建学到的：

1. **构建的文件列表按字符串匹配**。`compiler.py` 的 `at_beginning`/`at_last`/
   `jit_utils_core_files` 里写的是 `"src/..."` 相对名，而 `files4.remove(...)` 是
   `list.remove`，**比的是字符串不是路径**。搬动后这些字面量要跟着改（我改了四项：
   `init.cc`、`profiler/profiler.cc`、`event_queue.cc`、`executor.cc`）。
2. **Python 侧按路径打开具体文件**。`pyjt_compiler.py` 读 `var_holder.h`、`compiler.py`
   读 `ops/{name}_op.cc`。前者随 `1.05` 进了 `core/` 要改；后者只用于
   `binary`/`unary`/`reduce` 三个元算子，它们留在 `ops/`，所以不用改。

## 4. 搬 `src` 那一刀的做法，值得照抄

**不要逐处硬改路径。** 我最初估「产品引用只有 3 处」是错的——那个 grep 只匹配
`os.path.join(jittor_path, "src")`，漏了字符串拼接 `jittor_path+"/src/**/*.cc"`、相对
路径列表、以及靠**前缀长度切片**做归一化的 `f[len(jittor_path)+1:]`。光 `compiler.py`
一个文件就有 21 行涉及 `src/`。

正确做法是找**咽喉**：`compile()` 里解析相对源码名的那一处（`os.path.join(jittor_path,
name)`）加一个 `_source_path()`，把开头为 `src/` 的名字路由到 `core_root`，几十个相对
列表一次全对；再把 4 处 glob 与那处归一化改成按 `core_root` 算，**并保持产出
`"src/..."` 的历史拼写**——因为上面第 3 节那些 `list.remove` 是按字符串匹配的，拼写一变
就碎。

`core_root()` 与 `backend_root()` 同形（checkout 下 `<repo>/src`、安装后 `<package>/src`），
**标记文件是 `core/common.h` 而不是 `isdir`**：`1.05` 把 `common.h` 也搬进了 `core/`，
标记跟着内容一起移，而 `isdir` 会在一个空的遗留目录上通过、然后交给你一个 `-I` 什么都
找不到的根。我在 `1.05` 之后确实撞了这一下（`core_root` 抛 `FileNotFoundError`）。

## 4bis. 先改这一个常量，后面每一步都受益

`jittor_utils/__init__.py:437`：

```python
pool_size = min(16, max(int(mem_gib // 3), 1))
```

这台机器 **128 核、1007 GB 内存**（算出来是 335），却被 `min(16, ...)` 硬顶在 16 个
编译进程。178 个翻译单元的全量重建因此要 5–9 分钟，而那正是布局验证的单位成本。
放开这个上限之前，别指望验证快起来——整个会话里机器负载最高 8.15、多数时候 1–3，
**利用率约 6%**，慢的从来不是机器。

（改它要小心的是内存而不是核数：并行编译的峰值内存按进程数线性涨，`mem_gib // 3`
就是为这个留的。128 核 / 1007 GB 上取 64 是安全的；真要取满 128 先量一次峰值。）

## 5. 验证口径：只报 passed 是看不出问题的

**同一棵树、同一条命令，`tests/structure` 在两种配置下不是同一组用例**：

| 配置 | collected | 结果 |
| --- | --- | --- |
| `nvcc_path=""` | 907 | 15 failed / 888 passed |
| 带 `nvcc_path` | 930 | 14 failed / 914 passed |

差的 23 条是随 CUDA 可用性参数化出来的。所以**这两组之间不能直接 A/B**，passed 差 26
不代表修好了 26 条。报数字的最小口径：命令、`nvcc_path` 取值、`JITTOR_TORCH_SHIM`
取值、`collected`、failed/passed/skipped/xfailed 五个数、**以及失败的 nodeid 集合**。
判据是失败集合**逐条相同**，不是数目相同。

三套门禁（原生 CPU、CPU torch 模式、CUDA）加带 CUDA 的 `import jittor` + matmul 冒烟，
缺一不可。`tests/structure` 必须带 `JITTOR_TORCH_SHIM=1`，不带会 collect 0 个、
读起来像全过。

每搬一刀顺手跑一次 `tests/structure/test_refactor_board_contract.py`（秒级），它会抓住
看板写坏。**备注里不能出现裸竖线**——那在 Markdown 表格里是列分隔符，会把行撑破、
让这条合同报红从而挡住所有人的 `tests/structure`。我自己踩过一次（写了 `"src"|"extern"`）。

## 5bis. 两个操作层面的教训

**`git commit <路径列表>` 只提交列出的路径。** `git mv` 会把 rename 两侧都暂存，但带路径
的 commit 只带你列的那些，旧位置的删除会留在暂存区。我在四刀里**每刀都要补一个「删除侧」
提交**，最后一次还漏了 `backends/` 侧 166 个文件的 include 重写（分家会让树在两个提交
之间是坏的）。改用不带路径的 `git commit`，配合先看 `git diff --cached --stat`。

**同一分区不要并发两个 pytest session**：会把 `$JITTOR_HOME` 搞坏，`import jittor` 在
`setup_cub` 里 abort，造出一批假失败。要并行就给第二个进程另一个 `JITTOR_HOME`——但
**新建的空 `JITTOR_HOME` 现在会因为第 0 节那个 preflight 直接失败**，所以在那条修掉之前，
验证只能串行复用热缓存。

## 6. 剩下的顺序，以及每条的边界

| 序 | 任务 | 边界（能划掉的那个终态） |
| --- | --- | --- |
| 0 | preflight | CPU-only 冷构建（空 `JITTOR_HOME`）能起来；CUDA 冷构建缺 cub 且无网时**仍然明确失败**并给出离线办法（别把这条一起放过） |
| 1 | `4.15` acl | `python/jittor/extern/{acl 非 hccl}` 不存在；entry point 改 `jittor.backends.acl`；**provider 与源码同一次提交** |
| 2 | `8.19` | 三个通信后端（`extern/cuda/nccl`、`extern/mpi`、`extern/acl/hccl`）进 `backends/comm/`；`extern/` 清空后删 `compiler.py:1724` 的 `-I{jittor_path}/extern` 与 `MANIFEST.in:8` 那行 |
| 3 | `5.26` | 包根下只剩 `__init__.py`/`__init__.pyi`/`selftest.py`；无 > 1500 行的 `.py` |
| 4 | `0.20` | 仓库只有一棵文档树。**要等没有别的 agent 在写看板**，否则每次 rebase 都撞 |
| 5 | `9.19` | `tools/` 与 `python/jittor/tools/` 职责不重叠 |
| — | `10.23` | **不在这一波**：卡 `7.18`，`7.18` 卡 `7.12`（实测 `torch is jittor` 仍是 True，模块身份那层没开始） |

`4.15` 的两条验收按字面都达不到，已记在看板：`python/jittor/` 下无 `.cc/.cu` 还需
`7.18`（compat 的 15 个）与 `5.26`（`math_util` 的 1 个）；`grep cuda_src == 0` 不可能，
因为它是 `jt.code()` 的公开参数名（24 处在 `__init__.pyi`、25 处在 `code_op.{cc,h}`）。

## 7. git 约定（这一条我犯了四次）

**`git mv` 会把 rename 两侧都暂存，但 `git commit <路径列表>` 只提交你列出的路径**——
旧位置的删除留在暂存区。四刀里我每刀都要补一个「删除侧」提交，最后一次还漏了
`backends/` 侧 166 个文件的 include 重写（分家会让树在两个提交之间是坏的）。
用不带路径的 `git commit`，并先看 `git diff --cached --stat`。

禁止 `git add -A`、force-push、`git stash`、碰 `2.0` 分支。**rebase 冲突逐块解，禁整
文件取一侧**——2026-09-03 真发生过整文件取一侧把另一分区已合并的 `7.16` 整段还原，
40 个 handler 变回 `except: pass`，提交是绿的、diff 看起来只是「我的文件」，没有任何
东西报警。解完必须跑 `JITTOR_TORCH_SHIM=1 pytest tests/structure -q`。

13 个工作树共用一个 `.git`，push 可能撞 `cannot lock ref`：**重试，绝不 force-push**，
重试前先 `git log --oneline origin/2.0-refactor -3` 确认自己那次是不是已经落地了。

三个文件属于别人永不提交：`agent/manuals/README.md`、`tests/core/test_setitem.py`、
`agent/results/2026-08-12-repository-modernization-review.md`。

## 8. 环境

```
JITTOR_HOME=/home/zy/jittor-lab/refactor/_home/pyops \
TMPDIR=/home/zy/jittor-lab/refactor/_tmp/pyops \
CUDA_VISIBLE_DEVICES=6 nvcc_path=/usr/local/cuda/bin/nvcc \
PATH=/usr/local/cuda/bin:$PATH \
taskset -c 64-79 /home/zy/miniconda3/envs/jt311/bin/python -m pytest <路径> -q
```

手写 python 命令必须显式加 `PYTHONPATH=.../pyops/python`（pytest 不需要）。
CPU-only 加 `JITTOR_TEST_DEVICES=cpu nvcc_path=""`，但加了之后 `jt.has_cuda` 就是 0，
**不要因此得出「本机无 CUDA」**——这台机器有 8 张 RTX 4090、nvcc 12.2.140、sm_89。

rebase 之后第一次 `import jittor` 会重建 `jit_utils` 然后按设计 `sys.exit(3)`，pytest
报成 `SystemExit: 3` 的 collection error，**那不是真失败**：先单独跑一次
`python -c "import jittor"` 消化掉。同一分区一次只跑一个 pytest——并发两个会把
`$JITTOR_HOME` 搞坏，`import jittor` 在 `setup_cub` 里 abort，造出一批假失败。
