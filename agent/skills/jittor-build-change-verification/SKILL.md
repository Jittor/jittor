---
name: jittor-build-change-verification
description: 改了 Jittor 构建系统（jittor_utils、compiler.py、compile_extern.py、cache_compile.cc、lock.cc、pyproject 的 pytest 配置）之后，怎么确认没把别人的构建弄坏。给出冷缓存 / 热缓存 / 并发 / 切 flag 四种情形各自的验证命令与判据，以及多 worktree 并行时哪些状态是全局共享的。另含 §2.5「怎么可复现地量 import jittor 的耗时并归因到具体一步」（三种造冷缓存的办法、为什么不能用 profiler 得结论、配套脚本 measure_import_cost.py）。凡是会改变缓存路径、锁、编译命令行、探测流程或 import 耗时的改动都要按这个走一遍再推。
---

# 改了构建系统之后怎么确认没弄坏别人

构建系统的改动和算子的改动不一样：**算子改坏了你自己的测试会红，构建改坏了是别人的
测试会红**，而且症状与你的改动毫无关系（散布的梯度不符、段错误、"op doesn't have cuda
version"、莫名其妙的 40 分钟卡死）。所以推之前必须自己先跑完下面四种情形。

四种情形对应四类真实故障：

| 情形 | 覆盖的故障 |
| --- | --- |
| 冷缓存 | 缓存路径拼错、首次生成顺序错、探测失败没兜底 |
| 热缓存 | 缓存键少算了东西（改了不重编）或多算了东西（每次全重编） |
| 并发两个进程 | 锁没生效、后写者替换前者已 dlopen 的产物 |
| 切 flag 再跑 | 不同配置共用同一个目录、互相重编 |

## 0. 先决条件

隔离环境（少一个就是在测别人的代码或污染别人的缓存），见
`jittor-worktree-verification`：

```bash
WT=<你的 worktree>            # 例如 .../refactor/<分区>
JH=<你的 JITTOR_HOME>          # 例如 .../refactor/_home/<分区>
TD=<你的 TMPDIR>
E="JITTOR_HOME=$JH TMPDIR=$TD PYTHONPATH=$WT/python nvcc_path=/usr/local/cuda/bin/nvcc"
```

`pytest` 在 worktree 里自动导入本树，**手写的 `python -c` 必须显式带 `PYTHONPATH`**。

跑之前 `df -h`。磁盘满的症状和并发损坏一模一样（散布的编译失败加段错误）。

**一条命令里只要有一个分支漏了 PYTHONPATH，整条就作废。** `cmd_a || cmd_b` 这种写法
最容易漏：`cmd_a` 带了、`cmd_b` 忘了，而失败时跑的恰恰是 `cmd_b`。同一个坑还有
`for` 循环里只给第一次带、以及 `env A=1 python` 后面接的第二条命令。判据不是"我写没写"，
而是**看日志第一行**：

```
[i ...] Jittor(1.3.11.0) src: /path/to/some/tree/python/jittor
```

这个路径不是你的 worktree，后面的一切结论全部作废——而且它会安安静静地把**别人那棵树**
从头编一遍（本机实测 156 个文件），你只会觉得"这条命令怎么这么慢"。

**验证用的临时缓存长得比想象中快**：本文第 1、3 节各要一个全新的 `JITTOR_HOME`，
一次冷启动加一次并发冷启动在本机占了 24 GB。跑完就删：

```bash
rm -rf $JH.cold $JH.conc
```

## 0.5 一条命令跑完四种情形

四节都手跑一遍要半小时，还容易漏掉判据。同目录下的 `verify_build_change.sh` 把四种
情形连同它们的断言写成了一个脚本，退出码为 0 才算通过：

```bash
JITTOR_SRC=$WT JITTOR_HOME=$JH TMPDIR=$TD \
PYTHON=<解释器> NVCC=/usr/local/cuda/bin/nvcc JOBS=4 \
bash agent/skills/jittor-build-change-verification/verify_build_change.sh
```

**它会 `rm -rf $JITTOR_HOME`**，所以给它一个专用的缓存目录。每一步都写一份日志到
`$TMPDIR/verify-build/`，失败时先看那里。它自己也断言了「导入的是不是本 worktree」，
因为其余每一条断言在导错树的时候都是空的。

脚本跑绿之后，下面各节仍然要读——它自动化的是**判据**，不是**判断**。本机一次完整
运行的量级：冷缓存 71s，热缓存 2s，4 路并发冷启动 76s，切 flag 后每次 1-2s。

## 1. 冷缓存

```bash
rm -rf $JH.cold && mkdir -p $JH.cold
env $E JITTOR_HOME=$JH.cold python -c "
import time,os; t=time.time()
import jittor
print('COLD_OK', round(time.time()-t,1))
print('CACHE', jittor.flags.cache_path)
print('HAS_CUDA', jittor.has_cuda)"
```

判据：

- 退出码 0，`HAS_CUDA` 与你预期一致（有 nvcc 就该是 1；变成 0 通常是缓存路径改动
  让 CUDA 版 `jittor_core` 被 CPU 版遮蔽）。
- `CACHE` 打出的路径**逐段**看一遍。这里最容易出的事故是把日志文本拼进了目录名：
  `compiler.py` 用子进程查 GPU 算力，**子进程的任何一行输出都会被当成 arch 号**切进
  cuda key（历史上真发生过，目录名里出现 `..._sm_0902_215850..._Create_[i_file...`）。
  凡是目录里出现日期、方括号、路径片段，就是有子进程往 stdout/stderr 写了东西。
- 冷启动耗时记下来（本机整核心约 85 s），后面对比用。

## 2. 热缓存

同一条命令再跑一次：

```bash
env $E python -c "import time;t=time.time();import jittor;print('WARM_OK',round(time.time()-t,1))"
```

判据：

- 秒级（本机约 3 s）。如果热缓存也要几十秒，说明你把某个**每次都变的东西**放进了
  缓存键（时间戳、pid、随机数、`os.environ` 里被自己写回去的值）。
- 连跑三次都不再出现 `jit_utils updated, please rerun your command.`。
  改了 `src/utils/{cache_compile,log,tracer,jit_utils,str_utils}.cc` 之后第一次必然出现
  一次（这是设计如此，见 `jittor-core-cpp-edit-loop`），但**第二次还出现就是 bug**：
  说明缓存键不收敛。

## 2.5 量 import 耗时，并把它归因到具体一步

「热缓存 import 变慢了 0.3 s」这句话没法行动。同目录下的 `measure_import_cost.py`
把总时间拆成能改的项：

```bash
EXPECT_JITTOR_SRC=$WT/python env $E \
  python agent/skills/jittor-build-change-verification/measure_import_cost.py \
      --json $TD/before.json
```

`EXPECT_JITTOR_SRC` 不是可选的：不设它，你量到的可能是**装好的那份** jittor，而
事后没有任何办法分辨。脚本自己会断言。改完代码再跑一次 `--json $TD/after.json`，
两份 JSON 直接对比。

### 三个层次，各看得见不同的东西

| 层 | 机制 | 粒度 | 盲区 |
| --- | --- | --- | --- |
| 1 | 子进程 `-X importtime` | 一个模块的 self / cumulative | `compiler.py` 的模块体是一个 1.4 s 的整块 |
| 2 | import **之前**替换 `jittor_utils.run_cmds` / `run_cmd` | 每次构建扇出的命令条数与耗时 | 只覆盖走 `jit_utils` 的调用 |
| 3 | import **之后**再调一次生成器 | `gen_jit_flags` / `gen_jit_tests` / `pyjt_compiler.compile` | 只对幂等的东西成立 |

第 2 层为什么必须从外面打：要量的东西在 `compiler.py` 的**模块体**里，等到
`jittor.compiler` 这个模块对象存在时它已经跑完了，没有可以事后包的函数。
`jittor_utils` 是另一个包、且 `compiler.py` 是按属性查找调用它的
（`jit_utils.run_cmds(...)`），所以在 `import jittor` 之前替换 `jittor_utils` 上的
名字能生效。**这条手法对任何「模块体里的副作用」都适用**：找它调用的、位于别的
模块里的那个函数，从外面包。

### 不要用 profiler 得结论

`cProfile` 在这条路径上加约 40%（2.5 s → 3.7 s），而且**会改变各项的排序**：
`strip_cxx_comments` 有 2.3 M 次 `str.startswith`，profiler 对调用次数敏感，于是它
被放大得比实际严重。profiler 用来**找**候选（`sort_stats('cumulative')` 一眼就能看到
`compile` / `run_cmds` / `gen_jit_flags`），墙钟用来**定价**。

### 冷缓存怎么可复现地造

「冷」有三种，代价与覆盖面不同，报数字时必须说清是哪一种：

| 造法 | 命令 | 会重编 | 用来验什么 |
| --- | --- | --- | --- |
| 全新缓存 | `rm -rf $JH.cold` 再用它做 `JITTOR_HOME` | jit_utils_core + 核心 + 全部 extern op | 首次安装体验；探测失败有没有兜底 |
| 换配置 | 同一个 `JITTOR_HOME`，把 `nvcc_path` 在 `""` 与真路径之间切 | 核心（另一个 `cfg*` 目录）| **切门禁的真实代价**；这是最容易被忽略的一种 |
| 碰源文件 | `touch python/jittor/src/executor.cc` | 该 TU + 链接 | 依赖跟踪对不对 |

第二种是本机实测 40 s 的那一种，注意它**不是空缓存**：CUDA 配置与 CPU-only 配置的
`cfg*` 指纹不同，各自要一份完整核心，所以在三套门禁之间来回切每次都付一次全量
核心编译。报「冷编译 40 s」而不说是哪一种，下一个人会以为是空缓存。

`touch` 那种要注意：**改注释验证不了重编**（见 §5），要 `touch` 或改符号。

### 判据

- 热缓存 import 的数字**必须报配置**。本机同一棵树：CPU-only 1.33 s、CUDA 2.46 s。
  只报一个数、不说 `nvcc_path` 是什么，等于没报。
- 连续跑三次取后两次。第一次可能撞上 jit_utils 重建（`§2`）或别的 agent 刚推的改动。
- 空转的构建扇出成本**与核心 TU 数成正比**（本机约 6-8 ms/条墙钟，16 路并行）。
  所以增加 TU 的改动要报「+N 个 TU」，比报「+0.1 s」有用——后者换台机器就不成立。
- 归因表的各项之和应当接近总时间。差得多说明漏了一层，别把差额记成「Python 启动开销」
  就算了。

一份写完的归因表见 `agent/results/2026-09-04-import-jittor-cost-attribution.md`。

## 3. 并发两个进程

这是唯一能验证锁的办法。两个进程共用一个**空**缓存：

```bash
rm -rf $JH.conc && mkdir -p $JH.conc
for i in 1 2; do
  env $E JITTOR_HOME=$JH.conc JT_LOCK_REPORT_AFTER=10 \
    python -c "import jittor as jt;print('OK',float((jt.array([1.,2.,3.])*2).sum()))" \
    > $TD/conc_$i.log 2>&1 &
done; wait
grep -h "OK\|waiting for\|Error\|error" $TD/conc_*.log
```

判据：

- 两个都 `OK`，算出的值正确。
- 耗时应当是 **T + ε**，不是 2T 也不是 T：`import jittor` 全程持锁，所以第二个进程
  等第一个跑完（T），然后自己走热缓存（ε）。如果两个都是 T，说明它们真的在并行编译
  ——锁没生效。
- 第二个进程应打印一行 `waiting for build lock ... held by pid N ...`。**没有这行就是
  锁的诊断没接上**，将来卡死时又只能靠猜。

## 4. 切 flag 再跑

编译配置（`cc_flags` / `nvcc_flags` / `kernel_flags` / `cuda_archs` / `enable_lto`）不属于
缓存目录的自然分区，必须靠指纹分开。验证：

```bash
for F in "" " --fmad=false --prec-div=true --prec-sqrt=true "; do
  env $E nvcc_flags="$F" python -c "import jittor_utils as j;print(j.cache_path)"
done
```

判据：两行**必须不同**，且 `<那个目录>/build_config.json` 记着这套 flag。相同就意味着
两套不同的目标码写进同一个 `jit/`：切一次全重编，并发时后写者替换前者已 dlopen 的 `.so`。

反过来也要看一眼：`jittor.lock` 的位置**不应**随配置变化。

这是一条通用判据：**锁的作用域必须匹配它保护的资源的作用域**。`jittor.lock` 保护的
不只是本配置的 `jit/`，还有 `~/.cache/jittor/{mkl,cutt,cub}` 这些所有配置共享的
下载目录——所以它必须待在配置目录**之上**。给缓存路径加一段分区的时候，顺手问一句：
这把锁现在还盖得住它要保护的全部东西吗？盖不住就是把互斥悄悄削掉了，症状是并发
下载互相覆盖，看起来像网络问题。

## 4.5 改了 core 的导出接口之后：扫一遍起裸 python 的地方

C++ 侧改了 pyjt 导出的名字或签名（`@pyjt(...)`、`core.xxx`），除了自己的测试之外
还要扫一遍**测试里起子进程的地方**：

```bash
grep -rn "sys.executable\|getoutput\|os.system\|subprocess" tests/ | grep -v PYTHONPATH
```

失败的形状是这样的：某个用例用 `sp.getoutput(sys.executable + ' ' + fname)` 起一个
裸 python。pytest 只把本 checkout 放进**自己进程**的 `sys.path`，不导出 `PYTHONPATH`，
所以这个子进程导入的是**装好的那份** jittor（主树）。但父进程把 `cache_name` 写进了
环境，于是子进程加载到的是**你这份分支刚编出来的 core**——新 core 配旧 Python 层。
两边一旦对不上就炸：

```
AttributeError: module 'jittor_core' has no attribute 'set_lock_path'.
Did you mean: 'set_lock_fd'?
```

而这条报错和你的改动看起来毫无关系（改的是锁，红的是 tracer）。修法是给子进程显式
传环境：

```python
environment = dict(os.environ)
environment["PYTHONPATH"] = REPO_PYTHON + os.pathsep + environment.get("PYTHONPATH", "")
sp.run((sys.executable, fname), env=environment, ...)
```

判据：**任何以 `sys.executable` 起、又会 `import jittor` 的子进程，都必须显式带
PYTHONPATH**。这条比"我的改动有没有效果"更狠：不带的话，跑的是两棵树的嵌合体。

## 4.6 改了编译命令行：import 成功证明不了任何事

`cache_compile` 会**解析编译命令行**来找源文件，规则是"不是选项的 token 就是源文件"。
所以一个分成两段的 flag 会被当成文件名：

```
 -gencode arch=compute_89,code=[sm_89,compute_89]     # 两个 token
 → Check failed: src.size()  Source read failed: arch=compute_89,code=...
```

**而 `import jittor` 全程是绿的**——核心走的是另一条编译路径，只有 JIT 算子经过
`cache_compile`。所以改了 `nvcc_flags` / `cc_flags` 的组装之后，必须**真的编一个算子
出来再看产物**：

```bash
$E python -c "
import jittor as jt, jittor.compiler as c, glob, os
jt.flags.use_cuda = 1
(jt.random((256,256)) @ jt.random((256,256))).sum().item()      # 逼它编一个 CUDA kernel
so = max(glob.glob(os.path.join(c.cache_path,'jit','*.so')), key=os.path.getmtime)
print(so)"
cuobjdump -lelf -lptx <那个 .so>        # cubin 有几个？PTX 有没有？
```

两条硬性要求：**每个 flag 必须是一个 token**（用 `--generate-code=...`，不要
`-gencode ...`），**token 里不能有 shell 通配符**（命令要经过 shell，`[...]` 是通配
模式）。

## 4.7 子进程本来就该崩的用例，中间必须留一层 shell

这条是 4.5 的直接陷阱。把 `getoutput(f"{sys.executable} {f}")` 改成
`subprocess.run([sys.executable, f], env=...)` 是给子进程补 PYTHONPATH 的自然写法，
但它同时**去掉了中间那层 shell**。如果这个子进程本来就应该 abort（例如用例测的正是
kernel 里的 assert 触发时的报错），那么它现在是 pytest 的**直接子进程**，jittor 装在
父进程里的 SIGCHLD handler 看到 `si_code=3`（CLD_DUMPED）就判定 OOM 并 quick exit：

```
[e ... log.cc:250] Caught SIGCHLD. Maybe out of memory ... si_status: 6 , quick exit
```

**整个 pytest 进程消失，一行输出都没有**，`-q` 下连 summary 都不打——看起来像是挂了，
不像是失败。写法：保留 shell，同时传 env。

```python
subprocess.run("%s %s" % (sys.executable, path), shell=True, env=child_env(),
               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
```

判据：**用例跑完退出码非 0 但没有任何输出**，第一个要怀疑的就是这个。

`tests/_helpers/child_process.py` 的 `child_env()` 是这两节共用的那份环境构造，新写
子进程时直接用它，不要再抄一份。

## 5. 别被这些假象骗了

- **改一行注释验证不了重编。** 注释不改变 `.o` 的字节，链接的缓存键因此不变，`.so`
  不会重链。要验证"产物真的被重建了"，改一个**符号**（加个函数），或者直接比 `.o` 的
  inode：`stat -c %i`。inode 变了说明是"写新文件再改名"，没变说明是原地覆盖。
- **`import jittor` 成功不代表构建对了。** 还要看 `jittor.flags.cache_path` 是不是你以为
  的那个目录、`jt.has_cuda` 对不对。CPU 版 core 遮蔽 CUDA 版时，import 是成功的，错要到
  第一个 CUDA 算子才报。
- **在 jittor 进程里 `kill` 子进程会把自己也带走。** jittor 装了 SIGCHLD handler，看到
  子进程被信号杀死就判定 OOM 并 `quick exit`，测试进程整个消失、不打任何 traceback。
  测试里要结束子进程，让它自己退（读 stdin 一行、或用哨兵文件），别 `kill()`。
- **加了一个过滤/守卫之后，先问它在"输入为空"时退化成什么。** 这类改动的失败形状
  不是报错，是**静默地什么都不做**，于是测试全绿而保护为零。今晚一晚上撞了四次：
  「项目外的文件不下钻」用路径前缀判断，而 C++ TEST harness 传的 roots 是**空的**，
  于是什么都不算项目内、什么都不扫，改头文件不再触发重编（空 roots 现在等于"全扫"）；
  `@onlyCPU` 在 CUDA 门禁下生成**零个**用例；`SharedReducePass` 因为默认 flag 从不运行；
  `is_type<NanoString>` 万能匹配。写完守卫，把它的输入置空跑一遍，看它是不是变成了空操作。
- **不要 `kill -9` 正在编译的 jittor 进程。** 会留下损坏的缓存，下一次运行在毫不相干的
  算子上大面积报错。真要杀，先把整个 `JITTOR_HOME` 删掉。

## 6. 多 worktree 并行时，哪些状态是共享的

十个 agent 各有一个 worktree，很容易以为"我的目录是我的"。实际上：

| 资源 | 隔离吗 |
| --- | --- |
| 工作区文件、index（暂存区） | **是**，每个 worktree 独立 |
| `git stash` 栈 | **否**——存在公共的 `.git` 里，`pop` 出来的可能是别人的改动（已经发生过一次两个分区 WIP 对调） |
| 分支、tag、`origin/*` 等 refs | **否**，全仓共用 |
| `JITTOR_HOME` 缓存 | 只有在你显式设了才隔离；不设就是全机共用 |
| `TMPDIR` | 同上 |
| `~/.cache/jittor/{mkl,cutt,cub}` 等第三方下载 | **否**，按 `JITTOR_HOME` 共用 |
| `jittor.lock` | 按 `JITTOR_HOME` 共用（这是它该做的事） |

推论：

- **不要用 `git stash`。** 需要临时搁置：
  `git diff <文件> > $TD/wip.patch && git checkout -- <文件>`，恢复用 `git apply`。
- 一次只做一个任务，改完就测、测完就提交，别让无关的 WIP 留在树里。
- 这也是审计里那批 bug 的共同形状：**看起来每个进程独立、实际上是全局共享的资源**
  ——进程级的 `ssl._create_default_https_context`、写回 `os.environ` 的 `cache_name`、
  两个进程都以为自己独占的 `jittor.lock`。改构建系统时，每加一个全局状态就问一句：
  这台机器上另一个 jittor 进程会看到它吗？

## 7. 推之前

- 跑 `tests/compiler`（构建系统的回归都在这里），至少和你改动前的结果对齐。
  **先记录一份改动前的失败清单**：这个目录本来就带着与你无关的失败（例如别人改了
  归约实现之后 `test_atomic_tuner` 就红了），不先记就会花一小时查一个不是你的 bug。
- 提交说明里必须写一句「**其他 agent 需要做什么**」：要不要清缓存、会不会触发一次全量
  重编、有没有改动公开 API（例如 `core.set_lock_path` 改名）。别人 rebase 之后第一件事
  就是看这句。
