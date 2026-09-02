---
name: jittor-build-change-verification
description: 改了 Jittor 构建系统（jittor_utils、compiler.py、compile_extern.py、cache_compile.cc、lock.cc、pyproject 的 pytest 配置）之后，怎么确认没把别人的构建弄坏。给出冷缓存 / 热缓存 / 并发 / 切 flag 四种情形各自的验证命令与判据，以及多 worktree 并行时哪些状态是全局共享的。凡是会改变缓存路径、锁、编译命令行或探测流程的改动都要按这个走一遍再推。
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

**验证用的临时缓存长得比想象中快**：本文第 1、3 节各要一个全新的 `JITTOR_HOME`，
一次冷启动加一次并发冷启动在本机占了 24 GB。跑完就删：

```bash
rm -rf $JH.cold $JH.conc
```

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

反过来也要看一眼：`jittor.lock` 的位置**不应**随配置变化（它同时保护 mkl/cutt/cub 这些
所有配置共享的下载）。

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
