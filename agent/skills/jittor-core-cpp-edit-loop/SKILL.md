---
name: jittor-core-cpp-edit-loop
description: 改 python/jittor/src 下 C++ 核心时的编辑—重编—验证循环。包含隔离缓存与解释器选树的自检、把重编从 ~10 分钟压到 ~30 秒的 CPU-only 循环、每次 C++ 改动后第一次 pytest 必然失败的"jit_utils updated"陷阱，以及"读到未初始化字节"这类静默错值的复现判据。
---

# 改 Jittor C++ 核心的验证循环

改 `python/jittor/src/**` 的人每次都会重新踩同样四个坑：导错源码树、缓存互相污染、
每改一行等十分钟、以及改完第一次跑测试拿到一个假失败。本 skill 是这四件事的固定答案。

## 1. 先自检：解释器导入的是哪棵树

开发环境里 jittor 是 **editable 安装**，`site-packages` 的 `.pth` 指向**某一棵固定的源码树**。
在 git worktree 里裸跑 `python -c "import jittor"` 导入的是那棵树，**不是你改的这棵**。
一行复现（`jt.array(np.uint8([200])).item()` 这种）最容易这样跑，然后得出"已经是对的"的错误结论。

- `pytest` 是安全的：`tests/conftest.py` 让它跑在哪个 worktree 就导哪棵树（`JITTOR_SOURCE_ROOT` 可覆盖）。
- **任何手写的 `python -c`、`python 脚本.py`、子进程，都必须显式加 `PYTHONPATH=<你的 worktree>/python`。**

每轮验证前先跑一次这个，必须打印出你自己的目录：

```bash
PYTHONPATH=$WT/python JITTOR_HOME=$JH TMPDIR=$TD \
  python -c "import jittor, os; print(os.path.dirname(jittor.__file__))"
```

## 2. 缓存隔离是硬性的

每个并发进程一套 `JITTOR_HOME` 与 `TMPDIR`。共享缓存的并发运行会互相损坏，表现是
**在与你的改动完全无关的算子上大面积报梯度不符**，看起来像回归。同理：不要 `kill -9`
正在编译的 Jittor 进程；真要杀，先把该进程的 `JITTOR_HOME` 整个删掉再重跑。

## 3. 把重编从十分钟压到半分钟

完整的 CUDA 构建（`nvcc_path=/usr/local/cuda/bin/nvcc`）第一次约 10 分钟。
只验证与设备无关的核心逻辑时用 CPU-only：

```bash
JITTOR_TEST_DEVICES=cpu nvcc_path="" pytest tests/core/test_xxx.py -q
```

CPU-only 是**另一个缓存分区**（路径里没有 `cu12.x` 段），核心 156 个编译单元约 25–35 秒重编，
可以一分钟一轮。代价是 `jt.has_cuda` 为假，所有 CUDA 用例被 skip——所以**收尾时必须再跑一次带
`nvcc_path` 的 CUDA 轮**，两轮都绿才算完。

缓存目录名里带**源码内容哈希**与**分支名**，所以改一行 C++ 就换一个目录：老目录不会被复用，
但也不会被清理。跑几十轮之后 `du -sh $JITTOR_HOME`，超过 20G 就整个删掉重来。

## 4. 每次 C++ 改动后的第一次运行必然"失败"

改完 `python/jittor/src/**` 后第一次跑 pytest，会在 collect 阶段拿到：

```
E   SystemExit: 0
[e ... compiler.py:927] jit_utils updated, please rerun your command.
```

这**不是**你的改动有问题：jittor 重新生成了 `jit_utils` 之后要求换一个新进程。
**原样重跑同一条命令**即可。不要因为这个去改代码。

## 5. 静默错值的复现判据

C++ 核心里的"静默算错"分两类，各有一个能一眼定性的判据：

**读到未初始化字节** —— 判据是**同一个值连读多次结果不同**。POD 结构体没有值初始化时，
高位字节是栈上的残留，随调用栈变化：

```python
var = jt.array(np.array([200], dtype="uint8"), dtype="uint8")
first = var.item()
assert [first] * 8 == [var.item() for _ in range(8)]   # 修前会差最后一两次
```

只断言"等于 200"是不够的：它可能碰巧对一次。**同时**断言跨调用一致，才能把
"没初始化"与"转换分支写错"这两个原因分开——两者都会让值不对，但只有前者不稳定。

**dtype 相关的测试必须钉死 dtype**：`jt.array(np.ones(4, "float64"))` 会静默变 float32，
int64 变 int32。一律写 `jt.array(v, dtype="float64")`。

## 6. 一轮完整验证

```bash
# 1. 自检导入树（见 §1）
# 2. 定向测试，CPU-only，跑两次（第一次吃掉 jit_utils updated）
JITTOR_TEST_DEVICES=cpu nvcc_path="" pytest tests/core/test_xxx.py -q
JITTOR_TEST_DEVICES=cpu nvcc_path="" pytest tests/core/test_xxx.py -q
# 3. 受影响目录，CPU-only
JITTOR_TEST_DEVICES=cpu nvcc_path="" pytest tests/core -q
# 4. CUDA 轮（编译更久，放最后一次跑）
nvcc_path=/usr/local/cuda/bin/nvcc pytest tests/core/test_xxx.py -q
```

## 7. 改代码生成器：先证明生成结果没变，或只按预期变

改 `opt/pass/**`、`op_compiler.cc`、tuner 这类**产出源码**的代码时，「测试绿了」
是很弱的证据：生成器的输出空间远大于测试覆盖的那几种形状，一个只在
`parallel depth=4 且有归约` 时才出现的形状变化，全套测试可以一条都不碰。

可靠的判据是**逐字比对生成的源码**。Jittor 把每个 JIT kernel 的源码落盘，
`jt.profile_scope` 的报告里就有路径：

```python
with jt.profile_scope(compile_options={"parallel":1, "max_parallel_depth":d}) as rep:
    b = (a+a).data
assert np.allclose(...)          # 数值也要断言，形状对不代表值对
src_path = rep[1][1]             # rep[0] 是表头；[1] 是 FileName 列
open(out_dir + tag + ".cc", "w").write(open(src_path).read())
```

做法：写一个 dump 脚本，覆盖你这次改动可能触及的**每一个分支**（CPU/CUDA、
有无归约、`max_parallel_depth` 1..N、`merge_loop_var` 0/1），改前跑一次存一份，
改后跑一次再存一份，`diff` 两个目录。

判据分三档：

- **该没变的必须零字节变化。** 例如只动 CPU 分支时，CUDA 的每一个 dump 都必须
  一个字节不差。有一个变了就说明你以为的「只动 CPU」不成立。
- **该变的，变化必须逐条能解释**，并且你能说出为什么新旧两种写法数值恒等。
- 数值断言与 dump 写在同一个脚本里，**不要分开跑**：形状对而值错是最容易漏的一种。

dump 脚本按分区放在自己的 `$TMPDIR` 下，不要提交；把 diff 摘要写进提交说明。

### 顺带会查出来的东西：「它对，但对的原因是别人恰好不动它」

代码生成器里有大量跨 pass 的隐式约定（谁先跑、谁设了哪个 attr、生成的语句
是什么文本形状）。改动时经常发现某段代码**只在另一个 pass 恰好不碰它时才正确**。

判断方法：把这段代码依赖的前提写成一句话，然后问「**谁保证这句话成立**」。
如果答案是「另一个 pass 的某个 if 恰好把它排除了」，而那个 if 的条件与这段代码
毫无关系（例如按 `dtype` 里有没有 `"__global__ void"` 来过滤），那就是一个
未声明的巧合——不是 bug，但下一次有人放宽那个 if 就会静默出错。

这类发现**比修复本身值钱**，务必写进提交说明：写清依赖的前提、今天由谁兜底、
以及你的改动是消除了这个依赖还是只是搬了个位置。
