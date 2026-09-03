---
name: jittor-core-cpp-edit-loop
description: 改 python/jittor/src 下 C++ 核心时的编辑—重编—验证循环。包含隔离缓存与解释器选树的自检、把重编从 ~10 分钟压到 ~30 秒的 CPU-only 循环、每次 C++ 改动后第一次 pytest 必然失败的"jit_utils updated"陷阱、"读到未初始化字节"这类静默错值的复现判据，以及怎么给 UB 类改动找到可达后果、并在禁止 git stash 的前提下跑出「修前失败」那一轮。
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

### 四个会让 A/B 比对说谎的坑（都踩过）

1. **JIT 缓存会让「改后」重用「改前」的源码。** 同一个 `JITTOR_HOME`、同一组
   `compile_options`，第二次跑直接命中缓存里的 `.cc`，diff 全绿——而你什么都没测到。
   解法是给每一轮一个**只改 jit key、不进生成文本**的整数标记：
   `co["_ab"] = 1001` / `1002`。别为此换 `JITTOR_HOME`，那是一次十分钟的冷编译。
2. **`compile_options` 的值必须是 int。** 写成字符串会抛
   `Check failed: is_type<typename T::key_type>(key) && is_type<...>(value)`，
   来自 `py_converter.h`，与代码生成毫无关系，很容易被当成自己改坏了。
3. **`rep[1][1]` 只是其中一个 kernel。** 一条表达式常常产生**多个**融合算子
   （`mean` 加 broadcast 就是两个），profile 表里各占一行，而行的顺序不稳定：
   两轮各抓到不同的那一个，diff 里就出现一段与你的改动无关的"差异"。
   要比对就把每一行里存在的 `.cc` 路径**全部**读出来，排序后拼成一份再 diff。
4. **函数名里有 jit key 的哈希**（`func_b8376c61c41d5631_0`）。加了 §1 的标记之后
   这个哈希必然不同，比对前统一替换掉：`sed -E 's/func_[0-9a-f]{8,}_/func_HASH_/g'`。

另外：dump 脚本里的数值断言不要只写 `rtol`。`a - mean(a)` 这类结果贴着零，
`np.allclose(x, y, rtol=1e-4)` 会失败，看起来像"改动把值算错了"，其实是少写了 `atol`。

### 顺带会查出来的东西：「它对，但对的原因是别人恰好不动它」

代码生成器里有大量跨 pass 的隐式约定（谁先跑、谁设了哪个 attr、生成的语句
是什么文本形状）。改动时经常发现某段代码**只在另一个 pass 恰好不碰它时才正确**。

判断方法：把这段代码依赖的前提写成一句话，然后问「**谁保证这句话成立**」。
如果答案是「另一个 pass 的某个 if 恰好把它排除了」，而那个 if 的条件与这段代码
毫无关系（例如按 `dtype` 里有没有 `"__global__ void"` 来过滤），那就是一个
未声明的巧合——不是 bug，但下一次有人放宽那个 if 就会静默出错。

这类发现**比修复本身值钱**，务必写进提交说明：写清依赖的前提、今天由谁兜底、
以及你的改动是消除了这个依赖还是只是搬了个位置。

## 7. 给 UB 和「只是不该这么写」的改动写修前失败的测试

整改里有一类任务没有错的返回值可断言：字段被类型双关、`std::next(end())`、setter 与赋值的
顺序。**不要试图直接观测 UB**——它按定义没有稳定表现，写出来的断言换个编译器就翻。
做法是找**同一个缺陷的可达后果**：先问「这个缺陷让代码没法做到什么」，再从那里找必然的失败。

| 缺陷形态 | 直接观测 | 可达后果（拿它写用例） |
| --- | --- | --- |
| 字段被类型双关（`Var::allocator` 里存 `Var*`） | 对 Var 发虚调用，行为不定 | 双关让 `alloc` **没法**问「源 var 分配过没有」，于是 `x->allocator->share_with(...)` 是空指针虚调用。`b.share_with(a)` 而 a 从没执行过 → 必然段错误，`gdb` 栈顶就是 `Var::alloc` |
| 迭代器 `std::next(end())` | libstdc++ 绕回 `begin()`，"看起来没事" | `sync_ptr` 变成 `begin()` → `top_weak_sync` 第一行就 break → weak sync 从此静默不工作。判据用 `jt.number_of_lived_ops()`（是函数不是属性），**两个子进程跑同一个探针**（触发/不触发各一次）比大小，比断言绝对值稳 |
| 共享的 per-node 草稿槽（`Node::custom_data`）、共享的全局 epoch（`tflag`） | 构造两个遍历交错的时序 | **去找代码里已有的那份手工补丁，把它删掉**。补丁的存在就是证据：`memory_profiler.cc` 把整个 `custom_data` 抄出来再抄回去（它从 `run_sync` 的算子循环里被调用），`grad.cc` 把下标抄进 `id_buffer`（构造反向算子会重入 `run_sync`）。删掉 memory_profiler 那六行、别的不动 → `fused_op.cc Check failed: outputs().size()`。失败点离原因隔两个文件，正是这类缺陷的形状 |
| 编译期不变量只写在注释里（「bit 28 在两套布局里都空着」） | 无法运行期观测 | 两头都要：`static_assert` 把不变量变成编译错误，**加**一条静态用例断言那些 `static_assert` 存在、且位号是枚举生成的而不是手写的数字。允许的名字要从头文件**自己减出来**（共享枚举减去两个私有枚举），手写清单在改动前会**空洞地通过** |
| setter 在赋值之前跑 | 每个 setter 自己回写，行为看着正确 | 找**旧顺序真的把 setter 的工作抹掉**的那一处：`setter_use_cuda` 无设备时把值回退成 0，紧接着的 `name = value` 又把 1 写回去 |
| 环境变量解析静默回退 | 有一句 warning | warning 是 'w' 级，`log_silent=1` 时被 `send_log` 吞掉：`log_v="1 " log_silent=1` 下**一个字都不打**，flag 还是 0。断言"什么都没打印" |

三条里有两条的可达后果是**进程级崩溃或静默失效**，所以这类用例一律 `subprocess.run`
起子进程断言退出码与输出，并显式传 `PYTHONPATH=<worktree>/python`（见 §1）。

如果确实找不到可达后果（纯可维护性重构），**在提交说明里直说**，用往返用例把契约钉住，
不要编一个看着像回归测试的空断言。

## 8. 怎么跑出「修前失败」这一轮（禁止 `git stash`）

C++ 改动没有开关可切，只能真的拿旧代码编一次。两种做法，按代价选：

**A. 换回文件重编**（同一个缓存，适合改动只在一两个文件里）

```bash
cp python/jittor/src/xxx.cc $SCRATCH/xxx.cc.fixed      # 先存好，别只靠 git
git checkout -- python/jittor/src/xxx.cc
<跑用例，记录失败输出>
cp $SCRATCH/xxx.cc.fixed python/jittor/src/xxx.cc      # 换回来
```

**B. 拿一棵干净的源码树另建一个缓存**（不动自己的工作树，可以在别的任务跑着的时候做）

```bash
git archive HEAD | tar -x -C $SCRATCH/pristine          # HEAD = 改动之前的状态
PYTHONPATH=$SCRATCH/pristine/python \
JITTOR_HOME=$SCRATCH/pristine_home TMPDIR=$SCRATCH/pristine_tmp \
nvcc_path="" python $SCRATCH/probe.py                   # CPU-only：整棵树 86MB，冷编约 1 分钟
```

B 的好处：**工作树一直是修好的状态**，不用来回换文件，也不会跟正在跑的长测试打架；
CPU-only 缓存小得多。前提是复现路径与 CUDA 无关——`Var::alloc` 的空指针、weak sync、
环境变量解析都满足，`setter_use_cuda` 的回退不满足（要 `HAS_CUDA`），那一条只能用 A，
或者在应用该任务的补丁**之前**先在当前构建上跑一次探针。

两种做法都要记得：换过源码之后第一次 pytest 必然吃一个 "jit_utils updated"（见 §4），
要原样重跑一遍。

### 类型系统能替你找出「靠巧合才对」的读法

把一个「谁都能读」的字段改成「必须先知道种类才能读」之后，**编译错误的清单就是审计
报告**。2.01 把 Var 与 Op 的 flag 拆成两个枚举、私有位只能经 `Var::flag`/`Op::flag`
读写，编译器立刻指出 `executor.cc` 的 `run_sync` 在一个同时装 var 和 op 的队列上读
Op 专用的 `_has_gopt`，随后对**任何答是的节点**调 `n->op()->graph_optimize()`。它今天
不出事，只因为那个位号在 Var 布局里恰好没人用。

**做法**：改完类型先只跑 `g++ -fsyntax-only`（全树 146 个核心 .cc，`xargs -P 8`，
十几秒），把错误逐条读成「这里为什么可以不问种类」。比编译整个核心快一个数量级，
而且这一步的产出（每条错误对应的判断）比改动本身值钱。

## 8. 构建锁会把你饿死：换一个 `JITTOR_HOME`，不要杀进程

`JITTOR_HOME` 里有一把全局构建锁（`.../jittor.lock`）。同一个 `JITTOR_HOME` 下的
所有进程共用它，**一个长跑的门禁会不停地拿锁放锁**（每编一个 JIT kernel 一次），
后来的进程可能几十分钟拿不到锁。表象是「我的测试卡住不动、日志一个字都没有」。

判据：

```bash
grep "waiting for build lock" <你的日志>   # 拿不到锁，不是编译慢
```

正确做法是**再开一个 `JITTOR_HOME`**，不是杀掉那个门禁：

```bash
export JITTOR_HOME=/home/zy/jittor-lab/refactor/_home/<分区>-b
export TMPDIR=/home/zy/jittor-lab/refactor/_tmp/<分区>-b
```

代价是一次冷构建（CPU-only 约 60s，CUDA 约 10 分钟）与约 350MB 磁盘，
远小于等锁的时间。跑完把 `-b` 目录删掉。

**不要为了抢锁去 kill 那个门禁进程**：它可能正在编译，留下的损坏缓存会在
毫不相干的算子上大面积报错，而且你还得把整个 `JITTOR_HOME` 删掉重来。

## 9. 「修前失败」怎么证明：换文件，不要 stash

修「静默算错」必须有一个修前失败、修后通过的测试。C++ 改动里证明这一点的
正确做法是**只回退实现、保留测试**：

```bash
cp <改过的实现文件> $MYTMP/impl.fixed        # 先备份
git checkout -- <改过的实现文件>              # 只回退实现，测试还在
<跑测试>                                     # 必须失败，而且失败信息要对得上症状
cp $MYTMP/impl.fixed <改过的实现文件>         # 还原
<跑测试>                                     # 必须通过
```

**不要用 `git stash`**（栈是全仓库共用的，见 `git-worktree-shared-state`）。

判据不是「红了」，是**红的原因就是你要修的那个**。例如把死代码消除的判据从
「语句含 `void` 一词」改成结构化识别时，修前的失败信息里打印出来的整个函数体
是空的——这才叫对得上。只看到 `assert False` 说明不了什么。

## 10. 缺陷只在某个配置下可达时，测试要绕开那个配置

代码生成器里有整段只在特定编译器或 flag 下才跑的代码（`UnrollPass` 只在
`cc_type == "icc"` 时进流水线；`SharedReducePass` 只在 `para_opt_level >= 4`
时进）。这类地方的缺陷，**运行期怎么测都测不到**——包括在注册/初始化时加校验，
因为那段根本不会执行。

做法是把断言下移到不需要那个配置的层面：

- 名字冲突 → 把每个 pass 各构造一个，断言名字两两不同（构造 pass 不需要流水线）；
- 默认关着的 pass → 用例里显式打开那个 flag（`jt.flags.para_opt_level = 4`），
  并且**每个用例换一个 `compile_options` 值**，否则第二个用例拿到的是缓存里
  上一个配置编出来的 kernel；
- 只在某个后端出现的形状 → 直接构造那段 IR 喂给要测的函数，不要指望端到端跑出来。

写用例时把「为什么不能端到端测」写进 docstring，否则下一个人会以为你偷懒。
