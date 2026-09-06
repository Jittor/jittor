# 审查 1.01：`data.gz` 五个翻译单元的反混淆还原是否忠实

- Status: Accepted；结论「有差异但无害」，逐条差异见下；原验收判为不充分
- Date: 2026-09-06
- Baseline: 被审对象 `ecb6a1128`（含其后两个补注提交 `812714d54`、`dbe693378`）；
  审查在 `wk/pyother` 的 `f5a0ec5a1` 上做，两个被比对的源码树用 `git archive`
  从 `ecb6a1128^` 与 `ecb6a1128` 导出，不改任何分支
- Owner: Jittor 2.0 整改审查者（独立于 1.01 的执行者）
- Review when: 五个 TU 中任一个的语义被改动；或再有「只以二进制形式发布的翻译单元」
  需要还原时（本文第 6 节的验证流程可直接复用）

## 结论

**有差异但无害。** 还原忠实于 `data.gz` 里的混淆原文，覆盖率是**五个翻译单元的
全部 token**（原文 13185 个 token 逐个对上，无一未归类）。共发现 **7 类、约 70 处**
文本差异，逐条见第 3 节；其中没有一处在可达输入上改变语义，两处只在原实现本身
已经越界（UB）的输入上表现不同。

但**原验收（「31 个符号 + 三套门禁不变」）不足以支撑这个结论**——它对 7 类差异中的
7 类全都是盲的。见第 5 节。

三处此前未被披露、我认为应当披露的差异：
`run_liveness_queue` 里**多加了两条 `LOGvvvv`**（原文没有，见 3.6，这是「加」不是
「还原」）；`try_atomic` 的 `||` 链里**两条逐字重复的子句被合并**（3.3）；
`tune_atomic` 少了一个**从不被读的参数**（3.4）。三者都可证明无害，但读者若拿还原
后的代码与二进制对照，会对前者得出错误结论。

未登记 `1.06`：没有实质差异。

## 1. 先证明「我手里的原文就是原文」

`python/jittor/utils/data.gz`（437798 字节，sha256 `18d6f920…`）是普通 gzip，
解出 569 行、1558091 字节的 `data.cc`。按 skill 描述的手法独立重做一遍展开：

```
sed -E 's|^#include (.*)$|JTINCLUDE \1 JTENDINCLUDE|' data.cc > noinc.cc
cpp -P -D'_P(...)=' noinc.cc > expanded.cc          # 69141 字节，13185 个 token
```

**这一步必须自己验证，否则后面的比对全部悬空。** 把 `expanded.cc` 的 `JTINCLUDE`
标记还原成 `#include`，用 `compiler.py` 当年那条命令的**同一套 flag**
（`(cc_flags+opt_flags)` 去掉 `-Wall`/`-Werror`/`-shared`，加 `-include src/utils/vdp`）
编译，与构建自己产出的 `data.o` 对比：

| 对比项 | 结果 |
| --- | --- |
| 我的 flag 是否复现构建的 `data.o` | 是。598 个符号逐条相同，只差 TU 名符号 `_GLOBAL__sub_I_*` |
| 符号（去掉 `.LC*` 常量池标签） | **240 : 240，完全相同**，只差 TU 名符号 |
| 反汇编指令条数 | **67866 : 67866** |
| 归一化后的指令流（地址、常量池标签、TU 名归一） | 67866 行里 **60 行**不同 |
| 这 60 行的差异内容 | 只有 8 处 `lea`/`mov` 的寄存器选择（`%rbx`/`%rbp` ↔ `%r12`） |
| **只取助记符序列逐条 diff** | **完全相同，退出码 0** |

即：每一条指令、顺序都一样，只有 8 个寄存器分配不同（`__FILE__` 从 `data.cc` 变成
`myexp.cc`，字符串长度变了，常量池随之微调）。**所以 `expanded.cc` 就是 `data.cc`**，
后面拿它当基准是成立的。

`#include` 分组给出五个 TU 的顺序：`atomic_tuner_pass`、`node`、`fuser`、
`shared_reduce_pass`、`parallel_pass`。这与代码体的实际顺序一致（第一段就有
`"tune_atomic"` 字符串）。

## 2. 符号级：独立确认 31，且两侧集合逐条相同

`data.o` 的 `B`/`D`/`T` 符号：**31 个 `T` + 7 个 `B`/`D` = 38**，与验收声称的 31 一致。
还原后五个 `.o` 取并集：**38**。差异只有 10 条，一一对应，参数类型逐字相同：

| `data.o` | 五个 TU | 类型 |
| --- | --- | --- |
| `D x57074` | `D get_thread_range_log_src` | `const char*` 全局，内容是一段生成代码的源文本，逐字相同 |
| `T x1840(Op*, Var*, bool)` | `get_reduce_init_code` | 同签名 |
| `T x4674(int)` | `round_down_pow2` | 同签名 |
| `T x42722(string const&, int)` | `parse_int_at` | 同签名 |
| `T x48817(KernelIR*, expr::Expr*)` | `expand_offset_expr` | 同签名 |
| `T x54412(unique_ptr<KernelIR>&, vector<unique_ptr<KernelIR>>&)` | `find_atomic_kernel` | 同签名 |
| `T x95099(unique_ptr<KernelIR>&, FusedOp*)` | `find_reduce_op_id` | 同签名 |
| `T x27987(unique_ptr<KernelIR>&)` | `rewrite_atomics_to_shared_reduce` | 同签名 |
| `T x79738(unique_ptr<KernelIR>&, ReduceOp*)` | `plan_reduce_thread_order` | 同签名 |
| `T x15175(unique_ptr<KernelIR>&, unique_ptr<KernelIR>&, ReduceOp*)` | `apply_reduce_thread_order` | 同签名 |

没有多出来的符号（没有把 static 提成外部可见），也没有少掉的（没有丢函数）。
提交说明里那张对照表**逐条复核无误**。

## 3. 语义级：token 逐字对照 + 每函数 α 等价

方法：把 `expanded.cc` 与还原后五个 `.cc` 都 token 化（字符串统一解 `\xNN` 转义、
整数字面量统一按值归一，所以 `"\x72\x76"` 与 `"rv"`、`0x100` 与 `256` 会判为相同），
丢掉注释、空白与 `#include`，然后贪心对齐：token 相同则前进；原文是 `xNNNN`
而还原侧是标识符则记入改名表并前进；否则报为结构差异并用 difflib 重新同步。

原文 13185 token，还原侧 12707 token，恢复出 **213 个混淆标识符**的映射，
**99 处结构差异**，全部归类如下。

### 3.1 成员函数指针（34 处）— 已披露

原文把 `&Node::release_forward_liveness` 这种**成员指针**强转成 `void(*)(Node*)`
再 `op(node)` 调用。这是 UB；`-Wall` 独立确认了这一点：**35 条
`-Wpmf-conversions` 警告**。还原改成真正的 `void (Node::*)()` 与 `(node->*op)()`。
队列是文件内 static，不影响任何导出符号。`812714d54` 已写明。无害。

### 3.2 有符号/无符号与类型（约 25 处）— 未披露，且**没有必要做**

`int i` → `uint`/`size_t`/`auto`；新增 `(int)` 强转；`!= -1` → `!= string::npos`；
`i < n-1` → `i+1 < n`。

我原以为这是为了在核心的 `-Wall` 下编译干净——**实测不是**。`src/common.h:24`
有 `#pragma GCC diagnostic ignored "-Wsign-compare"`，全仓库每个 TU 都吃这条；
把去混淆后的原文加 `-Wall` 编一遍，`-Wsign-compare` **0 条**（只有 35 条
`-Wpmf-conversions`、10 条 `-Wmisleading-indentation`（我单行排版的产物）、
2 条 `-Wunused-variable`）。也就是说这些类型改动既非 `-Werror` 所迫，也不消除任何
警告，是**多余的改动**，出现在一个以「忠实还原」为目的的文件里。

逐处核对后仍判无害：所有比较在可达取值上（下标非负、容器规模 < INT_MAX）
两侧结果一致。两处只在原实现已经越界时不同：

- `apply_reduce_thread_order`：原文 `int i=0; i < order.size()-1`（`int` 提升为
  无符号）在 `order` 为空时 `size()-1` 是 `SIZE_MAX`，循环会越界读；还原的
  `i+1 < order.size()` 不会。
- `tune_atomic` 的 `for (int j = order.size()-1; j >= split_pos; j--)`：`split_pos`
  是 `uint`，所以原文这个比较是**无符号**的，`j` 变成 `-1` 时会判真。实际到不了：
  `split_pos` 的求法（`order_owner[0] != order_owner[0]` 恒假，不会在 0 处 break）
  保证 `split_pos >= 1`，循环在 `j == split_pos-1 >= 0` 时结束。

### 3.3 `try_atomic` 链里两条逐字重复的子句被合并（1 处）— 未披露

原文的 `||` 链有 **13 个** `try_atomic(...)` 子句，其中 `::max` 与 `::min` 各出现
**两次，六个实参逐字相同**；还原后是 **11 个**，每种一次。

可证明等价：`||` 短路，第二次同参调用只在第一次返回 false 时才会执行；`try_atomic`
的两条 false 出口都在任何修改之前（`expr::match` 不匹配、`(c[d])` 不匹配），
唯一被写的是传引用的 `matches` 暂存向量，同参重算写入相同内容。所以调一次与调两次
的可观察状态和返回值都一样。

### 3.4 `tune_atomic` 去掉一个从不被读的参数（1 处，diff 里表现为 4 处）— 未披露

原文 `static void x74384(Pass*, KernelIR*, bool, int x18677, vector<vector<int>>&, vector<string>&)`，
`x18677` 在函数体里**一次都没出现**，唯一调用点传常量 `4`。还原去掉了它。
`tune_atomic` 是 static，不影响符号表。无害。
（`x18677` 在 `parallel_pass` 的 `replace_with_atomic` 里确实被用作 `parallel_depth`，
那是另一个函数——混淆器按原名映射，同一个 `xNNNN` 会横跨多个函数。）

### 3.5 六个死变量被删（未披露，无害）

| 原文 | 位置 | 为什么是死的 |
| --- | --- | --- |
| `static vector<int64> x27600` | node.cc | 全文只出现 2 次：声明与 `.clear()`，只写不读 |
| `auto x9433 = liveness_queue.size()` | `run_liveness_queue` | 只出现 1 次，`-Wunused-variable` 独立报出 |
| `auto need_release = forward_liveness && backward_liveness` | `finish_pending_liveness` | 只赋值不读，`-Wunused-variable` 独立报出 |
| `vector<pair<unique_ptr<KernelIR>*,string>> x94021` | `rewrite_atomics_to_shared_reduce` | 全文只出现 1 次 |
| `string x68756 = code.substr(code.find("&(")+2, …)` | 同上 | 该函数内只出现 1 次 |
| `string x71901 = found.first` | `SharedReducePass::run` | 该函数内只出现 1 次 |
| `int root = -1`（随即被覆盖） | `count_fuse` | 死初值 |

一处需要点明：`x68756` 那句在 `find("&(")` 返回 `npos` 时会 `substr(npos+2, …)` 抛
`std::out_of_range`。删掉它去掉了一条**在本函数处理的字符串上不可能发生**的抛出
（代码形如 `atomicAdd(&(x[i]), y);`，必含 `&(`）。

### 3.6 `run_liveness_queue` 里多加了两条 `LOGvvvv`（未披露）— 这是「加」

原文的 `run_liveness_queue` **没有任何日志**，而 `liveness_op_name`（原 `x8006`）
在整个 `data.cc` 里**只有定义、没有调用点**（全文 1 次出现）。还原后加了

```
LOGvvvv << "run liveness queue from" << caller << "size" << liveness_queue.size();
…
LOGvvvv << "liveness" << liveness_op_name(op) << (void*)node;
```

于是原本是死代码的 `liveness_op_name` 被激活。node.cc 区段的 `LOGvvvv` 数量
原文 3 条、还原后 5 条。默认日志等级下（`LOGvvvv` 要 v>=4）不产生任何输出，
不改变数值结果；两者都是文件内 static，符号表看不见。**但这不是还原，是添加**，
提交说明与两个补注提交都没提。上游 jittor 大概本来有这两行（否则不会留一个
只为日志服务的死函数），可我审的是「还原是否忠实于 `data.gz`」，就这一点而言
它不忠实。

### 3.7 纯排版（其余若干处，无害）

函数末尾多余的 `return;`、单语句外的花括号、初值列表末尾多一个逗号、
`int t = f(); v.push_back(t)` 合成 `v.push_back(f())`、把参数名 `ir` 改成 `call`
（这些是自由函数，`ir` 只能是那个参数，不存在与 Pass 成员 `ir` 混淆的问题）。

### 3.8 串位检查：改名映射在每个函数内是否双射

全局改名表里有「一个 `xNNNN` 对多个名字」和「一个名字对多个 `xNNNN`」，但这**不是
错误信号**：混淆器按原名映射，所以同一个 `xNNNN` 会横跨多个作用域。要判串位得看
**函数内**。做法是按顶层定义切段，把两侧参与改名的 token 各按**在该定义内首次出现
的顺序**编号，再逐位比较序号序列——纯改名时两条序号序列相同，不论选了哪个英文名；
若把两个变量合并成一个名字，还原侧会复用较早的序号，序列在该处分叉。

37 个顶层定义中 **29 个逐位 α 等价**。8 个分叉，其中 6 个是「还原侧序号更大」
= **拆分**（原文一个名字在不相交作用域里复用，还原按用途分别命名，安全，
如 `apply_reduce_thread_order` 里 `x10662` 在值循环里叫 `tn`、在下标循环里叫 `i`，
两处都对）。只有 2 个是「还原侧序号更小」= 合并方向，都已逐行读过：

- `count_fuse`：`other` 既是 `edge_fusable` 的第三个形参、又是
  `for_each_neighbor` 里的兄弟算子。两个**不同的 lambda**，作用域不相交。逐行读过
  整个 `count_fuse`（并查集、`edge_fusable` 的两组规则、`for_each_neighbor` 的三个
  分支、最后的 `var_fused` 三态判定），包括
  `edge_fusable(var, other, op, 0)` 与 `edge_fusable(var, consumer, producer, 1)`
  两个调用点的**实参顺序**，全部与原文一致。
- `rewrite_atomics_to_shared_reduce`：删掉死变量 `x68756`（3.5）导致其后序号整体
  平移，非合并。

## 4. 差分测试

两棵源码树，各自独立构建一份核心，跑同一批用例，**串行、同一顺序**
（`tests/core` 的存活 Var 计数在并发下不稳，必须串行）。

### 4.1 CPU

`tests/core tests/compiler tests/ops`：

| | `data.gz` 构建的核心 | 还原后五个 TU 构建的核心 |
| --- | --- | --- |
| 结果 | **722 passed, 322 skipped, 126 warnings** | **722 passed, 322 skipped, 126 warnings** |

两份日志去掉树路径前缀与耗时后**逐字节相同**（81 行对 81 行，唯一差异是那行墙上时间）：
同样的用例通过、同样的用例跳过、同样的警告出现在同样的行号。

### 4.2 CUDA（三个 pass 是 CUDA 代码生成器，纯 CPU 覆盖不到）

`pass/fail` 对代码生成器是钝器，所以改成**比生成物**：同一批固定负载在 GPU 上跑完，
记录每个结果的数值，并把 JIT 生成的每个 kernel 源码整篇存下来比对。

| | old（data.gz） | new（还原） |
| --- | --- | --- |
| 负载 | 77 | 77 |
| 生成的 kernel 个数 | 29 | 29 |
| kernel 文件名集合 | — | **完全相同**（文件名里含 JIT key，即 pass 的决策） |
| **kernel 源码逐字节** | — | **29/29 相同，0 处不同** |
| 数值逐比特相同 | — | 50/77 |

剩下 27 个数值差在 `1e-9 ~ 9.2e-7` 相对量级。**对照实验**：把 old 这一侧**再跑一遍**，
与它自己第一次比 —— 差异的是**同样的 27 个负载，集合完全一致**；
「跨核心不同但同核心两次能复现」的负载有 **0 个**。所以这 27 个是 GPU 原子加的
求和顺序不确定造成的 float32 舍入，不是还原引入的。

### 4.3 `SharedReducePass`：默认关着，我把它打开了

还原出来的 `SharedReducePass::run()` 第二行确实是 `if (para_opt_level < 4) return;`，
而默认值是 3——**所以三套门禁对这个 pass 的覆盖是零**，两侧都是零，
「门禁不变」在这里不构成任何证据。强行 `jt.flags.para_opt_level = 4` 再比：

| | old | new |
| --- | --- | --- |
| 6 个归约负载 | 全部 ok | 全部 ok |
| 生成的 kernel | 31 | 31 |
| **kernel 源码逐字节** | — | **31/31 相同** |
| 真正用到 `shared_reduce<>` 的 kernel | 5 | **同样的 5 个** |
| 数值逐比特相同 | — | 5/6（第 6 个是全量归约，走原子加，3e-7） |

即：这个在整个负载里从未生效的 pass，被强制打开后两侧生成**逐字节相同**的代码。

### 4.4 覆盖率与边界

- **静态对照覆盖五个 TU 的 100%**：原文 13185 个 token 全部参与对齐，99 处差异全部
  归类，没有「没看的部分」。CUDA-only 分支同样被这层覆盖——token 比对不关心代码
  最终在哪个设备上跑。
- **动态对照的覆盖是不完整的，且我没有量化它**：没有统计行覆盖率。
  `AtomicTunerPass` 与 `ParallelPass` 在 CPU（OpenMP）与 CUDA 两侧都跑到了；
  `SharedReducePass` 只有 4.3 那 6 个负载、5 个 kernel。
- **没有验证**：ROCm/NPU 路径；多卡；`para_opt_level` 的其他取值；
  `fuser.cc` 的 `_force_fuse`/`_out_hint` 少见分支是否被 4.1 的用例走到（未测量）。
- **不做任何计时**：机器上八个分区并行，负载在 7–22 之间波动，本文任何结论都不
  依赖耗时。

## 5. 原验收「31 个符号 + 三套门禁不变」够不够

**不够。** 第 3 节的 7 类差异，这套验收**一类都发现不了**：

1. **符号相同不代表函数体相同。** 3.1–3.7 全部发生在函数体内部，符号集合纹丝不动。
2. **符号表对 static 完全失明。** 提交说明自己写了这一点，是诚实的；但恰好有四项
   发现落在 static 里：`tune_atomic` 少的那个参数、`run_liveness_queue` 多的两条
   日志、死掉的 `x27600`、被激活的 `liveness_op_name`。
3. **「三套门禁不变」在 `SharedReducePass` 上是空判据**：它默认关着（4.3），
   门禁跑到的行数是 0。用「门禁不变」为它背书，等于用没执行过的代码作证。
4. 门禁是「不变 = 没变坏」的弱信号，而反混淆的典型错误（改名串位、常量抄错、
   `||` 漏一项）恰恰是**静默**的：本轮 CPU 全绿、CUDA 生成物全同，也没能替我
   免掉第 3 节的逐条阅读——那 7 类差异全是读出来的，不是跑出来的。

**够用而且便宜的验收**，两步，都是分钟级，见第 6 节已写进 skill：
先证明「展开出来的原文就是原文」（编译展开结果，与随包的 `.o` 比助记符序列），
再证明「还原就是那份原文」（token 流逐字对齐 + 每函数 α 等价）。
第一步把「我读的东西对不对」钉死，第二步把「改写有没有越界」钉死；
符号比对只该当**冒烟检查**，不该当验收。

## 6. 删除旧路径有没有留下悬空引用

**没有功能性悬空引用。** 逐项核实：

- `compiler.py` 那 34 行删得干净。`use_data_gz`、`data.md5`、`data_o_path`
  全仓库无残留。被删代码里唯一用到的 `hashlib` 在同文件另外三处仍在用
  （`:77`、`:913`、`:1910`），不是失效 import。
- `python/jittor/src/utils/vdp` 已删，唯一用它的就是那条 `-include`。
- `__data__` 回退分支：`python/jittor/src/__data__` 确实不存在，两个分支都在过滤
  一个不会出现的路径，删得对。剩下的 `__data__` 提及无害：`.gitignore:76`、
  `agent/scripts/check_wheel_contents.py:59` 的排除名单。
- `tools/release/legacy/polish.py` 仍引用 `src/__data__` 与
  `jt.compiler.files`，但它在 `data_path.is_dir()` 处就 `raise`，是死代码，
  归 11.01，这里不动是对的。
- `agent/baselines/wheel-contents-*.txt` 里的 `data.gz`/`vdp` 条目是**历史快照**，
  描述的是过去的 wheel，不该改。

三处**文档残留**（不影响构建，但会误导读者）：

1. `agent/skills/jittor-refactor-gates/SKILL.md:247-249` 说三个 pass「git 里根本
   没有」、实现在 `data.gz` 里、`log_vprefix` 要写 `data=100` 才抓得到。1.01 之后
   全部不成立，而这是一份**给后续 agent 读的 skill**，照它写 `data=100` 会抓不到
   日志。**本次已改。**
2. `docs/locales/zh_CN/LC_MESSAGES/architecture/source-architecture.po:248` 还留着
   `python/jittor/utils/data.gz` 的 msgid，而英文源已删该条。未改（属翻译目录维护）。
3. `agent/design/refactor-plan.md:160`（1.05）与 `refactor-board.md:516` 仍把
   `data.gz`/`vdp` 的删除列为待办，实际 1.01 已做。**未改**：计划任务内容不由
   审查者改写。

## 7. 命令口径

两棵树用 `git archive` 导出，不新建 worktree、不动任何分支：

```
cd /home/zy/jittor-lab/refactor/pyother
git archive ecb6a1128^ | tar -x -C <R>/tree_old      # 含 data.gz
git archive ecb6a1128  | tar -x -C <R>/tree_new      # 含还原的五个 TU
```

构建与跑用例（`PYTHONPATH` 必须显式给，jt311 里 jittor 是 editable 安装、
`.pth` 指向别的树；每次首跑都打印过 `os.path.dirname(jittor.__file__)` 自检）：

```
cd <R>/tree_{old,new}
PYTHONPATH=<R>/tree_{old,new}/python \
JITTOR_HOME=<自己的 home>/review101{old,new} \
TMPDIR=<R> JITTOR_TEST_DEVICES=cpu nvcc_path="" \
taskset -c 4-11 <python> -m pytest tests/core tests/compiler tests/ops -q
```

CUDA 一侧加 `CUDA_VISIBLE_DEVICES=0 nvcc_path=/usr/local/cuda/bin/nvcc`
（本机 CUDA 可用，`cuda archs: [89]`；两棵树各自预热自己的 `JITTOR_HOME`，
不碰别人的）。符号与反汇编比对用的是 `nm -C --defined-only` 与
`objdump -dr --no-show-raw-insn`，同机、同 `g++ 12.3.0`、同 flag。

对照用的脚本（token 化、对齐、α 等价、生成物比对）留在 `_tmp/review101`，
未入库；方法本身写进了
`agent/skills/restoring-obfuscated-cpp-sources/SKILL.md` 新增的「验证还原是忠实的」
一节，那里给了可直接复制的命令。

## 8. 边界

- 本文只判 `ecb6a1128`（以及其后两个补注提交）对 `data.gz` 的**还原是否忠实**。
  不判这五个 TU 的**原有逻辑**是否正确，也不判 `ecb6a1128..HEAD` 之间 21 个提交
  对它们的后续改动——那些改动已使 HEAD 版本与被审版本不同（`fuser.cc` 262→266 行等），
  比对一律用 `git show ecb6a1128:<path>` 取被审版本。
- 未验证：ROCm/NPU、多卡、`para_opt_level` 的其他取值、动态对照的行覆盖率。
- 不做性能测量。
