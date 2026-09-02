---
name: jit-compile-failure-attribution
description: 怎么让一个算子的 JIT 编译失败——用来验证「编译错误报的是不是闯祸的那个算子」，以及为什么关掉 use_parallel_op_compiler 修不了这件事。测试类要按算子逐条生成、想动 use_parallel_op_compiler、或者遇到「上一条用例的错误出现在下一条用例上」时读这一篇。
---

# 让一个算子编译失败，然后看错误报给了谁

## 什么时候用

- 你要改一个「一条用例一个算子」的电池组（`test_ops.py`、`test_device_parity.py`），
  想开并行编译或者分片。
- 你看到「A 用例单独跑绿，跟在 B 后面跑红，报的还是 B 的错」。
- 你打算写 `use_parallel_op_compiler = 0` 来「让错误归到对的算子头上」。
  **先读下面第 3 节，这个方子不管用。**

## 1. 怎么让恰好一个算子编译失败

坏的 `-D` 没有用：没人用的宏照样编过。要让**编译驱动本身**失败：

```python
BAD = {"FLAGS: -include /nonexistent-header-for-this-probe.h ": 1}
with jt.flag_scope(compile_options=BAD):
    poisoned = a.sqr() * 7.0        # 只有这个子图带上坏 flag
```

`compile_options` 里所有 `FLAGS:` 开头、值为真的键，去掉前缀后被追加进这个算子的
编译命令（`op_compiler.cc` 的 `add_compile_flags`）。`jt.flag_scope(compile_options=...)`
给作用域内新建的 Var 打上这个 loop_option，`var.compile_options` 可以读回来确认打上了。

**确认它真的生效**：`print(poisoned.compile_options)`。打不上就是白测。

## 2. 报错里有什么

`jt.sync_all(True)` 抛出的信息里有三样东西：

- `[Error] source file location: <cache>/.../jit/__opkey0_...cc` —— 出事的 JIT 源文件；
- `Compile fused operator(i/n) failed: [Op(...), Op(...), ...]` —— **融合算子会把成员全列出来**，
  所以「归因到算子」在有融合的情况下天然是「归因到一组算子」；
- `Reason: ... Source read failed: /nonexistent-header-for-this-probe.h` —— 你下的毒。

判据写成「报错里出现我下毒的那个字符串」，不要写成「报错里出现某个算子名」——后者会被融合
和 pass 改写弄脏。

## 3. 关并行编译器修不了「错误跑到下一条用例上」

**这条是本 skill 的核心结论，实测得到，两个方向都验过：**

```
poisoned = <编译失败的 Var>；保持引用
sync_all(True)   -> 抛出，带毒标记          （第一条用例：正确归因）
unrelated = jt.array([4.,5.]) * 2
sync_all(True)   -> 又抛出同一个错误        （下一条用例：被冤枉）
unrelated.sum().item()  -> 18.0            （只 fetch 它自己是好的）
del poisoned; gc.collect()
sync_all(True)   -> 干净                    （放掉引用才解毒）
```

**第二步在 `use_parallel_op_compiler=0` 下同样复现。** 这是生命周期问题不是并发问题：
编译失败的 Var 只要还被引用就留在图里，此后每一次**整图** sync 都把它再跑一次。
串行编译只是让第一次的报错里融合算子小一点，它不改变「下一条用例被冤枉」。

**失败用例里是谁在持有那个 Var**：异常的 traceback——失败帧的局部变量。测试方法返回之后
没有任何活着的名字指向它，但 pytest 要留着 traceback 做报告。所以解法是显式放掉：

```python
try:
    self._compare(op)
except Exception:
    traceback.clear_frames(sys.exc_info()[2])   # 放掉失败帧的局部变量
    gc.collect()
    raise                                       # 报告的行号和信息都还在
```

代价只有 `--showlocals` 会少看到东西。

## 4. 必须在子进程里测

被下毒的运行时按定义会外溢到同进程的后续用例——那正是本 skill 要证明的性质。用
`_helpers.child_process.run_child_script` 起子进程，父进程里断言子进程的输出。
在父进程里直接下毒 = 把整条测试文件的后半段一起毒掉。

## 5. 现成的实现

`tests/compiler/test_parallel_compile_attribution.py`（4 条，CPU，27 秒）。
要改电池组的并行策略之前，先跑它一遍；要新增一种「编译失败该怎么报」的契约，
加在那里而不是新开一个文件。
