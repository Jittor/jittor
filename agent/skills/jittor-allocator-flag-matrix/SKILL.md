---
name: jittor-allocator-flag-matrix
description: 复现和证明 Jittor 分配器缺陷的手法：flag 矩阵、下毒再断言、share_with 别名的观察与证伪、跨流时序竞争的稳定复现、check_graph 的覆盖判据。改动 src/mem/**、Var::alloc/share_with、migrate_to_cpu/gpu、fetch_op、getitem/setitem 的 inplace 路径，或看到"只在某些配置下结果不对/段错误/偶发不对"时用它——默认配置会掩盖几乎所有分配器 bug，测试不切 flag、不下毒、不做够轮数就等于没测。
---

# 分配器 flag 矩阵

## 为什么默认配置测不出分配器 bug

`get_allocator()` 组出来的是一摞分配器。默认那摞最外层是 SFRL，它**总是**写回
`allocation`（块 id）、**总是**支持 `share_with`。于是底层分配器的契约违反全被吃掉。
把 SFRL 摘掉或换掉，底层的行为才暴露出来。

| 配置 | `exe.allocator` 实际是什么 | 能暴露什么 |
| --- | --- | --- |
| 默认 | SFRL over (cuda_host \| cuda_device \| aligned) | 只有 SFRL 自身的问题 |
| `use_sfrl_allocator=0` | 裸的 aligned / cuda_host / cuda_device | 裸分配器不写 `allocation`；`share_with` 返回 false 的路径；零字节分配 |
| `use_nfef_allocator=1` | NFEF over 裸分配器（**从不释放**，且复用指针不经过底层） | 复用路径不写 `allocation`；指针复用导致的读到上一轮数据 |
| `use_temp_allocator=0` | workspace 不再走 TempAllocator | TempAllocator 的统计/嵌套问题 |
| `use_stat_allocator=1/2` | 中间插一层 StatAllocator | 分层转发是否把 `allocation` 透传 |
| `use_cuda_managed_allocator=1` | cudaMallocManaged | managed 路径 |
| `use_cuda_host_allocator=0` | CPU 侧不用 pinned 内存 | 有 CUDA 设备时 CPU 分配器的选择 |

**注意**：`cpu_allocator`（`array_op`、`fetch_op`、`Allocation(cpu_allocator, …)` 用的那个）
**永远是 SFRL**，不受 `use_sfrl_allocator` 影响。所以

> 直接用 `jt.array(...)` 造出来的 Var 走的是 SFRL，**测不出**裸分配器的问题。

要让被测 Var 由 `exe.allocator` 分配，必须让它是**某个算子的输出**：

```python
a = jt.array(np_value) * 1.0     # 这个 * 1.0 是必需的，不是凑数
a.sync()
```

## 复现脚本

[`probe_allocator_matrix.py`](probe_allocator_matrix.py) 跑「getitem/setitem 别名误判」
这一类，按 (设备 × flag 组合 × 切片下标) 报坏掉的比例：

```bash
cd <你的 worktree>
PYTHONPATH=$PWD/python \
JITTOR_HOME=<你的缓存> TMPDIR=<你的 tmp> \
CUDA_VISIBLE_DEVICES=<你的卡> nvcc_path=/usr/local/cuda/bin/nvcc \
taskset -c <你的核> python agent/skills/jittor-allocator-flag-matrix/probe_allocator_matrix.py
```

**`PYTHONPATH` 不能省**：jt311 里的 jittor 是 editable 安装，`.pth` 指向主树；
手写的 `python -c` / `python 脚本.py` 不加它就是在测别人的代码，症状是「bug 复现不出来」。
pytest 不需要（`tests/conftest.py` 已经处理）。

## 判据

1. **偏移量要选 0。** `share_with(x, offset)` 把 `offset` 写进 `allocation`。
   底层分配器不写回时残值多半也是 0（`free_var` 在释放前把 `allocation` 清成 0，
   Var 对象复用同一块堆内存），所以 `a[0]` 稳定复现、`a[1]` 基本不复现。
   写测试用 `a[0]`；只测 `a[1]` 会得到假绿。
2. **要跑够轮数并 `gc.collect()`。** 触发条件是 Var 对象的堆内存被复用，单次不一定命中。
   10 轮足够，坏的时候是 10/10。
3. **CPU 和 CUDA 都要跑。** 同一个 flag 组合在两边走不同的裸分配器
   （aligned/cuda_host vs cuda_device）。
4. **改了 `src/**` 就要重编。** 每个新进程约 1–10 分钟；把多条改动攒在一起编译一次。

## 已知的、与你的改动无关的失败（别去追）

在非默认配置下，下面这些在 2026-09 时就已经是坏的，改分配器不会修好它们，
写测试时避开或单独立项：

- `jt.arange()` / `jt.index()` 在 `use_sfrl_allocator=0` 下返回**全 0**（CPU 与 CUDA 都是）。
  所以测试的数据源要用 `jt.array(np.…)`，不要用 `jt.arange`。
- CUDA 上的归约（`x.sum()`）在 `use_sfrl_allocator=0` 与 `use_nfef_allocator=1` 下结果错误
  （只累加了一小部分）。CPU 正常。
- `src/test/test_sfrl_allocator.cc` 的两个计时用例（400 ms 硬上限，一次分配 20 GB）
  在有 CUDA 设备的机器上走 pinned 内存，本身就远超时限，且对机器负载极其敏感；
  它不在门禁里，不要把它当性能基准。

## 怎么证明「读到的是别人的内存」——下毒再断言

分配器 bug 的症状是「值不对」，而「值不对」有一百种原因。要把它钉死成
**读到了不该读的那块内存**，唯一可靠的办法是**先在那块内存里写一个不可能出现的值，
再断言读回来的不是它**。

```python
MAGIC  = 98765.0     # 被测数据
POISON = 3.0         # 覆盖者
```

三条规则，缺一条就得到假绿：

1. **下毒的缓冲区必须和被测缓冲区同形同 dtype。** SFRL 按 size 分桶，形状不同就
   拿不到同一个块。pyops 分区的做法是一次造 128 个同形缓冲全填 `98765.0` 再全释放，
   把 free list 铺满，下一次分配几乎必然命中其中一个。
2. **第 0 轮永远是干净的。** free list 一开始是空的，第一轮没有可复用的块，
   复现要从第 1 轮起。测试至少跑 3 轮，只跑 1 轮 = 没测。
3. **下毒的写必须走 device kernel，不要走 `jt.array(np.…)`。** `jt.array` 是主机侧
   分配加一次同步 H2D 拷贝，中间有几十毫秒的 Python/numpy 时间，异步窗口早就关上了。
   正确写法是从一个**已经在设备上**的 var 派生：

   ```python
   seed = jt.array(np.zeros(n, "float32")) + 0.0   # 这个 + 0.0 把它挪到设备上
   seed.sync()
   poison = [seed + POISON for _ in range(8)]      # 纯 device kernel，没有主机拷贝
   ```

## 别名（`share_with`）怎么观察与怎么证伪

`Var::share_with(x, offset)` 只是把 `allocator` 字段临时塞成父 Var 的指针；真正建立
别名的是 `Var::alloc`，它成功后把子 var 的 `allocator`/`allocation` 覆盖成父块的，
**从此没有任何字段记得「我是某块的子区间」**。建立别名的算子：`reshape`、`clone`、
`tape`、`getitem`/`setitem` 的 inplace 路径、`code_op`、`fused_adamw`。

从 Python 观察别名只有一条路：`operator<<(ostream&, const Var&)` 会把 `mem_ptr`
以十六进制打出来，而 `jt.dump_all_graphs().nodes_info` 是唯一能读到它的接口。

```python
def var_mem_ptrs():
    out = {}
    for info in jt.dump_all_graphs().nodes_info:
        if not info.startswith("Var("):
            continue
        body = info[len("Var("):]
        # Var(id:f:b:p:iN:oN:sN:nN:gN,dtype,name,memptr)shape
        out[body.split(":", 1)[0]] = body.split(",")[3].split(")")[0]
    return out
```

两个 var 的 mem_ptr 相等就是别名（offset=0 的情形，`reshape` 就是）。

**最短的一条别名断裂复现**（`use_cuda=1`）：

```python
a = jt.array(np.zeros(6, "float32")) + 0.0
b = a.reshape((2, 3))        # b 与 a 共享同一块
jt.sync_all(True)
b.numpy()                    # fetch_sync -> migrate_to_cpu，把 b 单方面搬走
a[1] = 7.0                   # setitem inplace，写进 a 那块（还在显存里）
assert b.numpy()[0][1] == 7.0
```

`.numpy()` / `.item()` / `jt.fetch_sync` 在非 managed 配置下都是**真搬家**
（`migrate_to_cpu` 新分配加 memcpy 加 free 原块），不是拷一份出来看。这是别名断裂
最容易踩到的入口，比「混合 CPU-CUDA 图」好构造得多。

## 跨流的时序 bug 怎么稳定复现

`fetch_op` 在自己的非阻塞流上拷贝，源 var 的块在 `run_sync` 末尾
（`executor.cc` 的 `fetcher_to_free.clear()`，**在 `cudaDeviceSynchronize` 之前**）
就回了 free list。要赢这个竞争，得让副流上堆着足够多的活，而主流这边尽快把块要回来：

1. **一次 fetch 多个 var**：`jt.fetch(v1, …, v8, cb)` 会在同一条流上排 8 组
   D2D+D2H。32 MB 的 D2H 走 PCIe 约 2.7 ms，8 组就是 ~20 ms 的积压——后面几个 var
   的 D2D 是在 20 ms 之后才执行的，源块早被抢走了。
2. **中间一次设备同步都不能有**：`jt.sync_all(True)` 会
   `cudaDeviceSynchronize()`，把所有流（包括副流）等干净，窗口就关了。释放那一步用
   `jt.sync_all()`（不带 True）。
3. 顺序是：`fetch` → `del srcs` → `jt.sync_all()`（这一步才真正释放块）→
   造 poison → `jt.sync_all()` → 最后 `jt.sync_all(True)` 收货。

这样构造，8 个 var 里有 7 个 100% 读到 poison。同样的场景改成「一次只 fetch 一个
64 MB 的 var」则**一次都复现不了**——D2D 只要 60 µs，主机侧那点 Python 时间足够它跑完。
所以「试了几次没复现」不等于没 bug，**要先把副流的积压做够**。

## `check_graph` 与 `NODE_MEMCHECK`

`jt.graph_check()` 分两半：liveness 三个计数器的一致性校验（一直在跑），和
「活着但从 hold_vars 到不了」的悬挂节点扫描（走 `lived_nodes`）。后者的登记表原来
只在 `#ifdef NODE_MEMCHECK` 下填，也就是**只有 `debug=1` 的构建**里才有内容；
正式构建里它扫一张空表然后报告成功。

现在登记跟着 `check_graph` 走：`jt.flag_scope(check_graph=1)` 里新建的节点才进表。
判据是 `jt.graph_check()` 的返回值（扫过的节点数）：

```python
with jt.flag_scope(check_graph=1):
    x = jt.array(...); (x*2).sum().sync()
    assert jt.graph_check() > 0        # 0 表示这一半根本没在检查
```

**开着 `check_graph=1` 的进程里每次 `sync()` 都会跑一遍全图校验**，慢，只在查
liveness 问题时开。

## 换页（swap / save_mem）怎么测

`save_mem` 是**编译期**常量，不是运行期 flag（`jt.flags.save_mem` 不存在，这是
故意的：swap.h 顶上那张 TODO 还没做完，而 `if (save_mem)` 挂在每一次 Var 释放上）。
`export JT_SAVE_MEM=1` 现在会被翻译成 `-DJT_SAVE_MEM=1`，并且它**自带一个缓存目录**
（`jittor_utils.save_mem_build_flags` 进了构建配置指纹），所以开关它只会各编一次，
不会互相顶掉对方的产物：

```bash
export JT_SAVE_MEM=1                   # 会打一条 experimental 的警告，是对的
```

然后在 Python 里把限额压到比工作集小：

```python
jt.flags.use_parallel_op_compiler = 0   # 要 fork 的话必须关，否则子进程卡在幽灵线程池
jt.flags.use_cuda_host_allocator = 0    # 要 fork 的话必须关，fork 后子进程用不了 CUDA
jt.flags.cpu_mem_limit = 4 << 20
jt.flags.device_mem_limit = 4 << 20
keep = [jt.array(...) * 2.0 for _ in range(12)]   # 必须**留着**，否则没有可换出的 var
```

判据：`<cache_path>/tmp/` 下出现 `swap-<pid>-<token>-<var id>.bin`，且
`display_memory_info()` 的 `swap: total(...)` 不为 0。只看「没报错」不算数——
限额设得不对时它只会打一条 `unable to alloc var` 的 warning 然后照常分配。

fork 场景的坑（都不是 swap 自己的问题，别去修）：
- 子进程继承的编译线程池是幽灵，`wait_all` 每个线程等 5 秒；先在父进程把所有
  kernel 编译完，并且关掉 `use_parallel_op_compiler`。
- CUDA context 不能跨 fork，子进程里任何 CUDA 调用都会
  `cudaErrorInitializationError`；要测 fork 就整个用例走 CPU。
