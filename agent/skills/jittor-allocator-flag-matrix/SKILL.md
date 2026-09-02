---
name: jittor-allocator-flag-matrix
description: 复现和验证 Jittor 分配器相关缺陷的 flag 矩阵与判据。改动 src/mem/**、Var::alloc/share_with、getitem/setitem 的 inplace 路径，或看到"只在某些配置下结果不对/段错误"时用它——默认配置会掩盖几乎所有分配器 bug，测试不切 flag 就等于没测。
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

## 换页（swap / save_mem）怎么测

`save_mem` 是**编译期**常量，不是运行期 flag。文档里的 `export JT_SAVE_MEM=1`
当前**无效**（仓库里没有把它变成 `-DJT_SAVE_MEM` 的地方）。要跑到 `mem/swap.cc`：

```bash
export cc_flags=" -DJT_SAVE_MEM=1 "
export JITTOR_HOME=<另一个缓存目录>     # 别污染你平时那个，切回去要重编
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
