# 调试 Jittor 程序

Jittor 惰性执行、运行时编译算子。下面这些诊断手段都是**用性能换更准确的失败信息**，
只在复现问题时打开。

## 定位 NaN 和 inf

在启动进程前打开输出检查与 Python 源码追踪：

```bash
export JT_CHECK_NAN=1
export trace_py_var=3
python reproduce.py
```

`JT_CHECK_NAN=1` 会在算子产生非法浮点值时停下；`trace_py_var=3` 记录生成算子对应的
Python 调用位置。**两者都会触发重新编译并显著拖慢速度。**

## 让异步失败变成本地失败

惰性执行会在创建出错算子的那行 Python 之后才报错——这就是为什么堆栈常常指向
`print` 或 `.numpy()`。做最小复现时临时切成即时执行：

```bash
export lazy_execution=0
```

进程内的等价写法：

```python
import jittor as jt
jt.flags.lazy_execution = 0
```

## 诊断显存耗尽

单次迭代就超出显存时，先减小模型或批次。显式同步加回收有助于判断是不是临时图被
留住了：

```python
for batch in dataset:
    train_step(batch)
    jt.sync_all()
    jt.gc()
```

CUDA 卷积可以限制 cuDNN 的 workspace 来降低峰值分配，代价是速度：

```python
jt.cudnn.set_max_workspace_ratio(0.0)
```

如果显存**逐迭代增长**，检查图与持有变量的计数：

```python
for batch in dataset:
    train_step(batch)
    jt.sync_all()
    jt.display_memory_info()
```

`lived_var` 或 `lived_op` 持续上升，通常意味着某张图被留住了，或者某个全局变量还
参与在求导里。

> `nvidia-smi` 显示的显存在释放后不下降是**正常的**——那是缓存分配器没把块还给驱动。
> 用 `jt.clean()` 才会真正交还。

## 调试段错误

带调试信息构建并自动挂 GDB：

```bash
export JT_BUILD_DEBUG=1
export JT_GDB_ATTACH=1
python reproduce.py
```

Jittor 从环境读取的设置只有两个命名空间：**`JT_BUILD_*` 决定编译出什么，`JT_*`
决定编译好的核心怎么行为**。`python -m jittor_utils.env_manifest` 会列出全部。

多数设置也接受历史上的无前缀小写名（`gdb_attach=1` 仍然有效，并会在启动时提示一次）。
`debug` 是例外：一个普通英文单词更可能是别的工具的变量，所以它只以 `JT_BUILD_DEBUG`
的形式被读取。

报告原生崩溃时，请附最小复现、完整日志、编译器版本和设备信息。

## 管理编译缓存

Jittor 把编译好的 kernel、下载的工具链、数据集和权重都放在缓存目录里。升级编译器、
驱动或系统之后，清空全部缓存：

```bash
python -m jittor_utils.clean_cache all
```

查看更细的清理目标：

```bash
python -m jittor_utils.clean_cache help
```
