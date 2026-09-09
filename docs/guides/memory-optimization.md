# 显存优化

先用内存分析器找出占用大的活跃张量，再考虑开启换出。**换出能降低显存下限，但替代
不了修好一张被意外留住的图。**

## 分析大块分配

内存分析器把峰值分配归因到 Python 调用点：

```python
import jittor as jt
from jittor import models

net = models.resnet18()
with jt.flag_scope(trace_py_var=3, profile_memory_enable=1):
    images = jt.randn((1, 3, 224, 224))
    net(images).sync()
    jt.get_max_memory_treemap()
```

看报告里最大的几个分支。在动全局显存上限之前，先做这三件事：缩小临时张量、释放
那些让图活着的 Python 引用、把工作拆成更小的批次。

## 自动换出（实验性）

达到上限时，Jittor 可以在显存、主机内存和磁盘之间搬运数据。上限单位是字节，`-1`
表示不限。

`JT_SAVE_MEM` 是**构建开关，不是运行时 flag**：它会变成编译期常量，好让没启用它的
构建在每次释放值时不必付出换出分支的代价。因此设置它等于选择了另一套构建配置，
设置或清除之后的第一次运行会重新编译。

**换出功能尚未完成**：别名值（`share_with`）、设备间迁移、双分配器和外部分配器都还
在它自己的 TODO 里。只有在你**实测确认需要**时才开。

```bash
export JT_SAVE_MEM=1
export cpu_mem_limit=16000000000
export device_mem_limit=8000000000
python train.py
```

PowerShell 用对应的 `$env:JT_SAVE_MEM`、`$env:cpu_mem_limit`、`$env:device_mem_limit`。
把换出文件放在空间足够的本地磁盘上，并预期换出期间吞吐下降。

清理残留的换出文件：

```bash
python -m jittor_utils.clean_cache swap
```

## 相关

- 显存到底去哪了、缓存分配器为什么不还给驱动：见[调试指南](debugging.md)
- 把大张量搬回主机：`.cpu()` 的目标缓冲分配在主机侧，见[设备与放置](../notes/device-placement.md)
