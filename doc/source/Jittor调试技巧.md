Jittor调试技巧
=====================

该文档包含了几种异常情况的调试方法和技巧。

## 爆Nan、Inf

在模型训练的过程中，可能因为数值不稳定而出现Nan或者Inf，为了帮助您定位出现nan的代码，您可以设置如下环境变量：

```bash
export JT_CHECK_NAN=1
export trace_py_var=3
```

其中，环境变量`JT_CHECK_NAN=1`的用途是：当算子的输出出现异常浮点数时，自动报错并停止程序，环境变量`trace_py_var=3`的用途是：输出算子对应的Python代码行数，3代表输出的详细等级，为最高等级。

需要注意的是，开启这两个特性之后，jittor速度会大幅下降，并且触发重编译，请不要在训练环境或者生产环境开启该模式，也不要长时间开启该模式。

## 错误信息定位不准确

Jittor框架默认采用延迟执行（Lazy execution）的方式进行加速，算子的执行和创建是不同步的，这可能导致报错信息定位不准确，您可以手动关闭延迟执行，采取立刻执行（eager execution）的模式，使用如下环境变量即可：

```bash
export lazy_execution=0
```

或者在python代码中通过flag关闭
```python
jt.flags.lazy_execution=0
```

## 内存不足

当您发现Jittor由于内存相关问题，无法运行时，Jittor会向您报告内存使用情况，内存不足可能有两种情况：

1. 训练模型过大，一个迭代就崩溃报错。
2. 多次迭代的过程中，内存占用不断增长，直到最后内存耗尽报错。

**对于第一种情况** ，您可能需要调整模型或者数据大小，或者使用[多卡训练](jittor.mpi)，此外，您还可以在每个迭代内部，让Jittor强制回收内存：

```python
for ...:
    ...
    jt.sync_all()
    jt.gc()
```

如果您使用到了CUDA和卷积，还有可能是卷积消耗的临时空间过大，在这种情况下，可以关闭cudnn的临时内存申请，请将如下代码插入到最开始：

```python
jt.cudnn.set_max_workspace_ratio(0.0)
```

**对于第二种情况**，可能是存在内存内存泄漏，请检查您是否存在全局变量没有释放，或者全局变量没有停止梯度，导致计算图不断增加，检查方法如下，您可以在每个迭代内部，插入如下调试代码：

```python
for ...:
    ...
    jt.sync_all()
    jt.display_memory_info()
```

Jittor会输出内存消耗，以及计算图的大小`lived_var,lived_op`，以及用户持有的变量数`hold_var`, 如果计算图规模不断增大，请检查代码，或者提交github issue联系我们，并且附上错误日志和代码复现脚本。


## 段错误

如果Jittor出现了段错误，建议您将错误提交github issue联系我们，并且附上错误日志和代码复现脚本。您也可以使用如下环境变量对程序以及框架进行诊断：

```bash
export debug=1
export gdb_attach=1
```

其中，环境变量`debug=1`代表开启jittor的debug模式，性能会大幅下降，但会保留调试信息，`gdb_attach=1`将会自动将gdb贴在jittor的主进程上，方便您进行单步调试。关于gdb的使用，您可以参考[GDB Cheat Sheet](https://darkdust.net/files/GDB%20Cheat%20Sheet.pdf)


## 管理Jittor cache

Jittor会在`～/.cache/jittor`目录下创建cache， cache里面可能包括 core（内核）、cuda编译器、cuda库、数据集（dataset）、预训练参数等等，在某些情况下cache可能失效，如系统更新、驱动更新等等，这种情况可能需要用户手动清除cache， 清除的方法如下：

```
python3 -m jittor_utils.clean_cache all
```

以上命令会清除jittor的所有cache，如果您不想全部清除，可以参考命令行帮助：

```
python3 -m jittor_utils.clean_cache help
```