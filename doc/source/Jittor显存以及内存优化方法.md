Jittor显存以及内存优化方法
=====================

您可以主要通过两种方法，来改进内存消耗：

1. 优化消耗内存比较大的变量
2. 使用Jittor自动交换技术，将变量在显存-内存-硬盘之间交换，降低运行部署门槛。

## 优化消耗内存比较大的变量

您可以使用jittor的memory profiler，来分析显存消耗较大的代码，并且针对特定代码进行优化。使用方法如下：

```
net = jt.models.resnet18()
with jt.flag_scope(trace_py_var=3, profile_memory_enable=1):
    imgs = jt.randn((1,3,224,224))
    net(imgs).sync()
    jt.get_max_memory_treemap()
```

输出如下：
```
    | 
    ├─./python/jittor/test/test_memory_profiler.py:100(test_sample)
    | [19.03 MB; 29.67%]
    | ./python/jittor/test/test_memory_profiler.py:100
    |    | 
    |    └─./python/jittor/__init__.py:730(__call__)
    |      [19.03 MB; 29.67%]
    |      ./python/jittor/__init__.py:730
    |         | 
    |         └─./python/jittor/models/resnet.py:152(execute)
    |           [19.03 MB; 29.67%]
    |           ./python/jittor/models/resnet.py:152
    |              | 
    |              ├─./python/jittor/models/resnet.py:142(_forward_impl)
    |              | [6.13 MB; 9.55%]
    |              | ./python/jittor/models/resnet.py:142
    |              |    | 
```


## 使用自动交换技术

该技术确保Jittor在显存或者内存不足的情况下，都能以一定速度运行。

节省内存方法，请安装Jittor版本大于1.3.7.5，并添加如下环境变量：

```bash
export JT_SAVE_MEM=1
# 限制cpu最多使用16G
export cpu_mem_limit=16000000000
# 限制device内存（如gpu、tpu等）最多使用8G
export device_mem_limit=8000000000
# windows 用户，请使用powershell
# $env:JT_SAVE_MEM="1"
# $env:cpu_mem_limit="16000000000"
# $env:device_mem_limit="8000000000"
```
用户可以自由设定cpu和设备内存的使用量，如果不希望对内存进行限制，可以设置为`-1`。
```bash
# 限制cpu最多使用16G
export cpu_mem_limit=-1
# 限制device内存（如gpu、tpu等）最多使用8G
export device_mem_limit=-1
# windows 用户，请使用powershell
# $env:JT_SAVE_MEM="1"
# $env:cpu_mem_limit="-1"
# $env:device_mem_limit="-1"
```

如果想要清理磁盘交换文件，可以运行如下命令
```bash
python3 -m jittor_utils.clean_cache swap
```