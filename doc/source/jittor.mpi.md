jittor.mpi
=====================

这里是Jittor的MPI模块的API文档，您可以通过`from jittor import mpi`来获取该模块。

## 如何从单卡代码适配多卡代码

使用`mpirun`时，以下几种模块会自动检测mpi环境并且自动切换成多卡版本：

* jittor.optimizer: 自动同步梯度
* jittor.nn.BatchNorm*： 同步batch norm
* jittor.dataset： 自动数据并行

大部分情况下，单卡训练的代码可以直接使用`mpirun`实现分布式多卡运行。 但仍然如下几种情况下，需要对代码进行调整：

1. 对硬盘进行写操作（保存模型，保存曲线）
2. 需要统计全局信息（validation 上的全局准确率）

### 对硬盘进行写操作

对于第一点，假设原来您的代码如下：

```python
for i, (images, labels) in enumerate(dataset):
    output = model(images)
    loss = nn.cross_entropy_loss(output, labels)
    acc1 = accuracy(output, labels)
    SGD.step(loss)
    loss_data = loss.data
    writer.add_scalar("Train/loss")
```

更改后的代码如下：

```python
for i, (images, labels) in enumerate(dataset):
    output = model(images)
    loss = nn.cross_entropy_loss(output, labels)
    acc1 = accuracy(output, labels)
    SGD.step(loss)
    loss_data = loss.data
    if jt.rank == 0:
        writer.add_scalar("Train/loss")
```

这里我们使用了 jt.rank 来限制，只允许第一个进程可以写 loss，这个代码在单卡下也是有效的，因为单卡的 jt.rank 值为 0， 需要注意的是，在 `if jt.rank == 0` 代码块里面的代码，不允许调用任何jittor的api，因为这很有可能导致多卡之间的api调用不一致而产生**死锁**!

### 需要统计全局信息

统计全局信息有两种方法，第一种是使用提供的 mpi op 来实现全局信息统计, 如下所示， 是一个validation的代码：

```python
def val(epoch):
    global min_error
    model.eval()
    correct_nums = 0
    for i, (images, labels) in enumerate(valdataset):
        output = model(images)
        correct_nums += top1error(output, labels)
        correct_nums.sync()
    top1_error = (valdataset.total_len - correct_nums.data[0]) / valdataset.total_len
    if top1_error < min_error:
        print("[*] Best model is updated ...")
        model.save('model_best.pkl')
```

更改方案如下：

```python
def val(epoch):
    global min_error
    model.eval()
    correct_nums = 0
    for i, (images, labels) in enumerate(valdataset):
        output = model(images)
        correct_nums += top1error(output, labels)
        correct_nums.sync()
    if jt.in_mpi:
        correct_nums = correct_nums.mpi_all_reduce()
    top1_error = (valdataset.total_len - correct_nums.data[0]) / valdataset.total_len
    if jt.rank == 0 and top1_error < min_error:
        print("[*] Best model is updated ...")
        model.save('model_best.pkl')
```

可以留意到我们首先使用了 `mpi_all_reduce`， 来统计多卡的正确数量(mpi_all_reduce会将多个mpi进程的结果累加起来)， 然后在 `jt.rank == 0` 的情况下才更新模型。

第二种方法是使用`@jt.single_process_scope()`，被装饰的代码会直接以单进程的方式执行，无需处理多卡。

```python
@jt.single_process_scope()
def val(epoch):
    ......
```

下面是 jittor 的 mpi api reference.

```eval_rst
.. automodule:: jittor_mpi_core
   :members:
   :undoc-members:
.. automodule:: jittor_mpi_core.ops
   :members:
   :undoc-members:
```
