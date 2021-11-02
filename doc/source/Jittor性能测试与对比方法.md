Jittor性能测试与对比方法
=====================

下面代码以AlexNet为例，用于演示 Jittor 性能测试的正确方法：

```python
import time
import jittor as jt
from jittor.models import resnet50
jt.flags.use_cuda = jt.has_cuda

warmup = 10
rerun = 100
batch_size = 8
data = jt.random((batch_size, 3, 224, 224))
model = resnet50()
model.eval()

# 此段代码对jittor进行热身，确保时间测试准确
jt.sync_all(True)
for i in range(warmup):
    pred = model(data)
    # sync是把计算图发送到计算设备上
    pred.sync()
# sync_all(true)是把计算图发射到计算设备上，并且同步。
# 只有运行了jt.sync_all(True)才会真正地运行，时间才是有效的，因此执行forward前后都要执行这句话
jt.sync_all(True)

# 开始测试运行时间
start = time.time()
for i in range(rerun):
    pred = model(data)
    pred.sync()
jt.sync_all(True)
end = time.time()

print("Jittor FPS:", (rerun*batch_size)/(end-start))

```

在这段代码中，我们定义了几个参数`batch_size`, `warmup`, `rerun`, batch_size代表批大小，warmup是用于热身的循环次数，而rerun是用于测速的循环次数，最终输出FPS，对Jittor进行正确测速的关键是 热身部分和同步部分，热身部分确保测试时间稳定，没有包含编译用的时间，而同步部分确保计算完成，因为jittor是一个异步框架，只有同步操作能保证计算完成。

以上代码的运行结果如下（RTX Titan，batch 8）：

```
Compiling Operators(8/8) used: 7.35s eta:    0s
Compiling Operators(13/13) used: 8.36s eta:    0s
Jittor FPS: 908.9853866375396
```

我们还可以使用类似的代码测试 PyTorch的性能：

```python
import time
import torch
from torchvision.models import resnet50

warmup = 10
rerun = 100
batch_size = 8
data = torch.randn((batch_size, 3, 224, 224)).cuda()
model = resnet50()
model.cuda()
model.eval()

# 此段代码对pytorch进行热身，确保时间测试准确
torch.cuda.synchronize()
for i in range(warmup):
    pred = model(data)
# synchronize用于确保PyTorch计算完成
torch.cuda.synchronize()

# 开始测试运行时间
start = time.time()
for i in range(rerun):
    pred = model(data)
torch.cuda.synchronize()
end = time.time()

print("PyTorch FPS:", (rerun*batch_size)/(end-start))
```


以上代码的运行结果如下（RTX Titan，batch 8）：

```
PyTorch FPS: 807.4806873965665
```

我们还可以对这两段代码合并，并对比结果的一致性：

```python
import time
import jittor as jt
from jittor.models import resnet50
jt.flags.use_cuda = jt.has_cuda

warmup = 100
rerun = 1000
batch_size = 8
data = jt.random((batch_size, 3, 224, 224))
model = resnet50()
model.eval()

# 此段代码对jittor进行热身，确保时间测试准确
jt.sync_all(True)
for i in range(warmup):
    pred = model(data)
    # sync是把计算图发送到计算设备上
    pred.sync()
# sync_all(true)是把计算图发射到计算设备上，并且同步。
# 只有运行了jt.sync_all(True)才会真正地运行，时间才是有效的，因此执行forward前后都要执行这句话
jt.sync_all(True)

# 开始测试运行时间
start = time.time()
for i in range(rerun):
    pred = model(data)
    pred.sync()
jt.sync_all(True)
end = time.time()

print("Jittor FPS:", (rerun*batch_size)/(end-start))
# 将 jittor 数据和参数导出为 numpy 和 torch 格式
jittor_data = pred.numpy()
jittor_param = model.state_dict(to="torch")

import numpy as np
import torch
from torchvision.models import resnet50
data = torch.Tensor(data.numpy()).cuda()
model = resnet50()
# 加载 jittor 参数
model.load_state_dict(jittor_param)
model.cuda()
model.eval()

# 此段代码对pytorch进行热身，确保时间测试准确
torch.cuda.synchronize()
for i in range(warmup):
    pred = model(data)
# synchronize用于确保PyTorch计算完成
torch.cuda.synchronize()

# 开始测试运行时间
start = time.time()
for i in range(rerun):
    pred = model(data)
torch.cuda.synchronize()
end = time.time()

print("PyTorch FPS:", (rerun*batch_size)/(end-start))
pytorch_data = pred.detach().cpu().numpy()
err = np.mean(np.abs(pytorch_data - jittor_data))
print("mean error:", err)

```


以上代码运行结果如下：

```
Jittor FPS: 908.9853866375396
PyTorch FPS: 807.4806873965665
mean error: 1e-5
```

误差输出为1e-5, 在可接受范围内。正确测速与对比的几大关键点为：

1. 充分热身，除去框架的准备时间。
2. 多次运行，确保测试时间稳定。
3. 加上同步语句，确保测试时间准确。
4. 保证显存充足，在显存不足时，jittor会调用统一内存来弥补，会产生性能损失，请密切关注`nvidia-smi`的输出结果。
5. 保证对比模型的一致性，检查输出结果的一致。

如果您对测试结果有疑问，或者有优化需求，欢迎随时联系Jittor开发团队。
