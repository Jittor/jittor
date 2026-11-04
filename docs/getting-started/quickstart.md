# 快速开始：张量与训练

[本栏目目录](index.md) · [English](quickstart.en.md)

完成[安装](installation.md)后，把下面的代码保存为脚本运行。示例只用 CPU 和内存中的小数据，不需要下载模型。

## 做一次计算并求导

```python
import jittor as jt

jt.flags.use_cuda = 0
x = jt.array([2.0, 3.0])
y = (x * x).sum()
print(y.item())                 # 13.0
print(jt.grad(y, x).numpy())    # [4. 6.]
```

`Var` 是原生 Jittor 的张量。读取 `.item()` 或 `.numpy()` 会等待计算完成；构建表达式与取得数值是两个时机。

## 训练一个小网络

```python
import jittor as jt
from jittor import nn

jt.flags.use_cuda = 0
model = nn.Sequential(nn.Linear(1, 8), nn.Relu(), nn.Linear(8, 1))
optimizer = nn.SGD(model.parameters(), lr=0.05)
x = jt.array([[-1.0], [0.0], [1.0], [2.0]])
y = x * 2 + 1

for step in range(100):
    loss = ((model(x) - y) ** 2).mean()
    optimizer.step(loss)
    if step % 20 == 0:
        print(step, loss.item())
```

原生训练通过 `optimizer.step(loss)` 求导并更新参数。初始化会影响具体 loss 数字，这个例子用于理解训练循环，不是精度或性能验收。

## 继续学习

- [完整入门教程](https://github.com/Jittor/jittor/blob/2.0-refactor/examples/notebooks/getting_started.md)：从张量到模型定义，再到真实数据。
- [教程目录](https://github.com/Jittor/jittor/blob/2.0-refactor/examples/notebooks/README.md)：设备、数据、训练与进阶主题。
- [API 参考](../api/index.md)：查找层、优化器和数据处理接口。
- [Torch 兼容使用指南](../compatibility/torch.md)：使用 `loss.backward()` 和 `optimizer.step()` 风格的已有脚本。
