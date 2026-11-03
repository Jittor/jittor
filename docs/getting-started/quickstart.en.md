# Quickstart: tensors and training

[Section contents](index.en.md) · [中文](quickstart.md)

After [installation](installation.en.md), save the examples below as Python scripts and run them. They use a CPU and small in-memory data, with no model downloads.

## Compute a value and its gradient

```python
import jittor as jt

jt.flags.use_cuda = 0
x = jt.array([2.0, 3.0])
y = (x * x).sum()
print(y.item())                 # 13.0
print(jt.grad(y, x).numpy())    # [4. 6.]
```

`Var` is Jittor's native tensor. Reading `.item()` or `.numpy()` waits for computation to finish; building an expression and obtaining its value are separate steps.

## Train a small network

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

Native Jittor uses `optimizer.step(loss)` to compute gradients and update parameters. Initialization affects the exact loss values. This example teaches the training loop; it is not an accuracy or performance acceptance test.

## Continue learning

- [Complete introductory tutorial](https://github.com/Jittor/jittor/blob/2.0-refactor/examples/notebooks/getting_started.md): tensors, model definitions and real data.
- [Tutorial directory](https://github.com/Jittor/jittor/blob/2.0-refactor/examples/notebooks/README.md): devices, data, training and advanced topics.
- [API reference](../api/index.en.md): layers, optimizers and data processing.
- [Torch compatibility guide](../compatibility/torch.en.md): existing scripts using `loss.backward()` and `optimizer.step()`.
