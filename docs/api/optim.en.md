# Optimizer API

[Section contents](index.en.md) · [中文](optim.md)

Optimizers live in `jittor.optim`, and learning-rate schedules live in
`jittor.lr_scheduler`.

Native training commonly passes the loss to `step(loss)`. Torch-compatible training has a different calling convention; see the [Torch guide](../compatibility/torch.en.md).

:::{autopublicmodule} jittor.optim
:::

## Learning-rate schedules

:::{autopublicmodule} jittor.lr_scheduler
:::

## Related pages

- [Neural Network API](nn.en.md)
- [Core API](core.en.md)
- [Quickstart](../getting-started/quickstart.en.md)
