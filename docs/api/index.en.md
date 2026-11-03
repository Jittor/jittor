# API reference

[Documentation](../index.en.md) · [中文](index.md)

Find native Jittor modules and functions by task. Parameters and public objects come from the package installed when the documentation is built; availability also depends on your version and device. For existing Torch code, read the [Torch compatibility guide](../compatibility/torch.en.md).

- Tensors and gradients: [core](core.en.md).
- Models and training: [layers](nn.en.md), [attention](nn-attention.en.md), [models](models.en.md), [initialization](init.en.md), [optimizers and schedulers](optim.en.md).
- Data preparation: [datasets](dataset.en.md), [transforms](transform.en.md).
- Specialized computation: [linear algebra](linalg.en.md), [distributions](distributions.en.md), [3D losses](loss3d.en.md), [contributed utilities](contrib.en.md).

If you have not written a training script yet, start with the [quickstart](../getting-started/quickstart.en.md).

:::{autosummary}
jittor
jittor.nn
jittor.nn.attention
jittor.models
jittor.optim
jittor.lr_scheduler
jittor.init
jittor.contrib
jittor.dataset
jittor.transform
jittor.linalg
jittor.distributions
jittor.loss3d
:::

:::{toctree}
:hidden:
:maxdepth: 2

core.en
nn.en
nn-attention.en
models.en
optim.en
init.en
contrib.en
dataset.en
transform.en
linalg.en
distributions.en
loss3d.en
:::
