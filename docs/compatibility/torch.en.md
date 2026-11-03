# Run existing Torch code

[Section contents](index.en.md) · [中文](torch.md)

Start with a CPU training loop, then decide how to migrate your script. Install matching Jittor and `jittor-compat` packages using the [installation guide](../getting-started/installation.en.md). Downstream support depends on the version, model and device; consult the integration repository's support scope.

## A complete CPU example

Run this in a new Python process. Activate the frontend before importing genuine PyTorch:

```python
import jittor as jt
jt.flags.use_cuda = 0

from jittor.compat.shim import activate
activate()
import torch

model = torch.nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
x = torch.tensor([[0.0], [1.0], [2.0]])
y = 2 * x + 1

for step in range(100):
    optimizer.zero_grad()
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    optimizer.step()
    if step % 20 == 0:
        print(step, loss.item())
```

`activate()` selects the compatibility frontend for the whole process; it is not a local context. Run independent PyTorch comparisons in another environment and process.

## Check these points when migrating

| Area | What to use |
| --- | --- |
| Imports | Plain `import jittor as jt` keeps native behavior. `import jittor as torch` is only a Python alias and does not activate compatibility. |
| Training loop | Torch mode uses `zero_grad()`, `backward()` and `step()`; native Jittor commonly uses `step(loss)`. |
| Installation | `jittor` and `jittor-compat` are separate distributions. Adding `PYTHONPATH=python` does not install compatibility. |
| Third-party libraries | Use versions required by the independent integration repository. A passing small model does not establish whole-library support. |
| Device and precision | Verify the target chip, dtype, forward pass, backward pass and training separately. CPU success cannot replace accelerator validation. |

The old `activate(independent_namespace=False)` and `JITTOR_TORCH_INDEPENDENT=0` modes have been removed. Use `activate()` and remove those switches. The Torch frontend shares Jittor's runtime but is a distinct Python module.

## Let an application import torch directly

The compatibility wheel provides the `torch` entry point. To deploy the bundled stubs, run:

```bash
jittor-torch-shim
jittor-torch-shim --check
```

This writes package files into the current environment; use an environment dedicated to the application. See [installation and deployment](torch-shim.en.md) for installation sources, import precedence and troubleshooting.

## Devices and mixed precision

Jittor's device execution model is not identical to PyTorch's. The compatibility frontend's `cuda` interface can also map to the ACL backend, so `torch.cuda.is_available()` alone does not identify the chip vendor. Consult the backend and integration documentation for NPU configuration and support.

Autocast uses Jittor precision controls rather than PyTorch's per-operator allowlist. Some operators and nested device contexts differ. Keep a float32 baseline while checking mixed-precision outputs, gradients and training curves. Memory usage alone does not prove where an operator executed.

## Use a downstream library

Each [JittorEcosystem](https://github.com/JittorEcosystem) repository maintains its installation combination, official example entry points, tests and feature/device support table. Check the features needed by your task before following the corresponding version's official tutorial.

For a problem report, keep a minimal script, package versions, device details and the complete error log. Import failures, compilation errors, numerical differences and performance problems need different evidence. See the [compatibility principles](../development/compatibility/principles.en.md) for acceptance criteria.

Continue with the [Torch API tutorial](https://github.com/Jittor/jittor/blob/2.0-refactor/examples/notebooks/torch_compat.md) and [script migration tutorial](https://github.com/Jittor/jittor/blob/2.0-refactor/examples/notebooks/torch_compat_migration.md).
