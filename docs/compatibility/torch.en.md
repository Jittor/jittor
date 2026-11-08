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

## State and resuming

CUDA and ACL retain random state in their respective native generators. In an ACL build, the compatibility `torch.cuda` RNG spelling and `torch.npu` delegate to the same ACL device stream; actual CUDA uses cuRAND seed/offset state. The returned CPU `uint8` tensors are opaque snapshots to save and return to the same backend. They cannot be exchanged between CUDA, ACL or native PyTorch. CPU state and individual device streams are saved and restored separately.

SGD and AdamW retain `foreach`/`fused` constructor settings, group overrides, added groups and checkpoint metadata, then select available native optimizer update implementations. Setting both options to `True` raises an error. Unsupported differentiable optimizer and AdamW capture options remain explicit refusals. Native `jt.optim` keeps its own default arithmetic policy.

Torch DCP retains per-rank snapshots (`metadata.json`) for ordinary state trees and uses native sharded manifests (`manifest.json`) for local DTensor intervals. Loading detects existing formats and preserves destination tensor device, dtype and storage identity. Native PyTorch `.metadata/.distcp` files and custom planners are outside these formats. Device, world-size and sharding support still follow the relevant API boundaries; see [native sharded checkpoints](../guides/distributed/native-sharded-checkpoint.en.md).

## Use a downstream library

Each [JittorEcosystem](https://github.com/JittorEcosystem) repository maintains its installation combination, official example entry points, tests and feature/device support table. Check the features needed by your task before following the corresponding version's official tutorial.

For a problem report, keep a minimal script, package versions, device details and the complete error log. Import failures, compilation errors, numerical differences and performance problems need different evidence. See the [compatibility principles](../development/compatibility/principles.en.md) for acceptance criteria.

Continue with the [Torch API tutorial](https://github.com/Jittor/jittor/blob/2.0-refactor/examples/notebooks/torch_compat.md) and [script migration tutorial](https://github.com/Jittor/jittor/blob/2.0-refactor/examples/notebooks/torch_compat_migration.md).


Torch NPU Linear reuses low-precision casts of trainable FP32 leaf tensors within an autocast region. Branch gradients therefore accumulate at the shared compute dtype before conversion back to FP32. Cache keys include dtype, device and the native storage version; leaving the outermost autocast region or calling `torch.clear_autocast_cache()` releases the holders. `cache_enabled=False` uses independent casts. This mechanism belongs to the Torch NPU Linear boundary; native Jittor operators own their native precision policy.
