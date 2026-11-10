# Install and deploy the Torch entry point

[Section contents](index.en.md) · [中文](torch-shim.md)

This page explains how an application finds the intended `torch` package and how compatibility activation works. First-time users should start with the [CPU training example](torch.en.md).

## Which packages to install

Core `jittor` provides the native computation runtime. `jittor-compat` provides the Torch frontend, deployment tools and shared compatibility mechanisms. Independent integration repositories supply project-specific patches.

For a source checkout, run from the repository root:

```bash
python -m pip install -e . -e ./compat
```

For wheels, install matching core and compatibility wheels with the same interpreter, respecting their metadata requirements. Genuine PyTorch and the compatibility package both provide `torch`; keep them in separate environments.

## Choose an activation method

For a script or notebook, explicitly activate before other Torch imports:

```python
from jittor.compat.shim import activate
activate()
import torch
```

`activate()` is process-wide and idempotent. Plain `import jittor` does not claim the `torch` namespace. A process that has imported genuine PyTorch cannot safely switch runtimes by activating afterward.

For an application that starts with `import torch`, deploy the entry point:

```bash
jittor-torch-shim
jittor-torch-shim --check
```

This command comes from `jittor-compat` and also deploys bundled third-party stubs and metadata. Do not copy only `torch/__init__.py`. After changing or upgrading compatibility resources, use `--check` to verify deployment.

`JITTOR_TORCH_SHIM=1` selects Torch mode but does not add uninstalled packages to Python's search path. Install the compatibility package first; setting the variable alone cannot fix `ModuleNotFoundError`.

## Optional torch_xla emulation

`jittor-compat` bundles a `torch_xla` client emulation so programs written against the PyTorch/XLA client API can run on Jittor. `jittor-torch-shim` deploys it with the other bundled stubs. It is not XLA: `distributed.xla_multiprocessing.spawn` gives each process one visible CUDA device, `core.xla_model` maps device/ordinal/rendezvous calls onto Jittor CUDA or CPU devices, and collectives exchange files through the rendezvous directory. There is no TPU, no XLA compilation, no SPMD sharding and no XLA FSDP.

Note that once `torch_xla` is importable, Transformers and Accelerate treat the environment as an XLA one by default (`USE_TORCH_XLA` defaults to true). Set `USE_TORCH_XLA=0`, or leave this stub undeployed, when running native Jittor CUDA work; otherwise unrelated Trainer paths enter the emulated XLA branches.

Launching matches upstream; `--num_cores` must not exceed the CUDA devices visible to the process, and a CPU-only host supports one process:

```bash
python examples/pytorch/xla_spawn.py --num_cores 2 train_script.py ...
```

## Caches and project extensions

Set `JITTOR_TORCH_RUNTIME_ROOT` before starting the application when you need a fixed runtime location. The default is `${XDG_CACHE_HOME:-~/.cache}/jittor/torch-shim/`. Native Jittor's compilation cache and the compatibility runtime are separate concepts; check their locations when maintaining multiple environments.

Compatibility bootstrap can discover source extensions and process them through Jittor-backed build interfaces. This does not imply that arbitrary PyTorch binary extensions can be loaded. Keep compiler output when a build fails and check the integration repository's extension requirements.

## CPU NumPy views

Integrations that pass a CPU tensor to a NumPy `out=` argument can use the public `jittor.compat.torch.numpy_interop.numpy_view(tensor)` API. It returns a writable view of the current CPU allocation without copying, and the array retains ownership of that allocation. Device tensors and tensors requiring gradients raise an error; the API does not transfer or detach them. After tensor storage is replaced, an old view still refers to the old allocation, so callers must obtain a new view. The API does not promise that a view will track later tensor writes or storage replacements.

```python
from jittor.compat.torch.numpy_interop import numpy_view

out = numpy_view(cpu_tensor)
numpy_operation(..., out=out)
```

## Downstream adapters

Install the integration you need using its [JittorEcosystem](https://github.com/JittorEcosystem) repository's instructions. Each repository maintains version constraints and device support. Installing an adapter does not establish support for every library feature.

Adapters register patches or extension discovery policies through `jittor.module_patches` and `jittor.external_backends`. See the [compatibility principles](../development/compatibility/principles.en.md) for ownership rules and the [architecture](../development/compatibility/torch-architecture.en.md) for implementation organization.

## Troubleshoot imports

| Symptom | Check first |
| --- | --- |
| `jittor.compat` cannot be found | Is `jittor-compat` installed for this interpreter, rather than only core `PYTHONPATH` being set? |
| `import torch` selects another runtime | Check the interpreter, environment and installation location; check whether genuine PyTorch was already imported. |
| Submodules are missing after deployment | Run `jittor-torch-shim --check` and inspect missing or inconsistent files. |
| A downstream version is rejected | Use a version supported by the integration repository instead of bypassing its check. |

`torch.__version__` may reflect the Jittor version; it does not establish that a particular native PyTorch release is installed. The deployment helper prints a command to inspect the entry point and Torch API version.
