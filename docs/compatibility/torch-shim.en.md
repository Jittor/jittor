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

## Caches and project extensions

Set `JITTOR_TORCH_RUNTIME_ROOT` before starting the application when you need a fixed runtime location. The default is `${XDG_CACHE_HOME:-~/.cache}/jittor/torch-shim/`. Native Jittor's compilation cache and the compatibility runtime are separate concepts; check their locations when maintaining multiple environments.

Compatibility bootstrap can discover source extensions and process them through Jittor-backed build interfaces. This does not imply that arbitrary PyTorch binary extensions can be loaded. Keep compiler output when a build fails and check the integration repository's extension requirements.

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
