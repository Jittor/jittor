# Torch shim

The torch shim lets a torch-oriented Python application use Jittor as its
runtime. The main Jittor distribution owns only the reusable torch API,
deployment, extension build, import-patch, and external-backend mechanisms.
Project-specific runtime policy is supplied by optional adapter distributions.

## Components

- `torch__init__.py` is deployed as `torch/__init__.py`.
- `stubs/` contains bundled compatibility packages such as `flash_attn`,
  `torchvision`, `torchaudio`, and `torchdata`.
- `cpp_extension/` provides the Jittor-backed `torch.utils.cpp_extension`
  surface used to build source extensions.
- `bootstrap.py` configures an isolated runtime, deploys the shim, discovers
  native extension roots, and loads registered optional adapters.
- `deploy.py` installs the complete shim tree into a target site-packages
  directory.

The canonical torch-compatible API lives in `jittor.compat.torch`. The shim
exports that API under the `torch` module name and wires the submodule paths
expected by third-party libraries.

## Bootstrap

Applications that need a project-local runtime can enable it explicitly:

```python
from jittor.torch_shim import enable

enable(project_root=__file__)
import torch
```

`enable()` creates a runtime below
`${XDG_CACHE_HOME:-~/.cache}/jittor/torch-shim/` unless
`JITTOR_TORCH_RUNTIME_ROOT` is set. It keeps Jittor, torch extension, CUDA,
Triton, pip, and temporary caches below that runtime and deploys the shim into
its local site-packages directory.

Local source extensions are discovered from `setup.py`, `pyproject.toml`, and
`CMakeLists.txt` signals. Missing or stale setuptools extensions are rebuilt
through the Jittor-backed cpp-extension API. Set
`JITTOR_TORCH_SKIP_EXT_BUILD=1` to skip warm-run build checks or pass explicit
`extension_dirs` when automatic discovery is not suitable.

For numerical parity, the bootstrap disables CUDA fast-math contraction for
Jittor JIT kernels unless `JITTOR_TORCH_KEEP_FAST_MATH=1` is set. Project
extensions retain the flags requested by their own build definitions.

## Optional adapters

Adapters use two public entry-point groups:

- `jittor.module_patches` registers exact module-path callbacks through
  `jittor.compat.module_patcher`.
- `jittor.external_backends` registers extension discovery policy through
  `jittor.compat.external_backend`.

The maintained adapters are separate distributions:

- `jittor-trellis` for TRELLIS.2 runtime policy and kernels.
- `jittor-gs` for graphdeco Gaussian Splatting runtime policy and launch tools.
- `jittor-hf-compat` for explicitly selected Transformers version adapters.

Installing an adapter makes its entry points discoverable. Applications may
also call the adapter's `install()` function explicitly. Jittor itself does not
import those projects, inspect their directory layouts, or install permanent
project-specific import finders.

## Deploy

Deploy the complete shim with the maintained helper:

```bash
python -m jittor.torch_shim.deploy --target /path/to/site-packages
```

The target contains the torch package, bundled stubs, and distribution metadata.
Do not copy only `torch__init__.py`; the nested stub modules and interfaces are
part of the runtime contract.
