# Torch shim

The torch shim lets a torch-oriented Python application use Jittor as its
runtime. The independent `jittor-torch` distribution owns the reusable Torch API,
deployment, extension build, import-patch, and external-backend mechanisms.
The core `jittor` distribution contains none of these compatibility files.
Project-specific runtime policy is supplied by optional adapter distributions.

## Components

- `compat/shim/resources/torch/__init__.py` is the thin installed and deployed
  `torch/__init__.py` entry point.
- `resources/stubs/` contains bundled compatibility packages such as `flash_attn`,
  `torchvision`, `torchaudio`, and `torchdata`.
- `cpp_extension/` provides the Jittor-backed `torch.utils.cpp_extension`
  surface used to build source extensions.
- `runtime.py` implements isolated runtime setup and native extension discovery;
  `bootstrap.py` is its stable, small public facade.
- `deploy.py` installs the complete shim tree into a target site-packages
  directory.

The canonical torch-compatible API lives in `jittor.compat.torch`. The shim
exports that API under the `torch` module name and wires the submodule paths
expected by third-party libraries.

`jittor.torch_shim` remains an import-compatible alias of
`jittor.compat.shim`; both names resolve to the same module objects.

## Bootstrap

For development, install both projects from the checkout root:

```bash
python -m pip install -e . -e ./compat
```

The optional project maps the top-level `compat/` source tree to the installed
`jittor.compat` package. A `PYTHONPATH=python` setting alone selects only core;
install the optional editable project in the same interpreter when using
compatibility APIs or running their tests. When testing a second checkout,
install that checkout's `compat` project too. Nox runtime gates do this explicitly.

For wheel deployment, install the matching core and `jittor-torch` wheels in one
environment. Build them separately with `python -m build .` and
`python -m build ./compat`; the latter declares its core version dependency.
The independent wheel provides `torch` itself. The deployment helper below
additionally installs the bundled third-party stubs when the application needs them.

Applications that need a project-local runtime can enable it explicitly:

```python
from jittor.compat.shim import activate, activation_status

activate(project_root=__file__)
assert activation_status().active
import torch
```

The default activation and deployed `import torch` publish independent Tensor,
Parameter and Module types: `torch is not jittor`. Native Var methods retain
their own contracts. For the legacy Jittor-alias mode, explicitly pass
`independent_namespace=False`; an already activated process cannot change modes.
The deployed entry sets `JITTOR_TORCH_INDEPENDENT=1` alongside its activation
flag so child interpreters select the same mode. `JITTOR_TORCH_SHIM=1` alone
continues to select the legacy import-time path.

`activate()` is process-wide and idempotent; repeated calls return the original
activation result without rescanning extensions or reapplying patches. It creates a runtime below
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
jittor-torch-shim --target /path/to/site-packages
```

The target contains the torch package, bundled stubs, and distribution metadata.
Do not copy only `resources/torch/__init__.py`; the nested stub modules and interfaces are
part of the runtime contract.
