# Torch shim

The Torch shim lets a Python application written against torch use Jittor as its
runtime. The reusable Torch API, deployment, extension building, import
patching, and external-backend mechanisms are all owned by the **separate
`jittor-torch` distribution**; the core `jittor` distribution **contains none of
these compatibility files**. Project-specific runtime policy is provided by
optional adapter distributions.

## Composition

| Path | Role |
| --- | --- |
| `compat/shim/resources/torch/__init__.py` | The installed and deployed `torch/__init__.py` entry point (very thin) |
| `resources/stubs/` | Compatibility packages shipped with the distribution, such as `flash_attn`, `torchvision`, `torchaudio`, `torchdata` |
| `cpp_extension/` | The Jittor-backed `torch.utils.cpp_extension` interface, used to build source extensions |
| `runtime.py` | Setup of the isolated runtime and native extension discovery |
| `bootstrap.py` | A stable, thin public facade over the above |
| `deploy.py` | Installs the complete shim tree into the target site-packages |

The canonical Torch compatibility API lives in `jittor.compat.torch`. The shim
exports that API under the `torch` module name and wires up the submodule paths
third-party libraries expect.

`jittor.torch_shim` remains an import-compatibility alias of
`jittor.compat.shim`; both names resolve to **the same module objects**.

## Bootstrap

During development, install both projects from the checkout root:

```bash
python -m pip install -e . -e ./compat
```

The optional project maps the top-level `compat/` source tree onto the installed
`jittor.compat` package. **Setting only `PYTHONPATH=python` selects the core
alone**; to use the compatibility API or run its tests, install this optional
editable project in the same interpreter. When testing a second checkout,
install that checkout's `compat` project as well. The nox runtime gate does
exactly this explicitly.

For wheel deployments, install the matching core wheel and the `jittor-torch`
wheel in the same environment. Build them with `python -m build .` and
`python -m build ./compat` respectively; the latter declares a dependency on the
core version. **`torch` itself is provided by this separate wheel.** The
deployment helper below additionally installs the bundled third-party stubs when
the application needs them.

Applications that need a project-local runtime can opt in explicitly:

```python
from jittor.compat.shim import activate, activation_status

activate(project_root=__file__)
assert activation_status().active
import torch
```

Activation and the deployed `import torch` publish **independent** Tensor,
Parameter, and Module types: **`torch is not jittor`**. Native Var methods keep
their own contracts. **This is the only form:** the historical alias mode that
installed the Torch API onto the native `jittor` module has been removed;
`independent_namespace=False` and `install(jittor)` raise a migration error
directly (`require_independent_frontend()` in `compat/shim/preflight.py`).

Setting `JITTOR_TORCH_SHIM=1` **is the switch**: `import jittor` assembles the
same independent `TorchNamespace`, without needing PYTHONPATH or a prior
deployment. It gives the same result as `activate()` — `torch is not jittor`,
`torch.Tensor is not Var`. The deployed entry
(`compat/shim/resources/torch/__init__.py`) only sets `JITTOR_TORCH_SHIM=1`; it
also **raises directly** if `JITTOR_TORCH_INDEPENDENT` is set to a falsy value,
because that value names the removed frontend. This variable does not need to be
set today.

There is one ordering requirement: the environment variable makes `torch`
importable, but **cannot put the module on `sys.path`**. So when `import torch`
is the first import in the process, a deployed `torch` package is still required
(or `compat/shim/resources` on PYTHONPATH); importing `jittor` first removes
that need. The contract is pinned by
`compat/tests/torch/test_env_activation_contract.py`.

`activate()` is **process-wide and idempotent**: repeated calls return the
original activation result and do not rescan extensions or re-patch. Unless
`JITTOR_TORCH_RUNTIME_ROOT` is set, it creates a runtime under
`${XDG_CACHE_HOME:-~/.cache}/jittor/torch-shim/`, placing Jittor, torch
extensions, CUDA, Triton, pip, and temporary caches under that runtime, and
deploys the shim into its local site-packages.

Local source extensions are discovered from signals in `setup.py`,
`pyproject.toml`, and `CMakeLists.txt`. Missing or stale setuptools extensions
are rebuilt through the Jittor-backed cpp-extension API. Set
`JITTOR_TORCH_SKIP_EXT_BUILD=1` to skip the warm-start build check, or pass
`extension_dirs` explicitly when auto-discovery is unsuitable.

**For numerical consistency**, unless `JITTOR_TORCH_KEEP_FAST_MATH=1` is set,
the bootstrap disables CUDA fast-math contraction for Jittor's JIT kernels. A
project's own extensions keep the options requested by their build definitions.

## Optional adapters

Adapters use two public entry-point groups:

- **`jittor.module_patches`** — register exact module-path callbacks through
  `jittor.compat.module_patcher`;
- **`jittor.external_backends`** — register extension-discovery policies through
  `jittor.compat.external_backend`.

The adapter maintained in this repository is a separate distribution:
`jittor-torch-adapters` (source in
[`adapters/`](https://github.com/Jittor/jittor/blob/master/adapters/README.md),
Python package `jittor_adapters`, depending on the same version of
`jittor-torch`). It registers three `jittor.module_patches` entry points:

| entry point | Version | Role |
| --- | --- | --- |
| `jittor_transformers` | Transformers 4.56.2, 5.5.3 | Make Transformers' own `torch_npu` probe return false |
| `jittor_torchmetrics` | TorchMetrics 1.7.4 | Private bincount / concatenation / safe-divide helpers preserve existing Jittor behavior |
| `jittor_vllm` | see `adapters/README.md` | vLLM's compiled extension surface and layer patches |

Installing it makes the entry points discoverable; when it is not installed they
are reported as `unavailable` and the frontend is unaffected. Versions are
explicit: an unrecognized version raises `UnsupportedAdapterVersion` when the
patch is applied. **Jittor itself does not import these third-party libraries,
inspect their directory structure, or install long-lived project-specific import
finders.** Glue code for other projects (TRELLIS, Gaussian Splatting, and the
like) is made into its own optional distribution with the same entry-point
shape, **not in this repository and not shipped with this release line**.

## Deployment

`jittor-torch-shim` is the command provided by the `jittor-torch` distribution
(`[project.scripts]` in `compat/pyproject.toml`, implemented in
`compat/shim/deploy.py`):

```bash
jittor-torch-shim                            # 部署进当前环境的 site-packages
jittor-torch-shim --target /path/to/site-packages
jittor-torch-shim --check                    # 校验已部署的文件与源文件一致
```

It copies four things: `resources/torch/__init__.py` to
`<site-packages>/torch/__init__.py`, `resources/stubs/<package>/**` to their
respective top-level packages (`torchvision`, `torchaudio`, `torchdata`,
`flash_attn`), and two distribution metadata entries,
`torch-<version>.dist-info` and `flash_attn-<version>.dist-info` (so libraries
that gate on version numbers can find them). The command is idempotent and can
be re-run after changing bundled resources. **Do not copy only
`resources/torch/__init__.py`** — the nested stub modules and interfaces are
part of the runtime contract, and `--check` lists each missing or mismatched
file.

After deployment it prints a verification command that reads `torch.__file__`,
`torch.__torch_version__`, and `torch.version.__version__` — that is, the
**Torch API level**, not the Jittor version exposed by `torch.__version__` on
the same module.
