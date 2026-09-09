# Native Python Core Packages

- Status: Implemented with focused runtime checks; broad gates deferred
- Date: 2026-09-07
- Baseline: `43e302888`
- Owner: coord
- Review when: native startup, public composition or compatibility flags change

## Change

The 3153-line `_runtime/core_api.py` implementation is replaced by a same-object
legacy alias. `_core/api.py` explicitly composes the existing public export set;
the package itself does not publish objects named `var`, `flags` or `hooks`,
which would shadow its implementation modules.

`var.py` owns tensor construction, operations and Var bindings; `module.py`
owns Module and parameter roles; `function.py` owns one-call autograd contexts;
`hooks.py` owns removable gradient hooks; `flags.py` owns scopes and the one
native Flags/RuntimeState initialization; `diagnostics.py` owns logging,
profiling and exit handling. Native saving/loading is now
`jittor.serialization.native`. Legacy public and core_api pickle globals still
resolve to the real objects, whose metadata identifies the new owners.

Imports are explicit. Runtime-only dependencies are local to their calls,
including Module/Function/hook edges and compiler logging. The existing import
checker reports 163 cyclic modules, down from 164, with all three contracts
satisfied and no threshold change. Native operator captures, dtype annotations,
kernel registration order and public Var aliases are retained. Exit hooks are
installed once by the composition entry before the remaining root startup.

## Focused Verification

Python 3.11; native CPU `nvcc_path=""`, native CUDA
`nvcc_path=/usr/local/cuda/bin/nvcc` (12.2, sm89), and CPU Torch mode
`JITTOR_TORCH_SHIM=1 JITTOR_TORCH_KEEP_HOME=1`. All use `PYTHONPATH=python`,
`use_mkl=0 use_mpi=0 use_nccl=0 DISABLE_MULTIPROCESSING=0` and isolated CPU/CUDA
JITTOR_HOME directories. The CUDA probe explicitly selects `use_cuda=1`.

All three processes completed successfully:

- square forward and custom Function backward against NumPy;
- Linear parameter gradients are finite;
- forward hook runs once and its handle can be removed;
- old core_api import resolves to the same composition module;
- old Module/Function/flag_scope/array/load/save pickle globals resolve;
- native save/load roundtrip of an integer NumPy array;
- final device synchronization.

After the runtime-only import cleanup, CPU import/square backward was checked
again. Static symbol and syntax checks accompanied each extraction. Existing
source-location, ownership and bootstrap-mock tests were updated; no test files
were added. The layout checker remains part of the commit check.

Per the user's request to prioritize large code migrations, this batch does
not repeat complete CPU/CUDA structure suites or rebuild a wheel. Those remain
required at the next larger layout checkpoint; the prior domain-package wheel
does not contain this core migration. NPU and other unavailable hardware are
not claimed as tested. 5.26 remains open for root/misc/pool/build/contrib work.
