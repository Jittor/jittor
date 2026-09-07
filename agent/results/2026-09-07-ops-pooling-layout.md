# Tensor Operations And Pooling Layout

- Status: Implemented with short CPU/CUDA/Torch checks; broad gates deferred
- Date: 2026-09-07
- Baseline: `830272fce`
- Owner: coord
- Review when: operation exports, bootstrap registration or pooling semantics change

## Ownership

`jittor.ops` is now a normal Python package. Its native operator attributes
resolve to the original `jittor_core.ops` callables, preserving low-level
signatures and live native registrations. The native module itself is unchanged.
Torch's existing OpsDispatcher captures this package during compatibility
composition; it is not replaced after Torch activation.

The 3058-line miscellaneous tensor file is split into fifteen implementation
domains and an ordered export/registration facade. Concatenation, basic indexing,
reductions and shape composition/transforms also live under `ops/`. No new
operator implementation file exceeds 500 lines. The old `misc` namespace keeps
its previous public surface; its six child files are same-object aliases, not
duplicate implementations. Dynamic `jt.misc` lookups for scan, CTC, meshgrid and
other override points are retained.

CUDA tensor registrations now have an explicit root bootstrap import before NN
composition. Compatibility attributes lazily return the same CUDA callable
objects; lazy attribute access is not responsible for backend registration.
Native argsort is captured before compatibility replacement, and Var bindings,
kernel keys and RNG initialization retain their original ordering.

Pooling mathematics and validation live in `nn.functional.pooling/`; modules
own constructor parameters and delegate their execution. The legacy pool
namespace and submodules only re-export canonical objects. The historical
AdaptiveAvgPool2d fixed-window algorithm is preserved separately from the
modern NN overlapping-bin behavior, with its original public class name.
Legacy AvgPool adapters are similarly explicit. `pool_use_code_op` has one
owner, with live read/write views through the old pool and NN spellings.

## Short Verification

Python 3.11, `PYTHONPATH=python`, `use_mkl=0 use_mpi=0 use_nccl=0`,
`DISABLE_MULTIPROCESSING=0`. CPU uses `nvcc_path=""`; CUDA uses
`nvcc_path=/usr/local/cuda/bin/nvcc` (12.2, sm89) and explicitly selects CUDA.
Separate CPU/CUDA JITTOR_HOME directories avoid concurrent shared-cache JIT.
CPU Torch also sets `JITTOR_TORCH_SHIM=1 JITTOR_TORCH_KEEP_HOME=1`.

All three processes completed successfully, checking:

- native ops callable identity and old miscellaneous submodule identity;
- cumsum and stack against NumPy;
- indexed-square gradients against explicit values;
- max/average pooling against explicit 4x4 reference values and finite gradients;
- old pool pickle globals and class aliases;
- shared pooling-switch writes, followed by restoration;
- final device synchronization.

The import checker reports 158 cyclic modules, down from 163, with all three
contracts passing and no relaxed threshold. Extraction checked 127 tensor
definitions and 22 pooling constructor/execution bodies after normalizing their
new owners; embedded CPU/CUDA source strings are unchanged. Existing source
location and metadata contracts were updated, without adding test files.

Full structure suites, broad numerical tests and a new wheel are deferred to
the next larger layout checkpoint under the user's requested faster workflow.
No NPU/ROCm/Corex hardware execution or optimization is claimed. 5.26 remains
open for build/contrib/root cleanup and the remaining oversized Python files.
