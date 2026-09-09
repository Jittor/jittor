# Independent training owners

- Status: optimizer installation and Module conversion migrated
- Baseline: `e6dffd1c9`
- Date: 2026-09-08
- Owner: coord
- Review when: optimizer wrappers, parameter roles or Module conversions change

Independent installation now owns a separate optim namespace and real
Optimizer/algorithm subclasses. Existing native algorithms and compatibility
adapters are reused; no new update equations were introduced. Native optimizer
class dictionaries remain unchanged. Constructor scopes cover native state
buffer allocation, including work after the base initializer returns. Iterable
parameters and per-group iterables are materialized at the frontend boundary.
Public state_dict step tensors are constructed by the Torch owner.

Scheduler publication uses the same target optim namespace. The current
optimizer reference belongs to the target, and Module zero_grad and gradient
clipping resolve that owner. Optimizer-kind recognition includes the installed
frontend algorithm classes, preserving the explicit step-identity comparison
used when a consumer must not substitute different update mathematics.

Independent Module conversions use the native role accessor, so parameters,
buffers and ParameterList/ParameterDict contents follow one path. Parameter
holders remain the same objects, tied references remain tied, and converted
storage remains a leaf rather than retaining conversion history. Existing
gradient holders are converted too; floating buffers follow the requested
dtype and integer buffers retain their dtype. The legacy conversion path is
unchanged.

Verification: the real independent integration and existing optimizer file
passed 25 CPU cases in 19.93 s. The optimizer file also reported retained
runtime objects at session end (2 holders, 4 Vars, 4 Ops); this run is not a
full liveness audit. Real CUDA integration passed 2 cases in 26.26 s. Covered:
native Optimizer/SGD/Adam/AdamW class bindings unchanged, iterable parameter
inputs, first-step SGD/Adam/AdamW values against explicit formulas, target
state tensors, state_dict roundtrip preserving parameter identity, StepLR,
zero_grad, current-optimizer ownership, Module float64 conversion with tied
parameters/gradients/buffers, and ParameterList conversion. Existing deployment
and failed-install/retry checks remain in the integration. No full suite,
wheel rebuild, NPU or distributed hardware run was performed.

7.12 remains open for broader independent API/model coverage and previously
recorded storage/stride, shared native-child and mixed-thread boundaries. The
existing compatibility Adam adapters and FSDP update implementation still have
their separate mathematical-consolidation work; this change does not close it
by relabeling the optimizer owner migration.
