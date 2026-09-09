# Native Tensor placement validation

- Status: Implemented; focused CPU/CUDA checks passed
- Baseline: `b2e03162b`, with the coordinator's current NN/distribution/protocol integration
- Date: 2026-09-08
- Owner: native Runtime and independent frontend maintainers
- Review when: graph placement, scalar conversion, fusion, copies or factories change

Explicit placement now belongs to native Var/Op graph state. CPU tensors created
or loaded while the Runtime defaults to CUDA retain CPU placement through later
operations. The implementation preserves native FollowRuntime behavior and the
shared graph/meta-operator architecture. The [contract](../architecture/tensor-backend-placement.md)
describes construction scopes, dispatch, fusion, per-segment allocation and copies.

Two actual defects drove the regression cases. Previously a checkpoint mapped to
CPU retained CUDA affinity, and its next operation moved it back to GPU while a
Python force_cpu hint still reported CPU. During the placement implementation,
the existing pending-scalar retargeter also changed published scalar tensors:
a CUDA scalar became CPU when used with a CPU vector, and a CPU scalar became
CUDA with a CUDA vector, including its returned gradient. A real-device probe
recorded both changes before the publication/constructor-adapter correction.

The final implementation marks published native Vars, prevents their retargeting,
and inserts a local differentiable CPU-0-D-to-accelerator copy before typed
operator construction. It preserves the original scalar dtype-promotion bit.
CUDA-scalar/CPU-vector operations reject the mismatch without changing the source.
CPU sum() and zeros(()) scalar sources are covered as well as direct scalar factories.

## Executed checks

The final focused run used two visible CUDA devices (sm_89, CUDA 12.2.140), with
`use_cuda=1`, and executed CPU and CUDA kernels through the same built core.
Independent Tensor mode was enabled. **9 passed, 0 skipped**, in **343.12 s**:

- Seven nodes in `tests/compat/torch/test_native_tensor_placement.py`: lazy CPU
  factories/views/reductions/matmul/argsort; mixed CPU/CUDA submission; CPU → CUDA
  device 1 → CUDA device 0 → CPU copies and gradients; disabled default CUDA with
  explicitly placed CUDA tensors plus native FollowRuntime; NumPy/Parameter CPU
  ownership; published scalar direction/value/both-gradient checks; and CPU/CUDA
  frequency factories followed by real computation.
- `test_serialization_api_owners.py::test_cuda_checkpoint_mapped_to_cpu_stays_on_cpu_for_subsequent_ops`:
  int64 checkpoint values survive CPU mapping and subsequent arithmetic. The
  strict expected-failure marker was removed. Materialization occurs in the test
  before inspecting allocation residency, not as an implementation workaround.
- `test_independent_frontend.py::test_independent_tensor_installation_preserves_native_type`:
  failure/retry, independent types, native owner preservation, training,
  parameter/container behavior, storage, deepcopy and pickle composition.

Earlier integrated validation of the requested 19 nodes reported 18 passed / 1
failed. Its child-process environment did not inherit the parent runner's compat
source finder; the shared editable installation could resolve coordinator Python
against the validation tree's changed core. The final run pins compatibility
imports for subprocesses with a temporary sitecustomize loader. It does not alter
the shared environment or tests' assertions. The final independent node passed.

The first six-node placement run was 4 passed / 2 failed: both failures called the
public CUDA-only `to_device(-1)` wrapper. The implementation now uses the existing
native differentiable `_copy_to_cpu` entry. The remaining new scalar/frequency
coverage was added because concrete gaps were found, not to repeat a full suite.

CPU-only and CUDA-conditional translation-unit syntax checks passed during native
implementation. The standalone host construction-scope contract passed (1 node,
0.91 s), covering nested exceptions, thread isolation and ACL enum representation.
No NPU, ROCm, Corex or multi-machine execution was performed. This report makes no
performance claim and does not certify the full repository gate.

## Reproduction and source ownership

Build once from the validation worktree with its own JITTOR_HOME, then run the
two files/one independent node above using the same core. The environment includes
`JITTOR_TORCH_INDEPENDENT=1`, `JITTOR_TORCH_SHIM=1`, `JT_BACKEND=cuda`,
`use_cuda=1`, explicit NVCC/Python-config paths and two visible devices. Prefix
PYTHONPATH with the source Torch entry and core Python directory; when using an
editable compatibility install from a different checkout, pin the compat package
for both the parent and all child interpreters too.

The complete run used the validation tree's native and Python sources. Coordinator
NN/distribution/protocol changes were copied in before the integrated run. Later
coordinator-only allocator comments and protocol annotation adjustments were not
silently represented as runtime-tested changes. Git integration and the final
source comparison belong to the coordinator.

Unversioned artifacts are under `$JITTOR_LAB_ROOT/_state/tensor-placement-integrated-20260908/`:
`run.py`, `bootstrap/sitecustomize.py`, `warm.log`, `focused.log`, `integrated.log`,
`scalar_before.py`, `scalar-before.log` and `final-placement.log`. The original
checkpoint failure was recorded in the coordinator's `serialization-cuda-cpu-op.log`.
