# Independent numerical namespaces

- Status: distributions/linalg/sparse publication ownership migrated
- Baseline: `37f9b0430`
- Date: 2026-09-08
- Owner: coord
- Review when: numerical namespace installation or scalar-index JIT changes

Independent linalg and sparse now have their own writable module facades;
sparse compatibility additions no longer modify the native module. Distribution
publication similarly owns a separate module tree and real subclasses matching
the native inheritance hierarchy. Native implementations are reused under the
Tensor frontend scope for constructors, sampling, probability methods and
properties. Constructors convert mathematical parameters while preserving
existing Tensor gradients. Normal accepts loc/scale spellings, and mutable
class metadata is copied. Reparameterized sampling checks current parameter
state without retaining a second parameter-reference list; opaque objects are
treated conservatively so their gradient paths are not discarded.

The actual Categorical.log_prob scalar-index call exposed an unrelated core
JIT defect: a rank-zero tensor index generated a negative stride-variable name.
Getitem and its setitem-add gradient now emit index strides only for positive
index rank; scalar index pointers continue reading element zero. Positive-rank
indexing and update mathematics are unchanged.

Verification: native linalg/sparse/distributions module bindings and native
distribution class bindings remain unchanged after independent installation.
Normal/Categorical/Bernoulli values and result types, Normal reparameterization
gradients and linalg inverse passed in the real integration. Final integration
checks: CPU 2 passed/2.88 s, CUDA 2 passed/7.32 s. The separate scalar-index case
passed CPU and CUDA forward/backward for positive/negative int32/int64 indices;
its first runs reproduced both getitem and setitem compilation failures. The
legacy indexing case reported retained Vars and was not a liveness audit.
No full suite, wheel rebuild or NPU execution was performed.

7.12 remains open for full API/model coverage, unsupported distribution
semantics, shared storage/stride fidelity and mixed-thread behavior. Existing
fallback distribution mathematics and constraint semantics were not expanded
or relabeled as fully Torch-equivalent.
