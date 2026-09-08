# Tensor copy boundaries

- Status: independent clone/copy constructors use explicit native copy storage
- Baseline: `861541393`
- Date: 2026-09-08
- Owner: coord
- Review when: clone, tensor/as_tensor or device-preserving copies change

Independent Tensor.clone and torch.clone now call the native CopyOp, which
allocates separate storage and retains gradient connectivity. Native CloneOp
shares storage and remains the explicit implementation for Tensor(existing)'s
alias behavior. A CPU tensor is copied under a CPU scope when the process is
otherwise using CUDA, preserving its actual residency.

torch.tensor(existing) copies and detaches before applying requires_grad, so a
requested trainable result is a new leaf and does not share the source's data
pointer. Native Var inputs use the same typed conversion boundary instead of a
NumPy roundtrip. as_tensor preserves an existing Tensor's identity when dtype
matches and creates an alias frontend holder for native Var inputs.

Verification: eleven focused CPU construction/NumPy/integration cases passed in
3.29 s, and two real CUDA integration cases passed in 7.61 s. Checks compare real
data pointers, verify clone gradients and tensor-copy leaf status, preserve
Tensor(existing) alias writes, test native Var interop and unchanged-dtype
as_tensor identity, and verify CPU copies remain on the host under CUDA. No
full suite, wheel rebuild or NPU execution was performed.

7.12 remains open for full API/model coverage and shared Storage/stride
semantics. This change does not claim zero-copy NumPy interoperability or
complete memory-format support.
