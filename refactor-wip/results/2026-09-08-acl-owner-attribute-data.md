# ACL production attribute data migration

Status: host implementation and checks; NPU validation remains required.
Baseline: `4a7aae78d`. Owner: backend maintainers. Recheck after any runner schema change.

All Python ACL attribute families now pass typed records through CodeOp data.
The existing Softmax/SoftmaxBackward/Triu/Flip/Cumsum/Gather/Scatter path is
extended to convolution, normalization, pooling, upsampling, concatenation,
stack/split, slice/assignment, range, activation/dropout, embedding backward,
NanToNum, FlashAttention/incremental attention/KV cache, transpose, roll,
matmul/batched matmul and truth reduction. Forward, backward and compound
runner programs share `_code.py`; repeated command builders are aliases.

`attribute_program(name, values, variable=..., slot=...)` combines a fixed C++
assignment statement with an encoded payload under `acl_payload.<slot>.`.
`code_program(parts)` merges fragments and rejects conflicting payloads.
The primary `attributes=` path retains `acl_attr.`; these namespaces are
disjoint. Slots identify different runner instances, including transpose's
inverse and both matmul derivatives. Generated source never reconstructs
attribute values from strings. The C++ decoder validates that the registered
schema belongs to the actual runner; text layouts are UTF-8 byte vectors with
explicit byte and supported-spelling checks.

NumPy integral and real scalars are normalized without accepting fractional
axes or treating bool as int64. Signed int64 limits remain exact in both wire
directions. Structural execution choices remain structural: notably
`stridedsliceassignv2_grad` still clears the destination before its slice write.

`CodeOp::grads` copies data to grouped backward CodeOps, removing only
`multi_grad`, `multi_grad_output`, and `multi_grad_input_count`. The new CPU
node `tests/ops/test_code_op.py::TestCodeOp::test_multi_output_grad_preserves_data_without_gradient_controls`
checks both data-dependent numerical gradients and absence of these controls.
Its runtime result must be recorded by the integrating maintainer against a
core containing this change; an old loaded core is insufficient.

Host verification: **50 passed in 3.57 s**, using `test_acl_production_attributes.py`,
`test_acl_code_data_wire.py`, and `test_acl_data_schema_normalizer.py`.
It exercises actual Python owner payloads, Python encoding consumed by C++,
real C++ attribute classes, schema/runner mismatch rejection, and NumPy scalar
acceptance. The ACL host syntax skill additionally parses real MatMul,
BatchMatMul, Roll, GroupNormBackward, FlashAttentionBackward,
StridedSliceAssignV2 and TruthReduce runner templates with CANN stub headers.
The runner-template TU passed. Undefined-name lint and whitespace checks also
passed. This does not validate CANN query signatures, NPU execution or numerical math.

Pre-existing math issues are deliberately unchanged: `StackACL` does not set
the `self.dim` read by its backward, and its backward split sizes describe
input dimensions rather than singleton stack entries; DropoutBackward still
uses scale 1.0 independent of forward probability. These require separate
semantic fixes and real NPU comparison. Descriptor caches and remaining ACL
resource organization are not completed by this attribute migration.
