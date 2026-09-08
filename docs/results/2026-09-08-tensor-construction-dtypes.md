# Tensor construction dtype boundaries

- Status: construction precision/defaults and Tensor alias entry repaired
- Baseline: `adc75ced3`
- Date: 2026-09-08
- Owner: coord
- Review when: tensor factories, default dtype or pointer introspection changes

Python values with explicit dtype are converted to the appropriate NumPy
transport dtype before native construction. Explicit float64 and DoubleTensor
therefore no longer pass through float32 first. Python floating/complex values
resolve the selected default dtype; NumPy arrays and scalars retain their own
dtype unless explicitly converted. Unsupported default complex combinations
raise rather than silently selecting a narrower complex dtype.

Independent Tensor shape/data construction and empty Parameter construction use
the current default dtype. set_default_dtype rejects non-floating types without
changing the previous selection. Tensor(existing_tensor) preserves dtype and
gradient connectivity through a native clone with the existing view writeback
record, instead of narrowing and disabling gradients. A separate real PyTorch
2.12.1 process confirmed the reference behavior: different Python object, same
data pointer/dtype, and a non-leaf alias for floating trainable input.

The alias check exposed Tensor.data_ptr returning id(self). A native read-only
_storage_address property now synchronizes and returns the actual first-element
address without migrating the tensor. Original host/device raw-pointer
interfaces retain their migration behavior. This is not a claim that the legacy
Storage facade now exposes a correct allocation base or shared-storage extent.

Verification: eleven focused CPU construction/NumPy/integration cases passed in
3.40 s, and two real CUDA integration cases passed in 7.33 s after incremental
core compilation. They include float64 values differing from 1 by 2**-52,
NumPy scalar preservation, default-float64 Tensor/Parameter construction,
invalid default rejection, DoubleTensor precision, alias dtype/leaf identity,
shared data_ptr, view writes and CUDA residency after pointer inspection.
No full suite, new wheel or NPU execution was performed. 7.12 and full storage
semantics remain open.
