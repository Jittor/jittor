# Independent checkpoint loading

- Status: checkpoint and safetensors frontend/dtype boundaries repaired
- Baseline: `e52c07857`
- Date: 2026-09-08
- Owner: coord
- Review when: storage rebuild, dtype conversion or safe_open adapters change

Portable torch.save payloads now retain dtype, requires_grad and Parameter role.
Loading reconstructs through the target Tensor/Parameter factories, including
BF16 whose NumPy transport representation is float32. Map-location conversion
preserves a loaded Parameter's holder identity and leaf state. Real Torch zip
tensor rebuilding uses the saved dtype and requires_grad rather than native
jt.array's automatic narrowing; Parameter rebuild restores its role and optional
state. Existing checked stride extraction remains responsible for values.

The safetensors Torch adapter likewise constructs target tensors with explicit
dtype and requested device. Slice handles retain dtype/device; scalar shape is
preserved. BF16 saving writes uint16 payload words from the float32 transport
bits and marks the header BF16. Unsupported float8/unknown dtype paths raise
instead of interpreting integer bytes as floating values. NumPy load/save are
no longer patched with Torch-returning functions, and non-Torch safe_open calls
delegate to the original implementation.

Verification: twelve focused CPU cases passed in 5.15 s and three real CUDA
integration cases passed in 14.64 s, no skips. Cases include portable int64/BF16
and Parameter roundtrip, wide int64/float64 strided zip rebuilding, safetensors
scalar/dtype preservation, CPU-targeted loading under a CUDA runtime, slices,
BF16 header/data roundtrip and unchanged NumPy ndarray ownership. Prior
deployment/training/failed-install integration remains included. No full suite,
new wheel or NPU execution was performed.

7.12 remains open. Materializing a strided checkpoint's correct values does not
establish preservation of shared storage aliases or arbitrary stride semantics.
The existing map_location/device and full external-model compatibility scope
still need their broader completion evidence.
