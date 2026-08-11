# Native Complex Dtype

- Status: Accepted with tracked limitations
- Last reviewed: 2026-08-12
- Baseline: `582fc51d`
- Owner: core dtype and linear-algebra maintainers
- Review when: complex128, second-order complex autograd, or native complex
  linear-algebra kernels land

This document records the durable contract for Jittor's native complex dtype.
Implementation history and individual experiment transcripts belong in Git and
`agent/results/`; they are intentionally not duplicated here.

## Decision

`complex64` is a first-class `jittor_core` dtype. Public tensor APIs should
consume and return native complex `Var` objects. The older
`jt.nn.ComplexNumber` real/imaginary pair remains only as an internal bridge for
linear-algebra algorithms that have not yet been rewritten. New public APIs must
not introduce another simulated complex representation.

`complex128` is not registered. Adding it requires expanding and auditing the
core dtype-size representation before exposing any constructor or promotion
rule.

## Supported contract

The maintained CPU and CUDA tests cover:

- construction from NumPy complex64, zero initialization, scalar assignment,
  NumPy round trips, and real/complex casts;
- add, subtract, multiply, divide, negate, scalar mixing, equality, inequality,
  and ternary selection;
- reshape, transpose, indexing, slicing, broadcasting, concatenation, and stack;
- `sum`, `mean`, `abs`, `conj`, 2-D and batched matrix multiplication;
- `exp`, `log`, `sin`, `cos`, and `sqrt`;
- `.real`, `.imag`, `.angle()`, `view_as_real`, `view_as_complex`, and `polar`;
- Wirtinger-style first-order gradients for supported arithmetic, magnitude,
  matrix multiplication, bridge operations, and transcendental functions;
- native-complex inputs and outputs for FFT, VJP, and the maintained
  linear-algebra surface.

The primary regression files are:

- [`tests/core/test_complex64_native.py`](../../tests/core/test_complex64_native.py)
- [`tests/core/test_complex64_linalg.py`](../../tests/core/test_complex64_linalg.py)
- [`tests/core/test_complex64_gradfunctional.py`](../../tests/core/test_complex64_gradfunctional.py)
- [`tests/core/test_complex.py`](../../tests/core/test_complex.py), for the
  remaining internal bridge

## Dtype and gradient invariants

- `complex64` has an eight-byte element size and is classified as complex, not
  floating-point or integer.
- Complex-to-real casts select the real component. Complex-to-bool is true when
  either component is nonzero.
- `abs(complex64)` returns `float32`; arithmetic and complex reductions retain
  `complex64` unless an explicit contract says otherwise.
- A differentiable complex value is allowed to carry gradients, but a scalar
  loss supplied to reverse-mode differentiation remains real.
- Complex binary gradients conjugate the opposite operand according to the
  real-loss convention used by Torch. Holomorphic unary gradients conjugate the
  local derivative.
- Unsupported gradients fail explicitly; they do not silently return zeros.

## Bridge boundary

`view_as_real` and its inverse convert between `complex64[...]` and
`float32[..., 2]` on device and preserve first-order gradients. The conversion is
an implementation bridge, not a promise of zero-copy aliasing.

Some functions in `jittor.linalg` currently convert native values to the
internal `ComplexNumber` substrate, execute an existing real/imaginary
algorithm, and convert outputs back to native complex tensors. That substrate is
deprecated for user-facing results but cannot be deleted until all such kernels
have native implementations and equivalent tests.

## Known limitations

- CUDA complex `prod` lacks the required atomic multiply implementation. CPU is
  covered; CUDA must fail rather than return a partial result.
- Native complex JVP relies on second-order autograd that is not implemented.
  `jvp` raises `NotImplementedError`; native complex VJP is supported.
- General complex eigendecomposition on CUDA depends on an available CuPy
  linear-algebra path and may be unavailable in otherwise valid CUDA setups.
- `complex128` and several transcendental operations are unsupported.
- Native complex linear algebra still uses the internal bridge described above.

These limitations are indexed in the
[known-issues ledger](../../agent/manuals/known-issues.md). A limitation is
removed only together with a focused regression that exercises the previously
unsupported backend or derivative order.

## Extension checklist

When adding a complex operation:

1. Specify input promotion and output dtype independently from kernel code.
2. Add CPU forward comparison against NumPy or a precise mathematical oracle.
3. Add CUDA/NPU execution and device-parity coverage where support is claimed.
4. Derive and test first-order gradients under the real-loss convention.
5. Test zero, branch-cut, empty, batched, and non-contiguous cases as applicable.
6. Raise an explicit error for unsupported derivative orders or backends.
7. Update this contract and the known-issues ledger without copying experiment
   history into either document.
