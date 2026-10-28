# Complex dtypes

- Status: accepted, limitations registered
- Last reviewed: 2026-09-22
- Current baseline: `refactor-2.0` at `c7d9650a` (after syncing origin)
- Review trigger: second-order complex autodiff, native complex linear-algebra kernels,
  or FFT complex128 support landing

Jittor's `complex64` and `complex128` are **native dtypes**, not simulated with two
real tensors. This page explains how far support reaches and what is explicitly
unsupported.

## Supported scope

Parts covered by maintained tests on CPU and CUDA (both `complex64` and
`complex128` apply unless noted):

- **Construction and conversion**: constructing from NumPy complex64/complex128,
  zero initialization, scalar assignment, NumPy round-trip, real-to-complex and
  complex-to-real conversion, `complex64`↔`complex128` precision conversion
  (including gradients);
- **Arithmetic**: add, subtract, multiply, divide, negation, mixing with scalars,
  equality/inequality, ternary select;
- **Promotion**: mixing `complex64` with operands that require double precision
  (`complex128`, or real values of `float64`/`int64`/`uint64` width) promotes to
  `complex128`; mixing with narrower operands keeps `complex64`;
- **Structural operations**: reshape, transpose (including CUDA), indexing,
  slicing, broadcasting, concatenation, stacking;
- **Reductions and linear algebra**: `sum`, `mean`, `abs`, `conj`, 2-D and batched
  matrix multiplication, `inv`/`svd`/`svdvals`/`qr`/`eigh`/`pinv` (bridged through
  `nn.ComplexNumber`; both precisions verified);
- **Transcendental functions**: `exp`, `log`, `sin`, `cos`, `sqrt`;
- **Real and imaginary views**: `.real`, `.imag`, `.angle()`, `view_as_real`,
  `view_as_complex`, `polar` (selects complex64/complex128 by input component
  width);
- **Scalar extraction**: `.item()` returns a Python `complex` for both widths;
- **First-order gradients**: Wirtinger-type gradients for the arithmetic above,
  magnitude, matrix multiplication, bridged operations, transcendental functions,
  and precision-conversion casts; the joint (U,S,Vh) backward for `svd`/`svdvals`
  (including non-square and batched);
- VJP and the maintained linear-algebra interfaces accept and return native
  complex values.

Second-order gradient and JVP support **depends on the participating operators;
it is not that the whole complex64 dtype is unsupported**. Targeted CPU
verification on the above baseline has covered:

- 0-D real/complex bridging that preserves the scalar shape, second-order
  gradients of a real loss on the squared magnitude, and `.item()` returning a
  Python complex;
- basic JVP for complex64 square and `exp`, and JVP where real inputs produce
  complex outputs; the square case also covers scalar/vector outputs and
  `create_graph=False/True`;
- existing native complex64 VJP regressions.

The evidence is the added [scalar and JVP contract test](../../tests/autograd/test_complex64_scalar_contract.py)
and the [functional AD test](../../tests/autograd/test_complex64_gradfunctional.py). The
6 CUDA cases added for the scalar contract also passed this round, covering 0-D
bridging/second-order gradients, `.item()`, and square JVP; the existing
functional AD file above ran on CPU only this round, so its exp/VJP coverage
cannot be extrapolated to mean it was tested on CUDA this round.

## dtype and gradient invariants

- A `complex64` element occupies 8 bytes and `complex128` occupies 16 bytes; both
  are classified as **complex**, neither floating point nor integer.
- Complex-to-real conversion takes the real part; complex-to-bool is true when
  either component is nonzero.
- `abs(complex64)` returns `float32` and `abs(complex128)` returns `float64`;
  arithmetic and complex reductions keep the input's complex width, and mixed
  widths follow the promotion rules above.
- A bare Python/NumPy complex **scalar** (not attached to any Var) is narrowed to
  `complex64` when constructing a Var, the same convention as a bare Python
  `float` narrowing to `float32`; it does not mean complex support is limited to
  complex64 — see `tests/core/test_complex_scalar_operand.py`.
- Differentiable complex values can carry gradients, but a **scalar loss in
  reverse mode must still be real**.
- Gradients of binary complex operators conjugate the other operand following
  torch's real-loss convention; holomorphic unary functions conjugate the local
  derivative.
- Unimplemented gradients must raise explicitly rather than silently returning
  zero; the confirmed SVD callback numerical gaps are listed separately below.

## Explicitly unsupported

- **complex128 kernel support for FFT is not yet wired in.** The `_NO_COMPLEX128`
  skip marker in `tests/opinfo/definitions/fft.py` remains — dtype registration
  alone is not enough for the cuFFT/kernel side to support this width; that is
  separate follow-up work.
- **Complex `prod` on CUDA** lacks the required atomic multiply implementation.
  CPU is covered; CUDA must raise instead of returning a partial result.
- **Native complex JVP** depends on second-order autodiff that is not yet
  implemented, and `jvp` raises `NotImplementedError`; native complex VJP is
  supported. Second-order complex autodiff (including the backward of the `svd`
  backward) is out of scope here.
- **Higher-order derivatives of the complex linear-algebra callbacks** are still
  unimplemented: the opaque `numpy_code` backward of complex `inv`, `QR`, `SVD`,
  and `eigh` cannot be considered supported for higher orders just because the
  second-order gradients/JVP of basic arithmetic pass. JVP uses a double
  backward and remains subject to the same limitation when composed with these
  operators.
- **General complex eigendecomposition on CUDA** relies on an available CuPy
  linear-algebra path, which may be unavailable in some otherwise normal CUDA
  environments.
- Some transcendental functions are not yet supported.
- The gradient of complex SVD at repeated/degenerate singular values is a
  mathematically non-differentiable boundary, not an implementation gap.

A limitation is removed only when it **comes with a targeted regression test
covering the previously unsupported backend or derivative order**.

## Internal bridge

`view_as_real` and its inverse perform the `complex64[...]`↔`float32[..., 2]`
conversion on the device and preserve gradients; the 0-D bridging and second-order
CPU/CUDA regressions for a basic real loss are the targeted evidence above, and
must not be extrapolated to arbitrary derivative orders of all complex operators.
This is an **implementation bridge; zero-copy aliasing is not promised**.

Some functions in `jittor.linalg` still convert native complex values into the
internal `ComplexNumber` real/imaginary representation, run the existing real
algorithms, and convert back to native complex. `jt.nn.ComplexNumber` is retained
only as the internal bridge for these not-yet-rewritten linear-algebra
algorithms; **new public APIs must not introduce a second simulated complex
representation**. The bridge is deprecated for user-visible results, but can only
be deleted once every such kernel has a native implementation and equivalence
tests.

## Checklist for new complex operators

1. Specify the input promotion rules and output dtype independently of the kernel
   code;
2. Compare CPU forward against NumPy or an exact mathematical reference;
3. Add CUDA/NPU execution and device-consistency coverage on every device claimed
   to be supported;
4. Derive and test first-order gradients under the real-loss convention;
5. Cover zeros, branch cuts, empty tensors, batched and non-contiguous cases;
6. **Raise explicitly** for unsupported derivative orders or backends;
7. Update this page and the issue ledger; do not copy the experimental process
   into either.

Main regression files (master-branch paths; on refactor-2.0 the corresponding
files are [`tests/linalg/test_complex64_linalg.py`](../../tests/linalg/test_complex64_linalg.py),
[`tests/core/test_complex128_native.py`](../../tests/core/test_complex128_native.py),
[`tests/linalg/test_complex_spectral_gradients.py`](../../tests/linalg/test_complex_spectral_gradients.py),
and so on):
[`test_complex64_native.py`](https://github.com/Jittor/jittor/blob/master/tests/type/test_complex64_native.py),
[`test_complex64_linalg.py`](https://github.com/Jittor/jittor/blob/master/tests/linalg/test_complex64_linalg.py),
[`test_complex64_gradfunctional.py`](https://github.com/Jittor/jittor/blob/master/tests/autograd/test_complex64_gradfunctional.py),
and
[`test_complex.py`](https://github.com/Jittor/jittor/blob/master/tests/type/test_complex.py),
which covers the remaining internal bridge.
