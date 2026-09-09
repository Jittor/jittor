# Independent Tensor.to copy and device behavior

- Status: same-dtype explicit copies and dtype-only placement corrected
- Baseline: `ed3263e12`
- Date: 2026-09-08
- Owner: coord
- Review when: Tensor.to, clone or memory-format handling changes

Tensor.to(copy=True) now explicitly clones before optional dtype conversion,
so requesting the existing dtype cannot collapse the copy into an alias.
Without an explicit device, conversion retains the input's device rather than
following the process default. A no-op conversion without copy still returns
the original object.

Independent clone accepts preserve_format and contiguous_format for the
currently supported representation. clone/to reject other requested formats
rather than silently ignoring them; this does not implement general strided or
channels-last storage layouts.

Verification: two real integration cases passed on CPU in 2.89 s and CUDA in
7.54 s. New assertions compare data pointers for same-dtype copy=True, preserve
gradient/non-leaf behavior, check no-op identity, exercise explicit clone format
and verify a CPU tensor remains physically on CPU after dtype conversion while
CUDA is enabled. No full suite, wheel rebuild or NPU execution was performed.
7.12 and general Storage/layout fidelity remain open.
