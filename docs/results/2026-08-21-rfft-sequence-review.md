# CUDA rFFT 序列风险复核

- Status: old finding withdrawn; CPU/CUDA regression retained
- Date: 2026-08-21
- Baseline: `a755309b`
- Owner: FFT and Torch compatibility maintainers
- Review when: native complex views or gradients, FFT construction, lazy host
  transfer, or CUDA graph scheduling changes

## Finding

KI-FFT-001 recorded that CUDA `rfft` could lose its imaginary half-spectrum after
complex forward and gradient work in the same process. The current implementation
does not reproduce that outcome. Before editing, the original aggregate operation
order was run in four fresh CUDA processes from revision `4c8a827f`. Every run had
the same NumPy-reference maximum error: about `6.188e-08` for `rfft` and
`2.235e-08` for `irfft(rfft(x))`; the spectrum imaginary magnitude was
`1.02396846`, so it had not been zeroed or dropped.

Repository history does not identify a dedicated rFFT correction between the old
report and this review. The old observation may have been a probe artifact or an
indirectly corrected interaction, but the surviving evidence is insufficient to
assign a root cause. No FFT implementation change was made on that basis.

## Regression and probe contract

`TestRFFT.test_rfft_after_complex_forward_backward_sequence` now materializes
native complex view bridges, arithmetic, reductions, matmul, complex elementary
functions, and four first-order complex gradient graphs before checking the
`rfft` real and imaginary parts against NumPy and checking the `irfft` round trip.
The test runs under both CPU and the available accelerator flag; this review used
a real CUDA device.

The aggregate complex parity comparator no longer exempts `rfft` or
`irfft_rfft` as sequence-sensitive. Either numerical mismatch now fails the
comparison normally.

## Verification

The maintained source ran with Python 3.12.13, CUDA 12.2, and an RTX 4090. The
oracle was independent binary PyTorch 2.12.1+cu126 from the same Python
environment, with its compiled `torch._C` origin checked before use. State,
temporary files, and logs were isolated below `$JITTOR_LAB_ROOT/_state/`; first
JIT work and all subsequent suites ran serially.

| Gate | Result |
| --- | --- |
| Original aggregate order, four fresh CUDA processes | all four clean and numerically identical |
| Deterministic complex-forward/backward then rFFT regression, CPU + real CUDA | `1 passed` |
| Full Torch-mode FFT/einsum/complex compatibility module, CPU + real CUDA | `43 passed` |
| FFT OpInfo NumPy forward reference, CPU + real CUDA | `8 passed` |
| FFT OpInfo autograd/gradgrad declarations | `8 skipped` by explicit unsupported metadata |
| Aggregate Jittor vs independent PyTorch: `rfft` | passed, max absolute error `1.333e-07` |
| Aggregate Jittor vs independent PyTorch: `irfft_rfft` | passed, max absolute error `5.960e-08` |
| Repository layout | passed |
| `tests/structure`, after warming one timed-out fresh-import node | `213 passed`, 1511 subtests passed |

The aggregate comparison as a whole exited nonzero because this Python
environment has no CuPy, leaving eight unrelated complex linalg bridge items as
Jittor errors. The Jittor side saved 35 arrays with 11 recorded errors; the
independent PyTorch side saved 46 arrays with no errors. All FFT items, all
complex prelude gradients, and both newly non-exempt rFFT items passed comparison.

## Reproduction commands

With the device and isolated environment configured as described in
`agent/manuals/environment.md`:

```bash
FFT_OUT="${JITTOR_LAB_ROOT:-../jittor-lab}/_state/fft-sequence/parity-current"
mkdir -p "$FFT_OUT"
python -m pytest -q \
  tests/compat/torch/test_torch_compat_fft_einsum.py
python -m pytest -q tests/ops/test_ops.py -k 'fft or rfft'
python agent/skills/jittor-torch-diff/complex_cuda_parity.py jt "$FFT_OUT"
python agent/skills/jittor-torch-diff/complex_cuda_parity.py rt "$FFT_OUT"
python agent/skills/jittor-torch-diff/complex_cuda_parity.py cmp "$FFT_OUT"
```

## Limits

This review covers CPU and one real NVIDIA CUDA configuration; it does not make
an NPU or ROCm claim. The FFT OpInfo definitions still declare autograd and
gradgrad unsupported, so their eight skips are recorded rather than treated as
gradient evidence. A future failure must preserve copied host arrays immediately
after spectrum and round-trip evaluation so output-buffer reuse cannot corrupt
the diagnostic artifact.
