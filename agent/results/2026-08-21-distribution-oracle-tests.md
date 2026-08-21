# 分布采样与独立 PyTorch oracle 测试收口

- Status: verified; KI-COMPAT-002 withdrawn
- Date: 2026-08-21
- Baseline: `a40b775b`
- Owner: distribution, test infrastructure, and Torch compatibility maintainers
- Review when: distribution sample-shape semantics, deployed Torch shim
  detection, or independent-oracle loading changes

## Finding

`Normal.sample(sample_shape)` already used the maintained
`sample_shape + batch_shape` implementation. Before editing, batched parameters
produced `(4, 2, 3)` for `sample((4, 2))` on CPU and real CUDA, and independent
binary PyTorch produced the same shape on both devices. The active known-issue
entry survived because the original focused regression still carried an old
`unittest.skip` decorator. The broader distribution-shape module had been
passing the same contract without removing that skip.

The expanded regression also found two test-infrastructure problems. In an
environment where the deployed Jittor-backed top-level `torch` package was
discoverable but not imported, `modules_available("torch")` returned true. The
module setup then correctly rejected it for not being preloaded independent
PyTorch, turning every test in the module into a setup error. With a real oracle
loaded, an old OneHotCategorical test supplied arbitrary category integers
instead of one-hot events, while the Geometric test used unseeded probabilities
over the open unit interval and could cross its float32 tolerance near an
endpoint.

## Change

The stale Normal skip and its obsolete defect prose were removed. The optional
dependency probe now parses a discoverable Torch package's source without
importing it and rejects the canonical
`_jittor_torch_shim_placeholder = True` assignment. A structure test locks this
pre-import behavior.

OneHotCategorical parity now constructs legal one-hot events from category
indices. Geometric entropy, log-probability, and KL checks use four deterministic
interior probability pairs and a `1e-5` tolerance; the measured CPU and CUDA
maximum absolute errors were at most `1.91e-6`.

## Verification

Tests used Python 3.11.15, GCC 12.3, Jittor 1.3.11.0, CUDA 12.2, and an NVIDIA
GeForce RTX 4090 (`sm_89`). The independent oracle was binary PyTorch
2.12.1+cu130, verified through the origin of `torch._C`. Jittor state was
isolated below `$JITTOR_LAB_ROOT/_state/`, and initial compilation was serial.

| Gate | Result |
| --- | --- |
| Pre-change Jittor Normal shape, CPU and real CUDA | `(4, 2, 3)` on both |
| Independent PyTorch Normal shape, CPU and CUDA | `(4, 2, 3)` on both |
| Pre-change compatibility distribution modules | `33 passed, 1 skipped` |
| Post-change compatibility distribution modules | `34 passed` |
| Native distribution modules without external Torch | `14 passed, 9 skipped` |
| Distribution modules with independent PyTorch | `23 passed` |
| Focused optional-dependency structure checks | `5 passed` |
| Full pytest-contract module | `16 passed` |
| Repository layout | passed |
| `tests/structure` | `215 passed` |

The first full structure invocation refreshed `jit_utils` and exited with the
documented request to rerun. The unchanged second invocation produced the result
above.

## Maintained commands

With environment variables configured according to
`agent/manuals/environment.md`:

```bash
python -m pytest -q tests/compat/torch/test_torch_compat_distributions.py \
  tests/compat/torch/test_torch_compat_distrib_shape.py
env -u REAL_TORCH_SITE python -m pytest -q \
  tests/core/test_distributions.py tests/core/test_gamma_distribution.py \
  tests/core/test_distributions_grad.py
REAL_TORCH_SITE="$REAL_TORCH_SITE" python -m pytest -q \
  tests/core/test_distributions.py tests/core/test_gamma_distribution.py \
  tests/core/test_distributions_grad.py
```

## Limits

This is a test-contract cleanup; distribution runtime code did not change and no
performance claim is made. NPU and ROCm were not available. The complete native
plus Torch two-process repository suite was not rerun end to end in this focused
change.
