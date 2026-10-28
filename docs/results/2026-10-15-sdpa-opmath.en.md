# SDPA low-precision accumulation and higher-order gradients

- Status: the CPU/CUDA numerical and gradient regressions scoped to this document are verified.
- Date: 2026-10-15; maintainer: native attention and CUDA backend maintainers.
- Synchronization baseline: `626dcd6823dfd4ee45200374fa447c39e91797f2`; the fix is determined by the commit to which this report belongs.
- Recheck conditions: changes to the attention math path, the Function tape, AMP, cuDNN, or input storage semantics.

The original math path rounded early in low-precision QK, the probability-times-V product,
and backward. In the saved real `[1,4,2,8]` Q/K/V reproduction, the BF16 forward maximum
normalized error exceeded the existing `0.002` budget.
Now the Q/K/V, softmax, and intermediate backward computations for FP16/BF16 use FP32,
converting back to the original dtype only at the public outputs and returned gradients;
AMP is disabled locally and the caller's settings are restored on leaving the scope.
FP64 stays FP64.

Backward recomputes the row inner products from the probabilities and the probability
gradients. Each Function call saves a stable reshape holder over the original graph before
the tape, so second-order gradients connect to the original inputs and a caller `assign`
rebinding or reuse of the same Function instance cannot overwrite the previous state.
This is not a deep copy of the underlying storage.

Real cuDNN can execute the above BF16 small-head inputs, but the output error is `0.002513`
and the dQ error is `0.03273`, exceeding the original forward/gradient `0.002/0.01` budgets.
Therefore BF16 with head_dim<32 uniformly takes the native FP32 math, including inference
and GQA; this is a numerical admission policy. The FP16 and existing D32-and-up cuDNN
admission conditions are retained.

The forced-MATH raw arrays of the standalone binary Torch 2.13.0+cu130 are unchanged.
The final CPU FP16/BF16 default and explicit math outputs and the three input gradients are
all bitwise equal; CUDA FP16 explicit math and BF16 default and explicit math are likewise
bitwise equal. The CUDA FP16 default cuDNN path has output/gradient maximum normalized
errors of `0.000628/0.004728`, satisfying the original budgets.
These are correctness results on fixed inputs and do not constitute full Omni model
acceptance or a performance conclusion.

Verification includes eight new CPU cases, plus 39 cases on real CUDA covering the new and
old attention and the full cuDNN targeted set, all passing with no skips; the CUDA memory
threshold also passes. The cases cover bool/additive masks, fully masked rows, AMP, the
small-head common path, input rebinding, and same-instance reuse.
The FP64 first-order and finite-difference HVP cases explicitly run on CPU. Core is
322 passed, 42 skipped, 1 pre-existing xfail.

Using separate CPU/CUDA compilation caches and selecting the actually available CUDA
toolchain; mixed-device tests default globally to CPU and switch to CUDA explicitly from the
tests, and all new device computations check residency and forbid backend fallback:

```bash
JITTOR_TORCH_SHIM=0 JITTOR_TEST_DEVICE=cuda python -m pytest -q \
  tests/nn/test_sdpa_opmath.py tests/nn/test_sdpa_memory_efficient.py \
  tests/backends/cuda/test_cudnn_fused_attention.py
python tools/run_test_suite.py --tier core --serial-compile
```

The platform is Linux x86_64, Python 3.11, RTX 4090/SM89, nvcc 12.2.140.
The raw logs, standalone Torch arrays, and field-by-field comparisons are in the
unversioned `$JITTOR_LAB_ROOT/_state/sdpa-opmath-integration-626/`; the original oracle is
at `$JITTOR_LAB_ROOT/_state/vllm-evidence/20261015/saved-sdpa-v1/`.
Other GPUs, ROCm, NPU, and full models are outside this acceptance scope.
