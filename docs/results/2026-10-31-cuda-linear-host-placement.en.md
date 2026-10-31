# CUDA `nn.Linear`'s cuBLASLt route: admit by operand placement, not by the process flag

- Status: verified on a real CUDA device. This changes `lt_linear_cuda`'s
  admission test only; the values it computes are unchanged.
- Date: 2026-10-31
- Baseline commit: `3dc5c7ef3` (reproduction); the fix is the commit this page
  lands with.
- Verification scope: RTX 4090 24 GB, CUDA 12.2.140 with the `jittor[cuda12]`
  component wheels (`pipcu122`), against PyTorch 2.11.0+cu128; both sides on the
  same card.
- Maintainer: Jittor core maintainers
- Review conditions: the admission tests of `lt_linear_cuda` /
  `lt_linear_train_cuda`, the meaning of `Var.placement_backend`, or how the
  Torch frontend records a host tensor's placement

## Symptom

The ConvNeXt case of `compat/tests/torch/test_torch_hf_models.py` raised on
`pwconv1(features)` (`modeling_convnext.py`) during a forward:

```
JittorError: code requires source for the selected backend (code_op.cc:408)
op: code
  in:  float32[1,8,8,32,], float32[128,32,], float32[128,]
  out: float32[1,8,8,128,]
```

That same case used to die earlier on a different defect (the `var_relay`
invariant, see `2026-10-31-cuda-var-relay-decline.md`), and the Var it left
behind poisons every later `sync`, so this one stayed invisible until each
`CFG` architecture was run on its own.

## Root cause

`lt_linear_cuda` carries only `cuda_src` -- it is a CUDA-only `jt.code` op --
and its admission test `_supports` asked only `jt.flags.use_cuda`. That flag
says where a *new* tensor would go, not where these live: the Torch frontend
keeps a `torch.tensor(...)` on the host until the caller moves it, while the
shim sets `use_cuda` for the process at startup. The op was therefore built for
CUDA and resolved to the CPU backend at execution, where it has no source.

## Fix

The admission test now reads each operand's *explicit* placement
(`Var.placement_backend`: `0` is `BackendId::Cpu`, `-1` means no explicit
placement, so the ambient one -- CUDA, since the flag is on -- decides):

```python
def _on_the_device(*values):
    for value in values:
        placement = int(value.placement_backend)
        if placement >= 0 and placement != 1:   # 1 is BackendId::Cuda
            return False
    return True
```

Both `_supports` (forward) and `_supports_training` use it now. The reading
follows `ops/concatenation.py`, which reads the same attribute the same way. It
costs 0.04 us an operand. The first implementation asked `dispatch_context`
instead: 3 us a layer, because it builds a tuple and backend names on every
call -- not acceptable in a module whose whole point is host time, so it was
replaced with the attribute read.

**The same operator has a native fast path.** A biased inference `nn.Linear`
also reaches `linear_with_bias_inference` in
`src/bindings/pyjt/py_module_call.cc`, which builds the same `code` operator
directly in C++ and saves the 8.6 us of Python graph construction. Its
admission test looked at `use_cuda` and `accelerator_backend_id()` the same way,
so the first version of this fix repaired only the Python route: the first
ConvNeXt forward passed and the second one failed on the same operator (the
error is raised by the Var the first forward left behind, on the next `sync`).
That path now applies the same placement rule: an explicit placement whose
backend is not CUDA declines.

## Results

- ConvNeXt (batch 4, `torch.no_grad()`) runs five forwards in a row, each
  returning `(1, 64, 4, 4)`. The first version of the fix let only the first
  forward through; with both routes repaired it does not reproduce.
- `compat/tests/torch/test_torch_hf_models.py::TestTorchHFModels::test_forward_and_eval_determinism`
  (33 architectures, two forwards each) is **1 passed** in 97 s once this fix is
  paired with the `var_relay` decline (`2026-10-31-cuda-var-relay-decline.md`).
- `tests/nn/test_lt_linear_cuda.py`: 17 passed, including the new
  `test_a_host_placed_operand_declines_even_when_use_cuda_is_on` (it stamps an
  explicit CPU placement on the operands with `_set_construction_placement(0,
  0)` and asserts the fast path declines while the portable path returns the
  same values).
- `compat/tests/torch/test_bfloat16_linear_rounding.py`,
  `test_torch_amp_boundaries.py`, `test_nn_layer_subclassing.py`: 40 passed.
- `tools/run_test_suite.py --tier core`: 327 passed / 1 xfailed.
- Paired benchmark (same 4090, `benchmarks/torch_compat`, TF32 on both sides):
  `vit_b16_train` 191.5 / 169.4 ms (0.89x), `qwen3_train` 300.0 / 177.6 ms
  (0.59x) -- the same as before the change, with no measurable host cost.

