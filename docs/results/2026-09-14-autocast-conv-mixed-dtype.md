# Auto-mixed-precision requested mixed-dtype convolutions

- Status: Fixed; verified on a real CUDA device
- Date: 2026-09-14
- Baseline commit: `63fc1485` (the fix itself, on top of `5a084737`)
- Owner: Jittor compatibility maintainers
- Review when: the amp register's operand handling changes, or the shim's
  `torch.autocast` stops recording only `jt.flags.amp_reg`

## Question

`torch.autocast(float16)` over float32 weights is the verified MiniMax-H3
video-decode recipe. On the shim it built a float32-input / float16-output
convolution -- a request no cuDNN algorithm serves -- and `cudnn_conv3d` died on
`best_algo_idx == -1`, aborting the decode. What is the root cause, and what is
the fix rather than the `--vae-dtype float16` workaround?

## Root cause

`compat/torch/grad.py`'s `_AutocastContext` maps `torch.autocast` onto
`jt.flags.amp_reg`. The register biases the dtype an operator *infers for its
result* (`src/type/nano_string.h`: `float_dtype` under `amp_prefer16`) but does
not cast the operator's operands. Torch's autocast does the opposite: it casts
an operator's inputs to the compute dtype, and the result is low precision
because the computation is.

The cuDNN convolution ops create their output as
`create_output(nullptr, dtype_infer(x->ns, w->ns))`
(`backends/cuda/kernels/cudnn/cudnn_conv{,3d}_op.cc`). Under `amp_prefer16` a
float32 pair was therefore asked for a float16 result. An elementwise kernel
absorbs that (its generated code converts on store), but cuDNN needs one dtype
for input, filter and output and has no such algorithm; the op then failed its
own `best_algo_idx != -1` invariant.

## Fix

- `src/core/var.{h,cc}`: `cast_operand_to_compute_dtype(value, dtype)` returns a
  cast of a floating operand to the operator's compute dtype, or nothing when no
  cast is needed. The caller keeps the owner alive until the operator it builds
  has taken the cast as an input.
- The six cuDNN convolution constructors (2-D and 3-D, forward and both backward
  directions) compute `dtype_infer` once, cast whichever floating operand
  differs, and delegate to a fresh operator through `forward` -- the pattern
  `BinaryOp` already uses for its integer-promotion cases.

## Verification

- Minimal repro `jt.nn.conv3d(float32, float32)` under `amp_reg=prefer16`:
  before the fix it raised at `cudnn_conv3d_op.cc:296 [check failed:
  best_algo_idx!=-1]` with `in: float32, float32 / out: float16`; after the fix
  it completes and returns float16.
- `tests/backends/cuda/test_cudnn_conv_amp_dtype.py` (real CUDA): 2-D and 3-D
  forward under `amp_prefer16` (result equals an explicit float16 convolution),
  3-D backward under `amp_prefer16`, and a mixed-operand pair without amp that
  promotes to float32. 4 passed.
- Normal path: `tests/backends/cuda/test_cudnn_conv3d_algo_cache.py` and the
  `conv3d` case of `tests/backends/cuda/test_cudnn_op.py` pass unchanged.
- End to end, MiniMax-H3 tiny 32x32 t2va with the VAE in float32 under
  autocast (`run-tiny-parity.sh`): `grep -c best_algo_idx` is 0, where it was 9
  on every prior run.

## Known boundary (separate defect, not covered here)

That tiny run still fails later, in the video VAE's attention: the shim's
`nn.Linear` is `matmul_transpose(x, weight) + bias`, and the shim's torch-style
binary promotion lifts the float16 matmul result back to float32 when the bias
parameter is float32 (`f16 + f32 -> f32` in the shim, where native jittor under
the same register gives float16). The RoPE multiplies then differ: q/k end up
float16 while v stays float32, and `scaled_dot_product_attention` rejects the
mixed triple. torch's autocast casts the bias as part of the linear operator, so
the situation does not arise there. This is a fidelity gap in the shim's
autocast, not in the convolution ops, and needs its own fix.

Raw logs and the tiny run live under `$JITTOR_LAB_ROOT/_state/h3/` and are
unversioned.
