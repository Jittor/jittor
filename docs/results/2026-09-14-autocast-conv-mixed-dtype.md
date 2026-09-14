# Auto-mixed-precision requested mixed-dtype convolutions

- Status: Fixed; verified on a real CUDA device
- Date: 2026-09-14
- Baseline commit: `4c3ab0e4` (the two fix commits, on top of `5a084737`)
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
  autocast (`run-tiny-parity.sh`): `grep -c best_algo_idx` is 0 (was 9 on every
  prior run), `video decode failed` is 0 (was 1), and the run writes a real
  124-frame video decoded by the VAE instead of the placeholder.

## The same root in matmul and linear (fixed in `4c3ab0e4`)

With the convolutions fixed the tiny run progressed past the convolution and
then failed in the video VAE's attention, `query, key and value must have the
same dtype`. That is the same root -- the amp register applied inconsistently
across operators -- in two more places:

- **The CUDA cuBLAS matmul ignored the register.** `cublas_matmul` and
  `cublas_batched_matmul` set their output dtype from the operands, so a float32
  product under `amp_prefer16` stayed float32, while the generic path the CPU
  and mixed-dtype cases use returned float16. One model, two precisions,
  depending on the backend. Both now take their dtype from `dtype_infer` and
  cast their operands, mirroring the convolution ops.
- **The decomposed `linear` let its bias undo the dtype.** `linear` is
  `matmul_transpose(x, weight) + bias`. The matmul returns the compute dtype,
  but the shim's torch-parity promotion lifts the sum back to float32 when the
  bias is a float32 Parameter (`f16 + f32 -> f32` in the shim, where native
  jittor under the same register gives float16). torch's autocast casts the bias
  with the rest of the linear operator. `linear` (and `nn.Linear.execute`, which
  used to transcribe it) now casts a wider bias to the product's dtype while the
  register is active. The RoPE multiplies then keep q, k and v in one dtype and
  the attention check passes.

Verification:

- `tests/backends/cuda/test_cublas_matmul_amp_dtype.py` (real CUDA): float32
  matmul and bmm under `amp_prefer16` are float16, and float32 without amp stays
  float32. 3 passed.
- Shim `torch.nn.functional.linear` and `torch.nn.Linear` with a float16 input,
  float32 weight and float32 bias under `torch.autocast(float16)` return
  float16. The case is added to `TestAutocast` in
  `compat/tests/torch/test_torch_compat_unimplemented.py`.
- Regression: `test_cublas_matmul_grad`, `test_fp16`, `test_bf16` and
  `test_cudnn_conv_amp_dtype` -- 74 passed, 2 xfailed.
- The tiny run's video decode now completes (`video decode failed` gone), which
  the first commit alone did not achieve.

## Remaining, unrelated

The tiny run still exits 134 with `corrupted double-linked list` after every
artifact is written. That is the exit-time heap corruption tracked as the
separate open issue, not an autocast defect.

Raw logs and the tiny run live under `$JITTOR_LAB_ROOT/_state/h3/` and are
unversioned.
