# 7.19 / 7.20 frontend precision isolation

## Delivered behavior

Native Jittor retains one public Runtime precision setting which updates both
native matmul and cuDNN tier fields. Independent Torch owns its own matmul
and cuDNN fields; changing either does not change the other field or native
Runtime settings. Defaults are native highest/highest and Torch highest/high.

An explicit frontend scope carries the pair in thread-local native state.
Each Op captures it, fusion checks equality, and graph rewriting, compilation,
execution and native gradient construction restore the captured pair. This
preserves lazy graph precision after later setting changes. Native operations
without an explicit frontend continue following Runtime policy. No temporary
global flags or forced tensor synchronization implement the boundary.

RNN descriptors use the shared cuDNN precision rule. Reserve-space cache keys
include math type and the flattened-weight offset cache includes the cuDNN
tier. The real Torch RNN gradient test also exposed a stopped zero buffer in
the native weight-flattening owner: indexed assignment preserved that stopped
destination and disconnected the parameter-copy graph. The common owner now
starts that buffer's gradient when parameters require gradients, retaining the
same packing and recurrent mathematics.

The native private binding pair `_set_float32_precision` /
`_reset_float32_precision` is used by the existing synchronous
`tensor_frontend` context, including no-tensor-argument descriptor queries.
Generated Tensor binding scopes also infer the pair from Tensor ownership.
The public policy interface remains the existing native Runtime setting and
Torch backend/set-precision methods.

## Focused validation

Source: `refactor-tensor-placement-integrated-20260908`, base `86c93`, with this
batch overlaid. The existing source path and JIT cache were reused. Interpreter
was the `compat-dist-OIR3DI` venv; CUDA 12.2.140, visible physical GPUs 6/7,
architectures `[89, 89]`. The tests execute on the default visible CUDA device;
this is not a two-device communication test.

Command: `python -m pytest --noconftest -q -s
compat/tests/torch/test_precision_policy_isolation.py
compat/tests/torch/test_torch_backends_tf32.py`.

Final result: **17 passed, 0 skipped, 104.17 seconds**, including binding and
library incremental compilation after the first-run fixes. Log:
`_state/tensor-placement-integrated-20260908/precision-policy-fixed.log`.

The four new nodes cover:

- Native combined setter, independent Torch domains, nested scope and exception
  restoration, without native-policy mutation.
- Pending strict/medium Torch matmul and strict Torch convolution evaluated
  after state changes alongside native high matmul/convolution. Library logs
  show all three GEMM compute tiers and both convolution math modes; output
  values are compared with NumPy references.
- Independent float64 NumPy LSTM forward/BPTT, with finite-difference controls
  for input and all four parameter families.
- CUDA LSTM sequence length 5, batch 4, input 32, hidden 64, fixed weights and
  loss coefficients. Output and gradients for input, input/hidden weights and
  both biases are compared to that CPU oracle. Largest normalized max error:
  native highest **2.24003e-7**, Torch cuDNN false **2.24003e-7**, Torch cuDNN
  true **5.36036e-4**. Descriptor logs prove FMA versus TF32-allowed selection.

The first run was **2 passed / 2 failed** and is retained in
`precision-policy.log`: the explicit Python scope lacked precision TLS for
no-tensor calls, and the independent RNN weight buffer lost gradients. Both
were fixed before the final run. Ten changed core C++ translation units also
passed host syntax checks before the actual build; those are not substituted
for the CUDA results.

No full-suite, model-performance, NPU/CANN or multi-machine claim is made.
The inherited vLLM optional-entry-point warning in this validation base is
outside this batch. Real PyTorch 2.12.1 confirms the highest/True defaults;
its mixed old/new precision-control APIs can reject boolean roundtrips. The
shim retains its existing high/medium roundtrip approximation and does not
claim exact compatibility with that rejection policy.

## Grouped-gradient review follow-up

Integration review found the separate `op->grads(douts, dins)` path lacked
the forward precision scope present on `op->grad(...)`. Both entrances now
restore the forward Op's policy. The additional real CUDA node
`test_grouped_rnn_backward_retains_forward_precision_after_policy_switch`
constructs an FMA LSTM, enables Torch cuDNN TF32 before backward, then checks
all forward/backward descriptors remain FMA and all oracle errors stay below
1e-6. The current public cuDNN switch remains enabled after backward.

Before the one-line grouped-branch fix: **1 failed / 2.35 seconds**, reporting
three highest/FMA descriptors followed by a high/TF32-allowed backward
descriptor. After: **1 passed / 11.88 seconds**, including the single core-TU
incremental compile. Logs are `precision-grouped-before.log` and
`precision-grouped-after.log` in the same state directory. Together with the
previous concentrated run, **18 distinct nodes passed**; the 17-node run
preceded this final grouped-only correction and was not needlessly repeated.
