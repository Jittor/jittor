# Convolution Plans on the cuDNN Backend API

Once the device is kept busy, a step is bounded by the CPU work per
operator. For convolution-heavy models that work was dominated by cuDNN's
legacy API itself: on cuDNN 8.9 `cudnnGetConvolution*WorkspaceSize` plus
`cudnnConvolution*` cost about 110 µs of CPU per call (measured with a
standalone program: 59 + 49 µs, descriptors 3 µs), and a diffusers UNet2D
step made about 150 such calls, 20 ms of its 42 ms of CPU time.

`extern/cuda/cudnn/inc/cudnn_conv_plan.h` executes the three 2-D
convolution ops through the backend (graph) API with a plan cache keyed by
kind, shapes, strides, dtypes, parameters and numerics. A plan is built once
per key (heuristics mode B, then A, then the fallback list; engines needing
runtime compilation are skipped, and for fp32 operands without tensor-op
permission so are engines that use tensor cores or down-convert inputs) and
executed for about 12 µs. With `cudnn_benchmark` set, up to six candidate
plans are timed on the real buffers, as the legacy `cudnnFind` path does.
Anything the backend declines is remembered and served by the legacy path.

The UNet2D step went from 1.48× to 1.10× of PyTorch. The per-key build
costs a few milliseconds, paid once per shape; a workload whose shapes never
repeat pays it on every call, which is why the cache is bounded and the
legacy path remains.
