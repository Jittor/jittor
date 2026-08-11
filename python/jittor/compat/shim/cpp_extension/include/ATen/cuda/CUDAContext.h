#pragma once
// ATen/cuda/CUDAContext.h — only ever included by .cu TUs (nvcc), so it MAY pull
// the CUDA stream shim (which includes <cuda_runtime.h>). The public
// torch/extension.h must stay CUDA-free, so the stream API lives here, not there.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
