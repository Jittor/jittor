#pragma once

#include "core/var.h"
#include <miopen/miopen.h>
#include <functional>

namespace jittor {

// Share the stream-bound per-device handle with other native MIOpen ops.
void with_miopen_handle(int device, const std::function<void(miopenHandle_t)>& launch);

enum class MiopenDirection : int { Forward, BackwardData, BackwardWeights };

// Query all three directions before admitting a differentiable convolution.
// No tensor data is needed; the descriptors and solutions are cached.
bool miopen_training_available(const NanoVector& x, const NanoVector& w,
    const NanoVector& y, NanoString dtype, int device,
    const NanoVector& stride, const NanoVector& padding, const NanoVector& dilation);

// x, w and y always describe the original forward convolution. The selected
// direction identifies which of the three buffers is the output.
bool try_miopen_conv(Var* x, Var* weight, Var* y, MiopenDirection direction,
    const NanoVector& stride, const NanoVector& padding, const NanoVector& dilation);

bool try_miopen_conv_inference(Var* x, Var* weight, Var* y,
    const NanoVector& stride, const NanoVector& padding, const NanoVector& dilation);

} // namespace jittor
