#pragma once

#include "core/var.h"

namespace jittor {

// Returns false when no bounded native MIOpen solution is available.
// The caller then uses its existing same-device convolution implementation.
bool try_miopen_conv_inference(Var* x, Var* weight, Var* y,
    const NanoVector& stride, const NanoVector& padding, const NanoVector& dilation);

} // namespace jittor
