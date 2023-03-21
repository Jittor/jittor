#pragma once
#include "extension.h"
#include "type/fp16_compute.h"
#include <cuda_fp16.h>

namespace jittor {
    namespace at {
        using Half = __half;
    }
}