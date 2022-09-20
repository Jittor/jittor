#pragma once
#include "var.h"
#include "var_holder.h"
#include "executor.h"
#include "ops/getitem_op.h"
#include "ops/op_register.h"
#include "pyjt/py_converter.h"
#include "misc/cuda_flags.h"
#include "type/fp16_compute.h"
#include <cuda_runtime.h>
#include "helper_cuda.h"

namespace c10 {
    using Half = jittor::float16; 
}