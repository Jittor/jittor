#include "runtime/tensor_placement.h"

namespace jittor {
namespace {
thread_local TensorPlacement construction_placement;
}

TensorPlacement current_tensor_placement() { return construction_placement; }
void set_tensor_placement(TensorPlacement placement) { construction_placement = placement; }
} // namespace jittor
