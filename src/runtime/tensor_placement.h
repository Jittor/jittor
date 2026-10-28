#pragma once
#include <atomic>
#include "runtime/backend.h"

namespace jittor {

// Whether any explicit placement has been made in this process. Every one is
// made by the explicit constructor below, and a default-constructed placement
// is not one, so while this is false no Var can carry one -- which is what
// lets `Op::graph_placement` answer without walking an op's outputs. Sticky:
// a placement made once may be copied onto any Var later.
EXTERN_LIB std::atomic<bool> explicit_tensor_placement_made;

// A graph constraint, not a statement about the current allocation. Native
// Vars default to following the Runtime; explicit frontend Vars retain their
// backend even when an allocation is staged on the host or Runtime changes.
struct TensorPlacement {
    bool explicit_backend = false;
    Device device;

    TensorPlacement() = default;
    explicit TensorPlacement(Device target) : explicit_backend(true), device(target) {
        if (!explicit_tensor_placement_made.load(std::memory_order_relaxed))
            explicit_tensor_placement_made.store(true, std::memory_order_relaxed);
    }
    bool operator==(const TensorPlacement& other) const {
        return explicit_backend == other.explicit_backend &&
            (!explicit_backend || (device.backend == other.device.backend &&
                (device.backend == BackendId::Cpu || device.index == other.device.index)));
    }
    bool operator!=(const TensorPlacement& other) const { return !(*this == other); }
};

EXTERN_LIB TensorPlacement current_tensor_placement();
EXTERN_LIB void set_tensor_placement(TensorPlacement placement);

// Used only while constructing a native graph. No flag setter, kernel launch,
// allocator change or synchronization occurs at this boundary.
struct TensorPlacementScope {
    TensorPlacement previous;
    explicit TensorPlacementScope(TensorPlacement placement)
        : previous(current_tensor_placement()) { set_tensor_placement(placement); }
    ~TensorPlacementScope() { set_tensor_placement(previous); }
    TensorPlacementScope(const TensorPlacementScope&) = delete;
    TensorPlacementScope& operator=(const TensorPlacementScope&) = delete;
};

} // namespace jittor
