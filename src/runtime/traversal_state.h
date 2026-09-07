#pragma once
#include "common.h"

namespace jittor {

struct TraversalEpoch;

class RuntimeTraversalState {
public:
    RuntimeTraversalState() = default;
    RuntimeTraversalState(const RuntimeTraversalState&) = delete;
    RuntimeTraversalState& operator=(const RuntimeTraversalState&) = delete;

    int64 stamp_count() const { return stamp_count_; }
    int active_epochs() const { return active_epochs_; }

private:
    friend struct TraversalEpoch;
    int64 enter() {
        ++active_epochs_;
        return ++stamp_count_;
    }
    void leave() { --active_epochs_; }

    int64 stamp_count_ = 0;
    int active_epochs_ = 0;
};

EXTERN_LIB RuntimeTraversalState& runtime_traversal_state();

} // namespace jittor
