#pragma once
#include <atomic>

#include "core/common.h"

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
    // Atomic, and not only to avoid a torn read: a stamp is an identity. Two
    // traversals that run concurrently -- the compile workers reach this through
    // `add_relay_group` -> `bfs_backward` -- must not be handed the same number,
    // because `TraversalEpoch::mark` reads a matching stamp as "this node is
    // already mine" and the walk then skips or re-releases a node the other
    // traversal owns. `fetch_add` makes every stamp unique; the counter is
    // otherwise just a monotone tag.
    int64 enter() {
        active_epochs_.fetch_add(1, std::memory_order_relaxed);
        return stamp_count_.fetch_add(1, std::memory_order_relaxed) + 1;
    }
    void leave() { active_epochs_.fetch_sub(1, std::memory_order_relaxed); }

    std::atomic<int64> stamp_count_{0};
    std::atomic<int> active_epochs_{0};
};

EXTERN_LIB RuntimeTraversalState& runtime_traversal_state();

} // namespace jittor
