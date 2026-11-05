#pragma once
#include <atomic>
#include "core/common.h"

namespace jittor {
struct Var;
struct Op;

// Shared before allocation, so a pending detach/view observes writes through
// its source too. This owns no Var, holder, graph or Python object.
struct StorageMutationVersion {
    uint64 identity;
    std::atomic<uint64> value{0};
    StorageMutationVersion();
};

struct SavedStorageVersion {
    shared_ptr<StorageMutationVersion> token;
    uint64 version;
    int64 original_id;
};

// A native gradient checks a saved value only when its derivative actually
// reads that value. Merely having been a forward input is insufficient: the
// gradient of x+1 must remain valid after x is edited.
struct BackwardStorageReadScope {
    const Op* op;
    const BackwardStorageReadScope* previous;
    explicit BackwardStorageReadScope(const Op* op);
    ~BackwardStorageReadScope();
};
void check_backward_storage_read(Var* value);
} // namespace jittor
