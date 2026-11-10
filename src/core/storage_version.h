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
    // Raw pointer consumers are outside graph dependency tracking. Once an
    // address escapes, storage-preserving writes must keep their eager store.
    std::atomic<bool> externally_visible{false};
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
// Re-snapshot the saved-version entries that name `value` after a lazy
// graph optimization adopts a different storage token for it. The
// adoption is part of producing the value, not a user in-place write; a
// write after this point still bumps the adopted token and is caught.
void rebase_saved_storage_versions(Var* value);
} // namespace jittor
