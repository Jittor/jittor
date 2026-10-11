#pragma once
#include "core/common.h"
#include <map>
#include <mutex>

namespace jittor {
struct Allocator;
struct RuntimePoolBytes {
    int device;
    int64 live = 0, reserved = 0;
    bool supported;
    RuntimePoolBytes(int device, bool supported) : device(device), supported(supported) {}
};
struct RuntimeDevicePoolBytes {
    int64 live = 0, reserved = 0, peak_live = 0, peak_reserved = 0;
    bool unsupported = false;
};
// NativeRuntime owns this ledger. All events/query/reset serialize under mutex;
// callers may hold an allocator lock, but ledger readers never take pool locks.
struct RuntimeMemoryState {
    std::mutex mutex;
    std::map<const Allocator*, RuntimePoolBytes> pools;
    std::map<int, RuntimeDevicePoolBytes> devices;
};
} // namespace jittor
