#pragma once
#include <functional>
#include "core/common.h"
#include "runtime/backend.h"

namespace jittor {
struct Var;

// The executor has allocated these buffers and ordered their producers before
// this stream. A callback may enqueue work, but must not reenter the executor.
struct KernelLaunch {
    vector<Var*> inputs, outputs;
    Device device;
    int64 stream;
};

struct KernelFunc {
    std::function<void(const KernelLaunch&)> callback;
    std::function<void()> deleter;
    KernelFunc(std::function<void(const KernelLaunch&)> callback,
               std::function<void()> deleter)
        : callback(move(callback)), deleter(move(deleter)) {}
    KernelFunc(KernelFunc&& other)
        : callback(move(other.callback)), deleter(move(other.deleter)) {
        other.deleter = nullptr;
    }
    KernelFunc(const KernelFunc&) = delete;
    ~KernelFunc() { if (deleter) deleter(); }
};
} // namespace jittor
