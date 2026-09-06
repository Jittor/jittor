#include "acl_workspace.h"
#include "acl_runtime.h"
#include "executor.h"
#include "mem/allocator.h"

#include <cstdio>
#include <limits>
#include <map>

namespace jittor {
namespace {

struct Workspace {
    void* address = nullptr;
    size_t size = 0;
    Allocator* allocator = nullptr;
    size_t allocation = 0;
    aclrtStream stream = nullptr;
};

std::map<int, Workspace>*& workspace_state() {
    // The backend explicitly drains these before destroying CANN resources.
    static std::map<int, Workspace>* state = nullptr;
    return state;
}

std::map<int, Workspace>& workspaces() {
    auto*& state = workspace_state();
    if (!state)
        state = new std::map<int, Workspace>();
    return *state;
}

void release_workspace(Workspace& workspace) {
    if (!workspace.address)
        return;
    auto status = aclrtSynchronizeStream(workspace.stream);
    if (status != ACL_SUCCESS)
        LOGf << "ACL workspace synchronization failed, return code" << status;

    const Workspace previous = workspace;
    workspace = Workspace();
    ASSERT(previous.allocator != nullptr);
    previous.allocator->free(previous.address, previous.size, previous.allocation);
    previous.allocator->gc();
}

} // namespace

void* acl_workspace_address() {
    const auto found = workspaces().find(acl_runtime_current_device());
    return found == workspaces().end() ? nullptr : found->second.address;
}

void releaseWorkSpace() {
    const auto found = workspaces().find(acl_runtime_current_device());
    if (found != workspaces().end())
        release_workspace(found->second);
}

void* mallocWorkSpace(uint64_t size) {
    if (size == 0)
        return nullptr;
    if (size > std::numeric_limits<size_t>::max() - 31)
        LOGf << "ACL workspace allocation failed: workspace requested bytes"
             << size << "overflow alignment";
    const size_t alloc_size = (size + 31) / 32 * 32;
    const int device = acl_runtime_current_device();
    auto& workspace = workspaces()[device];
    if (alloc_size <= workspace.size)
        return workspace.address;

    release_workspace(workspace);
    Allocator* allocator = runtime_executor().temp_allocator;
    if (!allocator || allocator->device() != device)
        allocator = get_allocator(device, true);
    if (!allocator || allocator->device() != device)
        LOGf << "ACL workspace allocation failed: workspace allocator is not"
             << "ready for device" << device << "workspace requested bytes"
             << alloc_size;
    const aclrtStream stream = acl_current_stream();
    size_t allocation = 0;
    void* address = nullptr;
    try {
        address = allocator->alloc(alloc_size, allocation);
    } catch (const std::exception& error) {
        LOGf << "ACL workspace allocation failed: workspace requested bytes"
             << alloc_size << "workspace allocator" << allocator->name()
             << error.what();
    }
    if (!address)
        LOGf << "ACL workspace allocation failed: workspace requested bytes"
             << alloc_size << "workspace allocator" << allocator->name();
    workspace.address = address;
    workspace.size = alloc_size;
    workspace.allocator = allocator;
    workspace.allocation = allocation;
    workspace.stream = stream;
    return address;
}

void release_all_acl_workspaces() noexcept {
    if (!workspace_state())
        return;
    int32_t previous_device = 0;
    const auto previous_status = aclrtGetDevice(&previous_device);
    if (previous_status != ACL_SUCCESS)
        std::fprintf(stderr, "ACL workspace shutdown: cannot query current device (%d)\n",
                     static_cast<int>(previous_status));
    for (auto& entry : workspaces()) {
        if (!entry.second.address)
            continue;
        const auto status = aclrtSetDevice(entry.first);
        if (status != ACL_SUCCESS) {
            std::fprintf(stderr, "ACL workspace shutdown: cannot select device %d (%d)\n",
                         entry.first, static_cast<int>(status));
            continue;
        }
        try {
            release_workspace(entry.second);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "ACL workspace release failed during shutdown on device %d: %s\n",
                         entry.first, error.what());
        } catch (...) {
            std::fprintf(stderr, "ACL workspace release failed during shutdown on device %d: unknown exception\n",
                         entry.first);
        }
    }
    if (previous_status == ACL_SUCCESS) {
        const auto status = aclrtSetDevice(previous_device);
        if (status != ACL_SUCCESS)
            std::fprintf(stderr, "ACL workspace shutdown: cannot restore device %d (%d)\n",
                         previous_device, static_cast<int>(status));
    }
}

} // namespace jittor
