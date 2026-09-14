#include "acl_foreach_coefficients.h"
#include "acl_runtime.h"

#include <cstdio>
#include <map>

namespace jittor {
namespace {

// Uploading a coefficient has two ordering constraints and they pull opposite
// ways. The write has to be stream-ordered, or a kernel still queued behind us
// reads the next launch's value instead of its own; that rules out the
// blocking `aclrtMemcpy`, which runs on its own stream. And the host bytes an
// async copy reads have to stay put until the copy engine reaches them, which
// rules out reusing one host slot per launch.
//
// A ring settles both. Within one lap no slot is written twice, so no pending
// copy can have its source overwritten; the lap that wraps drains the stream
// first, which retires every copy issued in the lap before it. With this many
// slots and a handful of coefficients per optimizer step, that drain lands
// roughly once per hundred steps on a stream the executor is already
// synchronising for its own reasons.
const int coefficient_slots = 512;

struct CoefficientRing {
    void* device = nullptr;
    float* host = nullptr;
    aclrtStream stream = nullptr;
    // The first reservation wraps. Draining a stream that has never carried one
    // of these copies costs nothing and keeps the wrap path on one branch.
    int cursor = coefficient_slots;

    void* slot(int index) const {
        return static_cast<char*>(device) + size_t(index) * sizeof(float);
    }
};

std::map<int, CoefficientRing>*& ring_state() {
    // The backend drains these before it destroys CANN's resources.
    static std::map<int, CoefficientRing>* state = nullptr;
    return state;
}

std::map<int, CoefficientRing>& rings() {
    auto*& state = ring_state();
    if (!state)
        state = new std::map<int, CoefficientRing>();
    return *state;
}

void drain(aclrtStream stream) {
    if (!stream)
        return;
    const auto status = aclrtSynchronizeStream(stream);
    if (status != ACL_SUCCESS)
        LOGf << "ACL foreach coefficient drain failed, return code" << status;
}

void release_ring(CoefficientRing& ring) {
    const CoefficientRing previous = ring;
    ring = CoefficientRing();
    if (previous.device) {
        drain(previous.stream);
        const auto status = aclrtFree(previous.device);
        if (status != ACL_SUCCESS)
            LOGf << "ACL foreach coefficient device release failed, return code" << status;
    }
    if (previous.host) {
        const auto status = aclrtFreeHost(previous.host);
        if (status != ACL_SUCCESS)
            LOGf << "ACL foreach coefficient host release failed, return code" << status;
    }
}

void allocate_ring(CoefficientRing& ring) {
    const size_t bytes = size_t(coefficient_slots) * sizeof(float);
    void* device = nullptr;
    auto status = aclrtMalloc(&device, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (status != ACL_SUCCESS)
        LOGf << "ACL foreach coefficient device allocation failed, return code" << status;
    void* host = nullptr;
    status = aclrtMallocHost(&host, bytes);
    if (status != ACL_SUCCESS) {
        aclrtFree(device);
        LOGf << "ACL foreach coefficient host allocation failed, return code" << status;
    }
    ring.device = device;
    ring.host = static_cast<float*>(host);
}

} // namespace

void acl_stage_foreach_coefficients(const float* values, int count,
                                    aclTensor** tensors) {
    if (count <= 0)
        return;
    if (count > coefficient_slots)
        LOGf << "ACL foreach coefficient staging asked for" << count
             << "coefficients, the ring holds" << coefficient_slots;
    auto& ring = rings()[acl_runtime_current_device()];
    if (!ring.device)
        allocate_ring(ring);
    const aclrtStream stream = acl_current_stream();
    if (ring.stream != stream) {
        // A different stream cannot be ordered against the copies already in
        // flight on the old one, so retire those before reusing any slot.
        drain(ring.stream);
        ring.stream = stream;
        ring.cursor = coefficient_slots;
    }
    if (ring.cursor + count > coefficient_slots) {
        drain(stream);
        ring.cursor = 0;
    }
    const int base = ring.cursor;
    ring.cursor += count;
    for (int index = 0; index < count; ++index)
        ring.host[base + index] = values[index];
    const size_t bytes = size_t(count) * sizeof(float);
    const auto status = aclrtMemcpyAsync(ring.slot(base), bytes,
                                         ring.host + base, bytes,
                                         ACL_MEMCPY_HOST_TO_DEVICE, stream);
    if (status != ACL_SUCCESS)
        LOGf << "ACL foreach coefficient upload failed, return code" << status;
    static const int64_t one = 1;
    for (int index = 0; index < count; ++index) {
        tensors[index] = aclCreateTensor(
            &one, 1, ACL_FLOAT, &one, 0, ACL_FORMAT_ND, &one, 1,
            ring.slot(base + index));
        if (!tensors[index])
            LOGf << "ACL foreach coefficient tensor creation failed at index" << index;
    }
}

void release_all_acl_foreach_coefficients() noexcept {
    if (!ring_state())
        return;
    int32_t previous_device = 0;
    const auto previous_status = aclrtGetDevice(&previous_device);
    if (previous_status != ACL_SUCCESS)
        std::fprintf(stderr,
                     "ACL foreach coefficient shutdown: cannot query current device (%d)\n",
                     static_cast<int>(previous_status));
    for (auto& entry : rings()) {
        if (!entry.second.device && !entry.second.host)
            continue;
        const auto selected = aclrtSetDevice(entry.first);
        if (selected != ACL_SUCCESS) {
            std::fprintf(stderr,
                         "ACL foreach coefficient shutdown: cannot select device %d (%d)\n",
                         entry.first, static_cast<int>(selected));
            continue;
        }
        try {
            release_ring(entry.second);
        } catch (const std::exception& error) {
            std::fprintf(stderr,
                         "ACL foreach coefficient release failed on device %d: %s\n",
                         entry.first, error.what());
        } catch (...) {
            std::fprintf(stderr,
                         "ACL foreach coefficient release failed on device %d\n",
                         entry.first);
        }
    }
    if (previous_status == ACL_SUCCESS) {
        const auto status = aclrtSetDevice(previous_device);
        if (status != ACL_SUCCESS)
            std::fprintf(stderr,
                         "ACL foreach coefficient shutdown: cannot restore device %d (%d)\n",
                         previous_device, static_cast<int>(status));
    }
}

} // namespace jittor
