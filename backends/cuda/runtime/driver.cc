#include "runtime/backend.h"
#include "runtime/backend_streams.h"
#include "runtime/device_state.h"
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <exception>

namespace jittor {
DECLARE_FLAG(int, cuda_device_allocator_managed_fallback);
EXTERN_LIB bool no_device_error_when_free;
void cuda_backend_check_nan(Var*, Op*);
namespace {
int accelerator_count() {
    auto& count = runtime_device_state().device_count;
    if (count == -1 && cudaGetDeviceCount(&count) != cudaSuccess) count = 0;
    return count;
}

// The device the calling thread is actually bound to, or -1 when unknown.
//
// CUDA's current device is per-host-thread and starts at 0 on every new thread;
// jittor's is a single value for the whole process
// (RuntimeDeviceState::current_device). A worker thread that has never called
// cudaSetDevice therefore sits on device 0 while the process device is 1, and
// everything that resolves a device through jittor and then issues a call on
// *this thread's* context -- an allocation, a memory query, a copy, an event,
// a library handle -- silently works on device 0 under device-1 bookkeeping.
// That is a cross-device mismatch, and its signature is a Xid 31 MMU fault and
// a context-sticky cudaErrorIllegalAddress that surfaces later, on a call like
// cudaMemGetInfo that only reports that the context is gone. It is invariant
// for rank 1 of a multi-process run and never happens on rank 0, whose threads
// default to the device it uses anyway.
//
// record_event() already works around this by calling cudaSetDevice itself
// ("a thread whose CUDA context is still the process default"). Do it once
// here instead, so "the device jittor reports" and "the device this thread is
// bound to" cannot disagree for anything that goes through this entry point.
//
// Only the first call on a thread (or the first after the process device moves)
// pays for cudaSetDevice; until then `accelerator_current` consults this cache
// with one thread-local read.
static thread_local int tls_bound_device = -1;

int accelerator_current() {
    auto& state = runtime_device_state();
    if (state.current_device < 0) {
        if (accelerator_count() <= 0) return -1;
        int device = 0;
        if (cudaGetDevice(&device) != cudaSuccess) {
            cudaGetLastError();
            return -1;
        }
        state.current_device = state.device_id = device;
        tls_bound_device = device;
        return device;
    }
    if (tls_bound_device != state.current_device) {
        int device = state.current_device;
        if (cudaSetDevice(device) != cudaSuccess) {
            // Leave the cache unset rather than claim a binding that is not
            // there; the next call retries, and the caller still gets the
            // device jittor is configured for.
            cudaGetLastError();
            return device;
        }
        tls_bound_device = device;
        // Set before the hooks: a hook may ask for the current device, and
        // re-entering the branch above would switch device a second time.
        for (auto hook : state.switch_hooks) hook(device);
    }
    return state.current_device;
}

void accelerator_set(int device) {
    int count = accelerator_count();
    CHECK(device >= 0 && device < count)
        << "Invalid CUDA device index" << device >> ", visible device count is" << count;
    int previous = accelerator_current();
    auto& state = runtime_device_state();
    state.device_id = device;
    if (device == previous) return;
    checkCudaErrors(cudaSetDevice(device));
    tls_bound_device = device;
    state.current_device = device;
    for (auto hook : state.switch_hooks) hook(device);
}

void accelerator_sync(uint64 devices) {
    LaunchErrorScope error_scope({BackendId::Cuda, accelerator_current()});
    checkCudaErrors(cudaGetLastError());
    if (!devices) {
        checkCudaErrors(cudaDeviceSynchronize());
        return;
    }
    int previous = accelerator_current();
    try {
        for (int device = 0; device < 64; ++device) {
            if (!(devices & (1ull << device))) continue;
            if (device != accelerator_current()) accelerator_set(device);
            LaunchErrorScope device_error_scope({BackendId::Cuda, device});
            checkCudaErrors(cudaDeviceSynchronize());
        }
    } catch (...) {
        auto failure = std::current_exception();
        try {
            if (previous >= 0 && previous != accelerator_current()) accelerator_set(previous);
        } catch (...) {
            LOGe << "Could not restore device after backend synchronization failed";
        }
        std::rethrow_exception(failure);
    }
    if (previous >= 0 && previous != accelerator_current()) accelerator_set(previous);
}

void accelerator_peer(int from, int to) {
    if (from == to || from < 0 || to < 0) return;
    int count = accelerator_count();
    if (from >= count || to >= count) return;
    auto& enabled = runtime_device_state().peer_enabled;
    if ((int)enabled.size() < count * count) enabled.resize(count * count, 0);
    auto& done = enabled[from * count + to];
    if (done) return;
    done = 1;
    int can = 0;
    if (cudaDeviceCanAccessPeer(&can, to, from) != cudaSuccess || !can) {
        cudaGetLastError();
        return;
    }
    int previous = accelerator_current();
    checkCudaErrors(cudaSetDevice(to));
    auto error = cudaDeviceEnablePeerAccess(from, 0);
    if (error != cudaSuccess && error != cudaErrorPeerAccessAlreadyEnabled)
        LOGw << "cudaDeviceEnablePeerAccess(" >> from << "->" >> to >> ") failed:"
            << cudaGetErrorString(error);
    cudaGetLastError();
    checkCudaErrors(cudaSetDevice(previous));
}

template<class Func>
auto on_device(int device, Func&& func) -> decltype(func()) {
    LaunchErrorScope error_scope({BackendId::Cuda, device});
    int previous = accelerator_current();
    try {
        if (device != previous) accelerator_set(device);
        auto result = func();
        if (previous >= 0 && previous != accelerator_current()) accelerator_set(previous);
        return result;
    } catch (...) {
        auto failure = std::current_exception();
        try {
            if (previous >= 0 && previous != accelerator_current()) accelerator_set(previous);
        } catch (...) {
            LOGe << "Could not restore device after backend operation failed";
        }
        std::rethrow_exception(failure);
    }
}
template<class Func>
void on_device_void(int device, Func&& func) {
    on_device(device, [&] { func(); return 0; });
}

void* raw_allocate(int device, BackendMemoryKind kind, size_t size) {
    if (!size) return nullptr;
    return on_device(device, [&]() -> void* {
        void* ptr = nullptr;
        if (kind == BackendMemoryKind::Pinned) {
            checkCudaErrors(cudaMallocHost(&ptr, size));
        } else if (kind == BackendMemoryKind::Managed) {
            checkCudaErrors(cudaMallocManaged(&ptr, size));
        } else {
            auto error = cudaMalloc(&ptr, size);
            if (error != cudaSuccess) {
                cudaGetLastError();
                if (!cuda_device_allocator_managed_fallback) {
                    // "cudaMalloc failed" on its own tells the reader nothing
                    // they can act on: not how much was asked for, not how much
                    // the device had, not which device. The commonest cause is
                    // another process holding the card, and that is exactly the
                    // case the bare message cannot distinguish from a model that
                    // is too large. The caching allocators above have already
                    // released their cached blocks and retried by the time this
                    // is thrown, so these numbers are the real ones.
                    size_t device_free = 0, device_total = 0;
                    auto info = cudaMemGetInfo(&device_free, &device_total);
                    if (info != cudaSuccess) cudaGetLastError();
                    string where = info == cudaSuccess
                        ? (", device " + S(device) + " has " + S(device_free >> 20) +
                           " MiB free of " + S(device_total >> 20) + " MiB")
                        : (", and device " + S(device) + "'s free memory could not be read");
                    throw std::runtime_error(
                        "out of memory on the accelerator: could not allocate " +
                        S(size >> 20) + " MiB (" + S(size) + " bytes)" + where +
                        ". CUDA said: " + cudaGetErrorString(error) +
                        ". Another process may be holding the device; reduce the "
                        "batch size, or set auto_flush_ops=0 to let the whole "
                        "graph be scheduled at once, which frees intermediates "
                        "earlier.");
                }
                LOGw << "Unable to alloc cuda device memory for size" << size
                     << ", falling back to cudaMallocManaged";
                checkCudaErrors(cudaMallocManaged(&ptr, size));
            }
        }
        return ptr;
    });
}

void raw_free(int device, BackendMemoryKind kind, void* ptr) {
    if (!ptr || no_device_error_when_free) return;
    on_device_void(device, [&] {
        if (kind == BackendMemoryKind::Pinned) checkCudaErrors(cudaFreeHost(ptr));
        else checkCudaErrors(cudaFree(ptr));
    });
}
void memory_info(int device, size_t& free, size_t& total) {
    on_device_void(device, [&] { checkCudaErrors(cudaMemGetInfo(&free, &total)); });
}
void check_error() { checkCudaErrors(cudaGetLastError()); }

// The one stream every jittor CUDA launch, copy and library call goes on.
//
// It is `cudaStreamPerThread`, not the legacy default stream (0), for exactly
// one reason: **the legacy stream cannot be captured into a CUDA graph.**
// Capturing a repeated step is what turns its ~190 individual kernel launches
// (7.9 us of host time each, against 2.6 us for the launch itself) into a
// single `cudaGraphLaunch`; measured on this machine, 200 launches go from
// 346 us to 2.1 us.
//
// Everything has to agree on this stream, because `cudaStreamPerThread` and
// the legacy stream do NOT synchronise with each other -- a straggler left on
// the legacy stream is an unordered race that raises no error and produces no
// message. The three things that make them agree:
//   - jittor's own kernels: `--default-stream per-thread` in the nvcc flags,
//     which maps a bare `<<<>>>` (all 104 of them, generated and hand-written)
//     onto this stream;
//   - jittor's own copies and events: this function, plus the few sites that
//     spell a stream out;
//   - every library handle: `cublasSetStream`/`cudnnSetStream`/... at creation.
// Grep for `cudaStreamPerThread` to audit the set.
//
// The same argument applies to two *threads*, which this stream makes into two
// streams while the graph, the Vars and their buffers stay process-global and
// carry no stream affinity. That one is not fixed here: it is ordered at the
// two ends of `run_exec_plan` by `backend_compute_stream_acquire`/`_release`
// (src/runtime/backend_streams.cc), which is what keeps a graph built on one
// Python thread and finished on another from reading buffers the first thread
// is still writing.
void* compute_stream(int device) {
    CHECK(device >= 0 && device < accelerator_count()) << "Invalid compute stream device";
    return reinterpret_cast<void*>(cudaStreamPerThread);
}
// -- graph capture ---------------------------------------------------------
// Record everything issued on the compute stream instead of running it, then
// hand back one executable graph that re-issues the lot.  This is worth doing
// because jittor pays about 4 us of host time per operator on top of the
// launch, and a captured graph pays it once for the whole recording: measured
// on this machine, 200 launches go from 346 us to 2.1 us.
//
// Two things make a capture fail, and both are the caller's to avoid:
//   - work on a stream that is not being captured (the legacy default stream
//     above all, which is why `compute_stream` is `cudaStreamPerThread`);
//   - anything that has to talk to the driver synchronously -- an allocation,
//     a readback, an event query.  The replay path only ever re-runs a graph
//     whose buffers are already allocated, so it has none of these.
// A failed capture is not an error here: `capture_end` returns nullptr and the
// caller keeps launching one kernel at a time, which is what it did before.
// Is the compute stream currently being recorded? Anything that needs a
// synchronous answer from the driver has to take a different path while it is.
static inline bool capturing() {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(cudaStreamPerThread, &status) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return status != cudaStreamCaptureStatusNone;
}

bool graph_capture_begin(int device) {
    return on_device(device, [&]() -> bool {
        cudaGetLastError();
        const auto status = cudaStreamBeginCapture(cudaStreamPerThread,
                                                   cudaStreamCaptureModeThreadLocal);
        if (status != cudaSuccess) {
            cudaGetLastError();
            LOGvv << "graph capture could not start:" << cudaGetErrorString(status);
            return false;
        }
        return true;
    });
}

void* graph_capture_end(int device) {
    return on_device(device, [&]() -> void* {
        cudaGraph_t graph = nullptr;
        auto status = cudaStreamEndCapture(cudaStreamPerThread, &graph);
        if (status != cudaSuccess || !graph) {
            cudaGetLastError();
            LOGvv << "graph capture did not close:" << cudaGetErrorString(status);
            return nullptr;
        }
        size_t nodes = 0;
        cudaGraphGetNodes(graph, nullptr, &nodes);
        if (!nodes) {
            // An empty recording instantiates happily and then does nothing,
            // which as a replay is a silently frozen answer. Refuse it.
            cudaGraphDestroy(graph);
            LOGvv << "graph capture recorded no work";
            return nullptr;
        }
        LOGvv << "graph captured" << nodes << "nodes";
        cudaGraphExec_t exec = nullptr;
        status = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
        cudaGraphDestroy(graph);
        if (status != cudaSuccess || !exec) {
            cudaGetLastError();
            LOGvv << "graph would not instantiate:" << cudaGetErrorString(status);
            return nullptr;
        }
        return reinterpret_cast<void*>(exec);
    });
}

void graph_launch(void* graph, int device) {
    on_device_void(device, [&] {
        LaunchErrorScope error_scope(
            {accelerator_backend_id(), device}, true,
            reinterpret_cast<uintptr_t>(cudaStreamPerThread));
        checkCudaErrors(cudaGraphLaunch(
            reinterpret_cast<cudaGraphExec_t>(graph), cudaStreamPerThread));
    });
}

void graph_release(void* graph, int device) {
    on_device_void(device, [&] {
        const auto status = cudaGraphExecDestroy(
            reinterpret_cast<cudaGraphExec_t>(graph));
        if (status != cudaSuccess)
            LOGe << "graph release failed:" << cudaGetErrorString(status);
    });
}

void* create_stream(int device, bool nonblocking) {
    return on_device(device, [&]() -> void* {
        cudaStream_t stream;
        checkCudaErrors(cudaStreamCreateWithFlags(&stream,
            nonblocking ? cudaStreamNonBlocking : cudaStreamDefault));
        return reinterpret_cast<void*>(stream);
    });
}
void destroy_stream(BackendStream stream) {
    on_device_void(stream.device.index, [&] {
        checkCudaErrors(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream.handle)));
    });
}
void synchronize_stream(BackendStream stream) {
    on_device_void(stream.device.index, [&] {
        LaunchErrorScope error_scope(stream.device, true, reinterpret_cast<uintptr_t>(stream.handle));
        checkCudaErrors(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream.handle)));
    });
}
void* create_event(int device, bool timing) {
    return on_device(device, [&]() -> void* {
        cudaEvent_t event;
        checkCudaErrors(cudaEventCreateWithFlags(&event, timing ? cudaEventDefault : cudaEventDisableTiming));
        return reinterpret_cast<void*>(event);
    });
}
void destroy_event(BackendEvent event) {
    on_device_void(event.device.index, [&] {
        checkCudaErrors(cudaEventDestroy(reinterpret_cast<cudaEvent_t>(event.handle)));
    });
}
void record_event(BackendEvent event, BackendStream stream) {
    CHECK(event.device.index == stream.device.index) << "Event and recording stream must share a device";
    // Two things this used to get wrong on a device that is not the process
    // default, both observed as cudaErrorInvalidResourceHandle(400) from
    // cudaEventRecord inside a fused operator (rank 1 of a TP=2 run, and never
    // rank 0):
    //
    //  - a caller that leaves the stream unset passes a null handle, which
    //    reaches CUDA as the *legacy default stream*. This file avoids that
    //    stream on purpose (see compute_stream: it is what breaks graph
    //    capture), so resolve it to this device's compute stream.
    //  - `on_device_void` only drives jittor's own device switch (ops.set_device
    //    = accelerator_set). A thread whose CUDA context is still the process
    //    default then records a device-1 event on a device-0 context, which CUDA
    //    rejects while the device *indices* jittor compares still agree.
    //    nccl_init calls both; do the same here.
    BackendStream target = stream;
    if (!target.handle)
        target.handle = accelerator_backend_stream(
            target.device.index, BackendStreamKind::Compute);
    on_device_void(target.device.index, [&] {
        checkCudaErrors(cudaSetDevice(target.device.index));
        LaunchErrorScope error_scope(target.device, true, reinterpret_cast<uintptr_t>(target.handle));
        checkCudaErrors(cudaEventRecord(reinterpret_cast<cudaEvent_t>(event.handle),
                                      reinterpret_cast<cudaStream_t>(target.handle)));
    });
}
void synchronize_event(BackendEvent event) {
    on_device_void(event.device.index, [&] {
        checkCudaErrors(cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(event.handle)));
    });
}
float elapsed_event(BackendEvent start, BackendEvent end) {
    return on_device(start.device.index, [&] {
        float milliseconds = 0;
        checkCudaErrors(cudaEventElapsedTime(&milliseconds, reinterpret_cast<cudaEvent_t>(start.handle),
                                            reinterpret_cast<cudaEvent_t>(end.handle)));
        return milliseconds;
    });
}
void wait_event(BackendStream stream, BackendEvent event) {
    on_device_void(stream.device.index, [&] {
        LaunchErrorScope error_scope(stream.device, true, reinterpret_cast<uintptr_t>(stream.handle));
        checkCudaErrors(cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(stream.handle),
                                          reinterpret_cast<cudaEvent_t>(event.handle), 0));
    });
}
void host_callback(BackendStream stream, void (*callback)(void*), void* context) {
    on_device_void(stream.device.index, [&] {
        LaunchErrorScope error_scope(stream.device, true, reinterpret_cast<uintptr_t>(stream.handle));
        checkCudaErrors(cudaLaunchHostFunc(reinterpret_cast<cudaStream_t>(stream.handle), callback, context));
    });
}
vector<int> architectures() {
    vector<int> result;
    for (int device = 0; device < accelerator_count(); ++device) {
        cudaDeviceProp properties;
        checkCudaErrors(cudaGetDeviceProperties(&properties, device));
        result.push_back(properties.major * 10 + properties.minor);
    }
    return result;
}

auto copy_kind(Device dst, Device src) {
    if (src.backend == BackendId::Cpu) return cudaMemcpyHostToDevice;
    if (dst.backend == BackendId::Cpu) return cudaMemcpyDeviceToHost;
    if (src.index != dst.index) return cudaMemcpyDefault;
    return cudaMemcpyDeviceToDevice;
}
void copy_async(void* dst, Device target, const void* src, Device source, size_t size, BackendStream stream) {
    if (!size) return;
    on_device_void(stream.device.index, [&] {
        LaunchErrorScope error_scope(stream.device, true, reinterpret_cast<uintptr_t>(stream.handle));
        record_active_launch(stream);
        checkCudaErrors(cudaMemcpyAsync(dst, src, size, copy_kind(target, source),
                                       reinterpret_cast<cudaStream_t>(stream.handle)));
    });
}
void copy(void* dst, Device target, const void* src, Device source, size_t size, bool ordered) {
    if (!size) return;
    int device = target.backend == BackendId::Cpu ? source.index : target.index;
    on_device_void(device, [&] {
        if (ordered && source.backend != BackendId::Cpu && target.backend != BackendId::Cpu) {
            if (source.index != device) accelerator_peer(source.index, device);
            auto stream = backend_stream(target, BackendStreamKind::Copy);
            backend_side_stream_wait_default(BackendStreamKind::Copy, device, source.index);
            copy_async(dst, target, src, source, size, stream);
            backend_default_stream_wait_side(BackendStreamKind::Copy, device, device);
            if (source.index != device)
                backend_default_stream_wait_side(BackendStreamKind::Copy, device, source.index);
        } else if (ordered && source.backend == BackendId::Cpu) {
            // A host input, ordered against the compute stream rather than
            // blocking on it. The blocking form is pathologically slow for
            // small transfers on some drivers -- a 2 KB copy measures 2.3 ms
            // here against 1.5 us for the same bytes issued asynchronously,
            // and the cliff sits exactly at the 64 KB pageable staging
            // threshold. A caller that asks for `ordered` only needs the bytes
            // to land before the kernels that read them, which stream order
            // already gives.
            //
            // Pageable source memory is staged into the driver's own buffer
            // before cudaMemcpyAsync returns, so the caller may reuse it the
            // moment this call does. Pinned memory carries no such guarantee,
            // so that case still waits for the copy to drain.
            cudaPointerAttributes attr{};
            const auto query = cudaPointerGetAttributes(&attr, src);
            // An unregistered pointer is the ordinary case, not a failure;
            // older drivers report it as an error, so clear the sticky flag.
            if (query != cudaSuccess) cudaGetLastError();
            const bool pageable = query != cudaSuccess
                || attr.type == cudaMemoryTypeUnregistered;
            checkCudaErrors(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice,
                                            cudaStreamPerThread));
            if (!pageable) {
                LaunchErrorScope error_scope(
                    target, true, reinterpret_cast<uintptr_t>(cudaStreamPerThread));
                checkCudaErrors(cudaStreamSynchronize(cudaStreamPerThread));
            }
        } else if (target.backend == BackendId::Cpu && source.backend != BackendId::Cpu) {
            LaunchErrorScope error_scope(
                source, true, reinterpret_cast<uintptr_t>(cudaStreamPerThread));
            // Readback waits for its producer stream's event, not the device.
            // This also serves data-dependent shape counts and scalar item().
            cudaEvent_t done;
            checkCudaErrors(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
            try {
                checkCudaErrors(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost,
                                                cudaStreamPerThread));
                checkCudaErrors(cudaEventRecord(done, cudaStreamPerThread));
                checkCudaErrors(cudaEventSynchronize(done));
            } catch (...) {
                const auto cleanup = cudaEventDestroy(done);
                if (cleanup != cudaSuccess)
                    LOGe << "Readback event cleanup failed:" << cudaGetErrorString(cleanup);
                throw;
            }
            checkCudaErrors(cudaEventDestroy(done));
        } else if (capturing()) {
            // A blocking copy is impossible while the stream is being
            // recorded -- `cudaMemcpy` reports cudaErrorStreamCaptureImplicit
            // and poisons the capture. The copy that lands here during a
            // capture is an operator staging its own constants (every `x * 2`
            // builds an array op, and a kept graph re-executes it on every
            // run), so issuing it stream-ordered is both legal and what the
            // graph needs: the node reads the operator's host buffer on each
            // launch, and a kept graph holds that buffer for as long as it
            // holds the operator.
            checkCudaErrors(cudaMemcpyAsync(dst, src, size,
                                            copy_kind(target, source),
                                            cudaStreamPerThread));
        } else {
            checkCudaErrors(cudaMemcpy(dst, src, size, copy_kind(target, source)));
        }
    });
}
} // namespace

BackendOps make_cuda_backend() {
    BackendOps ops;
    ops.id = BackendId::Cuda;
    ops.name = "cuda";
    ops.execution.supports_auto_flush = true;
    ops.device_count = accelerator_count;
    ops.current_device = accelerator_current;
    ops.set_device = accelerator_set;
    ops.allocator = accelerator_allocator_for;
    ops.copy = copy;
    ops.copy_async = copy_async;
    ops.synchronize = accelerator_sync;
    ops.stream = accelerator_backend_stream;
    ops.enable_peer = accelerator_peer;
    ops.memory_allocate = raw_allocate;
    ops.memory_free = raw_free;
    ops.memory_info = memory_info;
    ops.check_error = check_error;
    ops.compute_stream = compute_stream;
    ops.graph_capture_begin = graph_capture_begin;
    ops.graph_capture_end = graph_capture_end;
    ops.graph_launch = graph_launch;
    ops.graph_release = graph_release;
    ops.stream_create = create_stream;
    ops.stream_destroy = destroy_stream;
    ops.stream_synchronize = synchronize_stream;
    ops.event_create = create_event;
    ops.event_destroy = destroy_event;
    ops.event_record = record_event;
    ops.event_synchronize = synchronize_event;
    ops.event_elapsed = elapsed_event;
    ops.stream_wait_event = wait_event;
    ops.host_callback = host_callback;
    ops.architectures = architectures;
    ops.check_nan = cuda_backend_check_nan;
    return ops;
}
} // namespace jittor
