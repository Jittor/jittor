// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************

#include "common.h"
#include "runtime/device.h"
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include "helper_cuda.h"
#endif

namespace jittor {

DEFINE_RUNTIME_FLAG_WITH_SETTER(int, use_cuda, 0,
    "Use cuda or not. 1 for trying to use cuda, 2 for forcing to use cuda.");
// NB: compiler.gen_jit_flags extracts this doc with the regex
// DEFINE_FLAG...\((.*?)\); and then eval()s it as one Python expression, so
// the text must be a single literal on a single line and must not contain the
// two characters ");" -- a doc ending a parenthetical would truncate the match
// there and leave an unterminated string.
DEFINE_RUNTIME_FLAG_WITH_SETTER(int, device_id, -1,
    "The CUDA device new Vars are placed on, torch's current device. Setting it switches the device in place -- cudaSetDevice plus a handle swap in every library wrapper -- and never restarts the process; the other devices stay usable. Reads -1 only when no CUDA device exists.");
// This had a setter whose entire body was `if (sync_run == value) return;
// sync_run = value;` -- the assignment the macro was about to do anyway.
DEFINE_RUNTIME_FLAG(int, sync_run, 1,
    "Enable per-op-sync or not");

EXTERN_LIB void sync_all(bool device_sync);

#ifdef HAS_CUDA
int get_device_count() {
    auto& count = runtime_device_state().device_count;
    if (count==-1) {
        // cudaGetDeviceCount returns cudaErrorNoDevice (and may leave `count`
        // untouched at -1) when no GPU is visible (e.g. CUDA_VISIBLE_DEVICES="").
        // Treat any error as 0 devices so callers (the array_op static Init,
        // setter_use_cuda) take the CPU path instead of aborting on a CUDA call.
        if (cudaGetDeviceCount(&count) != cudaSuccess)
            count = 0;
    }
    return count;
}
#endif

void setter_use_cuda(const int& old_value, const int& requested) {
    if (old_value == requested) return;
    // Local: the CUDA branch below may downgrade what was asked for.
    int value = requested;
#ifdef HAS_CUDA
    if (value) {
        int count=0;
        cudaGetDeviceCount(&count);
        if (count == 0) {
            // No CUDA device visible at runtime (e.g. CUDA_VISIBLE_DEVICES="" in
            // a CPU-only process such as a Ray orchestrator actor with
            // num_gpus=0). Fall back to CPU instead of aborting, so importing
            // jittor / the torch-shim does not crash where there is no GPU.
            LOGw << "No CUDA device available; falling back to CPU (use_cuda=0).";
            value = 0;
        } else {
            LOGi << "CUDA enabled.";
            // Pin down which device is current now, so jt.flags.device_id
            // reads it rather than -1 and flag_scope(device_id=N) restores to
            // a real device instead of to "unset".
            current_device();
        }
    } else {
        LOGv << "CUDA disabled.";
    }
#else
    CHECK(value==0) << "No CUDA found.";
#endif
    if (old_value != value) {
        // Flush the pending graph while the *old* backend is still in force.
        //
        // The lazy graph standing here was built for `old_value`, and
        // `Op::do_jit_prepare` has already cleared the other backend's flag on
        // every op it prepared: an op prepared under CUDA has `_cpu` off for
        // good. Compiling that graph after the flag has dropped to 0 reaches
        // `Op::do_jit_prepare` again, takes the CPU branch, and dies on
        // `ASSERT(flag(OpFlags::_cpu))` -- "Op broadcast_to doesn't have cpu
        // version" -- from inside the parallel compiler, i.e. as a
        // `RuntimeError` raised by `flag_scope.__exit__` rather than by
        // anything the user wrote.
        //
        // Before [2.21] the macro called the setter *before* the assignment,
        // so this flush got the old value for free. [2.21] moved the
        // assignment first (so a setter can see, and correct, the new value)
        // and silently took that away: `sync_all` is the one thing in here
        // that must run under the old setting. Restore it around the flush.
        runtime_device_state().use_cuda = old_value;
        sync_all(0);
        // If sync_all throws, `use_cuda` stays at old_value and the macro's
        // rollback writes the same thing -- flag and side effect still agree.
    }
    // Not a write-back: the macro already assigned the requested value. This
    // publishes the *correction* made above when CUDA was asked for and no
    // device answered, and re-publishes it after the flush above.
    runtime_device_state().use_cuda = value;
}

#ifdef HAS_CUDA

// The device the CUDA runtime is on, cached so that placing a Var (which asks
// on every construction) is a load rather than a driver call. -1 means "not
// asked yet"; get_device_count()==0 keeps it there forever.
// The runtime accessor constructs the vector before static initializers in
// backend translation units can register their hooks.
static vector<device_switch_hook_t>& device_switch_hooks() {
    return runtime_device_state().switch_hooks;
}

int current_device() {
    auto& cur_device = runtime_device_state().current_device;
    if (cur_device < 0) {
        if (get_device_count() <= 0) return -1;
        int d = 0;
        if (cudaGetDevice(&d) != cudaSuccess) {
            cudaGetLastError();
            return -1;
        }
        cur_device = d;
        // Keep the flag readable as the current device from the first query
        // on. flag_scope saves whatever it reads on entry and writes it back
        // on exit, so a flag that still said -1 would restore to "unset" and
        // silently leave the scope's device current.
        runtime_device_state().device_id = d;
    }
    return cur_device;
}

void set_current_device(int device) {
    int count = get_device_count();
    CHECK(device >= 0 && device < count)
        << "Invalid CUDA device index" << device >> ", visible device count is" << count;
    int cur = current_device();
    // The flag names the current device even when nothing has to move.
    runtime_device_state().device_id = device;
    if (device == cur) return;
    checkCudaErrors(cudaSetDevice(device));
    runtime_device_state().current_device = device;
    for (auto hook : device_switch_hooks()) hook(device);
}

void add_device_switch_hook(device_switch_hook_t hook) {
    if (!get_device_count()) return;
    device_switch_hooks().push_back(hook);
    // The hook owns per-device state that its callers reach through one
    // global name, so it has to be run for the device that is current now --
    // otherwise that global stays null until the first switch, which may
    // never come in a single-device process.
    int d = current_device();
    if (d >= 0) hook(d);
}

void enable_peer_access(int from, int to) {
    if (from == to || from < 0 || to < 0) return;
    int n = get_device_count();
    if (from >= n || to >= n) return;
    // One entry per ordered pair; cudaDeviceEnablePeerAccess is per (context,
    // peer) and asking twice is an error rather than a no-op.
    auto& enabled = runtime_device_state().peer_enabled;
    if ((int)enabled.size() < n*n) enabled.resize(n*n, 0);
    auto& done = enabled[from*n+to];
    if (done) return;
    done = 1;
    int can = 0;
    if (cudaDeviceCanAccessPeer(&can, to, from) != cudaSuccess || !can) {
        // Not an error: without peer access cudaMemcpy stages through the
        // host on its own, which is slower but correct.
        cudaGetLastError();
        return;
    }
    int prev = current_device();
    checkCudaErrors(cudaSetDevice(to));
    auto err = cudaDeviceEnablePeerAccess(from, 0);
    if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled)
        LOGw << "cudaDeviceEnablePeerAccess(" >> from << "->" >> to >> ") failed:"
            << cudaGetErrorString(err);
    cudaGetLastError();
    checkCudaErrors(cudaSetDevice(prev));
}

void sync_devices(uint64 devices) {
    if (!devices) {
        checkCudaErrors(cudaDeviceSynchronize());
        return;
    }
    // cudaDeviceSynchronize only waits on the current device, so a run that
    // launched on two devices needs one call per device -- and the caller's
    // current device back afterwards.
    int prev = current_device();
    for (int d = 0; d < 64; d++) {
        if (!((devices >> d) & 1)) continue;
        if (d != current_device()) set_current_device(d);
        checkCudaErrors(cudaDeviceSynchronize());
    }
    if (prev >= 0 && prev != current_device()) set_current_device(prev);
}

#endif

void setter_device_id(const int& old_value, const int& value) {
#ifdef HAS_CUDA
    // Below zero is "unset", and it is also what this setter is handed at
    // static-init time from the flag's default. Return before touching the
    // CUDA runtime: initialising it this early (before NCCL picks a device,
    // before a fork) is exactly what the old restart-the-process setter was
    // careful to avoid. current_device() makes the flag truthful the first
    // time a Var is placed, so flag_scope never has a -1 to restore.
    if (value < 0) return;
    if (!get_device_count()) {
        LOGw << "No CUDA device available; ignoring device_id" << value;
        return;
    }
    set_current_device(value);
#else
    CHECK(value < 0) << "No CUDA found.";
#endif
}

} // jittor
