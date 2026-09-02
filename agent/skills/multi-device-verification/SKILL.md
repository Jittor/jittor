---
name: multi-device-verification
description: How to verify per-device placement in Jittor on a box with two or more GPUs — proving a tensor really lives on the card it claims (not just that Var.device_id says so), proving a cross-device copy is correctly ordered (and knowing when the hardware makes that unprovable), and the four traps that make a multi-device change look green while it is wrong. Use when touching Var::device_id, get_allocator(device, ...), the device-switch hooks, device_copy, or the torch.cuda device facade.
---

# Verifying multi-device placement

`Var.device_id` is bookkeeping. A change that sets it correctly and allocates on
the wrong card passes every test that only reads it back. Every claim below is
checked against something outside jittor's own bookkeeping.

## 0. Setup

Two cards are required; one card cannot distinguish "placed" from "ignored".

```bash
cd <worktree>
JITTOR_HOME=<worktree-home> TMPDIR=<worktree-tmp> \
CUDA_VISIBLE_DEVICES=<A>,<B> nvcc_path=/usr/local/cuda/bin/nvcc \
taskset -c <cores> <python> -m pytest tests/backends/cuda/test_multi_device.py -q
```

Indices are always *logical*: `CUDA_VISIBLE_DEVICES=3,7` means device 0 and
device 1 inside the process. Never write a physical card number into a test.

A hand-written `python -c`/script additionally needs
`PYTHONPATH=<worktree>/python` — without it you are testing the main tree
(see `AGENT-BRIEF.md` §1).

## 1. Proving a tensor is really on device N

### The driver's answer (per tensor, exact)

Ask the CUDA driver where a pointer lives. `Var.device_raw_ptr` syncs and hands
back the device pointer.

```python
import ctypes
_libcuda = ctypes.CDLL("libcuda.so.1")
CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9   # cuda.h. NOT 15.

def pointer_device(ptr):
    value = ctypes.c_int(-1)
    if _libcuda.cuPointerGetAttribute(
            ctypes.byref(value), CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
            ctypes.c_void_p(ptr)) != 0:
        return None
    return value.value

assert pointer_device(x.device_raw_ptr) == 1
```

* **Get the attribute number right.** 15 is
  `CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE`, which answers `0` for
  every pointer — so a wrong constant reports *every* tensor on device 0 and
  the test fails in a way that looks like a placement bug in the code under
  test. This cost a debugging round.
* Use `libcuda.so.1` (driver API, ships with the driver, stable signature), not
  the runtime's `cudaPointerGetAttributes` — that struct's layout changed
  between CUDA versions.

### The memory-accounting answer (whole pool, coarse but independent)

Useful when you suspect a whole pool is on the wrong card, or as a
cross-check that the pointer query itself is right.

```python
rt = ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so.12.2.140")
def free_mb():
    f, t = ctypes.c_size_t(), ctypes.c_size_t()
    rt.cudaMemGetInfo(ctypes.byref(f), ctypes.byref(t))
    return f.value >> 20

jt.set_device(0); before0 = free_mb()
jt.set_device(1); before1 = free_mb()
with jt.flag_scope(device_id=1):
    big = jt.empty((256 * 1024 * 1024 // 4,), "float32"); big.sync()
jt.set_device(0); after0 = free_mb()
jt.set_device(1); after1 = free_mb()
assert before1 - after1 >= 250 and before0 - after0 == 0
```

**Switch devices with `jt.set_device`, never with a ctypes `cudaSetDevice`.**
Jittor caches the current device and swaps library handles on its own switch;
moving the device behind its back leaves that cache lying, and everything
after it is measured on the wrong card.

## 2. Proving cross-device copy ordering

The claim: a copy from device A to device B must wait for the kernel that
produced the source, and the source's stream must then wait for the copy
before A may reuse the block. Each device drives its own default stream and
nothing orders them against each other.

The shape of the test:

```python
# long kernel on the source, copy issued immediately, value read on the target
s = a
for _ in range(6):
    s = jt.matmul(s, a)      # ~20 ms of device-0 work, nothing has run yet
b = s.to_device(1)           # queued while the chain is still on the device
got = b.numpy()
ref = s.numpy()              # the truth, from the source device
assert (got == ref).all()
```

Why the final device sync does not hide the bug: it makes the work *finish*,
not *happen in order*. An unordered copy has already read the block's previous
contents, and those wrong bytes are still in the destination afterwards. Make
the source block recognisable — allocate and free a same-sized buffer with a
distinctive fill first, so the SFRL pool hands the chain that recycled block —
and a race gives old data rather than plausible zeros.

### Know whether your hardware can fail this test

```python
rt.cudaDeviceCanAccessPeer(ctypes.byref(can), 0, 1)
```

If `can == 0`, the driver stages every cross-device copy through host memory
and serialises it against the source device *itself*. The ordering is then
unobservable: deleting the `cudaEventRecord`/`cudaStreamWaitEvent` pair from
`DeviceCopyOp::run` leaves every test passing. Verified, not assumed — all
eight GPUs on this box report `canAccessPeer == 0` for every pair (consumer
cards; NVIDIA disables P2P), so **on this box the ordering test is a
correctness assertion, not a regression guard.** Run it on a peer-capable pair
(`nvidia-smi topo -m` showing NV#/PIX) to get the guard.

Report the regime rather than reporting a pass: `tests/backends/cuda/
test_device_copy.py::_peer_regime` does this, so a green run is not mistaken
for a proof.

## 3. Proving `flag_scope` / the facade did not merely lie back to you

`jt.flags.device_id` reads back whatever the setter left. Check the *effect*,
not the flag:

* after `with jt.flag_scope(device_id=1): ...`, assert `jt.current_device() == 0`
  **and** that a tensor built inside reports pointer-device 1;
* for the torch facade, assert on `Tensor.device.index` **and** on a value
  computed there, since `torch.cuda.current_device()` can be right while every
  allocation still lands on 0 (that was the previous state of the world).

## 4. The four traps

1. **A `migrate_to_*` that uses the op's allocator.** Every op but
   `device_copy` has its inputs on its own device, so `migrate_to_gpu(v,
   op_allocator)` looks right forever — until a `device_copy` input is
   host-resident and gets uploaded to the *destination* card while
   `Var::device_id` still says the source. The next kernel reading it faults
   with `cudaErrorIllegalAddress`, several statements away. Migrate by
   `v->device_id`, not by the op's.
   Repro: `x=jt.array(a); x1=x.to_device(1); x1.numpy(); x1.to_device(0);
   (x1*2).numpy()`.
2. **Memory handed to a foreign library.** `jittor_cuda_malloc` (cupy) and
   cuTT's `jt_alloc` allocated and freed through `exe.allocator`. That pointer
   now changes with the device, so a plan built on device 1 and destroyed at
   exit is freed into device 0's pool: `allocation not found: N` from
   `sfrl_allocator.cc`, thrown out of a static destructor, so the process
   `terminate`s *after* the test summary prints. **A clean "N passed" line
   followed by `terminate called` is a failure.** Record the allocator with the
   pointer.
3. **Streams and events belong to a device.** `cudaEventRecord(event, 0)` fails
   with `invalid resource handle` when `event` was created on another device
   than the current one — so any global stream/event pair (`array_op.cc`,
   `fetch_op.cc`) needs one instance per device, swapped by the device-switch
   hook. A stream *may* wait on another device's event and *may* copy across
   devices; only the record is pinned.
4. **The flag's doc string is parsed.** `compiler.gen_jit_flags` extracts a
   `DEFINE_FLAG...` doc with `\((.*?)\);` and `eval`s it as one Python
   expression. A doc spanning two C++ string literals, or containing the two
   characters `");`, truncates the match and `import jittor` dies with
   `SyntaxError: unterminated string literal` before anything runs. One
   literal, one line, no parenthetical at the end.

## 5. What a full check looks like

```bash
# native placement + per-device pools/handles
pytest tests/backends/cuda/test_multi_device.py -q
# the cross-device copy op
pytest tests/backends/cuda/test_device_copy.py -q
# the torch facade -- separate command, shim mode is process-global
JITTOR_TORCH_SHIM=1 pytest tests/compat/torch/test_multi_device.py -q
```

Never put `tests/compat/torch` and a native path in one pytest command: the
shim is process-global and switches the whole tree into torch semantics
(`AGENT-BRIEF.md` §4, board §"正确跑法").
