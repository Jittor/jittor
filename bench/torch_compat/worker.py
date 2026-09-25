"""Run one workload under one runtime and print one JSON result line.

Launched by ``run.py``, once per (workload, runtime), in a fresh interpreter so
neither memory nor JIT state carries from one measurement into the next.

    python worker.py <workload> --runtime {torch,jittor} --device cuda

The last stdout line is ``BENCH_RESULT <json>``. A failure still prints a
result line, with ``status`` set and the exception attached, so one broken
workload is a row in the table rather than a missing one.
"""

import argparse
import json
import os
import platform
import statistics
import sys
import time
import traceback
from contextlib import ExitStack
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

MARKER = "BENCH_RESULT "


def import_torch(runtime):
    """Return ``torch`` for the requested runtime, and refuse the other one.

    The Jittor side is a plain ``import torch`` in an environment with
    ``jittor-torch`` installed -- what a user does. The in-process
    ``JITTOR_TORCH_SHIM=1`` switch the unit tests use installs the module but
    not the runtime site that publishes ``torch``'s distribution metadata, so
    Transformers and Diffusers decide PyTorch is absent and refuse to build a
    model. A shim that silently resolved to real PyTorch -- or the reverse --
    would make every number below compare a framework with itself.
    """
    os.environ.pop("JITTOR_TORCH_SHIM", None)
    if runtime == "jittor":
        import torch

        if not hasattr(torch, "_torch_compat_install_context"):
            raise SystemExit("torch did not resolve to the Jittor shim; "
                             "install jittor-torch (pip install -e compat)")
        return torch
    import torch

    if not hasattr(torch, "_C") or hasattr(torch, "_torch_compat_install_context"):
        raise SystemExit("torch did not resolve to an independent PyTorch")
    return torch


def configure(torch, device, tf32, cudnn_benchmark):
    if device != "cuda":
        return {}
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cudnn.benchmark = cudnn_benchmark
    set_precision = getattr(torch, "set_float32_matmul_precision", None)
    if callable(set_precision):
        set_precision("high" if tf32 else "highest")
    get_precision = getattr(torch, "get_float32_matmul_precision", None)
    return {
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "matmul_precision": get_precision() if callable(get_precision) else None,
    }


class Synchronizer:
    """Force a step's outputs and wait for the device, per runtime.

    Jittor is lazy: until something asks for a value, a step is only a graph.
    ``jt.sync`` on the returned tensors executes exactly that step and then
    waits for the device. PyTorch only needs its CUDA queue drained.
    """

    def __init__(self, torch, runtime, device):
        self.torch = torch
        self.runtime = runtime
        self.device = device
        if runtime == "jittor":
            import jittor

            self.jt = jittor

    def __call__(self, outputs):
        if self.runtime == "jittor":
            self.jt.sync(list(outputs), device_sync=self.device != "cpu")
            self.jt.sync_all(self.device != "cpu")
        elif self.device == "cuda":
            self.torch.cuda.synchronize()


class DeviceMemorySampler:
    """Peak device memory this process holds, read from the driver (NVML).

    The runtimes' own counters are not comparable: Jittor's
    ``torch.cuda.max_memory_allocated`` reported 0.1-1.3 GB for training steps
    that ran out of a 22 GB card. NVML sees what the process actually holds --
    context, allocator caches and all -- the same way for both, which is the
    number that decides whether a workload fits.
    """

    def __init__(self, interval=0.02):
        self.interval = interval
        self.peak = None
        self._stop = None
        self._thread = None

    def start(self):
        try:
            import pynvml

            pynvml.nvmlInit()
        except Exception:  # no NVML: the column is reported as unknown
            return self
        import threading

        self._nvml = pynvml
        self._handles = [pynvml.nvmlDeviceGetHandleByIndex(index)
                         for index in range(pynvml.nvmlDeviceGetCount())]
        self._pid = os.getpid()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _sample(self):
        used = 0
        for handle in self._handles:
            try:
                processes = self._nvml.nvmlDeviceGetComputeRunningProcesses(handle)
            except self._nvml.NVMLError:
                continue
            used += sum(p.usedGpuMemory or 0 for p in processes
                        if p.pid == self._pid)
        if used and (self.peak is None or used > self.peak):
            self.peak = used

    def _run(self):
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)

    def stop(self):
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self._sample()
        return self.peak


def peak_memory(torch, runtime, device):
    if device != "cuda":
        return None
    try:
        value = int(torch.cuda.max_memory_allocated())
    except Exception:  # an unreported number is recorded as unknown
        return None
    return value or None


def versions(torch, runtime):
    report = {"python": platform.python_version(),
              "torch": getattr(torch, "__version__", None)}
    for name in ("transformers", "diffusers", "numpy"):
        try:
            report[name] = __import__(name).__version__
        except Exception:
            report[name] = None
    if runtime == "jittor":
        import jittor

        report["jittor"] = jittor.__version__
        report["torch_api"] = getattr(torch, "__torch_version__", None)
    else:
        report["cuda"] = getattr(torch.version, "cuda", None)
        cudnn = getattr(torch.backends, "cudnn", None)
        report["cudnn"] = cudnn.version() if cudnn is not None else None
    return report


def is_device_oom(error, text):
    """Only the allocator's own out-of-memory, not any text mentioning it.

    Jittor appends "This might be an overcommit issue or out of memory" to every
    failed compiler invocation, so a plain substring test filed nvcc errors
    under OOM and hid real compile failures.
    """
    if type(error).__name__ == "OutOfMemoryError":
        return True
    lowered = text.lower()
    return any(marker in lowered for marker in (
        "cuda out of memory", "out of memory on the accelerator",
        "gpu memory is overflow",
        "cudaerrormemoryallocation",
        "cuda_error_out_of_memory", "cublas_status_alloc_failed"))


#: Warmup steps allowed beyond a workload's configured count while its step
#: time is still moving.
MAX_EXTRA_WARMUP = 20


def settled(warm, tolerance=0.10):
    """Whether the last two warmup steps agree within ``tolerance``."""
    if len(warm) < 2:
        return False
    a, b = warm[-2], warm[-1]
    return abs(a - b) <= tolerance * min(a, b)


def scalar(tensor):
    try:
        if tensor.numel() == 1:
            return float(tensor.detach().float().cpu().item())
    except Exception:
        return None
    return None


def measure(options, stack, result):
    torch = import_torch(options.runtime)
    result["versions"] = versions(torch, options.runtime)
    result["settings"] = configure(torch, options.device, options.tf32,
                                   options.cudnn_benchmark)
    if options.runtime == "jittor":
        import jittor as jt

        if not options.allow_fallback:
            # A CPU fallback inside a CUDA measurement makes the number mean
            # something else entirely; fail the row instead of reporting it.
            stack.enter_context(jt.runtime.scope(backend_fallback="error"))
        result["fallbacks_before"] = int(jt.core.backend_fallback_count())

    from workloads import WORKLOADS

    cls = WORKLOADS[options.workload]
    dtype = options.dtype or cls.default_dtype
    result.update(workload=cls.name, family=cls.family, mode=cls.mode,
                  unit=cls.unit, dtype=dtype, description=cls.description)
    warmup, repeats = cls.iterations
    if options.size == "tiny":
        warmup, repeats = 1, 2
    warmup = options.warmup if options.warmup is not None else warmup
    repeats = options.repeats if options.repeats is not None else repeats

    torch.manual_seed(options.seed)
    sync = Synchronizer(torch, options.runtime, options.device)
    workload = cls(torch, options.device, dtype, options.size,
                   batch=options.batch)

    started = time.perf_counter()
    workload.setup()
    sync([])
    result["setup_s"] = time.perf_counter() - started
    result["config"] = workload.config()
    result["parameters"] = workload.parameter_count()
    result["items_per_step"] = workload.items

    # Warm up at least `warmup` steps, then until two consecutive steps agree
    # within 10% (at most MAX_EXTRA_WARMUP more). A fixed count is not enough
    # for Jittor: a later step can still meet a first-seen shape and compile
    # (sd15_vae_decode's second step took 21.7 s against a 36 ms steady state),
    # and a timed window that straddles two states is not one measurement.
    warm = []
    while (len(warm) < max(1, warmup)
           or (len(warm) < max(1, warmup) + MAX_EXTRA_WARMUP
               and not settled(warm))):
        started = time.perf_counter()
        outputs = workload.step()
        sync(outputs)
        warm.append(time.perf_counter() - started)
        if len(warm) == 1:
            result["first_value"] = scalar(outputs[0])
        del outputs
    result["first_step_s"] = warm[0]
    result["warmup_s"] = warm
    result["warmup_settled"] = settled(warm)

    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        outputs = workload.step()
        sync(outputs)
        samples.append(time.perf_counter() - started)
        last = outputs
    result["last_value"] = scalar(last[0])
    del last

    result["samples_s"] = samples
    result["median_s"] = statistics.median(samples)
    result["min_s"] = min(samples)
    result["mean_s"] = statistics.fmean(samples)
    result["stdev_s"] = statistics.pstdev(samples)
    result["throughput"] = workload.items / result["median_s"]
    result["peak_memory_bytes"] = peak_memory(torch, options.runtime,
                                              options.device)
    if options.runtime == "jittor":
        import jittor as jt

        result["fallbacks"] = (int(jt.core.backend_fallback_count())
                               - result.pop("fallbacks_before"))
    result["status"] = "ok"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("workload")
    parser.add_argument("--runtime", choices=("torch", "jittor"), required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--size", choices=("full", "tiny"), default="full")
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=None,
                        help="override the workload's batch size")
    parser.add_argument("--no-tf32", dest="tf32", action="store_false")
    parser.add_argument("--cudnn-benchmark", action="store_true")
    parser.add_argument("--allow-fallback", action="store_true")
    options = parser.parse_args()

    result = {"workload": options.workload, "runtime": options.runtime,
              "device": options.device, "size": options.size}
    sampler = DeviceMemorySampler().start() if options.device == "cuda" else None
    try:
        with ExitStack() as stack:
            measure(options, stack, result)
    except BaseException as error:  # reported as the row's status, never hidden
        if isinstance(error, KeyboardInterrupt):
            raise
        text = "".join(traceback.format_exception_only(type(error), error))
        result["status"] = "oom" if is_device_oom(error, text) else "error"
        result["error"] = text.strip()[-2000:]
        traceback.print_exc()
    if sampler is not None:
        # Recorded on failure too: an OOM row shows how high it got.
        result["peak_device_bytes"] = sampler.stop()
    sys.stdout.flush()
    print(MARKER + json.dumps(result), flush=True)
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
