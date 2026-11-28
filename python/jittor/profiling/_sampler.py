"""Host-side samplers that run next to a profile: Python call sites and NVML.

The Python sampler answers "which lines of the program is the host spending
its time on", including time inside jittor's frontend: a thread reads the
profiled thread's current frame every ``interval`` seconds and charges the
sample to the innermost frame outside the ``jittor``/``torch`` packages --
the same rule ``Op::launch_origin`` uses, so the two site tables agree on what
a call site is. It needs the GIL to look, so it only sees the profiled thread
at the interpreter's switch points; the switch interval is lowered to the
sampling interval while it runs and restored after. Sampling is statistical:
a site with fewer than a handful of samples is noise.

The NVML sampler reads what the driver says this process holds on the
profiled device -- CUDA context, modules and library workspaces included,
which no allocator counter sees.
"""

import os
import sys
import threading
import time
from collections import Counter


def _is_internal(frame):
    name = frame.f_globals.get("__name__", "")
    return (name == "jittor" or name.startswith("jittor.") or name == "torch"
            or name.startswith("torch."))


def call_site(frame):
    """(site, api) for a frame: the first frame outside jittor/torch, and the
    innermost framework function it was calling (or None)."""
    api = None
    depth = 0
    while frame is not None and depth < 128:
        if not _is_internal(frame):
            code = frame.f_code
            return "%s:%d" % (code.co_filename, frame.f_lineno), api
        if api is None:
            api = "%s.%s" % (frame.f_globals.get("__name__", "?"), frame.f_code.co_name)
        frame = frame.f_back
        depth += 1
    return "(framework only)", api


class PythonSampler:
    def __init__(self, interval=0.001, thread_id=None):
        self.interval = float(interval)
        if self.interval <= 0:
            raise ValueError("interval must be positive, not %r" % (interval,))
        self.thread_id = thread_id if thread_id is not None else threading.get_ident()
        self.sites = Counter()
        self.sites_ns = Counter()
        self.apis = Counter()
        self.samples = 0
        self._stop = threading.Event()
        self.armed = False
        self._thread = None
        self._switch = None
        self.started = self.stopped = None

    def _run(self):
        frames = sys._current_frames
        last_ns = time.perf_counter_ns()
        while not self._stop.is_set():
            now_ns = time.perf_counter_ns()
            dt_ns = max(now_ns - last_ns, 0)
            last_ns = now_ns
            frame = frames().get(self.thread_id) if self.armed else None
            if frame is not None:
                site, api = call_site(frame)
                self.sites[site] += 1
                self.sites_ns[site] += dt_ns
                if api:
                    self.apis[api] += 1
                self.samples += 1
            del frame
            self._stop.wait(self.interval)

    def start(self):
        self._switch = sys.getswitchinterval()
        sys.setswitchinterval(max(self.interval / 2, 1e-5))
        self.started = time.perf_counter_ns()
        self._thread = threading.Thread(target=self._run, name="jittor-profile-sampler", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self.stopped = time.perf_counter_ns()
        if self._switch is not None:
            sys.setswitchinterval(self._switch)
            self._switch = None
        return self


_nvml_lock = threading.Lock()
_nvml_mod = None
_nvml_err = None
_nvml_probed = False


def _nvml():
    global _nvml_mod, _nvml_err, _nvml_probed
    if _nvml_probed:
        return _nvml_mod, _nvml_err
    with _nvml_lock:
        if _nvml_probed:
            return _nvml_mod, _nvml_err
        try:
            import pynvml
        except ImportError:
            _nvml_err = "pynvml (nvidia-ml-py) is not installed"
            _nvml_probed = True
            return None, _nvml_err
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError as exc:
            _nvml_err = "nvmlInit failed: %s" % exc
            _nvml_probed = True
            return None, _nvml_err
        _nvml_mod = pynvml
        _nvml_probed = True
        return _nvml_mod, None


def nvml_handle(pynvml, ordinal):
    """The NVML handle of CUDA ordinal ``ordinal`` under CUDA_VISIBLE_DEVICES."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or visible.strip() == "":
        return pynvml.nvmlDeviceGetHandleByIndex(ordinal)
    entries = [item.strip() for item in visible.split(",") if item.strip()]
    entry = entries[ordinal]
    if entry.isdigit():
        return pynvml.nvmlDeviceGetHandleByIndex(int(entry))
    return pynvml.nvmlDeviceGetHandleByUUID(entry)


class NvmlSampler:
    """Process memory on one device as NVML reports it."""

    def __init__(self, ordinal, interval=0.002):
        self.ordinal = ordinal
        self.interval = interval
        self.error = None
        self.start_bytes = self.end_bytes = self.peak_bytes = None
        self.samples = 0
        self._stop = threading.Event()
        self._thread = None
        self._pynvml, self.error = _nvml()
        if self._pynvml is not None:
            try:
                self._handle = nvml_handle(self._pynvml, ordinal)
            except (self._pynvml.NVMLError, IndexError, ValueError) as exc:
                self.error = "cannot map CUDA device %d to NVML: %s" % (ordinal, exc)
                self._pynvml = None

    def read(self):
        pid = os.getpid()
        processes = self._pynvml.nvmlDeviceGetComputeRunningProcesses(self._handle)
        total = 0
        matched = False
        for p in processes:
            if p.pid == pid:
                matched = True
                if p.usedGpuMemory is None:
                    return None
                total += p.usedGpuMemory
        return total if matched else 0

    def _run(self):
        while not self._stop.is_set():
            try:
                value = self.read()
            except self._pynvml.NVMLError as exc:
                self.error = "nvml read failed: %s" % exc
                break
            if value is not None:
                self.samples += 1
                if self.peak_bytes is None or value > self.peak_bytes:
                    self.peak_bytes = value
            self._stop.wait(self.interval)

    def start(self):
        if self._pynvml is None:
            return self
        try:
            self.start_bytes = self.peak_bytes = self.read()
        except self._pynvml.NVMLError as exc:
            self.error = "nvml read failed: %s" % exc
            self._pynvml = None
            return self
        self._thread = threading.Thread(target=self._run, name="jittor-profile-nvml", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        if self._pynvml is None:
            return self
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        try:
            end_val = self.read()
        except self._pynvml.NVMLError as exc:
            self.error = self.error or ("nvml read failed: %s" % exc)
            return self
        self.end_bytes = end_val
        if end_val is not None:
            self.peak_bytes = max(self.peak_bytes or 0, end_val)
        return self
