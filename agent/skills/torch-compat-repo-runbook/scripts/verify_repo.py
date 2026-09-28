"""Four-axis verification of a per-repo torch-compat runbook.

A runbook that has not been run is a hypothesis. This driver executes the
repo's registered ecosystem cases on **both** runtimes and reports the four
axes the team signs off on:

* 支持的模型/case 清单 -- which cases the repo covers, and whether each ran;
* 精度 -- max abs / max scaled-relative error of every output and gradient
  against the independent PyTorch oracle, from the *same* weights and inputs;
* 显存 -- device memory of the child process on the same two axes for both
  runtimes: live bytes, and the runtime's own pool. Each runtime is asked
  through its per-device API so the pair means the same thing on both sides;
* 速度 -- minimum wall time over repeated samples, and the jittor/torch ratio.

The single-case execution is delegated to the project's own
`compat/tests/torch/_ecosystem_runner.py`, so the numbers come from the same
code path the ecosystem gate uses, not from a private reimplementation.

Usage:
    python verify_repo.py --repo transformers \
        --cases transformers_gpt2,transformers_bert \
        --oracle-python <real-torch>/bin/python \
        --device cuda --repeats 5 --out <dir>

Notes / honest limits:
* Device memory is asked of each runtime rather than sampled externally: on this
  box `nvidia-smi --query-compute-apps` does not list the process, and per-GPU
  `memory.used` includes co-tenants. Both runtimes are asked for live bytes and
  for their own pool, so `*_peak_bytes` is finally like-for-like. It used to
  pair torch's `max_memory_allocated` against Jittor's `total_cuda_used`, which
  is used+cached-free summed over every device -- reserved against live -- and
  that inflated the Jittor side. Numbers in reports written before this change
  are that old pair and must not be quoted as a memory ratio. Jittor's profiling
  can perturb timing, so the memory pass is a separate run from the timing pass.
* Wall clock on this box is contention-sensitive. Quote the minimum and keep
  the CPU/GPU and the case identical between the two runtimes.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
RUNNER = REPO / "compat" / "tests" / "torch" / "_ecosystem_runner.py"
RESULT_RE = re.compile(r"^ECOSYSTEM_RESULT (\{.*\})$", re.M)
PEAK_RE = re.compile(r"^MEMORY_PEAK_BYTES (-?\d+)$", re.M)
RESERVED_RE = re.compile(r"^MEMORY_RESERVED_BYTES (-?\d+)$", re.M)
UNMEASURABLE_RE = re.compile(r"^MEMORY_UNMEASURABLE (.+)$", re.M)

#: Runs a case while sampling the runtime's own device-memory accounting.
#:
#: Neither external sampler works on this box: `nvidia-smi --query-compute-apps`
#: does not list the process (container pid mapping), and per-GPU `memory.used`
#: includes co-tenants. So ask each runtime instead, and ask both for the *same
#: two numbers*: live bytes (torch `memory_allocated`, Jittor
#: `device_memory_used`) and pool bytes (torch `memory_reserved`, Jittor
#: `device_memory_reserved`).
#:
#: Do not use `jt.get_mem_info().total_cuda_used` here. It is used+cached-free
#: summed over *every* device (mem_info.cc:316-322), so pairing it with torch's
#: `max_memory_allocated` compares reserved against live and makes Jittor look
#: several times larger than it is. Jittor's per-device calls are the pair that
#: matches; when the jittor under test predates them, report *no* number rather
#: than one that means something else.
#:
#: `jt.core.get_peak_allocator_used_memory()` looks like the right counter -- it
#: is the allocator's own live high-water, the exact shape of torch's
#: `max_memory_allocated` -- but `MemoryProfiler::get_memory_info` sums
#: `SFRLAllocator::sfrl_allocators` with no CUDA filter (memory_profiler.cc:58),
#: so it counts host *and* device. Measured: 512 MiB of CPU-only Var moved it
#: 512 MiB, and a further 256 MiB CUDA Var moved it to 768. Only
#: `device_memory_used`/`device_memory_reserved` and the CUDA-only pool
#: (`total_cuda_used`) can be put beside a torch number.
#:
#: The peak therefore comes from `get_peak_device_used_memory(device)`, which is
#: that same high-water restricted to one device's pools. **It must not be read
#: by sampling `device_memory_used` from this thread**, which is what this
#: wrapper used to do: a python sampler only runs when the interpreter releases
#: the GIL -- during a step, only at device waits -- so on a fast step it reports
#: whichever intermediate value it happened to catch, while torch's
#: `max_memory_allocated` is maintained exactly by its allocator. Measured on an
#: 8-layer llama step (batch 4, seq 512): the sampler's max was 3406.5 MiB, the
#: runtime's own high-water was 7272 MiB, and torch's was 6655 MiB. So a sampled
#: "peak" understates by more than 2x on exactly the fast steps a run mostly
#: consists of, and the ratio printed from it measures the sampler's luck rather
#: than the two runtimes. It also manufactured a phantom "first-step 2x
#: transient" -- a slow first step's device waits are long enough for the sampler
#: to win the GIL, the fast later ones are not -- which two separate
#: investigations chased before the sampler itself was identified as the cause.
#:
#: Jittor's memory profiling can perturb timing, so the run that measures memory
#: is *separate* from the run that measures speed.
_MEMORY_WRAPPER = r'''
import os, runpy, sys, threading, time

# Resolve the runtime exactly as `_ecosystem_runner._import_torch` does, and do
# it here in the main thread: the sampler must not be the first thing to import
# `torch`, or the shim's "import Jittor before torch" rule is violated.
unmeasurable = None

if os.environ.get("VERIFY_RUNTIME") == "jittor":
    os.environ["JITTOR_TORCH_SHIM"] = "1"
    import jittor as jt
    import torch
    jt.flags.profile_memory_enable = 1
    if hasattr(jt.core, "device_memory_used"):
        device = int(jt.current_device())

        def read():
            return (int(jt.core.device_memory_used(device)),
                    int(jt.core.device_memory_reserved(device)))

        def reserved():
            return int(jt.core.device_memory_reserved(device))
    else:
        unmeasurable = (
            "jittor has no device_memory_used; its only counter "
            "(get_mem_info().total_cuda_used) is used+cached-free summed over "
            "every device, which cannot be compared with torch's "
            "max_memory_allocated")

    if hasattr(jt.core, "get_peak_device_used_memory"):
        # The peak has to come from the runtime, not from this thread. The
        # sampler below only runs when the interpreter releases the GIL -- which
        # during a step means only at device waits -- so on a fast step it
        # reports whichever intermediate value it happened to catch, while
        # torch's `max_memory_allocated` is maintained exactly by its allocator.
        # Measured on an 8-layer llama step: this sampler's max was 3406.5 MiB
        # and the runtime's own high-water was 7272 MiB, against torch's 6655.
        def peak():
            return int(jt.core.get_peak_device_used_memory(device))
    elif not unmeasurable:
        unmeasurable = (
            "this jittor predates get_peak_device_used_memory, whose absence "
            "leaves no exact peer for torch's max_memory_allocated -- a python "
            "sampler misses the peak on any fast step, so report no number "
            "rather than one that is not the peak")
else:
    os.environ.pop("JITTOR_TORCH_SHIM", None)
    import torch

    def read():
        return (int(torch.cuda.max_memory_allocated()),
                int(torch.cuda.max_memory_reserved()))

    def peak():
        return int(torch.cuda.max_memory_allocated())

    def reserved():
        return int(torch.cuda.max_memory_reserved())

pool = []
stop = threading.Event()


def sampler():
    # The sampler's only job is the pool: the peak comes from the runtime's own
    # high-water (`peak`, see above). A pool reading is still taken here rather
    # than read once at the end, because `memory_reserved` can *fall* when a
    # runtime releases cached blocks, so the high-water is not always the value
    # at the end. Sampling can only ever *miss*, never invent, which is why the
    # terminal read is unioned with these below.
    while not stop.is_set():
        try:
            _, b = read()
        except Exception:
            time.sleep(0.01)
            continue
        pool.append(b)
        time.sleep(0.01)


thread = threading.Thread(target=sampler, daemon=True)
thread.start()
runner, rest = sys.argv[1], sys.argv[2:]
sys.argv = [runner] + rest
try:
    runpy.run_path(runner, run_name="__main__")
finally:
    stop.set()
    thread.join(timeout=2)
    if unmeasurable:
        print("MEMORY_PEAK_BYTES -1")
        print("MEMORY_RESERVED_BYTES -1")
        print("MEMORY_UNMEASURABLE " + unmeasurable)
    else:
        # Both numbers now come from the runtime rather than from `max(live)` and
        # `max(pool)`. The peak has to -- this thread only runs when the
        # interpreter releases the GIL, so it misses it on any step with short
        # device waits. The pool's terminal reading is unioned in for the same
        # reason: on a 5-step probe the sampler caught the pool at 512 MiB while
        # the runtime held 1536 MiB.
        print("MEMORY_PEAK_BYTES %d" % peak())
        print("MEMORY_RESERVED_BYTES %d" % max(pool + [reserved()]))
'''


def _cases_module():
    """Import `_ecosystem_cases` from the runner's directory."""
    sys.path.insert(0, str(RUNNER.parent))
    import _ecosystem_cases  # noqa: E402

    return _ecosystem_cases


def _run_case(runner_python, case, out_npz, env, device, repeats, seed,
              weights=None):
    """Run one case in one runtime; return (result_dict, wall_s)."""
    cmd = [runner_python, str(RUNNER), case, str(out_npz),
           "--runtime", "jittor" if weights else "torch",
           "--device", device, "--repeats", str(repeats), "--seed", str(seed)]
    if weights:
        cmd += ["--weights", str(weights)]
    started = time.perf_counter()
    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    wall = time.perf_counter() - started
    match = RESULT_RE.search(proc.stdout or "")
    if match is None:
        tail = "\n".join((proc.stdout or "").splitlines()[-15:])
        raise SystemExit("[verify] %s failed (exit %s):\n%s"
                         % (case, proc.returncode, tail))
    return json.loads(match.group(1)), wall


def _peak_memory_bytes(runner_python, runtime, case, env, device, seed, npz):
    """Device memory for one case in one runtime.

    Returns ``{"live": bytes, "reserved": bytes, "note": str|None}``. ``live`` is
    the axis the two runtimes can be compared on; ``reserved`` is each runtime's
    own pool and is reported beside it, never instead of it. Either is -1 when
    the runtime could not answer, with ``note`` saying why.
    """
    cmd = [runner_python, "-c", _MEMORY_WRAPPER, str(RUNNER), case, str(npz),
           "--runtime", runtime, "--device", device, "--repeats", "1",
           "--seed", str(seed)]
    run_env = dict(env, VERIFY_RUNTIME=runtime)
    proc = subprocess.run(cmd, env=run_env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    text = proc.stdout or ""
    peak = PEAK_RE.search(text)
    reserved = RESERVED_RE.search(text)
    note = UNMEASURABLE_RE.search(text)
    return {
        "live": int(peak.group(1)) if peak else -1,
        "reserved": int(reserved.group(1)) if reserved else -1,
        "note": note.group(1) if note else None,
    }


def _compare(reference_path, candidate_path):
    """Max abs error, and max error relative to the whole field's magnitude.

    The relative figure is divided by the largest magnitude in the *entire*
    reference field, not by each array's own scale. A near-zero gradient would
    otherwise report a relative error of ~1 while agreeing to 1e-5 absolute,
    which is the classic way a correct run gets read as a failure.
    """
    import numpy as np

    ref = np.load(reference_path)
    cand = np.load(candidate_path)
    rows, worst_abs, worst_rel, worst_key, worst_abs_key = [], 0.0, 0.0, "-", "-"
    global_scale = 1e-12
    for key in ref.files:
        if key in cand.files:
            global_scale = max(global_scale,
                               float(np.abs(ref[key].astype("float64")).max()))
    for key in sorted(set(ref.files) & set(cand.files)):
        a, b = ref[key], cand[key]
        if a.shape != b.shape:
            rows.append((key, "shape %s vs %s" % (a.shape, b.shape), float("nan")))
            continue
        diff = np.abs(a.astype("float64") - b.astype("float64"))
        abs_err = float(diff.max()) if diff.size else 0.0
        rel_err = abs_err / global_scale
        rows.append((key, abs_err, rel_err))
        if abs_err > worst_abs:
            worst_abs, worst_abs_key = abs_err, key
        if rel_err > worst_rel:
            worst_rel, worst_key = rel_err, key
    missing = sorted(set(ref.files) ^ set(cand.files))
    return rows, worst_abs, worst_rel, worst_key, worst_abs_key, missing


def _case_list(cases_module, repo):
    """Cases whose name mentions the repo, with their required distributions.

    Case names use underscores where the distribution uses dashes (``ms-swift``
    registers ``ms_swift_lora_llama``), so normalise before matching.
    """
    key = repo.replace("-", "_")
    names = [n for n in cases_module.CASES if n.startswith(key + "_")
             or n.startswith("large_" + key + "_")]
    return [(n, cases_module.CASES[n][1]) for n in sorted(names)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--cases", default="")
    parser.add_argument("--oracle-python",
                        default=os.environ.get("REAL_TORCH_PYTHON", ""))
    parser.add_argument("--shim-python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="")
    parser.add_argument("--no-memory", dest="memory", action="store_false",
                        help="skip the separate peak-memory pass")
    parser.add_argument("--list-only", action="store_true")
    options = parser.parse_args()

    cases_module = _cases_module()
    available = _case_list(cases_module, options.repo)
    if options.list_only:
        for name, deps in available:
            print("CASE %-28s requires %s" % (name, ",".join(deps) or "-"))
        return 0
    if not options.out:
        raise SystemExit("--out is required")
    if not options.oracle_python:
        raise SystemExit("--oracle-python or REAL_TORCH_PYTHON is required; "
                         "comparing the shim against itself proves nothing")
    wanted = [c for c in options.cases.split(",") if c]
    chosen = [(n, d) for n, d in available if not wanted or n in wanted]
    if not chosen:
        raise SystemExit("no cases matched for %s" % options.repo)

    out_dir = Path(options.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    report = {"repo": options.repo, "device": options.device,
              "repeats": options.repeats, "cases": []}
    for name, deps in chosen:
        entry = {"case": name, "requires": list(deps)}
        try:
            ref_npz = out_dir / ("%s.ref.npz" % name)
            ref, _ = _run_case(
                options.oracle_python, name, ref_npz, env, options.device,
                options.repeats, options.seed)
            weights = Path(str(ref_npz)[:-len(".npz")] + ".weights.npz")
            cand_npz = out_dir / ("%s.shim.npz" % name)
            cand, _ = _run_case(
                options.shim_python, name, cand_npz, env, options.device,
                options.repeats, options.seed, weights=weights)
            rows, worst_abs, worst_rel, worst_key, worst_abs_key, missing = (
                _compare(ref_npz, cand_npz))
            mem = {"live": -1, "reserved": -1, "note": None}
            ref_mem = cand_mem = mem
            if options.memory:
                ref_mem = _peak_memory_bytes(
                    options.oracle_python, "torch", name, env, options.device,
                    options.seed, out_dir / ("%s.ref.mem.npz" % name))
                cand_mem = _peak_memory_bytes(
                    options.shim_python, "jittor", name, env, options.device,
                    options.seed, out_dir / ("%s.shim.mem.npz" % name))
            entry.update({
                "status": "ran",
                "tensors": len(rows),
                "missing": missing,
                "worst_abs_err": worst_abs,
                "worst_abs_key": worst_abs_key,
                "worst_rel_err_vs_field": worst_rel,
                "worst_rel_key": worst_key,
                "torch_seconds": ref.get("seconds"),
                "jittor_seconds": cand.get("seconds"),
                "speed_ratio": (cand.get("seconds") / ref["seconds"]
                                if ref.get("seconds") else None),
                "torch_peak_bytes": ref_mem["live"],
                "torch_reserved_bytes": ref_mem["reserved"],
                "jittor_peak_bytes": cand_mem["live"],
                "jittor_reserved_bytes": cand_mem["reserved"],
                "memory_note": cand_mem["note"] or ref_mem["note"],
                "jittor_fallback_count": cand.get("fallback_count"),
                "device_agreement": ref.get("device") == cand.get("device"),
            })
        except SystemExit as exc:
            entry.update({"status": "failed", "error": str(exc)})
        report["cases"].append(entry)
        print("[verify] " + json.dumps(entry, ensure_ascii=False), flush=True)

    (out_dir / "verify-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("[verify] wrote %s" % (out_dir / "verify-report.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
