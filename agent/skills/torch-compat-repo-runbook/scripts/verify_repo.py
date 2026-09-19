"""Four-axis verification of a per-repo torch-compat runbook.

A runbook that has not been run is a hypothesis. This driver executes the
repo's registered ecosystem cases on **both** runtimes and reports the four
axes the team signs off on:

* 支持的模型/case 清单 -- which cases the repo covers, and whether each ran;
* 精度 -- max abs / max scaled-relative error of every output and gradient
  against the independent PyTorch oracle, from the *same* weights and inputs;
* 显存 -- peak device memory of the child process, sampled externally so the
  number means the same thing on both runtimes;
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
* Peak memory is asked of each runtime, not sampled externally: on this box
  `nvidia-smi --query-compute-apps` does not list the process, and per-GPU
  `memory.used` includes co-tenants. Real PyTorch reports its true peak; Jittor
  needs `profile_memory_enable` and reports *current* device use, which a
  sampler thread turns into a peak. Jittor's profiling can perturb timing, so
  the memory pass is a separate run from the timing pass.
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

#: Runs a case while sampling the runtime's own device-memory accounting.
#:
#: Neither external sampler works on this box: `nvidia-smi --query-compute-apps`
#: does not list the process (container pid mapping), and per-GPU `memory.used`
#: includes co-tenants. So ask each runtime instead. Real PyTorch knows its true
#: peak; Jittor needs `profile_memory_enable` and then reports current device use,
#: which a sampler thread turns into a peak. Jittor's memory profiling can
#: perturb timing, so the run that measures memory is *separate* from the run
#: that measures speed.
_MEMORY_WRAPPER = r'''
import os, runpy, sys, threading, time

# Resolve the runtime exactly as `_ecosystem_runner._import_torch` does, and do
# it here in the main thread: the sampler must not be the first thing to import
# `torch`, or the shim's "import Jittor before torch" rule is violated.
if os.environ.get("VERIFY_RUNTIME") == "jittor":
    os.environ["JITTOR_TORCH_SHIM"] = "1"
    import jittor as jt
    import torch
    jt.flags.profile_memory_enable = 1

    def read():
        return int(jt.get_mem_info().total_cuda_used)
else:
    os.environ.pop("JITTOR_TORCH_SHIM", None)
    import torch

    def read():
        return int(torch.cuda.max_memory_allocated())

peak = []
stop = threading.Event()


def sampler():
    while not stop.is_set():
        try:
            peak.append(read())
        except Exception:
            pass
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
    print("MEMORY_PEAK_BYTES %d" % (max(peak) if peak else -1))
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
    """Peak device memory in bytes for one case in one runtime, or -1."""
    cmd = [runner_python, "-c", _MEMORY_WRAPPER, str(RUNNER), case, str(npz),
           "--runtime", runtime, "--device", device, "--repeats", "1",
           "--seed", str(seed)]
    run_env = dict(env, VERIFY_RUNTIME=runtime)
    proc = subprocess.run(cmd, env=run_env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    match = PEAK_RE.search(proc.stdout or "")
    return int(match.group(1)) if match else -1


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
            ref_mem = cand_mem = -1
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
                "torch_peak_bytes": ref_mem,
                "jittor_peak_bytes": cand_mem,
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
