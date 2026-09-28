#!/usr/bin/env python
"""Compare Jittor's Torch compatibility layer against PyTorch on real models.

Each workload in ``workloads.py`` is plain PyTorch code. It runs twice, in two
fresh interpreters: ``--torch-python`` (an independent binary PyTorch) and
``--jittor-python`` (whose ``torch`` is Jittor's shim, from this checkout).

    python bench/torch_compat/run.py \\
        --torch-python  /path/to/torch-env/bin/python \\
        --jittor-python /path/to/jittor-env/bin/python

    python bench/torch_compat/run.py --workloads qwen3_decode,sd15_sample
    python bench/torch_compat/run.py --size tiny          # harness self-check
    python bench/torch_compat/run.py --compile none,reduce-overhead
    python bench/torch_compat/run.py --list

Ratio = Jittor time / PyTorch time; below 1 means Jittor is faster. Results go
to ``$JITTOR_LAB_ROOT/_state/bench-torch-compat/<stamp>/`` as ``results.json``
and ``results.md``; ``report.py`` re-renders or compares saved runs.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))

from report import render_markdown, render_table  # noqa: E402

MARKER = "BENCH_RESULT "

#: Variables that would let the reference interpreter import Jittor's facade,
#: or point Jittor at another checkout's cache. Cleared for the torch side.
JITTOR_VARIABLES = (
    "JITTOR_SOURCE_ROOT", "JITTOR_HOME", "JITTOR_TORCH_CACHE_ROOT",
    "JITTOR_TORCH_SHIM", "JITTOR_TORCH_KEEP_HOME", "JT_BACKEND", "JT_USE_CUDA",
    "use_cuda", "cache_name",
)


def lab_root():
    root = os.environ.get("JITTOR_LAB_ROOT")
    return Path(root) if root else REPO.parent / "jittor-lab"


def list_workloads():
    # Parsed, not imported: listing must not need transformers or diffusers.
    import ast

    tree = ast.parse((HERE / "workloads.py").read_text())
    rows = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        fields = {}
        for item in node.body:
            if (isinstance(item, ast.Assign) and len(item.targets) == 1
                    and isinstance(item.targets[0], ast.Name)):
                try:
                    fields[item.targets[0].id] = ast.literal_eval(item.value)
                except ValueError:
                    pass
        if fields.get("name"):
            rows.append(fields)
    return rows


def environment_for(runtime, options, state):
    env = os.environ.copy()
    if runtime == "torch":
        env["PYTHONPATH"] = ""
        for name in JITTOR_VARIABLES + ("JT_BUILD_NVCC_PATH", "nvcc_path"):
            env.pop(name, None)
    else:
        # Pin this checkout, and a cache that no test run shares.
        env["PYTHONPATH"] = str(REPO / "python")
        env["JITTOR_HOME"] = str(state / "jittor-home")
        env["JITTOR_TORCH_CACHE_ROOT"] = str(state / "torch-shim")
        env["cache_name"] = "bench-torch-compat"
    if options.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = options.gpu
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env["TMPDIR"] = str(state / "tmp")
    return env


def run_one(workload, runtime, options, state, log_dir, mode="none"):
    python = options.torch_python if runtime == "torch" else options.jittor_python
    command = [python, str(HERE / "worker.py"), workload, "--runtime", runtime,
               "--device", options.device, "--size", options.size,
               "--seed", str(options.seed), "--compile", mode]
    for flag, value in (("--dtype", options.dtype),
                        ("--warmup", options.warmup),
                        ("--repeats", options.repeats),
                        ("--batch", options.batch)):
        if value is not None:
            command += [flag, str(value)]
    if not options.tf32:
        command.append("--no-tf32")
    if options.cudnn_benchmark:
        command.append("--cudnn-benchmark")
    if options.allow_fallback:
        command.append("--allow-fallback")

    suffix = "" if mode == "none" else "." + mode
    log = log_dir / ("%s.%s%s.log" % (workload, runtime, suffix))
    print("  %-8s %-16s %s ..." % (runtime, mode, workload), end="", flush=True)
    try:
        completed = subprocess.run(
            command, env=environment_for(runtime, options, state),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", timeout=options.timeout)
        output, code = completed.stdout, completed.returncode
    except subprocess.TimeoutExpired as expired:
        output = expired.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", "replace")
        code = None
    log.write_text(output)

    result = None
    for line in reversed(output.splitlines()):
        if line.startswith(MARKER):
            result = json.loads(line[len(MARKER):])
            break
    if result is None:
        result = {"workload": workload, "runtime": runtime, "compile": mode,
                  "status": "timeout" if code is None else "crash",
                  "error": "\n".join(output.splitlines()[-15:])}
    result["log"] = str(log)
    result["command"] = command
    status = result.get("status")
    if status == "ok":
        print(" %.4f s/step" % result["median_s"])
    else:
        print(" %s" % status.upper())
    return result


def git(*args):
    try:
        return subprocess.run(["git", "-C", str(REPO)] + list(args),
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def gpu_name(options):
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version",
             "--format=csv,noheader"], stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    rows = [row.strip() for row in query.splitlines() if row.strip()]
    index = int((options.gpu or "0").split(",")[0]) if rows else 0
    return rows[index] if index < len(rows) else (rows[0] if rows else None)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workloads", default="all",
                        help="comma-separated names, a family, or 'all'")
    parser.add_argument("--runtimes", default="torch,jittor")
    parser.add_argument("--torch-python",
                        default=os.environ.get("REAL_TORCH_PYTHON"))
    parser.add_argument("--jittor-python", default=sys.executable)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--gpu", default=None,
                        help="CUDA_VISIBLE_DEVICES for both runtimes")
    parser.add_argument("--dtype", default=None,
                        help="override every workload's default dtype")
    parser.add_argument("--size", choices=("full", "tiny"), default="full")
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None,
                        help="override every selected workload's batch size "
                             "(both runtimes), e.g. to fit a smaller card")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-tf32", dest="tf32", action="store_false")
    parser.add_argument("--cudnn-benchmark", action="store_true")
    parser.add_argument("--allow-fallback", action="store_true",
                        help="report Jittor CPU fallbacks instead of failing")
    parser.add_argument("--compile", default="none",
                        help="comma-separated torch.compile modes to run each "
                             "workload under, e.g. none,reduce-overhead")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--out", default=None, help="result directory")
    parser.add_argument("--label", default=None)
    parser.add_argument("--list", action="store_true")
    options = parser.parse_args()

    catalogue = list_workloads()
    if options.list:
        for row in catalogue:
            print("%-18s %-6s %-9s %s" % (row["name"], row.get("mode", ""),
                                          row.get("default_dtype", "float32"),
                                          row.get("description", "")))
        return 0

    names = [row["name"] for row in catalogue]
    if options.workloads == "all":
        selected = names
    else:
        selected = []
        for token in options.workloads.split(","):
            token = token.strip()
            match = [n for n in names if n == token or n.startswith(token + "_")]
            if not match:
                parser.error("unknown workload %r; see --list" % token)
            selected += [n for n in match if n not in selected]
    runtimes = [r.strip() for r in options.runtimes.split(",") if r.strip()]
    if "torch" in runtimes and not options.torch_python:
        parser.error("--torch-python (or REAL_TORCH_PYTHON) is required")

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(options.out) if options.out else (
        lab_root() / "_state" / "bench-torch-compat" / stamp)
    state = lab_root() / "_state" / "bench-torch-compat" / ("cache-" + options.size)
    (state / "tmp").mkdir(parents=True, exist_ok=True)
    log_dir = out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "label": options.label,
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "gpu": gpu_name(options) if options.device == "cuda" else None,
        "device": options.device, "size": options.size, "tf32": options.tf32,
        "batch": options.batch,
        "compile": options.compile,
        "cudnn_benchmark": options.cudnn_benchmark,
        "torch_python": options.torch_python,
        "jittor_python": options.jittor_python,
    }
    print("commit %s%s  gpu %s  size %s  -> %s" % (
        (meta["commit"] or "?")[:10], " (dirty)" if meta["dirty"] else "",
        meta["gpu"], options.size, out))

    modes = [m.strip() for m in options.compile.split(",") if m.strip()]
    results = []
    for workload in selected:
        for mode in modes:
            for runtime in runtimes:
                results.append(run_one(workload, runtime, options, state,
                                       log_dir, mode))
                (out / "results.json").write_text(json.dumps(
                    {"meta": meta, "results": results}, indent=1))

    meta["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
    document = {"meta": meta, "results": results}
    (out / "results.json").write_text(json.dumps(document, indent=1))
    (out / "results.md").write_text(render_markdown(document))
    print()
    print(render_table(document))
    print("\nsaved %s" % (out / "results.json"))
    failed = [r for r in results if r.get("status") != "ok"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
