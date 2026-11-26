"""Compare CUDA model workloads on Jittor and PyTorch.

Reference/candidate versions:
- Reference: PyTorch 2.11+cu128 (e.g. ``torch==2.11.0+cu128``, ``torchvision==0.26.0+cu128``)
- Candidate: Jittor 2.0.2 with CUDA 12.8 (``nvcc 12.8.x``, ``cudnn 9.x``)

Use separate Python environments with matching versions of NumPy, Diffusers
(validated with 0.35.1), Transformers, Accelerate, and Safetensors. The video
case also needs ftfy. Install binary PyTorch in the reference environment and
Jittor with jittor-compat in the candidate environment.

    python benchmarks/torch_compat/benchmark_cuda_models.py \
        --checkpoint /models/Wan2.2-TI2V-5B-Diffusers \
        --torch-python /envs/pytorch/bin/python --gpu 0

Use --list to see the presets. Select one or more with --case. ti2v-480 is
the default, and --case wan selects the five Wan presets.
ti2v-4400 measures the full denoiser at a longer sequence. ti2v-video measures
256 by 384, 17 frames, and eight steps with guidance scale 5. It also records
text encoding separately. t2v-a14b-block and i2v-a14b-block use the requested
transformer/config.json with one block at 2,240 tokens. These cases use
deterministic synthetic weights by default and accept shared local Safetensors
fixtures through --t2v-weights and --i2v-weights.

The reference measures torch.compile(mode="reduce-overhead") before eager PyTorch.
The faster validated reference is used for the comparison. The default Jittor mode follows the
measured presets, with reduce-overhead for short inference and video and
default execution for ti2v-4400. Set --jittor-mode default for an eager API
comparison. CUDA graph launches and refusals are recorded in results.json.
Each denoiser gets two independent process rounds. Video gets two steady
samples after three warmups. Loading and warmup have a separate timeout from
the measurement window. Use an isolated --cache-dir to reuse Jittor builds
across serial benchmark invocations.

--case stock selects the 14 existing Torch compatibility workloads, including
Qwen3, Stable Diffusion, DDPM, ResNet, ViT and BERT. --case stock-dual selects
the six two-GPU DDP training variants. Stock and ecosystem
inference measure Jittor replay first and eager next where both are supported.
The comparison uses the fastest result that passes the output checks.
Deterministic stock inference checks every output over one input-pool cycle.
--case training selects
the 53 CV and NLP training configurations from the RTX 4090 matrix. Training
uses the pinned jittor-accelerate benchmarking tools and a matching upstream
Accelerate checkout. Supply the tools directory from jittor-accelerate
1947ffbb and its pinned Accelerate checkout via --training-tools and
--accelerate-checkout.
The training fixtures directory holds <model>/initial.safetensors and, for NLP,
<model>/model/ with the local checkpoint. Existing metadata.json fixtures
with model and initial_state paths also work. Supply local Pets images via
--cv-data and a cached MRPC dataset with its GLUE metric via the Hugging Face
cache and --nlp-metric. Use --gpu 0,1 for the two-rank cases.

Training repeats one real NLP batch per rank or uses batches of real CV
images. NLP training disables stochastic layers in all modes for loss parity.
Eight warmup steps precede 16 timed steps by default. Training reports
mean step time using the slowest rank per step. Wan and the stock workloads
report median latency. Results and rank logs are saved outside the checkout.
A single selection defaults to two process rounds. Stock and training cases
run once each when selected as a suite. Video uses two samples in one process.

Use --case ecosystem for inference presets spanning CV, NLP, generation,
audio, graphs and forecasting. --list prints the scope of these presets.
Synthetic input shapes and deterministic weights are used for architecture
presets. Supply local snapshots via --model-assets for checkpoint presets.
With a reference Python, image and text processors run once on the host and
both CUDA workers read the same input tensors. Preprocessing is outside timing.
Ecosystem eager runs suspend automatic graph replay to provide an eager
baseline. --model-mode reduce-overhead explicitly measures replay.
"""

import argparse
from collections.abc import Mapping
from contextlib import nullcontext
import ctypes
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import itertools
import json
import math
import os
from pathlib import Path
import signal
import socket
from statistics import mean, median
import subprocess
import sys
import tempfile
from time import monotonic, perf_counter

from run import lab_root, list_workloads
from model_cases import MODEL_CASES, Inputs, needs_processor, prepare_model, processor_batch
from worker import import_torch


REPO = Path(__file__).resolve().parents[2]
WAN_CASES = {
    "ti2v-480": ((5, 16, 24), 6, 12),
    "ti2v-4400": ((5, 44, 80), 5, 8),
    "ti2v-video": (None, 3, 2),
    "t2v-a14b-block": ((5, 32, 56), 6, 12),
    "i2v-a14b-block": ((5, 32, 56), 6, 12),
}
VIDEO = "ti2v-video"
CUDA_DEVICE_ORDINAL = 9
TRAINING_LOSS_ATOL = 0.02
TRAINING_LOSS_RTOL = 0.03
CV_MODELS = {
    16: (
        "convnext_xlarge.fb_in22k_ft_in1k", "convnextv2_huge.fcmae_ft_in1k",
        "vit_large_patch16_224.augreg_in21k_ft_in1k", "vit_huge_patch14_224.orig_in21k",
        "swin_large_patch4_window7_224.ms_in22k_ft_in1k",
    ),
    64: (
        "convnext_tiny.fb_in1k", "efficientnet_b0.ra_in1k", "mobilenetv3_large_100.ra_in1k",
        "resnet18.a1_in1k", "swin_tiny_patch4_window7_224.ms_in1k",
        "vit_tiny_patch16_224.augreg_in21k_ft_in1k", "convnext_base.fb_in22k_ft_in1k",
        "resnet101.a1h_in1k", "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
        "vit_base_patch16_224.augreg_in21k_ft_in1k", "resnet50d",
    ),
}
NLP_MODELS = {
    4: ("bert-large-cased", "roberta-large"),
    16: (
        "albert-base-v2", "deberta-v3-small", "distilbert-base-cased", "gpt2",
        "roberta-base", "t5-small", "bert-base-cased",
    ),
}
TRAINING_CASES = {
    f"{model}-r{ranks}-b{batch}": (kind, model, ranks, batch)
    for kind, groups in (("cv", CV_MODELS), ("nlp", NLP_MODELS))
    for batch, models in groups.items()
    for model in models
    for ranks in (1, 2)
}
TRAINING_CASES.update({
    f"deberta-v3-large-r{ranks}-b{batch}": ("nlp", "deberta-v3-large", ranks, batch)
    for ranks in (1, 2)
    for batch in (1, 2)
})
EAGER_TRAINING = {
    "bert-large-cased-r2-b4", "roberta-large-r2-b4",
    # T5's training step reads back tensor values and cannot be replayed.
    "t5-small-r1-b16", "t5-small-r2-b16",
}


def parse_args():
    stock_cases = {case["name"]: case for case in list_workloads()}
    stock_single = list(stock_cases)
    stock_dual = {
        f"{name}-r2": dict(case, workload=name, ranks=2)
        for name, case in stock_cases.items() if case["mode"] == "train"
    }
    stock_cases.update(stock_dual)
    groups = {"wan": list(WAN_CASES), "stock": stock_single, "stock-dual": list(stock_dual),
              "training": list(TRAINING_CASES)}
    groups["ecosystem"] = list(MODEL_CASES)
    groups["all"] = list(itertools.chain.from_iterable(groups.values()))
    for name, case in MODEL_CASES.items():
        groups.setdefault(case.group, []).append(name)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", action="append", metavar="NAME",
                        help="a preset or group, including wan, stock, stock-dual, training, ecosystem, all")
    parser.add_argument("--list", action="store_true", help="list presets without loading model dependencies")
    parser.add_argument("--checkpoint", type=Path, help="local TI2V-5B Diffusers snapshot")
    for variant in ("t2v", "i2v"):
        parser.add_argument(f"--{variant}-config", type=Path,
                            help=f"{variant.upper()}-A14B transformer/config.json")
        parser.add_argument(f"--{variant}-weights", type=Path,
                            help="optional one-block Safetensors fixture")
    parser.add_argument("--torch-python", default=os.environ.get("REAL_TORCH_PYTHON"))
    parser.add_argument("--jittor-python", default=sys.executable)
    parser.add_argument("--source-root", type=Path, default=REPO,
                        help="Jittor checkout used by the candidate worker")
    parser.add_argument("--runtimes", choices=("torch,jittor", "torch", "jittor"),
                        default="torch,jittor")
    parser.add_argument("--torch-mode", choices=("both", "eager", "compiled"), default="both",
                        help="native PyTorch reference modes to measure")
    parser.add_argument("--jittor-mode", choices=("preset", "default", "reduce-overhead"),
                        default="preset", help="execution mode for the Wan cases")
    parser.add_argument("--stock-size", choices=("full", "tiny"), default="full")
    parser.add_argument("--stock-mode", choices=("both", "reduce-overhead", "eager"), default="both",
                        help="Jittor execution mode for stock inference workloads")
    parser.add_argument("--training-tools", type=Path,
                        default=REPO / "benchmarks/torch_compat/training_tools",
                        help="directory containing bench_cv.py and bench_nlp.py")
    parser.add_argument("--accelerate-checkout", type=Path, help="the upstream Accelerate checkout")
    parser.add_argument("--training-fixtures", type=Path, help="local training snapshots by model name")
    parser.add_argument("--cv-data", type=Path, help="Oxford-IIIT Pet image directory")
    parser.add_argument("--nlp-metric", type=Path, help="local GLUE metric script")
    parser.add_argument("--model-assets", type=Path, help="directory containing local model assets by preset name")
    parser.add_argument("--model-mode", choices=("both", "reduce-overhead", "eager"), default="both",
                        help="Jittor execution mode for ecosystem presets")
    parser.add_argument("--profile-model", action="store_true", help="profile a separate ecosystem inference call")
    parser.add_argument("--gpu", help="comma-separated physical CUDA indices or UUIDs")
    parser.add_argument("--warmup", type=int, help="override complete warmup calls per worker")
    parser.add_argument("--samples", type=int, help="override measured calls per worker")
    parser.add_argument("--rounds", type=int, help="fresh process rounds per runtime")
    parser.add_argument("--measurement-timeout", type=float, default=300,
                        help="steady-state budget in seconds per round across the selected runtimes")
    parser.add_argument("--setup-timeout", type=float, default=1800)
    parser.add_argument("--decode-workspace-ratio", type=float, default=0.06,
                        help="cuDNN workspace search fraction for Jittor video replay")
    parser.add_argument("--seed", type=int, help="override the Wan or stock input seed")
    parser.add_argument("--output-dir", type=Path, help="new result directory outside the checkout")
    parser.add_argument("--cache-dir", type=Path, help="isolated Jittor benchmark cache")
    args = parser.parse_args()
    if args.list:
        for group in ("wan", "stock", "stock-dual", "training"):
            for name in groups[group]:
                print(f"{group:<9} {name}")
        for name, case in MODEL_CASES.items():
            scope = f"{case.scope}, {case.dtype}"
            if case.checkpoint:
                scope += f", checkpoint {case.checkpoint}"
            print(f"{case.group:<9} {name}  [{scope}]")
        parser.exit()
    if any(n is not None and n < 1 for n in (args.warmup, args.samples, args.rounds)):
        parser.error("warmup, samples and rounds must be positive")
    if not 0 < args.decode_workspace_ratio <= 1:
        parser.error("--decode-workspace-ratio must be in (0, 1]")
    if any(n <= 0 or not math.isfinite(n) for n in (args.setup_timeout, args.measurement_timeout)):
        parser.error("timeouts must be positive and finite")
    if "torch" in args.runtimes and not args.torch_python:
        parser.error("--torch-python or REAL_TORCH_PYTHON is required for the reference")
    if "jittor" in args.runtimes and not (args.source_root / "python/jittor/__init__.py").is_file():
        parser.error("--source-root must point to a Jittor checkout")
    expanded = []
    for name in args.case or ["ti2v-480"]:
        if name not in groups and name not in groups["all"]:
            parser.error(f"unknown case {name!r}, use --list")
        expanded.extend(groups.get(name, [name]))
    args.case = list(dict.fromkeys(expanded))
    args.stock_cases = stock_cases
    for name in args.case:
        if name.startswith("ti2v") and (args.checkpoint is None or not args.checkpoint.is_dir()):
            parser.error(f"{name} requires --checkpoint with a local Diffusers snapshot")
        if name.endswith("-block"):
            variant = name.split("-")[0]
            config = getattr(args, f"{variant}_config")
            weights = getattr(args, f"{variant}_weights")
            if config is None or not config.is_file():
                parser.error(f"{name} requires --{variant}-config")
            if weights is not None and not weights.is_file():
                parser.error(f"--{variant}-weights must point to a Safetensors file")
    if any(name in TRAINING_CASES for name in args.case):
        for name in ("training_tools", "accelerate_checkout", "training_fixtures"):
            if getattr(args, name) is None or not getattr(args, name).is_dir():
                parser.error(f"training cases require --{name.replace('_', '-')}")
        kinds = {TRAINING_CASES[name][0] for name in args.case if name in TRAINING_CASES}
        if "cv" in kinds and (args.cv_data is None or not args.cv_data.is_dir()):
            parser.error("CV training requires --cv-data")
        if "nlp" in kinds and (args.nlp_metric is None or not args.nlp_metric.is_file()):
            parser.error("NLP training requires --nlp-metric")
    return args


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def revision(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def worker_env(args, output, runtime):
    env = os.environ.copy()
    env.pop("JITTOR_TORCH_SHIM", None)
    env.pop("JITTOR_BENCH_JIT_UTILS_RESTARTED", None)
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
        env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    env.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_DATASETS_OFFLINE="1", HF_EVALUATE_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false", USE_TORCH_XLA="0", TMPDIR=str(args.ipc_root))
    env.setdefault("HF_HOME", str(output / "hf"))
    env.setdefault("TORCH_HOME", str(output / "torch-home"))
    env.setdefault("CUDA_CACHE_PATH", str(output / "cuda-cache"))
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(output / "inductor"))
    env.setdefault("TRITON_CACHE_DIR", str(output / "triton"))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env.setdefault(key, "4")
    if runtime == "jittor":
        search_path = [str(args.source_root.resolve() / "python")]
        if env.get("PYTHONPATH"):
            search_path.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(search_path)
        env["JITTOR_HOME"] = str(args.cache_dir.resolve() if args.cache_dir else output / "jittor-home")
        env["JITTOR_TORCH_RUNTIME_ROOT"] = str(output / "torch-shim")
        env.update(JITTOR_TORCH_KEEP_HOME="1", JITTOR_TORCH_KEEP_TMPDIR="1",
                   JT_BACKEND="cuda", JT_USE_CUDA="1")
        env["cache_name"] = "bench-wan22"
    return env


def make_job(args, name, runtime, round_id, mode=None):
    torch_mode = mode if runtime == "torch" else None
    job = {"case": name, "runtime": runtime, "round": round_id,
           "source_root": str(args.source_root.resolve()), "torch_mode": torch_mode,
           "jittor_mode": mode if runtime == "jittor" else None}
    inference_mode = "reduce-overhead" if mode == "compiled" else "eager"
    if name in TRAINING_CASES:
        return training_job(args, job)
    if name in MODEL_CASES:
        case = MODEL_CASES[name]
        assets = args.model_assets.resolve() if args.model_assets else None
        if case.checkpoint and (assets is None or not (assets / name).exists()):
            raise FileNotFoundError(f"{name} needs {case.checkpoint} at --model-assets/{name}")
        return dict(job, kind="model", mode=inference_mode, profile=args.profile_model,
                    group=case.group, scope=case.scope, model=case.model, model_assets=str(assets) if assets else None,
                    dtype=case.dtype, shape=list(case.shape), warmup=args.warmup or 6, samples=args.samples or 12,
                    seed=args.seed if args.seed is not None else 0)
    if name in args.stock_cases:
        stock = args.stock_cases[name]
        ranks = stock.get("ranks", 1)
        warmup, samples = (1, 2) if args.stock_size == "tiny" else (6, 12)
        return dict(job, kind="stock", stock_workload=stock.get("workload", name), ranks=ranks,
                    mode=inference_mode, size=args.stock_size,
                    seed=args.seed if args.seed is not None else 0,
                    warmup=args.warmup or warmup, samples=args.samples or samples)
    shape, warmup, samples = WAN_CASES[name]
    mode = args.jittor_mode
    if mode == "preset":
        mode = "default" if name == "ti2v-4400" else "reduce-overhead"
    variant = name.split("-")[0]
    checkpoint = args.checkpoint if variant == "ti2v" else None
    config = checkpoint / "transformer/config.json" if checkpoint else getattr(args, f"{variant}_config")
    weights = None if checkpoint else getattr(args, f"{variant}_weights")
    return dict(job, **{
        "kind": "wan",
        "mode": mode if runtime == "jittor" else inference_mode,
        "checkpoint": str(checkpoint.resolve()) if checkpoint else None,
        "config": str(config.resolve()),
        "weights": str(weights.resolve()) if weights else None,
        "shape": shape, "seed": args.seed if args.seed is not None else 20261123,
        "warmup": args.warmup or warmup, "samples": args.samples or samples,
        "input_slots": 1 if name == VIDEO or (name == "ti2v-4400" and mode == "default") else 3,
        "decode_workspace_ratio": args.decode_workspace_ratio,
    })


def training_job(args, job):
    family, model, ranks, batch = TRAINING_CASES[job["case"]]
    fixture = args.training_fixtures.resolve() / model
    metadata_file = fixture / "metadata.json"
    metadata = json.loads(metadata_file.read_text()) if metadata_file.is_file() else {}
    initial = fixture / "initial.safetensors"
    if not initial.is_file() and "initial_state" in metadata:
        initial = fixture / metadata["initial_state"]
    model_path = fixture / "model"
    if not model_path.is_dir() and "model" in metadata:
        model_path = fixture / metadata["model"]
    if not initial.is_file() or (family == "nlp" and not model_path.is_dir()):
        raise ValueError(f"Missing training checkpoint or state in {fixture}")
    eager = "eager" if family == "cv" else "step-eager"
    mode = "compiled-step" if job["runtime"] == "jittor" and job["case"] not in EAGER_TRAINING else eager
    if job["torch_mode"] == "compiled":
        mode = "compiled-step"
    return dict(job, kind="training", family=family, mode=mode, ranks=ranks, batch=batch,
                model=model if family == "cv" else str(model_path), initial_state=str(initial),
                training_tools=str(args.training_tools.resolve()), checkout=str(args.accelerate_checkout.resolve()),
                cv_data=str(args.cv_data.resolve()) if args.cv_data else None,
                metric=str(args.nlp_metric.resolve()) if args.nlp_metric else None,
                warmup=args.warmup or 8, samples=args.samples or 16)


def wait_for_workers(processes, directories, setup_timeout, measurement_timeout):
    measuring = False
    deadline = monotonic() + setup_timeout
    started = None
    finished = None
    while True:
        codes = [process.poll() for process in processes]
        now = monotonic()
        in_timing_window = (all((d / "warmup.done").exists() for d in directories)
                            and not all((d / "measurement.done").exists() for d in directories))
        if in_timing_window != measuring:
            measuring = in_timing_window
            deadline = now + (measurement_timeout if measuring else setup_timeout)
            if measuring:
                started = now
            else:
                finished = now
        elapsed = (finished or now) - started if started is not None else 0
        if any(code not in (None, 0) for code in codes):
            return "failed", elapsed
        if all(code == 0 for code in codes):
            return "ok", elapsed
        if now >= deadline:
            return ("measurement-timeout" if measuring else "worker-timeout"), elapsed
        try:
            next(process for process in processes if process.returncode is None).wait(timeout=0.25)
        except subprocess.TimeoutExpired:
            pass


def rank_env(env, job, directory, rank, port):
    ranks = job["ranks"]
    env.update(RANK=str(rank), WORLD_SIZE=str(ranks), LOCAL_WORLD_SIZE=str(ranks),
               MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), NVIDIA_TF32_OVERRIDE="0",
               TIMM_FUSED_ATTN="2")
    if ranks > 1:
        env["LOCAL_RANK"] = str(rank)
    else:
        env.pop("LOCAL_RANK", None)
    if (job["case"] == "qwen3_train-r2" and job["runtime"] == "torch"
            and "PYTORCH_ALLOC_CONF" not in env):
        # Two FP32 replicas at the preset batch fit on 24 GiB cards when the
        # allocator can grow existing segments between DDP warmup steps.
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if job["runtime"] == "jittor" and ranks > 1:
        env.update(
            JITTOR_HOME=str(Path(env["JITTOR_HOME"]) / f"rank{rank}"),
            JITTOR_TORCH_RUNTIME_ROOT=str(Path(env["JITTOR_TORCH_RUNTIME_ROOT"]) / f"rank{rank}"),
            JT_BUILD_USE_MPI="0", JT_BUILD_USE_NCCL="1", JITTOR_TORCH_DISTRIBUTED_AUTO_INIT="1",
            JT_NCCL_WORLD_SIZE=str(ranks), JT_NCCL_RANK=str(rank), JT_NCCL_LOCAL_RANK=str(rank),
            JT_NCCL_ROOTINFO_FILE=str(directory / "nccl.bin"), JITTOR_DIST_RENDEZVOUS_DIR=str(directory),
        )
    return env


def stop_workers(processes):
    for process in processes:
        if process.poll() is not None:
            continue
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        except ProcessLookupError:
            process.wait()


def _worker_oom_error(directories):
    for directory in directories:
        log_path = directory / "worker.log"
        try:
            with log_path.open("rb") as stream:
                stream.seek(max(0, log_path.stat().st_size - 8000))
                lines = stream.read().decode(errors="replace").splitlines()
        except OSError:
            continue
        for line in reversed(lines):
            lower = line.lower()
            if "outofmemoryerror" in lower or "cuda out of memory" in lower or "out of memory" in lower:
                return line.strip()[:350]
    return None


def run_one(args, output, job, budget):
    name, runtime = job["case"], job["runtime"]
    execution_label = "torch-compiled" if job.get("torch_mode") == "compiled" else runtime
    if job.get("jittor_mode"):
        execution_label = f"jittor-{job['jittor_mode']}"
    directory = output / f"{name}-{execution_label}-{job['round']}"
    directory.mkdir()
    if budget <= 0:
        return dict(job, status="measurement-timeout", directory=str(directory), measured_seconds=0)
    python = args.torch_python if runtime == "torch" else args.jittor_python
    ranks = job.get("ranks", 1)
    directories = [directory / f"rank{rank}" for rank in range(ranks)] if ranks > 1 else [directory]
    with socket.socket() as address:
        address.bind(("127.0.0.1", 0))
        port = address.getsockname()[1]
    processes = []
    try:
        for rank, workdir in enumerate(directories):
            workdir.mkdir(exist_ok=True)
            spec = workdir / "job.json"
            write_json(spec, job)
            command = [python, "-u", str(Path(__file__).resolve()), "--worker", str(spec)]
            env = worker_env(args, output, runtime)
            if job["kind"] == "training" or ranks > 1:
                env = rank_env(env, job, directory, rank, port)
            with (workdir / "worker.log").open("w") as log:
                processes.append(subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                  start_new_session=True))
        status, elapsed = wait_for_workers(processes, directories, args.setup_timeout, budget)
    finally:
        stop_workers(processes)
    paths = [workdir / "result.json" for workdir in directories]
    result = job.copy()
    if status == "ok" and all(path.is_file() for path in paths):
        records = [json.loads(path.read_text()) for path in paths]
        if job["kind"] == "training":
            result = aggregate_training(job, records)
        elif ranks > 1:
            result = aggregate_stock_training(records)
        else:
            result = records[0]
        elapsed = sum(sample["total_s"] for sample in result["measurements"])
        if elapsed > budget:
            status = "measurement-timeout"
    elif status == "ok":
        status = "missing-result"
    elif status == "failed":
        oom_line = _worker_oom_error(directories)
        if oom_line is not None:
            status = "oom"
            result["error"] = oom_line
    result.update(status=status, returncodes=[p.returncode for p in processes],
                  directory=str(directory), measured_seconds=elapsed)
    if status == "ok":
        statistic = "mean" if job["kind"] == "training" else "median"
        elapsed_ms = result[f"{statistic}_s"] * 1000
        print(f"{name} {runtime} {result['execution']} round {job['round']} "
              f"{elapsed_ms:.3f} ms {statistic}", flush=True)
    else:
        print(f"{name} {execution_label} round {job['round']} {status} ({directory})", flush=True)
    return result


def aggregate_training(job, records):
    samples = [max(step["total"] for step in steps) for steps in zip(*(row["records"] for row in records))]
    warm, measured = samples[:job["warmup"]], samples[job["warmup"]:]
    peak_alloc = max((row.get("peak_memory_bytes") or 0 for row in records), default=0)
    peak_reserved = max((row.get("peak_reserved_bytes") or 0 for row in records), default=0)
    strategy = next((row.get("strategy") for row in records if row.get("strategy")), None)
    if not strategy:
        strategy = "DDP" if job.get("ranks", 1) > 1 else "single-gpu"
    dtype = next((row.get("dtype") for row in records if row.get("dtype")), "float32")
    return dict(job, execution=job["mode"], mean_s=mean(measured), median_s=median(measured),
                warmups=[{"total_s": seconds} for seconds in warm],
                measurements=[{"total_s": seconds} for seconds in measured],
                compile_warmup_s=sum(warm), first_call_s=warm[0] if warm else None,
                peak_memory_bytes=peak_alloc or None,
                peak_reserved_bytes=peak_reserved or None,
                peak_vram_gb=round(peak_alloc / (1024 ** 3), 4) if peak_alloc else None,
                strategy=strategy, dtype=dtype,
                rank_results=records)


def aggregate_stock_training(records):
    result = records[0].copy()
    for key in ("warmups", "measurements"):
        samples = [row[key] for row in records]
        if len(set(map(len, samples))) != 1:
            raise RuntimeError(f"Stock training ranks disagree on the number of {key}")
        result[key] = [{"total_s": max(step["total_s"] for step in steps)} for steps in zip(*samples)]
    peak_alloc = max(row["peak_memory_bytes"] for row in records)
    modes = sorted({row["execution"] for row in records})
    return dict(result, execution=" / ".join(modes), strategy="DDP",
                median_s=median(step["total_s"] for step in result["measurements"]),
                compile_warmup_s=sum(step["total_s"] for step in result["warmups"]),
                first_call_s=result["warmups"][0]["total_s"],
                peak_memory_bytes=peak_alloc, peak_vram_gb=round(peak_alloc / (1024 ** 3), 4),
                peak_reserved_bytes=max(row["peak_reserved_bytes"] for row in records),
                rank_results=records)


def cuda_residency(torch, jt, tensor):
    if tensor.numel() == 0:
        if tensor.device.type != "cuda":
            raise RuntimeError("Empty tensor is on the wrong device")
        return {"device": str(tensor.device), "ordinal": None,
                "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
    if jt is not None:
        tensor.sync()
        if tensor.location() != "device" or jt.core.dispatch_context([tensor])[0] != "cuda":
            raise RuntimeError("Jittor tensor failed the CUDA residency check")
    driver = ctypes.CDLL("libcuda.so.1")
    ordinal = ctypes.c_int(-1)
    status = driver.cuPointerGetAttribute(
        ctypes.byref(ordinal), CUDA_DEVICE_ORDINAL, ctypes.c_void_p(tensor.data_ptr()),
    )
    if tensor.device.type != "cuda" or status != 0 or ordinal.value != torch.cuda.current_device():
        raise RuntimeError(f"Tensor has no allocation on the measured CUDA device ({status})")
    return {"device": str(tensor.device), "ordinal": ordinal.value,
            "shape": list(tensor.shape), "dtype": str(tensor.dtype)}


def prepare_denoiser(torch, np, job):
    from diffusers import WanTransformer3DModel

    config = json.loads(Path(job["config"]).read_text())
    if job["checkpoint"]:
        model = WanTransformer3DModel.from_pretrained(
            job["checkpoint"], subfolder="transformer", torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
    else:
        model = WanTransformer3DModel.from_config(config, num_layers=1)
        if job["weights"]:
            from safetensors.numpy import load_file

            weights = load_file(job["weights"])
            model.load_state_dict({name: torch.from_numpy(value) for name, value in weights.items()})
        else:
            from workloads import deterministic_init

            deterministic_init(torch, model, job["seed"])
        model = model.to(dtype=torch.bfloat16)
    model = model.eval().to("cuda")
    frames, height, width = job["shape"]
    pf, ph, pw = config["patch_size"]
    tokens = (frames // pf) * (height // ph) * (width // pw)
    latent_shape = (1, config["in_channels"], frames, height, width)
    rng = np.random.default_rng(job["seed"])
    latents = [rng.standard_normal(latent_shape, dtype=np.float32)]
    context = rng.standard_normal((1, 512, config["text_dim"]), dtype=np.float32)
    for _ in range(job["input_slots"] - 1):
        latents.append(rng.standard_normal(latent_shape, dtype=np.float32))
    encoder_states = torch.from_numpy(context).to(device="cuda", dtype=torch.bfloat16)
    latents = [torch.from_numpy(value).to(device="cuda", dtype=torch.bfloat16) for value in latents]
    time_shape = (1, tokens) if job["checkpoint"] else (1,)
    times = [torch.full(time_shape, 734 - 97 * index, dtype=torch.int64, device="cuda")
             for index in range(len(latents))]
    weight = next(model.parameters())
    if job["mode"] == "reduce-overhead":
        model = torch.compile(model, mode="reduce-overhead")

    def call(index):
        slot = index % len(latents)
        if job.get("torch_mode") == "compiled":
            torch.compiler.cudagraph_mark_step_begin()
        start = perf_counter()
        out = model(hidden_states=latents[slot], encoder_hidden_states=encoder_states,
                    timestep=times[slot], return_dict=False)[0]
        torch.cuda.synchronize()
        return {"denoised": out}, {"total_s": perf_counter() - start}

    return call, model, {"tokens": tokens}, latents[0], weight


def prepare_video(torch, jt, np, job):
    from diffusers import AutoencoderKLWan, WanPipeline

    checkpoint = job["checkpoint"]
    vae = AutoencoderKLWan.from_pretrained(
        checkpoint, subfolder="vae", torch_dtype=torch.float32, local_files_only=True,
    )
    pipe = WanPipeline.from_pretrained(
        checkpoint, vae=vae, torch_dtype=torch.bfloat16, local_files_only=True,
    )
    pipe.set_progress_bar_config(disable=True)
    pipe.text_encoder.to("cuda")
    prompt = {
        "prompt": "A red kite floats above a grassy meadow on a cloudy day, static camera.",
        "negative_prompt": "blurry, distorted, static noise",
        "device": torch.device("cuda"), "dtype": torch.bfloat16, "max_sequence_length": 512,
    }
    pipe.encode_prompt(**prompt)
    torch.cuda.synchronize()
    start = perf_counter()
    positive, negative = pipe.encode_prompt(**prompt)
    torch.cuda.synchronize()
    text_s = perf_counter() - start
    pipe.register_modules(text_encoder=None)
    gc.collect()
    torch.cuda.empty_cache()
    pipe.to("cuda")
    weight = next(pipe.transformer.parameters())
    cap_workspace = jt is not None and job["mode"] == "reduce-overhead"
    if job["mode"] == "reduce-overhead":
        if jt is None:
            pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead")
        else:
            # Step capture can replay the transformer's in-place storage
            # updates. Module graph replay refuses opaque writes.
            pipe.transformer.forward = torch.compile(pipe.transformer.forward, mode="reduce-overhead")
    if cap_workspace:
        jt.cudnn.set_max_workspace_ratio(job["decode_workspace_ratio"])
    height, width, frames, steps = 256, 384, 17, 8
    channels = vae.config.z_dim
    spatial = vae.config.scale_factor_spatial
    temporal = vae.config.scale_factor_temporal
    shape = (1, channels, (frames - 1) // temporal + 1, height // spatial, width // spatial)
    noise = np.random.default_rng(job["seed"]).standard_normal(shape, dtype=np.float32)
    scale = torch.tensor(vae.config.latents_std, dtype=torch.float32, device="cuda").view(1, channels, 1, 1, 1)
    mean = torch.tensor(vae.config.latents_mean, dtype=torch.float32, device="cuda").view(1, channels, 1, 1, 1)

    def call(_):
        if job.get("torch_mode") == "compiled":
            torch.compiler.cudagraph_mark_step_begin()
        latents = torch.from_numpy(noise.copy()).to("cuda")
        torch.cuda.synchronize()
        start = perf_counter()
        pred = pipe(prompt_embeds=positive, negative_prompt_embeds=negative, latents=latents,
                    output_type="latent", height=height, width=width, num_frames=frames,
                    num_inference_steps=steps, guidance_scale=5.0, return_dict=False)[0]
        torch.cuda.synchronize()
        midpoint = perf_counter()
        if cap_workspace:
            previous = jt.cudnn.get_benchmark()
            jt.cudnn.set_benchmark(1)
        try:
            video = vae.decode(pred.to(vae.dtype) * scale + mean, return_dict=False)[0]
            torch.cuda.synchronize()
        finally:
            if cap_workspace:
                jt.cudnn.set_benchmark(previous)
        end = perf_counter()
        return {"latents": pred, "video": video}, {
            "denoise_s": midpoint - start, "decode_s": end - midpoint, "total_s": end - start,
        }

    metadata = {"text_s": text_s, "height": height, "width": width, "frames": frames, "steps": steps,
                "workspace_ratio": job["decode_workspace_ratio"] if cap_workspace else None}
    return call, pipe.transformer, metadata, positive, weight


def checked_runtime(job):
    torch = import_torch(job["runtime"])
    jt = None
    guard = nullcontext()
    if job["runtime"] == "jittor":
        import jittor as jt
        from jittor._runtime.fallback import forbid_backend_fallbacks

        root = Path(job["source_root"])
        if Path(jt.__file__).resolve().parent != (root / "python/jittor").resolve():
            raise RuntimeError(f"Expected Jittor from {root}, got {jt.__file__}")
        job.update(jittor_sha=revision(root, "rev-parse", "HEAD"),
                   jittor_dirty=bool(revision(root, "status", "--porcelain")),
                   jittor_module=jt.__file__)
        if job["kind"] != "training":
            guard = forbid_backend_fallbacks()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this worker")
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    return torch, jt, guard


def warm_up(job, call, settle=None):
    warmups = [call(index)[1] for index in range(job["warmup"])]
    index = len(warmups)
    if settle is not None:
        from worker import MAX_EXTRA_WARMUP

        while not settle([step["total_s"] for step in warmups]) and index < job["warmup"] + MAX_EXTRA_WARMUP:
            warmups.append(call(index)[1])
            index += 1
    return warmups, index


def native_graphs(job):
    if job.get("torch_mode") != "compiled":
        return None
    from torch._dynamo.utils import counters

    return int(counters["stats"]["unique_graphs"])


def check_native_compile(job, warm_graphs, final_graphs):
    if warm_graphs is None:
        return
    if warm_graphs == 0:
        raise RuntimeError("Torch compiled worker produced no Dynamo graphs")
    if final_graphs != warm_graphs:
        raise RuntimeError("Torch compiled worker traced new graphs during steady-state timing")
    job["native_compile"] = {"mode": "reduce-overhead", "graphs": warm_graphs}


def timed_calls(directory, job, call, index):
    warm_graphs = native_graphs(job)
    (directory / "warmup.done").touch()
    measurements = []
    for step in range(index, index + job["samples"]):
        outputs, sample = call(step)
        measurements.append(sample)
    (directory / "measurement.done").touch()
    check_native_compile(job, warm_graphs, native_graphs(job))
    return outputs, measurements


def run_wan(job, directory, torch, jt, np):
    torch.manual_seed(job["seed"])
    start = perf_counter()
    if job["case"] == VIDEO:
        call, model, metadata, input_tensor, weight = prepare_video(torch, jt, np, job)
    else:
        call, model, metadata, input_tensor, weight = prepare_denoiser(torch, np, job)
    residency = {"input": cuda_residency(torch, jt, input_tensor),
                 "weight": cuda_residency(torch, jt, weight)}
    torch.cuda.synchronize()
    setup_s = perf_counter() - start
    from worker import settled

    warmups, index = warm_up(job, call, settled if job.get("torch_mode") == "compiled" else None)
    replay = None
    if jt is not None:
        if job["case"] == VIDEO and job["mode"] == "reduce-overhead":
            replay = model.forward
        else:
            replay = model.__dict__.get("_replay")
    replay_before = dict(replay.stats) if replay is not None else {}
    _, measurements = timed_calls(directory, job, call, index)
    execution = job["mode"]
    if jt is not None and job["mode"] == "reduce-overhead":
        counts = {name: replay.stats[name] - before for name, before in replay_before.items()}
        execution = "eager-fallback"
        if counts.get("graph"):
            execution = "cuda-graph"
        elif counts.get("replayed"):
            execution = "executor-replay"
        job.update(measured_replay=counts, replay_refused=replay.refused if replay else None,
                   device_graph_refused=replay._graph_refused if replay else None)
    outputs, _ = call(0)
    for name, tensor in outputs.items():
        residency[name] = cuda_residency(torch, jt, tensor)
        value = tensor.detach().float().cpu().numpy()
        if not np.isfinite(value).all():
            raise RuntimeError(f"Nonfinite output in {name}")
        np.save(directory / f"{name}.npy", value, allow_pickle=False)
    return dict(metadata, execution=execution, setup_s=setup_s, warmups=warmups,
                measurements=measurements, residency=residency)


def run_stock(job, directory, torch, jt, np):
    from worker import Synchronizer, compile_report, configure, settled
    from workloads import WORKLOADS

    settings = configure(torch, "cuda", False, False, job["runtime"])
    if job["runtime"] == "torch":
        allocator = os.environ.get("PYTORCH_ALLOC_CONF") or os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
        if allocator:
            settings["cuda_allocator_config"] = allocator
    torch.manual_seed(job["seed"])
    task = WORKLOADS[job.get("stock_workload", job["case"])]
    workload = task(torch, "cuda", task.default_dtype, job["size"], seed=job["seed"])
    distributed = job.get("ranks", 1) > 1
    if distributed:
        rank = int(os.environ["LOCAL_RANK"])
        if rank:
            workload.rng = np.random.default_rng(np.random.SeedSequence(job["seed"], spawn_key=(rank,)))
    sync = Synchronizer(torch, job["runtime"], "cuda")
    start = perf_counter()
    workload.setup()
    if task.mode == "train" and task.stochastic:
        from training_tools.model_profile import disable_stochastic_layers

        disable_stochastic_layers(workload.model, torch)
    if distributed:
        workload.model = torch.nn.parallel.DistributedDataParallel(
            workload.model, device_ids=[rank])
    jittor_compile = jt is not None and job["mode"] == "reduce-overhead"
    compile_requested = (job.get("torch_mode") == "compiled"
                         or (jittor_compile and task.compile_what is not None))
    compile_inference = task.mode == "infer" and compile_requested
    compile_training = task.mode == "train" and compile_requested
    if compile_inference:
        workload.compile("reduce-overhead")
    sync([])
    setup_s = perf_counter() - start

    def call(index):
        start = perf_counter()
        values = workload.step()
        sync(values)
        elapsed = perf_counter() - start
        value = values[0].detach().float().mean().cpu().item()
        if not math.isfinite(value):
            raise RuntimeError("Nonfinite workload output")
        if index == 0 and compile_training:
            # Let the eager warmup allocate optimizer state outside capture.
            workload.compile("reduce-overhead")
        return values, {"total_s": elapsed, "value": float(value)}

    warmup_job = dict(job, warmup=max(3, job["warmup"])) if compile_training else job
    # A rank cannot independently choose to do more gradient reductions.
    warmups, index = warm_up(warmup_job, call, None if distributed else settled)
    capture_before = compile_report(torch, "jittor", workload) if jittor_compile else None
    outputs, measurements = timed_calls(directory, job, call, index)
    residency = cuda_residency(torch, jt, outputs[0])
    execution = job["mode"]
    replay = {}
    refusal = None
    if jittor_compile:
        capture_after = compile_report(torch, "jittor", workload)
        if capture_after is not None:
            before = (capture_before or {}).get("stats", {})
            replay = {name: count - before.get(name, 0) for name, count in capture_after["stats"].items()}
            refusal = capture_after["refused"] or capture_after["graph_refused"]
        else:
            refusal = "stock workload does not expose a replay capture"
        execution = ("cuda-graph" if replay.get("graph") else
                     "executor-replay" if replay.get("replayed") else "eager-fallback")

    checked_batches = 0
    if task.mode == "infer" and not task.stochastic:
        # Save a full cycle in input-pool order, independent of warmup length.
        arrays = {}
        checked_batches = len(workload.pool)
        for step in range(index + job["samples"], index + job["samples"] + checked_batches):
            values = workload.step()
            sync(values)
            batch = step % checked_batches
            for i, value in enumerate(values):
                arrays[f"batch.{batch}.{i}"] = output_array(np, torch, value)
        np.savez_compressed(directory / "outputs.npz", **arrays)
    return dict(execution=execution, measured_replay=replay, replay_refused=refusal,
                setup_s=setup_s, warmups=warmups, measurements=measurements,
                config=workload.config(), dtype=task.default_dtype, observed_output=residency,
                settings=settings, workload_mode=task.mode,
                stochastic=task.stochastic and task.mode != "train",
                stochastic_layers_disabled=task.stochastic and task.mode == "train",
                checked_output_batches=checked_batches)


def output_tensors(torch, result, prefix="output"):
    if isinstance(result, torch.Tensor):
        yield prefix, result
    elif isinstance(result, Mapping):
        for name, item in result.items():
            yield from output_tensors(torch, item, f"{prefix}.{name}")
    elif isinstance(result, (list, tuple)):
        for index, item in enumerate(result):
            yield from output_tensors(torch, item, f"{prefix}.{index}")


def output_array(np, torch, tensor, *, key="output", infinity_sign=None):
    value = tensor.detach().cpu()
    if value.dtype in (torch.bfloat16, torch.float16):
        value = value.float()
    array = value.numpy()
    finite = np.isfinite(array)
    if not finite.all():
        if infinity_sign is None or not finite.any() or not (finite | (array == infinity_sign * np.inf)).all():
            raise RuntimeError(f"Unexpected nonfinite inference values in {key}")
    return array


def run_model(job, directory, torch, jt, np):
    from importlib.metadata import version

    from worker import Synchronizer, configure, settled

    configure(torch, "cuda", False, False, job["runtime"])
    sync = Synchronizer(torch, job["runtime"], "cuda")
    start = perf_counter()
    forward, inputs, model, assets = prepare_model(
        torch, job["case"], job["model_assets"], job["seed"], job.get("processed_inputs"))
    residency = {"inputs": [cuda_residency(torch, jt, tensor) for tensor in inputs],
                 "weight": cuda_residency(torch, jt, next(model.parameters()))}
    if job["mode"] == "reduce-overhead":
        if jt is None:
            if MODEL_CASES[job["case"]].options.get("torch_emulate_precision_casts"):
                # Preserve eager's lower precision rounding between fused ops.
                from torch._inductor import config as inductor_config

                inductor_config.emulate_precision_casts = True
                job["torch_emulate_precision_casts"] = True
            # Materialize any lazily cached module buffers (e.g. detector anchors/strides)
            # in eager memory before torch.compile so CUDAGraph replay does not overwrite them.
            forward()
            sync([])
            if MODEL_CASES[job["case"]].loader == "recurrent":
                from torch._dynamo import config

                config.allow_rnn = True
        forward = torch.compile(forward, mode="reduce-overhead", dynamic=False)
    sync([])
    setup_s = perf_counter() - start

    def call(_):
        if job["mode"] == "reduce-overhead":
            torch.compiler.cudagraph_mark_step_begin()
        start = perf_counter()
        outputs = dict(output_tensors(torch, forward()))
        if not outputs:
            raise RuntimeError("Model forward returned no tensors")
        sync(outputs.values())
        return outputs, {"total_s": perf_counter() - start}

    warmups, index = warm_up(job, call, settled)
    replay_before = dict(getattr(forward, "stats", {}))
    outputs, measurements = timed_calls(directory, job, call, index)
    residency["outputs"] = {key: cuda_residency(torch, jt, tensor) for key, tensor in outputs.items()}
    masked_outputs = MODEL_CASES[job["case"]].masked_outputs
    arrays = {key: output_array(np, torch, tensor, key=key, infinity_sign=masked_outputs.get(key))
              for key, tensor in outputs.items()}
    np.savez_compressed(directory / "outputs.npz", **arrays)
    execution = job["mode"]
    replay = {}
    if jt is not None and job["mode"] == "reduce-overhead":
        replay = {name: forward.stats[name] - before for name, before in replay_before.items()}
        if replay.get("graph"):
            execution = "cuda-graph"
        elif replay.get("replayed"):
            execution = "executor-replay"
        else:
            execution = "eager-fallback"
    if job["profile"]:
        import cProfile
        import pstats

        with cProfile.Profile() as profiler:
            call(index)
        with (directory / "profile.txt").open("w") as log:
            pstats.Stats(profiler, stream=log).sort_stats("cumulative").print_stats(70)
    loader = MODEL_CASES[job["case"]].loader
    package = {"pyg": "torch-geometric", "chronos": "chronos-forecasting"}.get(loader, loader)
    dependency = None if loader in ("recurrent", "cosyvoice", "torchvision") else {package: version(package)}
    return dict(execution=execution, measured_replay=replay, replay_refused=getattr(forward, "refused", None),
                dependency=dependency, weights=str(assets.resolve()) if assets else "deterministic initialization",
                setup_s=setup_s, warmups=warmups, measurements=measurements, residency=residency)


def prepare_shared_inputs(args, output, job):
    inputs = output / "inputs"
    inputs.mkdir(exist_ok=True)
    path = inputs / (job["case"] + ".npz")
    if not path.exists():
        spec = path.with_suffix(".json")
        write_json(spec, job)
        env = worker_env(args, output, "torch")
        env["CUDA_VISIBLE_DEVICES"] = ""
        with path.with_suffix(".log").open("w") as log:
            subprocess.run([args.torch_python, str(Path(__file__).resolve()), "--preprocess", str(spec)],
                           env=env, stdout=log, stderr=subprocess.STDOUT,
                           timeout=args.setup_timeout, check=True)
    return dict(processed_inputs=str(path), input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def preprocess(spec):
    import numpy as np

    torch = import_torch("torch")
    job = json.loads(spec.read_text())
    name = job["case"]
    case = MODEL_CASES[name]
    batch = processor_batch(case, Path(job["model_assets"]) / name, Inputs(torch, job["seed"], case.dtype))
    arrays = {key: (value.float() if value.dtype == torch.bfloat16 else value).numpy()
              for key, value in batch.items() if isinstance(value, torch.Tensor)}
    if not arrays:
        raise RuntimeError("Model processor returned no tensors")
    np.savez(spec.with_suffix(".npz"), **arrays)


def limit_cv_batches(job, steps):
    import cv_shared_input

    class ShortBatches(cv_shared_input.SharedBatches):
        def __iter__(self):
            full = (batch for batch in super().__iter__() if len(batch) == self.batch_size)
            return itertools.islice(full, len(self))

        def __len__(self):
            return steps * job["ranks"]

    class ShortValidation(cv_shared_input.SharedPets):
        def __init__(self, files, labels, training=True, seed=42):
            if not training:
                limit = job["ranks"] * job["batch"]
                files, labels = files[:limit], labels[:limit]
            super().__init__(files, labels, training=training, seed=seed)

    cv_shared_input.SharedBatches = ShortBatches
    cv_shared_input.SharedPets = ShortValidation


def trace_nlp_inputs(job, steps):
    from accelerate import Accelerator
    import nlp_step_loop

    prepare_loader = Accelerator.prepare_data_loader
    train_steps = nlp_step_loop.train_steps
    plan = {}

    def recorded_loader(self, loader, *args, **kwargs):
        count = job["batch"] * job["ranks"]
        if len(loader.dataset) == steps * count:
            rows = json.dumps(loader.dataset[:count], sort_keys=True).encode()
            plan["sha256"] = hashlib.sha256(rows).hexdigest()
        return prepare_loader(self, loader, *args, **kwargs)

    def without_validation(*args, **kwargs):
        prepare = Accelerator.prepare

        def prepare_training(self, model, optimizer, train, evaluation, scheduler):
            model, optimizer, train, _, scheduler = prepare(self, model, optimizer, train, evaluation, scheduler)
            return model, optimizer, train, (), scheduler

        Accelerator.prepare = prepare_training
        try:
            return train_steps(*args, **kwargs)
        finally:
            Accelerator.prepare = prepare

    Accelerator.prepare_data_loader = recorded_loader
    nlp_step_loop.train_steps = without_validation
    return plan


def run_training_rank(job, directory, jt):
    sys.path.insert(0, job["training_tools"])
    steps = job["warmup"] + job["samples"]
    tool = importlib.import_module("bench_cv" if job["family"] == "cv" else "bench_nlp")
    result_dir = directory / "training"
    sys.argv = [tool.__file__, "--backend", job["runtime"], "--output", str(result_dir),
                "--checkout", job["checkout"], "--model", job["model"], "--initial-state", job["initial_state"],
                "--execution", job["mode"], "--batch-size", str(job["batch"]),
                "--epochs", "1", "--warmup-steps", str(job["warmup"]), "--window-steps", "1"]
    if job["family"] == "cv":
        limit_cv_batches(job, steps)
        cycles = "2" if job["ranks"] == 2 and job["batch"] == 64 else "1"
        sys.argv += ["--data-dir", job["cv_data"], "--cycles", cycles, "--num-workers", "8",
                     "--persistent-workers", "--shared-input-pipeline", "--disable-stochastic-layers"]
        input_plan = {}
    else:
        input_plan = trace_nlp_inputs(job, steps)
        # Native eager and compiled dropout can use different random streams
        # even with the same seed. Compare the same training profile in all modes.
        sys.argv += ["--batches", str(job["ranks"]), "--cycles", str(steps),
                     "--metric", job["metric"], "--disable-stochastic-layers"]
        if job.get("torch_mode") == "compiled":
            sys.argv.append("--tensor-learning-rate")

    warm_graphs = None
    def print_progress(*values, **kwargs):
        nonlocal warm_graphs
        if (len(values) == 1 and isinstance(values[0], str)
                and values[0].startswith("{") and '"step":' in values[0] and '"total":' in values[0]):
            record = json.loads(values[0])
            if record["step"] == job["warmup"] - 1:
                warm_graphs = native_graphs(job)
                (directory / "warmup.done").touch()
            if record["step"] == steps - 1:
                (directory / "measurement.done").touch()
                check_native_compile(job, warm_graphs, native_graphs(job))
        print(*values, **kwargs)

    tool.print = print_progress
    tool.main()
    rank = json.loads((result_dir / "result.json").read_text())
    if (not rank["completed"] or len(rank["records"]) != steps
            or rank["local_batch_size"] != job["batch"]
            or rank["world_size"] != job["ranks"] or rank["rank"] != int(os.environ["RANK"])):
        raise RuntimeError(f"Unexpected training result for rank {os.environ['RANK']}: "
                           f"{len(rank['records'])} steps, rank {rank['rank']}/{rank['world_size']}")
    if rank["execution"] != job["mode"] or any(not math.isfinite(row["loss"]) for row in rank["records"]):
        raise RuntimeError("Training rank returned unexpected mode or loss")
    if jt is not None and rank["fallback"] != 0:
        raise RuntimeError("Training rank used backend fallback")
    return dict(rank, training_input_sha256=rank.get("input_plan_sha256") or input_plan.get("sha256"))


def run_worker(spec):
    import numpy as np
    from worker import versions

    job = json.loads(spec.read_text())
    directory = spec.parent
    try:
        torch, jt, guard = checked_runtime(job)
    except SystemExit as error:
        # The build uses exit code 3 to request a fresh import after replacing
        # jit_utils. Restart before the worker touches a distributed group.
        restart_key = "JITTOR_BENCH_JIT_UTILS_RESTARTED"
        if job["runtime"] != "jittor" or error.code != 3 or os.environ.get(restart_key):
            raise
        os.environ[restart_key] = "1"
        os.execv(sys.executable, [sys.executable, "-u", *sys.argv])
    stock_distributed = job["kind"] == "stock" and job.get("ranks", 1) > 1
    if stock_distributed:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
        if torch.distributed.get_world_size() != job["ranks"]:
            raise RuntimeError("Stock training worker has the wrong distributed world size")
    torch.cuda.reset_peak_memory_stats()
    before_fallback = jt.core.backend_fallback_count() if jt is not None else None
    with guard:
        if job["kind"] == "training":
            result = run_training_rank(job, directory, jt)
        elif job["kind"] == "stock":
            from workloads import WORKLOADS

            inference = WORKLOADS[job.get("stock_workload", job["case"])].mode == "infer"
            grad_scope = torch.no_grad() if inference else nullcontext()
            replay_scope = jt.flag_scope(auto_graph_replay=0) if inference and jt is not None else nullcontext()
            with grad_scope, replay_scope:
                result = run_stock(job, directory, torch, jt, np)
        elif job["kind"] == "model":
            replay_scope = jt.flag_scope(auto_graph_replay=0) if jt is not None else nullcontext()
            with torch.no_grad(), replay_scope:
                result = run_model(job, directory, torch, jt, np)
        else:
            with torch.no_grad():
                result = run_wan(job, directory, torch, jt, np)
    if job["kind"] != "training":
        result["median_s"] = median(sample["total_s"] for sample in result["measurements"])
        warmups = result.get("warmups") or ()
        result["compile_warmup_s"] = sum(sample["total_s"] for sample in warmups)
        result["first_call_s"] = warmups[0]["total_s"] if warmups else None
    peak_alloc = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    result.update(peak_memory_bytes=peak_alloc, peak_reserved_bytes=peak_reserved,
                  peak_vram_gb=round(peak_alloc / (1024 ** 3), 4))
    packages = versions(torch, job["runtime"])
    if job.get("torch_mode") == "compiled":
        if not job.get("native_compile"):
            raise RuntimeError("Torch compiled worker did not complete timing with a compiled graph")
        result["execution"] = "compiled"
    jittor_version = getattr(jt, "__version__", None) if jt is not None else None
    jittor_cuda_version = None
    if jt is not None:
        try:
            from jittor.build import compiler as _jt_compiler
            jittor_cuda_version = getattr(_jt_compiler, "nvcc_version", None) or getattr(_jt_compiler, "cuda_version", None)
            if isinstance(jittor_cuda_version, (list, tuple)):
                jittor_cuda_version = ".".join(str(x) for x in jittor_cuda_version)
        except Exception:
            jittor_cuda_version = None
        if not jittor_cuda_version:
            jittor_cuda_version = packages.get("cuda")
    torch_version = torch.__version__ if job["runtime"] == "torch" else None
    torch_cuda_version = getattr(getattr(torch, "version", None), "cuda", None) if job["runtime"] == "torch" else None
    job.update(result, gpu=torch.cuda.get_device_name(), python=sys.executable, torch=torch.__version__,
               torch_version=torch_version, torch_cuda_version=torch_cuda_version,
               jittor_version=jittor_version, jittor_cuda_version=jittor_cuda_version,
               versions=packages,
               fallback_count=jt.core.backend_fallback_count() - before_fallback if jt is not None else None)
    write_json(directory / "result.json", job)
    if stock_distributed:
        torch.distributed.destroy_process_group()


def compare_training(np, reference, candidate):
    expected, actual = reference["rank_results"], candidate["rank_results"]
    if len(expected) != len(actual):
        return dict(same_inputs=False, parity=False, comparable=False,
                    parity_error="Training rank counts differ")
    keys = ("training_input_sha256", "initial_state_sha256")
    inputs_match = all(left.get(key) is not None and left[key] == right.get(key)
                       for left, right in zip(expected, actual) for key in keys)
    profile_match = all(left.get("stochastic_layers_disabled") == right.get("stochastic_layers_disabled")
                        for left, right in zip(expected, actual))
    errors = []
    metrics = {}
    for rank, (left, right) in enumerate(zip(expected, actual)):
        expected_steps = [row["step"] for row in left["records"]]
        actual_steps = [row["step"] for row in right["records"]]
        if not expected_steps or expected_steps != actual_steps:
            errors.append(dict(rank=rank, parity_error="Training steps are missing or differ"))
            continue
        inputs_match &= all(a.get("batch_sha256") == b.get("batch_sha256")
                            for a, b in zip(left["records"], right["records"]))
        reference_loss = np.array([row["loss"] for row in left["records"]])
        candidate_loss = np.array([row["loss"] for row in right["records"]])
        metrics[f"rank{rank}.loss"] = compare_arrays(
            np, reference_loss, candidate_loss, TRAINING_LOSS_ATOL, TRAINING_LOSS_RTOL)
        failed = np.flatnonzero(~np.isclose(candidate_loss, reference_loss,
                                           atol=TRAINING_LOSS_ATOL, rtol=TRAINING_LOSS_RTOL))
        if failed.size:
            first = failed[0]
            errors.append(dict(rank=rank, first_step=expected_steps[first], steps_outside_tolerance=int(failed.size),
                               reference_loss=float(reference_loss[first]), candidate_loss=float(candidate_loss[first]),
                               max_abs=float(np.abs(reference_loss - candidate_loss).max())))
    parity = bool(metrics) and all(check["parity"] for check in metrics.values()) and not errors
    return dict(same_inputs=inputs_match, same_training_profile=profile_match, parity=parity,
                comparable=inputs_match and profile_match and parity,
                output_errors=metrics, loss_errors=errors,
                atol=TRAINING_LOSS_ATOL, rtol=TRAINING_LOSS_RTOL)


def compare_stock_losses(np, reference, candidate):
    # Compiled and eager workers may settle after a different number of warmup
    # steps. Compare their common prefix from the same initial state.
    def losses(row):
        samples = row["warmups"] + row["measurements"]
        return np.array([sample["value"] for sample in samples], dtype=np.float64)

    expected, actual = losses(reference), losses(candidate)
    steps = min(expected.size, actual.size)
    if not steps:
        return dict(parity=False, parity_error="No matching training loss steps")
    errors = compare_arrays(np, expected[:steps], actual[:steps],
                            TRAINING_LOSS_ATOL, TRAINING_LOSS_RTOL)
    return dict(parity=errors["parity"], loss_steps=steps, output_errors={"loss": errors},
                atol=TRAINING_LOSS_ATOL, rtol=TRAINING_LOSS_RTOL)


def compare_pair(reference, candidate):
    import numpy as np

    kind = reference["kind"]
    if kind == "training":
        return compare_training(np, reference, candidate)
    if kind == "stock" and (reference.get("rank_results") or candidate.get("rank_results")):
        left, right = reference.get("rank_results", ()), candidate.get("rank_results", ())
        if len(left) != len(right):
            return dict(parity=False, comparable=False, parity_error="Stock training rank counts differ")
        checks = [compare_pair(a, b) for a, b in zip(left, right)]
        errors = {f"rank{rank}.{name}": metrics for rank, check in enumerate(checks)
                  for name, metrics in check.get("output_errors", {}).items()}
        return dict(parity=all(check.get("parity") is True for check in checks),
                    comparable=all(check.get("comparable") and check.get("parity") is True for check in checks),
                    same_inputs=all(check.get("same_inputs") for check in checks),
                    same_dependencies=all(check.get("same_dependencies") for check in checks),
                    output_errors=errors, rank_checks=checks)
    dependencies_match = all(reference["versions"].get(key) == candidate["versions"].get(key)
                             for key in ("diffusers", "transformers"))
    if kind == "model":
        output = compare_model_outputs(np, reference, candidate)
        dependencies_match = dependencies_match and reference["dependency"] == candidate["dependency"]
        inputs_match = (reference["seed"] == candidate["seed"]
                        and reference.get("input_sha256") == candidate.get("input_sha256"))
        output.update(same_dependencies=dependencies_match, same_inputs=inputs_match,
                      comparable=output["parity"] and dependencies_match and inputs_match
                      and reference["weights"] == candidate["weights"])
        return output
    if kind == "wan":
        errors = {}
        first, second = Path(reference["directory"]), Path(candidate["directory"])
        names = {p.name for p in first.glob("*.npy")}
        if not names or names != {p.name for p in second.glob("*.npy")}:
            return dict(parity=False, comparable=False, parity_error="Output fields differ")
        case_name = reference.get("case")
        multistep = case_name in MULTI_STEP_INFERENCE_CASES
        if multistep:
            tol = MULTI_STEP_INFERENCE_TOLERANCES["bfloat16"]
            tier = "inference-multistep:bfloat16"
        else:
            # Wan2.2-TI2V-5B runs a 5B-parameter BF16 DiT where PyTorch compiled vs
            # eager shows ~0.16 max_abs / 0.998 cosine; allow BF16 tail outliers with
            # strict non-finite and high-cosine gates.
            tol = dict(atol=0.08, rtol=0.08, max_outside_fraction=5e-3, min_cosine=0.998,
                       max_abs_cap=0.25, max_scale_ratio=0.15)
            tier = "inference-wan:bfloat16"
        for name in sorted(names):
            a, b = (np.load(directory / name, allow_pickle=False) for directory in (first, second))
            errors[name] = compare_arrays(
                np, a, b, tol["atol"], tol["rtol"],
                max_outside_fraction=tol["max_outside_fraction"],
                min_cosine=tol["min_cosine"],
                max_abs_cap=tol["max_abs_cap"],
                max_scale_ratio=tol["max_scale_ratio"],
            )
        parity = all(item["parity"] for item in errors.values())
        return dict(parity=parity, comparable=parity and dependencies_match, output_errors=errors,
                    same_dependencies=dependencies_match, parity_tier=tier,
                    atol=tol["atol"], rtol=tol["rtol"],
                    max_outside_fraction=tol["max_outside_fraction"], min_cosine=tol["min_cosine"])
    inputs_match = (reference["config"] == candidate["config"] and reference["seed"] == candidate["seed"]
                    and reference.get("stochastic_layers_disabled") == candidate.get("stochastic_layers_disabled"))
    result = dict(same_dependencies=dependencies_match, same_inputs=inputs_match,
                  comparable=dependencies_match and inputs_match)
    if reference.get("checked_output_batches") and candidate.get("checked_output_batches"):
        result.update(compare_model_outputs(np, reference, candidate))
    elif (reference.get("workload_mode") == candidate.get("workload_mode") == "train"
          and not (reference["stochastic"] or candidate["stochastic"])):
        result.update(compare_stock_losses(np, reference, candidate))
    if "parity" in result:
        result["comparable"] = result["comparable"] and result["parity"]
    return result


def compare_results(rows):
    if not rows:
        return []
    modes = {}
    for row in rows:
        mode = row.get("torch_mode") or ("eager" if row["runtime"] == "torch" else "jittor")
        if row.get("jittor_mode"):
            mode = f"jittor_{row['jittor_mode']}"
        modes[mode] = row
    ok = {mode: row for mode, row in modes.items() if row["status"] == "ok"}
    case, round_id = rows[0]["case"], rows[0]["round"]
    statistic = "mean" if any(row.get("kind") == "training" for row in rows) else "median"
    times = {mode: row[f"{statistic}_s"] * 1000 for mode, row in ok.items()}
    summary = dict(case=case, round=round_id, statistic=statistic,
                   torch_eager_ms=times.get("eager"), torch_compiled_ms=times.get("compiled"),
                   jittor_eager_ms=times.get("jittor_eager"), jittor_compiled_ms=times.get("jittor_compiled"),
                   jittor_ms=None, jittor_execution=None,
                   statuses={mode: row["status"] for mode, row in modes.items()})
    def passed(check):
        return check.get("comparable") is True and check.get("parity") is True

    best = None
    if "eager" in ok and "compiled" in ok:
        compiled_check = compare_pair(ok["eager"], ok["compiled"])
        summary["torch_compile_check"] = compiled_check
        if passed(compiled_check):
            best = min(("eager", "compiled"), key=times.__getitem__)
            summary.update(torch_baseline_mode=best, torch_baseline_ms=times[best], torch_ms=times[best])
    candidates = [mode for mode in ("jittor_compiled", "jittor_eager", "jittor") if mode in ok]
    checks = {}
    if "eager" in ok:
        checks = {mode: compare_pair(ok["eager"], ok[mode]) for mode in candidates}
        summary["jittor_checks"] = checks
    if candidates:
        verified = [mode for mode in candidates if passed(checks.get(mode, {}))]
        # A refused capture ran eager. Prefer the separately measured eager mode.
        if "jittor_eager" in verified and ok.get("jittor_compiled", {}).get("execution") == "eager-fallback":
            verified = [mode for mode in verified if mode != "jittor_compiled"]
        selected = min(verified or candidates, key=times.__getitem__)
        summary.update(jittor_selected_mode=selected, jittor_ms=times[selected],
                       jittor_execution=ok[selected]["execution"])
        if checks:
            summary.update(checks[selected])
            if best and passed(checks[selected]) and times[best] > 0 and times[selected] > 0:
                summary["jittor_over_torch"] = times[selected] / times[best]
    return [summary]


INFERENCE_TOLERANCES = {
    "float32": dict(atol=1e-4, rtol=1e-3, max_outside_fraction=0.0, min_cosine=0.9999,
                    max_abs_cap=None, max_scale_ratio=None),
    "float16": dict(atol=1e-2, rtol=3e-2, max_outside_fraction=1e-3, min_cosine=0.999,
                    max_abs_cap=0.1, max_scale_ratio=0.1),
    "bfloat16": dict(atol=5e-2, rtol=8e-2, max_outside_fraction=5e-3, min_cosine=0.998,
                     max_abs_cap=0.25, max_scale_ratio=0.15),
}

# Deep FP32 vision/deformable-attention models (GroundingDINO, OWLv2, SigLIP2)
# accumulate tiny FP32 FMA/grid_sample tail rounding across 18-24+ layers: even
# PyTorch compiled vs eager shows 0.001%-0.11% pointwise outliers above 1e-4
# while maintaining >= 0.9999999 cosine similarity.
DEFORMABLE_FP32_CASES = {"grounding-dino", "owlv2", "siglip2"}
DEFORMABLE_FP32_TOLERANCE = dict(
    atol=1e-4, rtol=1e-3, max_outside_fraction=2e-3, min_cosine=0.99999,
    max_abs_cap=1e-2, max_scale_ratio=1e-2,
)

# Iterative multi-step diffusion sampling (e.g. 20-step DDIM with CFG 7.5) compounds
# low-precision rounding across steps: even PyTorch compiled vs eager shows ~5.5%
# pointwise outliers with 0.99999 cosine similarity. Keep strict non-finite and
# high-cosine checks so genuine computation errors still fail.
MULTI_STEP_INFERENCE_CASES = {"sd15_sample", "ti2v-video"}
MULTI_STEP_INFERENCE_TOLERANCES = {
    "float16": dict(atol=5e-2, rtol=5e-2, max_outside_fraction=0.10, min_cosine=0.9995,
                    max_abs_cap=0.5, max_scale_ratio=0.15),
    "bfloat16": dict(atol=8e-2, rtol=8e-2, max_outside_fraction=0.10, min_cosine=0.999,
                     max_abs_cap=0.75, max_scale_ratio=0.2),
}


def compare_arrays(np, a, b, atol, rtol, *, max_outside_fraction=0.0, min_cosine=None,
                   max_abs_cap=None, max_scale_ratio=None):
    if a.shape != b.shape or a.dtype != b.dtype:
        return dict(parity=False, reference_shape=a.shape, candidate_shape=b.shape,
                    reference_dtype=str(a.dtype), candidate_dtype=str(b.dtype))
    is_float = np.issubdtype(a.dtype, np.floating)
    close = np.isclose(b, a, atol=atol, rtol=rtol) if is_float else (a == b)
    finite_a = np.isfinite(a) if is_float else np.ones(a.shape, dtype=bool)
    finite_b = np.isfinite(b) if is_float else np.ones(b.shape, dtype=bool)
    nonfinite_mismatch = int(np.count_nonzero(finite_a != finite_b))
    both_finite = finite_a & finite_b
    reference = a[both_finite].astype(np.float64, copy=False)
    actual = b[both_finite].astype(np.float64, copy=False)
    with np.errstate(over="ignore", invalid="ignore"):
        difference = np.abs(reference - actual)
        max_abs = float(np.max(difference, initial=0.0))
        relative = difference / np.maximum(np.abs(reference), np.finfo(np.float64).eps)
        max_rel = float(np.max(relative, initial=0.0))
    reference_scale = float(np.max(np.abs(reference), initial=0.0))
    actual_scale = float(np.max(np.abs(actual), initial=0.0))
    cosine = None
    if reference_scale and actual_scale:
        reference_norm = reference / reference_scale
        actual_norm = actual / actual_scale
        dot = np.dot(reference_norm, actual_norm)
        cosine = float(np.clip(dot / (np.linalg.norm(reference_norm) * np.linalg.norm(actual_norm)), -1, 1))
    elif reference.size:
        cosine = 1.0 if reference_scale == actual_scale else 0.0
    outside = int(a.size - np.count_nonzero(close))
    outside_fraction = (outside / a.size) if a.size else 0.0
    max_abs = max_abs if math.isfinite(max_abs) else None
    parity = nonfinite_mismatch == 0 and outside == 0
    if not parity and is_float and nonfinite_mismatch == 0 and max_outside_fraction > 0.0:
        cosine_ok = (min_cosine is None) or (cosine is not None and cosine >= min_cosine)
        abs_limit = max(max_abs_cap or 0.0, (max_scale_ratio or 0.0) * reference_scale)
        abs_ok = (max_abs_cap is None and max_scale_ratio is None) or (max_abs is not None and max_abs <= abs_limit)
        parity = (outside_fraction <= max_outside_fraction) and cosine_ok and abs_ok
    return dict(parity=parity, max_abs=max_abs, max_abs_diff=max_abs,
                max_rel_diff=max_rel if math.isfinite(max_rel) else None, cosine_similarity=cosine,
                elements=a.size, elements_outside_tolerance=outside,
                outside_fraction=outside_fraction, nonfinite_mismatch=nonfinite_mismatch)


def compare_model_outputs(np, reference, candidate):
    ref = Path(reference["directory"]) / "outputs.npz"
    actual = Path(candidate["directory"]) / "outputs.npz"
    dtype_name = reference["dtype"]
    case_name = reference.get("case")
    multistep = case_name in MULTI_STEP_INFERENCE_CASES and dtype_name in MULTI_STEP_INFERENCE_TOLERANCES
    deformable = case_name in DEFORMABLE_FP32_CASES and dtype_name == "float32"
    if multistep:
        tol = MULTI_STEP_INFERENCE_TOLERANCES[dtype_name]
        tier_prefix = "inference-multistep"
    elif deformable:
        tol = DEFORMABLE_FP32_TOLERANCE
        tier_prefix = "inference-deformable"
    else:
        tol = INFERENCE_TOLERANCES[dtype_name]
        tier_prefix = "inference"
    atol, rtol = tol["atol"], tol["rtol"]
    errors = {}
    with np.load(ref, allow_pickle=False) as left, np.load(actual, allow_pickle=False) as right:
        if not left.files or set(left.files) != set(right.files):
            return dict(parity=False, parity_error="Output fields differ")
        for key in left.files:
            errors[key] = compare_arrays(
                np, left[key], right[key], atol, rtol,
                max_outside_fraction=tol["max_outside_fraction"],
                min_cosine=tol["min_cosine"],
                max_abs_cap=tol["max_abs_cap"],
                max_scale_ratio=tol["max_scale_ratio"],
            )
    return dict(parity=all(v["parity"] for v in errors.values()), output_errors=errors,
                atol=atol, rtol=rtol, parity_tier=f"{tier_prefix}:{dtype_name}",
                max_outside_fraction=tol["max_outside_fraction"], min_cosine=tol["min_cosine"])


def main():
    if sys.argv[1:2] == ["--worker"]:
        run_worker(Path(sys.argv[2]))
        return 0
    if sys.argv[1:2] == ["--preprocess"]:
        preprocess(Path(sys.argv[2]))
        return 0
    args = parse_args()
    if args.output_dir:
        output = args.output_dir.resolve()
        output.mkdir(parents=True)
    else:
        parent = lab_root().resolve() / "_state" / "wan22-benchmark"
        parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-")
        output = Path(tempfile.mkdtemp(prefix=stamp, dir=parent))
    (output / "tmp").mkdir()
    print(f"Results in {output}", flush=True)
    # DataLoader creates Unix sockets below TMPDIR. Use a short alias while
    # keeping the actual temporary files with the benchmark state.
    with tempfile.TemporaryDirectory(prefix="jt-bench-", dir="/tmp") as ipc:
        args.ipc_root = Path(ipc) / "tmp"
        args.ipc_root.symlink_to(output / "tmp", target_is_directory=True)
        return run_matrix(args, output)


def execution_order(args, case):
    torch_modes = ("compiled", "eager") if args.torch_mode == "both" else (args.torch_mode,)
    stock = args.stock_cases.get(case)
    if case in MODEL_CASES:
        preference = args.model_mode
    elif stock is not None and stock["compile_what"] is None:
        preference = "eager"
    elif stock is not None:
        preference = args.stock_mode
    else:
        preference = None
    if preference == "both":
        jittor_modes = ("compiled", "eager")
    elif preference == "reduce-overhead":
        jittor_modes = ("compiled",)
    else:
        jittor_modes = (preference,)
    modes = {"torch": torch_modes, "jittor": jittor_modes}
    executions = []
    for runtime in args.runtimes.split(","):
        executions.extend((runtime, mode) for mode in modes[runtime])
    return sorted(executions, key=lambda item: item[1] != "compiled")


def run_matrix(args, output):
    rows = []
    comparisons = []
    planned_rounds = dict.fromkeys(args.case, args.rounds or (1 if len(args.case) > 1 else 2))
    meta = {"state": "running", "pid": os.getpid(), "planned": args.case,
            "started": datetime.now(timezone.utc).isoformat(), "planned_rounds": planned_rounds}

    def publish(current=None):
        meta["current"] = current
        write_json(output / "results.json", {"meta": meta, "runs": rows, "comparisons": comparisons})

    for name in args.case:
        executions = execution_order(args, name)
        rounds = planned_rounds[name]
        processed = None
        preprocessing_error = None
        for round_id in range(1, rounds + 1):
            budget = args.measurement_timeout
            round_rows = []
            for runtime, mode in executions:
                phase = dict(case=name, runtime=runtime, round=round_id,
                             torch_mode=mode if runtime == "torch" else None,
                             jittor_mode=mode if runtime == "jittor" else None)
                publish(phase)
                try:
                    job = make_job(args, name, runtime, round_id, mode)
                except (OSError, ValueError) as error:
                    status = "missing-assets" if isinstance(error, FileNotFoundError) else "invalid-input"
                    row = dict(phase, status=status, error=str(error))
                    print(f"{name} {runtime}: {error}", flush=True)
                else:
                    if (job["kind"] == "model" and args.torch_python
                            and needs_processor(MODEL_CASES[name]) and processed is None
                            and preprocessing_error is None):
                        try:
                            processed = prepare_shared_inputs(args, output, job)
                        except (OSError, subprocess.SubprocessError) as error:
                            preprocessing_error = str(error)
                    if preprocessing_error is not None:
                        row = dict(job, status="failed", error=preprocessing_error,
                                   directory=str(output / "inputs"))
                    else:
                        job.update(processed or {})
                        row = run_one(args, output, job, budget)
                        budget -= row["measured_seconds"]
                rows.append(row)
                round_rows.append(row)
                if row.get("torch_version"):
                    meta["torch_version"] = row["torch_version"]
                if row.get("torch_cuda_version"):
                    meta["torch_cuda_version"] = row["torch_cuda_version"]
                if row.get("jittor_version"):
                    meta["jittor_version"] = row["jittor_version"]
                if row.get("jittor_cuda_version"):
                    meta["jittor_cuda_version"] = row["jittor_cuda_version"]
                if row.get("jittor_sha"):
                    meta["jittor_sha"] = row["jittor_sha"]
            comparisons.extend(compare_results(round_rows))
            publish()
    failed = (any(row["status"] != "ok" for row in rows)
              or any(not summary.get("comparable", True)
                     or not summary.get("torch_compile_check", {}).get("comparable", True)
                     or any(not check["comparable"] for check in summary.get("jittor_checks", {}).values())
                     for summary in comparisons))
    meta.update(state="completed-with-issues" if failed else "completed",
                finished=datetime.now(timezone.utc).isoformat())
    publish()
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
