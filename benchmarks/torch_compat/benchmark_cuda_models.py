"""Compare CUDA model workloads on Jittor and PyTorch.

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
Qwen3, Stable Diffusion, DDPM, ResNet, ViT and BERT. Stock and ecosystem
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
images. Eight warmup steps precede 16 timed steps by default. Training reports
mean step time using the slowest rank per step. Wan and the stock workloads
report median latency. Results and rank logs are saved outside the checkout.
A single selection defaults to two process rounds. Stock and training cases
run once each when selected as a suite. Video uses two samples in one process.

Use --case ecosystem for inference presets spanning CV, NLP, generation,
audio, graphs and forecasting. --list prints the scope of these presets.
Synthetic input shapes and deterministic weights are used for architecture
presets. Supply local snapshots via --model-assets for checkpoint presets.
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
from model_cases import MODEL_CASES, prepare_model
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
    groups = {"wan": list(WAN_CASES), "stock": list(stock_cases), "training": list(TRAINING_CASES)}
    groups["ecosystem"] = list(MODEL_CASES)
    groups["all"] = groups["wan"] + groups["stock"] + groups["training"] + groups["ecosystem"]
    for name, case in MODEL_CASES.items():
        groups.setdefault(case.group, []).append(name)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", action="append", metavar="NAME",
                        help="a preset or group, including wan, stock, training, ecosystem, all")
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
    parser.add_argument("--training-tools", type=Path, help="directory containing bench_cv.py and bench_nlp.py")
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
        for group in ("wan", "stock", "training"):
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
                    dtype=case.dtype, warmup=args.warmup or 6, samples=args.samples or 12,
                    seed=args.seed if args.seed is not None else 0)
    if name in args.stock_cases:
        warmup, samples = (1, 2) if args.stock_size == "tiny" else (6, 12)
        return dict(job, kind="stock", mode=inference_mode, size=args.stock_size,
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
            if job["kind"] == "training":
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
        result = aggregate_training(job, records) if job["kind"] == "training" else records[0]
        elapsed = sum(sample["total_s"] for sample in result["measurements"])
        if elapsed > budget:
            status = "measurement-timeout"
    elif status == "ok":
        status = "missing-result"
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
    return dict(job, execution=job["mode"], mean_s=mean(measured), median_s=median(measured),
                warmups=[{"total_s": seconds} for seconds in warm],
                measurements=[{"total_s": seconds} for seconds in measured],
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
        pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead")
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
    replay = model.__dict__.get("_replay") if jt is not None else None
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
    torch.manual_seed(job["seed"])
    task = WORKLOADS[job["case"]]
    workload = task(torch, "cuda", task.default_dtype, job["size"], seed=job["seed"])
    sync = Synchronizer(torch, job["runtime"], "cuda")
    start = perf_counter()
    workload.setup()
    jittor_compile = jt is not None and job["mode"] == "reduce-overhead"
    compile_inference = task.mode == "infer" and (job.get("torch_mode") == "compiled"
                                                  or (jittor_compile and task.compile_what is not None))
    compile_training = task.mode == "train" and job.get("torch_mode") == "compiled"
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

    warmups, index = warm_up(job, call, settled)
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
                settings=settings, stochastic=task.stochastic, checked_output_batches=checked_batches)


def output_tensors(torch, result, prefix="output"):
    if isinstance(result, torch.Tensor):
        yield prefix, result
    elif isinstance(result, Mapping):
        for name, item in result.items():
            yield from output_tensors(torch, item, f"{prefix}.{name}")
    elif isinstance(result, (list, tuple)):
        for index, item in enumerate(result):
            yield from output_tensors(torch, item, f"{prefix}.{index}")


def output_array(np, torch, tensor):
    value = tensor.detach().cpu()
    if value.dtype in (torch.bfloat16, torch.float16):
        value = value.float()
    array = value.numpy()
    if not np.isfinite(array).all():
        raise RuntimeError("Nonfinite inference output")
    return array


def run_model(job, directory, torch, jt, np):
    from importlib.metadata import version

    from worker import Synchronizer, configure, settled

    configure(torch, "cuda", False, False, job["runtime"])
    sync = Synchronizer(torch, job["runtime"], "cuda")
    start = perf_counter()
    forward, inputs, model, assets = prepare_model(torch, job["case"], job["model_assets"], job["seed"])
    residency = {"inputs": [cuda_residency(torch, jt, tensor) for tensor in inputs],
                 "weight": cuda_residency(torch, jt, next(model.parameters()))}
    if job["mode"] == "reduce-overhead":
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
    arrays = {key: output_array(np, torch, tensor) for key, tensor in outputs.items()}
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
                dependency=dependency, weights=str(assets) if assets else "deterministic initialization",
                setup_s=setup_s, warmups=warmups, measurements=measurements, residency=residency)


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
                     "--persistent-workers", "--shared-input-pipeline"]
        input_plan = {}
    else:
        input_plan = trace_nlp_inputs(job, steps)
        sys.argv += ["--batches", str(job["ranks"]), "--cycles", str(steps), "--metric", job["metric"]]
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
    torch, jt, guard = checked_runtime(job)
    before_fallback = jt.core.backend_fallback_count() if jt is not None else None
    with guard:
        if job["kind"] == "training":
            result = run_training_rank(job, directory, jt)
        elif job["kind"] == "stock":
            from workloads import WORKLOADS

            inference = WORKLOADS[job["case"]].mode == "infer"
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
    packages = versions(torch, job["runtime"])
    if job.get("torch_mode") == "compiled":
        if not job.get("native_compile"):
            raise RuntimeError("Torch compiled worker did not complete timing with a compiled graph")
        result["execution"] = "compiled"
    job.update(result, gpu=torch.cuda.get_device_name(), python=sys.executable, torch=torch.__version__,
               versions=packages,
               fallback_count=jt.core.backend_fallback_count() - before_fallback if jt is not None else None)
    write_json(directory / "result.json", job)


def compare_pair(reference, candidate):
    import numpy as np

    kind = reference["kind"]
    if kind == "training":
        keys = ("training_input_sha256", "initial_state_sha256")
        rank_pairs = list(zip(reference["rank_results"], candidate["rank_results"]))
        inputs_match = all(left.get(key) is not None and left[key] == right.get(key)
                           for left, right in rank_pairs for key in keys)
        loss_matches = all(np.allclose([row["loss"] for row in left["records"]],
                                       [row["loss"] for row in right["records"]], atol=0.02, rtol=0.03)
                           for left, right in rank_pairs)
        return dict(same_inputs=inputs_match, parity=loss_matches, comparable=inputs_match and loss_matches)
    dependencies_match = all(reference["versions"].get(key) == candidate["versions"].get(key)
                             for key in ("diffusers", "transformers"))
    if kind == "model":
        output = compare_model_outputs(np, reference, candidate)
        dependencies_match = dependencies_match and reference["dependency"] == candidate["dependency"]
        output.update(same_dependencies=dependencies_match,
                      comparable=output["parity"] and dependencies_match
                      and reference["weights"] == candidate["weights"])
        return output
    if kind == "wan":
        errors = {}
        first, second = Path(reference["directory"]), Path(candidate["directory"])
        names = {p.name for p in first.glob("*.npy")}
        if not names or names != {p.name for p in second.glob("*.npy")}:
            return dict(parity=False, comparable=False, parity_error="Output fields differ")
        for name in sorted(names):
            a, b = (np.load(directory / name, allow_pickle=False) for directory in (first, second))
            errors[name] = compare_arrays(np, a, b, 0.02, 0.08)
        parity = all(item["parity"] for item in errors.values())
        return dict(parity=parity, comparable=parity and dependencies_match, output_errors=errors,
                    same_dependencies=dependencies_match)
    inputs_match = reference["config"] == candidate["config"] and reference["seed"] == candidate["seed"]
    result = dict(same_dependencies=dependencies_match, same_inputs=inputs_match,
                  comparable=dependencies_match and inputs_match)
    if reference.get("checked_output_batches") and candidate.get("checked_output_batches"):
        result.update(compare_model_outputs(np, reference, candidate))
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
    references = [mode for mode in ("eager", "compiled") if mode in ok]
    if "eager" in ok and "compiled" in ok:
        compiled_check = compare_pair(ok["eager"], ok["compiled"])
        summary["torch_compile_check"] = compiled_check
        if not compiled_check["comparable"]:
            references.remove("compiled")
    candidates = [mode for mode in ("jittor_compiled", "jittor_eager", "jittor") if mode in ok]
    checks = {}
    if references:
        best = min(references, key=times.__getitem__)
        summary.update(torch_baseline_mode=best, torch_baseline_ms=times[best], torch_ms=times[best])
        reference = ok.get("eager", ok[best])
        checks = {mode: compare_pair(reference, ok[mode]) for mode in candidates}
        summary["jittor_checks"] = checks
    if candidates:
        verified = [mode for mode in candidates if checks.get(mode, {}).get("comparable")]
        # A refused capture ran eager. Prefer the separately measured eager mode.
        if "jittor_eager" in verified and ok.get("jittor_compiled", {}).get("execution") == "eager-fallback":
            verified = [mode for mode in verified if mode != "jittor_compiled"]
        selected = min(verified or candidates, key=times.__getitem__)
        summary.update(jittor_selected_mode=selected, jittor_ms=times[selected],
                       jittor_execution=ok[selected]["execution"])
        if checks:
            summary.update(checks[selected])
            if summary["comparable"]:
                summary["jittor_over_torch"] = times[selected] / times[best]
    return [summary]


def compare_arrays(np, a, b, atol, rtol):
    if a.shape != b.shape or a.dtype != b.dtype:
        return dict(parity=False, reference_shape=a.shape, candidate_shape=b.shape,
                    reference_dtype=str(a.dtype), candidate_dtype=str(b.dtype))
    equal = (np.allclose(a, b, atol=atol, rtol=rtol)
             if np.issubdtype(a.dtype, np.floating) else np.array_equal(a, b))
    max_abs = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0
    return dict(parity=bool(equal), max_abs=max_abs)


def compare_model_outputs(np, reference, candidate):
    ref = Path(reference["directory"]) / "outputs.npz"
    actual = Path(candidate["directory"]) / "outputs.npz"
    atol, rtol = {"float32": (1e-4, 1e-3), "float16": (5e-3, 3e-2),
                  "bfloat16": (2e-2, 8e-2)}[reference["dtype"]]
    errors = {}
    with np.load(ref, allow_pickle=False) as left, np.load(actual, allow_pickle=False) as right:
        if set(left.files) != set(right.files):
            return dict(parity=False, parity_error="Output fields differ")
        for key in left.files:
            errors[key] = compare_arrays(np, left[key], right[key], atol, rtol)
    return dict(parity=all(v["parity"] for v in errors.values()), output_errors=errors, atol=atol, rtol=rtol)


def main():
    if sys.argv[1:2] == ["--worker"]:
        run_worker(Path(sys.argv[2]))
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
    elif stock is not None and (stock["mode"] != "infer" or stock["compile_what"] is None):
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
    meta = {"state": "running", "pid": os.getpid(), "planned": args.case,
            "started": datetime.now(timezone.utc).isoformat()}

    def publish(current=None):
        meta["current"] = current
        write_json(output / "results.json", {"meta": meta, "runs": rows, "comparisons": comparisons})

    for name in args.case:
        executions = execution_order(args, name)
        rounds = args.rounds or (1 if len(args.case) > 1 and name not in WAN_CASES else 2)
        for round_id in range(1, 2 if name == VIDEO else rounds + 1):
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
                    row = run_one(args, output, job, budget)
                    budget -= row["measured_seconds"]
                rows.append(row)
                round_rows.append(row)
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
