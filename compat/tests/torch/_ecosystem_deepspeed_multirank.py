"""Isolated real Engine rank launch; numerical rules live in the shared harness."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from _helpers.child_process import child_env


def file_rendezvous(directory):
    path = (Path(directory) / "process-group-store").resolve()
    if path.exists():
        raise FileExistsError("DeepSpeed rendezvous store already exists: " + str(path))
    return path.as_uri()


def reserve_case(root, stage, optimizer, dtype, world_size, perturbation):
    if (stage not in (1, 2) or optimizer != "AdamW" or dtype != "bfloat16"
            or world_size != 2 or perturbation not in (0., .125)):
        raise ValueError("Unverified DeepSpeed multirank protocol")
    root = Path(root).expanduser().resolve()
    repository = Path(__file__).resolve().parents[3]
    if root == repository or repository in root.parents:
        raise ValueError("DeepSpeed artifacts must be outside the checkout")
    case = root / ("zero%d" % stage) / optimizer / dtype / ("world%d" % world_size) / (
        "perturb-%g" % perturbation)
    # Exclusive creation also rejects an empty pre-existing case. Partial and
    # successful runs must both retain their evidence on a subsequent launch.
    case.mkdir(parents=True, exist_ok=False)
    (case / "protocol.json").write_text(json.dumps({
        "zero_stage": stage, "optimizer": optimizer, "dtype": dtype,
        "world_size": world_size, "perturbation": perturbation,
    }, indent=2, sort_keys=True) + "\n")
    return case


def _terminate(process):
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        process.wait()


def wait_for_ranks(processes, logs, timeout):
    deadline = time.monotonic() + timeout
    try:
        while True:
            statuses = [process.poll() for process in processes]
            failed = next((i for i, code in enumerate(statuses) if code not in (None, 0)), None)
            if failed is not None:
                raise RuntimeError("DeepSpeed rank %d failed (%s):\n%s" % (
                    failed, statuses[failed], Path(logs[failed]).read_text(errors="replace")[-10000:]))
            if all(code == 0 for code in statuses):
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("DeepSpeed ranks exceeded timeout; logs: %s" % list(map(str, logs)))
            time.sleep(.1)
    finally:
        for process in processes:
            _terminate(process)


def _rank_environment(runtime, rank, gpu, cache_root, package_site):
    environment = os.environ.copy()
    for name in list(environment):
        if name.startswith("JT_NCCL_") or (runtime == "torch" and name.startswith(("JT_", "JITTOR_"))):
            environment.pop(name)
    if runtime == "torch":
        environment.pop("LD_LIBRARY_PATH", None)
        environment.pop("REAL_TORCH_SITE", None)
    state = (Path(cache_root) / runtime / ("rank%d" % rank)).resolve()
    repository = Path(__file__).resolve().parents[3]
    if state == repository or repository in state.parents:
        raise ValueError("DeepSpeed rank caches must be outside the checkout")
    for name in ("home", "tmp", "jittor-home", "lab", "xdg-cache", "cuda-cache", "torch-extensions"):
        (state / name).mkdir(parents=True, exist_ok=True)
    threads = environment.get("OMP_NUM_THREADS", "4")
    environment.update({
        "HOME": str(state / "home"), "TMPDIR": str(state / "tmp"),
        "XDG_CACHE_HOME": str(state / "xdg-cache"), "CUDA_CACHE_PATH": str(state / "cuda-cache"),
        "TORCH_EXTENSIONS_DIR": str(state / "torch-extensions"),
        "CUDA_VISIBLE_DEVICES": gpu, "OMP_NUM_THREADS": threads,
        "MKL_NUM_THREADS": threads, "OPENBLAS_NUM_THREADS": threads,
        "NUMEXPR_NUM_THREADS": threads, "BLIS_NUM_THREADS": threads,
        "VECLIB_MAXIMUM_THREADS": threads, "DS_ACCELERATOR": "cuda",
        "RANK": str(rank), "WORLD_SIZE": "2", "LOCAL_RANK": "0",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    if package_site:
        environment["JITTOR_ECOSYSTEM_PACKAGE_SITE"] = package_site
    else:
        environment.pop("JITTOR_ECOSYSTEM_PACKAGE_SITE", None)
    if runtime == "jittor":
        environment.update({
            "JITTOR_HOME": str(state / "jittor-home"),
            "JITTOR_LAB_ROOT": str(state / "lab"), "JITTOR_TORCH_SHIM": "1",
            "JT_BUILD_CACHE_NAME": "deepspeed-rank%d" % rank,
            "cache_name": "deepspeed-rank%d" % rank,
            "JITTOR_TORCH_DISTRIBUTED_AUTO_INIT": "1",
            "JT_BUILD_USE_MPI": "0", "JT_BUILD_USE_CUTT": "0", "JT_BUILD_USE_NCCL": "1",
            "DISABLE_MULTIPROCESSING": "1", "JT_USE_PARALLEL_OP_COMPILER": "0",
        })
        environment.pop("use_nccl", None)
        environment.pop("use_cutt", None)
    return environment


def pair_environments(runtime, devices, cache_root, package_site, directory):
    environments = [_rank_environment(runtime, rank, gpu, cache_root, package_site)
                    for rank, gpu in enumerate(devices)]
    for environment in environments:
        # FileStore supplies the process-group bootstrap. Inherited TCP store
        # endpoints must not redirect any library to an unrelated launcher.
        environment.pop("MASTER_ADDR", None)
        environment.pop("MASTER_PORT", None)
    if runtime == "jittor":
        # This is the shared communicator control plane, including watchdog
        # heartbeats and subgroup IDs. It must not live in private rank caches.
        rendezvous = Path(directory) / "rendezvous"
        rendezvous.mkdir(parents=True, exist_ok=False)
        for environment in environments:
            environment["JITTOR_DIST_RENDEZVOUS_DIR"] = str(rendezvous)
    return environments


_WARM_NATIVE = """
import jittor as jt
from jittor._runtime import forbid_backend_fallbacks
jt.flags.use_cuda = 1
jt.compile_extern.setup_nccl()
assert jt.compile_extern.nccl_ops is not None
initial = jt.core.backend_fallback_count()
with forbid_backend_fallbacks():
    value = jt.ones((2, 3))
    result = value.sum()
    value.sync(); result.sync()
    assert value.location() == result.location() == 'device'
    assert jt.core.backend_fallback_count() == initial
print('DeepSpeed serial rank warmup: CUDA residency, NCCL, fallback delta 0')
"""


def run_pair(python, runtime, case, cache_root, devices, package_site, timeout=1800):
    if runtime not in ("jittor", "torch") or len(devices) != 2 or len(set(devices)) != 2:
        raise ValueError("DeepSpeed requires two distinct allocated CUDA devices")
    protocol = json.loads((case / "protocol.json").read_text())
    directory = case / runtime
    directory.mkdir(exist_ok=False)
    environments = pair_environments(runtime, devices, cache_root, package_site, directory)
    # Every rank owns a cache. Build core/extensions in sequence before any
    # collective rank launch, including a cold first invocation of this gate.
    if runtime == "jittor":
        for rank, environment in enumerate(environments):
            warm_environment = dict(environment, JT_NCCL_WORLD_SIZE="1", JT_NCCL_RANK="0",
                                    JT_NCCL_LOCAL_RANK="0", WORLD_SIZE="1", RANK="0",
                                    JITTOR_TORCH_DISTRIBUTED_AUTO_INIT="0")
            with (directory / ("warm-rank%d.log" % rank)).open("w") as log:
                subprocess.run([python, "-c", _WARM_NATIVE], check=True, timeout=timeout,
                               stdout=log, stderr=subprocess.STDOUT,
                               env=child_env(warm_environment, inherit=False))
    rendezvous = file_rendezvous(directory)
    runner = Path(__file__).with_name("_ecosystem_deepspeed_runner.py")
    processes, logs, handles, outputs = [], [], [], []
    try:
        for rank, environment in enumerate(environments):
            output = directory / ("rank%d.npz" % rank)
            path = directory / ("rank%d.log" % rank)
            log = path.open("w")
            logs.append(path)
            handles.append(log)
            outputs.append(output)
            arguments = [str(runner), "--runtime", runtime, "--device", "cuda",
                         "--optimizer", protocol["optimizer"], "--dtype", protocol["dtype"],
                         "--world-size", str(protocol["world_size"]), "--rank", str(rank),
                         "--zero-stage", str(protocol["zero_stage"]),
                         "--perturbation", str(protocol["perturbation"]),
                         "--rendezvous", rendezvous, "--checkpoint", str(directory / "checkpoint"),
                         "--output", str(output)]
            processes.append(subprocess.Popen(
                [python, *arguments], stdout=log, stderr=subprocess.STDOUT,
                start_new_session=(os.name == "posix"),
                env=child_env(environment, inherit=False, repo_paths=(runtime == "jittor"),
                              without_torch_mode=(runtime == "torch"))))
        wait_for_ranks(processes, logs, timeout)
    finally:
        for process in processes:
            _terminate(process)
        for handle in handles:
            handle.close()
    if not all(path.is_file() for path in outputs):
        raise AssertionError("DeepSpeed ranks omitted their numerical observations")
    return outputs
