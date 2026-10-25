"""Isolated real Engine rank launch; numerical rules live in the shared harness."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))
from _helpers.child_process import child_env, PYTHON


def file_rendezvous(directory):
    path = (Path(directory) / "process-group-store").resolve()
    if path.exists():
        raise FileExistsError("DeepSpeed rendezvous store already exists: " + str(path))
    return path.as_uri()


def reserve_case(root, stage, optimizer, dtype, world_size, perturbation, *, device="cuda"):
    if (stage not in (1, 2) or optimizer != "AdamW" or dtype != "bfloat16"
            or world_size != 2 or perturbation not in (0., .125) or device not in ("cpu", "cuda")):
        raise ValueError("Unverified DeepSpeed multirank protocol")
    root = Path(root).expanduser().resolve()
    repository = Path(__file__).resolve().parents[3]
    if root == repository or repository in root.parents:
        raise ValueError("DeepSpeed artifacts must be outside the checkout")
    case = root / device / ("zero%d" % stage) / optimizer / dtype / ("world%d" % world_size) / (
        "perturb-%g" % perturbation)
    # Exclusive creation also rejects an empty pre-existing case. Partial and
    # successful runs must both retain their evidence on a subsequent launch.
    case.mkdir(parents=True, exist_ok=False)
    (case / "protocol.json").write_text(json.dumps({
        "zero_stage": stage, "optimizer": optimizer, "dtype": dtype,
        "world_size": world_size, "perturbation": perturbation, "device": device,
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


def _rank_environment(runtime, rank, gpu, cache_root, package_site, *, device="cuda"):
    if device not in ("cpu", "cuda") or runtime not in ("jittor", "torch") or rank not in (0, 1):
        raise ValueError("Unverified DeepSpeed rank environment")
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
        "CUDA_VISIBLE_DEVICES": gpu if device == "cuda" else "", "OMP_NUM_THREADS": threads,
        "MAX_JOBS": threads,
        "MKL_NUM_THREADS": threads, "OPENBLAS_NUM_THREADS": threads,
        "NUMEXPR_NUM_THREADS": threads, "BLIS_NUM_THREADS": threads,
        "VECLIB_MAXIMUM_THREADS": threads, "DS_ACCELERATOR": device,
        "RANK": str(rank), "WORLD_SIZE": "2",
        "LOCAL_RANK": str(rank) if runtime == "jittor" and device == "cpu" else "0",
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
            "JITTOR_TORCH_DISTRIBUTED_AUTO_INIT": "0",
            "JT_BUILD_USE_MPI": "1" if device == "cpu" else "0",
            "JT_BUILD_USE_CUTT": "0", "JT_BUILD_USE_NCCL": "1" if device == "cuda" else "0",
            "DISABLE_MULTIPROCESSING": "1", "JT_USE_PARALLEL_OP_COMPILER": "0",
        })
        environment.pop("use_nccl", None)
        environment.pop("use_cutt", None)
        if device == "cpu":
            environment.update(JT_BUILD_NVCC_PATH="", nvcc_path="", JT_USE_CUDA="0", use_cuda="0")
        else:
            environment.update(JT_USE_CUDA="1", use_cuda="1")
    return environment


def pair_environments(runtime, devices, cache_root, package_site, directory, *, device="cuda"):
    assignments = devices if device == "cuda" else ("", "")
    environments = [_rank_environment(runtime, rank, gpu, cache_root, package_site, device=device)
                    for rank, gpu in enumerate(assignments)]
    for environment in environments:
        if device == "cuda":
            environment["JITTOR_DEEPSPEED_DEVICES"] = ",".join(devices)
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


_WARM_NATIVE_CPU = """
import jittor as jt
from jittor._runtime import forbid_backend_fallbacks
jt.flags.use_cuda = 0
assert jt.compile_extern.has_mpi, 'DeepSpeed CPU ranks require the actual MPI provider'
initial = jt.core.backend_fallback_count()
with forbid_backend_fallbacks():
    value = jt.ones((2, 8)).bfloat16()
    weight = jt.ones((4, 8)).bfloat16()
    bias = jt.ones((4,)).bfloat16()
    result = jt.nn.layer_norm(jt.nn.linear(value, weight, bias), (4,))
    gradients = jt.grad(result.float32().sqr().sum(), [value, weight, bias])
    for tensor in (value, weight, bias, result, *gradients):
        tensor.sync()
        assert tensor.location() == 'cpu'
    assert jt.core.backend_fallback_count() == initial
print('DeepSpeed serial rank warmup: CPU BF16, MPI, fallback delta 0')
"""


_WARM_TORCH_CPU = """
import sys
sys.path.insert(0, sys.argv[1])
from _ecosystem_runner import _import_torch
torch = _import_torch('torch')
from deepspeed.ops.op_builder.cpu.comm import ShareMemCommBuilder
builder = ShareMemCommBuilder()
if builder.is_compatible():
    module = builder.load()
    assert module.__file__.endswith(('.so', '.pyd'))
    print('DeepSpeed serial CPU oracle extension', module.__file__)
"""


def _runner_arguments(protocol, runtime, directory, rendezvous, rank):
    runner = Path(__file__).with_name("_ecosystem_deepspeed_runner.py")
    return [str(runner), "--runtime", runtime, "--device", protocol["device"],
            "--optimizer", protocol["optimizer"], "--dtype", protocol["dtype"],
            "--world-size", str(protocol["world_size"]), "--rank", str(rank),
            "--zero-stage", str(protocol["zero_stage"]),
            "--perturbation", str(protocol["perturbation"]),
            "--rendezvous", rendezvous, "--checkpoint", str(directory / "checkpoint"),
            "--output", str(directory / ("rank%d.npz" % rank))]


def _rank_main(arguments, launcher):
    """Isolate caches after the real launcher has assigned rank and device."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--" + launcher + "-case", type=Path, required=True, dest="case")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--package-site", default="")
    parser.add_argument("--rendezvous", required=True)
    args = parser.parse_args(arguments)
    device = "cpu" if launcher == "mpi" else "cuda"
    if launcher == "mpi":
        if os.environ.get("OMPI_COMM_WORLD_SIZE") != "2":
            raise ValueError("The verified CPU launcher requires two actual OpenMPI ranks")
        rank_text = os.environ.get("OMPI_COMM_WORLD_RANK")
        if rank_text not in ("0", "1") or os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK") != rank_text:
            raise ValueError("The verified CPU launcher requires distinct local OpenMPI ranks")
    else:
        rank_text = os.environ.get("RANK")
        if (os.environ.get("WORLD_SIZE") != "2" or rank_text not in ("0", "1")
                or os.environ.get("LOCAL_RANK") != "0"
                or os.environ.get("JT_NCCL_WORLD_SIZE") != "2"
                or os.environ.get("JT_NCCL_RANK") != rank_text
                or os.environ.get("JT_NCCL_LOCAL_RANK") != "0"
                or not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
                or "," in os.environ.get("CUDA_VISIBLE_DEVICES", "")):
            raise ValueError("The verified CUDA launcher requires two actual jtrun NCCL ranks")
    rank = int(rank_text)
    directory = args.case.resolve()
    protocol = json.loads((directory.parent / "protocol.json").read_text())
    if protocol["device"] != device:
        raise ValueError("Engine launcher differs from the requested protocol device")
    environment = _rank_environment("jittor", rank, os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                                    args.cache_root, args.package_site, device=device)
    if launcher == "jtrun":
        # Do not replace the public launcher's rank/device/store controls. The
        # wrapper execs the actual Engine rank, so the launcher-owned TCPStore
        # must remain reachable for Jittor's NCCL initialization in that rank.
        required = ("MASTER_ADDR", "MASTER_PORT", "JITTOR_LAUNCHER_RUN_ID")
        if (os.environ.get("JITTOR_LAUNCHER_EXTERNAL_STORE") != "1"
                or any(not os.environ.get(name) for name in required)):
            raise ValueError("jtrun must provide its external store endpoint to Engine ranks")
        environment.update({name: value for name, value in os.environ.items()
                            if name.startswith("JT_NCCL_") or name in required
                            or name == "JITTOR_LAUNCHER_EXTERNAL_STORE"})
    else:
        environment.pop("MASTER_ADDR", None)
        environment.pop("MASTER_PORT", None)
    control = directory / "rendezvous"
    if not control.is_dir():
        raise ValueError("Engine ranks require their shared case control plane")
    if launcher == "jtrun" and os.environ.get("JITTOR_DIST_RENDEZVOUS_DIR") != str(control):
        raise ValueError("jtrun must own the shared communicator control plane")
    environment["JITTOR_DIST_RENDEZVOUS_DIR"] = str(control)
    environment["JITTOR_DEEPSPEED_LAUNCHER"] = (
        "jtrun-module" if launcher == "jtrun" else "mpi-backend-verification")
    command = _runner_arguments(protocol, "jittor", directory, args.rendezvous, rank)
    # An exclusive claim protects logs/outputs even when this entry is called
    # outside run_pair. Record placement controls without persisting secrets
    # from the inherited environment.
    controls = ("HOME", "TMPDIR", "JITTOR_HOME", "RANK", "WORLD_SIZE", "LOCAL_RANK",
                "CUDA_VISIBLE_DEVICES", "JITTOR_DIST_RENDEZVOUS_DIR", "JT_BUILD_USE_MPI",
                "JT_BUILD_USE_NCCL", "JITTOR_DEEPSPEED_LAUNCHER")
    with (directory / ("rank%d-launch.json" % rank)).open("x") as handle:
        json.dump({"command": command, "environment": {key: environment[key] for key in controls}},
                  handle, indent=2, sort_keys=True)
    with (directory / ("rank%d.log" % rank)).open("x") as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        os.execve(PYTHON, [PYTHON, *command], child_env(environment, inherit=False))


def _mpi_rank_main(arguments):
    _rank_main(arguments, "mpi")


def _jtrun_rank_main(arguments):
    _rank_main(arguments, "jtrun")


def run_pair(python, runtime, case, cache_root, devices, package_site, timeout=1800):
    protocol = json.loads((case / "protocol.json").read_text())
    device = protocol["device"]
    if runtime not in ("jittor", "torch") or device not in ("cpu", "cuda"):
        raise ValueError("Unverified DeepSpeed runtime or device")
    if device == "cuda" and (len(devices) != 2 or len(set(devices)) != 2 or not all(devices)):
        raise ValueError("DeepSpeed requires two distinct allocated CUDA devices")
    if device == "cpu" and devices:
        raise ValueError("CPU Engine ranks must not allocate CUDA devices")
    mpirun = None
    if runtime == "jittor" and device == "cpu":
        mpirun = shutil.which(os.environ.get("JITTOR_MPIEXEC", "mpirun"))
        if mpirun is None:
            raise ValueError("DeepSpeed CPU ranks require a real OpenMPI mpirun")
    directory = case / runtime
    directory.mkdir(exist_ok=False)
    environments = pair_environments(runtime, devices, cache_root, package_site, directory, device=device)
    # Every rank owns a cache. Build core/extensions in sequence before any
    # collective rank launch, including a cold first invocation of this gate.
    if runtime == "jittor" or device == "cpu":
        for rank, environment in enumerate(environments):
            warm_environment = dict(environment, WORLD_SIZE="1", RANK="0",
                                    LOCAL_RANK="0", JITTOR_TORCH_DISTRIBUTED_AUTO_INIT="0")
            if device == "cuda":
                warm_environment.update(JT_NCCL_WORLD_SIZE="1", JT_NCCL_RANK="0", JT_NCCL_LOCAL_RANK="0")
            code = (_WARM_NATIVE if device == "cuda" else _WARM_NATIVE_CPU) if runtime == "jittor" else _WARM_TORCH_CPU
            with (directory / ("warm-rank%d.log" % rank)).open("w") as log:
                subprocess.run([python, "-c", code, str(Path(__file__).parent)], check=True, timeout=timeout,
                               stdout=log, stderr=subprocess.STDOUT,
                               env=child_env(warm_environment, inherit=False,
                                             repo_paths=(runtime == "jittor"),
                                             without_torch_mode=(runtime == "torch")))
    rendezvous = file_rendezvous(directory)
    outputs = [directory / ("rank%d.npz" % rank) for rank in (0, 1)]
    if runtime == "jittor":
        if device == "cpu":
            command = [mpirun, "--bind-to", "none", "-n", "2", python,
                       "-S", str(Path(__file__).resolve()), "--mpi-case", str(directory)]
            launch_environment = child_env(os.environ.copy(), inherit=False)
            path = directory / "mpi-launch.log"
        else:
            # The documented -m entry is the console jtrun owner. -S keeps
            # site hooks from importing Jittor before the wrapper picks its
            # private rank cache. jtrun owns all rank/device assignments.
            command = [python, "-m", "jittor.distributed.launch", "--standalone",
                       "--backend", "nccl", "--nproc-per-node", "2", "--device-ids", ",".join(devices),
                       "--timeout", str(timeout), "--log-dir", str(directory / "rendezvous"),
                       "--", python, "-S", str(Path(__file__).resolve()),
                       "--jtrun-case", str(directory)]
            launch_environment = child_env(dict(environments[0], WORLD_SIZE="1", RANK="0",
                                               CUDA_VISIBLE_DEVICES=",".join(devices)), inherit=False)
            path = directory / "jtrun-launch.log"
        command += ["--cache-root", str(cache_root), "--package-site", package_site or "",
                    "--rendezvous", rendezvous]
        (directory / "launcher.json").write_text(json.dumps({"command": command}, indent=2) + "\n")
        with path.open("x") as log:
            process = subprocess.Popen(
                command,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                env=launch_environment)
            try:
                wait_for_ranks([process], [path], timeout)
            except (RuntimeError, TimeoutError) as error:
                details = "\n".join(output.with_suffix(".log").read_text(errors="replace")[-10000:]
                                    for output in outputs if output.with_suffix(".log").is_file())
                raise type(error)(str(error) + "\n" + details) from error
        if not all(output.is_file() for output in outputs):
            raise AssertionError("Engine ranks omitted numerical observations")
        return outputs
    processes, logs, handles, outputs = [], [], [], []
    try:
        for rank, environment in enumerate(environments):
            output = directory / ("rank%d.npz" % rank)
            path = directory / ("rank%d.log" % rank)
            log = path.open("w")
            logs.append(path)
            handles.append(log)
            outputs.append(output)
            arguments = _runner_arguments(protocol, runtime, directory, rendezvous, rank)
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


if __name__ == "__main__":
    if "--jtrun-case" in sys.argv:
        _jtrun_rank_main(sys.argv[1:])
    else:
        _mpi_rank_main(sys.argv[1:])
