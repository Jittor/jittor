"""Fresh-process roundtrip of verl's actual RNG checkpoint helper on NPU.

This saves CPU/NPU/Python/NumPy RNG state, not model/optimizer/scheduler state.
It does not instantiate TrainingWorker or establish complete training resume.
Run baseline and resume separately under each runtime. All state/arrays remain
on the execution server; JSON contains only provenance and equality summaries.
"""
import argparse
from contextlib import ExitStack
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import random
import sys
import uuid

import numpy as np

from probe_training_workflow import initialize


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_draws(values):
    required = {"cpu_uniform", "cpu_normal", "npu_uniform", "npu_normal", "python", "numpy"}
    if set(values) != required:
        raise ValueError("incomplete RNG streams")
    for name in sorted(required):
        value = np.asarray(values[name])
        size = 9 if name.endswith("normal") or name == "numpy" else 7
        dtype = np.dtype("float64" if name in ("python", "numpy") else "float32")
        if (value.shape != (size,) or value.dtype != dtype
                or not np.isfinite(value).all() or np.unique(value).size < 2):
            raise ValueError("invalid RNG sample: " + name)
    return sorted(required)


def compare_draws(expected, actual):
    streams = validate_draws(expected)
    validate_draws(actual)
    for name in streams:
        if not np.array_equal(expected[name], actual[name]):
            raise ValueError("RNG continuation mismatch: " + name)
    return streams


def require_advanced(before, after):
    streams = validate_draws(before)
    validate_draws(after)
    for name in streams:
        if np.array_equal(before[name], after[name]):
            raise ValueError("RNG stream did not advance: " + name)
    return streams


def validate_continuation(expected, unrestored, restored):
    # A matching suffix alone is insufficient if restoring state did nothing.
    require_advanced(unrestored, expected)
    return compare_draws(expected, restored)


def provider_identity(function):
    """Observe the actual callable binding, never install or invoke it."""
    if not callable(function):
        raise TypeError("RNG provider must be callable")
    name = getattr(function, "__qualname__", None)
    module_name = getattr(function, "__module__", None)
    if not module_name:
        module_name = type(getattr(function, "__self__", None)).__module__
    path = None
    if inspect.isfunction(function) or inspect.ismethod(function):
        # Cython exposes function-like RNG bindings which inspect.getfile
        # explicitly rejects; their loaded extension remains inspectable.
        try:
            path = inspect.getsourcefile(function)
        except TypeError:
            path = None
    if path and Path(path).is_file():
        kind = "python_source"
    else:
        owner = sys.modules.get(module_name)
        path = getattr(owner, "__file__", None)
        kind = "loaded_module_artifact"
    if not name or not path or not Path(path).is_file():
        raise RuntimeError("RNG provider has no inspectable source/artifact: " + str(name))
    return dict(module=module_name, callable=name, artifact_kind=kind,
                artifact_sha256=digest(path))


def runtime_identity(torch, jt):
    versions = dict(python=sys.version, python_implementation=sys.implementation.name,
                    numpy=np.__version__, torch=str(torch.__version__))
    if jt is not None:
        versions["jittor"] = str(jt.__version__)
        native = jt.core
    else:
        # The official runtime must already have loaded torch_npu in initialize.
        versions["torch_npu"] = importlib.metadata.version("torch_npu")
        native = torch._C
    functions = dict(cpu_get=torch.get_rng_state, cpu_set=torch.set_rng_state,
                     npu_get=torch.npu.get_rng_state, npu_set=torch.npu.set_rng_state,
                     uniform=torch.rand, normal=torch.randn,
                     python_get=random.getstate, python_set=random.setstate,
                     numpy_get=np.random.get_state, numpy_set=np.random.set_state)
    if jt is not None:
        functions.update(native_cpu_get=jt.core.get_cpu_rng_state,
                         native_cpu_set=jt.core.set_cpu_rng_state)
    path = getattr(native, "__file__", None)
    if not path or not Path(path).is_file():
        raise RuntimeError("RNG runtime has no inspectable native core artifact")
    return dict(versions=versions, native_core_sha256=digest(path),
                providers={name: provider_identity(fn) for name, fn in functions.items()})


def validate_resume_provenance(receipt, protocol, pid, token):
    if not all(receipt.get(key) == value for key, value in protocol.items()):
        raise ValueError("RNG baseline protocol differs")
    baseline_pid = receipt.get("baseline_pid")
    if type(baseline_pid) is not int or baseline_pid <= 0 or baseline_pid == pid:
        raise ValueError("RNG resume must use a different process on the same execution host")
    if not isinstance(receipt.get("baseline_token"), str) or receipt["baseline_token"] == token:
        raise ValueError("RNG baseline process token invalid")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=("torch", "jittor"), required=True)
    parser.add_argument("--phase", choices=("baseline", "resume"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.device = "npu"
    if args.output.exists():
        parser.error("output must be a fresh path")
    if args.phase == "baseline":
        args.run_dir.mkdir(parents=True, exist_ok=False)
    elif not args.run_dir.is_dir():
        parser.error("resume requires the original baseline directory")
    with ExitStack() as stack:
        torch, jt, _, synchronize, _ = initialize(args, stack)
        torch.npu.set_device(args.device_index)
        # Preserve the already validated TrainingWorker import sequence. In
        # particular HF's optional-dependency discovery must run in the native
        # oracle environment before the shared site becomes visible.
        from transformers import Qwen3ForCausalLM, Qwen3ForTokenClassification, ProcessorMixin  # noqa: F401
        # Match the worker harness: load the independent native runtime first,
        # then append an explicitly supplied shared dependency site for verl.
        # Never prepend it or re-resolve torch from the candidate environment.
        extra = os.environ.get("VERL_EXTRA_SITE_PACKAGES")
        if extra:
            if not Path(extra).is_dir():
                raise ValueError("VERL_EXTRA_SITE_PACKAGES must be an existing directory")
            if extra not in sys.path:
                sys.path.append(extra)
        from verl.workers.engine_workers import TrainingWorker  # noqa: F401: import only, never instantiated
        from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager
        from verl.utils.device import get_device_name, is_device_available

        assert get_device_name() == "npu" and is_device_available()
        assert importlib.metadata.version("verl") == "0.9.1"
        before_fallback = int(jt.core.backend_fallback_count()) if jt is not None else None
        token = uuid.uuid4().hex
        protocol = dict(schema=2, framework=args.framework, device_index=args.device_index,
                        verl_version="0.9.1", script_sha256=digest(__file__),
                        runtime_helper_sha256=digest(inspect.getfile(initialize)),
                        manager_sha256=digest(inspect.getfile(BaseCheckpointManager)),
                        runtime=runtime_identity(torch, jt),
                        scope="actual verl RNG helper only; no model/optimizer/worker resume")
        state_path, expected_path, receipt_path = [args.run_dir / name for name in
            ("rng-state.pt", "expected.npz", "receipt.json")]

        def draw():
            values = {}
            for device in ("cpu", "npu:%d" % args.device_index):
                for distribution, size in (("uniform", 7), ("normal", 9)):
                    value = (torch.rand if distribution == "uniform" else torch.randn)(
                        size, device=device, dtype=torch.float32)
                    synchronize()
                    assert value.device.type == device.split(":")[0]
                    if device != "cpu":
                        assert value.device.index == args.device_index
                    if jt is not None:
                        native = getattr(value, "_var", value)
                        assert native.location() == ("cpu" if device == "cpu" else "device")
                        if device != "cpu":
                            assert native.device_id == args.device_index
                    values[device.split(":")[0] + "_" + distribution] = value.detach().cpu().numpy().copy()
            values["python"] = np.asarray([random.random() for _ in range(7)])
            values["numpy"] = np.random.normal(size=9)
            return values

        torch.manual_seed(1701 if args.phase == "baseline" else 92)
        torch.npu.manual_seed(1701 if args.phase == "baseline" else 92)
        random.seed(1701 if args.phase == "baseline" else 92)
        np.random.seed(1701 if args.phase == "baseline" else 92)
        prefix = draw()  # Capture only after every stream has actually advanced.
        if args.phase == "baseline":
            state = BaseCheckpointManager.get_rng_state()
            assert set(state) == {"cpu", "npu", "numpy", "random"}
            for name in ("cpu", "npu"):
                assert state[name].dtype == torch.uint8 and state[name].device.type == "cpu"
            torch.save(state, state_path)
            expected = draw()
            advanced = require_advanced(prefix, expected)
            with expected_path.open("xb") as stream:
                np.savez(stream, **expected)
            receipt = dict(protocol, baseline_token=token, baseline_pid=os.getpid(), state_sha256=digest(state_path),
                           expected_sha256=digest(expected_path), advanced_streams=advanced)
            with receipt_path.open("x") as stream:
                json.dump(receipt, stream, indent=2)
            streams = validate_draws(expected)
            status = "baseline_recorded"
        else:
            receipt = json.loads(receipt_path.read_text())
            validate_resume_provenance(receipt, protocol, os.getpid(), token)
            assert digest(state_path) == receipt["state_sha256"]
            assert digest(expected_path) == receipt["expected_sha256"]
            with np.load(expected_path, allow_pickle=False) as saved:
                expected = dict(saved)
            if receipt.get("advanced_streams") != validate_draws(expected):
                raise ValueError("baseline RNG advancement evidence is incomplete")
            unrestored = draw()
            advanced = require_advanced(prefix, unrestored)
            require_advanced(unrestored, expected)
            # Only load the task-owned file whose hash was checked above.
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            assert set(state) == {"cpu", "npu", "numpy", "random"}
            BaseCheckpointManager.load_rng_state(state)
            streams = validate_continuation(expected, unrestored, draw())
            status = "fresh_process_rng_continuation_exact"
        assert torch.npu.current_device() == args.device_index
        fallback = int(jt.core.backend_fallback_count()) - before_fallback if jt is not None else None
        assert fallback in (0, None)
        result = dict(protocol, phase=args.phase, status=status, process_token=token, process_pid=os.getpid(),
                      baseline_token=receipt["baseline_token"], checked_streams=streams,
                      advanced_streams=advanced,
                      fallback_count=fallback, oracle_fallback_instrumented=False)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
