"""Execute one downstream-library case in whichever runtime owns ``torch``.

Invoked as a subprocess by ``test_ecosystem_parity.py``::

    python _ecosystem_runner.py <case> <output.npz> [--weights weights.npz]

Without ``--weights`` the runner builds the model, saves its state dict next to
the result and reports the reference numbers.  With ``--weights`` it loads that
exact state before running, so the two runtimes differ only in operator
semantics.  The measured wall time is reported too, because the 2.0 goal asks
for parity *and* for no speed regression.
"""

import argparse
from contextlib import ExitStack, nullcontext
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

# This helper is also launched directly, outside pytest's path bootstrap.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))
from _helpers import capability as _test_capability

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import _ecosystem_cases  # noqa: E402


def _activate_package_site():
    raw_site = os.environ.get("JITTOR_ECOSYSTEM_PACKAGE_SITE", "").strip()
    if not raw_site:
        return ""
    site = Path(raw_site).expanduser().resolve()
    if not site.is_dir():
        raise SystemExit(
            "JITTOR_ECOSYSTEM_PACKAGE_SITE is not a directory: {}".format(site)
        )
    site_text = str(site)
    sys.path[:] = [entry for entry in sys.path if entry != site_text]
    sys.path.insert(0, site_text)
    return site_text


def _dependency_report(requirements):
    report = {}
    for name in requirements:
        module = importlib.import_module(name)
        report[name] = {
            "version": str(getattr(module, "__version__", "unknown")),
            "origin": str(Path(getattr(module, "__file__", "")).resolve()),
        }
    return report


def _import_torch(runtime):
    """Return the ``torch`` module for the requested runtime.

    Jittor claims the ``torch`` namespace from inside its own import, and it
    refuses to install over a Torch module graph that already exists.  So the
    shim runtime has to import Jittor *first*; importing the deployed ``torch``
    package first would leave a half-initialized module for Jittor to reject.
    """
    if runtime == "jittor":
        os.environ["JITTOR_TORCH_SHIM"] = "1"
        import jittor  # noqa: F401

        import torch

        if getattr(torch, "__name__", None) != "jittor" and not hasattr(
            torch, "_torch_compat_install_context"
        ):
            raise SystemExit("torch did not resolve to the Jittor shim")
        _activate_package_site()
        return torch

    os.environ.pop("JITTOR_TORCH_SHIM", None)
    import torch

    if not hasattr(torch, "_C") or hasattr(torch, "_torch_compat_install_context"):
        raise SystemExit("torch did not resolve to an independent PyTorch")
    # Claim torchvision from the real-Torch environment before the shared
    # Python package site can expose Jittor's deployed torchvision facade.
    try:
        import torchvision  # noqa: F401
    except Exception:
        # torchvision is optional for the transformer cases.  Some reference
        # environments ship a mismatched torchvision wheel whose import raises
        # RuntimeError while registering compiled operators; that must not
        # prevent unrelated torch-only models from running.
        pass
    _activate_package_site()
    return torch


def _to_device_callable():
    """Move a host-backed tensor onto Jittor's selected accelerator.

    Jittor's global device flag covers tensors it creates itself; a tensor built
    from a host array keeps its host residency, so it must be moved explicitly
    before it can take part in a device-resident graph. Real PyTorch's
    ``from_numpy`` is host-resident for the same reason and the caller moves it.
    """
    def move(tensor):
        cuda = getattr(tensor, "cuda", None)
        return cuda() if callable(cuda) else tensor

    return move


def _select_device(torch, runtime, device, *, policy_stack=None):
    """Keep default placement selected for the caller-owned workload lifetime."""
    if runtime == "jittor":
        import jittor as jt
        if policy_stack is None:
            raise RuntimeError("Jittor device selection requires a caller-owned policy stack")
        if device == "cuda":
            if not _test_capability.check_accelerator("cuda", backend=jt).enabled:
                raise SystemExit("CUDA is unavailable in this Jittor build")
            policy_stack.enter_context(jt.runtime.scope(use_cuda=1))
            # Turning the global device flag on places tensors Jittor *creates*,
            # which is why the CPU branch below can stay identity. It does not
            # relocate a host-backed tensor: `from_numpy` stays host-resident
            # (exactly as real PyTorch's `from_numpy` returns a CPU tensor), and
            # an op that mixes it with a device-resident Parameter fails
            # dispatch_context's same-backend check. Real PyTorch has the same
            # residency and the caller moves it, so move it here too.
            return _to_device_callable()
        elif device == "npu":
            if not _test_capability.check_accelerator("acl", backend=jt).enabled:
                raise SystemExit("ACL is unavailable in this Jittor build")
            policy_stack.enter_context(jt.runtime.scope(use_cuda=1, use_acl=1))
            # Same residency issue as CUDA, but the ACL path is verified on the
            # NPU gate rather than here; leave it identity until it can be run.
            return lambda tensor: tensor
        else:
            policy_stack.enter_context(jt.runtime.scope(use_cuda=0))
        return lambda tensor: tensor
    if device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is unavailable in this PyTorch build")
        return lambda tensor: tensor.cuda()
    if device == "npu":
        try:
            importlib.import_module("torch_npu")
        except ImportError as error:
            raise SystemExit("torch_npu is unavailable: {}".format(error))
        npu = getattr(torch, "npu", None)
        if npu is None or not npu.is_available():
            raise SystemExit("NPU is unavailable in this PyTorch build")
        return lambda tensor: tensor.to("npu")
    return lambda tensor: tensor


def _device_in_use(torch, runtime, device):
    """Where the work actually ran, read back from the runtime itself.

    Reported alongside the timings so the caller can assert it instead of
    trusting that requesting a device was enough.
    """
    if runtime == "jittor":
        import jittor as jt

        if (
            device == "npu"
            and _test_capability.check_accelerator('acl', backend=jt).enabled
            and jt.introspection.policy.runtime.use_cuda
        ):
            return "npu"
        return "cuda" if jt.introspection.policy.runtime.use_cuda else "cpu"
    if device == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "npu":
        npu = getattr(torch, "npu", None)
        return "npu" if npu is not None and npu.is_available() else "cpu"
    return "cpu"


def _backend_report(runtime):
    if runtime == "jittor":
        import jittor as jt

        return {
            "has_acl": bool(_test_capability.check_accelerator('acl', backend=jt).enabled),
            "use_acl": bool(_test_capability.check_accelerator("acl", backend=jt).enabled
                            and jt.introspection.policy.runtime.use_cuda),
            "use_cuda": bool(jt.introspection.policy.runtime.use_cuda),
        }
    return {}


def _configure_cpu_threads(torch, runtime, device):
    """Make the independent CPU oracle honor the declared OpenMP budget.

    Some PyTorch builds initialize their pool to physical cores despite a
    different OMP_NUM_THREADS environment value. Configure the real pool before
    recording conditions; the Jittor process keeps its native OpenMP setup.
    """
    if runtime != "torch" or device != "cpu":
        return
    requested = os.environ.get("OMP_NUM_THREADS", "").strip()
    if not requested:
        return
    try:
        count = int(requested)
    except ValueError as error:
        raise ValueError("OMP_NUM_THREADS must be a positive integer") from error
    if count <= 0:
        raise ValueError("OMP_NUM_THREADS must be a positive integer")
    torch.set_num_threads(count)


def _runtime_conditions(torch, tf32):
    affinity = []
    if hasattr(os, "sched_getaffinity"):
        affinity = sorted(os.sched_getaffinity(0))
    get_threads = getattr(torch, "get_num_threads", None)
    return {
        "affinity": affinity,
        "runtime_threads": int(get_threads()) if callable(get_threads) else None,
        "thread_env": {
            name: os.environ.get(name, "")
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                         "OPENBLAS_NUM_THREADS")
        },
        "precision": tf32,
    }


def _configure_tf32(torch, device):
    enabled = os.environ.get("JITTOR_ECOSYSTEM_TF32", "1").strip().lower()
    enabled = enabled not in ("", "0", "false", "no", "off")
    benchmark = os.environ.get("JITTOR_ECOSYSTEM_CUDNN_BENCHMARK", "0").strip().lower()
    benchmark = benchmark not in ("", "0", "false", "no", "off")
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = enabled
        torch.backends.cudnn.allow_tf32 = enabled
        torch.backends.cudnn.benchmark = benchmark
        set_precision = getattr(torch, "set_float32_matmul_precision", None)
        if callable(set_precision):
            set_precision("high" if enabled else "highest")
    # Each runtime reports its own switches back, which is the comparison that
    # means something: on both sides these read the policy the runtime's own
    # ops execute with -- for Jittor the frontend-owned tier that every op it
    # builds captures at construction, not a value parked beside an unaffected
    # library call. The tier string is reported alongside the booleans because
    # "tf32 on" is three states in torch, and two runtimes can agree on the
    # boolean while one accumulates in bfloat16.
    get_precision = getattr(torch, "get_float32_matmul_precision", None)
    return {
        "matmul": bool(torch.backends.cuda.matmul.allow_tf32) if device == "cuda" else False,
        "cudnn": bool(torch.backends.cudnn.allow_tf32) if device == "cuda" else False,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark) if device == "cuda" else False,
        "matmul_precision": get_precision() if callable(get_precision) else "",
    }


def _synchronize(torch, runtime, device):
    # Jittor is lazy on every device: without a sync the timed step only pays
    # for the values it fetches, and the pending backward graph is discarded by
    # the next ``zero_grad``. PyTorch only needs the CUDA queue drained.
    if runtime == "jittor":
        import jittor as jt

        jt.sync_all(device != "cpu")
    elif device == "cuda":
        torch.cuda.synchronize()
    elif device == "npu":
        torch.npu.synchronize()


def _make_inputs(torch, spec, seed, to_device):
    generator = np.random.RandomState(seed)
    tensors = {}
    for name, (dtype, shape, high) in spec.items():
        if dtype == "int64":
            array = generator.randint(0, high, size=shape).astype("int64")
            tensors[name] = to_device(torch.from_numpy(array))
        else:
            array = generator.randn(*shape).astype("float32")
            tensor = to_device(torch.from_numpy(array))
            tensor.requires_grad_(True)
            tensors[name] = tensor
    return tensors


def _primary_output(result):
    """Reduce a library-specific output object to one differentiable tensor."""
    for attribute in ("logits", "sample", "last_hidden_state", "prediction_logits"):
        value = getattr(result, attribute, None)
        if value is not None:
            return value
    if isinstance(result, (tuple, list)):
        return result[0]
    if isinstance(result, dict):
        return next(iter(result.values()))
    return result


def _numpy_snapshot(value, *, preserve_dtype=False):
    return np.array(value.detach().cpu().numpy(),
                    dtype=None if preserve_dtype else "float32", copy=True)


def _tensor_manifest(value):
    """Portable public metadata; never infer sparse layout from tensor shape."""
    layout = str(value.layout).replace("torch.", "", 1)
    if layout not in ("strided", "sparse_coo"):
        raise NotImplementedError("ecosystem state transfer does not support " + layout)
    result = {
        "layout": layout,
        "shape": list(value.shape),
        "dtype": str(value.dtype).replace("torch.", "", 1),
    }
    if layout == "sparse_coo":
        result.update(sparse_dim=value.sparse_dim(), dense_dim=value.dense_dim())
        if result["dense_dim"] or not value.is_coalesced():
            raise NotImplementedError("ecosystem state transfer requires full, coalesced COO")
    return result


def _snapshot_state(entries):
    arrays, manifest = {}, {}
    for name, value in entries:
        manifest[name] = _tensor_manifest(value)
        value = value.detach()
        if manifest[name]["layout"] == "sparse_coo":
            value = value.to_dense()
        arrays[name] = np.array(value.cpu().numpy(), copy=True)
    return arrays, manifest


def _restore_state(torch, entries, arrays, manifest, to_device):
    available = dict(entries)
    if set(arrays) != set(available):
        raise ValueError("saved state keys differ: missing=%s unexpected=%s" % (
            sorted(set(available) - set(arrays)), sorted(set(arrays) - set(available))))
    if manifest is not None and set(manifest) != set(available):
        raise ValueError("saved state manifest keys differ from tensor keys")
    # Validate every tensor before mutating any parameters or buffers. Legacy
    # dense NPZ consumers need no sidecar, but COO must declare its layout.
    for name, value in available.items():
        actual = _tensor_manifest(value)
        if manifest is None:
            if actual["layout"] != "strided":
                raise ValueError("sparse state requires a layout manifest: " + name)
        elif actual != manifest[name]:
            raise ValueError("saved state metadata differs for %s: %r != %r" % (
                name, manifest[name], actual))
        array = arrays[name]
        if list(array.shape) != actual["shape"] or str(array.dtype) != actual["dtype"]:
            raise ValueError("saved state array shape/dtype differs for " + name)
    with torch.no_grad():
        for name, value in available.items():
            source = to_device(torch.from_numpy(arrays[name]))
            if _tensor_manifest(value)["layout"] == "sparse_coo":
                source = source.to_sparse()
            value.copy_(source)


def _state_fingerprints(arrays):
    return {
        name: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        for name, value in arrays.items()
    }


def _collect_arrays(model, inputs, output, *, required_parameters=None):
    """Strict cases fail on missing/nonfinite gradients before parity scoring."""
    strict = required_parameters is not None
    required = set(required_parameters or ())
    arrays = {"__output__": _numpy_snapshot(output, preserve_dtype=strict)}
    required_inputs = {name for name, tensor in inputs.items() if tensor.requires_grad}
    for prefix, entries, expected in (
        ("grad::", model.named_parameters(), required),
        ("ingrad::", inputs.items(), required_inputs),
    ):
        for name, tensor in entries:
            grad = getattr(tensor, "grad", None)
            if grad is None:
                if strict and name in expected:
                    raise AssertionError("missing required gradient: " + prefix + name)
                continue
            array = _numpy_snapshot(grad, preserve_dtype=strict)
            if strict and tuple(array.shape) != tuple(tensor.shape):
                raise AssertionError("gradient shape differs: " + prefix + name)
            arrays[prefix + name] = array
    if strict:
        expected = {"__output__"}
        expected.update("grad::" + name for name in required)
        expected.update("ingrad::" + name for name in required_inputs)
        if set(arrays) != expected:
            raise AssertionError("gradient tensor set differs from the trainable contract")
        for name, array in arrays.items():
            if not np.isfinite(array).all():
                raise AssertionError("nonfinite output/gradient: " + name)
    return arrays


def main():
    with ExitStack() as policy_stack:
        return _run(policy_stack)


def _run(policy_stack):
    parser = argparse.ArgumentParser()
    parser.add_argument("case")
    parser.add_argument("output")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runtime", choices=("torch", "jittor"), default="torch")
    parser.add_argument("--device", choices=("cpu", "cuda", "npu"), default="cpu")
    options = parser.parse_args()

    torch = _import_torch(options.runtime)
    to_device = _select_device(torch, options.runtime, options.device, policy_stack=policy_stack)
    _configure_cpu_threads(torch, options.runtime, options.device)
    tf32 = _configure_tf32(torch, options.device)
    runtime_conditions = _runtime_conditions(torch, tf32)

    fallback_scope = nullcontext()
    fallback_before = None
    if options.runtime == "jittor":
        import jittor as jt
        from jittor._runtime.fallback import forbid_backend_fallbacks

        policy_stack.enter_context(jt.runtime.scope(backend_fallback="error"))
        fallback_before = jt.core.backend_fallback_count()
        fallback_scope = forbid_backend_fallbacks()

    with fallback_scope:
        torch.manual_seed(options.seed)
        builder, requirements = _ecosystem_cases.CASES[options.case]
        model, input_spec = builder(torch)
        dependencies = _dependency_report(requirements)
        strict_gradients = options.case in _ecosystem_cases.STRICT_GRADIENT_CASES
        trainable_parameters = {
            name for name, value in model.named_parameters() if value.requires_grad
        }
        model.eval()
        if options.runtime == "torch" and options.device != "cpu":
            model.to(options.device)

        # ``state_dict`` is not always complete: ms-swift's tuner deliberately
        # reports only its adapter, so transferring it would leave the two runtimes
        # with independently initialized backbones and a meaningless comparison.
        # Enumerating parameters and buffers is complete by construction.
        def transferable():
            entries = list(model.named_parameters())
            named_buffers = getattr(model, "named_buffers", None)
            if callable(named_buffers):
                entries += list(named_buffers())
            return entries

        if options.weights:
            manifest_path = Path(options.weights + ".manifest.json")
            manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                        if manifest_path.exists() else None)
            if strict_gradients and manifest is None:
                raise ValueError("this case requires a complete state layout manifest")
            with np.load(options.weights, allow_pickle=False) as loaded:
                _restore_state(torch, transferable(), loaded, manifest, to_device)
        else:
            weights_path = os.path.splitext(options.output)[0] + ".weights.npz"
            saved, manifest = _snapshot_state(transferable())
            np.savez(weights_path, **saved)
            Path(weights_path + ".manifest.json").write_text(
                json.dumps(manifest, sort_keys=True), encoding="utf-8")
        state_manifest, state_fingerprints = None, None
        if strict_gradients:
            captured, state_manifest = _snapshot_state(transferable())
            state_fingerprints = _state_fingerprints(captured)

        # ``eval()`` in Jittor also stops gradients on every parameter; PyTorch's
        # does not.  Re-enable them so both runtimes differentiate the same graph.
        for name, parameter in model.named_parameters():
            if strict_gradients and name not in trainable_parameters:
                continue
            start_grad = getattr(parameter, "start_grad", None)
            if callable(start_grad):
                start_grad()
            else:
                parameter.requires_grad_(True)

        inputs = _make_inputs(torch, input_spec, options.seed + 1, to_device)
        output = _primary_output(model(**inputs))

        weights = np.random.RandomState(options.seed + 2)
        loss_weights = weights.randn(*tuple(output.shape)).astype("float32")
        loss = (output * to_device(torch.from_numpy(loss_weights))).sum()
        loss.backward()

        _synchronize(torch, options.runtime, options.device)
        arrays = _collect_arrays(
            model, inputs, output,
            required_parameters=trainable_parameters if strict_gradients else None,
        )
        if strict_gradients and not np.isfinite(_numpy_snapshot(loss)).all():
            raise AssertionError("nonfinite diagnostic loss")

        # Timing runs after correctness capture. Inputs and loss weights are already
        # resident on the requested device, so the number excludes allocation/H2D.
        timing_slots = [
            _make_inputs(torch, input_spec, options.seed + 10 + index, to_device)
            for index in range(4)
        ]
        timing_loss_weights = to_device(torch.from_numpy(loss_weights))

        def one_step(step_inputs):
            step_output = _primary_output(model(**step_inputs))
            step_loss = (step_output * timing_loss_weights).sum()
            model.zero_grad(set_to_none=False)
            step_loss.backward()
            gradients = []
            for parameter in model.parameters():
                grad = getattr(parameter, "grad", None)
                if grad is not None:
                    gradients.append(grad)
            for tensor in step_inputs.values():
                grad = getattr(tensor, "grad", None)
                if grad is not None:
                    gradients.append(grad)
            if options.runtime == "jittor":
                import jittor as jt

                # Explicit targets force every lazy gradient without introducing
                # hundreds of per-tensor D2H copies into the training measurement.
                # Submit exactly the observed training graph and wait for its
                # device work here. A following sync_all() would traverse every
                # live Var a second time even though these targets already cover
                # the complete forward/backward step.
                jt.sync(
                    [step_loss] + gradients,
                    device_sync=options.device != "cpu",
                )
            # Only the loss is handed back. These are per-step tensors, and a
            # caller that keeps them alive holds one step's whole gradient set
            # per step wherever the runtime returns a fresh ``.grad`` object
            # instead of accumulating into the existing one. `gradients` exists
            # to force the lazy graph and dies with this call; the parameters
            # keep their own ``.grad`` regardless.
            return step_loss

        resident_values = [timing_loss_weights]
        for slot in timing_slots:
            resident_values.extend(slot.values())
        if options.runtime == "jittor":
            import jittor as jt

            jt.sync(resident_values)
        warm_values = [one_step(slot) for slot in timing_slots]
        _synchronize(torch, options.runtime, options.device)
        durations = []
        for index in range(max(1, options.repeats)):
            started = time.perf_counter()
            values = one_step(timing_slots[index % len(timing_slots)])
            if options.runtime != "jittor":
                _synchronize(torch, options.runtime, options.device)
            durations.append(time.perf_counter() - started)
        del warm_values, values

    fallback_count = (jt.core.backend_fallback_count() - fallback_before
                      if fallback_before is not None else 0)

    np.savez(options.output, **arrays)
    print(
        "ECOSYSTEM_RESULT "
        + json.dumps(
            {
                "case": options.case,
                "tensors": len(arrays),
                "state_manifest": state_manifest,
                "state_fingerprints": state_fingerprints,
                "required_parameter_gradients": (
                    sorted(trainable_parameters) if strict_gradients else None),
                "required_input_gradients": sorted(
                    name for name, value in inputs.items() if value.requires_grad
                ) if strict_gradients else None,
                "timing_scope": (
                    "synthetic correctness fixture" if options.case in
                    _ecosystem_cases.REPORT_ONLY_TIMING_CASES else "case configuration"),
                "seconds": min(durations),
                "loss": float(loss.detach().cpu().numpy().reshape(-1)[0]),
                "device": _device_in_use(torch, options.runtime, options.device),
                "backend": _backend_report(options.runtime),
                "fallback_count": fallback_count,
                "fallback_policy": "error" if options.runtime == "jittor" else None,
                "package_site": os.environ.get("JITTOR_ECOSYSTEM_PACKAGE_SITE", ""),
                "dependencies": dependencies,
                "tf32": tf32,
                "runtime_conditions": runtime_conditions,
            }
        )
    )


if __name__ == "__main__":
    main()
