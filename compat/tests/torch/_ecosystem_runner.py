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
            # from_numpy has explicit host placement even with ACL selected.
            # Move inputs and loaded state just as the native NPU runner does.
            return lambda tensor: tensor.to("npu")
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


def _validate_transfer_state(available, loaded):
    """Reject incomplete or coercible state before changing any model tensor."""
    saved_keys = set(loaded.files)
    if set(available) != saved_keys:
        raise AssertionError("transfer state keys differ: missing={}, extra={}".format(
            sorted(set(available) - saved_keys), sorted(saved_keys - set(available))))
    for name, value in available.items():
        source = loaded[name]
        if tuple(value.shape) != tuple(source.shape):
            raise AssertionError("transfer shape mismatch: {}".format(name))
        target_dtype = str(value.dtype).split(".")[-1]
        source_dtype = str(source.dtype).split(".")[-1]
        if target_dtype != source_dtype:
            raise AssertionError("transfer dtype mismatch: {} ({} != {})".format(
                name, source_dtype, target_dtype))


def _parameter_grad_policy(model):
    """Capture the tuner's declared trainable set before mode/state transitions."""
    return {name: bool(value.requires_grad)
            for name, value in model.named_parameters()}


def _assert_parameter_grad_policy(model, policy):
    """Reject trainability drift; evidence collection must never repair it."""
    parameters = dict(model.named_parameters())
    if set(parameters) != set(policy):
        raise AssertionError("parameter set changed after mode/state transition")
    changed = sorted(name for name, value in parameters.items()
                     if bool(value.requires_grad) != policy[name])
    if changed:
        raise AssertionError("parameter requires_grad changed: {}".format(changed))


def _required_gradients(model, inputs, policy):
    """Reject missing trainable and differentiable-input gradients."""
    parameters = dict(model.named_parameters())
    if set(parameters) != set(policy):
        raise AssertionError("parameter set changed during forward")
    gradients = {}
    for name, value in parameters.items():
        grad = getattr(value, "grad", None)
        if policy[name]:
            if grad is None:
                raise AssertionError("missing parameter gradient: {}".format(name))
            gradients["grad::" + name] = grad
        elif grad is not None:
            raise AssertionError("frozen parameter has gradient: {}".format(name))
    not_applicable = []
    for name, value in inputs.items():
        grad = getattr(value, "grad", None)
        if bool(value.requires_grad):
            if grad is None:
                raise AssertionError("missing input gradient: {}".format(name))
            gradients["ingrad::" + name] = grad
        else:
            if grad is not None:
                raise AssertionError("non-differentiable input has gradient: {}".format(name))
            not_applicable.append(name)
    return gradients, sorted(not_applicable)


def _output_structure(result, primary):
    report = {"type": type(result).__name__, "primary_shape": list(primary.shape)}
    if isinstance(result, dict):
        report["keys"] = list(result.keys())
    elif isinstance(result, (tuple, list)):
        report["length"] = len(result)
    return report


def _npu_tensor_evidence(value, runtime, label, expected_device=0):
    """Observe only; callers synchronize before this and before any D2H copy."""
    shape = tuple(int(size) for size in value.shape)
    record = {"shape": list(shape), "dtype": str(value.dtype).split(".")[-1]}
    empty = any(size == 0 for size in shape)
    if runtime == "jittor":
        # FollowRuntime (-1) is valid under the asserted ACL runtime; physical
        # residency and device identity below must still independently agree.
        placement = int(value.placement_backend)
        device_id = int(value.device_id)
        location = value.location()
        record.update(placement_backend=placement, device_id=device_id, location=location)
        if placement not in (-1, 2) or device_id != expected_device:
            raise AssertionError("{}: expected ACL-compatible placement on device {}".format(label, expected_device))
        if location != "device":
            if empty and location == "none":
                record["residency_exception"] = "zero-sized tensor has no allocation"
            else:
                raise AssertionError("{}: non-device residency {}".format(label, location))
    else:
        device_type, device_id = value.device.type, value.device.index
        record.update(device_type=device_type, device_id=device_id)
        if device_type != "npu" or device_id != expected_device:
            raise AssertionError("{}: expected native NPU tensor on device {}".format(label, expected_device))
    return record


def _npu_evidence(torch, runtime, model, inputs, output, gradients):
    """Require independent runtime identity and every observed tensor on NPU."""
    if runtime == "jittor":
        import jittor as jt
        if not hasattr(torch, "_torch_compat_install_context"):
            raise AssertionError("candidate torch is not Jittor shim")
        registered = list(jt.core.registered_backends())
        config = jt.compiler.build_config
        count = jt.core.backend_device_count("acl")
        if "acl" not in registered or config.backend != "acl" or not config.has_acl or count != 1:
            raise AssertionError("candidate is not a single-device ACL build")
        backend = {"runtime": "jittor", "module": jt.__file__, "build_backend": config.backend,
                   "registered_backends": registered, "device_count": count}
    else:
        torch_npu = importlib.import_module("torch_npu")
        if hasattr(torch, "_torch_compat_install_context") or not hasattr(torch, "_C"):
            raise AssertionError("oracle torch is not independent native PyTorch")
        count = torch.npu.device_count()
        if not torch.npu.is_available() or count != 1:
            raise AssertionError("oracle requires exactly one native NPU")
        backend = {"runtime": "torch_npu", "module": torch_npu.__file__,
                   "torch_module": torch.__file__, "device_count": count}
    entries = [("parameter::" + name, value) for name, value in model.named_parameters()]
    entries += [("buffer::" + name, value) for name, value in model.named_buffers()]
    entries += [("input::" + name, value) for name, value in inputs.items()]
    entries += [("primary_output", output)]
    entries += [(name, value) for name, value in gradients.items()]
    records = {}
    for name, value in entries:
        if name in records:
            raise AssertionError("duplicate NPU evidence key: " + name)
        records[name] = _npu_tensor_evidence(value, runtime, name)
    return {"backend_identity": backend, "tensors": records,
            "primary_dtype": records["primary_output"]["dtype"],
            "input_dtypes": {name: records["input::" + name]["dtype"] for name in inputs}}


def _numpy_snapshot(value):
    return np.array(value.detach().cpu().numpy(), dtype="float32", copy=True)


def _training_trajectory_keys(names, steps=3):
    keys = {"initial::" + name for name in names}
    for step in range(steps):
        prefix = "step::{}::".format(step)
        keys.update(prefix + "grad::" + name for name in names)
        keys.update(prefix + "param::" + name for name in names)
        keys.update(prefix + "delta::" + name for name in names)
        keys.add(prefix + "loss")
    return keys


def _run_adamw3(torch, model, inputs, policy, options, dependencies,
                tf32, runtime_conditions, fallback_before):
    """Three real updates; snapshots are correctness evidence, never timing."""
    if options.device != "npu":
        raise AssertionError("ms-swift AdamW3 is an Ascend-only protocol")
    if options.runtime == "jittor" and not options.weights:
        raise AssertionError("candidate AdamW3 requires oracle initial weights")
    names = sorted(name for name, enabled in policy.items() if enabled)
    if len(names) != 8 or any("lora_" not in name for name in names):
        raise AssertionError("expected exactly eight ms-swift LoRA parameters")
    model.train()
    _assert_parameter_grad_policy(model, policy)
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    frozen = {name: _numpy_snapshot(value) for name, value in parameters.items()
              if not policy[name]}
    initial_buffers = {name: _numpy_snapshot(value) for name, value in buffers.items()}
    arrays = {"initial::" + name: _numpy_snapshot(parameters[name]) for name in names}
    optimizer_config = dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=0.01, fused=False)
    optimizer = torch.optim.AdamW([parameters[name] for name in names], **optimizer_config)
    observations = []
    output_structure = None
    input_grad_na = None
    for step in range(3):
        if _parameter_grad_policy(model) != policy:
            raise AssertionError("trainable parameter policy changed during training")
        optimizer.zero_grad(set_to_none=True)
        result = model(**inputs, labels=inputs["input_ids"], use_cache=False)
        output = _primary_output(result)
        structure = _output_structure(result, output)
        if output_structure is not None and structure != output_structure:
            raise AssertionError("output structure changed during training")
        output_structure = structure
        loss = result.loss
        loss.backward()
        _synchronize(torch, options.runtime, options.device)
        gradients, input_grad_na = _required_gradients(model, inputs, policy)
        if set(gradients) != {"grad::" + name for name in names}:
            raise AssertionError("training gradient set differs from all eight LoRA parameters")
        backward_evidence = _npu_evidence(torch, options.runtime, model, inputs, output, gradients)
        loss_evidence = _npu_tensor_evidence(loss, options.runtime, "loss")
        prefix = "step::{}::".format(step)
        arrays[prefix + "loss"] = _numpy_snapshot(loss)
        arrays.update({prefix + name: _numpy_snapshot(grad)
                       for name, grad in gradients.items()})
        for key, value in arrays.items():
            if not np.isfinite(value).all():
                raise AssertionError("non-finite training snapshot: " + key)
        optimizer.step()
        # Force the updated values on lazy runtimes before taking snapshots.
        if options.runtime == "jittor":
            import jittor as jt
            jt.sync([parameters[name] for name in names], device_sync=True)
        _synchronize(torch, options.runtime, options.device)
        update_evidence = _npu_evidence(torch, options.runtime, model, inputs, output, gradients)
        arrays.update({prefix + "param::" + name: _numpy_snapshot(parameters[name])
                       for name in names})
        previous_prefix = "initial::" if step == 0 else "step::{}::param::".format(step - 1)
        for name in names:
            arrays[prefix + "delta::" + name] = (
                arrays[prefix + "param::" + name].astype("float64")
                - arrays[previous_prefix + name].astype("float64"))
        if _parameter_grad_policy(model) != policy:
            raise AssertionError("optimizer changed trainable parameter policy")
        for name in names:
            if not np.isfinite(arrays[prefix + "param::" + name]).all():
                raise AssertionError("non-finite updated parameter: " + name)
        for name, expected in frozen.items():
            if not np.array_equal(_numpy_snapshot(parameters[name]), expected):
                raise AssertionError("frozen parameter changed: " + name)
        if set(dict(model.named_buffers())) != set(initial_buffers):
            raise AssertionError("buffer set changed during training")
        for name, expected in initial_buffers.items():
            if not np.array_equal(_numpy_snapshot(dict(model.named_buffers())[name]), expected):
                raise AssertionError("buffer changed: " + name)
        count = None
        if options.runtime == "jittor":
            count = jt.core.backend_fallback_count() - fallback_before
        actual_device = _device_in_use(torch, options.runtime, options.device)
        if actual_device != "npu" or (options.runtime == "jittor" and count != 0):
            raise AssertionError("training left NPU or used backend fallback")
        observations.append({"step": step, "device": actual_device,
                             "backend": _backend_report(options.runtime),
                             "trainable_parameters": names,
                             "backward_npu_evidence": backward_evidence,
                             "update_npu_evidence": update_evidence,
                             "loss_dtype": loss_evidence["dtype"],
                             "fallback_count": count})
    if set(arrays) != _training_trajectory_keys(names):
        raise AssertionError("incomplete three-step training trajectory")
    for key, value in arrays.items():
        if not np.isfinite(value).all():
            raise AssertionError("non-finite training snapshot: " + key)
    # LoRA A can legitimately have zero gradient on the first update while B
    # starts at zero. Require a genuine update, not every parameter every step.
    if not any(not np.array_equal(arrays["initial::" + name],
                                  arrays["step::2::param::" + name]) for name in names):
        raise AssertionError("AdamW completed without updating any parameter")
    np.savez(options.output, **arrays)
    print("ECOSYSTEM_RESULT " + json.dumps({
        "case": options.case, "protocol": "adamw3", "steps": 3,
        "optimizer": optimizer_config, "step_observations": observations,
        "trajectory_dtypes": {key: str(value.dtype) for key, value in arrays.items()},
        "npu_evidence": observations[-1]["update_npu_evidence"],
        "trainable_parameters": names,
        "frozen_parameters": sorted(frozen), "input_grad_not_applicable": input_grad_na,
        "output_structure": output_structure, "tensors": len(arrays),
        "device": observations[-1]["device"], "backend": observations[-1]["backend"],
        "fallback_count": observations[-1]["fallback_count"],
        "fallback_policy": "error" if options.runtime == "jittor" else None,
        "package_site": os.environ.get("JITTOR_ECOSYSTEM_PACKAGE_SITE", ""),
        "dependencies": dependencies, "tf32": tf32, "runtime_conditions": runtime_conditions,
    }))


def _performance_statistics(durations, tokens=512):
    import math
    import statistics
    if len(durations) < 10 or any(not math.isfinite(x) or x <= 0 for x in durations):
        raise AssertionError("performance requires at least ten finite positive durations")
    ordered = sorted(durations)
    def quantile(q):
        index = (len(ordered) - 1) * q
        lower = int(index)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)
    median = statistics.median(ordered)
    return {"durations_seconds": list(durations), "samples": len(durations),
            "median_seconds": median, "min_seconds": ordered[0],
            "p10_seconds": quantile(0.1), "p90_seconds": quantile(0.9),
            "quantile_method": "linear", "input_tokens_per_second": tokens / median,
            "loss_tokens_per_second": (tokens - 1) / median}


def _performance_precision(torch, runtime):
    if runtime == "torch":
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
        torch.npu.matmul.cube_math_type = torch.npu.CubeMathType.KEEP_DTYPE
        if torch.npu.matmul.allow_hf32 or torch.npu.conv.allow_hf32:
            raise AssertionError("native HF32 disable did not take effect")
        cube = torch.npu.matmul.cube_math_type
        if cube != torch.npu.CubeMathType.KEEP_DTYPE:
            raise AssertionError("native cube math is not KEEP_DTYPE")
        return {"dtype": "float32", "allow_hf32": False,
                "cube_math_type": int(cube), "readback": "torch_npu native options"}
    import jittor as jt
    jt.acl_allow_hf32 = False
    if jt.acl_allow_hf32:
        raise AssertionError("candidate HF32 disable did not take effect")
    return {"dtype": "float32", "allow_hf32": False, "cube_math_type": 0,
            "readback": "jt.acl_allow_hf32; mapped_matmul passes cubeMathType=0"}


def _performance_memory(torch, runtime):
    if runtime == "torch":
        return {"source": "torch.npu allocator", "units": "bytes",
                "allocated": int(torch.npu.memory_allocated(0)),
                "reserved": int(torch.npu.memory_reserved(0)),
                "peak_allocated": int(torch.npu.max_memory_allocated(0)),
                "peak_reserved": int(torch.npu.max_memory_reserved(0)),
                "peak_kind": "allocator peak since post-warmup reset"}
    import jittor as jt
    return {"source": "jt.core.device_memory_used/reserved(0)", "units": "bytes",
            "allocated": int(jt.core.device_memory_used(0)),
            "reserved": int(jt.core.device_memory_reserved(0)),
            "peak_allocated": None, "peak_reserved": None,
            "peak_kind": "unavailable; synchronized boundary samples only"}


def _run_lora_performance(torch, model, inputs, policy, options, dependencies,
                          tf32, runtime_conditions, fallback_before):
    """Explicit realistic-size NPU workload; no host snapshots in timed steps."""
    if options.device != "npu" or options.repeats < 10:
        raise AssertionError("large LoRA performance requires NPU and repeats>=10")
    if options.runtime == "jittor" and not options.weights:
        raise AssertionError("candidate requires complete oracle initial state")
    precision = _performance_precision(torch, options.runtime)
    if model.config._attn_implementation != "eager":
        raise AssertionError("both performance runtimes must use eager attention")
    model.train()
    _assert_parameter_grad_policy(model, policy)
    parameters = dict(model.named_parameters())
    names = sorted(name for name, enabled in policy.items() if enabled)
    if len(names) != 88 or any("lora_" not in name for name in names):
        raise AssertionError("expected 88 LoRA matrices across 22 layers")
    total_parameters = sum(int(value.numel()) for value in parameters.values())
    trainable_parameters = sum(int(parameters[name].numel()) for name in names)
    if not 1000000000 <= total_parameters <= 1200000000 or trainable_parameters != 563200:
        raise AssertionError("unexpected model/trainable parameter count")
    optimizer_config = dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=0.01, fused=False)
    optimizer = torch.optim.AdamW([parameters[name] for name in names], **optimizer_config)

    def step():
        optimizer.zero_grad(set_to_none=True)
        result = model(**inputs, labels=inputs["input_ids"], use_cache=False)
        loss = result.loss
        loss.backward()
        gradients, _ = _required_gradients(model, inputs, policy)
        optimizer.step()
        # A detached loss is a new lazy node on Jittor. Materialize the same
        # scalar we retain before ending the synchronized measurement.
        detached_loss = loss.detach()
        if options.runtime == "jittor":
            import jittor as jt
            jt.sync([parameters[name] for name in names] + [detached_loss], device_sync=True)
        _synchronize(torch, options.runtime, "npu")
        return detached_loss, result, gradients

    initial = {name: _numpy_snapshot(parameters[name]) for name in names}
    losses, warmup_durations, durations, fallback_counts = [], [], [], []
    loss_evidence = []
    output_structure = None
    for _ in range(3):
        started = time.perf_counter()
        loss, result, gradients = step()
        warmup_durations.append(time.perf_counter() - started)
        losses.append(loss)
        output = _primary_output(result)
        observed_structure = _output_structure(result, output)
        if output_structure is not None and observed_structure != output_structure:
            raise AssertionError("performance output structure changed")
        output_structure = observed_structure
        loss_evidence.append(_npu_tensor_evidence(loss, options.runtime, "loss"))
        if loss_evidence[-1]["dtype"] != "float32" or loss_evidence[-1]["shape"] != []:
            raise AssertionError("performance loss must be scalar FP32 on NPU")
    warm_evidence = _npu_evidence(torch, options.runtime, model, inputs, output, gradients)
    _assert_parameter_grad_policy(model, policy)
    if options.runtime == "torch":
        torch.npu.reset_peak_memory_stats(0)
    memory_start = _performance_memory(torch, options.runtime)
    _synchronize(torch, options.runtime, "npu")
    for _ in range(options.repeats):
        started = time.perf_counter()
        loss, result, gradients = step()
        durations.append(time.perf_counter() - started)
        losses.append(loss)
        output = _primary_output(result)
        observed_structure = _output_structure(result, output)
        if output_structure is not None and observed_structure != output_structure:
            raise AssertionError("performance output structure changed")
        output_structure = observed_structure
        loss_evidence.append(_npu_tensor_evidence(loss, options.runtime, "loss"))
        if loss_evidence[-1]["dtype"] != "float32" or loss_evidence[-1]["shape"] != []:
            raise AssertionError("performance loss must be scalar FP32 on NPU")
        _assert_parameter_grad_policy(model, policy)
        count = None
        if options.runtime == "jittor":
            import jittor as jt
            count = jt.core.backend_fallback_count() - fallback_before
        if options.runtime == "jittor" and (type(count) is not int or count != 0):
            raise AssertionError("performance workload used backend fallback")
        fallback_counts.append(count)
    final_evidence = _npu_evidence(torch, options.runtime, model, inputs, output, gradients)
    memory_end = _performance_memory(torch, options.runtime)
    arrays = {"losses": np.asarray([float(_numpy_snapshot(loss).reshape(-1)[0])
                                   for loss in losses], dtype="float32")}
    arrays.update({"final::" + name: _numpy_snapshot(parameters[name]) for name in names})
    arrays.update({name: _numpy_snapshot(value) for name, value in gradients.items()})
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise AssertionError("non-finite performance artifact")
    if not any(not np.array_equal(initial[name], arrays["final::" + name]) for name in names):
        raise AssertionError("performance optimizer never updated parameters")
    if final_evidence["primary_dtype"] != "float32" or final_evidence["input_dtypes"] != {"input_ids": "int64"}:
        raise AssertionError("unexpected workload precision")
    statistics = _performance_statistics(durations)
    np.savez(options.output, **arrays)
    print("ECOSYSTEM_RESULT " + json.dumps({
        "case": options.case, "protocol": "lora_1b_train_performance_v1",
        "warmup_steps": 3, "timed_steps": options.repeats,
        "warmup_durations_seconds": warmup_durations, "statistics": statistics,
        "seconds": statistics["median_seconds"], "memory_start": memory_start,
        "memory_end": memory_end, "precision": precision, "optimizer": optimizer_config,
        "config": model.config.to_dict(), "attention": "eager",
        "tuner": {"owner": "swift.tuners.LoRAConfig", "r": 4, "lora_alpha": 8,
                  "lora_dropout": 0.0, "target_modules": ["q_proj", "v_proj"]},
        "parameter_count": total_parameters, "trainable_parameter_count": trainable_parameters,
        "trainable_parameters": names, "frozen_parameters": sorted(set(parameters) - set(names)),
        "input_grad_not_applicable": ["input_ids"], "output_structure": output_structure,
        "device": "npu", "backend": _backend_report(options.runtime),
        "loss_npu_evidence": loss_evidence,
        "artifact_dtypes": {key: str(value.dtype) for key, value in arrays.items()},
        "npu_evidence": final_evidence, "warmup_npu_evidence": warm_evidence,
        "fallback_policy": "error" if options.runtime == "jittor" else None,
        "fallback_count": fallback_counts[-1], "step_fallback_counts": fallback_counts,
        "native_fallback_observation": "separate full-process log guard; no validated universal counter",
        "dependencies": dependencies, "tf32": tf32, "runtime_conditions": runtime_conditions,
        "package_site": os.environ.get("JITTOR_ECOSYSTEM_PACKAGE_SITE", ""),
    }))


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
        grad_policy = _parameter_grad_policy(model)
        model.eval()
        _assert_parameter_grad_policy(model, grad_policy)
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
            loaded = np.load(options.weights)
            available = dict(transferable())
            _validate_transfer_state(available, loaded)
            for name, value in available.items():
                source = to_device(torch.from_numpy(loaded[name]))
                with_no_grad = getattr(torch, "no_grad", None)
                if with_no_grad is not None:
                    with with_no_grad():
                        value.copy_(source)
                else:
                    value.copy_(source)
        else:
            weights_path = os.path.splitext(options.output)[0] + ".weights.npz"
            np.savez(
                weights_path,
                **{
                    name: value.detach().cpu().numpy()
                    for name, value in transferable()
                },
            )

        _assert_parameter_grad_policy(model, grad_policy)

        inputs = _make_inputs(torch, input_spec, options.seed + 1, to_device)
        if options.case == "large_ms_swift_lora_llama_1b_train":
            return _run_lora_performance(torch, model, inputs, grad_policy, options,
                                         dependencies, tf32, runtime_conditions, fallback_before)
        if options.case == "ms_swift_lora_llama_adamw3":
            return _run_adamw3(torch, model, inputs, grad_policy, options,
                               dependencies, tf32, runtime_conditions, fallback_before)
        result = model(**inputs)
        output = _primary_output(result)
        output_structure = _output_structure(result, output)

        weights = np.random.RandomState(options.seed + 2)
        loss_weights = weights.randn(*tuple(output.shape)).astype("float32")
        loss = (output * to_device(torch.from_numpy(loss_weights))).sum()
        loss.backward()

        _synchronize(torch, options.runtime, options.device)
        gradients, input_grad_na = _required_gradients(model, inputs, grad_policy)
        npu_evidence = None
        if options.device == "npu":
            npu_evidence = _npu_evidence(
                torch, options.runtime, model, inputs, output, gradients)
        arrays = {"__output__": _numpy_snapshot(output)}
        arrays.update({name: _numpy_snapshot(grad)
                       for name, grad in gradients.items()})

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
                      if fallback_before is not None else None)

    np.savez(options.output, **arrays)
    print(
        "ECOSYSTEM_RESULT "
        + json.dumps(
            {
                "case": options.case,
                **({"npu_evidence": npu_evidence} if npu_evidence is not None else {}),
                "trainable_parameters": sorted(name for name, enabled in grad_policy.items() if enabled),
                "frozen_parameters": sorted(name for name, enabled in grad_policy.items() if not enabled),
                "input_grad_not_applicable": input_grad_na,
                "output_structure": output_structure,
                "tensors": len(arrays),
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
