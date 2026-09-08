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
        elif device == "npu":
            if not _test_capability.check_accelerator("acl", backend=jt).enabled:
                raise SystemExit("ACL is unavailable in this Jittor build")
            policy_stack.enter_context(jt.runtime.scope(use_cuda=1, use_acl=1))
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
    return {
        "matmul": bool(torch.backends.cuda.matmul.allow_tf32) if device == "cuda" else False,
        "cudnn": bool(torch.backends.cudnn.allow_tf32) if device == "cuda" else False,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark) if device == "cuda" else False,
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


def _numpy_snapshot(value):
    return np.array(value.detach().cpu().numpy(), dtype="float32", copy=True)


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
            loaded = np.load(options.weights)
            available = dict(transferable())
            missing = sorted(key for key in loaded.files if key not in available)
            if missing:
                raise SystemExit("no counterpart for saved weights: %s" % missing[:5])
            unset = sorted(key for key in available if key not in loaded.files)
            if unset:
                raise SystemExit("no saved weight for: %s" % unset[:5])
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

        # ``eval()`` in Jittor also stops gradients on every parameter; PyTorch's
        # does not.  Re-enable them so both runtimes differentiate the same graph.
        for parameter in model.parameters():
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
        arrays = {"__output__": _numpy_snapshot(output)}
        for name, parameter in model.named_parameters():
            grad = getattr(parameter, "grad", None)
            if grad is None:
                continue
            arrays["grad::" + name] = _numpy_snapshot(grad)
        for name, tensor in inputs.items():
            grad = getattr(tensor, "grad", None)
            if grad is not None:
                arrays["ingrad::" + name] = _numpy_snapshot(grad)

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
            return step_loss, gradients

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
