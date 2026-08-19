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
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import _ecosystem_cases  # noqa: E402


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
        return torch

    os.environ.pop("JITTOR_TORCH_SHIM", None)
    import torch

    if not hasattr(torch, "_C") or hasattr(torch, "_torch_compat_install_context"):
        raise SystemExit("torch did not resolve to an independent PyTorch")
    return torch


def _select_device(torch, runtime, device):
    """Put both runtimes on the requested device using each one's own idiom.

    Jittor has no per-tensor device; a single global flag moves the whole graph,
    so the two runtimes need different code here even though everything else in
    this file is spelled once.
    """
    if device != "cuda":
        return lambda tensor: tensor
    if runtime == "jittor":
        import jittor as jt

        if not jt.has_cuda:
            raise SystemExit("CUDA is unavailable in this Jittor build")
        jt.flags.use_cuda = 1
        return lambda tensor: tensor
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this PyTorch build")
    return lambda tensor: tensor.cuda()


def _synchronize(torch, runtime, device):
    # Jittor is lazy on every device: without a sync the timed step only pays
    # for the values it fetches, and the pending backward graph is discarded by
    # the next ``zero_grad``. PyTorch only needs the CUDA queue drained.
    if runtime == "jittor":
        import jittor as jt

        jt.sync_all(device == "cuda")
    elif device == "cuda":
        torch.cuda.synchronize()


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case")
    parser.add_argument("output")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runtime", choices=("torch", "jittor"), default="torch")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    options = parser.parse_args()

    torch = _import_torch(options.runtime)
    to_device = _select_device(torch, options.runtime, options.device)

    torch.manual_seed(options.seed)
    builder, _requirements = _ecosystem_cases.CASES[options.case]
    model, input_spec = builder(torch)
    model.eval()
    if options.device == "cuda" and options.runtime == "torch":
        model.cuda()

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
    arrays = {"__output__": np.asarray(output.detach().cpu().numpy(), dtype="float32")}
    for name, parameter in model.named_parameters():
        grad = getattr(parameter, "grad", None)
        if grad is None:
            continue
        arrays["grad::" + name] = np.asarray(grad.detach().cpu().numpy(), dtype="float32")
    for name, tensor in inputs.items():
        grad = getattr(tensor, "grad", None)
        if grad is not None:
            arrays["ingrad::" + name] = np.asarray(
                grad.detach().cpu().numpy(), dtype="float32"
            )

    # Timing runs after the correctness capture so a lazily built graph is
    # already compiled and the number reflects steady-state execution.
    def one_step():
        step_inputs = _make_inputs(torch, input_spec, options.seed + 1, to_device)
        step_output = _primary_output(model(**step_inputs))
        step_loss = (step_output * to_device(torch.from_numpy(loss_weights))).sum()
        model.zero_grad(set_to_none=False)
        step_loss.backward()
        return float(step_loss.detach().cpu().numpy().reshape(-1)[0])

    one_step()
    _synchronize(torch, options.runtime, options.device)
    durations = []
    for _ in range(max(1, options.repeats)):
        started = time.perf_counter()
        one_step()
        _synchronize(torch, options.runtime, options.device)
        durations.append(time.perf_counter() - started)

    np.savez(options.output, **arrays)
    print(
        "ECOSYSTEM_RESULT "
        + json.dumps(
            {
                "case": options.case,
                "tensors": len(arrays),
                "seconds": min(durations),
                "loss": float(loss.detach().cpu().numpy().reshape(-1)[0]),
            }
        )
    )


if __name__ == "__main__":
    main()
