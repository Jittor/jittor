"""Run deterministic multi-step SGD for an admitted ecosystem case in one runtime.

Invoke twice in separate interpreters; pass the reference's complete initial
weights to the candidate. Models are the shared ecosystem builders, and all
outputs, requested gradients and updated parameters are saved for comparison.
"""
import argparse
from contextlib import ExitStack, nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import subprocess


def options(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--case", required=True)
    parser.add_argument("--runtime", choices=("torch", "jittor"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "npu"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.steps < 2:
        parser.error("--steps must be at least 2 for a training trajectory")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    args.root = args.root.resolve()
    args.output = args.output.resolve()
    if args.output == args.root or args.root in args.output.parents:
        parser.error("--output must be outside the checkout")
    runner = args.root / "compat/tests/torch/_ecosystem_runner.py"
    if not runner.is_file():
        parser.error("--root must contain the shared ecosystem runner")
    return args


def main(argv=None):
    args = options(argv)
    if args.device == "cpu":
        # The CPU ladder must stay CPU even on a CUDA-equipped host.
        os.environ.update(CUDA_VISIBLE_DEVICES="", nvcc_path="", JT_BUILD_NVCC_PATH="", use_cuda="0", JT_USE_CUDA="0")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    sys.path.insert(0, str(args.root / "compat/tests/torch"))
    import _ecosystem_cases as cases
    if args.case not in cases.CASES:
        raise SystemExit("Unknown registered case: " + args.case)
    import numpy as np
    import _ecosystem_runner as runner
    args.output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}

    def snapshot(value):
        return np.array(value.detach().cpu().numpy(), copy=True)
    with ExitStack() as stack:
        torch = runner._import_torch(args.runtime)
        to_device = runner._select_device(torch, args.runtime, args.device, policy_stack=stack)
        tf32 = runner._configure_tf32(torch, args.device)
        conditions = runner._runtime_conditions(torch, tf32)
        before = None
        scope = nullcontext()
        if args.runtime == "jittor":
            import jittor as jt
            from jittor._runtime.fallback import forbid_backend_fallbacks
            stack.enter_context(jt.runtime.scope(backend_fallback="error"))
            before = jt.core.backend_fallback_count()
            scope = forbid_backend_fallbacks()
        with scope:
            torch.manual_seed(args.seed)
            builder, requirements = cases.CASES[args.case]
            model, input_spec = builder(torch)
            original_trainable = {name: bool(parameter.requires_grad)
                                  for name, parameter in model.named_parameters()}
            if args.device != "cpu":
                model.to(args.device)
            transferable = list(model.named_parameters()) + list(model.named_buffers())
            if args.weights:
                with np.load(args.weights) as initial:
                    available = dict(transferable)
                    if set(initial.files) != set(available):
                        raise RuntimeError("Complete parameter/buffer names differ between runtimes")
                    with torch.no_grad():
                        for name, parameter in available.items():
                            parameter.copy_(to_device(torch.from_numpy(initial[name])))
                weights_path = args.weights
            else:
                weights_path = args.output.with_name(args.output.stem + ".weights.npz")
                np.savez(weights_path, **{name: snapshot(value)
                                         for name, value in transferable})
            model.train()
            # eval()/train() has broader gradient side effects in Jittor. Honor
            # the upstream tuner's real frozen/trainable split on both sides.
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(original_trainable[name])
            names = sorted(name for name, value in original_trainable.items() if value)
            if not names:
                raise RuntimeError("Training case has no trainable parameters")
            initial_parameters = {name: snapshot(value)
                                  for name, value in model.named_parameters()}
            expected_inputs = sorted(name for name, value in input_spec.items()
                                     if value[0] != "int64")
            loss_history = []
            for step in range(args.steps):
                model.zero_grad(set_to_none=False)
                inputs = runner._make_inputs(torch, input_spec, args.seed + step + 1, to_device)
                output = runner._primary_output(model(**inputs))
                loss_weights = np.random.RandomState(args.seed + 100).randn(*tuple(output.shape)).astype("float32")
                loss = (output * to_device(torch.from_numpy(loss_weights))).mean()
                loss.backward()
                prefix = "step/%d/" % step
                arrays[prefix + "output"] = snapshot(output)
                arrays[prefix + "loss"] = snapshot(loss)
                loss_history.append(float(arrays[prefix + "loss"]))
                parameters = dict(model.named_parameters())
                for name in names:
                    gradient = parameters[name].grad
                    if gradient is None:
                        raise RuntimeError("Missing requested parameter gradient: " + name)
                    arrays[prefix + "grad::" + name] = snapshot(gradient)
                for name in expected_inputs:
                    gradient = inputs[name].grad
                    if gradient is None:
                        raise RuntimeError("Missing requested input gradient: " + name)
                    arrays[prefix + "ingrad::" + name] = snapshot(gradient)
                # Materialize every old gradient before updating lazy Vars.
                with torch.no_grad():
                    for name in names:
                        value = parameters[name]
                        gradient = to_device(torch.from_numpy(arrays[prefix + "grad::" + name]))
                        value.copy_(value - args.learning_rate * gradient)
                for name, value in parameters.items():
                    parameter_snapshot = snapshot(value)
                    arrays[prefix + "parameter::" + name] = parameter_snapshot
                    if not original_trainable[name] and not np.array_equal(initial_parameters[name], parameter_snapshot):
                        raise RuntimeError("Frozen upstream parameter changed: " + name)
            if not any(not np.array_equal(initial_parameters[name], arrays[prefix + "parameter::" + name])
                       for name in names):
                raise RuntimeError("No trainable parameter changed across the trajectory")
            if any(not np.isfinite(value).all() for value in arrays.values()):
                raise RuntimeError("Non-finite training trajectory")
            runner._synchronize(torch, args.runtime, args.device)
            observed = runner._device_in_use(torch, args.runtime, args.device)
            if observed != args.device:
                raise RuntimeError("Requested and observed training devices differ")
            result = {"case": args.case, "runtime": args.runtime, "device": observed,
                      "jittor_commit": subprocess.check_output(["git", "-C", str(args.root), "rev-parse", "HEAD"], text=True).strip(),
                      "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE") == "1",
                      "steps": args.steps, "learning_rate": args.learning_rate,
                      "mode": "train", "seed": args.seed, "loss_history": loss_history,
                      "trainable_parameters": names, "frozen_parameters": sorted(set(original_trainable) - set(names)),
                      "input_gradient_names": expected_inputs, "tensors": len(arrays),
                      "weights": str(weights_path), "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
                      "backend": runner._backend_report(args.runtime), "tf32": tf32,
                      "runtime_conditions": conditions, "dependencies": runner._dependency_report(requirements),
                      "fallback_count": jt.core.backend_fallback_count() - before if before is not None else 0,
                      "fallback_policy": "error" if before is not None else None}
    np.savez(args.output, **arrays)
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print("TRAINING_RESULT " + json.dumps(result))


if __name__ == "__main__":
    main()
