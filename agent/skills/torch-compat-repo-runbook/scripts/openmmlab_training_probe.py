"""Compare real MMEngine BaseModel.train_step, OptimWrapper and momentum SGD.

The optimizer subclass observes gradients immediately before the real SGD step.
MMEngine owns preprocessing, forward(mode='loss'), loss parsing, backward,
clipping, step and zero_grad. This is deliberately separate from the ordinary
SGD probe, which does not exercise those downstream training owners.
"""

from contextlib import ExitStack, nullcontext
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from training_probe import options


def main(argv=None):
    args = options(argv)
    if args.device not in ("cpu", "cuda"):
        raise SystemExit("OpenMMLab probe admits only CPU and CUDA")
    if args.device == "cpu":
        os.environ.update(
            CUDA_VISIBLE_DEVICES="",
            nvcc_path="",
            JT_BUILD_NVCC_PATH="",
            use_cuda="0",
            JT_USE_CUDA="0",
        )
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sys.path.insert(0, str(args.root / "compat/tests/torch"))
    import numpy as np
    import _ecosystem_cases as cases
    import _ecosystem_runner as runner
    from _ecosystem_openmmlab_cases import PARAMETERS_BY_CASE

    if args.case not in PARAMETERS_BY_CASE:
        raise SystemExit("Not an OpenMMLab training-owner case: " + args.case)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}

    def snapshot(value):
        return np.array(value.detach().cpu().numpy(), copy=True)

    with ExitStack() as stack:
        torch = runner._import_torch(args.runtime)
        to_device = runner._select_device(torch, args.runtime, args.device, policy_stack=stack)
        tf32 = runner._configure_tf32(torch, args.device)
        conditions = runner._runtime_conditions(torch, tf32)
        from mmengine.model import BaseModel
        from mmengine.optim import OptimWrapper
        from mmengine.registry import OPTIM_WRAPPERS

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
            model.to(args.device)
            model.train()
            parameters = dict(model.named_parameters())
            names = sorted(parameters)
            if names != sorted(PARAMETERS_BY_CASE[args.case]):
                raise RuntimeError("Upstream model parameter names changed")
            if not all(parameter.requires_grad for parameter in parameters.values()):
                raise RuntimeError("Every declared model parameter must be trainable")
            values = dict(list(parameters.items()) + list(model.named_buffers()))
            if args.weights:
                with np.load(args.weights) as initial, torch.no_grad():
                    if set(initial.files) != set(values):
                        raise RuntimeError("Complete initial parameter/buffer names differ")
                    for name, value in values.items():
                        if tuple(value.shape) != initial[name].shape:
                            raise RuntimeError("Initial parameter/buffer shape differs: " + name)
                        value.copy_(to_device(torch.from_numpy(initial[name])))
                weights = args.weights
            else:
                weights = args.output.with_name(args.output.stem + ".weights.npz")
                np.savez(weights, **{name: snapshot(value) for name, value in values.items()})
            initial_parameters = {name: snapshot(value) for name, value in parameters.items()}

            class RecordingSGD(torch.optim.SGD):
                def __init__(self, params, **kwargs):
                    super().__init__(params, **kwargs)
                    self.updates = 0

                def step(self, *step_args, **step_kwargs):
                    prefix = "step/%d/" % self.updates
                    for name, parameter in parameters.items():
                        if parameter.grad is None:
                            raise RuntimeError("Missing requested parameter gradient: " + name)
                        arrays[prefix + "grad::" + name] = snapshot(parameter.grad)
                    result = super().step(*step_args, **step_kwargs)
                    self.updates += 1
                    return result

            optimizer = RecordingSGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
            wrapper = OPTIM_WRAPPERS.build(
                dict(
                    type="OptimWrapper",
                    optimizer=optimizer,
                    clip_grad=dict(max_norm=1.0),
                )
            )
            if (
                type(wrapper) is not OptimWrapper
                or model.train_step.__func__ is not BaseModel.train_step
            ):
                raise RuntimeError("Training must use the unmodified upstream owners")
            loss_history = []
            log_keys = ["loss", "loss_regression", "loss_regularization"]
            for step in range(args.steps):
                prefix = "step/%d/" % step
                inputs = runner._make_inputs(torch, input_spec, args.seed + step + 1, to_device)
                target_array = (
                    np.random.RandomState(args.seed + 101 + step)
                    .randn(input_spec["sample"][1][0], 2)
                    .astype("float32")
                )
                targets = to_device(torch.from_numpy(target_array))
                arrays[prefix + "input::sample"] = snapshot(inputs["sample"])
                arrays[prefix + "target"] = target_array
                log_vars = model.train_step(dict(**inputs, targets=targets), wrapper)
                if sorted(log_vars) != log_keys:
                    raise RuntimeError("BaseModel.parse_losses returned unexpected log keys")
                if optimizer.updates != step + 1 or model.loss_forward_calls != step + 1:
                    raise RuntimeError(
                        "Each train_step must forward, backward and update exactly once"
                    )
                arrays[prefix + "output"] = snapshot(model.last_prediction)
                for name, value in log_vars.items():
                    arrays[prefix + name] = snapshot(value)
                if not np.allclose(
                    arrays[prefix + "loss"],
                    arrays[prefix + "loss_regression"] + arrays[prefix + "loss_regularization"],
                    rtol=1e-6,
                    atol=1e-7,
                ):
                    raise RuntimeError("Parsed total loss differs from scalar/list components")
                loss_history.append(float(arrays[prefix + "loss"]))
                if inputs["sample"].grad is None:
                    raise RuntimeError("Missing requested input gradient: sample")
                arrays[prefix + "ingrad::sample"] = snapshot(inputs["sample"].grad)
                for name, value in parameters.items():
                    arrays[prefix + "parameter::" + name] = snapshot(value)
                    state = optimizer.state[value]
                    if "momentum_buffer" not in state:
                        raise RuntimeError("SGD did not create its genuine momentum state: " + name)
                    arrays[prefix + "momentum::" + name] = snapshot(state["momentum_buffer"])
                for name, value in model.named_buffers():
                    arrays[prefix + "buffer::" + name] = snapshot(value)
            if not any(
                not np.array_equal(initial_parameters[name], arrays[prefix + "parameter::" + name])
                for name in names
            ):
                raise RuntimeError("No model parameter changed across train_step calls")
            if any(not np.isfinite(value).all() for value in arrays.values()):
                raise RuntimeError("Non-finite training trajectory")
            runner._synchronize(torch, args.runtime, args.device)
            observed = runner._device_in_use(torch, args.runtime, args.device)
            if observed != args.device:
                raise RuntimeError("Requested and observed training devices differ")
            fallback_count = jt.core.backend_fallback_count() - before if before is not None else 0
            if fallback_count:
                raise RuntimeError("Backend fallback during real train_step")
            result = {
                "case": args.case,
                "runtime": args.runtime,
                "device": observed,
                "jittor_commit": subprocess.check_output(
                    ["git", "-C", str(args.root), "rev-parse", "HEAD"], text=True
                ).strip(),
                "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE") == "1",
                "steps": args.steps,
                "learning_rate": args.learning_rate,
                "mode": "train",
                "seed": args.seed,
                "loss_history": loss_history,
                "trainable_parameters": names,
                "frozen_parameters": [],
                "input_gradient_names": ["sample"],
                "target_requires_grad": False,
                "buffer_names": sorted(dict(model.named_buffers())),
                "tensors": len(arrays),
                "weights": str(weights),
                "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                "backend": runner._backend_report(args.runtime),
                "tf32": tf32,
                "runtime_conditions": conditions,
                "dependencies": runner._dependency_report(requirements),
                "fallback_count": fallback_count,
                "fallback_policy": "error" if before is not None else None,
                "training_protocol": {
                    "owner": "mmengine.model.BaseModel.train_step",
                    "wrapper": "mmengine.optim.OptimWrapper",
                    "wrapper_built_by": "mmengine.registry.OPTIM_WRAPPERS.build",
                    "optimizer": "torch.optim.SGD",
                    "momentum": 0.9,
                    "clip_grad": {"max_norm": 1.0},
                    "gradient_stage": "after_clip_before_sgd_step",
                    "optimizer_updates": optimizer.updates,
                    "loss_forward_calls": model.loss_forward_calls,
                    "log_keys": log_keys,
                },
            }
    np.savez(args.output, **arrays)
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print("TRAINING_RESULT " + json.dumps(result))
    print("ECOSYSTEM_RESULT " + json.dumps(result))


if __name__ == "__main__":
    main()
