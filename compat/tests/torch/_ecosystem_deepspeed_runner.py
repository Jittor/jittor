"""Exercise the upstream Engine training/optimizer/checkpoint lifecycle.

This protocol deliberately calls Engine.backward and Engine.step. Generic
model runners that construct another optimizer cannot validate DeepSpeed.
"""
import argparse
from contextlib import ExitStack
import json
import os
import numbers
from pathlib import Path
import socket
import tempfile

import numpy as np

from _ecosystem_runner import (
    _backend_report, _configure_tf32, _dependency_report, _import_torch,
    _numpy_snapshot, _runtime_conditions, _select_device, _synchronize,
)
import _ecosystem_deepspeed_cases as cases


def _assert_device(value, device, runtime):
    if value.device.type != device:
        raise AssertionError("DeepSpeed tensor is on %s, requested %s" % (value.device, device))
    if runtime == "jittor":
        value.sync()
        if value.location() != ("device" if device == "cuda" else "cpu"):
            raise AssertionError("DeepSpeed tensor residency does not match requested " + device)


def _snapshot_optimizer(optimizer, arrays, prefix):
    state = optimizer.state_dict()
    if "base_optimizer_state" in state:
        if "base_optimizer_state_step" in state:
            arrays[prefix + "/wrapper_step"] = np.asarray(state["base_optimizer_state_step"])
        state = state["base_optimizer_state"]
    for key, fields in state["state"].items():
        for name, value in fields.items():
            if hasattr(value, "detach"):
                arrays[prefix + "/%s/%s" % (key, name)] = _numpy_snapshot(value)
            elif isinstance(value, numbers.Number):
                arrays[prefix + "/%s/%s" % (key, name)] = np.asarray(value)
    return [{key: value for key, value in group.items() if key in (
        "lr", "betas", "eps", "weight_decay", "foreach", "params")}
            for group in state["param_groups"]]


def _microstep(torch, deepspeed, engine, args, index, arrays, prefix, incoming):
    incoming.clear()
    inputs = torch.arange(16, dtype=torch.float32, device=engine.device).reshape(2, 8)
    inputs = (inputs * .03 + .01 * index + args.perturbation).requires_grad_(True)
    target = torch.arange(8, dtype=torch.float32, device=engine.device).reshape(2, 4)
    target = (target * -.007 - args.perturbation).requires_grad_(True)
    output = engine(inputs)
    for value in (inputs, target, output, *engine.module.parameters()):
        _assert_device(value, args.device, args.runtime)
    loss = (output - target).square().mean()
    engine.backward(loss)
    arrays[prefix + "/output"] = _numpy_snapshot(output)
    arrays[prefix + "/loss"] = _numpy_snapshot(loss)
    for name, value in (("inputs", inputs), ("target", target)):
        if value.grad is None:
            raise AssertionError("Missing requested DeepSpeed input gradient: " + name)
        _assert_device(value.grad, args.device, args.runtime)
        arrays[prefix + "/input_grad/" + name] = _numpy_snapshot(value.grad)
    for name, parameter in engine.module.named_parameters():
        if name not in incoming:
            raise AssertionError("DeepSpeed did not compute the requested parameter gradient: " + name)
        arrays[prefix + "/incoming_parameter_grad/" + name] = incoming[name]
        # ZeRO 2 exposes full accumulated gradients only at the GAS boundary.
        # Observe every microstep's incoming gradients through real public
        # Tensor hooks; request partitioned accumulated storage when upstream
        # actually makes that public getter available.
        if args.zero_stage == 2 and not engine.is_gradient_accumulation_boundary():
            continue
        gradient = deepspeed.utils.safe_get_full_grad(parameter)
        if gradient is None:
            raise AssertionError("Missing requested DeepSpeed parameter gradient: " + name)
        _assert_device(gradient, args.device, args.runtime)
        arrays[prefix + "/parameter_grad/" + name] = _numpy_snapshot(gradient)
    engine.step()
    for name, parameter in engine.module.named_parameters():
        arrays[prefix + "/parameter/" + name] = _numpy_snapshot(parameter)
    _synchronize(torch, args.runtime, args.device)
    from deepspeed.runtime.engine import _ENGINE_BACKWARD_GRAPH_TRACKER
    if _ENGINE_BACKWARD_GRAPH_TRACKER.active_registration_count():
        raise AssertionError("DeepSpeed retained a finished backward graph")


def _execute(args):
    if args.runtime == "jittor":
        os.environ["JITTOR_TORCH_SHIM"] = "1"
    os.environ["DS_ACCELERATOR"] = args.device
    os.environ["LOCAL_RANK"] = "0"
    torch = _import_torch(args.runtime)
    with ExitStack() as stack:
        _select_device(torch, args.runtime, args.device, policy_stack=stack)
        if args.runtime == "jittor":
            import jittor as jt
            from jittor._runtime import forbid_backend_fallbacks
            from jittor.compat.module_patcher import install_module_patches
            report = install_module_patches(expected_entry_points=("jittor_deepspeed",))
            if not any(item.name == "jittor_deepspeed" and item.status == "loaded"
                       for item in report.results):
                raise RuntimeError("The independent DeepSpeed adapter is not installed")
            stack.enter_context(forbid_backend_fallbacks())
            initial_fallbacks = jt.core.backend_fallback_count()
        precision = _configure_tf32(torch, args.device)
        import deepspeed
        dependencies = _dependency_report(("deepspeed",))
        if dependencies["deepspeed"]["version"].split("+", 1)[0] != "0.19.7":
            raise RuntimeError("DeepSpeed gate requires the verified 0.19.7 source")
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        deepspeed.init_distributed(
            dist_backend="nccl" if args.device == "cuda" else "gloo",
            auto_mpi_discovery=False, init_method="tcp://127.0.0.1:%d" % port,
            rank=0, world_size=1,
        )
        arrays = {}
        def initialize():
            original = cases.model(torch)
            engine, optimizer, _, _ = deepspeed.initialize(
                model=original, model_parameters=original.parameters(),
                config=cases.config(args.optimizer, args.zero_stage),
            )
            if not isinstance(engine, deepspeed.DeepSpeedEngine):
                raise AssertionError("Training did not use the real upstream DeepSpeedEngine")
            incoming = {}
            for name, parameter in engine.module.named_parameters():
                def observe(gradient, name=name):
                    _assert_device(gradient, args.device, args.runtime)
                    incoming[name] = _numpy_snapshot(gradient)
                stack.callback(parameter.register_hook(observe).remove)
            return engine, optimizer, incoming
        try:
            engine, optimizer, incoming = initialize()
            with tempfile.TemporaryDirectory(prefix="deepspeed-checkpoint-") as checkpoint:
                for index in range(6):
                    _microstep(torch, deepspeed, engine, args, index, arrays, "step%d" % index, incoming)
                    if index == 3:
                        if engine.global_steps != 2:
                            raise AssertionError("Gradient accumulation did not produce two updates")
                        if not engine.save_checkpoint(checkpoint, tag="resume", client_state={"microstep": 4}):
                            raise AssertionError("DeepSpeed failed to save the checkpoint")
                        _snapshot_optimizer(optimizer, arrays, "saved_optimizer")
                groups = _snapshot_optimizer(optimizer, arrays, "final_optimizer")
                if engine.global_steps != 3:
                    raise AssertionError("Gradient accumulation did not produce three updates")
                resumed, resumed_optimizer, resumed_incoming = initialize()
                path, client_state = resumed.load_checkpoint(checkpoint, tag="resume")
                if path is None or client_state.get("microstep") != 4 or resumed.global_steps != 2:
                    raise AssertionError("DeepSpeed failed to restore checkpoint/client state")
                _snapshot_optimizer(resumed_optimizer, arrays, "restored_optimizer")
                for index in (4, 5):
                    _microstep(torch, deepspeed, resumed, args, index, arrays, "resumed%d" % index, resumed_incoming)
                    for suffix in ("output", "loss"):
                        np.testing.assert_allclose(arrays["resumed%d/" % index + suffix],
                                                   arrays["step%d/" % index + suffix], rtol=2e-6, atol=2e-7)
                    for name, _ in resumed.module.named_parameters():
                        np.testing.assert_allclose(arrays["resumed%d/parameter/" % index + name],
                                                   arrays["step%d/parameter/" % index + name], rtol=2e-6, atol=2e-7)
                _snapshot_optimizer(resumed_optimizer, arrays, "resumed_optimizer")
                if resumed.global_steps != 3:
                    raise AssertionError("Resumed training did not advance its update count")
            _synchronize(torch, args.runtime, args.device)
            fallback_count = (jt.core.backend_fallback_count() - initial_fallbacks
                              if args.runtime == "jittor" else 0)
            metadata = {
                "runtime": args.runtime, "device": args.device, "world_size": 1,
                "optimizer": args.optimizer, "zero_stage": args.zero_stage,
                "perturbation": args.perturbation, "engine_class": type(engine).__module__ + "." + type(engine).__name__,
                "parameter_gradients_per_microstep": len(list(engine.module.parameters())),
                "input_gradients_per_microstep": 2, "microsteps": 6, "global_steps": 3,
                "dependencies": dependencies, "backend": _backend_report(args.runtime),
                "runtime_conditions": _runtime_conditions(torch, precision),
                "fallback_policy": "error", "fallback_count": fallback_count,
                "optimizer_groups": groups, "torch_version": str(torch.__version__),
                "torch_origin": str(torch.__file__),
            }
            if args.runtime == "torch":
                metadata["torch_binary"] = str(torch._C.__file__)
            arrays["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
            np.savez(args.output, **arrays)
            print(json.dumps(metadata, sort_keys=True))
        finally:
            torch.distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", choices=("jittor", "torch"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--optimizer", choices=("Adam", "AdamW"), default="AdamW")
    parser.add_argument("--zero-stage", type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument("--perturbation", type=float, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _execute(args)


if __name__ == "__main__":
    main()
