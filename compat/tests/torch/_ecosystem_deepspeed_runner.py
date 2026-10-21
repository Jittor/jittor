"""Exercise the upstream Engine training/optimizer/checkpoint lifecycle.

This protocol deliberately calls Engine.backward and Engine.step. Generic
model runners that construct another optimizer cannot validate DeepSpeed.
"""
import argparse
import ctypes
from contextlib import ExitStack
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
import os
import numbers
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import uuid

import numpy as np

from _ecosystem_runner import (
    _backend_report, _configure_tf32, _dependency_report, _import_torch,
    _numpy_snapshot, _runtime_conditions, _select_device, _synchronize,
)
from _helpers.child_process import PYTHON
import _ecosystem_deepspeed_cases as cases
from _ecosystem_deepspeed_multirank import file_rendezvous


def _snapshot(value):
    # NumPy has no Torch bf16 dtype. Observe values in fp32 while separately
    # asserting the operands/output actually used the requested bf16 dtype.
    if str(value.dtype) in ("bfloat16", "torch.bfloat16"):
        value = value.float()
    return _numpy_snapshot(value)


def _weights_digest(values):
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        value = np.ascontiguousarray(value)
        digest.update(json.dumps([name, value.dtype.str, list(value.shape)],
                                 separators=(",", ":")).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _protocol_spec(args):
    return {
        "id": "deepspeed-zero-training", "world_size": args.world_size,
        "precision": "bf16" if args.dtype == "bfloat16" else "fp32",
        "ranks": list(range(args.world_size)), "zero_stage": args.zero_stage,
        "optimizer": args.optimizer, "gradient_accumulation_steps": 2,
        "updates": 3, "microsteps": 6, "checkpoint_after_microstep": 4,
        "resume_microsteps": [4, 5], "seed": 0,
        "input_perturbations": [0., .125],
        "model": {"id": "residual-linear-gelu-layernorm-v1", "micro_batch": 2,
                  "input_features": 8, "hidden_features": 16, "output_features": 4},
    }


def _measurement_sources():
    directory = Path(__file__).resolve().parent
    repository = directory.parents[2]
    return {str(path.relative_to(repository)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (directory / name for name in (
                "_ecosystem_deepspeed_runner.py", "_ecosystem_deepspeed_multirank.py",
                "_ecosystem_deepspeed_cases.py", "_ecosystem_deepspeed_comparison.py"))}


def _cuda_identity(torch, device):
    if device != "cuda":
        return {"cuda_visible_devices": "", "rank_device_uuid": None, "logical_device_index": None}
    logical = int(torch.cuda.current_device())
    driver = ctypes.CDLL("libcuda.so.1")
    physical, identifier = ctypes.c_int(), ctypes.create_string_buffer(16)
    if driver.cuDeviceGet(ctypes.byref(physical), logical) or driver.cuDeviceGetUuid(identifier, physical):
        raise RuntimeError("Cannot bind the actual CUDA device UUID")
    return {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "rank_device_uuid": "GPU-" + str(uuid.UUID(bytes=identifier.raw)),
            "logical_device_index": logical}


def _measurement_platform(torch, args, precision):
    """Observe the actual process, including its assigned physical devices."""
    def command(*arguments):
        return subprocess.check_output(list(arguments), text=True).strip()
    cpu = platform.processor() or platform.machine()
    if Path("/proc/cpuinfo").is_file():
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), cpu)
    allocation = os.environ.get("JITTOR_DEEPSPEED_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "")).split(",") \
        if args.device == "cuda" else []
    selected = []
    if args.device == "cuda":
        rows = [line.split(", ") for line in command("nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader").splitlines()]
        for device in allocation:
            matches = [row for row in rows if device in row[:2]]
            if len(matches) != 1:
                raise ValueError("Cannot bind the actual DeepSpeed CUDA allocation")
            selected.append(matches[0])
    native = None
    if args.runtime == "jittor":
        import jittor as jt
        native = jt
    gcc = native.compiler.cc_path if native else shutil.which("g++")
    nvcc = native.compiler.nvcc_path if native else shutil.which("nvcc")
    flags = {"use_cuda": int(native.flags.use_cuda) if native else int(args.device == "cuda"),
             "use_acl": 0,
             "graph_replay": "default" if native and native.flags.auto_graph_replay else "eager",
             "tf32": precision["matmul"], "hf32": False}
    return {"os": platform.platform(), "cpu": cpu,
            "accelerator_model": [row[2] for row in selected] or "not-applicable",
            "card_count": len(selected), "card_indices": allocation,
            "physical_devices": [{"index": row[0], "uuid": row[1]} for row in selected],
            "driver": selected[0][3] if selected else "not-applicable",
            "cuda": str(torch.version.cuda) if args.device == "cuda" else "not-applicable",
            "cann": "not-applicable", "nvcc": command(nvcc, "--version") if args.device == "cuda" else "not-applicable",
            "gcc": command(gcc, "--version"),
            "nvidia_wheels": {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
                              if dist.metadata.get("Name", "").lower().startswith("nvidia-")},
            "jittor_flags": flags, "precision": precision}


def _assert_device(value, device, runtime, dtype=None):
    if value.device.type != device:
        raise AssertionError("DeepSpeed tensor is on %s, requested %s" % (value.device, device))
    if dtype is not None and value.dtype != dtype:
        raise AssertionError("DeepSpeed gradient dtype %s differs from requested %s" % (value.dtype, dtype))
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
                arrays[prefix + "/%s/%s" % (key, name)] = _snapshot(value)
            elif isinstance(value, numbers.Number):
                arrays[prefix + "/%s/%s" % (key, name)] = np.asarray(value)
    return [{key: value for key, value in group.items() if key in (
        "lr", "betas", "eps", "weight_decay", "foreach", "params")}
            for group in state["param_groups"]]


def _snapshot_partitions(torch, optimizer, model, args, arrays, prefix):
    wrapped = getattr(optimizer, "optimizer", optimizer)
    sizes = []
    for index, group in enumerate(wrapped.param_groups):
        for slot, parameter in enumerate(group["params"]):
            _assert_device(parameter, args.device, args.runtime)
            if args.dtype == "bfloat16" and parameter.dtype != torch.float32:
                raise AssertionError("DeepSpeed master partition is not fp32")
            arrays[prefix + "/%d/%d" % (index, slot)] = _snapshot(parameter)
            sizes.append(int(parameter.numel()))
    total = sum(int(parameter.numel()) for parameter in model.parameters())
    if args.world_size > 1 and not 0 < sum(sizes) < total:
        raise AssertionError("DeepSpeed optimizer did not actually partition model parameters")
    return sizes


def _full_gradient_dtype(torch, parameter):
    # Upstream safe_get_full_grad returns an existing param.grad directly.
    # BF16_Optimizer retains it in bf16; the partitioned ZeRO2 getter returns
    # the configured fp32 accumulation storage when param.grad is absent.
    gradient = parameter.grad
    return gradient.dtype if gradient is not None else torch.float32


def _assert_strict_resume(arrays, gradient_dtypes=None):
    count = 0
    for before, after in (("saved_optimizer/", "restored_optimizer/"),
                          ("saved_master_partition/", "restored_master_partition/"),
                          ("final_optimizer/", "resumed_optimizer/"),
                          ("final_master_partition/", "resumed_master_partition/"),
                          ("step4/", "resumed4/"), ("step5/", "resumed5/")):
        left = {key[len(before):] for key in arrays if key.startswith(before)}
        right = {key[len(after):] for key in arrays if key.startswith(after)}
        if not left or left != right:
            raise AssertionError("Incomplete strict resume schema: " + before)
        for suffix in left:
            np.testing.assert_array_equal(arrays[before + suffix], arrays[after + suffix],
                                          err_msg=before + suffix)
            if gradient_dtypes is not None and "_grad/" in suffix:
                if gradient_dtypes[before + suffix] != gradient_dtypes[after + suffix]:
                    raise AssertionError("Resumed gradient dtype differs: " + before + suffix)
            count += 1
    changed = sum(np.count_nonzero(arrays[key] != arrays[
        "final_master_partition/" + key[len("saved_master_partition/"):]])
                  for key in arrays if key.startswith("saved_master_partition/"))
    if not changed:
        raise AssertionError("DeepSpeed's final optimizer step did not update a master partition")
    return count, int(changed)


def _microstep(torch, deepspeed, engine, args, index, arrays, prefix, incoming, gradient_dtypes):
    incoming.clear()
    inputs = torch.arange(16, dtype=torch.float32, device=engine.device).reshape(2, 8)
    perturbation = args.perturbation + .125 * args.rank
    inputs = inputs * .03 + .01 * index + perturbation
    target = torch.arange(8, dtype=torch.float32, device=engine.device).reshape(2, 4)
    target = target * -.007 - perturbation
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    inputs = inputs.to(dtype).requires_grad_(True)
    target = target.to(dtype).requires_grad_(True)
    output = engine(inputs)
    for value in (inputs, target, output, *engine.module.parameters()):
        _assert_device(value, args.device, args.runtime)
        if value.dtype != dtype:
            raise AssertionError("DeepSpeed did not execute the requested " + args.dtype)
    loss = (output.float() - target.float()).square().mean()
    engine.backward(loss)
    arrays[prefix + "/output"] = _snapshot(output)
    arrays[prefix + "/loss"] = _snapshot(loss)
    for name, value in (("inputs", inputs), ("target", target)):
        arrays[prefix + "/input/" + name] = _snapshot(value)
        if value.grad is None:
            raise AssertionError("Missing requested DeepSpeed input gradient: " + name)
        _assert_device(value.grad, args.device, args.runtime, value.dtype)
        key = prefix + "/input_grad/" + name
        arrays[key] = _snapshot(value.grad)
        gradient_dtypes[key] = str(value.grad.dtype).rsplit(".", 1)[-1]
    for name, parameter in engine.module.named_parameters():
        if name not in incoming:
            raise AssertionError("DeepSpeed did not compute the requested parameter gradient: " + name)
        key = prefix + "/incoming_parameter_grad/" + name
        arrays[key], gradient_dtypes[key] = incoming[name]
        # ZeRO 2 exposes full accumulated gradients only at the GAS boundary.
        # Observe every microstep's incoming gradients through real public
        # Tensor hooks; request partitioned accumulated storage when upstream
        # actually makes that public getter available.
        if args.zero_stage == 2 and not engine.is_gradient_accumulation_boundary():
            continue
        gradient = deepspeed.utils.safe_get_full_grad(parameter)
        if gradient is None:
            raise AssertionError("Missing requested DeepSpeed parameter gradient: " + name)
        expected = _full_gradient_dtype(torch, parameter)
        _assert_device(gradient, args.device, args.runtime, expected)
        key = prefix + "/parameter_grad/" + name
        arrays[key] = _snapshot(gradient)
        gradient_dtypes[key] = str(gradient.dtype).rsplit(".", 1)[-1]
    engine.step()
    for name, parameter in engine.module.named_parameters():
        arrays[prefix + "/parameter/" + name] = _snapshot(parameter)
    _synchronize(torch, args.runtime, args.device)
    from deepspeed.runtime.engine import _ENGINE_BACKWARD_GRAPH_TRACKER
    if _ENGINE_BACKWARD_GRAPH_TRACKER.active_registration_count():
        raise AssertionError("DeepSpeed retained a finished backward graph")


def _execute(args):
    if args.world_size > 1 and (not args.rendezvous or not args.checkpoint):
        raise ValueError("Multi-rank Engine requires shared rendezvous and checkpoint paths")
    if args.runtime == "jittor":
        os.environ["JITTOR_TORCH_SHIM"] = "1"
    os.environ["DS_ACCELERATOR"] = args.device
    os.environ["LOCAL_RANK"] = (os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", str(args.rank))
                                if args.runtime == "jittor" and args.device == "cpu" else "0")
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
        torch.manual_seed(0)
        import deepspeed
        dependencies = _dependency_report(("deepspeed",))
        if dependencies["deepspeed"]["version"].split("+", 1)[0] != "0.19.7":
            raise RuntimeError("DeepSpeed gate requires the verified 0.19.7 source")
        if args.rendezvous:
            rendezvous = args.rendezvous
        else:
            directory = stack.enter_context(tempfile.TemporaryDirectory(
                prefix="deepspeed-rendezvous-"))
            rendezvous = file_rendezvous(directory)
        transport = ("nccl" if args.device == "cuda" else
                     "mpi" if args.runtime == "jittor" and args.world_size > 1 else "gloo")
        deepspeed.init_distributed(
            dist_backend=transport,
            auto_mpi_discovery=False, init_method=rendezvous,
            rank=args.rank, world_size=args.world_size,
        )
        arrays = {}
        gradient_dtypes = {}
        def initialize():
            original = cases.model(torch)
            configuration = cases.config(args.optimizer, args.zero_stage)
            configuration["train_batch_size"] *= args.world_size
            if args.dtype == "bfloat16":
                configuration["bf16"] = {"enabled": True}
                configuration["communication_data_type"] = "fp32"
                configuration["data_types"] = {"grad_accum_dtype": "fp32"}
                configuration["zero_optimization"].update(
                    reduce_bucket_size=100, allgather_bucket_size=100)
            engine, optimizer, _, _ = deepspeed.initialize(
                model=original, model_parameters=original.parameters(),
                config=configuration,
            )
            if not isinstance(engine, deepspeed.DeepSpeedEngine):
                raise AssertionError("Training did not use the real upstream DeepSpeedEngine")
            # Observe the actual initial training weights, after upstream has
            # performed its requested precision conversion and before step 0.
            # The temporary FP32 fixture is not the BF16 model being trained.
            expected = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
            for name, parameter in engine.module.named_parameters():
                _assert_device(parameter, args.device, args.runtime, expected)
                key, value = "initial/parameter/" + name, _snapshot(parameter)
                if key in arrays:
                    np.testing.assert_array_equal(arrays[key], value,
                                                  err_msg="Fresh Engine changed original weights: " + name)
                else:
                    arrays[key] = value
            incoming = {}
            for name, parameter in engine.module.named_parameters():
                def observe(gradient, name=name, dtype=parameter.dtype):
                    _assert_device(gradient, args.device, args.runtime, dtype)
                    incoming[name] = (_snapshot(gradient), str(gradient.dtype).rsplit(".", 1)[-1])
                stack.callback(parameter.register_hook(observe).remove)
            return engine, optimizer, incoming
        try:
            engine, optimizer, incoming = initialize()
            checkpoint_context = (nullcontext(str(args.checkpoint)) if args.checkpoint
                                  else tempfile.TemporaryDirectory(prefix="deepspeed-checkpoint-"))
            with checkpoint_context as checkpoint:
                for index in range(6):
                    _microstep(torch, deepspeed, engine, args, index, arrays, "step%d" % index,
                               incoming, gradient_dtypes)
                    if index == 3:
                        if engine.global_steps != 2:
                            raise AssertionError("Gradient accumulation did not produce two updates")
                        if not engine.save_checkpoint(checkpoint, tag="resume", client_state={"microstep": 4}):
                            raise AssertionError("DeepSpeed failed to save the checkpoint")
                        _snapshot_optimizer(optimizer, arrays, "saved_optimizer")
                        _snapshot_partitions(torch, optimizer, engine.module, args, arrays, "saved_master_partition")
                groups = _snapshot_optimizer(optimizer, arrays, "final_optimizer")
                partition_sizes = _snapshot_partitions(
                    torch, optimizer, engine.module, args, arrays, "final_master_partition")
                if engine.global_steps != 3:
                    raise AssertionError("Gradient accumulation did not produce three updates")
                resumed, resumed_optimizer, resumed_incoming = initialize()
                path, client_state = resumed.load_checkpoint(checkpoint, tag="resume")
                if path is None or client_state.get("microstep") != 4 or resumed.global_steps != 2:
                    raise AssertionError("DeepSpeed failed to restore checkpoint/client state")
                _snapshot_optimizer(resumed_optimizer, arrays, "restored_optimizer")
                _snapshot_partitions(torch, resumed_optimizer, resumed.module, args, arrays, "restored_master_partition")
                for key in list(arrays):
                    if key.startswith(("saved_optimizer/", "saved_master_partition/")):
                        restored_key = "restored_" + key[len("saved_"): ]
                        np.testing.assert_array_equal(arrays[restored_key], arrays[key])
                for index in (4, 5):
                    _microstep(torch, deepspeed, resumed, args, index, arrays, "resumed%d" % index,
                               resumed_incoming, gradient_dtypes)
                    for suffix in ("output", "loss"):
                        np.testing.assert_array_equal(arrays["resumed%d/" % index + suffix],
                                                      arrays["step%d/" % index + suffix])
                    for name, _ in resumed.module.named_parameters():
                        np.testing.assert_array_equal(arrays["resumed%d/parameter/" % index + name],
                                                      arrays["step%d/parameter/" % index + name])
                _snapshot_optimizer(resumed_optimizer, arrays, "resumed_optimizer")
                _snapshot_partitions(torch, resumed_optimizer, resumed.module, args, arrays, "resumed_master_partition")
                if resumed.global_steps != 3:
                    raise AssertionError("Resumed training did not advance its update count")
                strict_fields, changed_elements = _assert_strict_resume(arrays, gradient_dtypes)
            _synchronize(torch, args.runtime, args.device)
            fallback_count = (jt.core.backend_fallback_count() - initial_fallbacks
                              if args.runtime == "jittor" else 0)
            metadata = {
                "runtime": args.runtime, "device": args.device, "world_size": args.world_size,
                "seed": 0, "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE") == "1",
                "protocol": _protocol_spec(args),
                "process_group_backend": str(torch.distributed.get_backend()),
                "launch_entrypoint": os.environ.get("JITTOR_DEEPSPEED_LAUNCHER", "independent-rank-process"),
                "measurement_source_sha256": _measurement_sources(),
                "initial_weights_sha256": _weights_digest({key[len("initial/parameter/"):]: value
                    for key, value in arrays.items() if key.startswith("initial/parameter/")}),
                "interpreter": {"executable": PYTHON, "pid": os.getpid(),
                                "is_shim": hasattr(torch, "_torch_compat_install_context"),
                                "torch_version": str(torch.__version__),
                                "binary_extension": str(torch._C.__file__) if args.runtime == "torch" else None,
                                "binary_sha256": hashlib.sha256(Path(torch._C.__file__).read_bytes()).hexdigest()
                                    if args.runtime == "torch" else None},
                "rank": args.rank, "dtype": args.dtype,
                "optimizer_partition_sizes": partition_sizes,
                "model_parameter_elements": sum(int(value.numel()) for value in engine.module.parameters()),
                "optimizer_class": type(optimizer).__module__ + "." + type(optimizer).__name__,
                "optimizer": args.optimizer, "zero_stage": args.zero_stage,
                "perturbation": args.perturbation, "engine_class": type(engine).__module__ + "." + type(engine).__name__,
                "parameter_gradients_per_microstep": len(list(engine.module.parameters())),
                "input_gradients_per_microstep": 2, "microsteps": 6, "global_steps": 3,
                "dependencies": dependencies, "backend": _backend_report(args.runtime),
                "runtime_conditions": _runtime_conditions(torch, precision),
                "fallback_policy": "error", "fallback_count": fallback_count,
                "optimizer_groups": groups, "torch_version": str(torch.__version__),
                "torch_origin": str(torch.__file__),
                "strict_resume_fields": strict_fields, "updated_master_elements": changed_elements,
                "gradient_dtypes": gradient_dtypes,
                "trainable_parameters": sorted(name for name, value in engine.module.named_parameters()
                                                if value.requires_grad),
                "frozen_parameters": sorted(name for name, value in engine.module.named_parameters()
                                             if not value.requires_grad),
                "jittor_commit": subprocess.check_output(
                    ["git", "-C", str(Path(__file__).resolve().parents[3]), "rev-parse", "HEAD"],
                    text=True).strip(),
            }
            metadata.update(_cuda_identity(torch, args.device))
            metadata["platform"] = _measurement_platform(torch, args, precision)
            if args.runtime == "jittor":
                metadata["native_runtime_flags"] = {name: int(getattr(jt.flags, name)) for name in
                    ("use_cuda", "auto_graph_replay", "auto_graph_replay_retain_bytes")}
                if args.device == "cpu":
                    from jittor._runtime.backend_libraries import get_library_ops
                    metadata["cpu_matmul_provider"] = (
                        "oneDNN" if get_library_ops("mkl", load=False) is not None else "native_opmath")
            if args.runtime == "torch":
                metadata["torch_binary"] = str(torch._C.__file__)
                metadata["torch_binary_sha256"] = hashlib.sha256(Path(torch._C.__file__).read_bytes()).hexdigest()
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
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--world-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--rank", type=int, choices=(0, 1), default=0)
    parser.add_argument("--rendezvous")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _execute(args)


if __name__ == "__main__":
    main()
