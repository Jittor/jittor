"""Pinned H3 forward/gradient diagnostics, never a coverage or training grant.

CPU is requested honestly: unsupported upstream platform/collective errors
propagate. --torch-math is a separate genuine-Torch calibration, not permission
to relabel a failed default-backend comparison. Raw artifacts must live outside
the source tree. Run the shared scorecard auditor for coverage admission.
"""

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


PINS = {
    "vllm": "98dff2a81d747d1dba01a47f939f48c3526d4206",
    "vllm-omni": "446c2b5474dabffccf9a0034836eeab77c62b4b8",
}


def operation_routes(model):
    """Observe selected methods; a vLLM request does not control Omni dispatch."""
    routes = []
    for name, module in model.named_modules():
        method = getattr(module, "_forward_method", None)
        if method is None:
            continue
        if not callable(method):
            raise RuntimeError("Selected operation route is not callable: " + name)
        routes.append({
            "module": name,
            "class": type(module).__module__ + "." + type(module).__qualname__,
            "method_module": method.__module__,
            "method": method.__qualname__,
        })
    return routes


def require_complete_gradients(records):
    if len(records) != 2 or {row.get("seed") for row in records} != {7, 19}:
        raise RuntimeError("Expected both fixed-input gradient audits")
    for row in records:
        required = {
            "expected_parameters",
            "expected_inputs",
            "missing_parameters",
            "missing_inputs",
        }
        if not required.issubset(row):
            raise RuntimeError("Incomplete gradient audit metadata")
        if not row.get("expected_parameters") or not row.get("expected_inputs"):
            raise RuntimeError("A full-model audit needs expected parameter and input names")
        if row.get("missing_parameters") or row.get("missing_inputs"):
            raise RuntimeError("Full-model gradient contract is incomplete; see gradient_audit")


def source_identity(root):
    sys.path.insert(0, str(root / "agent/skills/torch-compat-repo-runbook/scripts"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _omni_installation import framework_identity

    return framework_identity(root)


def checkout_identity(path, expected):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()

    sha = git("rev-parse", "HEAD")
    if sha != expected or git("status", "--porcelain"):
        raise RuntimeError("The recipe requires the documented clean upstream checkout")
    return {"sha": sha, "remote": git("remote", "get-url", "origin")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=("torch", "jittor"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--mode", choices=("forward", "grad-audit"), default="forward")
    parser.add_argument(
        "--model-mode", choices=("eval", "train"), default="eval",
        help="Set the real Module training flag; does not replace inference kernels",
    )
    parser.add_argument(
        "--torch-math", action="store_true", help="separate genuine-Torch MATH calibration"
    )
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--vllm-checkout", type=Path, required=True)
    parser.add_argument("--omni-checkout", type=Path, required=True)
    parser.add_argument("--vllm-wheel", type=Path, required=True)
    parser.add_argument("--omni-wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.torch_math and args.runtime != "torch":
        parser.error("--torch-math selects only the genuine-Torch calibration")
    args.source_root = args.source_root.resolve()
    expected_probe = args.source_root / "agent/skills/vllm-omni-torch-compat/scripts/probe_h3.py"
    if Path(__file__).resolve() != expected_probe:
        parser.error("The executed probe must belong to the declared source root")
    args.output = args.output.resolve()
    if args.output.suffix != ".npz":
        parser.error("--output must end in .npz")
    try:
        args.output.relative_to(args.source_root)
    except ValueError:
        pass
    else:
        parser.error("Keep raw artifacts outside the source tree")
    for ancestor in (args.output.parent, *args.output.parents):
        if (ancestor / ".git").exists() or (ancestor / ".git").is_symlink():
            parser.error("Keep raw artifacts outside every Git checkout")
    for path in (
        args.output,
        args.output.with_suffix(".json"),
        args.output.with_suffix(".weights.npz"),
        args.output.with_suffix(".installation.json"),
    ):
        if path.exists():
            raise FileExistsError("Choose a fresh output path; preserving existing evidence")
    for path in (args.vllm_wheel, args.omni_wheel):
        if not path.is_file():
            parser.error("Provide the real original source-matched wheel: " + str(path))
    identities = {
        "vllm": checkout_identity(args.vllm_checkout, PINS["vllm"]),
        "vllm-omni": checkout_identity(args.omni_checkout, PINS["vllm-omni"]),
    }
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np

    with ExitStack() as stack:
        run(args, np, stack, identities)


def run(args, np, stack, identities):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    # The shared distribution capture binds vLLM through this wheel input.
    # Use the explicit CLI artifact even if the parent carries another recipe.
    os.environ["JITTOR_VLLM_WHEEL"] = str(args.vllm_wheel)
    before = None
    source_before = source_identity(args.source_root)
    observations = []
    if args.runtime == "jittor":
        os.environ["JT_USE_CUDA"] = str(int(args.device == "cuda"))
        import jittor as jt

        if Path(jt.__file__).resolve() != args.source_root / "python/jittor/__init__.py":
            raise RuntimeError("Candidate imported a different Jittor source tree")
        import jittor_adapters.vllm.layers as adapter_layers

        if Path(adapter_layers.__file__).resolve() != (
            args.source_root / "adapters/jittor_adapters/vllm/layers.py"
        ):
            raise RuntimeError("Candidate imported a different adapter source tree")
        jt.flags.use_cuda = int(args.device == "cuda")
        from jittor._runtime.fallback import forbid_backend_fallbacks

        stack.enter_context(jt.runtime.scope(backend_fallback="error"))
        stack.enter_context(forbid_backend_fallbacks())
        before = jt.core.backend_fallback_count()
    import torch

    if args.torch_math:
        from torch.nn.attention import sdpa_kernel, SDPBackend

        stack.enter_context(sdpa_kernel(SDPBackend.MATH))
    assert hasattr(torch, "_torch_compat_install_context") == (args.runtime == "jittor")
    if args.runtime == "torch" and torch.__version__.split("+")[0] != "2.13.0":
        raise RuntimeError("The genuine oracle recipe is pinned to Torch 2.13.0")

    def snapshot(value):
        if args.runtime == "jittor":
            value.sync()
            if value.location() != ("device" if args.device == "cuda" else "cpu"):
                raise RuntimeError("A measured tensor is on the wrong physical device")
        elif value.device.type != args.device:
            raise RuntimeError("The genuine oracle tensor is on the wrong device")
        row = {"device": value.device.type, "dtype": str(value.dtype), "shape": list(value.shape)}
        if str(value.dtype) not in ("float32", "torch.float32"):
            raise RuntimeError("An observed H3 diagnostic tensor is not FP32")
        if args.runtime == "jittor":
            row.update(location=value.location(), dispatch=jt.core.dispatch_context([value])[0])
            if row["dispatch"] != args.device:
                raise RuntimeError("The H3 tensor dispatched on the wrong device")
        if args.device == "cuda":
            import ctypes

            ordinal = ctypes.c_int(-1)
            status = ctypes.CDLL("libcuda.so.1").cuPointerGetAttribute(
                ctypes.byref(ordinal), 9, ctypes.c_void_p(value.data_ptr()))
            if status != 0 or ordinal.value != value.device.index:
                raise RuntimeError("The actual H3 CUDA pointer belongs to another device")
            row["pointer_device"] = ordinal.value
        observations.append(row)
        array = value.detach().float().cpu().numpy().copy()
        if not np.isfinite(array).all():
            raise RuntimeError("Non-finite measured tensor")
        return array

    torch.set_num_threads(1)
    if args.device == "cuda":
        assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    import vllm
    import vllm_omni
    from _omni_installation import capture as capture_omni, validate as validate_omni
    from _vllm_installation import adapter_installation, device_controls
    from component_measure import installed_distribution
    from registry_inventory import inventory

    sources = {repo: inventory(repo, path) for repo, path in (
        ("vllm", args.vllm_checkout), ("vllm-omni", args.omni_checkout))}
    bindings = {
        "vllm": installed_distribution(sources["vllm"], vllm),
        "vllm-omni": capture_omni(sources["vllm-omni"], vllm_omni, args.omni_wheel),
    }
    controls = device_controls(torch, args.device)
    carrier = adapter_installation(args.source_root) if args.runtime == "jittor" else None
    conditions = {"threads": torch.get_num_threads(),
                  "environment": {key: os.environ.get(key) for key in (
                      "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                      "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}}
    if conditions["threads"] != 1 or any(conditions["environment"][key] != "1"
                                         for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                                     "OPENBLAS_NUM_THREADS")):
        raise RuntimeError("The H3 diagnostic requires matching single-thread runtime conditions")
    installed = {"phase": "installed-source-verified-before-model-import", "runtime": args.runtime,
                 "device": args.device, "source": source_before,
                 "upstream": {repo: value["upstream"] for repo, value in sources.items()},
                 "downstream_distributions": bindings, "device_controls": controls,
                 "runtime_conditions": conditions, "adapter_distribution": carrier,
                 "precision": {"matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                               "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
                               "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                               "matmul_precision": torch.get_float32_matmul_precision()},
                 "interpreter": {"executable": sys.executable,
                                 "is_shim": hasattr(torch, "_torch_compat_install_context"),
                                 "torch_version": str(torch.__version__),
                                 "binary_extension": str(torch._C.__file__)
                                 if args.runtime == "torch" else None,
                                 "binary_sha256": hashlib.sha256(Path(torch._C.__file__).read_bytes()).hexdigest()
                                 if args.runtime == "torch" else None},
                 "numerical_admission": "not_evaluated_by_this_diagnostic"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".installation.json").open("x") as stream:
        json.dump(installed, stream, indent=2)
        stream.write("\n")
    from vllm.config import VllmConfig, CompilationConfig, DeviceConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
        destroy_model_parallel,
        destroy_distributed_environment,
    )
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig, AttentionConfig
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel

    if vllm.__version__ != "0.29.0" or vllm_omni.__version__ != "0.29.0rc2.dev58+g446c2b547":
        raise RuntimeError(
            "This diagnostic recipe is pinned to vLLM 0.29.0 and the documented Omni SCM build"
        )
    import importlib

    executed_sources = {}
    modules = (
        ("vllm", args.vllm_checkout, "vllm.model_executor.layers.activation"),
        ("vllm", args.vllm_checkout, "vllm.model_executor.layers.layernorm"),
        ("vllm", args.vllm_checkout, "vllm.model_executor.layers.linear"),
        (
            "vllm-omni",
            args.omni_checkout,
            "vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer",
        ),
        (
            "vllm-omni",
            args.omni_checkout,
            "vllm_omni.diffusion.attention.ops.minimax_h3_modulation",
        ),
        ("vllm-omni", args.omni_checkout, "vllm_omni.diffusion.layers.norm"),
        ("vllm-omni", args.omni_checkout, "vllm_omni.diffusion.layers.activation"),
        ("vllm-omni", args.omni_checkout, "vllm_omni.diffusion.layers.custom_op"),
    )
    for repo, checkout, name in modules:
        loaded = Path(importlib.import_module(name).__file__)
        relative = Path(*name.split("."))
        relative = (
            relative / "__init__.py"
            if loaded.name == "__init__.py"
            else relative.with_suffix(".py")
        )
        observed = hashlib.sha256(loaded.read_bytes()).hexdigest()
        if observed != hashlib.sha256((checkout / relative).read_bytes()).hexdigest():
            raise RuntimeError("Executed module differs from the clean upstream: " + name)
        executed_sources[name] = {"repo": repo, "path": str(loaded), "sha256": observed}
    print("ACTUAL_IMPORT", torch.__version__, vllm.__version__, vllm_omni.__version__, flush=True)
    arch = dict(
        num_layers=1,
        token_refiner_num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        attention_head_dim=8,
        ffn_hidden_size=64,
        latents_dim=2,
        audio_latents_dim=2,
        patch_size=(1, 1, 1),
        text_dim=16,
        timestep_input_dim=8,
        time_embed_hidden_size=32,
        time_embed_dim=16,
        adaln_out_features=18 * 32,
        final_adaln_out_features=2 * 32,
        rope_inv_freq_len=1,
    )
    config = VllmConfig(
        device_config=DeviceConfig(device=args.device),
        compilation_config=CompilationConfig(mode=0, custom_ops=["none"]),
    )
    od = OmniDiffusionConfig(
        model_class_name="MiniMaxH3DiTModel",
        tf_model_config=TransformerConfig.from_dict(arch),
        diffusion_attention_config=AttentionConfig(default="TORCH_SDPA"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    store = args.output.parent / ("world-" + args.runtime + "-" + str(os.getpid()))
    with set_current_vllm_config(config), set_current_diffusion_config(od):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=store.as_uri(),
            backend="gloo" if args.device == "cpu" else "nccl",
        )
        stack.callback(destroy_distributed_environment)
        initialize_model_parallel(backend="gloo" if args.device == "cpu" else "nccl")
        stack.callback(destroy_model_parallel)
        print("REAL_WORLD_INITIALIZED", args.device, flush=True)
        model = MiniMaxH3DiTModel(od).to(args.device)
        constructed_parameter_requires_grad = {
            name: bool(value.requires_grad) for name, value in model.named_parameters()
        }
        rng = np.random.RandomState(3)
        initial = {}
        with torch.no_grad():
            for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
                array = rng.uniform(-0.1, 0.1, size=tuple(tensor.shape)).astype("float32")
                if "norm" in name and name.endswith("weight"):
                    array += 1
                if name == "rope.inv_freq":
                    array[:] = 1
                tensor.copy_(torch.tensor(array, device=args.device, dtype=tensor.dtype))
                initial[name] = snapshot(tensor)
        model.train(args.model_mode == "train")
        for parameter in model.parameters():
            parameter.requires_grad_(args.mode == "grad-audit")
        print("REAL_MODEL", sum(p.numel() for p in model.parameters()), len(initial), flush=True)
        np.savez(args.output.with_suffix(".weights.npz"), **initial)
        arrays = {}
        gradient_names = []
        gradient_audit = []

        def tensor(data, dtype=None):
            return torch.tensor(data, device=args.device, dtype=dtype)

        for seed in (7, 19):
            model.zero_grad(set_to_none=True)
            rng = np.random.RandomState(seed)
            x = tensor(rng.normal(0, 0.1, (1, 6, 2)).astype("float32"))
            x.requires_grad_(True)
            audio = tensor(rng.normal(0, 0.1, (1, 6, 2)).astype("float32"))
            audio.requires_grad_(True)
            text = tensor(rng.normal(0, 0.1, (2, 16)).astype("float32"))
            text.requires_grad_(True)
            time = tensor([0.2, 0.8], dtype=torch.float32)
            time.requires_grad_(True)
            kwargs = dict(
                x=x,
                audio_x=audio,
                prompt_embeds=text,
                unique_timesteps=time,
                img_position_ids=tensor(
                    [[[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1]]],
                    dtype=torch.int64,
                ),
                inverse_indices=tensor([0, 0, 0, 1, 1, 1], dtype=torch.int64),
                update_mask=tensor([1, 1], dtype=torch.float32),
                update_audio_mask=tensor([1, 1], dtype=torch.float32),
                token_tags=tensor([1, 1, 0, 0, 2, 2], dtype=torch.int64),
                img_pos_info={"position_ids": tensor([2, 3], dtype=torch.int64)},
                audio_pos_info={"position_ids": tensor([4, 5], dtype=torch.int64)},
                text_pos_info={"position_ids": tensor([0, 1], dtype=torch.int64)},
                img_pos_for_infer_output_info={"position_ids": tensor([2, 3], dtype=torch.int64)},
                packed_seq_params={
                    "cu_seqlens_q": tensor([0, 6], dtype=torch.int32),
                    "max_seqlen_q": 6,
                },
                refiner_packed_seq_params={
                    "cu_seqlens_q": tensor([0, 2], dtype=torch.int32),
                    "max_seqlen_q": 2,
                },
            )
            video, audio_out = model(**kwargs)
            print("REAL_FORWARD", seed, tuple(video.shape), tuple(audio_out.shape), flush=True)
            loss = (
                video * tensor(rng.uniform(0.2, 1.1, tuple(video.shape)).astype("float32"))
            ).sum() + (
                audio_out * tensor(rng.uniform(0.2, 1.1, tuple(audio_out.shape)).astype("float32"))
            ).sum()
            for name, value in [("video", video), ("audio", audio_out), ("loss", loss)]:
                arrays[f"seed{seed}/{name}"] = snapshot(value)
            if args.mode == "grad-audit":
                loss.backward()
                parameter_map = dict(model.named_parameters())
                input_map = dict(x=x, audio=audio, text=text, time=time)
                missing_parameters = [
                    name for name, value in parameter_map.items() if value.grad is None
                ]
                missing_inputs = [name for name, value in input_map.items() if value.grad is None]
                for category, mapping in [("paramgrad", parameter_map), ("ingrad", input_map)]:
                    for name, value in mapping.items():
                        if value.grad is not None:
                            key = f"seed{seed}/{category}::{name}"
                            arrays[key] = snapshot(value.grad)
                            gradient_names.append(key)
                gradient_audit.append(
                    dict(
                        seed=seed,
                        expected_parameters=sorted(parameter_map),
                        expected_inputs=sorted(input_map),
                        parameters_requiring_grad=sorted(
                            name for name, value in parameter_map.items() if value.requires_grad
                        ),
                        inputs_requiring_grad=sorted(
                            name for name, value in input_map.items() if value.requires_grad
                        ),
                        missing_parameters=missing_parameters,
                        missing_inputs=missing_inputs,
                    )
                )
                print("REAL_GRADIENT_AUDIT", json.dumps(gradient_audit[-1]), flush=True)

    assert all(np.isfinite(a).all() for a in arrays.values())
    np.savez(args.output, **arrays)
    metadata = dict(
        runtime=args.runtime,
        device=args.device,
        torch_version=str(torch.__version__),
        vllm_version=vllm.__version__,
        omni_version=vllm_omni.__version__,
        omni_origin=vllm_omni.__file__,
        arch=arch,
        gradient_names=gradient_names,
        gradient_audit=gradient_audit,
        trainable_parameters=sorted(
            name for name, value in model.named_parameters() if value.requires_grad
        ),
        frozen_parameters=sorted(
            name for name, value in model.named_parameters() if not value.requires_grad
        ),
        buffer_names=sorted(name for name, _ in model.named_buffers()),
        arrays=len(arrays),
        scope="Tiny actual MiniMaxH3 class diagnostic. Not a coverage collector, engine, checkpoint-family or training acceptance.",
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    metadata["gradient_contract_complete"] = (
        all(not row["missing_parameters"] and not row["missing_inputs"] for row in gradient_audit)
        if args.mode == "grad-audit"
        else None
    )
    metadata.update(
        mode=args.mode,
        source=source_before,
        fixed_input_seeds=[7, 19],
        initial_weight_seed=3,
        model_training=bool(model.training),
        constructed_parameter_requires_grad=constructed_parameter_requires_grad,
        custom_ops=["none"],
        vllm_custom_ops_request=["none"],
        operation_routes=operation_routes(model),
        grad_enabled=bool(torch.is_grad_enabled()),
        model_mode=args.model_mode,
        numerical_admission="not_evaluated_by_this_diagnostic",
        torch_attention_request="MATH" if args.torch_math else "default",
        precision={
            "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "matmul_precision": torch.get_float32_matmul_precision(),
            "candidate_cuda_kernel_math": str(jt.flags.cuda_kernel_math)
            if args.runtime == "jittor"
            else "not-applicable",
        },
        upstream_checkouts=identities,
        executed_module_sources=executed_sources,
        identity_scope="Complete source/wheel, actual installed payload/METADATA/RECORD and framework source binding",
        downstream_distributions=bindings,
        device_controls=controls,
        runtime_conditions=conditions,
        adapter_distribution=carrier,
        tensor_observations=observations,
        interpreter={
            "executable": sys.executable,
            "is_shim": args.runtime == "jittor",
            "binary_extension": str(torch._C.__file__) if args.runtime == "torch" else None,
            "binary_sha256": hashlib.sha256(Path(torch._C.__file__).read_bytes()).hexdigest()
            if args.runtime == "torch" else None,
        },
    )
    refreshed = installed_distribution(sources["vllm"], vllm)
    if refreshed != bindings["vllm"]:
        raise RuntimeError("Actual vLLM installation changed during the H3 diagnostic")
    validate_omni(bindings["vllm-omni"], sources["vllm-omni"], args.omni_wheel)
    if source_identity(args.source_root) != source_before:
        raise RuntimeError("Source changed during measurement")
    if before is not None:
        assert jt.core.backend_fallback_count() == before
        metadata["fallback_count"] = 0
        metadata["fallback_policy"] = "error"
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        "OMNI_H3_DIAGNOSTIC",
        json.dumps(
            {
                key: metadata[key]
                for key in (
                    "runtime",
                    "device",
                    "mode",
                    "arrays",
                    "gradient_contract_complete",
                    "torch_attention_request",
                )
            }
        ),
        flush=True,
    )

    if args.mode == "grad-audit":
        require_complete_gradients(gradient_audit)


if __name__ == "__main__":
    main()
