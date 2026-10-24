"""Exercise a minimal, single-process verl rollout/reward/advantage/update loop.

Run this exact script in separate official Torch and Jittor interpreters, using
``--weights-out`` on the oracle and ``--weights-in`` on the candidate. Outputs
are an NPZ containing every step's tensors, every parameter gradient and every
post-update parameter, plus a JSON manifest beside it. Compare both artifacts;
a successful standalone run is NOT a numerical parity result.

This is an integration fixture, NOT verl's Ray/FSDP trainer, human preference
data, realistic task-quality evaluation, or performance benchmark. ``ppo``
uses rule scores and GAE. ``grpo`` uses the same generated responses with the
real grouped outcome estimator. ``rlhf`` adds an independently materialized
frozen reference policy, a separate Transformer reward model, verl's KL reward,
GAE and a learned critic. ``rlvr`` uses a deterministic response verifier and
the grouped estimator. The tiny reward model is an execution fixture, not a
preference-trained model, and the verifier is not a benchmark-quality judge.
PPO uses a fixed zero value baseline unless --critic learned is selected; that
mode trains a separate Transformer backbone and scalar value head with verl's
clipped value loss and records every critic gradient and updated parameter.
GRPO/RLVR use synthetic prompt pairs differing in their final token and a
documented row-dependent target so greedy rollouts cannot make the group
variance identically zero. This deliberately tests the real grouping algorithm
without claiming on-policy task quality.
Integer token inputs have no gradient. Schema 4 saves the complete retained
embedding-input gradient, preserving the embedding parameter graph, as well as
the historical scalar multiplier gradient. Full snapshots are intended
for tiny models: ``--model`` can consume substantial host RAM and disk space.
Schema 4 is pinned to Transformers 5.9.0 dense Qwen3/Qwen3.5 text architectures;
visual/MTP branches are outside the declared text task, not silently frozen.
For PPO with a learned critic, ``baseline`` and ``resume`` phases additionally
exercise a fresh-process model+critic checkpoint continuation.  The checkpoint
is runtime-local (never loaded across frameworks); the shared official initial
weights and the saved trajectories remain the cross-framework oracle.  SGD has
no momentum state in this fixture, so there is deliberately no hidden optimizer
slot to restore.  Multi-process support is not claimed by this script.
"""

import argparse
import copy
from contextlib import ExitStack
import importlib.util
import importlib.metadata
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

import numpy as np


def parameter_spec(config, role):
    """Independent shape/alias oracle for the explicitly supported text models.

    This describes HF architecture, not runtime named_parameters: both runtimes
    dropping the same parameter must fail. Unknown architectures fail closed.
    """
    assert role in ("actor", "critic")
    kind = config["model_type"]
    assert kind in ("qwen3", "qwen3_5_text"), "unverified text architecture: " + kind
    assert not config.get("mlp_bias", False), "unverified MLP bias"
    assert not config.get("attention_bias", False), "unverified attention bias"
    h, m, v = (int(config[k]) for k in ("hidden_size", "intermediate_size", "vocab_size"))
    heads, kv = int(config["num_attention_heads"]), int(config["num_key_value_heads"])
    d = int(config.get("head_dim") or h // heads)
    layers = int(config["num_hidden_layers"])
    assert min(h, m, v, heads, kv, d, layers) > 0
    types = config.get("layer_types", ["full_attention"] * layers)
    assert len(types) == layers
    prefix = "model." if role == "actor" else "backbone."
    shapes = {prefix + "embed_tokens.weight": [v, h], prefix + "norm.weight": [h]}
    for index, layer_type in enumerate(types):
        base = prefix + "layers.{}.".format(index)
        entries = {"input_layernorm.weight": [h], "post_attention_layernorm.weight": [h],
                   "mlp.gate_proj.weight": [m, h], "mlp.up_proj.weight": [m, h],
                   "mlp.down_proj.weight": [h, m]}
        if layer_type in ("full_attention", "sliding_attention"):
            q = heads * d * (2 if kind == "qwen3_5_text" else 1)
            entries.update({"self_attn.q_proj.weight": [q, h],
                            "self_attn.k_proj.weight": [kv * d, h],
                            "self_attn.v_proj.weight": [kv * d, h],
                            "self_attn.o_proj.weight": [h, heads * d],
                            "self_attn.q_norm.weight": [d], "self_attn.k_norm.weight": [d]})
            if config.get("attention_bias", False):
                entries.update({"self_attn.q_proj.bias": [q], "self_attn.k_proj.bias": [kv * d],
                                "self_attn.v_proj.bias": [kv * d]})
        else:
            assert kind == "qwen3_5_text" and layer_type == "linear_attention", "unknown layer type"
            nk, nv, dk, dv, kernel = (int(config[k]) for k in
                ("linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim",
                 "linear_value_head_dim", "linear_conv_kernel_dim"))
            width = 2 * nk * dk + nv * dv
            entries.update({"linear_attn.A_log": [nv], "linear_attn.dt_bias": [nv],
                            "linear_attn.norm.weight": [dv],
                            "linear_attn.conv1d.weight": [width, 1, kernel],
                            "linear_attn.in_proj_qkv.weight": [width, h],
                            "linear_attn.in_proj_z.weight": [nv * dv, h],
                            "linear_attn.in_proj_a.weight": [nv, h],
                            "linear_attn.in_proj_b.weight": [nv, h],
                            "linear_attn.out_proj.weight": [h, nv * dv]})
        shapes.update({base + name: shape for name, shape in entries.items()})
    aliases = []
    if role == "critic":
        shapes["value_head.weight"] = [1, h]
    elif config.get("tie_word_embeddings", False):
        aliases = [["lm_head.weight", "model.embed_tokens.weight"]]
    else:
        shapes["lm_head.weight"] = [v, h]
    return {"shapes": dict(sorted(shapes.items())), "aliases": aliases,
            "numel": sum(int(np.prod(shape)) for shape in shapes.values()), "dtype": "torch.float32"}


def prepare_fp32_model(module, move):
    """Enforce actual floating state after loading, before device placement.

    In Transformers 5.9 AutoConfig can consume dtype=float32 before AutoModel
    selects a composite model's still-BF16 text config. A loader keyword is not
    proof of the loaded dtype. Public Module.float() converts parameters and
    floating buffers; observe_parameters subsequently rejects any missed cast.
    """
    return move(module.float())


def observe_parameters(module, config, role, torch, assert_device):
    expected = parameter_spec(config, role)
    actual = dict(module.named_parameters())
    assert {name: list(p.shape) for name, p in actual.items()} == expected["shapes"], \
        "parameter names/shapes differ from independent architecture: " + role
    identities = {}
    all_named = list(module.named_parameters(remove_duplicate=False))
    expected_all = set(expected["shapes"]).union(name for group in expected["aliases"] for name in group)
    assert len(all_named) == len(expected_all) and {name for name, _ in all_named} == expected_all, \
        "incomplete non-deduplicated parameter enumeration: " + role
    for name, parameter in all_named:
        identities.setdefault(id(parameter), []).append(name)
        assert parameter.dtype == torch.float32, "non-FP32 parameter: " + role + "/" + name
        assert_device(parameter)
    aliases = sorted(sorted(names) for names in identities.values() if len(names) > 1)
    assert aliases == expected["aliases"], "parameter aliases differ: " + role
    assert sum(int(p.numel()) for p in actual.values()) == expected["numel"]
    return expected


def validate_loading_info(info, config, role):
    """Permit only deliberately excluded vision/MTP or unused causal LM head.

    Never accept missing text weights (including a claimed tied LM head), shape
    mismatches, or an unmatched language_model prefix. HF owns key conversion.
    """
    assert isinstance(info, dict) and all(key in info for key in
        ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")), "missing loading diagnostics"
    result = {key: sorted(info[key]) for key in
              ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")}
    assert not result["missing_keys"], "missing pretrained text weights: " + repr(result)
    assert not result["mismatched_keys"] and not result["error_msgs"], "pretrained load error"
    hybrid = config["model_type"] == "qwen3_5_text"
    for name in result["unexpected_keys"]:
        excluded = hybrid and name.startswith(("visual.", "model.visual.", "mtp.", "model.mtp."))
        assert excluded or (role == "critic" and name == "lm_head.weight"), \
            "unexpected pretrained text weight: " + name
    return result


def embedding_input(module, ids):
    """Retain a real non-leaf input without detaching its parameter gradient."""
    embeddings = module.get_input_embeddings()(ids)
    assert embeddings.requires_grad, "embedding input disconnected from trainable parameters"
    embeddings.retain_grad()
    return embeddings


def observe_qwen35_adapter(config, framework):
    required = config["model_type"] == "qwen3_5_text"
    result = {"required": required, "active": False, "bindings": []}
    if required:
        assert importlib.metadata.version("transformers") == "5.9.0", "unverified Qwen3.5 version"
        from transformers.models.qwen3_5 import modeling_qwen3_5
        active = []
        for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
            function = getattr(modeling_qwen3_5, name)
            active.append(getattr(function, "_jittor_transformers_qwen35_functional", False) is True)
            result["bindings"].append({"name": name, "module": function.__module__,
                "qualname": function.__qualname__,
                "source_sha256": hashlib.sha256(inspect.getsource(function).encode()).hexdigest()})
        assert all(active) if framework == "jittor" else not any(active), \
            "incorrect Qwen3.5 functional adapter activation"
        result["active"] = all(active)
    return result


def observe_qwen35_instances(model, config, role, framework):
    """Observe, never overwrite, the callable selected by every actual layer.

    HF captures fused-or-fallback functions on each instance at construction;
    patching a module-level fallback alone does not prove it is selected.
    This declared profile excludes FLA fused replacements, even if they carry
    a copied adapter marker. Identity is checked before collecting source facts.
    """
    assert config["model_type"] == "qwen3_5_text" and role in ("actor", "critic")
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
    prefix = "model" if role == "actor" else "backbone"
    expected = ["{}.layers.{}.linear_attn".format(prefix, index)
                for index, kind in enumerate(config["layer_types"]) if kind == "linear_attention"]
    assert len(config["layer_types"]) == config["num_hidden_layers"]
    actual = {name: module for name, module in model.named_modules()
              if isinstance(module, hf.Qwen3_5GatedDeltaNet)}
    assert set(actual) == set(expected), "Qwen3.5 instance layer coverage differs: " + role
    layers = []
    for path in expected:
        layer = actual[path]
        assert type(layer) is hf.Qwen3_5GatedDeltaNet, "unverified Qwen3.5 instance subclass"
        bindings = []
        for attribute, target in (("chunk_gated_delta_rule", "torch_chunk_gated_delta_rule"),
                                  ("recurrent_gated_delta_rule", "torch_recurrent_gated_delta_rule")):
            function = getattr(layer, attribute)
            assert function is getattr(hf, target), "actual Qwen3.5 instance binding differs: " + path + "." + attribute
            active = getattr(function, "_jittor_transformers_qwen35_functional", False) is True
            assert active is (framework == "jittor"), "actual Qwen3.5 instance adapter marker differs"
            bindings.append({"attribute": attribute, "target": target, "identity_matches": True,
                "functional_adapter": active, "module": function.__module__, "qualname": function.__qualname__,
                "source_sha256": hashlib.sha256(inspect.getsource(function).encode()).hexdigest()})
        layers.append({"path": path, "class_module": type(layer).__module__,
                       "class_name": type(layer).__qualname__, "bindings": bindings})
    return {"expected_layers": len(expected), "layers": layers}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=("torch", "jittor"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "npu"), default="cpu")
    parser.add_argument("--model", help="local pretrained model; default: tiny Qwen3")
    parser.add_argument("--algorithm", choices=("ppo", "grpo", "rlhf", "rlvr"), default="ppo")
    parser.add_argument("--critic", choices=("zero", "learned"), default="zero")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights-in", type=Path)
    parser.add_argument("--weights-out", type=Path)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--phase", choices=("full", "baseline", "resume"), default="full")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-after", type=int, default=2)
    args = parser.parse_args()
    if args.steps < 1 or args.lr <= 0:
        parser.error("steps and lr must be positive")
    if args.critic == "learned" and args.algorithm not in ("ppo", "rlhf"):
        parser.error("a learned critic is only used by the PPO/RLHF fixtures")
    if args.algorithm == "rlhf" and args.critic != "learned":
        parser.error("the RLHF fixture requires --critic learned")
    if args.output.suffix != ".npz":
        parser.error("--output must end with .npz")
    for path in (args.weights_in, args.weights_out):
        if path is not None and path.suffix != ".npz":
            parser.error("weight paths must end with .npz")
    if args.framework == "jittor" and args.weights_in is None:
        parser.error("Jittor requires --weights-in from the independent oracle")
    if args.phase == "full":
        if args.checkpoint is not None:
            parser.error("--checkpoint is only valid for baseline/resume phases")
    else:
        if args.algorithm != "ppo" or args.critic != "learned":
            parser.error("checkpoint phases require --algorithm ppo --critic learned")
        if args.checkpoint is None or args.checkpoint.suffix != ".npz":
            parser.error("checkpoint phases require --checkpoint ending in .npz")
        if not 0 < args.checkpoint_after < args.steps:
            parser.error("checkpoint-after must be inside the requested trajectory")
        checkpoint_manifest = args.checkpoint.with_suffix(".json")
        if args.phase == "baseline" and (
                args.checkpoint.exists() or checkpoint_manifest.exists()):
            parser.error("baseline checkpoint and manifest paths must be fresh")
        if args.phase == "resume" and not (
                args.checkpoint.is_file() and checkpoint_manifest.is_file()):
            parser.error("resume checkpoint or manifest does not exist")
    return args


def cuda_runtime_conditions(torch):
    """Read the effective public precision policy, not just requested switches."""
    conditions = {
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    assert all(conditions[name] is False for name in
               ("cuda_matmul_allow_tf32", "cudnn_allow_tf32", "cudnn_benchmark")), \
        "CUDA precision policy changed or was not applied"
    assert conditions["float32_matmul_precision"] == "highest", \
        "CUDA precision policy changed or was not applied"
    return conditions


def initialize(args, stack):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    jt = None
    if args.framework == "jittor":
        os.environ["JITTOR_TORCH_SHIM"] = "1"
        import jittor as jt
        import torch
        from jittor.compat.shim.runtime import activation_status
        from jittor.compat.torch.publication import namespace_owner
        from jittor._runtime.fallback import forbid_backend_fallbacks

        assert namespace_owner(torch) is jt and activation_status(jt).active
        assert torch.Tensor is not jt.Var
        if args.device == "npu":
            assert jt.compiler.has_acl, "ACL unavailable"
            stack.enter_context(jt.runtime.scope(use_cuda=1, use_acl=1))
        elif args.device == "cuda":
            assert jt.compiler.has_cuda, "Jittor CUDA unavailable"
            assert not any(getattr(jt.compiler, name, False) for name in
                           ("has_acl", "has_rocm", "has_corex")), "build is not native CUDA"
            # use_acl/use_rocm/use_corex alias use_cuda, so setting those to
            # zero here would disable device execution rather than select CUDA.
            stack.enter_context(jt.runtime.scope(use_cuda=1))
        else:
            stack.enter_context(jt.runtime.scope(use_cuda=0))
        stack.enter_context(jt.runtime.scope(backend_fallback="error"))
        stack.enter_context(forbid_backend_fallbacks())
    else:
        os.environ.pop("JITTOR_TORCH_SHIM", None)
        import torch

        assert hasattr(torch, "_C"), "oracle must be official PyTorch"
        assert not hasattr(torch, "_torch_compat_install_context"), "oracle is shim"
        if args.device == "npu":
            import torch_npu  # noqa: F401

            assert torch.npu.is_available(), "native torch_npu unavailable"
        elif args.device == "cuda":
            assert torch.cuda.is_available(), "official PyTorch CUDA unavailable"

    if args.device == "cuda":
        # These frontend-owned switches configure both runtimes' actual math.
        # Read them back immediately and again in the final artifact to reject
        # no-op configuration or a dependency changing policy during training.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision("highest")
        cuda_runtime_conditions(torch)

    def move(value):
        if args.device == "cpu":
            return value
        return value.cuda() if jt is not None else value.to(args.device)

    def synchronize():
        if jt is not None:
            jt.sync_all(args.device != "cpu")
        elif args.device == "npu":
            torch.npu.synchronize()
        elif args.device == "cuda":
            torch.cuda.synchronize()

    def assert_device(value):
        if jt is not None:
            assert bool(jt.flags.use_cuda) == (args.device != "cpu"), \
                "Jittor device dispatch changed during the run"
            location = getattr(value, "_var", value).location()
            expected = "device" if args.device != "cpu" else "cpu"
            assert location == expected, (location, expected)
        else:
            location = str(value.device)
            assert value.device.type == args.device, (location, args.device)
        return str(location)

    return torch, jt, move, synchronize, assert_device


def block_unavailable_cpu_optionals(args):
    """Hide explicitly named, unusable accelerator-only packages on CPU.

    The shared dependency environment may contain ``torch_npu`` metadata while
    its CANN libraries are intentionally absent.  Transformers treats metadata
    presence as availability and imports the broken package even for a CPU
    model.  This opt-in filter must run before Transformers is imported and is
    recorded in the artifact; it is forbidden for device acceptance.
    """
    raw = os.environ.get("VERL_CPU_BLOCK_OPTIONAL_PACKAGES", "")
    blocked = frozenset(item.strip() for item in raw.split(",") if item.strip())
    if not blocked:
        return []
    if args.device != "cpu":
        raise RuntimeError("optional-package blocking is only valid for CPU runs")
    original = importlib.util.find_spec

    def filtered(name, package=None):
        if name.split(".", 1)[0] in blocked:
            return None
        return original(name, package)

    importlib.util.find_spec = filtered
    return sorted(blocked)


def snapshot(value):
    return np.array(value.detach().cpu().numpy(), copy=True)


def save_npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def main():
    args = parse_args()
    with ExitStack() as stack:
        run(args, stack)


def run(args, stack):
    torch, jt, move, synchronize, assert_device = initialize(args, stack)
    fallback_before = jt.core.backend_fallback_count() if jt is not None else None
    torch.manual_seed(args.seed)
    blocked_optional_packages = block_unavailable_cpu_optionals(args)
    from transformers import AutoModel, AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM, Qwen3Model
    if importlib.metadata.version("transformers") != "5.9.0":
        raise RuntimeError("Schema 4 architecture and loading evidence require Transformers 5.9.0")

    loading_info = []
    if args.model:
        def make_policy():
            policy, info = AutoModelForCausalLM.from_pretrained(
                args.model, local_files_only=True, trust_remote_code=False,
                dtype=torch.float32, attn_implementation="eager", output_loading_info=True,
            )
            loading_info.append({"role": "actor", "info": validate_loading_info(
                info, policy.config.get_text_config().to_dict(), "actor")})
            return policy
        model = make_policy()
    else:
        config = Qwen3Config(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            head_dim=8, max_position_embeddings=32, attention_dropout=0.0,
            tie_word_embeddings=False, bos_token_id=1, eos_token_id=None,
            pad_token_id=0,
        )
        config._attn_implementation = "eager"
        def make_policy():
            return Qwen3ForCausalLM(config)
        model = make_policy()
    # Explicit module placement: enabling a native accelerator does not change
    # torch's default factory device, including after the upstream default-CPU fix.
    model = prepare_fp32_model(model, move)
    text_config = model.config.get_text_config()
    architecture = text_config.to_dict()
    adapter_evidence = observe_qwen35_adapter(architecture, args.framework)
    class ValueModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            if args.model:
                self.backbone, info = AutoModel.from_pretrained(
                    args.model, local_files_only=True, trust_remote_code=False,
                    config=copy.deepcopy(text_config), dtype=torch.float32,
                    attn_implementation="eager", output_loading_info=True,
                )
                loading_info.append({"role": "critic", "info": validate_loading_info(
                    info, architecture, "critic")})
            else:
                self.backbone = Qwen3Model(config)
            self.value_head = torch.nn.Linear(model.config.hidden_size, 1, bias=False)

        def forward(self, ids, scale, inputs_embeds=None):
            hidden = self.backbone(
                input_ids=ids if inputs_embeds is None else None,
                inputs_embeds=inputs_embeds, attention_mask=torch.ones_like(ids), use_cache=False,
            ).last_hidden_state
            return self.value_head(hidden * scale).squeeze(-1)

    critic = None
    if args.critic == "learned":
        critic = prepare_fp32_model(ValueModel(), move)
        critic.eval()
    reference = None
    reward_model = None
    if args.algorithm == "rlhf":
        reference = prepare_fp32_model(make_policy(), move)
        # RLHF starts from a frozen copy of the same SFT policy. Later actor
        # updates make the KL term nonzero while the reference remains fixed.
        reference.load_state_dict(model.state_dict())
        reference.eval()
        reward_model = prepare_fp32_model(ValueModel(), move)
        reward_model.eval()
        for frozen in (reference, reward_model):
            for parameter in frozen.parameters():
                parameter.requires_grad_(False)
    # Eval disables dropout, not autograd. The update remains a real backward
    # and SGD step, while generation and old/new probability paths stay aligned.
    model.eval()
    model.generation_config.eos_token_id = None
    model.generation_config.pad_token_id = 0
    synchronize()
    parameter_contract = {"actor": observe_parameters(model, architecture, "actor", torch, assert_device)}
    if critic is not None:
        parameter_contract["critic"] = observe_parameters(critic, architecture, "critic", torch, assert_device)
    for auxiliary, role in ((reference, "actor"), (reward_model, "critic")):
        if auxiliary is not None:
            observe_parameters(auxiliary, architecture, role, torch, assert_device)
    if adapter_evidence["required"]:
        adapter_evidence["instance_profile"] = "hf_torch_gated_delta_instances_v1"
        adapter_evidence["instances"] = {
            name: observe_qwen35_instances(module, architecture, role, args.framework)
            for name, module, role in (("actor", model, "actor"), ("critic", critic, "critic"),
                                      ("reference", reference, "actor"), ("reward", reward_model, "critic"))
            if module is not None
        }
    transferable = dict(list(model.named_parameters()) + list(model.named_buffers()))
    if critic is not None:
        transferable.update({"critic/" + name: value for name, value in
                             list(critic.named_parameters()) + list(critic.named_buffers())})
    if reference is not None:
        transferable.update({"reference/" + name: value for name, value in
                             list(reference.named_parameters()) + list(reference.named_buffers())})
    if reward_model is not None:
        transferable.update({"reward/" + name: value for name, value in
                             list(reward_model.named_parameters()) + list(reward_model.named_buffers())})
    if args.weights_in:
        with np.load(args.weights_in, allow_pickle=False) as weights:
            if set(weights.files) != set(transferable):
                raise ValueError("oracle/candidate parameter or buffer keys differ")
            with torch.no_grad():
                for name, value in transferable.items():
                    if tuple(value.shape) != weights[name].shape:
                        raise ValueError("weight shape mismatch: " + name)
                    value.copy_(move(torch.from_numpy(weights[name])).to(value.dtype))
    synchronize()
    if args.weights_out:
        save_npz(args.weights_out, {name: snapshot(v) for name, v in transferable.items()})
    initial_weight_path = args.weights_in or args.weights_out
    initial_weight_sha256 = (None if initial_weight_path is None else
                            hashlib.sha256(initial_weight_path.read_bytes()).hexdigest())

    checkpoint_metadata = None
    checkpoint_sha256 = None
    start_step = 0
    if args.phase == "resume":
        checkpoint_metadata = json.loads(args.checkpoint.with_suffix(".json").read_text())
        expected = {
            "schema_version": 1,
            "framework": args.framework,
            "device": args.device,
            "algorithm": "ppo",
            "critic": "learned",
            "next_step": args.checkpoint_after,
            "total_steps": args.steps,
            "initial_weight_sha256": initial_weight_sha256,
            "state_names": sorted(transferable),
        }
        for key, value in expected.items():
            if checkpoint_metadata.get(key) != value:
                raise ValueError("checkpoint contract changed: " + key)
        checkpoint_sha256 = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        if checkpoint_metadata.get("checkpoint_sha256") != checkpoint_sha256:
            raise ValueError("checkpoint digest mismatch")
        with np.load(args.checkpoint, allow_pickle=False) as state:
            if set(state.files) != set(transferable):
                raise ValueError("checkpoint state keys differ")
            with torch.no_grad():
                for name, value in transferable.items():
                    if tuple(value.shape) != state[name].shape:
                        raise ValueError("checkpoint state shape mismatch: " + name)
                    value.copy_(move(torch.from_numpy(state[name])).to(value.dtype))
        synchronize()
        start_step = args.checkpoint_after

    # Load optional shared pure-Python dependencies only after Transformers has
    # materialized the model, to avoid shim optional packages polluting oracle.
    extra_site = os.environ.get("VERL_EXTRA_SITE_PACKAGES")
    if extra_site and extra_site not in sys.path:
        sys.path.append(extra_site)
    import verl
    from verl.trainer.ppo.core_algos import (
        compute_gae_advantage_return, compute_grpo_outcome_advantage,
        compute_policy_loss, compute_rewards, compute_value_loss,
    )

    if verl.__version__ != "0.9.1":
        raise RuntimeError("This fixture is pinned to verl 0.9.1")
    parameters = dict(model.named_parameters())
    if any(not p.requires_grad for p in parameters.values()):
        raise RuntimeError("fixture requires an unfrozen model")
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    critic_parameters = {} if critic is None else dict(critic.named_parameters())
    reference_parameters = {} if reference is None else dict(reference.named_parameters())
    reward_parameters = {} if reward_model is None else dict(reward_model.named_parameters())
    critic_optimizer = (None if critic is None else
                        torch.optim.SGD(critic.parameters(), lr=args.lr))
    prompt_ids = move(torch.tensor(
        [[1, 2, 3, 4], [1, 2, 3, 5], [1, 6, 7, 4], [1, 6, 7, 5]],
        dtype=torch.long,
    ))
    prompt_mask = torch.ones_like(prompt_ids)
    group_ids = np.array(["a", "a", "b", "b"])
    new_tokens = 4
    prompt_length = int(prompt_ids.shape[1])
    arrays = {}
    summaries = []

    def save(name, value):
        # A retained lazy diagnostic (for example a clip metric) may not have
        # been materialized by backward. Execute THIS tensor before checking
        # residency; never infer accelerator execution from the global flag.
        if jt is not None:
            value.sync()
        synchronize()
        try:
            assert_device(value)
        except AssertionError as error:
            raise AssertionError("snapshot device mismatch: " + name) from error
        result = snapshot(value)
        if not np.isfinite(result).all():
            raise AssertionError("nonfinite tensor: " + name)
        arrays[name] = result
        return result

    def log_probabilities(ids, scale, policy=None, inputs_embeds=None):
        policy = model if policy is None else policy
        logits = policy(input_ids=ids if inputs_embeds is None else None,
                        inputs_embeds=inputs_embeds, attention_mask=torch.ones_like(ids), use_cache=False).logits
        response_logits = logits[:, prompt_length - 1:-1, :] * scale
        log_probs = torch.log_softmax(response_logits, dim=-1)
        selected = torch.gather(log_probs, -1, ids[:, prompt_length:].unsqueeze(-1)).squeeze(-1)
        return response_logits, selected

    executed_steps = list(range(start_step, args.steps))
    for step in executed_steps:
        prefix = "step_{:03d}/".format(step)
        optimizer.zero_grad(set_to_none=True)
        before = {name: snapshot(p) for name, p in parameters.items()}
        critic_before = {name: snapshot(p) for name, p in critic_parameters.items()}
        if critic_optimizer is not None:
            critic_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            ids = model.generate(
                input_ids=prompt_ids, attention_mask=prompt_mask,
                do_sample=False, max_new_tokens=new_tokens,
                min_new_tokens=new_tokens, use_cache=False,
                eos_token_id=None, pad_token_id=0,
            )
            assert tuple(ids.shape) == (4, prompt_length + new_tokens)
            response = ids[:, prompt_length:]
            response_mask = torch.ones_like(response, dtype=torch.float32)
            # Explicit synthetic PPO/GRPO score: 0.01 per generated token, +1
            # on token 7, plus 0/0.25 row target at the final token. All reward
            # math stays on the selected device. The target ensures nonzero
            # group variance; it is not a task-quality metric.
            row_offset = move(torch.tensor([[0.0], [0.25], [0.0], [0.25]]))
            terminal = move(torch.tensor([[0.0, 0.0, 0.0, 1.0]]))
            rule_scores = response_mask * 0.01 + (response == 7).float() + row_offset * terminal
            values = (torch.zeros_like(response_mask) if critic is None else
                      critic(ids, 1.0)[:, prompt_length - 1:-1].detach())
            _, old_log_prob = log_probabilities(ids, 1.0)
            old_log_prob = old_log_prob.detach()
            ref_log_prob = None
            reward_model_scores = None
            verifier_scores = None
            if args.algorithm == "rlhf":
                reward_model_scores = reward_model(ids, 1.0)[:, -1:].detach()
                token_level_scores = reward_model_scores * terminal
                _, ref_log_prob = log_probabilities(ids, 1.0, policy=reference)
                ref_log_prob = ref_log_prob.detach()
                rewards = compute_rewards(
                    token_level_scores, old_log_prob, ref_log_prob, kl_ratio=0.1,
                )
            elif args.algorithm == "rlvr":
                # A deterministic graded verifier: the prompt's final token is
                # the target and the response's final token is the answer. The
                # inverse-distance score is nonconstant without framework RNG.
                verifier_scores = 1.0 / (
                    1.0 + torch.abs(response[:, -1:].float() - prompt_ids[:, -1:].float())
                )
                token_level_scores = verifier_scores * terminal
                rewards = token_level_scores
            else:
                token_level_scores = rule_scores
                rewards = token_level_scores
            if args.algorithm in ("ppo", "rlhf"):
                advantages, returns = compute_gae_advantage_return(
                    token_level_rewards=rewards, values=values,
                    response_mask=response_mask, gamma=0.99, lam=0.95,
                )
            else:
                advantages, returns = compute_grpo_outcome_advantage(
                    token_level_rewards=rewards, response_mask=response_mask,
                    index=group_ids,
                )
        input_scale = move(torch.tensor(1.0)).requires_grad_(True)
        actor_embeddings = embedding_input(model, ids)
        logits, new_log_prob = log_probabilities(ids, input_scale, inputs_embeds=actor_embeddings)
        loss, clip_fraction, ppo_kl, lower_clip_fraction = compute_policy_loss(
            old_log_prob, new_log_prob, advantages, response_mask, cliprange=0.2,
        )
        loss.backward()
        synchronize()
        for name, value in (
            ("input_ids", ids), ("response", response),
            ("token_level_scores", token_level_scores), ("rewards", rewards),
            ("response_mask", response_mask), ("values", values),
            ("advantages", advantages), ("returns", returns),
            ("old_log_prob", old_log_prob), ("log_prob", new_log_prob),
            ("logits", logits), ("loss", loss), ("clip_fraction", clip_fraction),
            ("ppo_kl", ppo_kl), ("lower_clip_fraction", lower_clip_fraction),
            ("input_scale", input_scale),
        ):
            save(prefix + name, value)
        for name, value in (("ref_log_prob", ref_log_prob),
                            ("reward_model_scores", reward_model_scores),
                            ("verifier_scores", verifier_scores)):
            if value is not None:
                save(prefix + name, value)
        if input_scale.grad is None:
            raise AssertionError("differentiable input has no gradient")
        save(prefix + "input_grad/input_scale", input_scale.grad)
        if actor_embeddings.grad is None:
            raise AssertionError("retained actor embedding input has no gradient")
        save(prefix + "input/embeddings", actor_embeddings)
        save(prefix + "input_grad/embeddings", actor_embeddings.grad)
        missing = [name for name, p in parameters.items() if p.grad is None]
        if missing:
            raise AssertionError("missing parameter gradients: " + repr(missing))
        for name, parameter in parameters.items():
            save(prefix + "param_grad/" + name, parameter.grad)
        critic_update = None
        if critic is not None:
            critic_scale = move(torch.tensor(1.0)).requires_grad_(True)
            critic_embeddings = embedding_input(critic.backbone, ids)
            predicted_values = critic(ids, critic_scale, inputs_embeds=critic_embeddings)[:, prompt_length - 1:-1]
            value_loss, value_clip_fraction = compute_value_loss(
                vpreds=predicted_values, returns=returns, values=values,
                response_mask=response_mask, cliprange_value=0.2,
            )
            value_loss.backward()
            synchronize()
            for name, value in (("value_predictions", predicted_values),
                                ("value_loss", value_loss),
                                ("value_clip_fraction", value_clip_fraction)):
                save(prefix + name, value)
            if critic_scale.grad is None:
                raise AssertionError("critic differentiable input has no gradient")
            save(prefix + "critic_input_grad/input_scale", critic_scale.grad)
            if critic_embeddings.grad is None:
                raise AssertionError("retained critic embedding input has no gradient")
            save(prefix + "critic_input/embeddings", critic_embeddings)
            save(prefix + "critic_input_grad/embeddings", critic_embeddings.grad)
            for name, parameter in critic_parameters.items():
                if parameter.grad is None:
                    raise AssertionError("missing critic gradient: " + name)
                save(prefix + "critic_param_grad/" + name, parameter.grad)
            if not any(np.any(arrays[prefix + "critic_param_grad/" + name])
                       for name in critic_parameters):
                raise AssertionError("all critic gradients are zero")
            critic_optimizer.step()
            synchronize()
            critic_update = 0.0
            for name, parameter in critic_parameters.items():
                updated = save(prefix + "critic_post_weight/" + name, parameter)
                critic_update = max(critic_update, float(np.max(np.abs(updated - critic_before[name]))))
            if critic_update == 0.0:
                raise AssertionError("SGD changed no critic parameter")
            with torch.no_grad():
                updated_values = critic(ids, 1.0)[:, prompt_length - 1:-1]
                updated_value_loss = compute_value_loss(
                    vpreds=updated_values, returns=returns, values=values,
                    response_mask=response_mask, cliprange_value=0.2,
                )[0]
            save(prefix + "post_update_values", updated_values)
            save(prefix + "post_update_value_loss", updated_value_loss)
        optimizer.step()
        synchronize()
        largest_update = 0.0
        for name, parameter in parameters.items():
            updated = save(prefix + "post_weight/" + name, parameter)
            largest_update = max(largest_update, float(np.max(np.abs(updated - before[name]))))
        if largest_update == 0.0:
            raise AssertionError("SGD changed no parameter")
        # At the start of an on-policy update ratio=1 and centered advantages
        # make the scalar loss near zero. Evaluate the updated model on the
        # SAME rollout to expose the actual policy change without interpreting
        # that near-zero pre-update loss as a lack of training.
        with torch.no_grad():
            _, post_log_prob = log_probabilities(ids, 1.0)
            post_loss = compute_policy_loss(
                old_log_prob, post_log_prob, advantages, response_mask, cliprange=0.2,
            )[0]
        synchronize()
        save(prefix + "post_update_log_prob", post_log_prob)
        save(prefix + "post_update_loss", post_loss)
        summaries.append({
            "step": step, "loss": float(arrays[prefix + "loss"].reshape(-1)[0]),
            "post_update_loss": float(arrays[prefix + "post_update_loss"].reshape(-1)[0]),
            "largest_parameter_update": largest_update,
            "gradient_count": len(parameters),
            "critic_gradient_count": len(critic_parameters),
            "largest_critic_update": critic_update,
        })
        if critic is not None:
            summaries[-1].update({
                "value_loss": float(arrays[prefix + "value_loss"].reshape(-1)[0]),
                "post_update_value_loss": float(arrays[prefix + "post_update_value_loss"].reshape(-1)[0]),
            })
        if args.phase == "baseline" and step + 1 == args.checkpoint_after:
            state = {name: snapshot(value) for name, value in transferable.items()}
            save_npz(args.checkpoint, state)
            checkpoint_sha256 = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
            checkpoint_metadata = {
                "schema_version": 1,
                "framework": args.framework,
                "device": args.device,
                "algorithm": args.algorithm,
                "critic": args.critic,
                "next_step": step + 1,
                "total_steps": args.steps,
                "initial_weight_sha256": initial_weight_sha256,
                "state_names": sorted(transferable),
                "optimizer": "SGD(lr={}, momentum=0, weight_decay=0)".format(args.lr),
                "optimizer_state": "empty",
                "checkpoint_sha256": checkpoint_sha256,
            }
            args.checkpoint.with_suffix(".json").write_text(
                json.dumps(checkpoint_metadata, indent=2) + "\n"
            )
        print(json.dumps(summaries[-1]), flush=True)

    synchronize()
    fallback_count = None if jt is None else int(jt.core.backend_fallback_count() - fallback_before)
    if fallback_count is not None and fallback_count != 0:
        raise AssertionError("backend fallback occurred")
    manifest = {
        "scope": "single-process synthetic rollout/reward/advantage/SGD integration",
        "schema_version": 4,
        "parameter_contract": parameter_contract,
        "text_only": True,
        "loading_info": loading_info,
        "qwen35_adapter": adapter_evidence,
        "script_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ("probe_training_workflow.py", "compare_training_workflow.py")},
        "initial_weight_sha256": initial_weight_sha256,
        "framework": args.framework, "device": args.device,
        "algorithm": args.algorithm, "steps": args.steps, "seed": args.seed,
        "phase": args.phase, "start_step": start_step, "executed_steps": executed_steps,
        "process_id": os.getpid(),
        "checkpoint_after": args.checkpoint_after if args.phase != "full" else None,
        "checkpoint_sha256": checkpoint_sha256,
        "critic": args.critic,
        "critic_parameter_count": sum(int(p.numel()) for p in critic_parameters.values()),
        "reference_parameter_count": sum(int(p.numel()) for p in reference_parameters.values()),
        "reward_parameter_count": sum(int(p.numel()) for p in reward_parameters.values()),
        "parameter_names": sorted(parameters),
        "critic_parameter_names": sorted(critic_parameters),
        "reference_parameter_names": sorted(reference_parameters),
        "reward_parameter_names": sorted(reward_parameters),
        "lr": args.lr, "dtype": "float32", "attention": "eager",
        "model": "local-pretrained" if args.model else "tiny-qwen3",
        "model_config": model.config.to_dict(),
        "versions": {name: importlib.metadata.version(name) for name in ("transformers", "verl")},
        "torch_api_version": getattr(torch, "__torch_version__", torch.__version__),
        "blocked_optional_packages": blocked_optional_packages,
        "parameter_count": sum(int(p.numel()) for p in parameters.values()),
        "parameter_location": assert_device(next(iter(parameters.values()))),
        "fallback_count": fallback_count,
        "input_gradients": ["input_scale", "embeddings"],
        "critic_input_gradients": [] if critic is None else ["input_scale", "embeddings"],
        "embedding_input_shape": [4, prompt_length + new_tokens, int(text_config.hidden_size)],
        "nondifferentiable_inputs": ["integer input_ids", "fixed rewards and advantages"],
        "reward_source": {"ppo": "synthetic_rule", "grpo": "synthetic_rule",
                          "rlhf": "independent_transformer_reward_model_plus_reference_kl",
                          "rlvr": "deterministic_inverse_token_distance_verifier"}[args.algorithm],
        "reward_rule": ("reward_model_final_score - 0.1*(old_log_prob-ref_log_prob)"
                        if args.algorithm == "rlhf" else
                        "inverse_distance(response_final,prompt_target)"
                        if args.algorithm == "rlvr" else
                        "0.01/token + token_id_7_hit + final_token_offset_[0,0.25,0,0.25]"),
        "reference_policy": reference is not None,
        "kl_ratio": 0.1 if args.algorithm == "rlhf" else None,
        "grpo_group_ids": group_ids.tolist(),
        "limitations": ["synthetic prompt-pair groups", "no human preference data",
                        "no Ray/FSDP trainer", "no distributed validation",
                        "not a performance or task-quality measurement"],
        "steps_summary": summaries,
    }
    if critic is None:
        manifest["limitations"].append("no learned critic")
    if args.device == "cuda":
        manifest["runtime_conditions"] = cuda_runtime_conditions(torch)
    if reward_model is None:
        manifest["limitations"].append("no learned reward model")
    else:
        manifest["limitations"].append("reward model is initialized fixture, not preference-trained")
    if args.algorithm == "rlvr":
        manifest["limitations"].append("verifier is deterministic execution fixture, not benchmark judge")
    if args.phase == "full":
        manifest["limitations"].append("checkpoint continuation not requested")
    else:
        manifest["limitations"].append(
            "runtime-local model+critic checkpoint; SGD has no momentum state"
        )
    save_npz(args.output, arrays)
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "tensor_count": len(arrays),
                      "fallback_count": fallback_count}), flush=True)


if __name__ == "__main__":
    main()
