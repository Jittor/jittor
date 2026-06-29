"""monkeypatch_ops — runtime monkeypatches that let external TRAINING libraries
(transformers / peft / trl / ms-swift) run on the jittor torch-shim.

WHAT THIS FILE IS (and isn't)
-----------------------------
Every function here patches the Python logic of an EXTERNAL library, NOT jittor.
They are NOT jittor operators or torch-compat shims — those (GradScaler, F.pad,
masked_scatter, in-place autograd, buffer/requires_grad semantics, conv list-kernel,
repeat_interleave, attention/SDPA, …) live by FUNCTION in jittor's own files
(`torch_compat.py`, `nn.py`, `attention.py`, `misc.py`, `__init__.py`,
`src/misc/cuda_atomic.h`). This file is the single, clearly-labelled home for the
irreducible glue that adapts a third-party training stack — it exists so those
workarounds are consolidated and documented in ONE place instead of scattered
per-library files.

WHY THEY LIVE IN JITTOR (a deliberate, relaxed constraint)
----------------------------------------------------------
The original project rule was "torch-external deps must NOT appear in the jittor
repo". By owner's decision that rule is relaxed FOR THIS FILE ONLY: the dep-specific
monkeypatches are collected here, in jittor, with clear comments — accepting that
the jittor repo now references transformers/peft/ms-swift symbols. It does NOT make
jittor DEPEND on them: every patch guards its own imports with try/except and is
version/architecture-scoped, so:
  * importing `jittor` (or this module) without those libraries is a no-op, and
  * running under REAL torch (the patches are env-symmetric where they fix
    transformers-version drift) is also safe.

USAGE — OPT-IN ONLY
-------------------
This is NOT applied automatically on `import jittor` (that would wrongly patch
transformers/peft for non-training jittor users — vLLM inference, mmdetection, etc.).
Call it explicitly, AFTER `import torch` (the shim) and BEFORE building the trainer:

    import torch                      # the jittor torch-shim
    import jittor.monkeypatch_ops as mp
    mp.apply()                        # idempotent
    from swift.pipelines import sft_main
    ...

`apply()` is idempotent and patches whatever of {transformers, peft, ms-swift} is
importable; absent libraries are silently skipped.

PATCH INVENTORY (each documented at its function)
-------------------------------------------------
transformers (version / loader / processor compat):
  _patch_transformers_legacy_symbols          re-add removed is_torch_fx_available
  _patch_transformers_default_rope_scaling     restore rope_scaling=None placeholder (MiniCPM)
  _patch_transformers_legacy_tied_weights_keys list-> dict _tied_weights_keys
  _patch_transformers_fused_moe_requires_grad  un-freeze fused MoE experts loaded under no_grad
  _patch_transformers_group_images_by_shape    hashable shape key for the VL fast image processor
peft:
  _patch_peft_disable_adapter                  don't toggle requires_grad in disable_adapter (jittor stop_grad severs graphs)
ms-swift (pipeline gaps):
  _patch_ppo_reward_value_seq_cls              give PPO reward/value models a .score head
  _patch_gkd_native_rollout_engine             attach a TransformersEngine for native (use_vllm=false) GKD rollout
"""

_INSTALLED = False


def apply():
    """Apply all external-library monkeypatches (idempotent, opt-in)."""
    global _INSTALLED
    if _INSTALLED:
        return
    _patch_transformers_legacy_symbols()
    _patch_transformers_legacy_tied_weights_keys()
    _patch_peft_disable_adapter()
    _patch_transformers_fused_moe_requires_grad()
    _patch_transformers_group_images_by_shape()
    _patch_gkd_native_rollout_engine()
    _patch_ppo_reward_value_seq_cls()
    _INSTALLED = True


# Back-compat alias: earlier harnesses / the swift_jittor shim call `install()`.
install = apply


def _patch_transformers_legacy_tied_weights_keys():
    """transformers 5.x changed `_tied_weights_keys` from a LIST of param names to a
    DICT mapping {tied_target: source}. `get_expanded_tied_weights_keys` now does
    `tied_mapping.keys() | tied_mapping.values()` and crashes with
    `AttributeError: 'list' object has no attribute 'keys'` on old trust_remote_code
    models that still ship the list form -- e.g. MiniCPM: `_tied_weights_keys =
    ["lm_head.weight"]` (the output head tied to the input embedding).

    Coerce a legacy LIST `_tied_weights_keys` into the new dict form before the
    original method runs: each listed name (the tie *target*, e.g. lm_head.weight)
    maps to the model's input-embedding weight (the tie *source*). Identical
    transformers behaviour in real torch (the list form is broken there too), so this
    is a transformers-version-compat shim for third-party code.
    """
    try:
        from transformers.modeling_utils import PreTrainedModel
    except Exception:
        return
    if getattr(PreTrainedModel, "_swift_jittor_tied_keys_patched", False):
        return
    orig = PreTrainedModel.get_expanded_tied_weights_keys

    def _input_emb_weight_name(self):
        # Find the parameter name of the input embedding weight (the tie source).
        try:
            emb = self.get_input_embeddings()
        except Exception:
            emb = None
        if emb is not None:
            emb_w = getattr(emb, "weight", None)
            if emb_w is not None:
                for n, p in self.named_parameters(remove_duplicate=False):
                    if p is emb_w:
                        return n
        # Fallback to the conventional name.
        return "model.embed_tokens.weight"

    def get_expanded_tied_weights_keys(self, *args, **kwargs):
        twk = getattr(self, "_tied_weights_keys", None)
        if isinstance(twk, (list, tuple)):
            src = _input_emb_weight_name(self)
            # Don't map the source onto itself; map every other listed target -> source.
            self._tied_weights_keys = {k: src for k in twk if k != src}
        return orig(self, *args, **kwargs)

    PreTrainedModel.get_expanded_tied_weights_keys = get_expanded_tied_weights_keys
    PreTrainedModel._swift_jittor_tied_keys_patched = True


def _patch_transformers_legacy_symbols():
    """Older trust_remote_code modeling files (shipped in the model dir and run via
    `trust_remote_code=True`) import helper symbols that transformers 5.x has since
    removed, so the module import dies before the model is even built:

      - MiniCPM (`modeling_minicpm.py`): `from transformers.utils.import_utils import
        is_torch_fx_available`  -> ImportError (removed in transformers>=5).

    These are pure transformers-version-compat shims (the model code is third-party
    and pinned to its release-time transformers API). Re-add the removed names with
    their original semantics. This is the SAME problem in real torch with transformers
    5.12.1 (the symbol is gone there too); restoring it is required for the custom code
    to import in either env.
    """
    try:
        import transformers.utils.import_utils as iu
    except Exception:
        return
    if not hasattr(iu, "is_torch_fx_available"):
        # Original returns whether torch.fx tracing is usable; True whenever a modern
        # torch (with torch.fx) is importable. Mirror that with a lazy probe.
        def is_torch_fx_available():
            try:
                import torch.fx  # noqa: F401
                return True
            except Exception:
                return False

        iu.is_torch_fx_available = is_torch_fx_available
        # also expose on the transformers.utils package (some code imports it there)
        try:
            import transformers.utils as tu
            if not hasattr(tu, "is_torch_fx_available"):
                tu.is_torch_fx_available = is_torch_fx_available
        except Exception:
            pass

    _patch_transformers_default_rope_scaling()


def _patch_transformers_default_rope_scaling():
    """transformers 5.x normalises a config that originally had `rope_scaling: null`
    into the placeholder dict `{'rope_type': 'default', 'rope_theta': ...}` (meaning
    "no scaling"). Old trust_remote_code modeling files read the LEGACY schema, e.g.
    MiniCPM's `_init_rope` does `self.config.rope_scaling["type"]` whenever
    `rope_scaling is not None` -> `KeyError: 'type'`, because the new placeholder has
    no `'type'` key. The model genuinely uses default (un-scaled) RoPE.

    Fix: when a loaded config carries the no-op default placeholder (rope_type ==
    'default' and NO legacy 'type' key), restore `rope_scaling = None` so the old
    code takes its `if rope_scaling is None:` branch. This is intentionally narrow:
    real scalings (Llama3 `rope_type='llama3'`, InternLM2 which still ships the legacy
    `'type'` key) are left untouched, and for models whose modern code reads
    `rope_type` (Qwen) 'default' and None are equivalent (no scaling). Same transformers
    behaviour in real torch, so applying it in both envs keeps the comparison fair.
    Transformers-version-compat for third-party modeling code.
    """
    try:
        from transformers.configuration_utils import PretrainedConfig
    except Exception:
        return
    if getattr(PretrainedConfig, "_swift_jittor_rope_default_patched", False):
        return
    orig_post_init = PretrainedConfig.__init__

    def __init__(self, *args, **kwargs):
        orig_post_init(self, *args, **kwargs)
        try:
            rs = getattr(self, "rope_scaling", None)
            if isinstance(rs, dict) and "type" not in rs and rs.get("rope_type") == "default":
                # Scope to the legacy custom architectures that read the old schema
                # (MiniCPM*). Models whose modern code understands `rope_type`
                # (Qwen/Llama/etc.) are left exactly as transformers produced them,
                # so none of the already-passing configs change behaviour.
                archs = getattr(self, "architectures", None) or []
                mt = (getattr(self, "model_type", "") or "")
                if any("MiniCPM" in a for a in archs) or mt.startswith("minicpm"):
                    self.rope_scaling = None
        except Exception:
            pass

    PretrainedConfig.__init__ = __init__
    PretrainedConfig._swift_jittor_rope_default_patched = True


def _patch_ppo_reward_value_seq_cls():
    """trl's PPOTrainer scores rollouts and computes the value baseline through a
    regression head: `get_reward` calls `reward_model.score(...)` and the
    PolicyAndValueWrapper calls `value_model.score(...)`. Both the reward AND the
    value model must therefore be sequence-classification models (`*ForSequence
    Classification`, which owns a `.score` Linear with num_labels=1).

    ms-swift's PPO pipeline (`SwiftRLHF._prepare_single_model`) derives the task
    type from the checkpoint via `_get_model_task_type`: a model is loaded as
    seq_cls only if its config/args.json already says so. A plain instruct
    checkpoint (e.g. Qwen2.5-0.5B-Instruct, `architectures=['Qwen2ForCausalLM']`,
    no num_labels) is therefore loaded as a CausalLM with no `.score`, so PPO dies
    with `'Qwen2ForCausalLM' object has no attribute 'score'` during the first
    rollout. (The PPO examples sidestep this by pointing --reward_model at a real
    reward checkpoint that is already seq_cls.)

    The correct behavior — and what trl expects — is for the reward/value models to
    carry a regression head: AutoModelForSequenceClassification with num_labels=1
    initialises a fresh `.score` ("adds a value head"). Force that here for the PPO
    reward and value models when the checkpoint didn't already resolve to seq_cls.
    This is ms-swift-pipeline-specific (not a torch-compat issue).
    """
    try:
        from swift.pipelines.train.rlhf import SwiftRLHF
    except Exception:
        return
    if getattr(SwiftRLHF, '_swift_jittor_ppo_seqcls_patched', False):
        return
    orig = SwiftRLHF._prepare_single_model

    def _prepare_single_model(self, key, origin_key, model_type, model_revision):
        args = self.args
        # Only PPO's reward/value heads need a regression head. (key=='value' is the
        # trainable critic; key=='reward' the frozen scorer — both go through .score.)
        if getattr(args, 'rlhf_type', None) == 'ppo' and key in {'reward', 'value'}:
            _stm = SwiftRLHF._get_model_task_type

            def _forced_task_type(model_dir, _stm=_stm):
                task_type, num_labels = _stm(model_dir)
                if task_type != 'seq_cls':
                    # plain CausalLM checkpoint -> load as seq_cls so a fresh
                    # num_labels=1 .score head is created (the PPO value head).
                    task_type, num_labels = 'seq_cls', 1
                elif num_labels is None:
                    num_labels = 1
                return task_type, num_labels

            # Temporarily override the task-type resolver for this one call.
            self._get_model_task_type = _forced_task_type
            try:
                return orig(self, key, origin_key, model_type, model_revision)
            finally:
                del self._get_model_task_type
        return orig(self, key, origin_key, model_type, model_revision)

    SwiftRLHF._prepare_single_model = _prepare_single_model
    SwiftRLHF._swift_jittor_ppo_seqcls_patched = True


def _patch_gkd_native_rollout_engine():
    """ms-swift's GKDTrainer only creates a rollout engine for vLLM: its __init__
    calls `prepare_rollout()` which (in RolloutTrainerMixin) builds `self.engine`
    ONLY in the `vllm_mode in {server, colocate}` branches. With `--use_vllm false`
    and `lmbda > 0` (on-policy: student generates its own responses), `_rollout`
    reaches `self.engine.infer(...)` and dies with `AttributeError: 'GKDTrainer'
    object has no attribute 'engine'`.

    GRPOTrainer handles the same native path by building a `TransformersEngine`
    inline in its __init__ (`if not self.args.use_vllm: self.engine =
    TransformersEngine(self.model, ...)`). GKDTrainer simply never got that branch
    — a plain ms-swift gap, unrelated to the shim (the off-policy lmbda=0 path,
    which needs no generation, works fine). Mirror GRPO's setup: after GKDTrainer
    finishes init, if no engine was created and vLLM is off, attach a
    TransformersEngine so native on-policy rollout works.

    ms-swift-specific.
    """
    try:
        from swift.rlhf_trainers.gkd_trainer import GKDTrainer
        from swift.infer_engine import TransformersEngine
    except Exception:
        return
    if getattr(GKDTrainer, '_swift_jittor_native_engine_patched', False):
        return
    orig_init = GKDTrainer.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        # Only when on-policy generation is actually needed and no engine exists.
        if not getattr(self.args, 'use_vllm', False) and getattr(self, 'engine', None) is None:
            from copy import copy
            infer_template = copy(self.template)
            # Generation needs the padded, non-SP, right-trimmed layout (same knobs GRPO uses).
            infer_template.padding_free = False
            infer_template.sequence_parallel_size = 1
            infer_template.remove_unused_columns = True
            self.engine = TransformersEngine(self.model, template=infer_template, max_batch_size=0)
            # The shared RolloutTrainerMixin reads these rollout-state attrs (e.g. in
            # _postprocess_rollout_outputs / padding logic) but GRPO sets them in its
            # OWN __init__, not in the mixin — GKD never does, so on the native path
            # they're missing. Set GRPO's same defaults (single-trajectory, no padding).
            if not hasattr(self, 'dynamic_num_samples'):
                self.dynamic_num_samples = False
            if not hasattr(self, 'rollout_pad_count'):
                self.rollout_pad_count = 0

    GKDTrainer.__init__ = __init__
    GKDTrainer._swift_jittor_native_engine_patched = True


def _patch_transformers_group_images_by_shape():
    """transformers' fast image processor (Qwen2/2.5-VL etc.) groups images with
    `grouped[image.shape[1:]] = ...` -- using the tensor shape as a DICT KEY. On
    jittor `Var.shape` is a NanoVector, which (unlike torch.Size) is NOT hashable,
    so the VL image pipeline dies with `unhashable type: NanoVector`.

    Fixing `.shape` globally to a tuple subclass breaks jittor C++ ops that require
    a real NanoVector, so coerce only at this transformers call site: re-wrap
    `_group_images_by_shape` to normalize each shape key to a plain int tuple. The
    stored (shape, idx) index entries become tuple-keyed too, and `reorder_images`
    keys `processed_images` with those same tuples, so it stays self-consistent.
    Transformers-loader/processor-specific.
    """
    try:
        import transformers.image_transforms as it
    except Exception:
        return
    if getattr(it, "_swift_jittor_group_patched", False):
        return
    orig = getattr(it, "_group_images_by_shape", None)
    if orig is None:
        return

    def _group_images_by_shape(nested_images, *paired_inputs, is_nested=False):
        # Replicate transformers' original grouping but with the shape key coerced to
        # a hashable plain-int tuple (the only change needed for jittor).
        from collections import defaultdict
        grouped_images = defaultdict(list)
        grouped_images_index = {}
        paired_grouped_values = [defaultdict(list) for _ in paired_inputs]
        normalized_images = [nested_images] if not is_nested else nested_images
        normalized_paired = [[p] if not is_nested else p for p in paired_inputs]
        for i, (sublist, *paired_sublists) in enumerate(zip(normalized_images, *normalized_paired)):
            for j, (image, *paired_values) in enumerate(zip(sublist, *paired_sublists)):
                key = (i, j) if is_nested else j
                shape = tuple(int(x) for x in image.shape[1:])  # hashable
                grouped_images[shape].append(image)
                for pidx, pval in enumerate(paired_values):
                    paired_grouped_values[pidx][shape].append(pval)
                grouped_images_index[key] = (shape, len(grouped_images[shape]) - 1)
        if is_nested:
            grouped_images_index["_num_sublists"] = len(normalized_images)
        return (grouped_images, *paired_grouped_values, grouped_images_index)

    it._group_images_by_shape = _group_images_by_shape
    it._swift_jittor_group_patched = True


def _patch_transformers_fused_moe_requires_grad():
    """Fused MoE expert weights (Qwen2/Qwen3-MoE `experts.gate_up_proj` /
    `experts.down_proj`) come out of `from_pretrained` FROZEN on the shim, so the
    experts never train (router learns, experts stay fixed) -- a silent accuracy
    bug, not a crash.

    Cause: transformers builds those fused 3D params by stacking the per-expert
    slices inside `MergeModulelist.convert`, which is decorated `@torch.no_grad`.
    On jittor every op under `no_grad` yields a `stop_grad` (requires_grad=False)
    Var. `set_param_for_module` would normally re-wrap a freshly-loaded tensor via
    `nn.Parameter(t, requires_grad=t.is_floating_point())` (which clears stop_grad),
    BUT it guards that with `if not isinstance(param_value, nn.Parameter)`, and the
    shim's Parameter metaclass reports EVERY Var as a Parameter -- so the re-wrap is
    skipped and the frozen stacked Var is stored as-is. (Dense params load via paths
    that don't run under no_grad, so only the fused MoE experts are hit.)

    torch's invariant is that `from_pretrained` leaves every floating-point param
    trainable (freezing is a separate later step). Restore that invariant: after
    each `set_param_for_module`, re-enable grad on the float param it just stored.
    Transformers-loader-specific.
    """
    try:
        import transformers.core_model_loading as cml
    except Exception:
        return
    if getattr(cml, "_swift_jittor_moe_grad_patched", False):
        return
    orig = cml.set_param_for_module

    def set_param_for_module(model, target_name, param_value, *args, **kwargs):
        orig(model, target_name, param_value, *args, **kwargs)
        try:
            # resolve the stored param the same way set_param_for_module does
            module_path, _, pname = target_name.rpartition(".")
            parent = model.get_submodule(module_path) if module_path else model
            p = getattr(parent, pname, None)
            # only float params, and only those that loading left frozen
            if (p is not None and hasattr(p, "is_stop_grad") and p.is_stop_grad()
                    and getattr(p, "is_floating_point", lambda: False)()):
                p.requires_grad = True
        except Exception:
            pass

    cml.set_param_for_module = set_param_for_module
    cml._swift_jittor_moe_grad_patched = True


def _patch_peft_disable_adapter():
    """peft's `enable_adapters(False)` (used by `disable_adapter()` for the DPO/RLHF
    reference pass) calls `requires_grad_(False)` on every adapter sub-layer.

    On jittor, `stop_grad()` PERMANENTLY severs an already-built autograd graph --
    `start_grad()` afterwards only reconnects FUTURE computations, not the policy
    forward that was built before `disable_adapter()`. So the per-step ref pass
    silently disconnects the policy logps from the LoRA params => grad_norm 0,
    DPO loss frozen at ln(2), no learning. (torch's requires_grad_ never severs
    existing graphs, so this only bites on jittor.)

    The requires_grad toggle is unnecessary: the ref forward runs under
    `torch.no_grad()` anyway, and `_disable_adapters=True` already makes the
    forward skip the LoRA contribution. So override `enable_adapters` to set ONLY
    the disable flag (no requires_grad toggle) -- ref output stays correct AND the
    policy graph stays connected.
    """
    try:
        from peft.tuners.tuners_utils import BaseTunerLayer
    except Exception:
        return

    def enable_adapters(self, enabled: bool) -> None:
        if enabled:
            self.set_adapter(self.active_adapters)
            self._disable_adapters = False
        else:
            # jittor: do NOT requires_grad_(False) here (it would sever the
            # already-built policy graph). Just flag the adapters disabled.
            self._disable_adapters = True

    BaseTunerLayer.enable_adapters = enable_adapters
