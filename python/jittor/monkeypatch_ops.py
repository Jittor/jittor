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
flex_gemm (TRELLIS.2 sparse-conv backend; triton-version compat):
  _patch_flexgemm_triton_autotuner             make real triton 3.1's Autotuner tolerate flex_gemm's triton>=3.2 API
transformers (TRELLIS.2 DINOv3 conditioner; version layout compat):
  _patch_dinov3_encoder_layer_layout           locate DINOv3ViTModel transformer layers across transformers 4.x/5.x
                                               (+ env-gated TRELLIS_DINOV3_PATH local weight redirect)
TRELLIS.2 pipeline construction glue:
  _patch_trellis2_rembg_lazy                   defer the gated RMBG-2.0 download (only needed for alpha-less inputs)

A separate, explicitly-called installer (NOT in apply(), trellis2-only):
  install_trellis2_sparse_conv_jittor          pure-jittor submanifold sparse-conv3d backend
                                               (flex_gemm's REAL Triton conv doesn't run on the bridge)
"""

_INSTALLED = False


def apply():
    """Apply all external-library monkeypatches (idempotent, opt-in).

    Safe to call MORE THAN ONCE: every individual patch self-guards (its own
    `_jt_*` / `_swift_jittor_*` sentinel or a try/except import), and the
    library-dependent ones (flex_gemm / trellis2 / transformers / peft / ms-swift)
    no-op until that library is importable. So a caller can call apply() early
    (before importing the dep) and AGAIN after — the dep-dependent patches take
    effect on the later call. We therefore always run the full list rather than
    short-circuiting on a global flag."""
    global _INSTALLED
    _patch_transformers_legacy_symbols()
    _patch_transformers_legacy_tied_weights_keys()
    _patch_peft_disable_adapter()
    _patch_transformers_fused_moe_requires_grad()
    _patch_transformers_group_images_by_shape()
    _patch_gkd_native_rollout_engine()
    _patch_ppo_reward_value_seq_cls()
    _patch_flexgemm_triton_autotuner()
    _patch_dinov3_encoder_layer_layout()
    _patch_trellis2_rembg_lazy()
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
    orig = getattr(PreTrainedModel, "get_expanded_tied_weights_keys", None)
    if orig is None:
        return

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
    try:
        import sys as _sys
        _torch_mod = _sys.modules.get("torch")
        if getattr(_torch_mod, "__name__", "") == "jittor" and not getattr(iu, "_jt_torch_backend_patched", False):
            def _jt_is_torch_available():
                return True
            def _jt_get_torch_version():
                # Transformers 5 gates DINOv3 behind torch>=2.4. The active
                # torch module is jittor's shim, not the real PyTorch package.
                return "2.4.0"
            try:
                iu.is_torch_available.cache_clear()
            except Exception:
                pass
            try:
                iu.get_torch_version.cache_clear()
            except Exception:
                pass
            iu.is_torch_available = _jt_is_torch_available
            iu.get_torch_version = _jt_get_torch_version
            if hasattr(iu, "BACKENDS_MAPPING") and "torch" in iu.BACKENDS_MAPPING:
                iu.BACKENDS_MAPPING["torch"] = (_jt_is_torch_available, getattr(iu, "PYTORCH_IMPORT_ERROR", ""))
            try:
                import transformers.utils as tu
                tu.is_torch_available = _jt_is_torch_available
                tu.get_torch_version = _jt_get_torch_version
            except Exception:
                pass
            try:
                import transformers.utils.generic as tg
                tg._is_torch_available = True
            except Exception:
                pass
            try:
                _refresh_transformers_lazy_torch_backend()
            except Exception:
                pass
            iu._jt_torch_backend_patched = True
    except Exception:
        pass
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


def _refresh_transformers_lazy_torch_backend():
    """Refresh already-created transformers LazyModules after enabling the shim.

    transformers 5 builds LazyModule import tables while importing
    transformers.utils.import_utils. If it decided torch was unavailable at that
    moment, symbols such as DINOv3ViTModel are cached as Placeholder classes.
    Once sys.modules["torch"] is the jittor module, remove only the missing
    "torch" backend mark and any Placeholder cached from it.
    """
    import sys
    try:
        from transformers.utils.import_utils import _LazyModule
    except Exception:
        return

    for mod in list(sys.modules.values()):
        if not isinstance(mod, _LazyModule):
            continue
        missing_map = getattr(mod, "_object_missing_backend", None)
        if not isinstance(missing_map, dict):
            continue
        for name, missing in list(missing_map.items()):
            if "torch" not in missing:
                continue
            rest = [b for b in missing if b != "torch"]
            if rest:
                missing_map[name] = rest
            else:
                missing_map.pop(name, None)
            cached = getattr(mod, "__dict__", {}).get(name, None)
            if getattr(cached, "_backends", None) is not None:
                try:
                    delattr(mod, name)
                except Exception:
                    mod.__dict__.pop(name, None)


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
    import sys
    if "swift" not in sys.modules:
        return
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
    import sys
    if "swift" not in sys.modules:
        return
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
    import sys
    if "peft" not in sys.modules:
        return
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


def _patch_flexgemm_triton_autotuner():
    """Make the REAL triton's Autotuner tolerate flex_gemm's newer-triton API so
    `import flex_gemm` succeeds AND its autotuned Triton kernels actually RUN on
    the installed triton 3.1.0.

    flex_gemm (TRELLIS.2's default sparse-conv backend, config.CONV='flex_gemm')
    ships `TritonPersistentCacheAutotuner(triton.runtime.Autotuner)`, written for
    triton>=3.2, with two incompatibilities against triton 3.1.0:

      (1) its __init__ passes one extra positional `do_bench` that triton 3.1.0's
          `Autotuner.__init__` does not accept -> ImportError at flex_gemm import
          time (the autotuner objects are created at module import via the
          `@triton_autotune` decorator).
      (2) its overridden `run()` reads `self.keys` (the list of key arg NAMES, a
          triton>=3.2 attribute); triton 3.1.0 stores only `self.key_idx`. Without
          `self.keys` the first autotuned kernel dies with AttributeError.

    Fix: wrap `Autotuner.__init__` to (a) truncate extra positionals / drop unknown
    kwargs so the real signature is satisfied, and (b) back-fill `self.keys` from
    the `key` argument. This is purely a flex_gemm-vs-triton-version compat shim —
    it would be required under real torch + triton 3.1.0 too — and it depends on
    NEITHER jittor internals NOR flex_gemm (it only touches triton's own class),
    so it is safe whenever a real triton is importable and a no-op otherwise.

    NOTE: this patches the autotuner's PYTHON wrapper only. The actual kernel
    execution (a 2-D tiled GEMM) runs through jittor's triton BRIDGE
    (jittor.triton_shim), which compiles the real @triton.jit kernel and launches
    the cubin on jittor Vars. Activating that bridge is a separate concern (import
    jittor.triton_shim); this function does not touch it.
    """
    try:
        import inspect
        import triton.runtime.autotuner as _A
    except Exception:
        return
    if getattr(_A.Autotuner.__init__, "_jt_tolerant", False):
        return
    _orig = _A.Autotuner.__init__
    sig = inspect.signature(_orig)
    params = list(sig.parameters)[1:]          # drop 'self'
    n_pos = len(params)
    accepted = set(params)
    key_pos = params.index("key") if "key" in params else None
    do_bench_pos = params.index("do_bench") if "do_bench" in params else None

    def _jt_do_bench():
        try:
            from jittor.triton_shim import backend as _jt_triton_backend
            return _jt_triton_backend.make_do_bench()
        except Exception:
            return None

    def _tolerant_init(self, *args, **kwargs):
        # recover the key-name list BEFORE truncating extras
        key_val = kwargs.get("key", None)
        if key_val is None and key_pos is not None and key_pos < len(args):
            key_val = args[key_pos]
        args = list(args[:n_pos])              # drop extras (e.g. do_bench)
        kwargs = {k: v for k, v in kwargs.items() if k in accepted}
        bench = _jt_do_bench()
        if bench is not None and do_bench_pos is not None:
            if do_bench_pos < len(args):
                if args[do_bench_pos] is None:
                    args[do_bench_pos] = bench
            elif kwargs.get("do_bench", None) is None:
                kwargs["do_bench"] = bench
        _orig(self, *args, **kwargs)
        # triton 3.1.0 has no self.keys (only self.key_idx); flex_gemm's run()
        # needs the name list. Provide it (newer-triton-compatible attribute).
        if not hasattr(self, "keys"):
            self.keys = list(key_val) if key_val is not None else []
    _tolerant_init._jt_tolerant = True
    _A.Autotuner.__init__ = _tolerant_init


def _patch_dinov3_encoder_layer_layout():
    """Locate the DINOv3ViTModel transformer-block ModuleList robustly across
    transformers versions, for TRELLIS.2's DinoV3FeatureExtractor.

    TRELLIS.2's conditioner (`trellis2.modules.image_feature_extractor`) does
    `extract_features` by iterating `self.model.layer`. In transformers 5.x the
    `DINOv3ViTModel` nests its encoder as `self.model.model` (a DINOv3ViTEncoder)
    whose layer ModuleList is `self.model.model.layer` -> the original code
    AttributeErrors on `.layer`. Re-implement `extract_features` to find the layer
    list across `.layer` / `.model.layer` / `.encoder.layer`, returning the
    pre-final-norm hidden states the conditioner then layer-norms itself.

    Same problem under real torch with transformers 5.x (the attribute moved
    there too), so this is a transformers-version-layout compat shim for the
    third-party TRELLIS.2 model code, not a jittor concern. No-op if trellis2 is
    not importable.

    ALSO (env-gated, model-path config — NOT a code workaround): the conditioner
    loads `facebook/dinov3-vitl16-pretrain-lvd1689m`, which is GATED on HF (403).
    If `TRELLIS_DINOV3_PATH` points at a local transformers-format DINOv3 dir
    (an ungated mirror with config.json + model.safetensors), redirect __init__ to
    load from there. This only swaps WHERE the identical weights come from; absent
    the env var, __init__ is left exactly as TRELLIS.2 ships it.
    """
    try:
        import importlib
        ife = importlib.import_module("trellis2.modules.image_feature_extractor")
    except Exception:
        return
    cls = getattr(ife, "DinoV3FeatureExtractor", None)
    if cls is None or getattr(getattr(cls, "extract_features", None),
                              "_jt_layer_compat", False):
        return
    try:
        import os as _os
        import torch.nn.functional as F
    except Exception:
        return

    # --- env-gated local weight-path redirect (gated facebook repo -> local) ---
    _dino_path = _os.environ.get("TRELLIS_DINOV3_PATH")
    if _dino_path and _os.path.isdir(_dino_path) \
            and not getattr(cls.__init__, "_jt_local_redirect", False):
        from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTModel
        from torchvision import transforms

        def _local_init(self, model_name, image_size=512, _p=_dino_path):
            self.model_name = _p
            self.model = DINOv3ViTModel.from_pretrained(_p)
            self.model.eval()
            self.image_size = image_size
            self.transform = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        _local_init._jt_local_redirect = True
        cls.__init__ = _local_init

    def _extract_features(self, image):
        model = self.model
        image = image.to(model.embeddings.patch_embeddings.weight.dtype)
        hidden_states = model.embeddings(image, bool_masked_pos=None)
        position_embeddings = model.rope_embeddings(image)
        # Locate the transformer-block ModuleList across transformers versions.
        if hasattr(model, "layer"):
            layers = model.layer
        elif hasattr(model, "model") and hasattr(model.model, "layer"):
            layers = model.model.layer
        elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
            layers = model.encoder.layer
        else:
            raise AttributeError(
                "DINOv3ViTModel: could not locate transformer layers "
                "(tried .layer, .model.layer, .encoder.layer)")
        for layer_module in layers:
            hidden_states = layer_module(
                hidden_states, position_embeddings=position_embeddings)
        return F.layer_norm(hidden_states, hidden_states.shape[-1:])
    _extract_features._jt_layer_compat = True
    cls.extract_features = _extract_features


def _patch_trellis2_rembg_lazy():
    """Make TRELLIS.2's `rembg.BiRefNet.__init__` lazy so building the pipeline
    does NOT eagerly download the GATED `briaai/RMBG-2.0` background-removal model.

    `Trellis2ImageTo3DPipeline.from_pretrained` unconditionally constructs
    `rembg.BiRefNet(...)`, whose __init__ calls
    `AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-2.0",
    trust_remote_code=True)` — a network download that 403s (the repo is gated).
    The model is ONLY used for background removal when the input image has no
    alpha channel; for an RGBA input it is never invoked. Defer the heavy load
    until the model is actually called, so pipeline construction (and the whole
    alpha-input path) needs no RMBG download. If an alpha-less image ever reaches
    it, it downloads then (and would still 403 — but that's correct: the model is
    genuinely required only there).

    Same behaviour under real torch (the repo is gated there too). Trellis2
    pipeline-construction glue, env-independent. No-op if rembg is not importable.
    """
    try:
        import importlib
        birefnet_mod = importlib.import_module("trellis2.pipelines.rembg.BiRefNet")
    except Exception:
        return
    BiRefNet = getattr(birefnet_mod, "BiRefNet", None)
    if BiRefNet is None or getattr(BiRefNet.__init__, "_jt_lazy", False):
        return
    from torchvision import transforms

    def _lazy_init(self, model_name="ZhengPeng7/BiRefNet"):
        self.model_name = model_name
        self.model = None
        self.transform_image = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    _lazy_init._jt_lazy = True

    _orig_call = BiRefNet.__call__

    def _ensure(self):
        if self.model is None:
            from transformers import AutoModelForImageSegmentation
            self.model = AutoModelForImageSegmentation.from_pretrained(
                self.model_name, trust_remote_code=True)
            self.model.eval()

    def _guarded_call(self, image):
        _ensure(self)
        return _orig_call(self, image)

    BiRefNet.__init__ = _lazy_init
    BiRefNet.__call__ = _guarded_call
    BiRefNet.to = lambda self, device: (self.model.to(device) if self.model is not None else None)
    BiRefNet.cuda = lambda self: (self.model.cuda() if self.model is not None else None)
    BiRefNet.cpu = lambda self: (self.model.cpu() if self.model is not None else None)


# ---------------------------------------------------------------------------- #
#  TRELLIS.2 sparse submanifold conv3d — pure-jittor backend (opt-in)
# ---------------------------------------------------------------------------- #
# NOT part of apply(): this is a separate, explicitly-called installer because it
# is only relevant when running TRELLIS.2 AND only needed because flex_gemm's REAL
# Triton conv does not run on jittor's triton-bridge (its CUDA neighbor-map kernels
# DO run + match a dense reference, but the bridge-launched Triton GEMM corrupts
# memory — verified). trellis2's conv dispatcher does
#   importlib.import_module(f'..conv_{config.CONV}', __name__)
# and config.CONV's env only accepts none/spconv/torchsparse/flex_gemm, so the
# pure-jittor backend cannot be selected by env; we pre-register it under the
# dotted module name the dispatcher will import and flip config.CONV to 'jittor'.
#
# The backend implements submanifold conv (stride=1, output coords == input
# coords) with a per-tap neighbour hashmap (gather + matmul + masked-accumulate),
# fully differentiable, matching flex_gemm's weight layout (Co, Kd, Kh, Kw, Ci) so
# flex_gemm checkpoints load unchanged. Accuracy vs a dense reference ~5e-8.

def install_trellis2_sparse_conv_jittor():
    """Register + select a pure-jittor submanifold sparse-conv3d backend for
    TRELLIS.2 (replaces flex_gemm conv, which doesn't run via the triton-bridge).
    Idempotent; no-op if trellis2 is not importable. Returns True on success."""
    import sys
    import importlib
    try:
        spcfg = importlib.import_module('trellis2.modules.sparse.config')
    except Exception:
        return False

    mod_name = 'trellis2.modules.sparse.conv.conv_jittor'
    if mod_name not in sys.modules:
        sys.modules[mod_name] = _build_trellis2_jittor_conv_module(mod_name)

    # set_conv_backend's type hint is Literal[...] but Python does not enforce it.
    spcfg.set_conv_backend('jittor')
    # clear any cached different backend in the dispatcher
    try:
        conv_mod = importlib.import_module('trellis2.modules.sparse.conv.conv')
        conv_mod._backends.pop('jittor', None)
    except Exception:
        pass
    return True


def _build_trellis2_jittor_conv_module(mod_name):
    """Build the conv_jittor backend module object (module-level API mirroring
    trellis2's conv_flex_gemm: sparse_conv3d_init / sparse_conv3d_forward / the
    inverse stubs). Lives here, injected into sys.modules under `mod_name`."""
    import types
    import math
    import torch
    import torch.nn as nn

    m = types.ModuleType(mod_name)
    m.__package__ = 'trellis2.modules.sparse.conv'

    def _encode(coords_xyz, base):
        x = coords_xyz[:, 0]; y = coords_xyz[:, 1]; z = coords_xyz[:, 2]
        return (x * base + y) * base + z

    def _build_hashmap(coords, dilation, kernel_size):
        coords_l = coords.long()
        b = coords_l[:, 0]; xyz = coords_l[:, 1:]
        Kd, Kh, Kw = kernel_size
        max_disp = max((Kd // 2), (Kh // 2), (Kw // 2)) * max(dilation)
        pad = int(max_disp) + 1
        T = coords.shape[0]
        max_coord = int(xyz.max().item()) if T > 0 else 0
        base = max_coord + 1 + 2 * pad
        xyz_shift = xyz + pad
        keys = b * (base * base * base) + _encode(xyz_shift, base)
        order = torch.argsort(keys)
        return {'sorted_keys': keys[order], 'order': order, 'base': base,
                'pad': pad, 'T': T, 'b': b, 'xyz': xyz}

    def _lookup(hashmap, query_keys):
        sorted_keys = hashmap['sorted_keys']; order = hashmap['order']; T = hashmap['T']
        if T == 0:
            z = torch.zeros_like(query_keys); return z, (z == 1)
        pos = torch.searchsorted(sorted_keys, query_keys).clamp(0, T - 1)
        valid = sorted_keys[pos] == query_keys
        src_index = torch.where(valid, order[pos], torch.zeros_like(order[pos]))
        return src_index, valid

    def _neighbor_cache(hashmap, kernel_size, dilation):
        Kd, Kh, Kw = kernel_size
        cd, ch, cw = Kd // 2, Kh // 2, Kw // 2
        dd, dh, dw = dilation
        base = hashmap['base']; pad = hashmap['pad']; b = hashmap['b']; xyz = hashmap['xyz']
        batch_block = b * (base * base * base)
        cache = []
        for kd in range(Kd):
            for kh in range(Kh):
                for kw in range(Kw):
                    nbr = xyz.clone()
                    nbr[:, 0] = nbr[:, 0] + (kd - cd) * dd
                    nbr[:, 1] = nbr[:, 1] + (kh - ch) * dh
                    nbr[:, 2] = nbr[:, 2] + (kw - cw) * dw
                    q_keys = batch_block + _encode(nbr + pad, base)
                    cache.append(_lookup(hashmap, q_keys))
        return cache

    def _conv_forward(feats, coords, weight, bias, neighbor_cache, dilation):
        Co, Kd, Kh, Kw, Ci = weight.shape
        T = feats.shape[0]
        if neighbor_cache is None:
            hashmap = _build_hashmap(coords, dilation, (Kd, Kh, Kw))
            neighbor_cache = _neighbor_cache(hashmap, (Kd, Kh, Kw), dilation)
        out = torch.zeros((T, Co), dtype=feats.dtype, device=feats.device)
        tap = 0
        for kd in range(Kd):
            for kh in range(Kh):
                for kw in range(Kw):
                    src_index, valid = neighbor_cache[tap]; tap += 1
                    w = weight[:, kd, kh, kw, :]
                    gathered = feats[src_index]
                    contrib = torch.matmul(gathered, w.transpose(0, 1))
                    out = out + contrib * valid.unsqueeze(1).to(contrib.dtype)
        if bias is not None:
            out = out + bias.reshape(1, Co)
        return out, neighbor_cache

    def sparse_conv3d_init(self, in_channels, out_channels, kernel_size, stride=1,
                           dilation=1, padding=None, bias=True, indice_key=None):
        is_unit_stride = (tuple(stride) == (1, 1, 1)) if isinstance(stride, (list, tuple)) else (stride == 1)
        assert is_unit_stride and padding is None, \
            'jittor sparse backend only supports submanifold conv (stride=1, padding=None)'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = tuple(kernel_size) if isinstance(kernel_size, (list, tuple)) else (kernel_size,) * 3
        self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride,) * 3
        self.dilation = tuple(dilation) if isinstance(dilation, (list, tuple)) else (dilation,) * 3
        self.weight = nn.Parameter(torch.empty((out_channels, in_channels, *self.kernel_size)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(self.bias, -bound, bound)
        # (Co, Ci, Kd, Kh, Kw) -> (Co, Kd, Kh, Kw, Ci) to match flex_gemm checkpoints.
        self.weight = nn.Parameter(self.weight.permute(0, 2, 3, 4, 1).contiguous())

    def sparse_conv3d_forward(self, x):
        Co, Kd, Kh, Kw, Ci = self.weight.shape
        key = f'SubMConv3d_jittor_neighbor_cache_{Kw}x{Kh}x{Kd}_dilation{self.dilation}'
        neighbor_cache = x.get_spatial_cache(key)
        out, neighbor_cache_ = _conv_forward(
            x.feats, x.coords, self.weight, self.bias, neighbor_cache, self.dilation)
        if neighbor_cache is None:
            x.register_spatial_cache(key, neighbor_cache_)
        return x.replace(out)

    def sparse_inverse_conv3d_init(self, *a, **k):
        raise NotImplementedError('SparseInverseConv3d (jittor backend) not implemented')

    def sparse_inverse_conv3d_forward(self, x):
        raise NotImplementedError('SparseInverseConv3d (jittor backend) not implemented')

    m.sparse_conv3d_init = sparse_conv3d_init
    m.sparse_conv3d_forward = sparse_conv3d_forward
    m.sparse_inverse_conv3d_init = sparse_inverse_conv3d_init
    m.sparse_inverse_conv3d_forward = sparse_inverse_conv3d_forward
    return m


# --------------------------------------------------------------------------- #
#  flex_gemm REAL conv on the bridge — select the bridge-correct algorithm
# --------------------------------------------------------------------------- #
# flex_gemm exposes 5 sparse-conv algorithms (flex_gemm.ops.spconv.Algorithm).
# Its DEFAULT, MASKED_IMPLICIT_GEMM_SPLITK, does not survive jittor's triton
# bridge: a single masked-split-K launch is correct (cos≈1 vs a dense ref), but
# executing that particular cubin **twice** via the driver-API launcher (which the
# autotuner does while sweeping configs) corrupts the CUDA launch state so the
# NEXT flex_gemm *CUDA-extension* kernel (neighbor_map_post_process_for_masked_
# implicit_gemm_2) faults with cudaErrorIllegalAddress — even though every input
# Var is byte-intact and plain jittor ops keep working. (Verified: args/grid/
# constexprs/shared/scratch all correct, n_ptx_params matches, a single launch is
# exact; the IMPLICIT_GEMM_SPLITK kernel additionally has a latent over-read —
# `weight + k_start*BK` when BK>Ci — that only torch's slack allocator hides.)
#
# The plain IMPLICIT_GEMM algorithm has NEITHER problem: it is a straight
# implicit-GEMM (no split-K partial-sum buffer, no masked valid-kernel callback
# tensors, correct weight pointer) and runs on the bridge matching a dense conv
# reference to rel ~7e-4. So to run flex_gemm's REAL Triton conv on jittor with NO
# pure-jittor fallback we select IMPLICIT_GEMM. This keeps trellis2's
# config.CONV='flex_gemm' (real flex_gemm conv: its CUDA neighbor-map kernels +
# the Triton implicit-GEMM through the bridge), just on the bridge-robust
# algorithm. A tiny launch counter on the spconv forward proves the conv really
# went through flex_gemm (not the jittor fallback).

#: incremented once per flex_gemm submanifold-conv forward we route through the bridge
FLEXGEMM_BRIDGE_CONV_CALLS = 0


def force_flexgemm_bridge_algorithm(algorithm="IMPLICIT_GEMM"):
    """Make flex_gemm's REAL sparse-conv run on jittor's triton bridge by selecting
    the bridge-correct algorithm (default ``IMPLICIT_GEMM``) and instrumenting the
    forward with a call counter.

    Returns the algorithm name on success (and leaves ``config.CONV='flex_gemm'``
    untouched so trellis2 uses flex_gemm's real conv), or ``None`` if flex_gemm is
    not importable. Idempotent. After this, ``FLEXGEMM_BRIDGE_CONV_CALLS`` counts
    how many submanifold convs went through flex_gemm — a nonzero value is proof
    the real flex_gemm conv ran (the pure-jittor fallback never touches it).
    """
    try:
        import flex_gemm.ops.spconv as spconv_ops
    except Exception:
        return None
    Algorithm = spconv_ops.Algorithm
    alg = getattr(Algorithm, algorithm, None)
    if alg is None:
        alg = algorithm  # allow passing the raw string value
    spconv_ops.set_algorithm(alg)

    # trellis2's conv_flex_gemm re-applies `set_algorithm(config.FLEX_GEMM_ALGO)`
    # on EVERY forward, so a global set_algorithm alone is reverted on the next
    # conv. Pin trellis2's own config knob too (the algorithm value is the same
    # lowercase string the flex_gemm Algorithm enum uses). No-op if trellis2 isn't
    # importable yet (re-call after importing trellis2, like apply()).
    try:
        import trellis2.modules.sparse.conv.config as _t2cfg
        _t2cfg.FLEX_GEMM_ALGO = alg if isinstance(alg, str) else str(alg)
    except Exception:
        pass

    # Instrument the submanifold-conv forward (once) so we can prove flex_gemm ran.
    try:
        from flex_gemm.ops.spconv import submanifold_conv3d as _sm
        fn = _sm.SubMConv3dFunction
        if not getattr(fn, "_jt_bridge_counted", False):
            _orig_fwd = fn._sparse_submanifold_conv_forward.__func__ \
                if hasattr(fn._sparse_submanifold_conv_forward, "__func__") \
                else fn._sparse_submanifold_conv_forward

            def _counted_fwd(feats, neighbor_cache, weight, bias=None):
                global FLEXGEMM_BRIDGE_CONV_CALLS
                FLEXGEMM_BRIDGE_CONV_CALLS += 1
                return _orig_fwd(feats, neighbor_cache, weight, bias)

            fn._sparse_submanifold_conv_forward = staticmethod(_counted_fwd)
            fn._jt_bridge_counted = True
    except Exception:
        pass
    return algorithm
