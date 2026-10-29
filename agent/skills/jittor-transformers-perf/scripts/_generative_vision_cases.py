"""Identical model definitions and CPU inputs for native Torch and the Torch shim.

This module contains no runtime selection or device placement. The caller selects
the imported ``torch`` implementation, saves tiny native checkpoints, moves the
model/inputs, synchronizes, and measures. Tiny models are compatibility fixtures;
only downloaded full checkpoints are valid model performance measurements.
"""

from pathlib import Path

import numpy as np


import _stock_vision_cases as stock_vision


CASES = ("wan", "sdxl", "siglip2", "dinov2", "sam2", "sam2_image") + stock_vision.CASES


def _case(case):
    if case not in CASES:
        raise ValueError(f"Unknown case {case!r}; expected one of {CASES}")
    return case


def _component_path(model_path, component):
    path = Path(model_path)
    return path / component if (path / component / "config.json").exists() else path


def _attention_config(config, implementation):
    # Transformers propagates the top-level choice to its nested pretrained
    # modules. Set nested configs explicitly as well for direct tiny construction.
    config._attn_implementation = implementation
    for name in ("vision_config", "backbone_config", "mask_decoder_config"):
        child = getattr(config, name, None)
        if child is not None:
            _attention_config(child, implementation)
    return config


def _tiny_sam_config(implementation):
    from transformers import (
        Sam2Config,
        Sam2HieraDetConfig,
        Sam2MaskDecoderConfig,
        Sam2PromptEncoderConfig,
        Sam2VisionConfig,
    )

    backbone = Sam2HieraDetConfig(
        hidden_size=16,
        image_size=[64, 64],
        blocks_per_stage=[1, 1, 2, 1],
        embed_dim_per_stage=[16, 32, 64, 128],
        num_attention_heads_per_stage=[1, 2, 4, 8],
        window_size_per_stage=[4, 4, 4, 2],
        window_positional_embedding_background_size=[4, 4],
        # Global attention must be a non-pooling block: the pinned upstream
        # implementation computes window_size/query_stride unconditionally
        # for pooling blocks, so a global pooling block divides by zero.
        global_attention_blocks=[3],
        mlp_ratio=2.0,
    )
    vision = Sam2VisionConfig(
        backbone_config=backbone,
        backbone_channel_list=[128, 64, 32, 16],
        backbone_feature_sizes=[[16, 16], [8, 8], [4, 4]],
        fpn_hidden_size=32,
    )
    prompt = Sam2PromptEncoderConfig(
        hidden_size=32, image_size=64, patch_size=16, mask_input_channels=8
    )
    decoder = Sam2MaskDecoderConfig(
        hidden_size=32,
        mlp_dim=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        iou_head_hidden_dim=32,
    )
    return _attention_config(
        Sam2Config(vision_config=vision, prompt_encoder_config=prompt, mask_decoder_config=decoder),
        implementation,
    )


def create_model(torch, case, model_path=None, tiny=False, dtype=None, attn_implementation="eager"):
    """Build a CPU model or load an offline local checkpoint.

    ``tiny=True, model_path=None`` creates a small randomly initialized model for
    the caller to save with ``save_pretrained``. Providing ``model_path`` always
    loads those exact weights, including when ``tiny=True``. Full SDXL accepts a
    pipeline root or its ``unet`` directory; Wan accepts its ``transformer``.

    Wan uses the official Diffusers native SDPA dispatcher for both choices:
    there is no separate eager Wan processor in this pinned Diffusers release.
    """
    _case(case)
    if case in stock_vision.CASES:
        return stock_vision.create_model(
            torch, case, model_path=model_path, tiny=tiny, dtype=dtype,
            attn_implementation=attn_implementation,
        )
    if attn_implementation not in ("eager", "sdpa"):
        raise ValueError("attn_implementation must be 'eager' or 'sdpa'")
    if not tiny and model_path is None:
        raise ValueError("Full models require a downloaded local model_path")

    if case == "wan":
        from diffusers import WanTransformer3DModel

        if model_path is not None:
            # Wan declares modules that remain FP32 during loading; Diffusers
            # requires its low-memory loader for this architecture.
            args = dict(local_files_only=True, use_safetensors=True, low_cpu_mem_usage=True)
            if dtype is not None:
                args["torch_dtype"] = dtype
            model = WanTransformer3DModel.from_pretrained(str(_component_path(model_path, "transformer")), **args)
        else:
            model = WanTransformer3DModel(
                patch_size=(1, 2, 2),
                num_attention_heads=2,
                attention_head_dim=16,
                in_channels=4,
                out_channels=4,
                text_dim=32,
                freq_dim=16,
                ffn_dim=64,
                num_layers=2,
                rope_max_seq_len=32,
            )

    elif case == "sdxl":
        from diffusers import UNet2DConditionModel
        from diffusers.models.attention_processor import AttnProcessor, AttnProcessor2_0

        if model_path is not None:
            args = dict(local_files_only=True, use_safetensors=True, low_cpu_mem_usage=False)
            if dtype is not None:
                args["torch_dtype"] = dtype
            path = _component_path(model_path, "unet")
            # The official downloaded SDXL file uses this suffix; tiny saved
            # checkpoints have the usual unsuffixed diffusion_pytorch_model.
            if (path / "diffusion_pytorch_model.fp16.safetensors").exists():
                args["variant"] = "fp16"
            model = UNet2DConditionModel.from_pretrained(str(path), **args)
        else:
            model = UNet2DConditionModel(
                sample_size=16,
                in_channels=4,
                out_channels=4,
                down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
                up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
                block_out_channels=(32, 64),
                layers_per_block=1,
                cross_attention_dim=32,
                attention_head_dim=(4, 8),
                norm_num_groups=8,
                addition_embed_type="text_time",
                addition_time_embed_dim=8,
                projection_class_embeddings_input_dim=64,
                use_linear_projection=True,
                transformer_layers_per_block=(1, 1),
            )
        model.set_attn_processor(AttnProcessor() if attn_implementation == "eager" else AttnProcessor2_0())

    else:
        from transformers import (
            Dinov2Config,
            Dinov2Model,
            Sam2Config,
            Sam2Model,
            SiglipVisionConfig,
            SiglipVisionModel,
        )

        if case == "siglip2":
            cls = SiglipVisionModel
            if model_path is not None:
                config = SiglipVisionConfig.from_pretrained(str(model_path), local_files_only=True)
            else:
                config = SiglipVisionConfig(
                    hidden_size=32,
                    intermediate_size=64,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    image_size=32,
                    patch_size=8,
                )
        elif case == "dinov2":
            cls = Dinov2Model
            if model_path is not None:
                config = Dinov2Config.from_pretrained(str(model_path), local_files_only=True)
            else:
                config = Dinov2Config(
                    hidden_size=32,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    image_size=28,
                    patch_size=14,
                    mlp_ratio=2,
                )
        else:
            cls = Sam2Model
            config = (
                Sam2Config.from_pretrained(str(model_path), local_files_only=True)
                if model_path is not None
                else _tiny_sam_config(attn_implementation)
            )
        _attention_config(config, attn_implementation)
        if model_path is not None:
            args = dict(
                config=config,
                local_files_only=True,
                use_safetensors=True,
                low_cpu_mem_usage=False,
                attn_implementation=attn_implementation,
            )
            if dtype is not None:
                args["torch_dtype"] = dtype
            model = cls.from_pretrained(str(model_path), **args)
        else:
            model = cls(config)

    if model_path is None and dtype is not None:
        model = model.to(device="cpu", dtype=dtype)
    if case == "wan":
        # 310P has no fp64 arithmetic; use the same explicit FP32 RoPE
        # buffers in both processes, preserving the checkpoint's official
        # keep-in-FP32 parameters (time embedding/norm/scale-shift tables).
        model.rope.float()
    return model.eval()


def _input_description(value):
    if isinstance(value, dict):
        return {key: _input_description(item) for key, item in value.items()}
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "device": "cpu"}
    return value


def make_inputs(torch, case, model, rng, tiny=False, profile="default"):
    """Return deterministic CPU kwargs and a JSON-safe description.

    ``rng`` is a NumPy Generator or RandomState, not a framework RNG. Copies of
    the same NumPy state give identical input arrays across both runtimes.
    ``profile='large'`` selects Wan 704x1280x17; the default is 320x512x17.
    Other full cases use SDXL1024, SigLIP224, DINO518, and SAM1024.
    """
    _case(case)
    if case in stock_vision.CASES:
        return stock_vision.make_inputs(torch, case, model, rng, tiny=tiny, profile=profile)
    if profile not in ("default", "large"):
        raise ValueError("profile must be 'default' or 'large'")
    dtype = model.patch_embedding.weight.dtype if case == "wan" else next(model.parameters()).dtype

    def normal(shape, scale=1.0):
        array = (rng.standard_normal(shape) * scale).astype(np.float32)
        return torch.tensor(array, dtype=dtype, device="cpu")

    def constant(value, dtype_override=None):
        return torch.tensor(value, dtype=dtype if dtype_override is None else dtype_override, device="cpu")

    if case == "wan":
        if tiny:
            latent_shape, text_length, resolution = (1, 4, 2, 4, 4), 8, None
        else:
            latent_shape = (1, 48, 5, 44, 80) if profile == "large" else (1, 48, 5, 20, 32)
            text_length = 512
            resolution = [704, 1280, 17] if profile == "large" else [320, 512, 17]
        patches = model.config.patch_size
        tokens = (latent_shape[2] // patches[0]) * (latent_shape[3] // patches[1]) * (latent_shape[4] // patches[2])
        kwargs = {
            "hidden_states": normal(latent_shape),
            # Wan2.2 TI2V expands time conditioning per token, including text to
            # video. A scalar timestep omits work from the official pipeline.
            "timestep": constant(np.full((1, tokens), 500, dtype=np.float32), torch.float32),
            "encoder_hidden_states": normal((1, text_length, model.config.text_dim)),
            "return_dict": True,
        }
        workload = {
            "component": "one_conditional_denoiser_forward",
            "video_height_width_frames": resolution,
            "self_attention_tokens": tokens,
            "attention": "diffusers_native_sdpa",
            "guidance_passes": 1,
        }

    elif case == "sdxl":
        size = 16 if tiny else 128
        text_dim = model.config.cross_attention_dim
        pool_dim = model.config.projection_class_embeddings_input_dim - 6 * model.config.addition_time_embed_dim
        image_size = size * 8
        kwargs = {
            "sample": normal((1, model.config.in_channels, size, size)),
            "timestep": constant([500], torch.int64),
            "encoder_hidden_states": normal((1, 8 if tiny else 77, text_dim)),
            "added_cond_kwargs": {
                "text_embeds": normal((1, pool_dim)),
                "time_ids": constant([[image_size, image_size, 0, 0, image_size, image_size]]),
            },
            "return_dict": True,
        }
        workload = {
            "component": "one_conditional_denoiser_forward",
            "image_height_width": [image_size, image_size],
            "guidance_passes": 1,
        }

    elif case in ("siglip2", "dinov2"):
        size = model.config.image_size if tiny or case == "siglip2" else 518
        kwargs = {"pixel_values": normal((1, 3, size, size)), "return_dict": True}
        workload = {"component": "vision_encoder", "image_height_width": [size, size]}

    else:
        size = model.config.prompt_encoder_config.image_size
        kwargs = {"pixel_values": normal((1, 3, size, size))}
        if case == "sam2":
            kwargs.update(
                input_points=constant([[[[size * 0.5, size * 0.5]]]]),
                input_labels=constant([[[1]]], torch.int64),
                multimask_output=True,
            )
        workload = {
            "component": "image_encoder" if case == "sam2_image" else "image_encoder_prompt_and_mask_decoder",
            "image_height_width": [size, size],
        }
    spec = {"case": case, "tiny": tiny, "profile": profile, **workload, "inputs": _input_description(kwargs)}
    return kwargs, spec


def make_inputs_from_npz(torch, case, model, path):
    """Load shared official-preprocessor inputs for the stock vision cases."""
    return stock_vision.make_inputs_from_npz(torch, case, model, path)


def preprocess_image(case, model_path, image_path, output_path):
    """Prepare one real image with its pinned official slow processor."""
    return stock_vision.preprocess_image(case, model_path, image_path, output_path)


def invoke(case, model, kwargs):
    """Call the actual imported model without changing the runtime or inputs."""
    _case(case)
    if case in stock_vision.CASES:
        return stock_vision.invoke(case, model, kwargs)
    return model.get_image_embeddings(**kwargs) if case == "sam2_image" else model(**kwargs)


def extract_output(case, output):
    """Keep all principal numerical outputs for independent NumPy comparison."""
    _case(case)
    if case in stock_vision.CASES:
        return stock_vision.extract_output(case, output)
    if case in ("wan", "sdxl"):
        return {"sample": output.sample}
    if case in ("siglip2", "dinov2"):
        return {"last_hidden_state": output.last_hidden_state, "pooler_output": output.pooler_output}
    if case == "sam2_image":
        return {f"image_embedding_{index}": value for index, value in enumerate(output)}
    result = {
        "pred_masks": output.pred_masks,
        "iou_scores": output.iou_scores,
        "object_score_logits": output.object_score_logits,
    }
    for index, value in enumerate(output.image_embeddings):
        result[f"image_embedding_{index}"] = value
    return result
