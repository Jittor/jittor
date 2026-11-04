"""Stock Transformers vision workloads shared by native Torch and its shim.

This module never selects a runtime, changes a model implementation, moves a
tensor to an accelerator, or downloads a checkpoint. The caller supplies the
``torch`` module and an offline checkpoint. Random tiny models are compatibility
fixtures only: save them with native CPU Torch, then load the same checkpoint in
each runtime. Only the complete pretrained checkpoints represent model speed.

Source inspected: Transformers 4.56.2. ConvNeXtV2, SwinV2, and RT-DETRv2 expose no
SDPA selection in that release. Depth Anything delegates attention to its DINOv2
backbone. RT-DETRv2 uses its stock grid_sample implementation; there is no
disable_custom_kernels setting in its model code.
"""

import hashlib
from pathlib import Path

import numpy as np


CASES = ("convnextv2", "swinv2", "depth_anything_v2", "rtdetr_v2")
CHECKPOINTS = {
    "convnextv2": "facebook/convnextv2-tiny-1k-224",
    "swinv2": "microsoft/swinv2-tiny-patch4-window8-256",
    "depth_anything_v2": "depth-anything/Depth-Anything-V2-Small-hf",
    "rtdetr_v2": "PekingU/rtdetr_v2_r18vd",
}
FULL_IMAGE_SIZES = {
    "convnextv2": 224,
    "swinv2": 256,
    "depth_anything_v2": 518,
    "rtdetr_v2": 640,
}
TINY_IMAGE_SIZES = {
    "convnextv2": 64,
    "swinv2": 64,
    "depth_anything_v2": 56,
    "rtdetr_v2": 64,
}


def _case(case):
    if case not in CASES:
        raise ValueError("Unknown case {!r}; expected one of {}".format(case, CASES))
    return case


def _classes(case):
    from transformers import (
        ConvNextV2Config,
        ConvNextV2ForImageClassification,
        DepthAnythingConfig,
        DepthAnythingForDepthEstimation,
        RTDetrV2Config,
        RTDetrV2ForObjectDetection,
        Swinv2Config,
        Swinv2ForImageClassification,
    )

    return {
        "convnextv2": (ConvNextV2Config, ConvNextV2ForImageClassification),
        "swinv2": (Swinv2Config, Swinv2ForImageClassification),
        "depth_anything_v2": (DepthAnythingConfig, DepthAnythingForDepthEstimation),
        "rtdetr_v2": (RTDetrV2Config, RTDetrV2ForObjectDetection),
    }[_case(case)]


def _tiny_config(case, config_class):
    if case == "convnextv2":
        return config_class(
            image_size=64,
            hidden_sizes=[8, 16, 32, 64],
            depths=[1, 1, 1, 1],
            num_labels=7,
        )
    if case == "swinv2":
        # Retain four stages and alternating shifted/unshifted windows. A single
        # block would miss the attention-mask and torch.roll paths under test.
        return config_class(
            image_size=64,
            embed_dim=16,
            depths=[2, 2, 2, 2],
            num_heads=[2, 4, 8, 16],
            window_size=4,
            mlp_ratio=2.0,
            num_labels=7,
            drop_path_rate=0.0,
        )
    if case == "depth_anything_v2":
        from transformers import Dinov2Config

        backbone = Dinov2Config(
            image_size=56,
            patch_size=14,
            hidden_size=32,
            num_hidden_layers=4,
            num_attention_heads=4,
            mlp_ratio=2,
            out_indices=[1, 2, 3, 4],
            apply_layernorm=True,
            reshape_hidden_states=False,
        )
        return config_class(
            backbone_config=backbone,
            patch_size=14,
            reassemble_hidden_size=32,
            reassemble_factors=[4, 2, 1, 0.5],
            neck_hidden_sizes=[8, 16, 24, 32],
            fusion_hidden_size=16,
            head_hidden_size=8,
            depth_estimation_type="relative",
        )
    from transformers import RTDetrResNetConfig

    backbone = RTDetrResNetConfig(
        embedding_size=16,
        hidden_sizes=[16, 32, 64, 128],
        depths=[1, 1, 1, 1],
        layer_type="basic",
        out_indices=[2, 3, 4],
    )
    return config_class(
        backbone_config=backbone,
        encoder_hidden_dim=32,
        encoder_in_channels=[32, 64, 128],
        encoder_layers=1,
        encoder_ffn_dim=64,
        encoder_attention_heads=4,
        d_model=32,
        num_queries=10,
        decoder_in_channels=[32, 32, 32],
        decoder_ffn_dim=64,
        decoder_layers=2,
        decoder_attention_heads=4,
        num_labels=7,
        num_denoising=0,
        # The stock initializer also scales BatchNorm weights. At .01 this
        # small network collapses all 84 anchor scores to the identical FP32
        # bias, so its top-k output tests unspecified tie ordering. Use the
        # public initialization setting to make this numerical fixture useful;
        # native Torch still initializes and saves the one shared checkpoint.
        initializer_range=0.1,
    )


def create_model(torch, case, model_path=None, tiny=False, dtype=None, attn_implementation="eager"):
    """Load a complete offline model, or construct a tiny native CPU fixture.

    Supplying a path always loads its exact weights, including ``tiny=True``.
    ``sdpa`` is supported only for Depth Anything's DINOv2 backbone. The other
    three stock models implement their own eager computations in this pinned
    Transformers version, so this function rejects misleading SDPA requests.
    """
    _case(case)
    if attn_implementation not in ("eager", "sdpa"):
        raise ValueError("attn_implementation must be 'eager' or 'sdpa'")
    if case != "depth_anything_v2" and attn_implementation != "eager":
        raise ValueError("{} has no stock SDPA selection in Transformers 4.56.2".format(case))
    if model_path is None and not tiny:
        raise ValueError("Full models require a downloaded local model_path")

    config_class, model_class = _classes(case)
    config = (
        config_class.from_pretrained(str(model_path), local_files_only=True)
        if model_path is not None
        else _tiny_config(case, config_class)
    )
    # The public loader accepts per-subconfig attention choices. The top-level
    # depth head has no attention; selecting SDPA there would raise upstream.
    attention = (
        {"": "eager", "backbone_config": attn_implementation}
        if case == "depth_anything_v2"
        else "eager"
    )
    config._attn_implementation = attention
    if model_path is None:
        model = model_class(config)
        if dtype is not None:
            model = model.to(device="cpu", dtype=dtype)
    else:
        kwargs = dict(
            config=config,
            local_files_only=True,
            use_safetensors=True,
            low_cpu_mem_usage=False,
            attn_implementation=attention,
        )
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        model = model_class.from_pretrained(str(model_path), **kwargs)
    return model.eval()


def _description(kwargs):
    return {
        key: {"shape": list(value.shape), "dtype": str(value.dtype), "device": "cpu"}
        if hasattr(value, "shape") and hasattr(value, "dtype")
        else value
        for key, value in kwargs.items()
    }


def _spec(case, model, kwargs, tiny, input_kind, profile="default"):
    size = list(kwargs["pixel_values"].shape[-2:])
    component = {
        "convnextv2": "image_classification_backbone_and_head",
        "swinv2": "image_classification_backbone_and_head",
        "depth_anything_v2": "depth_backbone_neck_and_head",
        "rtdetr_v2": "detection_backbone_encoder_and_decoder",
    }[case]
    attention = "none" if case == "convnextv2" else "stock_eager"
    if case == "depth_anything_v2":
        attention = "dinov2_backbone_" + model.config.backbone_config._attn_implementation
    return dict(
        case=case,
        tiny=tiny,
        profile=profile,
        component=component,
        image_height_width=size,
        input_kind=input_kind,
        attention=attention,
        inputs=_description(kwargs),
    )


def make_inputs(torch, case, model, rng, tiny=False, profile="default"):
    """Make equal CPU arrays for full stock forwards, including prediction heads.

    The square full-resolution sizes are 224, 256, 518, and 640 respectively.
    These random inputs measure speed/numerical agreement, not semantic quality.
    Real image inputs should use :func:`make_inputs_from_npz` instead.
    """
    _case(case)
    if profile != "default":
        raise ValueError("New vision cases currently define only profile='default'")
    size = (TINY_IMAGE_SIZES if tiny else FULL_IMAGE_SIZES)[case]
    array = rng.standard_normal((1, 3, size, size)).astype(np.float32)
    dtype = next(model.parameters()).dtype
    kwargs = {"pixel_values": torch.tensor(array, dtype=dtype, device="cpu"), "return_dict": True}
    return kwargs, _spec(case, model, kwargs, tiny, "seeded_random", profile)


def make_inputs_from_npz(torch, case, model, path):
    """Load one shared, preprocessed real-image input without reprocessing it.

    The file must contain only the model inputs ``pixel_values`` and optionally
    ``pixel_mask`` (RT-DETRv2). Labels/target sizes belong in a separate metadata
    file, not in the timed forward. No floating-point data is round-tripped via
    one runtime before being passed to the other.
    """
    _case(case)
    with np.load(path, allow_pickle=False) as saved:
        keys = set(saved.files)
        allowed = {"pixel_values", "pixel_mask"} if case == "rtdetr_v2" else {"pixel_values"}
        if "pixel_values" not in keys or not keys.issubset(allowed):
            raise ValueError("Unexpected model input keys: {}".format(sorted(keys)))
        pixel_values = saved["pixel_values"]
        if (pixel_values.ndim != 4 or pixel_values.shape[0] != 1 or
                pixel_values.shape[1] != 3 or min(pixel_values.shape[-2:]) < 1 or
                not np.issubdtype(pixel_values.dtype, np.floating) or
                not np.isfinite(pixel_values).all()):
            raise ValueError("pixel_values must be one finite floating-point NCHW RGB image")
        if "pixel_mask" in keys:
            mask = saved["pixel_mask"]
            if mask.shape != (1, pixel_values.shape[2], pixel_values.shape[3]):
                raise ValueError("pixel_mask must match the image batch and spatial shape")
            if not np.isfinite(mask).all() or not np.isin(mask, [0, 1]).all():
                raise ValueError("pixel_mask must contain only zero and one")
        dtype = next(model.parameters()).dtype
        kwargs = {
            key: torch.tensor(
                saved[key], dtype=torch.int64 if key == "pixel_mask" else dtype, device="cpu"
            )
            for key in saved.files
        }
    kwargs["return_dict"] = True
    return kwargs, _spec(case, model, kwargs, False, "shared_preprocessed_image")


def preprocess_image(case, model_path, image_path, output_path):
    """Run the pinned official slow processor once, outside both timed runs.

    Run this preparation in the independent native CPU environment. It reads
    only local files and uses each checkpoint's official preprocessing defaults.
    Depth Anything preserves aspect ratio, so its real-image shape may differ
    from the fixed square 518 benchmark. The returned metadata records this.
    """
    _case(case)
    from PIL import Image
    from transformers import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained(
        str(model_path), local_files_only=True, use_fast=False
    )
    with Image.open(image_path) as source:
        rgb = source.convert("RGB")
        height, width = rgb.height, rgb.width
        processed = processor(images=rgb, return_tensors="np")
    allowed = {"pixel_values", "pixel_mask"} if case == "rtdetr_v2" else {"pixel_values"}
    if "pixel_values" not in processed or not set(processed).issubset(allowed):
        raise ValueError("Unexpected processor input keys: {}".format(sorted(processed)))
    arrays = {key: np.asarray(value) for key, value in processed.items()}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as stream:
        np.savez(stream, **arrays)
    return dict(
        case=case,
        model_path=str(Path(model_path).resolve()),
        image_sha256=hashlib.sha256(Path(image_path).read_bytes()).hexdigest(),
        archive_sha256=hashlib.sha256(output_path.read_bytes()).hexdigest(),
        original_height_width=[height, width],
        input_shapes={key: list(value.shape) for key, value in arrays.items()},
        processor_class=type(processor).__name__,
        processor_config=processor.to_dict(),
    )


def invoke(case, model, kwargs):
    """Invoke the stock model's complete inference forward."""
    _case(case)
    return model(**kwargs)


def extract_output(case, output):
    """Keep every principal prediction tensor, without filtering detections."""
    _case(case)
    if case in ("convnextv2", "swinv2"):
        return {"logits": output.logits}
    if case == "depth_anything_v2":
        return {"predicted_depth": output.predicted_depth}
    return {"logits": output.logits, "pred_boxes": output.pred_boxes}
