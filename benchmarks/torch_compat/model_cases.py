"""Inference workloads built with public model APIs."""

from dataclasses import dataclass, field
from functools import partial
from inspect import signature
from pathlib import Path


@dataclass(frozen=True)
class ModelCase:
    group: str
    loader: str
    model: str
    scope: str
    shape: tuple = ()
    dtype: str = "float32"
    checkpoint: str = ""
    options: dict = field(default_factory=dict)
    masked_outputs: dict = field(default_factory=dict)


@dataclass
class PreparedModel:
    model: object
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    forward: object = None


MODEL_CASES = {
    name: ModelCase("cv", "timm", model, "classification", (1, 3, 224, 224), "float16")
    for name, model in (
        ("resnet18", "resnet18"),
        ("resnet50", "resnet50"),
        ("resnet101", "resnet101"),
        ("mobilenetv3-small", "mobilenetv3_small_100"),
        ("mobilenetv3-large", "mobilenetv3_large_100"),
        ("efficientnet-b0", "efficientnet_b0"),
        ("efficientnetv2-s", "tf_efficientnetv2_s"),
        ("convnext-tiny", "convnext_tiny"),
        ("vit-b16", "vit_base_patch16_224"),
        ("swin-tiny", "swin_tiny_patch4_window7_224"),
    )
}


def hf(group, cls, config, task, shape, *, dtype="float32", checkpoint="", masked_outputs=None, **config_options):
    return ModelCase(group, "transformers", cls, task, shape, dtype, checkpoint,
                     {"config_type": config, "config": config_options}, masked_outputs or {})


MODEL_CASES.update({
    "dinov2": hf("cv", "Dinov2Model", "dinov2", "image features", (1, 3, 224, 224),
                 hidden_size=384, num_attention_heads=6, image_size=224, patch_size=14),
    "dinov3": hf("cv", "DINOv3ViTModel", "dinov3_vit", "image features", (1, 3, 224, 224),
                 hidden_size=384, intermediate_size=1536, num_attention_heads=6, image_size=224),
    **{
        name: ModelCase("cv", "ultralytics", factory, "raw detector forward", (1, 3, 640, 640),
                        options={"yaml": config})
        for name, factory, config in (
            ("yolov8n", "YOLO", "yolov8n.yaml"),
            ("yolo11n", "YOLO", "yolo11n.yaml"),
            ("rt-detr", "RTDETR", "rtdetr-l.yaml"),
        )
    },
    **{
        name: ModelCase("cv", "torchvision", model, "detection with NMS", (3, 640, 640))
        for name, model in (("faster-rcnn", "fasterrcnn_resnet50_fpn"),
                            ("mask-rcnn", "maskrcnn_resnet50_fpn"))
    },
    "grounding-dino": hf("cv", "GroundingDinoForObjectDetection", "", "text-guided detection", (480, 640),
                         checkpoint="IDEA-Research/grounding-dino-tiny",
                         masked_outputs={"output.logits": -1, "output.enc_outputs_class": -1,
                                         "output.encoder_logits": -1, "output.enc_outputs_coord_logits": 1}),
    "owlv2": hf("cv", "Owlv2ForObjectDetection", "", "text-guided detection", (480, 640),
                 checkpoint="google/owlv2-base-patch16-ensemble"),
    "unet": ModelCase("cv", "diffusers", "UNet2DModel", "U-Net forward", (1, 3, 64, 64)),
    "mask2former": hf("cv", "Mask2FormerForUniversalSegmentation", "", "semantic mask prediction", (1, 3, 256, 256),
                       checkpoint="facebook/mask2former-swin-small-ade-semantic"),
    "sam": hf("cv", "SamModel", "", "image prompt segmentation", (512, 640),
              checkpoint="facebook/sam-vit-base"),
    "sam2": hf("cv", "Sam2Model", "", "image prompt segmentation", (512, 640),
               checkpoint="facebook/sam2.1-hiera-tiny"),
    "sam2-large": hf("cv", "Sam2Model", "", "image prompt segmentation", (512, 640),
                     checkpoint="facebook/sam2.1-hiera-large"),
    "sam3": hf("cv", "Sam3Model", "", "text-guided segmentation", (512, 640),
               checkpoint="facebook/sam3"),
    "bert": hf("nlp", "BertModel", "bert", "text encoder", (1, 512)),
    "roberta": hf("nlp", "RobertaModel", "roberta", "text encoder", (1, 512), max_position_embeddings=514),
    "modernbert": hf("nlp", "ModernBertModel", "modernbert", "long text encoder", (1, 2048),
                     attn_implementation="sdpa"),
    "bge-m3": hf("nlp", "XLMRobertaModel", "", "dense CLS embedding", (1, 512),
                 checkpoint="BAAI/bge-m3"),
    "qwen3-embedding": hf("nlp", "Qwen3Model", "", "dense last-token embedding", (1, 512),
                          dtype="bfloat16", checkpoint="Qwen/Qwen3-Embedding-0.6B"),
    "all-minilm-l6-v2": hf("nlp", "BertModel", "bert", "dense mean-pool embedding", (1, 128),
                          hidden_size=384, intermediate_size=1536, num_hidden_layers=6, num_attention_heads=12),
    "qwen3-0.6b": hf("nlp", "Qwen3ForCausalLM", "qwen3", "prefill", (1, 512), dtype="bfloat16",
                    vocab_size=151936, hidden_size=1024, intermediate_size=3072, num_hidden_layers=28,
                    num_attention_heads=16, num_key_value_heads=8, head_dim=128),
    "gemma4": hf("nlp", "Gemma4ForConditionalGeneration", "", "prefill", (1, 512), dtype="bfloat16",
                 checkpoint="google/gemma-4-E2B-it"),
    "clip": hf("multimodal", "CLIPModel", "clip", "image-text similarity", (1, 3, 224, 224),
               text_config={"hidden_size": 512, "intermediate_size": 2048, "num_attention_heads": 8,
                            "num_hidden_layers": 12, "vocab_size": 49408, "max_position_embeddings": 77},
               vision_config={"hidden_size": 768, "intermediate_size": 3072, "num_attention_heads": 12,
                              "num_hidden_layers": 12, "image_size": 224, "patch_size": 32}),
    "siglip2": hf("multimodal", "Siglip2Model", "", "image-text similarity", (224, 224),
                  checkpoint="google/siglip2-base-patch16-naflex"),
    "llava": hf("multimodal", "LlavaForConditionalGeneration", "", "vision-language prefill", (336, 336),
                dtype="bfloat16", checkpoint="llava-hf/llava-1.5-7b-hf"),
    "qwen-vl": hf("multimodal", "Qwen3VLForConditionalGeneration", "", "vision-language prefill", (336, 336),
                  dtype="bfloat16", checkpoint="Qwen/Qwen3-VL-2B-Instruct"),
    "flux": ModelCase("multimodal", "diffusers", "FluxTransformer2DModel", "compact denoising step",
                      (1, 1024, 64), "bfloat16"),
    "flux2": ModelCase("multimodal", "diffusers", "Flux2Transformer2DModel", "compact denoising step",
                       (1, 1024, 128), "bfloat16", options={"torch_emulate_precision_casts": True}),
    "sdxl": ModelCase("multimodal", "diffusers", "UNet2DConditionModel", "denoising step",
                      (1, 4, 96, 96), "float16", "stabilityai/stable-diffusion-xl-base-1.0 (unet)"),
    "sd3.5": ModelCase("multimodal", "diffusers", "SD3Transformer2DModel", "denoising step",
                       (1, 16, 64, 64), "bfloat16", "stabilityai/stable-diffusion-3.5-medium (transformer)"),
    "whisper": hf("audio", "WhisperForConditionalGeneration", "whisper", "speech encoder and decoder", (1, 80, 3000),
                   d_model=384, encoder_layers=4, decoder_layers=4, encoder_attention_heads=6,
                   decoder_attention_heads=6, encoder_ffn_dim=1536, decoder_ffn_dim=1536),
    "whisper-large-v3-turbo": hf("audio", "WhisperForConditionalGeneration", "", "speech encoder and decoder",
                                (1, 128, 3000), dtype="float16", checkpoint="openai/whisper-large-v3-turbo"),
    "wav2vec2": hf("audio", "Wav2Vec2Model", "wav2vec2", "speech features", (1, 16000),
                   num_conv_pos_embedding_groups=16),
    "hubert": hf("audio", "HubertModel", "hubert", "speech features", (1, 16000)),
    "encodec": hf("audio", "EncodecModel", "encodec", "audio encode and decode", (1, 1, 24000)),
    "bark": hf("audio", "BarkModel", "", "greedy speech synthesis", (1, 16),
               checkpoint="suno/bark-small"),
    "cosyvoice": ModelCase("audio", "cosyvoice", "AutoModel", "vocoder inference", (1, 80, 100),
                           checkpoint="FunAudioLLM/CosyVoice2-0.5B"),
    **{
        name: ModelCase("graph-time", "pyg", model, "three-layer graph forward", (8192, 65536, 128))
        for name, model in (("gcn", "GCN"), ("gat", "GAT"), ("graphsage", "GraphSAGE"))
    },
    "lstm": ModelCase("graph-time", "recurrent", "LSTM", "two-layer sequence forward", (8, 128, 256)),
    "gru": ModelCase("graph-time", "recurrent", "GRU", "two-layer sequence forward", (8, 128, 256)),
    "patchtst": hf("graph-time", "PatchTSTForPrediction", "patchtst", "forecast", (1, 512, 8),
                   context_length=512, num_input_channels=8, patch_length=16, patch_stride=8,
                   prediction_length=32),
    "chronos": ModelCase("graph-time", "chronos", "Chronos2Model", "quantile forecast", (4, 512),
                         checkpoint="amazon/chronos-2"),
})


class Inputs:
    def __init__(self, torch, seed, dtype, processed=None):
        import numpy as np

        self.torch = torch
        self.random = np.random.default_rng(seed)
        self.dtype = getattr(torch, dtype)
        self.tensors = []
        self.processed = processed

    def tensor(self, array, dtype):
        value = self.torch.from_numpy(array).to(device="cuda", dtype=dtype)
        self.tensors.append(value)
        return value

    def real(self, shape, dtype=None):
        return self.tensor(self.random.standard_normal(shape, dtype="float32"), dtype or self.dtype)

    def ids(self, shape, high, low=0):
        return self.tensor(self.random.integers(low, high, size=shape, dtype="int64"), self.torch.int64)

    def image(self, shape):
        from PIL import Image

        pixels = self.random.integers(0, 256, size=(*shape, 3), dtype="uint8")
        return Image.fromarray(pixels)

    def place(self, tensor):
        dtype = self.dtype if tensor.is_floating_point() else tensor.dtype
        value = tensor.to(device="cuda", dtype=dtype)
        self.tensors.append(value)
        return value


def timm_model(torch, case, assets, inputs):
    import timm

    kwargs = {"checkpoint_path": str(assets)} if assets else {}
    model = timm.create_model(case.model, pretrained=False, **kwargs)
    return PreparedModel(model, (inputs.real(case.shape),))


def torchvision_model(torch, case, assets, inputs):
    from torchvision.models import detection

    model = getattr(detection, case.model)(
        weights=None, weights_backbone=None, min_size=case.shape[-2], max_size=case.shape[-1],
        box_score_thresh=0.0, box_detections_per_img=20,
    )
    if assets:
        model.load_state_dict(torch.load(assets, map_location="cpu", weights_only=True))
    return PreparedModel(model, ([inputs.real(case.shape).sigmoid()],))


def ultralytics_model(torch, case, assets, inputs):
    import ultralytics

    model = getattr(ultralytics, case.model)(str(assets) if assets else case.options["yaml"]).model
    return PreparedModel(model, (inputs.real(case.shape).sigmoid(),))


def needs_processor(case):
    return (case.scope in ("text-guided detection", "text-guided segmentation", "image prompt segmentation",
                           "vision-language prefill") or case.model == "Siglip2Model")


def processor_batch(case, assets, inputs):
    if case.model == "Sam2Model":
        from transformers import Sam2Processor as factory
    else:
        from transformers import AutoProcessor as factory

    processor = factory.from_pretrained(assets, local_files_only=True)
    image = inputs.image(case.shape)
    if case.scope == "image prompt segmentation":
        point = [case.shape[1] // 2, case.shape[0] // 2]
        points, labels = ([[[point]]], [[[1]]]) if case.model == "Sam2Model" else ([[point]], [[1]])
        return processor(images=image, input_points=points, input_labels=labels, return_tensors="pt")
    if case.scope == "vision-language prefill":
        content = [{"type": "image", "image": image}, {"type": "text", "text": "Describe this image."}]
        prompt = processor.apply_chat_template([{"role": "user", "content": content}],
                                               tokenize=False, add_generation_prompt=True)
        return processor(text=[prompt], images=[image], return_tensors="pt")
    return processor(images=image, text=["a person"], return_tensors="pt")


def processor_inputs(torch, case, model, assets, inputs):
    supported = signature(model.forward).parameters
    if inputs.processed:
        import numpy as np

        with np.load(inputs.processed, allow_pickle=False) as arrays:
            return {key: inputs.place(torch.from_numpy(arrays[key])) for key in arrays.files if key in supported}
    batch = processor_batch(case, assets, inputs)
    return {key: inputs.place(value) for key, value in batch.items()
            if key in supported and isinstance(value, torch.Tensor)}


def transformer_model(torch, case, assets, inputs):
    import transformers

    factory = getattr(transformers, case.model)
    if assets:
        model = factory.from_pretrained(assets, local_files_only=True)
    else:
        config = transformers.AutoConfig.for_model(case.options["config_type"], **case.options["config"])
        model = factory(config)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    text_config = model.config.get_text_config()
    if case.scope.endswith("prefill") and hasattr(text_config, "use_cache"):
        text_config.use_cache = False
    kwargs = {}
    forward = None
    if needs_processor(case):
        kwargs = processor_inputs(torch, case, model, assets, inputs)
    elif case.scope in ("image features", "semantic mask prediction"):
        kwargs["pixel_values"] = inputs.real(case.shape)
    elif case.model == "CLIPModel":
        kwargs["pixel_values"] = inputs.real(case.shape)
        kwargs["input_ids"] = inputs.ids((case.shape[0], 77), text_config.vocab_size)
    elif case.scope == "speech encoder and decoder":
        kwargs["input_features"] = inputs.real(case.shape)
        kwargs["decoder_input_ids"] = inputs.ids((case.shape[0], 16), text_config.vocab_size)
    elif case.scope in ("speech features", "audio encode and decode"):
        kwargs["input_values"] = inputs.real(case.shape)
    elif case.scope == "forecast":
        kwargs["past_values"] = inputs.real(case.shape)
    elif case.scope == "greedy speech synthesis":
        import numpy as np

        max_length = model.generation_config.semantic_config["max_input_semantic_length"]
        batch, length = case.shape
        tokens = np.zeros((batch, max_length), dtype=np.int64)
        tokens[:, :length] = inputs.random.integers(0, 1000, size=case.shape, dtype=np.int64)
        mask = np.zeros_like(tokens)
        mask[:, :length] = 1
        kwargs.update(input_ids=inputs.tensor(tokens, torch.int64),
                      attention_mask=inputs.tensor(mask, torch.int64), do_sample=False,
                      semantic_max_new_tokens=32, fine_temperature=1.0)
        forward = model.generate
    else:
        kwargs["input_ids"] = inputs.ids(case.shape, text_config.vocab_size, low=3)
        if "attention_mask" in signature(model.forward).parameters:
            kwargs["attention_mask"] = inputs.place(torch.ones_like(kwargs["input_ids"]))
        if "logits_to_keep" in signature(model.forward).parameters:
            kwargs["logits_to_keep"] = 1
        if case.scope.startswith("dense "):
            def embedding(**batch):
                states = model(**batch).last_hidden_state
                if case.scope == "dense mean-pool embedding":
                    pooled = states.mean(dim=1)
                elif case.scope == "dense CLS embedding":
                    pooled = states[:, 0]
                else:
                    pooled = states[:, -1]
                return torch.nn.functional.normalize(pooled, dim=-1)
            forward = embedding
    return PreparedModel(model, kwargs=kwargs, forward=forward)


def diffusers_model(torch, case, assets, inputs):
    import diffusers
    import numpy as np

    factory = getattr(diffusers, case.model)
    if assets:
        model = factory.from_pretrained(assets, local_files_only=True)
    elif case.model == "UNet2DModel":
        model = factory(
            sample_size=64, block_out_channels=(64, 128, 256), layers_per_block=2,
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
        )
    else:
        options = dict(num_layers=2, num_single_layers=4, attention_head_dim=64, num_attention_heads=8,
                       in_channels=case.shape[-1], joint_attention_dim=768)
        if case.model == "FluxTransformer2DModel":
            options.update(axes_dims_rope=(16, 24, 24), pooled_projection_dim=768)
        else:
            options.update(axes_dims_rope=(16, 16, 16, 16))
        model = factory(**options)
    config = model.config
    values = inputs.real(case.shape)
    timestep = inputs.tensor(np.full((case.shape[0],), 0.5, dtype="float32"), inputs.dtype)
    if case.model == "UNet2DModel":
        return PreparedModel(model, (values, timestep))
    text_size = config.cross_attention_dim if case.model == "UNet2DConditionModel" else config.joint_attention_dim
    kwargs = {"encoder_hidden_states": inputs.real((case.shape[0], 77, text_size))}
    if case.model == "UNet2DConditionModel":
        text_size = config.projection_class_embeddings_input_dim - 6 * config.addition_time_embed_dim
        kwargs["added_cond_kwargs"] = {
            "text_embeds": inputs.real((case.shape[0], text_size)),
            "time_ids": inputs.tensor(np.array([[768, 768, 0, 0, 768, 768]], np.float32), inputs.dtype),
        }
    elif case.model != "Flux2Transformer2DModel":
        kwargs["pooled_projections"] = inputs.real((case.shape[0], config.pooled_projection_dim))
    if case.model.startswith("Flux"):
        axes = len(config.axes_dims_rope)
        image_ids = np.zeros((case.shape[1], axes), dtype="float32")
        image_ids[:, -2], image_ids[:, -1] = np.divmod(np.arange(case.shape[1]), 32)
        kwargs["img_ids"] = inputs.tensor(image_ids, inputs.dtype)
        kwargs["txt_ids"] = inputs.tensor(np.zeros((77, axes), dtype="float32"), inputs.dtype)
        needs_guidance = config.get("guidance_embeds", case.model == "Flux2Transformer2DModel")
        if needs_guidance:
            kwargs["guidance"] = inputs.tensor(np.full((case.shape[0],), 3.5, dtype="float32"), inputs.dtype)
    return PreparedModel(model, (values,), dict(kwargs, timestep=timestep))


def pyg_model(torch, case, assets, inputs):
    from torch_geometric.nn import models

    nodes, edges, width = case.shape
    model = getattr(models, case.model)(in_channels=width, hidden_channels=width, num_layers=3, out_channels=32)
    return PreparedModel(model, (inputs.real((nodes, width)), inputs.ids((2, edges), nodes)))


def recurrent_model(torch, case, assets, inputs):
    model = getattr(torch.nn, case.model)(case.shape[-1], case.shape[-1], num_layers=2, batch_first=True)
    args = (inputs.real(case.shape),)
    if case.model != "LSTM":
        return PreparedModel(model, args)

    def forward(x):
        sequence, (hidden, cell) = model(x)
        # PyTorch 2.11 Inductor can retain the singleton direction dimension.
        # The benchmark publishes the two unidirectional states in LSTM layout.
        state_shape = (model.num_layers, x.shape[0], model.hidden_size)
        return sequence, (hidden.reshape(state_shape), cell.reshape(state_shape))

    return PreparedModel(model, args, forward=forward)


def chronos_model(torch, case, assets, inputs):
    from chronos.chronos2.model import Chronos2Model

    model = Chronos2Model.from_pretrained(assets, local_files_only=True)
    return PreparedModel(model, kwargs={"context": inputs.real(case.shape), "num_output_patches": 2})


def cosyvoice_model(torch, case, assets, inputs):
    from cosyvoice.cli.cosyvoice import AutoModel

    reader = AutoModel(model_dir=str(assets), load_jit=False, load_trt=False, fp16=False)
    generator = reader.model.hift
    return PreparedModel(generator, kwargs={"speech_feat": inputs.real(case.shape)}, forward=generator.inference)


def prepare_model(torch, name, assets_root, seed, processed=None):
    from workloads import deterministic_init

    case = MODEL_CASES[name]
    assets = Path(assets_root) / name if assets_root else None
    if assets is not None and not assets.exists():
        assets = None
    if case.checkpoint and assets is None:
        raise FileNotFoundError(f"{name} requires a local snapshot of {case.checkpoint} at --model-assets/{name}")
    torch.manual_seed(seed)
    inputs = Inputs(torch, seed, case.dtype, processed)
    result = LOADERS[case.loader](torch, case, assets, inputs)
    model = result.model.eval()
    if assets is None:
        deterministic_init(torch, model, seed)
    model.to(device="cuda", dtype=inputs.dtype)
    forward = result.forward if result.forward is not None else model
    return partial(forward, *result.args, **result.kwargs), inputs.tensors, model, assets


LOADERS = {
    "timm": timm_model,
    "torchvision": torchvision_model,
    "ultralytics": ultralytics_model,
    "transformers": transformer_model,
    "diffusers": diffusers_model,
    "pyg": pyg_model,
    "recurrent": recurrent_model,
    "chronos": chronos_model,
    "cosyvoice": cosyvoice_model,
}
