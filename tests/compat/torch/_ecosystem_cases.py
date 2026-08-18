"""Downstream-library cases run identically under real PyTorch and under Jittor.

Both runtimes claim the ``torch`` namespace, so a single process cannot hold
them at once.  Each case is therefore written once, against plain ``torch``
spellings, and executed twice in separate interpreters by
``_ecosystem_runner.py``: once where ``torch`` is real PyTorch and once where
``import torch`` resolves to Jittor's shim.

A case returns a model, its example inputs and a scalar loss builder.  The
runner takes care of seeding, weight transfer and serialization, so a case
never has to know which runtime it is running under.
"""


def _tiny_gpt2(torch):
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.for_model(
        "gpt2",
        n_layer=2,
        n_embd=64,
        n_head=2,
        vocab_size=128,
        n_positions=64,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    model = AutoModelForCausalLM.from_config(config)
    inputs = {"input_ids": ("int64", (2, 8), 128)}
    return model, inputs


def _tiny_llama(torch):
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.for_model(
        "llama",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=64,
        attention_dropout=0.0,
    )
    model = AutoModelForCausalLM.from_config(config)
    inputs = {"input_ids": ("int64", (2, 8), 128)}
    return model, inputs


def _tiny_bert(torch):
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.for_model(
        "bert",
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        vocab_size=128,
        max_position_embeddings=64,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    model = AutoModel.from_config(config)
    inputs = {"input_ids": ("int64", (2, 8), 128)}
    return model, inputs


def _tiny_vit(torch):
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.for_model(
        "vit",
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        image_size=32,
        patch_size=8,
        num_channels=3,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    model = AutoModel.from_config(config)
    inputs = {"pixel_values": ("float32", (2, 3, 32, 32), None)}
    return model, inputs


def _diffusers_unet(torch):
    from diffusers import UNet2DModel

    model = UNet2DModel(
        sample_size=16,
        in_channels=3,
        out_channels=3,
        layers_per_block=1,
        block_out_channels=(16, 32),
        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        norm_num_groups=8,
        dropout=0.0,
    )
    inputs = {
        "sample": ("float32", (2, 3, 16, 16), None),
        "timestep": ("int64", (2,), 100),
    }
    return model, inputs


def _peft_lora_llama(torch):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.for_model(
        "llama",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=64,
        attention_dropout=0.0,
    )
    base = AutoModelForCausalLM.from_config(config)
    lora = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora)
    inputs = {"input_ids": ("int64", (2, 8), 128)}
    return model, inputs


def _mmcv_conv_module(torch):
    """OpenMMLab's basic conv/norm/activation block.

    ``mmcv.ops`` is a compiled PyTorch extension and is out of scope for a
    Python-level shim, but ``mmcv.cnn`` builds its layers from ordinary
    ``torch.nn`` pieces, so its numerics are directly comparable.
    """
    from mmcv.cnn import ConvModule
    from torch import nn

    class Stack(nn.Module):
        def __init__(self):
            super().__init__()
            self.block1 = ConvModule(
                3, 16, 3, padding=1, norm_cfg=dict(type="BN"), act_cfg=dict(type="ReLU")
            )
            self.block2 = ConvModule(
                16, 16, 3, padding=1, norm_cfg=dict(type="GN", num_groups=4),
                act_cfg=dict(type="LeakyReLU"),
            )
            self.head = nn.Conv2d(16, 3, 1)

        def forward(self, sample):
            return self.head(self.block2(self.block1(sample)))

    inputs = {"sample": ("float32", (2, 3, 16, 16), None)}
    return Stack(), inputs


def _mmengine_base_model(torch):
    from mmengine.model import BaseModule
    from torch import nn

    class Head(BaseModule):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(32)
            self.fc1 = nn.Linear(32, 64)
            self.fc2 = nn.Linear(64, 32)

        def forward(self, sample):
            hidden = nn.functional.relu(self.fc1(self.norm(sample)))
            return self.fc2(hidden) + sample

    inputs = {"sample": ("float32", (2, 12, 32), None)}
    return Head(), inputs


def _transformers_t5(torch):
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.for_model(
        "t5",
        d_model=64,
        d_ff=128,
        d_kv=16,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=2,
        vocab_size=128,
        dropout_rate=0.0,
    )
    model = AutoModel.from_config(config)
    inputs = {
        "input_ids": ("int64", (2, 8), 128),
        "decoder_input_ids": ("int64", (2, 8), 128),
    }
    return model, inputs


def _transformers_whisper(torch):
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.for_model(
        "whisper",
        d_model=64,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=128,
        decoder_ffn_dim=128,
        vocab_size=128,
        num_mel_bins=16,
        max_source_positions=32,
        max_target_positions=32,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        # Whisper's defaults point at its 51865-token vocabulary; the shrunken
        # config needs special ids that exist inside the smaller embedding.
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
    )
    model = AutoModel.from_config(config)
    inputs = {
        "input_features": ("float32", (2, 16, 64), None),
        "decoder_input_ids": ("int64", (2, 8), 128),
    }
    return model, inputs


def _diffusers_transformer(torch):
    """A DiT-style latent transformer, the backbone modern diffusion uses."""
    from diffusers import DiTTransformer2DModel

    model = DiTTransformer2DModel(
        num_attention_heads=2,
        attention_head_dim=16,
        in_channels=4,
        num_layers=2,
        sample_size=8,
        patch_size=2,
        num_embeds_ada_norm=10,
        dropout=0.0,
    )
    inputs = {
        "hidden_states": ("float32", (2, 4, 8, 8), None),
        "timestep": ("int64", (2,), 10),
        "class_labels": ("int64", (2,), 10),
    }
    return model, inputs


#: name -> (builder, required top-level distributions)
CASES = {
    "transformers_gpt2": (_tiny_gpt2, ("transformers",)),
    "transformers_llama": (_tiny_llama, ("transformers",)),
    "transformers_bert": (_tiny_bert, ("transformers",)),
    "transformers_vit": (_tiny_vit, ("transformers",)),
    "transformers_t5": (_transformers_t5, ("transformers",)),
    "transformers_whisper": (_transformers_whisper, ("transformers",)),
    "diffusers_unet2d": (_diffusers_unet, ("diffusers",)),
    "diffusers_dit": (_diffusers_transformer, ("diffusers",)),
    "peft_lora_llama": (_peft_lora_llama, ("transformers", "peft")),
    "mmcv_conv_module": (_mmcv_conv_module, ("mmcv", "mmengine")),
    "mmengine_base_module": (_mmengine_base_model, ("mmengine",)),
}
