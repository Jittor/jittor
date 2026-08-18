"""Realistic-size downstream models, for the speed half of the 2.0 goals.

The parity cases in ``_ecosystem_cases`` are deliberately tiny: correctness is an
operator-coverage question, and a two-layer model covers the same operators as a
thirty-layer one while keeping the gate fast. Wall-clock is a different
question. At batch 2 and hidden size 64 a step is dominated by Python dispatch
and kernel launch, so the ratio between two frameworks says almost nothing about
their kernels.

These configurations are sized so the accelerator is actually the bottleneck.
They are measured, never asserted for numerical parity -- that job belongs to
the small cases, which compare every gradient.
"""


def _llama(torch):
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.for_model(
        "llama",
        hidden_size=1024,
        intermediate_size=2816,
        num_hidden_layers=8,
        num_attention_heads=16,
        num_key_value_heads=16,
        vocab_size=32000,
        max_position_embeddings=1024,
        attention_dropout=0.0,
    )
    model = AutoModelForCausalLM.from_config(config)
    return model, {"input_ids": ("int64", (4, 512), 32000)}


def _gpt2(torch):
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.for_model(
        "gpt2",
        n_layer=8,
        n_embd=1024,
        n_head=16,
        vocab_size=32000,
        n_positions=1024,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    model = AutoModelForCausalLM.from_config(config)
    return model, {"input_ids": ("int64", (4, 512), 32000)}


def _bert(torch):
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.for_model(
        "bert",
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        vocab_size=30522,
        max_position_embeddings=512,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    model = AutoModel.from_config(config)
    return model, {"input_ids": ("int64", (8, 256), 30522)}


def _vit(torch):
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.for_model(
        "vit",
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        image_size=224,
        patch_size=16,
        num_channels=3,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    model = AutoModel.from_config(config)
    return model, {"pixel_values": ("float32", (8, 3, 224, 224), None)}


def _diffusers_unet(torch):
    from diffusers import UNet2DModel

    model = UNet2DModel(
        sample_size=64,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 384),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        norm_num_groups=32,
        dropout=0.0,
    )
    return model, {
        "sample": ("float32", (4, 3, 64, 64), None),
        "timestep": ("int64", (4,), 1000),
    }


def _resnet_like(torch):
    """A convolution stack, the shape Jittor's oneDNN/cuDNN relays serve."""
    from torch import nn

    class Stack(nn.Module):
        def __init__(self):
            super().__init__()
            channels = (64, 128, 256, 512)
            layers = [nn.Conv2d(3, channels[0], 7, stride=2, padding=3)]
            previous = channels[0]
            for width in channels:
                layers += [
                    nn.BatchNorm2d(previous),
                    nn.ReLU(),
                    nn.Conv2d(previous, width, 3, padding=1),
                    nn.BatchNorm2d(width),
                    nn.ReLU(),
                    nn.Conv2d(width, width, 3, padding=1),
                ]
                previous = width
            self.body = nn.Sequential(*layers)
            self.head = nn.Conv2d(previous, 10, 1)

        def forward(self, sample):
            return self.head(self.body(sample))

    return Stack(), {"sample": ("float32", (8, 3, 128, 128), None)}


#: name -> (builder, required top-level distributions)
CASES = {
    "large_transformers_llama": (_llama, ("transformers",)),
    "large_transformers_gpt2": (_gpt2, ("transformers",)),
    "large_transformers_bert": (_bert, ("transformers",)),
    "large_transformers_vit": (_vit, ("transformers",)),
    "large_diffusers_unet2d": (_diffusers_unet, ("diffusers",)),
    "large_convnet": (_resnet_like, ()),
}
