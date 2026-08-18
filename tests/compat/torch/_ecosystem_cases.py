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


#: name -> (builder, required top-level distributions)
CASES = {
    "transformers_gpt2": (_tiny_gpt2, ("transformers",)),
    "transformers_llama": (_tiny_llama, ("transformers",)),
    "transformers_bert": (_tiny_bert, ("transformers",)),
    "transformers_vit": (_tiny_vit, ("transformers",)),
    "diffusers_unet2d": (_diffusers_unet, ("diffusers",)),
    "peft_lora_llama": (_peft_lora_llama, ("transformers", "peft")),
}
