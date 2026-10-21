"""Real Transformers models for multimodal, sparse-expert and context coverage.

Builders use only public Torch and Transformers APIs and are identical in the
candidate and independent oracle processes. Configurations are deliberately
small; the context case still executes all 2,048 tokens.
"""


def with_pooler_output(torch, model):
    """Keep the encoder and its trainable pooler in the checked output graph."""
    class PooledEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = model

        def forward(self, **inputs):
            result = self.encoder(**inputs)
            return torch.cat((result.last_hidden_state.reshape(-1),
                              result.pooler_output.reshape(-1)))

    return PooledEncoder()


def clip_multimodal(torch):
    from transformers import CLIPConfig, CLIPModel

    config = CLIPConfig(
        text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4,
                         max_position_embeddings=16, attention_dropout=0.0,
                         pad_token_id=0, bos_token_id=1, eos_token_id=2),
        vision_config=dict(hidden_size=32, intermediate_size=64,
                           num_hidden_layers=2, num_attention_heads=4,
                           image_size=16, patch_size=8, attention_dropout=0.0),
        projection_dim=16,
    )

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.clip = CLIPModel(config)

        def forward(self, input_ids, pixel_values):
            return self.clip(input_ids=input_ids, pixel_values=pixel_values).logits_per_image

    return Model(), {"input_ids": ("int64", (2, 8), 64),
                     "pixel_values": ("float32", (2, 3, 16, 16), None)}


def mixtral_moe(torch):
    from transformers import MixtralConfig, MixtralForCausalLM

    config = MixtralConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                          num_hidden_layers=2, num_attention_heads=4,
                          num_key_value_heads=2, max_position_embeddings=128,
                          num_local_experts=4, num_experts_per_tok=2,
                          attention_dropout=0.0, router_jitter_noise=0.0,
                          output_router_logits=True, use_cache=False)
    return MixtralForCausalLM(config), {"input_ids": ("int64", (2, 32), 64)}


def llama_rope_2048(torch):
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=1, num_attention_heads=4,
                         num_key_value_heads=2, max_position_embeddings=512,
                         rope_scaling={"rope_type": "linear", "factor": 4.0},
                         attention_dropout=0.0, use_cache=False)
    return LlamaForCausalLM(config), {"input_ids": ("int64", (1, 2048), 64)}
