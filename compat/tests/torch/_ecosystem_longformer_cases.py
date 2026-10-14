"""Real Longformer at 32768 positions with local and first-token global attention.

The width and layer count are tiny; the full context length is exercised.
No downloads, adapter, replaced attention implementation or runtime branch.
"""
SEQUENCE_LENGTH = 32768
HIDDEN_SIZE = 32
ATTENTION_WINDOW = 256


def longformer_32768(torch):
    from transformers import LongformerConfig, LongformerModel

    config = LongformerConfig(
        vocab_size=64,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        attention_window=ATTENTION_WINDOW,
        max_position_embeddings=SEQUENCE_LENGTH + 2,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        attn_implementation="eager",
    )

    class LongformerWithPooler(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = LongformerModel(config, add_pooling_layer=True)

        def forward(self, input_ids, embedding_noise):
            assert tuple(input_ids.shape) == (1, SEQUENCE_LENGTH)
            assert tuple(embedding_noise.shape) == (1, SEQUENCE_LENGTH, HIDDEN_SIZE)
            embeddings = self.model.get_input_embeddings()(input_ids) + embedding_noise * 0.01
            attention_mask = torch.ones_like(input_ids)
            global_attention_mask = torch.zeros_like(input_ids)
            global_attention_mask[:, 0] = 1
            output = self.model(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                return_dict=True,
            )
            assert tuple(output.last_hidden_state.shape) == (1, SEQUENCE_LENGTH, HIDDEN_SIZE)
            assert tuple(output.pooler_output.shape) == (1, HIDDEN_SIZE)
            return torch.cat((output.last_hidden_state.reshape(-1), output.pooler_output.reshape(-1)))

    return LongformerWithPooler(), {
        "input_ids": ("int64", (1, SEQUENCE_LENGTH), 64),
        "embedding_noise": ("float32", (1, SEQUENCE_LENGTH, HIDDEN_SIZE), None),
    }

# Complete named-parameter contract of the pinned one-layer Longformer.
LONGFORMER_PARAMETER_NAMES = (
    'model.embeddings.LayerNorm.bias',
    'model.embeddings.LayerNorm.weight',
    'model.embeddings.position_embeddings.weight',
    'model.embeddings.token_type_embeddings.weight',
    'model.embeddings.word_embeddings.weight',
    'model.encoder.layer.0.attention.output.LayerNorm.bias',
    'model.encoder.layer.0.attention.output.LayerNorm.weight',
    'model.encoder.layer.0.attention.output.dense.bias',
    'model.encoder.layer.0.attention.output.dense.weight',
    'model.encoder.layer.0.attention.self.key.bias',
    'model.encoder.layer.0.attention.self.key.weight',
    'model.encoder.layer.0.attention.self.key_global.bias',
    'model.encoder.layer.0.attention.self.key_global.weight',
    'model.encoder.layer.0.attention.self.query.bias',
    'model.encoder.layer.0.attention.self.query.weight',
    'model.encoder.layer.0.attention.self.query_global.bias',
    'model.encoder.layer.0.attention.self.query_global.weight',
    'model.encoder.layer.0.attention.self.value.bias',
    'model.encoder.layer.0.attention.self.value.weight',
    'model.encoder.layer.0.attention.self.value_global.bias',
    'model.encoder.layer.0.attention.self.value_global.weight',
    'model.encoder.layer.0.intermediate.dense.bias',
    'model.encoder.layer.0.intermediate.dense.weight',
    'model.encoder.layer.0.output.LayerNorm.bias',
    'model.encoder.layer.0.output.LayerNorm.weight',
    'model.encoder.layer.0.output.dense.bias',
    'model.encoder.layer.0.output.dense.weight',
    'model.pooler.dense.bias',
    'model.pooler.dense.weight',
)
REQUIRED_GRADIENTS = tuple("grad::" + name for name in LONGFORMER_PARAMETER_NAMES) + (
    "ingrad::embedding_noise",
)
