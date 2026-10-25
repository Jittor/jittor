"""Host function identities only, NOT model operators or numerical evidence."""


class Qwen3RMSNorm:
    def forward(self, value):
        raise AssertionError("host policy contracts must not execute RMSNorm")


class Qwen3MLP:
    def forward(self, value):
        raise AssertionError("host policy contracts must not execute MLP")


def apply_rotary_pos_emb(*args):
    raise AssertionError("host policy contracts must not execute RoPE")
