"""Pinned vLLM decoder primitives through their public native-path modules.

This parameter-free inference composition verifies four input gradients. It
exercises neither a registered model family nor an engine/compiled extension.
"""
REQUIRED_GRADIENTS = (
    "ingrad::gates", "ingrad::hidden_states", "ingrad::query", "ingrad::key",
)


def decoder_primitives(torch):
    from vllm.config import (
        CompilationConfig, DeviceConfig, VllmConfig, set_current_vllm_config,
    )
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding

    def configuration(device):
        return VllmConfig(
            device_config=DeviceConfig(device=device),
            compilation_config=CompilationConfig(mode=0, custom_ops=["none"]),
        )

    class DecoderPrimitives(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Construction performs no inference. The runner migrates registered
            # buffers before forward; each call configures its actual input device.
            with set_current_vllm_config(configuration("cpu")):
                self.activation = SiluAndMul(compile_native=False)
                self.normalization = RMSNorm(16, has_weight=False, dtype=torch.float32)
                self.rotary = RotaryEmbedding(8, 8, 32, 10000., True, torch.float32)
            self._configurations = {}

        def forward(self, gates, hidden_states, query, key, positions):
            if tuple(gates.shape) != (5, 32) or any(
                tuple(value.shape) != (5, 16) for value in (hidden_states, query, key)
            ) or tuple(positions.shape) != (5,):
                raise ValueError("vLLM primitive input shapes differ from the fixed protocol")
            device = str(gates.device).split(":", 1)[0]
            if device not in ("cpu", "cuda"):
                raise ValueError("vLLM primitive gate is verified only on CPU and CUDA")
            if device not in self._configurations:
                self._configurations[device] = configuration(device)
            with set_current_vllm_config(self._configurations[device]):
                activated = self.activation(gates)
                normalized = self.normalization(hidden_states)
                rotated_query, rotated_key = self.rotary(positions, query, key)
            return torch.cat(tuple(value.reshape(-1) for value in (
                activated, normalized, rotated_query, rotated_key,
            )))

    return DecoderPrimitives(), {
        "gates": ("float32", (5, 32), None),
        "hidden_states": ("float32", (5, 16), None),
        "query": ("float32", (5, 16), None),
        "key": ("float32", (5, 16), None),
        "positions": ("int64", (5,), 32),
    }
