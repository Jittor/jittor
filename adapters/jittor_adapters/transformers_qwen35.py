"""Partial Qwen3.5 eager-adapter retirement for Transformers 5.9.

Both upstream fallback sources remain version/source guarded. The chunk callable
is left untouched; only the functional recurrent replacement is installed.
The old chunk helper stays available for separately scoped historical controls.

This process-wide route change is not a new NPU, cached-recurrence or full-model
acceptance claim. Stub contracts do not prove actual model-instance bindings,
device numerics or complete PPO correctness; those need fresh device evidence.
"""
import hashlib
import inspect

from ._common import require_version, required_patch, replace, UnsupportedAdapterVersion


SUPPORTED_VERSIONS = frozenset(("5.9.0",))
SOURCE_SHA256 = {
    "torch_chunk_gated_delta_rule": "1066875a52cdd601582d9b3b8ebb31aa6a3d563de5dc34eb500ae860febbc7d0",
    "torch_recurrent_gated_delta_rule": "deb51ab402b0127b55e20faf565c33f06ff6b85735d729e4ec33d26751a5cea4",
}


def _verify_source(function, name):
    try:
        source = inspect.getsource(function)
    except (TypeError, OSError) as error:
        raise UnsupportedAdapterVersion("Qwen3.5 fallback source is required: " + name) from error
    if hashlib.sha256(source.encode("utf-8")).hexdigest() != SOURCE_SHA256[name]:
        raise UnsupportedAdapterVersion("Qwen3.5 fallback source changed: " + name)


def _functional_chunk(module):
    torch, F, l2norm = module.torch, module.F, module.l2norm

    def unit_lower_closure(strict_lower):
        """Return ``(I - strict_lower) ** -1`` without slice assignment.

        The upstream row recurrence computes this unit-lower-triangular
        closure in place.  Recursively applying the block inverse identity
        keeps the graph functional while replacing the token-serial recurrence
        with batched matrix multiplications.
        """
        size = strict_lower.shape[-1]
        if size == 1:
            return torch.ones_like(strict_lower)
        split = size // 2
        upper_inverse = unit_lower_closure(
            strict_lower[..., :split, :split]
        )
        lower_inverse = unit_lower_closure(
            strict_lower[..., split:, split:]
        )
        lower_left = (
            lower_inverse
            @ strict_lower[..., split:, :split]
            @ upper_inverse
        )
        upper = torch.cat(
            (upper_inverse, torch.zeros_like(strict_lower[..., :split, split:])),
            dim=-1,
        )
        lower = torch.cat((lower_left, lower_inverse), dim=-1)
        return torch.cat((upper, lower), dim=-2)

    def implementation(
        query,
        key,
        value,
        g,
        beta,
        chunk_size=64,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        **kwargs,
    ):
        initial_dtype = query.dtype
        if use_qk_l2norm_in_kernel:
            query = l2norm(query, dim=-1, eps=1e-6)
            key = l2norm(key, dim=-1, eps=1e-6)
        query, key, value, beta, g = [
            x.transpose(1, 2).contiguous().to(torch.float32)
            for x in (query, key, value, beta, g)
        ]

        batch_size, num_heads, sequence_length, k_head_dim = key.shape
        v_head_dim = value.shape[-1]
        # Padding a single short block to the upstream default of 64 only adds
        # zero key/value/beta rows and identity recurrence steps.  Shrink that
        # block to the real sequence length: this is mathematically identical,
        # avoids a needless 64x64 closure for short PPO samples, and leaves the
        # multi-block path unchanged.
        if 0 < sequence_length < chunk_size:
            chunk_size = sequence_length
        pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
        query = F.pad(query, (0, 0, 0, pad_size))
        key = F.pad(key, (0, 0, 0, pad_size))
        value = F.pad(value, (0, 0, 0, pad_size))
        beta = F.pad(beta, (0, pad_size))
        g = F.pad(g, (0, pad_size))
        total_sequence_length = sequence_length + pad_size
        query = query * (1 / (query.shape[-1] ** 0.5))

        v_beta = value * beta.unsqueeze(-1)
        k_beta = key * beta.unsqueeze(-1)
        query, key, value, k_beta, v_beta = [
            x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
            for x in (query, key, value, k_beta, v_beta)
        ]
        g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
        mask = torch.triu(
            torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
            diagonal=0,
        )

        g = g.cumsum(dim=-1)
        decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
        base_attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
        attn = unit_lower_closure(base_attn)
        value = attn @ v_beta
        k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
        last_recurrent_state = (
            torch.zeros(
                batch_size, num_heads, k_head_dim, v_head_dim,
                dtype=value.dtype, device=value.device,
            )
            if initial_state is None else initial_state.to(value)
        )

        outputs = []
        for index in range(total_sequence_length // chunk_size):
            q_i, k_i, v_i = query[:, :, index], key[:, :, index], value[:, :, index]
            local_attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, index]
            v_prime = k_cumdecay[:, :, index] @ last_recurrent_state
            v_new = v_i - v_prime
            attn_inter = (
                q_i * g[:, :, index, :, None].exp()
            ) @ last_recurrent_state
            outputs.append(attn_inter + local_attn @ v_new)
            last_recurrent_state = (
                last_recurrent_state * g[:, :, index, -1, None, None].exp()
                + (
                    k_i
                    * (g[:, :, index, -1, None] - g[:, :, index]).exp()[..., None]
                ).transpose(-1, -2) @ v_new
            )

        if not output_final_state:
            last_recurrent_state = None
        core_attn_out = torch.stack(outputs, dim=2)
        core_attn_out = core_attn_out.reshape(
            core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
        )
        core_attn_out = core_attn_out[:, :, :sequence_length]
        core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
        return core_attn_out, last_recurrent_state

    implementation.__name__ = "torch_chunk_gated_delta_rule"
    implementation._jittor_transformers_qwen35_functional = True
    return implementation


def _functional_recurrent(module):
    torch, l2norm = module.torch, module.l2norm

    def implementation(
        query, key, value, g, beta, initial_state, output_final_state,
        use_qk_l2norm_in_kernel=False,
    ):
        initial_dtype = query.dtype
        if use_qk_l2norm_in_kernel:
            query = l2norm(query, dim=-1, eps=1e-6)
            key = l2norm(key, dim=-1, eps=1e-6)
        query, key, value, beta, g = [
            x.transpose(1, 2).contiguous().to(torch.float32)
            for x in (query, key, value, beta, g)
        ]

        batch_size, num_heads, sequence_length, k_head_dim = key.shape
        v_head_dim = value.shape[-1]
        query = query * (1 / (query.shape[-1] ** 0.5))
        last_recurrent_state = (
            torch.zeros(
                batch_size, num_heads, k_head_dim, v_head_dim,
                dtype=value.dtype, device=value.device,
            )
            if initial_state is None else initial_state.to(value)
        )

        outputs = []
        for index in range(sequence_length):
            q_t, k_t, v_t = query[:, :, index], key[:, :, index], value[:, :, index]
            g_t = g[:, :, index].exp().unsqueeze(-1).unsqueeze(-1)
            beta_t = beta[:, :, index].unsqueeze(-1)
            last_recurrent_state = last_recurrent_state * g_t
            kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * beta_t
            last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            outputs.append((last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2))

        if not output_final_state:
            last_recurrent_state = None
        core_attn_out = torch.stack(outputs, dim=2)
        core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
        return core_attn_out, last_recurrent_state

    implementation.__name__ = "torch_recurrent_gated_delta_rule"
    implementation._jittor_transformers_qwen35_functional = True
    return implementation


@required_patch
def patch_qwen35(module):
    require_version("transformers", SUPPORTED_VERSIONS)
    originals = {
        name: getattr(module, name, None) for name in SOURCE_SHA256
    }
    for name, function in originals.items():
        if getattr(function, "_jittor_transformers_qwen35_functional", False):
            return False
        if not callable(function):
            raise UnsupportedAdapterVersion("Qwen3.5 fallback is not callable: " + name)
        _verify_source(function, name)

    replacements = {
        "torch_recurrent_gated_delta_rule": _functional_recurrent(module),
    }
    for name, replacement in replacements.items():
        replace(module, name, replacement, originals[name])
    return True


def register(register_module_patch):
    register_module_patch(
        "transformers.models.qwen3_5.modeling_qwen3_5", patch_qwen35
    )
