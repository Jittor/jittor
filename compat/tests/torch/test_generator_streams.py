"""torch.Generator owns its own stream.

Two `manual_seed(s)` generators must produce the same numbers, and a draw must
not depend on how much the *process* has already sampled. The H3 pipeline builds
its initial latents from a seeded CPU generator, and its DiT shards weights, so
every TP rank has to denoise the *same* latent: when drawing went through
jittor's global stream instead, two ranks whose streams had advanced differently
drew different latents, each RowParallelLinear added halves computed from
different inputs, and the TP2 picture came out as noise while TP1 was fine.
"""
import numpy as np
import torch


def test_a_generator_is_reproducible_and_independent_of_the_global_stream():
    seed = 1234
    a = torch.randn(8, generator=torch.Generator(device="cpu").manual_seed(seed))
    b = torch.randn(8, generator=torch.Generator(device="cpu").manual_seed(seed))
    np.testing.assert_array_equal(a.numpy(), b.numpy())

    c = torch.randn(8, generator=torch.Generator(device="cpu").manual_seed(seed + 1))
    assert not np.array_equal(a.numpy(), c.numpy())

    # advance this process's own stream, then ask again: the answer cannot move
    for _ in range(5):
        torch.randn(97)
    d = torch.randn(8, generator=torch.Generator(device="cpu").manual_seed(seed))
    np.testing.assert_array_equal(a.numpy(), d.numpy())


def test_the_pipelines_latent_call_shape_is_reproducible():
    """`torch.randn(*size, generator=g, dtype=...)` -- the pipeline's own call."""
    def draw():
        g = torch.Generator(device="cpu").manual_seed(11223)
        return torch.randn(1, 24, 4, 8, 8, generator=g, dtype=torch.float32)

    v1, v2 = draw(), draw()
    np.testing.assert_array_equal(v1.numpy(), v2.numpy())
    assert v1.dtype == torch.float32

    audio = torch.randn(
        66, 32, generator=torch.Generator(device="cpu").manual_seed(11223),
        dtype=torch.float32)
    assert tuple(audio.shape) == (66, 32)
