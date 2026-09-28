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


def _twice(call, seed=4242):
    """The same seeded call, once fresh and once after the process stream moved.

    The two results must be identical: that is what "the generator owns its own
    stream" means, and the global stream is advanced in between precisely
    because a rank's own history is what used to leak into the answer.
    """
    first = call(torch.Generator(device="cpu").manual_seed(seed))
    for _ in range(5):
        torch.randn(97)
    second = call(torch.Generator(device="cpu").manual_seed(seed))
    return first, second


def test_every_spelling_of_the_shape_draws_from_the_generators_stream():
    """`randn(2, 3)`, `randn((2, 3))`, `randn(t.shape)` and `randn(size=(2, 3))`
    are four spellings of one call, and `normal`/`randint` take their shape
    after their value arguments.

    Under the shim a `torch.Size` is a jittor `NanoVector`, which is neither a
    tuple nor a list, and `size=` arrives as a keyword -- so a shape reader that
    only knows tuples and loose ints silently hands those calls back to the
    *global* stream. That is the rank-dependent draw this file exists to rule
    out, and it is invisible: the tensor still has the right shape and plausible
    numbers.
    """
    like = torch.zeros(2, 3)
    spellings = {
        "randn(2, 3)": lambda g: torch.randn(2, 3, generator=g),
        "randn((2, 3))": lambda g: torch.randn((2, 3), generator=g),
        "randn(t.shape)": lambda g: torch.randn(like.shape, generator=g),
        "randn(size=(2, 3))": lambda g: torch.randn(size=(2, 3), generator=g),
        "rand(t.shape)": lambda g: torch.rand(like.shape, generator=g),
        "normal(0., 1., (2, 3))": lambda g: torch.normal(0.0, 1.0, (2, 3), generator=g),
        "normal(mean=, std=, size=)": lambda g: torch.normal(
            mean=0.0, std=1.0, size=(2, 3), generator=g),
        "randint(0, 9, (2, 3))": lambda g: torch.randint(0, 9, (2, 3), generator=g),
        "randint(9, (2, 3))": lambda g: torch.randint(9, (2, 3), generator=g),
    }
    for label, call in spellings.items():
        first, second = _twice(call)
        assert tuple(first.shape) == (2, 3), label
        np.testing.assert_array_equal(first.numpy(), second.numpy(), err_msg=label)

    # the four spellings of `randn(2, 3)` are one call, so they draw one answer
    randn = [spellings[label] for label in
             ("randn(2, 3)", "randn((2, 3))", "randn(t.shape)", "randn(size=(2, 3))")]
    answers = [call(torch.Generator(device="cpu").manual_seed(77)).numpy()
               for call in randn]
    for other in answers[1:]:
        np.testing.assert_array_equal(answers[0], other)


def test_a_generator_changes_the_stream_and_nothing_else():
    """The same call with and without `generator=`: only the numbers may differ.

    Shape, dtype and requires_grad belong to the caller, not to the stream. A
    latent that comes back float32 under a float16 default, an index
    permutation that comes back float, a `*_like` that drops its source's dtype
    and an inference tensor that suddenly requires grad are all silent -- and
    each one is a way for "draw from the generator's own stream" to return
    something the plain call never would.
    """
    like16 = torch.zeros(2, 3).to(torch.float16)
    calls = {
        "randn(2, 3)": lambda **kw: torch.randn(2, 3, **kw),
        "randn(())": lambda **kw: torch.randn((), **kw),
        "randn(size=(2, 3))": lambda **kw: torch.randn(size=(2, 3), **kw),
        "randn(t.shape)": lambda **kw: torch.randn(like16.shape, **kw),
        "rand_like(fp16)": lambda **kw: torch.rand_like(like16, **kw),
        "randn_like(fp32)": lambda **kw: torch.randn_like(torch.zeros(4), **kw),
        "randperm(6)": lambda **kw: torch.randperm(6, **kw),
        "randint(0, 9, (2, 3))": lambda **kw: torch.randint(0, 9, (2, 3), **kw),
        "normal(0., 1., (2, 3))": lambda **kw: torch.normal(0.0, 1.0, (2, 3), **kw),
    }
    for label, call in calls.items():
        plain = call()
        seeded = call(generator=torch.Generator(device="cpu").manual_seed(99))
        assert tuple(seeded.shape) == tuple(plain.shape), label
        assert seeded.dtype == plain.dtype, label
        assert seeded.requires_grad == plain.requires_grad, label


def test_the_default_dtype_still_decides_what_a_seeded_draw_returns():
    """`torch.set_default_dtype` is how a pipeline asks for fp16/bf16 latents."""
    torch.set_default_dtype(torch.float64)
    try:
        g = torch.Generator(device="cpu").manual_seed(5)
        assert torch.randn(4, generator=g).dtype == torch.randn(4).dtype
        assert torch.rand(4, generator=g).dtype == torch.float64
    finally:
        torch.set_default_dtype(torch.float32)
