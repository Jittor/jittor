"""An explicit Generator has to be the one a sampler draws from.

`RandomSampler(..., generator=g)` promises that every index came from `g`, that
`g` advanced by exactly those draws, and that restoring `g`'s state replays
them. A sampler that quietly draws from somewhere else satisfies none of that
while looking fine on a first run -- the failure only shows up after a resume,
as a different data order.

These are the reproductions from the "显式 Generator 配合有放回采样" section of
the C5 issue doc, written against the CPU contract that section's option A
recommends settling first.
"""
import random

import torch
from torch.utils.data import RandomSampler


def test_replacement_sampler_replays_from_a_restored_generator():
    g = torch.Generator(device="cpu").manual_seed(777)
    state = g.get_state()
    first = list(RandomSampler(range(8), replacement=True, num_samples=12, generator=g))

    g.set_state(state)
    second = list(RandomSampler(range(8), replacement=True, num_samples=12, generator=g))
    assert first == second


def test_replacement_sampler_does_not_touch_the_global_stream():
    random.seed(4242)
    before = [random.random() for _ in range(4)]

    random.seed(4242)
    list(RandomSampler(range(8), replacement=True, num_samples=12,
                       generator=torch.Generator(device="cpu").manual_seed(1)))
    after = [random.random() for _ in range(4)]
    assert before == after


def test_two_generators_with_the_same_seed_sample_the_same_indices():
    a = list(RandomSampler(range(16), replacement=True, num_samples=10,
                           generator=torch.Generator(device="cpu").manual_seed(5)))
    b = list(RandomSampler(range(16), replacement=True, num_samples=10,
                           generator=torch.Generator(device="cpu").manual_seed(5)))
    assert a == b


def test_a_generator_advances_across_successive_samplers():
    g = torch.Generator(device="cpu").manual_seed(99)
    first = list(RandomSampler(range(16), replacement=True, num_samples=10, generator=g))
    second = list(RandomSampler(range(16), replacement=True, num_samples=10, generator=g))
    assert first != second, "the second draw restarted the stream"


def test_without_replacement_also_honours_the_generator():
    g = torch.Generator(device="cpu").manual_seed(31)
    state = g.get_state()
    first = list(RandomSampler(range(12), generator=g))
    g.set_state(state)
    second = list(RandomSampler(range(12), generator=g))
    assert first == second
    assert sorted(first) == list(range(12))


def test_generator_state_round_trips():
    g = torch.Generator(device="cpu").manual_seed(2024)
    torch.randint(0, 100, (5,), generator=g)          # advance it
    state = g.get_state()
    after_state = torch.randint(0, 100, (5,), generator=g).tolist()

    g.set_state(state)
    assert torch.randint(0, 100, (5,), generator=g).tolist() == after_state
