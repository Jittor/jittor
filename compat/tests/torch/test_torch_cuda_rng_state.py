"""The CUDA RNG state is refused, not faked.

`get_rng_state` returned the constant `[0]` and `set_rng_state` did nothing, so
`accelerator.save_state()` wrote a byte that meant nothing, `load_state()`
restored nothing, and the resumed run drew a different sequence than the one it
was continuing -- silently. A refusal is worse for the caller and better for
the training run.

From the "问题一：CUDA RNG 的完整状态保存和恢复" section of the C5 issue doc.
What is *not* claimed here is that the state is expressible: it is not, and the
reason is in the message.
"""
import pytest
import torch


def test_getting_the_cuda_rng_state_says_it_cannot():
    with pytest.raises(NotImplementedError) as caught:
        torch.cuda.get_rng_state()
    assert "manual_seed" in str(caught.value)


def test_setting_the_cuda_rng_state_says_it_cannot():
    with pytest.raises(NotImplementedError):
        torch.cuda.set_rng_state(torch.zeros(8, dtype=torch.uint8))


def test_get_rng_state_all_and_set_rng_state_all_refuse_too():
    with pytest.raises(NotImplementedError):
        torch.cuda.get_rng_state_all()
    with pytest.raises(NotImplementedError):
        torch.cuda.set_rng_state_all([])


def test_initial_seed_reports_the_seed_in_use():
    torch.cuda.manual_seed(4321)
    assert torch.cuda.initial_seed() == 4321


def test_seed_actually_reseeds():
    torch.cuda.manual_seed(11)
    torch.cuda.seed()
    assert torch.cuda.initial_seed() != 11


def test_manual_seed_still_gives_a_reproducible_sequence():
    import jittor as jt
    torch.cuda.manual_seed(7)
    first = jt.random((8,)).numpy().copy()
    torch.cuda.manual_seed(7)
    assert (jt.random((8,)).numpy() == first).all()
