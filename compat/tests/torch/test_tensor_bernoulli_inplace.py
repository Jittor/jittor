"""Bernoulli fills preserve device streams, storage aliases and gradients."""
import numpy as np
import pytest
import torch

from _helpers.common import get_all_device_types


@pytest.fixture(params=get_all_device_types())
def device(request):
    return request.param


def array(tensor):
    return tensor.detach().cpu().to(dtype=torch.float64).numpy().copy()


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32,
                                   torch.float64, torch.int32, torch.int64, torch.bool])
def test_endpoints_dtype_device_and_alias_identity(device, dtype):
    x = torch.empty((4, 16), device=device, dtype=dtype)
    alias = x.view(8, 8)
    pointer = x.data_ptr()
    for probability in (0., 1.):
        assert x.bernoulli_(probability) is x
        assert x.data_ptr() == pointer and x.dtype == dtype
        assert x.device.type == device and alias.device.type == device
        np.testing.assert_array_equal(array(alias), probability)


def test_probability_tensor_broadcast_and_distribution(device):
    probabilities = torch.tensor([[0., .25, .75, 1.]], device=device)
    x = torch.empty((8192, 4), device=device)
    assert x.bernoulli_(probabilities) is x
    values = array(x)
    assert np.all((values == 0) | (values == 1))
    np.testing.assert_array_equal(values[:, 0], 0.)
    np.testing.assert_array_equal(values[:, 3], 1.)
    np.testing.assert_allclose(values[:, 1:3].mean(0), [.25, .75], atol=.025, rtol=0)


def test_generator_replay_and_global_stream_isolation(device):
    generator = torch.Generator(device=device).manual_seed(827)
    state = generator.get_state()
    torch.manual_seed(19)
    expected_global = array(torch.rand(31, device=device))
    torch.manual_seed(19)
    first = array(torch.empty(1024, device=device).bernoulli_(.4, generator=generator))
    advanced = array(torch.empty(1024, device=device).bernoulli_(.4, generator=generator))
    assert not np.array_equal(first, advanced)
    np.testing.assert_array_equal(array(torch.rand(31, device=device)), expected_global)
    generator.set_state(state)
    np.testing.assert_array_equal(array(torch.empty(1024, device=device).bernoulli_(.4, generator=generator)), first)


@pytest.mark.parametrize('probability', [-.1, 1.1, float('nan')])
def test_invalid_probability_does_not_mutate_or_advance_generator(device, probability):
    generator = torch.Generator(device=device).manual_seed(22)
    state = array(generator.get_state())
    x = torch.full((16,), 7., device=device)
    with pytest.raises(RuntimeError, match='p.*\\[0, 1\\]'):
        x.bernoulli_(probability, generator=generator)
    np.testing.assert_array_equal(array(x), 7.)
    np.testing.assert_array_equal(array(generator.get_state()), state)


@pytest.mark.parametrize('probability', [-.1, 1.1, float('nan')])
def test_invalid_probability_tensor_is_atomic(device, probability):
    generator = torch.Generator(device=device).manual_seed(23)
    state = array(generator.get_state())
    x = torch.full((2, 2), 7., device=device)
    p = torch.tensor([.5, probability], device=device)
    with pytest.raises(RuntimeError, match='p.*\\[0, 1\\]'):
        x.bernoulli_(p, generator=generator)
    np.testing.assert_array_equal(array(x), 7.)
    np.testing.assert_array_equal(array(generator.get_state()), state)


def test_integer_probability_tensor_is_rejected_before_mutation(device):
    x = torch.full((2,), 7., device=device)
    with pytest.raises(RuntimeError, match='floating'):
        x.bernoulli_(torch.tensor([0, 1], dtype=torch.int64, device=device))
    np.testing.assert_array_equal(array(x), 7.)


def test_saved_input_detects_a_no_grad_fill(device):
    x = torch.ones(8, device=device, requires_grad=True)
    loss = (x * x).sum()
    with torch.no_grad():
        x.bernoulli_(.5)
    with pytest.raises(RuntimeError, match='[Ii]n.place|modified|version'):
        loss.backward()


def test_view_fill_retains_pending_readers(device):
    base = torch.full((4, 8), 3., device=device)
    selected = base[:, ::2]
    before = (base * 2).clone()
    assert selected.bernoulli_(1.) is selected
    np.testing.assert_array_equal(array(before), 6.)
    values = array(base)
    np.testing.assert_array_equal(values[:, ::2], 1.)
    np.testing.assert_array_equal(values[:, 1::2], 3.)


def test_leaf_rejection_and_zero_gradient_for_overwritten_nonleaf(device):
    leaf = torch.ones(8, device=device, requires_grad=True)
    with pytest.raises(RuntimeError, match='leaf'):
        leaf.bernoulli_()
    x = leaf * 2
    x.bernoulli_(.5).sum().backward()
    np.testing.assert_array_equal(array(leaf.grad), 0.)


def test_probability_tensor_receives_zero_gradient(device):
    probability = torch.tensor([.2, .8], device=device, requires_grad=True)
    output = torch.empty((4, 2), device=device).bernoulli_(probability)
    assert output.requires_grad
    output.sum().backward()
    np.testing.assert_array_equal(array(probability.grad), 0.)


def test_cuda_graph_replay_refreshes_bernoulli_masks(device):
    if device != 'cuda':
        pytest.skip('CUDA graph replay requires real CUDA')

    def draw(x):
        return x.new_empty(x.shape).bernoulli_(.5)

    runner = torch.compile(draw, mode='reduce-overhead', dynamic=False)
    x = torch.ones(4096, device=device)
    samples = []
    for _ in range(9):
        torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad():
            value = runner(x)
        torch.cuda.synchronize()
        samples.append(array(value.clone()))
    assert all(not np.array_equal(a, b) for a, b in zip(samples[4:-1], samples[5:]))
    assert all(np.all((a == 0) | (a == 1)) for a in samples)
    if hasattr(runner, 'stats'):
        assert runner.stats['graph'] > 0, runner.stats
