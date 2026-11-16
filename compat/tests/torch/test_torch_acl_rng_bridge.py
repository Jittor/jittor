"""CUDA-spelled compat RNG APIs own independent ACL device streams."""
import numpy as np
import pytest
import torch
import jittor as jt
from jittor.compat.torch.installers import core
from jittor.compat.torch.installers.cuda import api
from jittor._runtime.fallback import forbid_backend_fallbacks


def test_fork_rng_rejects_an_unrestorable_device(monkeypatch):
    class Unsupported:
        def get_rng_state(self, device):
            raise NotImplementedError('no device snapshot')
    monkeypatch.setattr(core._ForkRng, '_accelerator', lambda self: Unsupported())
    before = torch.get_rng_state().numpy().copy()
    with pytest.raises(NotImplementedError, match='no device snapshot'):
        with torch.random.fork_rng(devices=[0]):
            pytest.fail('an incomplete snapshot must not enter the block')
    np.testing.assert_array_equal(torch.get_rng_state().numpy(), before)


@pytest.mark.parametrize('name', ['manual_seed', 'manual_seed_all'])
def test_missing_cuda_rng_backend_does_not_reseed_cpu(monkeypatch, name):
    monkeypatch.setattr(api, 'device_count', lambda: 1)
    monkeypatch.setattr(api, '_uses_acl_rng', lambda: False)
    def unavailable(required=True):
        raise NotImplementedError('no CUDA RNG backend')
    monkeypatch.setattr(api, '_curand', unavailable)
    before = torch.get_rng_state().numpy().copy()
    with pytest.raises(NotImplementedError, match='no CUDA RNG backend'):
        getattr(torch.cuda, name)(123)
    np.testing.assert_array_equal(torch.get_rng_state().numpy(), before)


@pytest.mark.skipif(not getattr(jt.compiler, 'has_acl', False), reason='requires ACL')
@pytest.mark.parametrize('seed', [17, (1 << 63) + 3, (1 << 64) - 1])
def test_acl_rng_state_and_nested_fork_restore_real_device(seed):
    with jt.flag_scope(use_acl=1, use_cuda=1, backend_fallback='error'), forbid_backend_fallbacks():
        cpu = torch.get_rng_state().numpy().copy()
        torch.cuda.manual_seed(seed)
        assert torch.cuda.initial_seed() == seed
        assert api.cuda_rng_state_is_supported()
        torch.rand(5, device='cuda').sync()
        before = torch.cuda.get_rng_state()
        expected = torch.rand(31, device='cuda')
        assert expected.location() == 'device'
        expected = expected.cpu().numpy().copy()
        torch.cuda.set_rng_state(before)
        with torch.random.fork_rng(devices=[torch.device('cuda:0')]):
            torch.rand(11, device='cuda').sync()
            with pytest.raises(ValueError, match='inside'):
                with torch.random.fork_rng(devices=[0]):
                    torch.rand(7, device='cuda').sync()
                    raise ValueError('inside')
        actual = torch.rand(31, device='cuda')
        assert actual.location() == 'device'
        np.testing.assert_array_equal(actual.cpu().numpy(), expected)
        np.testing.assert_array_equal(torch.get_rng_state().numpy(), cpu)
        torch.cuda.manual_seed_all(seed)
        states = torch.cuda.get_rng_state_all()
        torch.rand(9, device='cuda').sync()
        torch.cuda.set_rng_state_all(states)
        for got, saved in zip(torch.cuda.get_rng_state_all(), states):
            np.testing.assert_array_equal(got.numpy(), saved.numpy())
        bad = torch.zeros_like(states[0])
        with pytest.raises((ValueError, RuntimeError)):
            torch.cuda.set_rng_state(bad)
        np.testing.assert_array_equal(torch.cuda.get_rng_state().numpy(), states[0].numpy())
