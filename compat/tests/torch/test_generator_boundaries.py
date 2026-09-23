"""Do not advertise a counter offset for an unrelated factory RNG stream."""
import numpy as np
import pytest


def test_factory_offset_replays_or_explicitly_rejects_unsupported_stream():
    import torch
    generator = torch.Generator(device='cuda').manual_seed(88)
    pristine = generator.get_state()
    before = generator.get_offset()
    expected = torch.randn((33,), device='cuda', generator=generator).cpu().numpy()
    if hasattr(torch, '_torch_compat_install_context'):
        with pytest.raises(NotImplementedError, match='factory'):
            generator.get_offset()
        with pytest.raises(NotImplementedError, match='factory'):
            generator.set_offset(before)
        advanced = generator.get_state()
        generator.manual_seed(88)
        assert generator.get_offset() == 0
        generator.set_state(advanced)
        with pytest.raises(NotImplementedError, match='factory'):
            generator.get_offset()
        generator.set_state(pristine)
        assert generator.get_offset() == before
        actual = torch.randn((33,), device='cuda', generator=generator).cpu().numpy()
    else:
        assert generator.get_offset() > before
        generator.set_offset(before)
        actual = torch.randn((33,), device='cuda', generator=generator).cpu().numpy()
    np.testing.assert_array_equal(actual, expected)
