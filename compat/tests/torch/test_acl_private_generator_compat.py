"""Real ACL private-owner draws; no SIMT/vLLM discard-unit acceptance.

These tests require an independently built matching ACL Core and externally
admitted device. They never substitute host RNG or a fake Tensor/provider.
"""
import gc
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import jittor as jt

from jittor._runtime.fallback import forbid_backend_fallbacks
from jittor.compat.torch.installers.factories import _native_generator


def array(value):
    return value.detach().cpu().numpy().copy()


def resident(value, ordinal):
    value.sync()
    assert isinstance(value, jt.Var)
    assert int(value.placement_backend) == 2 and int(value.device_id) == ordinal
    assert value.location() == 'device'
    return array(value)


@pytest.fixture
def acl():
    if not getattr(jt.compiler, 'has_acl', False):
        pytest.skip('requires real ACL Core/device')
    assert jt.compiler.build_config.backend == 'acl'
    assert jt.compiler.core_build_is_current()
    ordinal = max(int(jt.core.current_device()), 0)
    with jt.runtime.scope(use_cuda=1, device_id=ordinal, backend_fallback='error'), forbid_backend_fallbacks():
        before = int(jt.core.backend_fallback_count())
        state = torch.cuda.get_rng_state(ordinal)
        try:
            yield ordinal
            jt.sync_all(True)
            assert int(jt.core.backend_fallback_count()) == before
        finally:
            torch.cuda.set_rng_state(state, ordinal)


def draw(generator, ordinal, count=17):
    return torch.rand(count, dtype=torch.float32, device='npu:%d' % ordinal, generator=generator)


@pytest.mark.parametrize('spelling', ['npu', 'acl', 'cuda'])
def test_device_spellings_bind_real_acl_owner(acl, spelling):
    generator = torch.Generator(device='%s:%d' % (spelling, acl)).manual_seed(37)
    assert isinstance(generator._native, jt.Generator)
    assert generator._native.device == 'acl:%d' % acl
    assert generator.device.type == 'npu' and generator.device.index == acl
    output = resident(draw(generator, acl), acl)
    assert np.isfinite(output).all() and np.all((0 <= output) & (output < 1))
    assert generator.get_offset() == 17


def test_private_owner_interleaving_preserves_default_and_other_owner(acl):
    first = torch.Generator(device='npu:%d' % acl).manual_seed(193)
    other = torch.Generator(device='acl:%d' % acl).manual_seed(71)
    reference = torch.Generator(device='cuda:%d' % acl).manual_seed(193)
    torch.cuda.manual_seed(914)
    default_state = torch.cuda.get_rng_state(acl)
    expected_default = resident(torch.rand(11, device='cuda:%d' % acl), acl)
    torch.cuda.set_rng_state(default_state, acl)
    expected = [resident(draw(reference, acl), acl) for _ in range(2)]
    actual = [resident(draw(first, acl), acl)]
    resident(draw(other, acl, 33), acl)
    # Independent private draws have not advanced the process default stream.
    np.testing.assert_array_equal(resident(torch.rand(11, device='cuda:%d' % acl), acl), expected_default)
    actual.append(resident(draw(first, acl), acl))
    for got, wanted in zip(actual, expected):
        np.testing.assert_array_equal(got, wanted)
    assert first.get_offset() == reference.get_offset() == 34
    assert other.get_offset() == 33


@pytest.mark.parametrize('seed', [-1, (1 << 63) + 17, (1 << 64) - 1])
def test_full_seed_state_and_raw32_offset_replay(acl, seed):
    generator = torch.Generator(device='npu:%d' % acl).manual_seed(seed)
    assert generator.initial_seed() == seed & ((1 << 64) - 1)
    expected = resident(draw(generator, acl, 19), acl)
    assert generator.get_offset() == 19
    assert generator.set_offset(np.int64(0)) is generator
    np.testing.assert_array_equal(resident(draw(generator, acl, 19), acl), expected)
    state = generator.get_state()
    following = resident(draw(generator, acl, 13), acl)
    assert generator.set_state(state) is generator
    np.testing.assert_array_equal(resident(draw(generator, acl, 13), acl), following)
    assert generator.manual_seed(seed) is generator
    assert generator.get_offset() == 0
    np.testing.assert_array_equal(resident(draw(generator, acl, 19), acl), expected)


@pytest.mark.parametrize('bad,error', [(True, TypeError), (np.bool_(False), TypeError),
    (1.5, TypeError), ('3', TypeError), (-1, ValueError), (1 << 63, ValueError)])
def test_invalid_offset_is_atomic(acl, bad, error):
    generator = torch.Generator(device='npu:%d' % acl).manual_seed(27)
    resident(draw(generator, acl, 7), acl)
    before = array(generator.get_state())
    with pytest.raises(error):
        generator.set_offset(bad)
    np.testing.assert_array_equal(array(generator.get_state()), before)
    assert generator.get_offset() == 7


def test_invalid_state_is_atomic_and_cpu_frame_is_not_acl_state(acl):
    generator = torch.Generator(device='npu:%d' % acl).manual_seed(59)
    resident(draw(generator, acl, 9), acl)
    before = array(generator.get_state())
    corrupt = torch.tensor(np.concatenate((before, np.array([88], np.uint8))),
                           dtype=torch.uint8, device='cpu')
    for bad in (corrupt, torch.Generator(device='cpu').get_state(),
                torch.tensor([1], dtype=torch.int64, device='cpu')):
        with pytest.raises((ValueError, RuntimeError, TypeError)):
            generator.set_state(bad)
        np.testing.assert_array_equal(array(generator.get_state()), before)
        assert generator.initial_seed() == 59 and generator.get_offset() == 9


def test_empty_draw_and_split_draw_preserve_raw32_positions(acl):
    generator = torch.Generator(device='npu:%d' % acl).manual_seed(39)
    empty = draw(generator, acl, 0)
    empty.sync()
    assert tuple(empty.shape) == (0,) and int(empty.placement_backend) == 2
    assert generator.get_offset() == 0
    whole = resident(draw(generator, acl, 19), acl)
    generator.manual_seed(39)
    split = np.concatenate([resident(draw(generator, acl, n), acl) for n in (1, 3, 5, 10)])
    np.testing.assert_array_equal(split, whole)
    assert generator.get_offset() == 19


def test_pending_draw_keeps_seed_and_owner_lifetime(acl):
    generator = torch.Generator(device='npu:%d' % acl).manual_seed(81)
    pending = draw(generator, acl, 23)
    generator.manual_seed(5)
    # Re-seeding an owner cannot rewrite a previously issued draw.
    del generator
    gc.collect()
    reference = torch.Generator(device='npu:%d' % acl).manual_seed(81)
    np.testing.assert_array_equal(resident(pending, acl), resident(draw(reference, acl, 23), acl))
    temporary = draw(torch.Generator(device='npu:%d' % acl).manual_seed(81), acl, 23)
    gc.collect()
    np.testing.assert_array_equal(resident(temporary, acl), resident(draw(
        torch.Generator(device='npu:%d' % acl).manual_seed(81), acl, 23), acl))


@pytest.mark.parametrize('name,dtype', [('randn', 'float32'), ('rand', 'float16'),
                                      ('rand', 'float64'), ('rand', 'bfloat16')])
def test_unsupported_private_distribution_or_dtype_consumes_no_state(acl, name, dtype):
    generator = torch.Generator(device='npu:%d' % acl).manual_seed(15)
    before = array(generator.get_state())
    with pytest.raises(NotImplementedError, match='float32 uniform'):
        getattr(torch, name)(7, device='npu:%d' % acl, dtype=getattr(torch, dtype), generator=generator)
    np.testing.assert_array_equal(array(generator.get_state()), before)


def test_wrong_backend_and_spoof_owner_refused_without_consumption(acl):
    generator = torch.Generator(device='npu:%d' % acl).manual_seed(15)
    before = array(generator.get_state())
    with pytest.raises(RuntimeError, match='device type'):
        torch.rand(7, device='cpu', generator=generator)
    with pytest.raises(TypeError, match='torch.Generator'):
        torch.rand(7, device='npu:%d' % acl,
                   generator=SimpleNamespace(_native=generator._native))
    # Exercise ordinal matching without allocating on a second device. Placement
    # is a scoped metadata request; refusal occurs before any draw/allocation.
    token = jt.core._set_tensor_placement(2, acl + 1)
    try:
        with pytest.raises(RuntimeError, match='same ACL device'):
            _native_generator(generator)
    finally:
        jt.core._reset_tensor_placement(token)
    np.testing.assert_array_equal(array(generator.get_state()), before)


def test_exponential_is_inplace_on_real_acl_and_consumes_selected_owner(acl):
    generator = torch.Generator(device='npu:%d' % acl).manual_seed(331)
    target = torch.empty((2, 16), dtype=torch.float32, device='cuda:%d' % acl)
    target.sync()
    pointer = target.data_ptr()
    state = generator.get_state()
    assert target.exponential_(generator=generator) is target
    values = resident(target, acl)
    assert target.data_ptr() == pointer
    assert np.isfinite(values).all() and np.all(values >= 0)
    assert generator.get_offset() == 32
    generator.set_state(state)
    replay = torch.empty_like(target)
    replay.exponential_(generator=generator)
    np.testing.assert_array_equal(resident(replay, acl), values)


def test_historical_six_draw_reproducer_reaches_private_draw(acl, capsys, tmp_path):
    """The original diagnostic's explicit sixth draw must no longer refuse ACL."""
    path = Path(__file__).resolve().parents[3] / 'tools/probes/acl/private_generator_repro.py'
    spec = importlib.util.spec_from_file_location('acl_private_generator_repro', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.run(acl)
    raw = capsys.readouterr().out
    (tmp_path / 'private-generator-reproducer.json').write_text(raw)
    report = json.loads(raw)
    assert result == 0 and report['status'] == 'EXPLICIT_DRAW_RETURNED_NOT_FULL_CONTRACT'
    assert report['draw_calls'] == len(report['completed_draws']) == 6
    assert report['default_stream']['status'] == 'PASS'
    assert report['offset_interfaces'] == {'get_offset': True, 'set_offset': True}
    assert report['private_draw']['backend'] == 2 and report['private_draw']['ordinal'] == acl
    assert report['private_draw']['location'] == 'device'
    assert report['fallback_before'] == report['fallback_after'] == 0


def test_cpu_offsets_remain_explicitly_unsupported():
    for generator in (torch.default_generator, torch.Generator(device='cpu')):
        with pytest.raises(NotImplementedError, match='CPU Generator'):
            generator.get_offset()
        with pytest.raises(NotImplementedError, match='CPU Generator'):
            generator.set_offset(0)


@pytest.mark.parametrize('device', ['not_a_device', 'acl_typo:0', 'npu_typo:0', 'acl:-1'])
def test_unknown_generator_devices_rejected(device):
    with pytest.raises((ValueError, RuntimeError, NotImplementedError)):
        torch.Generator(device=device)


def test_offset_limit_and_draw_overflow_are_atomic(acl):
    generator = torch.Generator(device='npu:%d' % acl).manual_seed(23)
    assert generator.set_offset((1 << 63) - 1) is generator
    before = array(generator.get_state())
    with pytest.raises((RuntimeError, ValueError)):
        draw(generator, acl, 1).sync()
    np.testing.assert_array_equal(array(generator.get_state()), before)
    assert generator.get_offset() == (1 << 63) - 1


def test_cuda_offsets_remain_explicitly_unsupported():
    if jt.compiler.build_config.backend != 'cuda' or not torch.cuda.is_available():
        pytest.skip('requires real CUDA build/device')
    generator = torch.Generator(device='cuda')
    with pytest.raises(NotImplementedError, match='CUDA Generator'):
        generator.get_offset()
    with pytest.raises(NotImplementedError, match='CUDA Generator'):
        generator.set_offset(0)
