"""Cached launch flags must observe live environment changes and stay bounded."""
import pytest


@pytest.mark.parametrize('raw,truthy,falsey', [
    ('', False, False), ('  TRUE ', True, False), ('yes', True, False),
    ('1', True, False), ('off', False, True), (' 0 ', False, True),
    ('unknown', False, False),
])
def test_boolean_flags_follow_each_raw_value(monkeypatch, raw, truthy, falsey):
    from jittor.compat.triton import backend

    name = 'JITTOR_REVIEW_BOOLEAN_FLAG'
    monkeypatch.setenv(name, 'on')
    assert backend._truthy_env(name)
    monkeypatch.setenv(name, raw)
    assert backend._truthy_env(name) is truthy
    assert backend._falsey_env(name) is falsey
    monkeypatch.delenv(name)
    assert not backend._truthy_env(name)
    assert not backend._falsey_env(name)


def test_boolean_flag_cache_remains_bounded(monkeypatch):
    from jittor.compat.triton import backend

    monkeypatch.setattr(backend, '_ENV_FLAG_CACHE', {})
    for index in range(600):
        name = 'JITTOR_REVIEW_BOOLEAN_FLAG_%d' % index
        monkeypatch.setenv(name, 'on')
        assert backend._truthy_env(name)
        assert len(backend._ENV_FLAG_CACHE) <= 256
