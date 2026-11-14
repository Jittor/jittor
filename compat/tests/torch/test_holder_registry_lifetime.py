"""Repeated holder publication preserves weak cleanup and caller protocols."""
import gc
from unittest import mock
import weakref

from jittor.compat.torch.holder_registry import HolderRegistry


class IndependentHolder:
    _frontend_backend = object()


def test_republication_keeps_weak_ownership_and_public_callback_visibility():
    holder = IndependentHolder()
    key = id(holder)
    registry = HolderRegistry()
    registry[key] = holder
    observed = []
    first = weakref.ref(holder, lambda ref: observed.append(
        ("first", registry.get(key), list(registry.keys()))))
    second = weakref.ref(holder, lambda ref: observed.append(
        ("second", registry.get(key), list(registry.keys()))))
    for _ in range(3):
        registry[key] = holder
    assert registry[key] is holder and registry.is_weak(key)
    del holder
    gc.collect()
    assert first() is second() is None and len(registry) == 0
    assert observed == [("second", None, []), ("first", None, [])]


def test_expiration_cannot_remove_replacement_published_by_a_user_callback():
    holder, replacement = IndependentHolder(), IndependentHolder()
    key = id(holder)
    registry = HolderRegistry()
    registry[key] = holder
    reference = weakref.ref(holder, lambda ref: registry.__setitem__(key, replacement))
    registry[key] = holder
    del holder
    gc.collect()
    assert reference() is None and registry[key] is replacement
    del replacement
    gc.collect()
    assert len(registry) == 0


def test_republication_still_calls_a_custom_registry_setter():
    calls = []
    class Registry(HolderRegistry):
        def __setitem__(self, key, value):
            calls.append(key)
            return super().__setitem__(key, value)
    holder = IndependentHolder()
    registry = Registry()
    for _ in range(3):
        registry[id(holder)] = holder
    assert calls == [id(holder)] * 3


def test_custom_weakref_creator_is_not_skipped_for_an_existing_identity():
    registry, holder = HolderRegistry(), IndependentHolder()
    registry[id(holder)] = holder
    original = weakref.ref
    calls = []
    def create(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)
    with mock.patch.object(weakref, "ref", create):
        registry[id(holder)] = holder
        registry[id(holder)] = holder
    assert calls == [registry, holder, registry, holder]


def test_custom_identity_key_keeps_its_hash_protocol():
    calls = []
    class Key(int):
        def __hash__(self):
            calls.append(1)
            return int.__hash__(self)
    holder = IndependentHolder()
    key, registry = Key(id(holder)), HolderRegistry()
    registry[key] = holder
    registry[key] = holder
    assert calls == [1, 1]


def test_marker_changes_can_replace_a_weak_binding_with_strong_ownership():
    registry, holder = HolderRegistry(), IndependentHolder()
    key = id(holder)
    registry[key] = holder
    assert registry.is_weak(key)
    with mock.patch.object(IndependentHolder, "_frontend_backend", None):
        registry[key] = holder
        assert not registry.is_weak(key) and dict.get(registry, key) is holder
    registry[key] = holder
    assert registry.is_weak(key)
    reference = weakref.ref(holder)
    del holder
    gc.collect()
    assert reference() is None and len(registry) == 0


def test_legacy_values_keep_their_existing_strong_ownership():
    class LegacyHolder:
        pass
    holder = LegacyHolder()
    reference = weakref.ref(holder)
    key, registry = id(holder), HolderRegistry()
    registry[key] = holder
    registry[key] = holder
    del holder
    gc.collect()
    assert reference() is registry[key] and not registry.is_weak(key)
    registry.clear_legacy()
    gc.collect()
    assert reference() is None
