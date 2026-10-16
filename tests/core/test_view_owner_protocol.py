"""Native wrapper ownership is optional, GC-visible and independent of Torch."""

import gc
import sys
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import jittor as jt
from jittor._runtime.fallback import forbid_backend_fallbacks
from _helpers.device_types import instantiate_device_type_tests


class OwningTensor(jt.Var):
    _frontend_view_base_ownership = True


class NonOwningTensor(jt.Var):
    pass


def view(source):
    axes = jt.NanoVector([1, 0])
    return source._storage_permute(axes)._set_transpose_view_of(source, axes)


def physical(value, device):
    value.sync()
    assert value.location() == ('cpu' if device == 'cpu' else 'device')
    return value.numpy()


@pytest.fixture(autouse=True)
def native_only():
    assert 'torch' not in sys.modules and 'jittor.compat.torch' not in sys.modules
    with forbid_backend_fallbacks():
        yield
        jt.sync_all(True)


class TestViewOwnerProtocol:
    def test_optional_root_ownership_is_not_native_global_policy(self, device):
        assert jt.flags.transpose_storage_view == 0
        for tensor_type, retains in ((NonOwningTensor, False), (OwningTensor, True)):
            source = tensor_type(np.arange(6, dtype='float32').reshape(2, 3))
            alias = view(source)
            root = weakref.ref(source)
            assert alias._view_base_object() is source
            np.testing.assert_array_equal(physical(alias, device), [[0, 3], [1, 4], [2, 5]])
            del source
            gc.collect()
            assert (root() is not None) == retains
            assert alias._has_view_provenance()
            if retains:
                assert alias._view_base_object() is root()
            else:
                assert alias._view_base_object() is None
            del alias
            gc.collect()
            assert root() is None
        assert jt.flags.transpose_storage_view == 0

    def test_factory_conversion_constructor_and_temporary_holders(self, device):
        token = jt.core._set_tensor_frontend_type(OwningTensor)
        try:
            source = jt.array(np.arange(6, dtype='float32').reshape(2, 3))
            assert type(source) is OwningTensor
            alias = view(source)
            nested = view(alias)
            assert type(alias) is OwningTensor and nested._view_base_object() is source
            source_ref = weakref.ref(source)
            del source, alias
            gc.collect()
            assert source_ref() is not None
            np.testing.assert_array_equal(physical(nested, device), np.arange(6).reshape(2, 3))
            del nested
            gc.collect()
            assert source_ref() is None
        finally:
            jt.core._reset_tensor_frontend_type(token)
        assert type(jt.array([1.])) is jt.Var

    def test_gc_clear_handles_root_alias_and_type_cycles(self, device):
        for _ in range(3):
            class Tensor(OwningTensor):
                pass
            source = Tensor(np.arange(6, dtype='float32').reshape(2, 3))
            alias = view(source)
            physical(alias, device)
            source.alias = alias
            Tensor.instance = alias
            refs = tuple(weakref.ref(value) for value in (source, alias, Tensor))
            del source, alias, Tensor
            gc.collect()
            jt.gc()
            assert all(ref() is None for ref in refs)

    def test_reentrant_finalizer_sees_a_detached_record(self, device):
        observations = []
        alias_ref = None

        class FinalizingTensor(OwningTensor):
            def __del__(self):
                alias = alias_ref()
                observations.append(alias is None or alias._view_base_object() is None)

        source = FinalizingTensor(np.arange(6, dtype='float32').reshape(2, 3))
        alias = view(source)
        alias_ref = weakref.ref(alias)
        # Inspect residency before any numpy() call explicitly stages host data.
        alias.sync()
        assert alias.location() == ('cpu' if device == 'cpu' else 'device')
        del source
        # Replacing provenance with a self-view clears the old root first.
        alias._set_view_of(alias, Ellipsis)
        gc.collect()
        assert observations == [True]
        np.testing.assert_array_equal(physical(alias, device), [[0, 3], [1, 4], [2, 5]])

    def test_invalid_metadata_does_not_poison_next_conversion(self, device):
        class Invalid(jt.Var):
            _frontend_view_base_ownership = 1

        with pytest.raises(RuntimeError, match='ownership must be a bool'):
            Invalid([1., 2.])
        token = jt.core._set_tensor_frontend_type(Invalid)
        try:
            with pytest.raises(RuntimeError, match='ownership must be a bool'):
                jt.array([1., 2.])
        finally:
            jt.core._reset_tensor_frontend_type(token)
        source = OwningTensor(np.arange(6, dtype='float32').reshape(2, 3))
        alias = view(source)
        assert alias._view_base_object() is source
        np.testing.assert_array_equal(physical(alias, device), [[0, 3], [1, 4], [2, 5]])

    def test_native_copy_default_keeps_logical_order(self, device):
        source = jt.array(np.arange(6, dtype='float32').reshape(2, 3))
        alias = view(source)
        copied = jt.Var.copy(alias)
        assert copied._storage_is_contiguous()
        assert not copied._has_view_provenance()
        np.testing.assert_array_equal(physical(copied, device), [[0, 3], [1, 4], [2, 5]])
        copied._copy_into(jt.ones((3, 2)))
        np.testing.assert_array_equal(physical(source, device), np.arange(6).reshape(2, 3))

    def test_per_call_view_request_restores_native_default_on_error_and_threads(self, device):
        source = jt.array(np.arange(6, dtype='float32').reshape(2, 3))
        with pytest.raises(RuntimeError):
            source._storage_permute(jt.NanoVector([0]))
        assert jt.flags.transpose_storage_view == 0
        # Complete first JIT serially. Threads below construct/read existing
        # kernels and never share a mutable tensor or a compilation cache race.
        physical(view(source), device)
        physical(source.transpose(1, 0), device)

        def work(_):
            for _ in range(4):
                x = jt.array(np.arange(6, dtype='float32').reshape(2, 3))
                y = view(x)
                copied = x.transpose(1, 0)
                assert not y._storage_is_contiguous()
                assert copied._storage_is_contiguous()
                np.testing.assert_array_equal(physical(y, device), physical(copied, device))
                assert jt.flags.transpose_storage_view == 0

        with ThreadPoolExecutor(max_workers=3) as executor:
            list(executor.map(work, range(3)))
        assert jt.flags.transpose_storage_view == 0


instantiate_device_type_tests(TestViewOwnerProtocol, globals())
