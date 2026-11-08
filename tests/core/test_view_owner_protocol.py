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

    def test_native_materialized_transpose_copy_keeps_real_root_and_siblings(self, device):
        # Explicitly use the existing Native materialized owner. Public Torch
        # permutations keep their physical default and are tested separately.
        assert jt.flags.transpose_storage_view == 0
        axes = jt.NanoVector([1, 0])
        source = OwningTensor(np.arange(1, 7, dtype="float32").reshape(2, 3))
        logical = jt.origin_transpose(source, axes)
        sibling = jt.origin_transpose(source, axes)
        # These are genuine native transpose results; register their actual
        # transformation through the same provenance owner as Native transpose.
        logical._set_transpose_view_of(source, axes)
        sibling._set_transpose_view_of(source, axes)
        shared = view(source)
        values = (source, logical, sibling, shared)
        jt.sync_all()
        pointers = tuple(value._mem_ptr_now for value in values)
        strides = tuple(tuple(value._storage_strides()) for value in values)
        assert all(pointers)
        assert pointers[0] == pointers[3]
        assert pointers[1] != pointers[0] and pointers[2] != pointers[0]
        assert strides == ((3, 1), (2, 1), (2, 1), (1, 3))
        assert logical._has_view_provenance() and sibling._has_view_provenance()
        assert logical._view_base_object() is source and sibling._view_base_object() is source
        assert not logical._storage_view_is_physical()
        assert not sibling._storage_view_is_physical() and shared._storage_view_is_physical()
        np.testing.assert_array_equal(physical(logical, device), [[1., 4.], [2., 5.], [3., 6.]])
        replacement = jt.array([[10., 20.], [30., 40.], [50., 60.]])
        jt.sync_all()  # The imperative native owner requires pending readers drained.
        with jt.no_grad():
            assert logical._copy_view_into(replacement) is True
        assert tuple(value._mem_ptr_now for value in values) == pointers
        assert tuple(tuple(value._storage_strides()) for value in values) == strides
        np.testing.assert_array_equal(physical(source, device), [[10., 30., 50.], [20., 40., 60.]])
        for value in (logical, sibling, shared):
            np.testing.assert_array_equal(physical(value, device), [[10., 20.], [30., 40.], [50., 60.]])
        # A root write must refresh both retained materialized siblings too.
        replacement = jt.array([[7., 8., 9.], [10., 11., 12.]])
        jt.sync_all()
        with jt.no_grad():
            assert source._copy_view_into(replacement) is True
        assert tuple(value._mem_ptr_now for value in values) == pointers
        assert tuple(tuple(value._storage_strides()) for value in values) == strides
        np.testing.assert_array_equal(physical(source, device), [[7., 8., 9.], [10., 11., 12.]])
        for value in (logical, sibling, shared):
            np.testing.assert_array_equal(physical(value, device), [[7., 10.], [8., 11.], [9., 12.]])
        assert not logical._storage_view_is_physical()
        assert not sibling._storage_view_is_physical() and shared._storage_view_is_physical()
        assert jt.flags.transpose_storage_view == 0

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
