"""Real ACL acceptance for readonly bounded PFA zero-padding constants."""

from collections import OrderedDict

import jittor as jt
import numpy as np
import pytest

from _helpers import capability
from jittor._runtime.fallback import forbid_backend_fallbacks
from jittor.backends.acl.kernels._soc import soc_name
from jittor.backends.acl.kernels.ops import flashattention_op as attention


@pytest.fixture(autouse=True)
def fresh_acl_padding_cache(monkeypatch):
    if not capability.check_accelerator("acl", backend=jt).enabled:
        pytest.skip("real ACL device required")
    monkeypatch.setattr(attention, "_prompt_padding_cache", OrderedDict())
    monkeypatch.setattr(attention, "_prompt_padding_cache_bytes", 0)
    monkeypatch.setattr(attention, "_prompt_causal_mask_cache", OrderedDict())
    monkeypatch.setattr(attention, "_prompt_full_mask_cache", OrderedDict())
    before = jt.core.backend_fallback_count()
    with jt.runtime.scope(use_cuda=1, use_acl=1, transpose_storage_view=1), jt.no_grad(), forbid_backend_fallbacks():
        if not soc_name().startswith("Ascend310P"):
            pytest.skip("PFA prefill selection is verified for Ascend310P")
        yield
    assert jt.core.backend_fallback_count() == before


def original_prefill(query, key, value, scale, causal):
    """The uncached production computation, including its real padding copies."""
    batch, heads, length, width = map(int, query.shape)
    key_heads = int(key.shape[1])
    padded = (length + 127) // 128 * 128
    mask = attention._prompt_mask(query, length, padded, causal)
    padding = padded - length
    if padding:
        query, key, value = (jt.nn.pad(tensor, (0, 0, 0, padding), value=0)
                             for tensor in (query, key, value))
    if key_heads != heads:
        expanded = (batch, key_heads, heads // key_heads, padded, width)
        key = key.unsqueeze(2).broadcast(expanded).reshape(batch, heads, padded, width)
        value = value.unsqueeze(2).broadcast(expanded).reshape(batch, heads, padded, width)
    output = attention.PromptFlashAttentionACL(heads, scale, is_causal=causal)(query, key, value, mask)
    return output[:, :, :length, :] if padding else output


def independent_host_copy(tensor):
    copy = jt.Var.copy(tensor)
    copy.sync(True, False)
    assert copy._storage_address != tensor._storage_address
    result = copy.numpy().copy()
    assert tensor.location() == "device"
    return result


def assert_cache_unchanged_zero_and_bounded():
    cache = attention._prompt_padding_cache
    assert attention._prompt_padding_cache_bytes == sum(size for _, size in cache.values())
    assert len(cache) <= attention._prompt_padding_cache_limit
    assert attention._prompt_padding_cache_bytes <= attention._prompt_padding_cache_max_bytes
    for (device, dtype, shape), (zero, size) in cache.items():
        assert zero.is_stop_grad() and zero.is_finished
        assert tuple(zero.shape) == shape and str(zero.dtype) == dtype
        assert int(zero.device_id) == device
        assert size <= attention._prompt_padding_cache_max_entry_bytes
        assert np.count_nonzero(independent_host_copy(zero)) == 0


@pytest.mark.parametrize("batch,heads,key_heads,length,width,causal,strided", [
    (1, 12, 12, 196, 64, False, True),
    (1, 12, 12, 1370, 64, False, True),
    (1, 24, 24, 800, 128, False, True),
    (1, 4, 2, 129, 64, True, True),
    (2, 4, 4, 257, 128, True, False),
])
def test_prefill_matches_uncached_bytes_without_changing_inputs_or_cache(
        batch, heads, key_heads, length, width, causal, strided):
    rng = np.random.RandomState(150)
    for slot in range(2):
        hosts = [rng.uniform(-.3, .3, (batch, count, length, width)).astype(np.float16)
                 for count in (heads, key_heads, key_heads)]
        tensors = []
        for host in hosts:
            if strided:
                tensor = jt.core.ops.transpose(jt.array(host.transpose(0, 2, 1, 3).copy()), (0, 2, 1, 3))
                assert not tensor._storage_is_contiguous()
            else:
                tensor = jt.array(host)
            tensors.append(tensor)
        expected = original_prefill(*tensors, width ** -.5, causal)
        actual = attention._prompt_prefill_padded(*tensors, width ** -.5, is_causal=causal)
        jt.sync([expected, actual], True)
        assert expected.location() == actual.location() == "device"
        a, b = jt.fetch_sync([actual, expected])
        assert a.dtype == b.dtype == np.float16 and a.shape == b.shape
        assert np.isfinite(a).all() and a.tobytes() == b.tobytes()
        for tensor, host in zip(tensors, hosts):
            np.testing.assert_array_equal(independent_host_copy(tensor), host)
        assert_cache_unchanged_zero_and_bounded()


def test_aligned_prefill_never_calls_padding_helper(monkeypatch):
    tensors = [jt.array(np.random.RandomState(i).normal(size=(1, 2, 128, 64)).astype(np.float16))
               for i in range(3)]

    def unexpected(*args):
        raise AssertionError("aligned prefill must not allocate padding")

    monkeypatch.setattr(attention, "_prompt_pad_sequence", unexpected)
    output = attention._prompt_prefill_padded(*tensors, 64 ** -.5, is_causal=False)
    output.sync(True)
    assert output.location() == "device" and not attention._prompt_padding_cache


def test_cache_entry_count_byte_budget_lru_and_oversize_decline(monkeypatch):
    monkeypatch.setattr(attention, "_prompt_padding_cache_limit", 2)
    monkeypatch.setattr(attention, "_prompt_padding_cache_max_entry_bytes", 128)
    monkeypatch.setattr(attention, "_prompt_padding_cache_max_bytes", 192)
    query = jt.ones((1, 2, 5, 8), dtype="float16")
    query.sync(True, False)
    first = attention._prompt_padding_zero(query, 1)
    second = attention._prompt_padding_zero(query, 2)
    assert attention._prompt_padding_zero(query, 1) is first
    third = attention._prompt_padding_zero(query, 3)
    assert len(attention._prompt_padding_cache) == 2
    assert attention._prompt_padding_cache_bytes == 32 + 96
    assert all(value is not second for value, _ in attention._prompt_padding_cache.values())
    fourth = attention._prompt_padding_zero(query, 4)
    assert attention._prompt_padding_cache_bytes == 128
    assert len(attention._prompt_padding_cache) == 1
    assert attention._prompt_padding_zero(query, 5) is None
    assert attention._prompt_padding_cache_bytes == 128
    # Eviction releases the cache's ownership without changing constants a
    # graph or another caller still holds.
    for zero in (first, second, third, fourth):
        assert np.count_nonzero(independent_host_copy(zero)) == 0
    assert_cache_unchanged_zero_and_bounded()


@pytest.mark.parametrize("access", ["readback", "explicit_migration"])
def test_cached_host_read_or_migration_is_safe_before_next_device_concat(access):
    query = jt.ones((1, 2, 5, 8), dtype="float16")
    query.sync(True, False)
    zero = attention._prompt_padding_zero(query, 3)
    address = zero._storage_address
    assert np.count_nonzero(zero.numpy()) == 0
    # Upstream readback now copies without moving the source allocation.
    assert zero.location() == "device" and zero._storage_address == address
    if access == "explicit_migration":
        zero.migrate_to_cpu()
        assert zero.location() == "cpu"
    output = attention._prompt_pad_sequence(query, 3)
    output.sync(True)
    actual = output.numpy()
    np.testing.assert_array_equal(actual[:, :, :5, :], np.ones((1, 2, 5, 8), dtype=np.float16))
    assert np.count_nonzero(actual[:, :, 5:, :]) == 0
    assert_cache_unchanged_zero_and_bounded()


def test_shape_dtype_switches_and_capture_miss_produce_finished_constants():
    wanted = [((1, 2, 5, 8), "float16"), ((2, 2, 5, 8), "float16"),
              ((1, 4, 5, 8), "float16"), ((1, 2, 5, 16), "float16"),
              ((1, 2, 5, 8), "float32")]
    for shape, dtype in wanted:
        query = jt.ones(shape, dtype=dtype)
        query.sync(True, False)
        with jt.runtime.scope(keep_graph=2):
            zero = attention._prompt_padding_zero(query, 3)
        assert zero.is_finished and str(zero.dtype) == dtype
        assert tuple(zero.shape) == (shape[0], shape[1], 3, shape[3])
    assert len(attention._prompt_padding_cache) == len(wanted)
    assert_cache_unchanged_zero_and_bounded()
