import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "python/jittor/extern/acl/aclops/acl_data.py"
SPEC = importlib.util.spec_from_file_location("acl_data_contract", MODULE_PATH)
ACL_DATA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ACL_DATA)


def test_normalizer_applies_defaults_and_freezes_field_order():
    record = ACL_DATA.validate_acl_data(
        {
            "schema_version": 1,
            "op": "Softmax",
            "fields": {"dim": {"type": "int64", "value": -1}},
        },
        expected_op="Softmax",
        schema={
            "dim": {"type": "int64", "required": True},
            "half_to_float": {"type": "bool", "default": False},
        },
    )
    assert list(record["fields"]) == ["dim", "half_to_float"]
    assert record["fields"]["half_to_float"]["value"] is False
    assert record["cache_key"] == (
        1,
        "Softmax",
        (("dim", "int64", -1), ("half_to_float", "bool", False)),
    )


def test_cache_key_is_independent_of_input_order_and_addresses():
    first = ACL_DATA.validate_acl_data({
        "schema_version": 1,
        "op": "Flip",
        "fields": {
            "axes": {"type": "int64[]", "value": [2, 0]},
            "keep": {"type": "bool", "value": True},
        },
    })
    second = ACL_DATA.validate_acl_data({
        "schema_version": 1,
        "op": "Flip",
        "fields": {
            "keep": {"type": "bool", "value": True},
            "axes": {"type": "int64[]", "value": tuple([2, 0])},
        },
    })
    assert first["cache_key"] == second["cache_key"]
    assert "0x" not in repr(first["cache_key"])


@pytest.mark.parametrize("record, expected", [
    ({"schema_version": 2, "op": "Softmax"}, "version"),
    ({"schema_version": 1, "op": "Softmax", "fields": {"dim": {"type": "int64", "value": True}}}, "match"),
    ({"schema_version": 1, "op": "Softmax", "fields": {"other": {"type": "int64", "value": 0}}}, "unknown"),
])
def test_malformed_records_fail_before_any_acl_call(record, expected):
    with pytest.raises(ACL_DATA.AclDataUserError, match=expected):
        ACL_DATA.validate_acl_data(
            record,
            expected_op="Softmax",
            schema={"dim": {"type": "int64", "required": True}},
        )


def test_invalid_integration_schema_is_internal_error():
    with pytest.raises(ACL_DATA.AclDataInternalError):
        ACL_DATA.validate_acl_data(
            {"schema_version": 1, "op": "Softmax"},
            schema={"dim": {"type": "pointer", "required": True}},
        )


def test_descriptor_key_includes_shape_layout_dtype_and_device():
    record = ACL_DATA.validate_acl_data({
        "schema_version": 1,
        "op": "Softmax",
        "fields": {"dim": {"type": "int64", "value": -1}},
    })
    first = ACL_DATA.descriptor_cache_key(
        record, shape=(2, 4), dtype="float32", layout="contiguous", device="npu:0"
    )
    same = ACL_DATA.descriptor_cache_key(
        record, shape=[2, 4], dtype="float32", layout="contiguous", device="npu:0"
    )
    other = ACL_DATA.descriptor_cache_key(
        record, shape=(4, 2), dtype="float32", layout="contiguous", device="npu:0"
    )
    assert first == same
    assert first != other
    assert "0x" not in repr(first)


def test_descriptor_cache_builds_once_and_keeps_device_entries_separate():
    record = {"schema_version": 1, "op": "Scale", "fields": {}}
    key0 = ACL_DATA.descriptor_cache_key(
        record, shape=(), dtype="float32", layout="contiguous", device="npu:0"
    )
    key1 = ACL_DATA.descriptor_cache_key(
        record, shape=(), dtype="float32", layout="contiguous", device="npu:1"
    )
    cache = ACL_DATA.DescriptorCache()
    builds = []
    assert cache.device_generation("npu:0") == 0
    cache.clear()
    assert cache.device_generation("npu:0") == 0
    assert cache.erase_device("npu:0") == 0
    assert cache.device_generation("npu:0") == 1
    assert cache.get_or_create(key0, lambda key: builds.append(key) or "descriptor-0") == "descriptor-0"
    assert cache.get_or_create(key0, lambda key: builds.append(key) or "wrong") == "descriptor-0"
    assert cache.get_or_create(key1, lambda key: builds.append(key) or "descriptor-1") == "descriptor-1"
    assert len(cache) == 2
    assert len(builds) == 2
    assert cache.erase(key0) is True
    assert cache.erase(key0) is False
    assert len(cache) == 1
    # A global clear must invalidate a device that was only observed through
    # insertion, without requiring a prior per-device erase.
    assert cache.device_generation("npu:1") == 0
    assert cache.erase_device("npu:1") == 1
    assert cache.device_generation("npu:1") == 1
    assert len(cache) == 0
    assert cache.erase_device("npu:1") == 0
    assert cache.device_generation("npu:1") == 2
    with pytest.raises(ACL_DATA.AclDataInternalError):
        cache.erase_device(1)
    with pytest.raises(ACL_DATA.AclDataInternalError):
        cache.device_generation(1)
    cache.clear()
    assert len(cache) == 0
    assert cache.device_generation("npu:0") == 2
    assert cache.device_generation("npu:1") == 3


def test_descriptor_handle_rejects_single_and_device_teardown_staleness():
    record = ACL_DATA.validate_acl_data({
        "schema_version": 1,
        "op": "Scale",
        "fields": {},
    })
    key = ACL_DATA.descriptor_cache_key(
        record, shape=(2,), dtype="float32", layout="contiguous", device="npu:0"
    )
    cache = ACL_DATA.DescriptorCache()
    cache.get_or_create(key, lambda _: "descriptor")
    handle = cache.acquire(key)
    assert cache.is_current(handle)
    assert cache.get(handle) == "descriptor"
    assert cache.erase(key)
    assert not cache.is_current(handle)
    with pytest.raises(ACL_DATA.AclDataInternalError, match="stale"):
        cache.get(handle)
    cache.get_or_create(key, lambda _: "descriptor-2")
    assert not cache.is_current(handle)
    with pytest.raises(ACL_DATA.AclDataInternalError, match="stale"):
        cache.get(handle)
    fresh = cache.acquire(key)
    cache.erase_device("npu:0")
    assert not cache.is_current(fresh)
    with pytest.raises(ACL_DATA.AclDataInternalError, match="stale"):
        cache.get(fresh)
    with pytest.raises(ACL_DATA.AclDataInternalError, match="missing"):
        cache.acquire(key)


@pytest.mark.parametrize("kwargs", [
    {"shape": (-1,), "dtype": "float32", "layout": "contiguous", "device": "npu:0"},
    {"shape": (1,), "dtype": "", "layout": "contiguous", "device": "npu:0"},
    {"shape": (1,), "dtype": "float32", "layout": "contiguous", "device": ""},
])
def test_descriptor_key_rejects_invalid_identity_metadata(kwargs):
    record = {"schema_version": 1, "op": "Scale", "fields": {}}
    with pytest.raises((ACL_DATA.AclDataUserError, ACL_DATA.AclDataInternalError)):
        ACL_DATA.descriptor_cache_key(record, **kwargs)


def test_descriptor_cache_rejects_noncanonical_keys_before_insertion():
    cache = ACL_DATA.DescriptorCache()
    malformed = [
        ("not-a-key",),
        (1, (2, "Scale", ()), (2,), "float32", "contiguous", "npu:0"),
        (1, (1, "Scale", (("dim", "pointer", 0),)), (2,),
         "float32", "contiguous", "npu:0"),
        (1, (1, "Scale", (("dim", "int64", object()),)), (2,),
         "float32", "contiguous", "npu:0"),
        (1, (1, "Scale", (("axes", "int64[]", [1, 2]),)), (2,),
         "float32", "contiguous", "npu:0"),
        (1, (1, "Scale", ()), [2], "float32", "contiguous", "npu:0"),
    ]
    for key in malformed:
        with pytest.raises(ACL_DATA.AclDataInternalError):
            cache.get_or_create(key, lambda _: "must-not-build")
        assert len(cache) == 0


def test_descriptor_cache_release_does_not_delete_rebuilt_entry():
    cache = ACL_DATA.DescriptorCache()
    record = {"schema_version": 1, "op": "Scale", "fields": {}}
    key = ACL_DATA.descriptor_cache_key(
        record, shape=(2,), dtype="float32", layout="contiguous", device="npu:0"
    )
    cache.get_or_create(key, lambda _: "first")
    old = cache.acquire(key)
    assert cache.release(old)
    cache.get_or_create(key, lambda _: "replacement")
    assert not cache.release(old)
    fresh = cache.acquire(key)
    assert cache.get(fresh) == "replacement"
    assert cache.release(fresh)
    assert len(cache) == 0
