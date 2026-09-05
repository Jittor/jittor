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

