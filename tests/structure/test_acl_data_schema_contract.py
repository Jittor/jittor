import re

import pytest


def validate_acl_data_schema(record):
    assert record["schema_version"] == 1
    assert isinstance(record["op"], str) and record["op"]
    fields = record.get("fields", {})
    assert isinstance(fields, dict)
    for name, entry in fields.items():
        assert isinstance(name, str) and name
        assert entry["type"] in {"int64", "float64", "bool", "int64[]", "float64[]", "bool[]"}
        value = entry.get("value", entry.get("default"))
        assert value is not None
        if entry["type"].endswith("[]"):
            assert isinstance(value, list)
        elif entry["type"] == "bool":
            assert isinstance(value, bool)
        elif entry["type"] == "int64":
            assert isinstance(value, int) and not isinstance(value, bool)
        else:
            assert isinstance(value, (int, float)) and not isinstance(value, bool)
    key = record.get("cache_key", ())
    assert list(key) == sorted(key)
    assert not re.search(r"pointer|object.?id|0x[0-9a-f]+", repr(key), re.I)
    return record


def test_valid_acl_data_schema_is_decoder_ready():
    validate_acl_data_schema({
        "schema_version": 1,
        "op": "Softmax",
        "fields": {"dim": {"type": "int64", "value": 1}},
        "cache_key": [("dim", "int64", 1)],
    })


@pytest.mark.parametrize("record", [
    {"schema_version": 2, "op": "Softmax"},
    {"schema_version": 1, "op": "Softmax", "fields": {"dim": {"type": "pointer", "value": "0x1"}}},
    {"schema_version": 1, "op": "Softmax", "fields": {"dim": {"type": "int64", "value": True}}},
])
def test_invalid_acl_data_schema_is_rejected(record):
    with pytest.raises((AssertionError, KeyError)):
        validate_acl_data_schema(record)
