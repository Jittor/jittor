"""Host-side contract for ACL attribute data channels.

This module deliberately has no ACL/CANN dependency.  It is the shared
normalization point for the future C++ decoder: callers can validate and
freeze an operator's scalar/vector attributes before generated code or an
ACL executor consumes them.  The current ACL operators still use their
existing generated ``OpAttr`` path; this module does not silently switch that
path over.
"""

from __future__ import annotations

import math
from collections import OrderedDict


SCHEMA_VERSION = 1
_SCALAR_TYPES = {"int64", "float64", "bool"}
_VECTOR_TYPES = {"int64[]", "float64[]", "bool[]"}
_TYPES = _SCALAR_TYPES | _VECTOR_TYPES


class AclDataUserError(ValueError):
    """Malformed data supplied by an operator caller."""


class AclDataInternalError(RuntimeError):
    """An invalid schema supplied by ACL integration code."""


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_value(type_tag, value, field_name):
    if type_tag not in _TYPES:
        raise AclDataInternalError("unknown ACL data type for {}: {}".format(field_name, type_tag))
    if type_tag == "int64":
        valid = _is_int(value)
    elif type_tag == "float64":
        valid = _is_finite(value)
    elif type_tag == "bool":
        valid = isinstance(value, bool)
    else:
        if not isinstance(value, (list, tuple)):
            valid = False
        else:
            scalar_type = type_tag[:-2]
            valid = all(_validate_value(scalar_type, item, field_name) is None for item in value)
    if not valid:
        raise AclDataUserError(
            "ACL data field {!r} does not match {}".format(field_name, type_tag)
        )
    return None


def _schema_entry(entry, field_name):
    if not isinstance(entry, dict):
        raise AclDataInternalError("ACL schema entry {!r} is not a mapping".format(field_name))
    type_tag = entry.get("type")
    if type_tag not in _TYPES:
        raise AclDataInternalError("ACL schema entry {!r} has invalid type".format(field_name))
    has_default = "default" in entry
    if has_default:
        _validate_value(type_tag, entry["default"], field_name)
    return type_tag, bool(entry.get("required", not has_default)), has_default


def canonical_cache_key(record):
    """Return a deterministic, address-independent key for a validated record."""
    if not isinstance(record, dict):
        raise AclDataUserError("ACL data record must be a mapping")
    version = record.get("schema_version")
    op = record.get("op")
    if version != SCHEMA_VERSION or not isinstance(op, str) or not op:
        raise AclDataUserError("ACL data record has an invalid schema_version or op")
    fields = record.get("fields", {})
    if not isinstance(fields, dict):
        raise AclDataUserError("ACL data fields must be a mapping")
    normalized = []
    for name in sorted(fields):
        entry = fields[name]
        if not isinstance(entry, dict) or "type" not in entry:
            raise AclDataUserError("ACL data field {!r} is missing its type".format(name))
        type_tag = entry["type"]
        value = entry.get("value", entry.get("default"))
        _validate_value(type_tag, value, name)
        if isinstance(value, (list, tuple)):
            value = tuple(value)
        normalized.append((name, type_tag, value))
    return (version, op, tuple(normalized))


def validate_acl_data(record, *, expected_op=None, schema=None):
    """Validate and normalize an ACL data record without calling CANN.

    ``schema`` is a mapping of field name to ``{"type": ..., "required":
    ..., "default": ...}``.  The returned record uses an ordered field map
    and carries its canonical cache key, making it safe to pass across a
    future Python/C++ data-channel boundary.
    """
    if not isinstance(record, dict):
        raise AclDataUserError("ACL data record must be a mapping")
    if record.get("schema_version") != SCHEMA_VERSION:
        raise AclDataUserError("unsupported ACL data schema version")
    op = record.get("op")
    if not isinstance(op, str) or not op:
        raise AclDataUserError("ACL data operator name is required")
    if expected_op is not None and op != expected_op:
        raise AclDataUserError("ACL data is for {!r}, expected {!r}".format(op, expected_op))
    fields = record.get("fields", {})
    if not isinstance(fields, dict):
        raise AclDataUserError("ACL data fields must be a mapping")

    if schema is not None:
        if not isinstance(schema, dict):
            raise AclDataInternalError("ACL data schema must be a mapping")
        normalized_schema = {}
        for name, entry in schema.items():
            if not isinstance(name, str) or not name:
                raise AclDataInternalError("ACL schema field names must be non-empty strings")
            normalized_schema[name] = _schema_entry(entry, name)
        unknown = set(fields) - set(normalized_schema)
        if unknown:
            raise AclDataUserError("unknown ACL data fields: {}".format(sorted(unknown)))
        for name, (type_tag, required, has_default) in normalized_schema.items():
            if name not in fields:
                if required and not has_default:
                    raise AclDataUserError("missing required ACL data field: {}".format(name))
                if has_default:
                    fields = dict(fields)
                    fields[name] = {"type": type_tag, "value": entry_default(schema[name])}

    normalized_fields = OrderedDict()
    for name in sorted(fields):
        if not isinstance(name, str) or not name:
            raise AclDataUserError("ACL data field names must be non-empty strings")
        entry = fields[name]
        if not isinstance(entry, dict) or "type" not in entry:
            raise AclDataUserError("ACL data field {!r} is missing its type".format(name))
        type_tag = entry["type"]
        value = entry.get("value", entry.get("default"))
        if value is None:
            raise AclDataUserError("ACL data field {!r} has no value".format(name))
        _validate_value(type_tag, value, name)
        normalized_fields[name] = {
            "type": type_tag,
            "value": list(value) if isinstance(value, tuple) else value,
        }

    normalized = {
        "schema_version": SCHEMA_VERSION,
        "op": op,
        "fields": normalized_fields,
    }
    normalized["cache_key"] = canonical_cache_key(normalized)
    return normalized


def entry_default(entry):
    """Copy a schema default so callers cannot mutate the schema in-place."""
    value = entry["default"]
    return list(value) if isinstance(value, (list, tuple)) else value
