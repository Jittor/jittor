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
from collections import OrderedDict, defaultdict


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


def descriptor_cache_key(record, *, shape, dtype, layout, device="npu"):
    """Build a descriptor identity without exposing runtime addresses.

    Attribute values alone do not identify an ACL tensor descriptor: shape,
    dtype, layout, and device placement are part of the descriptor contract.
    The returned tuple is immutable and therefore safe for a future C++/ACL
    cache boundary.  This helper remains CANN-free; it only validates identity
    metadata and never creates an ``aclTensor``.
    """
    normalized = validate_acl_data(record)
    if not isinstance(shape, (list, tuple)):
        raise AclDataUserError("ACL descriptor shape must be a list or tuple")
    normalized_shape = []
    for dimension in shape:
        if not _is_int(dimension) or dimension < 0:
            raise AclDataUserError("ACL descriptor shape dimensions must be non-negative integers")
        normalized_shape.append(dimension)
    for name, value in (("dtype", dtype), ("layout", layout), ("device", device)):
        if not isinstance(value, str) or not value:
            raise AclDataInternalError("ACL descriptor {} must be a non-empty string".format(name))
    return (
        SCHEMA_VERSION,
        normalized["cache_key"],
        tuple(normalized_shape),
        dtype,
        layout,
        device,
    )


class DescriptorHandle:
    """Value-only lease for a host-side descriptor cache entry.

    The handle never owns an ACL object.  It records the canonical key and
    device generation so a future CANN consumer can reject work submitted
    with a descriptor invalidated by ``erase`` or device teardown.
    """

    __slots__ = ("key", "device", "generation", "entry_generation")

    def __init__(self, key, device, generation, entry_generation):
        self.key = key
        self.device = device
        self.generation = generation
        self.entry_generation = entry_generation


def _validate_descriptor_key(key):
    """Validate the immutable key shape before touching cache state.

    The C++ cache validates ``AclDescriptorKey`` on every cache entry point.
    Keep the Python mirror fail-closed as well: accepting an arbitrary tuple
    into ``_entries`` would make a later ``acquire``/teardown failure happen
    after an invalid value had already polluted the cache.
    """
    if not isinstance(key, tuple) or len(key) != 6:
        raise AclDataInternalError("ACL descriptor cache requires a canonical descriptor key")
    version, attribute_key, shape, dtype, layout, device = key
    if version != SCHEMA_VERSION or not isinstance(attribute_key, tuple):
        raise AclDataInternalError("ACL descriptor cache requires a canonical descriptor key")
    if (len(attribute_key) != 3 or attribute_key[0] != SCHEMA_VERSION or
            not isinstance(attribute_key[1], str) or not attribute_key[1] or
            not isinstance(attribute_key[2], tuple)):
        raise AclDataInternalError("ACL descriptor cache requires a canonical descriptor key")
    for field in attribute_key[2]:
        if (not isinstance(field, tuple) or len(field) != 3 or
                not isinstance(field[0], str) or not field[0] or
                field[1] not in _TYPES):
            raise AclDataInternalError("ACL descriptor cache requires a canonical descriptor key")
        try:
            _validate_value(field[1], field[2], field[0])
        except AclDataUserError as error:
            raise AclDataInternalError(
                "ACL descriptor cache requires typed attribute values"
            ) from error
        if field[1] in _VECTOR_TYPES and not isinstance(field[2], tuple):
            raise AclDataInternalError("ACL descriptor cache requires immutable vector values")
    if not isinstance(shape, tuple):
        raise AclDataInternalError("ACL descriptor shape must be an immutable tuple")
    for dimension in shape:
        if not _is_int(dimension) or dimension < 0:
            raise AclDataUserError("ACL descriptor shape dimensions must be non-negative integers")
    for name, value in (("dtype", dtype), ("layout", layout), ("device", device)):
        if not isinstance(value, str) or not value:
            raise AclDataInternalError("ACL descriptor {} must be a non-empty string".format(name))
    return key


class DescriptorCache:
    """Small host-only cache shell for a future ACL descriptor owner.

    Values are supplied by the caller, so this class cannot accidentally
    manufacture or retain a CANN handle.  A real runner can use the same key
    and replace the value with an ``aclTensor`` owner once CANN is available.
    """

    def __init__(self):
        self._entries = OrderedDict()
        # Device teardown advances this generation even when no cache entry
        # exists. A future CANN owner can reject stale handles without
        # exposing runtime pointers through the cache key.
        self._device_generations = defaultdict(int)
        # Keep per-key tombstone epochs so an old lease stays stale even
        # after an equivalent descriptor is rebuilt.
        self._entry_generations = defaultdict(int)

    def get_or_create(self, key, builder):
        _validate_descriptor_key(key)
        if key in self._entries:
            return self._entries[key]
        value = builder(key)
        self._entries[key] = value
        if isinstance(key, tuple) and len(key) == 6 and isinstance(key[-1], str):
            self._device_generations.setdefault(key[-1], 0)
            self._entry_generations.setdefault(key, 0)
        return value

    def __contains__(self, key):
        _validate_descriptor_key(key)
        return key in self._entries

    def __len__(self):
        return len(self._entries)

    def acquire(self, key):
        """Acquire a value-only lease for an existing descriptor entry."""
        _validate_descriptor_key(key)
        if key not in self._entries:
            raise AclDataInternalError("ACL descriptor handle acquired for missing key")
        if not isinstance(key, tuple) or len(key) != 6 or not isinstance(key[-1], str):
            raise AclDataInternalError("ACL descriptor handle requires a canonical descriptor key")
        return DescriptorHandle(
            key, key[-1], self.device_generation(key[-1]), self._entry_generations[key]
        )

    def is_current(self, handle):
        """Return whether a lease still names a live entry and generation."""
        if not isinstance(handle, DescriptorHandle):
            return False
        return (
            handle.key in self._entries
            and isinstance(handle.key, tuple)
            and len(handle.key) == 6
            and handle.key[-1] == handle.device
            and self.device_generation(handle.device) == handle.generation
            and self._entry_generations.get(handle.key) == handle.entry_generation
        )

    def get(self, handle):
        """Resolve a lease, rejecting stale handles before a backend call."""
        if not self.is_current(handle):
            raise AclDataInternalError("stale ACL descriptor handle")
        return self._entries[handle.key]

    def release(self, handle):
        """Release a current lease without deleting a rebuilt entry."""
        if not self.is_current(handle):
            return False
        return self.erase(handle.key)

    def erase(self, key):
        """Invalidate one descriptor identity without touching other devices."""
        _validate_descriptor_key(key)
        removed = self._entries.pop(key, None) is not None
        self._entry_generations[key] += 1
        return removed

    def erase_device(self, device):
        """Invalidate every descriptor identity belonging to ``device``.

        Device teardown can invalidate several descriptors at once. Keys are
        produced by :func:`descriptor_cache_key`, whose device identity is
        the final tuple component; unrelated caller keys are left untouched.
        """
        if not isinstance(device, str) or not device:
            raise AclDataInternalError("ACL descriptor device must be a non-empty string")
        self._device_generations[device] += 1
        removed = 0
        for key in list(self._entries):
            if isinstance(key, tuple) and len(key) == 6 and key[-1] == device:
                del self._entries[key]
                self._entry_generations[key] += 1
                removed += 1
        return removed

    def device_generation(self, device):
        """Return the host-side invalidation generation for ``device``."""
        if not isinstance(device, str) or not device:
            raise AclDataInternalError("ACL descriptor device must be a non-empty string")
        return self._device_generations.get(device, 0)

    def clear(self):
        # A global teardown invalidates descriptors on every known device.
        for device in list(self._device_generations):
            self._device_generations[device] += 1
        for key in list(self._entry_generations):
            self._entry_generations[key] += 1
        self._entries.clear()
