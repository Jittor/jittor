"""Queryable implementation-fidelity metadata for Torch compatibility APIs."""

from __future__ import absolute_import

from dataclasses import dataclass
from enum import Enum


class Fidelity(str, Enum):
    EXACT = "exact"
    APPROXIMATE = "approximate"
    UNIMPLEMENTED = "unimplemented"


@dataclass(frozen=True)
class FidelityRecord:
    api: str
    level: Fidelity
    detail: str
    implementation: object


_REGISTRY = {}


def register_fidelity(api, implementation, level, detail):
    """Register one stable API object and return it unchanged."""
    level = level if isinstance(level, Fidelity) else Fidelity(level)
    record = FidelityRecord(str(api), level, str(detail), implementation)
    _REGISTRY[record.api] = record
    implementation.__torch_fidelity__ = level.value
    implementation.__torch_fidelity_detail__ = record.detail
    return implementation


def fidelity_of(api):
    """Return the immutable fidelity record for a fully-qualified API name."""
    try:
        return _REGISTRY[str(api)]
    except KeyError:
        raise KeyError("no Torch compatibility fidelity metadata for %s" % api)


def fidelity_report(prefix=None):
    """Return registered records in deterministic API-name order."""
    records = _REGISTRY.values()
    if prefix is not None:
        prefix = str(prefix)
        records = (record for record in records if record.api.startswith(prefix))
    return tuple(sorted(records, key=lambda record: record.api))


__all__ = [
    "Fidelity", "FidelityRecord", "fidelity_of", "fidelity_report",
    "register_fidelity",
]
