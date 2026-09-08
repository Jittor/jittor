"""Queryable implementation-fidelity metadata for Torch compatibility APIs."""

from __future__ import absolute_import

from dataclasses import dataclass
from enum import Enum
from ..transaction import current_transaction, _MISSING


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
    transaction = current_transaction()
    if transaction is not None:
        transaction.record(_REGISTRY, record.api,
                           _REGISTRY.get(record.api, _MISSING), record)
    _REGISTRY[record.api] = record
    # Native descriptors are first-class API owners too, but CPython does not
    # give them a writable attribute dictionary. The registry is authoritative.
    if (isinstance(getattr(implementation, "__dict__", None), dict)
            or isinstance(implementation, type) and implementation.__flags__ & (1 << 9)):
        if transaction is None:
            implementation.__torch_fidelity__ = level.value
            implementation.__torch_fidelity_detail__ = record.detail
        else:
            transaction.mutate_attr(implementation, "__torch_fidelity__", level.value)
            transaction.mutate_attr(implementation, "__torch_fidelity_detail__", record.detail)
    return implementation


def register_api_bindings(namespace, prefix, names, level, detail):
    """Record an explicit family's installed callable owners, including natives."""
    for name in names:
        implementation = getattr(namespace, name, None)
        if isinstance(implementation, property):
            implementation = implementation.fget
        if not callable(implementation):
            continue
        api = prefix + "." + name
        previous = _REGISTRY.get(api)
        if previous is None or previous.implementation is not implementation:
            register_fidelity(api, implementation, level, detail)


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


def fidelity_table(prefix=None):
    """Render the installed implementation coverage as a Markdown table."""
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    rows = ["| API | Fidelity | Implementation owner | Restrictions |",
            "| --- | --- | --- | --- |"]
    for record in fidelity_report(prefix):
        implementation = record.implementation
        owner = getattr(implementation, "__module__", type(implementation).__module__)
        name = getattr(implementation, "__qualname__", type(implementation).__qualname__)
        rows.append("| %s | %s | %s | %s |" % (
            cell(record.api), record.level.value, cell(owner + "." + name),
            cell(record.detail)))
    return "\n".join(rows) + "\n"


__all__ = [
    "Fidelity", "FidelityRecord", "fidelity_of", "fidelity_report",
    "register_fidelity",
    "register_api_bindings",
    "fidelity_table",
]
