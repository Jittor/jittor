"""Immutable implementation delegates owned by one installation context."""
from types import MappingProxyType
from ..transaction import active_transaction, _MISSING


def bind_delegates(context, key, values):
    table = MappingProxyType(dict(values))
    transaction = active_transaction(context)
    if transaction is not None:
        transaction.record(context.state, key, context.state.get(key, _MISSING), table)
    context.state[key] = table
    return table
