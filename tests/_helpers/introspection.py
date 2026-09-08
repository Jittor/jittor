"""Detached projections used by tests comparing only graph liveness."""


def liveness_snapshot(backend):
    counters = backend.introspection.counters
    return dict(hold_vars=counters.held_vars, lived_vars=counters.live_vars,
                lived_ops=counters.live_ops)
