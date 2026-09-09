# Explicit partial graph submission

`jt.submit_pending(*vars, device_sync=False)` is the explicit boundary for
submitting selected pending graph roots. Each argument must be a `Var`; its
pending producer graph is submitted through the existing `Var.submit_pending`
binding. Other holder roots remain pending and the normal lazy/auto-flush
policy is unchanged. With `device_sync=True`, the selected roots are then
synchronized for immediate host consumption. The function returns the sole
root or a tuple of roots, preserving object identity.

This interface is intended for Function callbacks, fetch bridges, and execution
adapters that know their exact output roots. It does not perform an implicit
global flush, call garbage collection, or make unrelated pending graphs
executable. Empty input and non-Var arguments raise clear Python errors.

The implementation delegates to the existing executor partial submission path;
it owns no second graph, queue, liveness counter, or scheduling policy. CPU and
accelerator execution use the same boundary. Numerical and accelerator
verification remains a separate gate; the focused test exercises selected-root
identity, unrelated-root laziness, synchronization, and invalid arguments.
