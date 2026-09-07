"""Torch tensor distributed ownership."""

def _ddp_all_reduce_grads(leaves):
    """Average DDP-managed gradients across ranks, in a rank-stable order.

    ``_jittor_ddp_state`` is set only by ``DistributedDataParallel`` (see
    installers/nn.py), which sits *above* this file -- the marker carries the
    state so nothing here has to import it, the same inversion FSDP2 uses.

    Operates on the accumulated ``_torch_grad`` and assigns in place, because
    that Var is also the one in the optimizer's ``pg["grads"]``: one write
    updates ``p.grad`` and what ``step()`` consumes.

    The ordering matters and is not incidental. Jittor's collectives are graph
    ops, and every rank must issue them in the same sequence or they pair up
    wrongly and the run deadlocks or mixes gradients between parameters. The
    backward's own leaf collection is keyed by ``id()`` and differs between
    processes; DDP stamps ``_jittor_ddp_order`` in ``module.parameters()``
    order, identical on every rank, and that is what this sorts by. The
    dependency chain then stops the scheduler reordering them again -- the same
    guard jittor's own ``optim/base.py`` puts around its all-reduce.
    """
    from importlib import import_module as _import_module
    _owner = _import_module(__package__)
    if _owner._collectives._world_size() <= 1:
        return
    pending = []
    for leaf in leaves:
        state = getattr(leaf, "_jittor_ddp_state", None)
        if state is None or not getattr(state, "sync_enabled", False):
            continue
        order = getattr(leaf, "_jittor_ddp_order", None)
        grad = getattr(leaf, "_torch_grad", None)
        if order is None or not isinstance(grad, _owner.jt.Var):
            continue
        pending.append((order, grad))
    if not pending:
        return
    pending.sort(key=lambda item: item[0])
    dep = []
    for _order, grad in pending:
        grad.assign(_owner._collectives._all_reduce_mean(grad))
        try:
            producer = grad._input(0)
        except _owner.EXPECTED as exc:
            _owner.swallowed("torch/installers/tensor.py _ddp_all_reduce_grads: "
                      "grad._input(0) for the collective ordering chain", exc,
                      "the all-reduces may be scheduled in a different order "
                      "on different ranks")
        else:
            producer._add_dependency(dep)
            dep = [producer]
