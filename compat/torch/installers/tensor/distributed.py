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
    cuda_parameters = True
    for leaf in leaves:
        state = getattr(leaf, "_jittor_ddp_state", None)
        if state is None or not getattr(state, "sync_enabled", False):
            continue
        order = getattr(leaf, "_jittor_ddp_order", None)
        grad = getattr(leaf, "_torch_grad", None)
        if order is None or not isinstance(grad, _owner.jt.Var):
            continue
        pending.append((order, grad, state))
        # A lazy native gradient may not yet have resident device storage.
        # The parameter's declared device is authoritative for DDP placement.
        cuda_parameters = cuda_parameters and leaf.device.type == 'cuda'
    if not pending:
        return
    pending.sort(key=lambda item: item[0])
    if cuda_parameters:
        ops = getattr(_owner.jt.compile_extern, 'nccl_ops', None)
        if ops is not None and callable(getattr(ops, 'nccl_all_reduce_many', None)):
            _ddp_all_reduce_grouped(pending)
            return
    dep = []
    for _order, grad, _state in pending:
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


def _ddp_all_reduce_grouped(pending):
    from importlib import import_module
    from ...context import get_install_context
    owner = import_module(__package__)
    from jittor._runtime import step_capture
    state = pending[0][2]
    cap = getattr(state, 'bucket_cap_bytes', 25 * 1024**2)
    ops = getattr(owner.jt.compile_extern, 'nccl_ops', None)
    bucketed = (state is not None
            and all(item[2] is state for item in pending)
            and all(str(item[1].dtype).rsplit('.', 1)[-1] == 'float32' for item in pending)
            and sum(grad.numel()*grad.element_size() for _, grad, _ in pending) >= cap
            and callable(getattr(ops, 'nccl_all_reduce_deferred', None))
            and callable(getattr(ops, 'nccl_wait_many', None)))
    if bucketed and not step_capture.active() and not hasattr(state, 'bucket_param_order'):
        # Registration order can place shared, late gradients first (e.g.
        # relative-position embeddings), blocking all subsequent buckets.
        # Build once during the ordinary warmup, before tracing. Only rank
        # zero's metadata order is authoritative: local graph IDs differ.
        estimates = owner.jt.core._gradient_producer_orders([g for _, g, _ in pending])
        indices = sorted(range(len(pending)), key=lambda i: (estimates[i], -pending[i][0]))
        ordered = [pending[i][0] for i in indices]
        context = get_install_context(owner.jt)
        target = context.target_namespace
        tensor = target.tensor(ordered, dtype=target.int64,
                               device='cuda:%d' % owner.jt.core.current_device())
        agreed = ops.nccl_broadcast(tensor, 0).cpu().numpy().reshape(-1).tolist()
        if sorted(agreed) != sorted(order for order, _, _ in pending):
            raise RuntimeError('DDP bucket order does not cover the same parameters on every rank')
        state.bucket_param_order = tuple(int(order) for order in agreed)
    if bucketed and step_capture.active():
        order = getattr(state, 'bucket_param_order', None)
        lookup = {index: grad for index, grad, _ in pending}
        if order is not None and set(order) == set(lookup):
            gradients = [lookup[index] for index in order]
            step_capture.guard(lambda s=state: s.bucket_param_order)
        else:
            gradients = [grad for _, grad, _ in reversed(pending)]
        reduced = owner._collectives._all_reduce_mean_bucketed(gradients, cap)
        for grad, value in zip(gradients, reduced):
            grad.assign(value)
        return
    gradients = [grad for _, grad, _ in pending]
    reduced = owner._collectives._all_reduce_mean_many(gradients)
    for grad, value in zip(gradients, reduced):
        grad.assign(value)
