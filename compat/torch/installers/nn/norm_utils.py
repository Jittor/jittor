from ...grad import _clip_grads_with_norm_device, _get_total_norm_device
from ....diagnostics import EXPECTED, swallowed

import jittor as _jt

def _get_total_norm(tensors, norm_type=2.0, error_if_nonfinite=False,
                    foreach=None):
    del foreach
    if isinstance(tensors, _jt.Var):
        tensors = [tensors]
    return _get_total_norm_device(
        list(tensors), norm_type, error_if_nonfinite)

def get_total_norm(tensors, norm_type=2.0, error_if_nonfinite=False,
                   foreach=None):
    """Compute a plain or same-mesh DTensor norm through public Torch APIs."""
    if isinstance(tensors, _jt.Var):
        tensors = [tensors]
    tensors = list(tensors)
    if any(not isinstance(value, _jt.Var) for value in tensors):
        raise TypeError("get_total_norm expects tensors")
    from torch.distributed.tensor import DTensor, Replicate
    distributed = [isinstance(value, DTensor) for value in tensors]
    if not any(distributed):
        return _get_total_norm(tensors, norm_type, error_if_nonfinite, foreach)
    if not all(distributed):
        raise ValueError("get_total_norm requires one tensor placement group")
    first = tensors[0]
    mesh = first.device_mesh
    placements = tuple(first.placements)
    if (mesh.ndim != 1 or len(placements) != 1 or
            not (placements[0].is_shard() or placements[0].is_replicate())):
        raise NotImplementedError("DTensor norm requires a one-dimensional shard or replica mesh")
    for value in tensors[1:]:
        if value.device_mesh != mesh or tuple(value.placements) != placements:
            raise ValueError("get_total_norm requires matching mesh and placements")
    local = [value.to_local() for value in tensors]
    group = mesh.get_group() if placements[0].is_shard() else False
    total = _get_total_norm_device(local, norm_type, error_if_nonfinite,
                                   shard_reduce=group)
    return DTensor.from_local(total, mesh, (Replicate(),))


def _clip_grads_with_norm_(parameters, max_norm, total_norm,
                           foreach=None):
    del foreach
    if isinstance(parameters, _jt.Var):
        parameters = [parameters]
    params = list(parameters)
    from ...tensor_state import latest_optimizer
    opt = latest_optimizer(_jt)
    grads = []
    for parameter in params:
        grad = None
        if opt is not None:
            try:
                grad = opt.find_grad(parameter)
            except EXPECTED as exc:
                swallowed("torch/installers/nn.py _clip_grads_with_norm_: grad = opt.find_grad(parameter)", exc)
                grad = None
        if grad is None:
            grad = getattr(parameter, "grad", None)
        if grad is not None:
            grads.append(grad)
    _clip_grads_with_norm_device(grads, max_norm, total_norm)
