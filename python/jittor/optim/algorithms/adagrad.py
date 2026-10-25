"""Dense FP32 Adagrad with per-parameter accumulators and step counts."""
from numbers import Real
import jittor as jt
from jittor._core.dtypes import dtype_name
from ..base import Optimizer, _state_buffer, _update_preserve_dtype


def validate_adagrad_options(options):
    """Reject unsupported execution modes before changing parameters/state."""
    for name in ("lr", "lr_decay", "weight_decay", "initial_accumulator_value", "eps"):
        value = options[name]
        if isinstance(value, jt.Var):
            raise NotImplementedError("Adagrad tensor-valued hyperparameters are not supported")
        if not isinstance(value, Real) or not 0 <= value:
            raise ValueError("Invalid Adagrad %s: %r" % (name, value))
    for name in ("foreach", "differentiable", "fused"):
        if options.get(name):
            raise NotImplementedError("Adagrad %s=True is not supported" % name)


def _validate_tensor(value, role):
    if getattr(value, "is_sparse", False):
        raise NotImplementedError("Adagrad sparse %s are not supported" % role)
    if not isinstance(value, jt.Var):
        raise TypeError("Adagrad %s must be dense tensors" % role)
    if dtype_name(value.dtype) != "float32":
        raise NotImplementedError("Adagrad currently supports dense float32 %s only" % role)


def adagrad_step_tensor(value):
    """Create a native host counter; the Torch frontend supplies its type scope.

    Native array creation does not itself consume the Python frontend placement
    request, so retain an explicit host copy for native callers. Torch state
    restoration uses its own public placement-aware tensor factory.
    """
    token = jt.core._set_tensor_placement(0, 0)
    try:
        return jt.array(float(value), dtype="float32")._copy_to_cpu().stop_grad()
    finally:
        jt.core._reset_tensor_placement(token)


def adagrad_update(param, grad, accumulator, *, lr, lr_decay, weight_decay,
                   eps, maximize, step):
    """One dense update; shared by native and Torch optimizer frontends."""
    if maximize:
        grad = -grad
    if weight_decay:
        grad = grad + param * weight_decay
    _update_preserve_dtype(accumulator, accumulator + grad * grad)
    effective_lr = lr / (1 + (step - 1) * lr_decay)
    return param - effective_lr * grad / (jt.sqrt(accumulator) + eps)


class Adagrad(Optimizer):
    """Dense float32 Adagrad; state is eager and counters live on the host.

    Sparse/complex tensors, tensor hyperparameters, foreach, differentiable and
    fused updates are explicitly unsupported.
    """
    def __init__(self, params, lr=1e-2, lr_decay=0, weight_decay=0,
                 initial_accumulator_value=0, eps=1e-10, foreach=None, *,
                 maximize=False, differentiable=False, fused=None):
        self._adagrad_defaults = dict(
            lr=lr, lr_decay=lr_decay, weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value, eps=eps,
            foreach=foreach, maximize=maximize, differentiable=differentiable,
            fused=fused)
        validate_adagrad_options(self._adagrad_defaults)
        super().__init__(params, lr)
        for key, value in self._adagrad_defaults.items():
            setattr(self, key, value)
        for group in self.param_groups:
            self._initialize_group(group)

    def _initialize_group(self, group):
        for key, value in self._adagrad_defaults.items():
            group.setdefault(key, value)
        validate_adagrad_options(group)
        group["params"] = list(group["params"])
        for param in group["params"]:
            _validate_tensor(param, "parameters")
        group["values"] = [
            (_state_buffer(param) + group["initial_accumulator_value"]).stop_grad()
            for param in group["params"]]
        # Non-fused Torch Adagrad has CPU scalar counters on every backend.
        group["_adagrad_steps"] = [
            adagrad_step_tensor(0)
            for _ in group["params"]]

    def add_param_group(self, group):
        if not isinstance(group, dict):
            raise TypeError("optimizer parameter group must be a dictionary")
        group = dict(group)
        params = group["params"]
        group["params"] = [params] if isinstance(params, jt.Var) else list(params)
        existing = {id(p) for pg in self.param_groups for p in pg["params"]}
        if any(id(p) in existing for p in group["params"]):
            raise ValueError("some parameters appear in more than one parameter group")
        self._initialize_group(group)
        self.param_groups.append(group)

    def step(self, loss=None, retain_graph=False):
        self.pre_step(loss, retain_graph)
        try:
            # Reject every bad input before the first state/parameter write.
            for group in self.param_groups:
                validate_adagrad_options(group)
                for param, grad in zip(group["params"], group.get("grads", ())):
                    if grad is None:
                        continue
                    _validate_tensor(param, "parameters")
                    _validate_tensor(grad, "gradients")
                    if list(param.shape) != list(grad.shape):
                        raise ValueError("Adagrad gradient shape does not match parameter")
            for group in self.param_groups:
                for param, grad, accumulator, counter in zip(
                        group["params"], group.get("grads", ()),
                        group["values"], group["_adagrad_steps"]):
                    if grad is None:
                        continue
                    step = int(counter.item()) + 1
                    counter.update((counter + 1).stop_grad())
                    trainable = param.requires_grad
                    _update_preserve_dtype(param, adagrad_update(
                        param, grad, accumulator, lr=group["lr"],
                        lr_decay=group["lr_decay"], weight_decay=group["weight_decay"],
                        eps=group["eps"], maximize=group["maximize"], step=step))
                    if trainable:
                        param.start_grad()
            self.post_step()
        finally:
            jt.flags.node_order = 0
