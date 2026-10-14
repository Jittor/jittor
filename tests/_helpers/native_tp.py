"""Fixed native TP training fixtures and independent unsharded NumPy math."""

import numpy as np
import jittor as jt
from jittor.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear


def weights():
    return {"column.weight": np.arange(18, dtype=np.float32).reshape(6, 3) / 30 - .2,
            "column.bias": np.arange(6, dtype=np.float32) / 50 - .05,
            "row.weight": np.arange(24, dtype=np.float32).reshape(4, 6) / 40 - .3,
            "row.bias": np.arange(4, dtype=np.float32) / 25 - .1}


def data(step, shape=(4, 3)):
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 10 + step / 31
    y = np.arange(np.prod(shape[:-1]) * 4, dtype=np.float32).reshape(shape[:-1] + (4,)) / 20
    return x, y


class Model(jt.Module):
    def __init__(self, gather=False, initialize=True):
        self.column = ColumnParallelLinear(3, 6, gather_output=gather)
        self.row = RowParallelLinear(6, 4, input_is_parallel=not gather)
        if initialize:
            for name, parameter in self.named_parameters():
                parameter.update(jt.array(local(weights()[name], parameter), dtype="float32"))
                parameter.start_grad()

    def execute(self, x):
        return self.row(jt.tanh(self.column(x)))


def local(value, parameter):
    spec = parameter._jittor_tp_spec
    if spec["axis"] is None:
        return value.copy()
    index = [slice(None)] * value.ndim
    width = value.shape[spec["axis"]] // spec["world_size"]
    start = spec["rank"] * width
    index[spec["axis"]] = slice(start, start + width)
    return value[tuple(index)].copy()


def optimizer(model):
    result = jt.optim.SGD(model.parameters(), lr=.05, momentum=.6,
                          dampening=.1, weight_decay=.01, fused=False)
    result.param_sync_iter = 1
    return result


def reference(steps=5, shape=(4, 3)):
    p = {name: value.astype(np.float64) for name, value in weights().items()}
    momentum = {name: np.zeros_like(value) for name, value in p.items()}
    result = []
    for step in range(steps):
        x, target = data(step, shape)
        x = x.astype(np.float64)
        hidden = np.tanh(x @ p["column.weight"].T + p["column.bias"])
        output = hidden @ p["row.weight"].T + p["row.bias"]
        dy = 2 * (output - target) / output.size
        dh = (dy @ p["row.weight"]) * (1 - hidden ** 2)
        dx = dh @ p["column.weight"]
        grads = {"row.weight": dy.reshape(-1, 4).T @ hidden.reshape(-1, 6),
                 "row.bias": dy.reshape(-1, 4).sum(0),
                 "column.weight": dh.reshape(-1, 6).T @ x.reshape(-1, 3),
                 "column.bias": dh.reshape(-1, 6).sum(0)}
        for name in p:
            update = grads[name] + .01 * p[name]
            momentum[name] = update.copy() if step == 0 else .6 * momentum[name] + .9 * update
            p[name] -= .05 * momentum[name]
        result.append({"output": output.copy(), "input_grad": dx.copy(),
                       "gradients": {k: v.copy() for k, v in grads.items()},
                       "model": {k: v.copy() for k, v in p.items()},
                       "moments": {k: v.copy() for k, v in momentum.items()},
                       "loss": float(np.mean((output - target) ** 2))})
    return result


def step(model, opt, number, shape=(4, 3), observer=None):
    x, target = data(number, shape)
    x = jt.array(x, dtype="float32").start_grad()
    output = model(x)
    loss = ((output - jt.array(target, dtype="float32")) ** 2).mean()
    input_grad = jt.grad(loss, [x], retain_graph=True)[0]
    jt.sync([output, loss, input_grad], True, False)
    if observer:
        for value in (x, output, loss, input_grad):
            observer(value)
    opt.backward(loss)
    gradients = opt.param_groups[0]["grads"]
    jt.sync(gradients, True, False)
    if observer:
        for value in gradients:
            observer(value)
    trace = {"output": output.numpy().copy(), "input_grad": input_grad.numpy().copy(),
             "loss": float(loss.item()), "gradients": {
                 name: gradient.numpy().copy()
                 for (name, _), gradient in zip(model.named_parameters(), gradients)}}
    opt.step()
    jt.sync_all(True)
    if observer:
        for value in list(model.parameters()) + list(opt.param_groups[0]["values"]):
            observer(value)
    trace["model"] = {name: value.numpy().copy() for name, value in model.named_parameters()}
    trace["moments"] = {name: value.numpy().copy() for (name, _), value in
                        zip(model.named_parameters(), opt.param_groups[0]["values"])}
    return trace


def assert_trace(actual, expected, model, exact=False):
    for field in ("output", "input_grad"):
        if exact:
            np.testing.assert_array_equal(actual[field], expected[field])
        else:
            np.testing.assert_allclose(actual[field], expected[field], atol=2e-6, rtol=2e-5)
    for field in ("gradients", "model", "moments"):
        for name, parameter in model.named_parameters():
            target = expected[field][name] if exact else local(expected[field][name], parameter)
            if exact:
                np.testing.assert_array_equal(actual[field][name], target)
            else:
                np.testing.assert_allclose(actual[field][name], target, atol=2e-6, rtol=2e-5)
    assert actual["loss"] == expected["loss"] if exact else abs(actual["loss"] - expected["loss"]) < 2e-6
