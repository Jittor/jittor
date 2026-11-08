"""Native ZeRO fixtures and independent NumPy optimizer trajectories."""

import jittor as jt
import numpy as np
from contextlib import nullcontext
from jittor.distributed import ZeroRedundancyOptimizer
from _helpers.native_checkpoint import inputs


def model_optimizer(algorithm="Adam", *, zero=True, frozen=False, single=False):
    model = jt.nn.Linear(3, 2, bias=not single)
    model.weight.update(jt.array(np.arange(6, dtype=np.float32).reshape(2, 3) / 20))
    groups = [{"params": [model.weight], "lr": 0.01}]
    if not single:
        model.bias.update(jt.array([0.1, -0.2], dtype="float32"))
        if frozen:
            model.bias.stop_grad()
        groups.append({"params": [model.bias], "lr": 0.02})
    options = dict(lr=0.5, weight_decay=0.02)
    if algorithm == "SGD":
        options.update(momentum=0.6, dampening=0.1, fused=False)
    else:
        options.update(betas=(0.85, 0.95), eps=1e-8)
        if algorithm == "AdamW":
            options["fused"] = False
    cls = getattr(jt.optim, algorithm)
    optimizer = (ZeroRedundancyOptimizer(groups, cls, **options) if zero else cls(groups, **options))
    return model, optimizer


def step(model, optimizer, rank=0, *, accumulate=1, observer=None,
         storage_observer=None, ddp_accumulate=False):
    x, target = inputs(rank)
    for microbatch in range(accumulate):
        context = model.no_sync() if ddp_accumulate and microbatch < accumulate - 1 else nullcontext()
        with context:
            output = model(jt.array(x, dtype="float32"))
            loss = ((output - jt.array(target, dtype="float32")) ** 2).mean() / accumulate
            if observer:
                observer(output)
                observer(loss)
            value = float(loss.item()) * accumulate
            optimizer.backward(loss)
    gradients = []
    for group in optimizer.param_groups:
        for parameter, gradient in zip(group["params"], group["grads"]):
            if observer and bool(parameter.requires_grad):
                observer(gradient)
            if storage_observer:
                storage_observer(gradient)
            gradients.append(gradient.numpy().copy())
    optimizer.step()
    local = optimizer.local_optimizer if isinstance(optimizer, ZeroRedundancyOptimizer) else optimizer
    trace = {"model": {}, "gradients": gradients, "loss": value,
             "steps": [group["n_step"] for group in optimizer.param_groups], "local": []}
    for name, parameter in model.state_dict().items():
        if observer:
            observer(parameter)
        trace["model"][name] = parameter.numpy().copy()
    for group in local.param_groups:
        states = {}
        for name in ("m", "values"):
            if name in group:
                states[name] = []
                for parameter, state in zip(group["params"], group[name]):
                    if observer and bool(parameter.requires_grad):
                        observer(state)
                    if storage_observer:
                        storage_observer(state)
                    states[name].append(state.numpy().copy())
        trace["local"].append(states)
    return trace


def numpy_trajectory(algorithm, steps=5, *, world=1, frozen=False, single=False,
                     frozen_until=0, group_lrs=None):
    parameters = [(np.arange(6, dtype=np.float32).reshape(2, 3) / 20).astype(np.float64)]
    if not single:
        parameters.append(np.array([0.1, -0.2], dtype=np.float32).astype(np.float64))
    moment = [np.zeros_like(p) for p in parameters]
    variance = [np.zeros_like(p) for p in parameters]
    initialized = [False] * len(parameters)
    traces = []
    for index in range(1, steps + 1):
        grads = [np.zeros_like(p) for p in parameters]
        losses = []
        for rank in range(world):
            x, target = inputs(rank)
            error = x.astype(np.float64) @ parameters[0].T - target.astype(np.float64)
            if not single:
                error += parameters[1]
            losses.append(float(np.mean(error ** 2)))
            dout = 2 * error / error.size
            grads[0] += dout.T @ x / world
            if not single and not (frozen or index <= frozen_until):
                grads[1] += dout.sum(axis=0) / world
        for i, (parameter, gradient) in enumerate(zip(parameters, grads)):
            if i == 1 and (frozen or index <= frozen_until):
                continue
            lr = group_lrs[index - 1][i] if group_lrs else 0.01 if i == 0 else 0.02
            if algorithm == "SGD":
                update = gradient + 0.02 * parameter
                moment[i] = update.copy() if not initialized[i] else 0.6 * moment[i] + 0.9 * update
                initialized[i] = True
                parameter -= lr * moment[i]
            else:
                g = gradient if algorithm == "AdamW" else gradient + 0.02 * parameter
                moment[i] = 0.85 * moment[i] + 0.15 * g
                variance[i] = 0.95 * variance[i] + 0.05 * g * g
                if algorithm == "AdamW":
                    parameter *= 1 - lr * 0.02
                    denom = np.sqrt(variance[i]) / np.sqrt(1 - 0.95 ** index) + 1e-8
                    parameter -= moment[i] * (lr / (1 - 0.85 ** index)) / denom
                else:
                    scalar = lr * np.sqrt(1 - 0.95 ** index) / (1 - 0.85 ** index)
                    parameter -= moment[i] * scalar / (np.sqrt(variance[i]) + 1e-8)
        traces.append({"model": dict(zip(("weight", "bias"), [p.copy() for p in parameters])),
                       "gradients": [g.copy() for g in grads], "losses": losses,
                       "m": [m.copy() for m in moment], "v": [v.copy() for v in variance]})
    return traces


def assert_reference(trace, reference, optimizer, rank=0, prefix=""):
    for name, want in reference["model"].items():
        np.testing.assert_allclose(trace["model"][prefix + name], want, atol=2e-6, rtol=2e-5)
    for got, want in zip(trace["gradients"], reference["gradients"]):
        np.testing.assert_allclose(got, want, atol=2e-6, rtol=2e-5)
    assert abs(trace["loss"] - reference["losses"][rank]) < 2e-6
    indices = optimizer.state_partition()["local_indices"]
    for gi, (group, local_indices) in enumerate(zip(trace["local"], indices)):
        for field, values in group.items():
            source = "v" if field == "values" and "m" in group else "m"
            for value, index in zip(values, local_indices):
                assert index == 0  # Each fixture group contains one parameter.
                np.testing.assert_allclose(value, reference[source][gi], atol=2e-6, rtol=2e-5)


def assert_exact(actual, expected):
    for name in actual["model"]:
        np.testing.assert_array_equal(actual["model"][name], expected["model"][name])
    for got, want in zip(actual["gradients"], expected["gradients"]):
        np.testing.assert_array_equal(got, want)
    for actual_group, expected_group in zip(actual["local"], expected["local"]):
        for name in actual_group:
            for got, want in zip(actual_group[name], expected_group[name]):
                np.testing.assert_array_equal(got, want)
    assert actual["loss"] == expected["loss"]
    assert actual["steps"] == expected["steps"]


def check_bfloat16_resume(directory, observer=None):
    from _helpers.native_checkpoint import Bfloat16TrainingModel
    from jittor.distributed import checkpoint, get_rank, get_world_size

    def make():
        model = Bfloat16TrainingModel()
        opt = ZeroRedundancyOptimizer(model.parameters(), jt.optim.SGD,
                                      lr=0.125, momentum=0.5, fused=False)
        assert len(opt.local_optimizer.param_groups[0]["values"]) == (1 if get_rank() == 0 else 0)
        return model, opt

    def run(model, opt):
        loss = model(jt.array([0.5, -0.25], dtype="bfloat16").stop_grad())
        opt.backward(loss)
        if observer:
            observer(loss)
            observer(opt.param_groups[0]["grads"][0])
        gradient = opt.param_groups[0]["grads"][0].numpy().copy()
        opt.step()
        assert str(model.weight.dtype) == "bfloat16"
        states = opt.local_optimizer.param_groups[0]["values"]
        for tensor in [model.weight] + states:
            assert str(tensor.dtype) == "bfloat16"
            if observer:
                observer(tensor)
        return {"weight": jt.reinterpret_view(model.weight, (2,), "uint16").numpy().copy(),
                "momentum": [s.numpy().copy() for s in states], "gradient": gradient}

    model, opt = make()
    parameter = np.array([0.25, -0.5], dtype=np.float32)
    gradient = np.array([0.5, -0.25], dtype=np.float32)
    momentum = np.zeros(2, dtype=np.float32)
    expected = []
    for index in range(5):
        trace = run(model, opt)
        momentum = gradient.copy() if index == 0 else 0.5 * momentum + gradient
        parameter -= 0.125 * momentum
        assert np.all((parameter.view(np.uint32) & 0xffff) == 0)
        np.testing.assert_array_equal(trace["weight"], (parameter.view(np.uint32) >> 16).astype(np.uint16))
        np.testing.assert_array_equal(trace["gradient"], gradient)
        for actual in trace["momentum"]:
            np.testing.assert_array_equal(actual, momentum)
        if index == 2:
            checkpoint.save_training_state(model, opt, directory)
        if index >= 3:
            expected.append(trace)
    fresh, resumed = make()
    checkpoint.load_training_state(fresh, resumed, directory)
    for want in expected:
        actual = run(fresh, resumed)
        np.testing.assert_array_equal(actual["weight"], want["weight"])
        np.testing.assert_array_equal(actual["gradient"], want["gradient"])
        for got, expected_momentum in zip(actual["momentum"], want["momentum"]):
            np.testing.assert_array_equal(got, expected_momentum)
    return {"rank": get_rank(), "world_size": get_world_size(),
            "owned_bfloat16_state_numel": sum(int(v.numel()) for v in resumed.local_optimizer.param_groups[0]["values"])}
