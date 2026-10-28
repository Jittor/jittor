"""Framework-independent deterministic fixture; no sharding/update implementation."""

import numpy as np


def model(namespace, *, native, dropout=0.0):
    nn = namespace.nn

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(5, 7)
            self.dropout = nn.Dropout(p=dropout)
            self.fc2 = nn.Linear(7, 3)
            value = np.array([0.9, 1.0, 1.1], dtype=np.float32)
            tensor = (
                namespace.array(value, dtype="float32").stop_grad()
                if native
                else namespace.from_numpy(value)
            )
            self.register_buffer("gain", tensor)

        def execute(self, sample):
            return self.fc2(self.dropout(self.fc1(sample).tanh())) * self.gain

        forward = execute

    result = TinyModel()
    result.train()
    for index, (name, parameter) in enumerate(sorted(result.named_parameters())):
        value = (
            np.sin(np.arange(int(np.prod(tuple(parameter.shape))), dtype=np.float32) + index + 1)
            * 0.13
        ).reshape(tuple(parameter.shape))
        if native:
            parameter.update(namespace.array(value, dtype="float32"))
            parameter.start_grad()
        else:
            with namespace.no_grad():
                parameter.copy_(namespace.from_numpy(value).to(parameter.device))
    return result


def samples(step, rank):
    random = np.random.RandomState(1000 + step * 17 + rank)
    return random.normal(0.0, 0.3, (3, 5)).astype(np.float32), random.normal(
        0.0, 0.2, (3, 3)
    ).astype(np.float32)


def optimizer(namespace, module, kind):
    named = sorted(module.named_parameters())
    groups = [
        dict(params=[value for _, value in named[:2]], lr=0.013, weight_decay=0.02),
        dict(params=[value for _, value in named[2:]], lr=0.007, weight_decay=0.0),
    ]
    if kind == "sgd":
        return namespace.optim.SGD(groups, lr=0.01, momentum=0.9)
    return namespace.optim.AdamW(groups, lr=0.01, eps=1e-8, betas=(0.85, 0.95))


def run_cuda_trajectory(*, mode, kind, output, checkpoint_path=None, dtype="float32"):
    import ctypes
    import datetime
    import json
    import random
    import sys
    from types import SimpleNamespace

    import jittor as jt
    from jittor import distributed as dist
    from jittor.distributed import checkpoint
    from jittor.distributed.fsdp import FullyShardedDataParallel
    from jittor._runtime.fallback import forbid_backend_fallbacks

    a = SimpleNamespace(
        mode=mode, kind=kind, output=output, checkpoint=checkpoint_path, dtype=dtype
    )
    assert jt.compiler.has_cuda and jt.flags.use_cuda
    precision = list(jt.core.float32_precision_state())
    assert precision == ["highest", "highest"], precision
    assert "torch" not in sys.modules
    assert not any(n.startswith("jittor.compat") for n in sys.modules)
    if dist.get_default_store() is None:
        dist.init_process_group(
            "nccl", init_method="env://", timeout=datetime.timedelta(seconds=90)
        )
    assert dist.get_world_size() == 2 and dist.get_backend() == "nccl"
    rank = dist.get_rank()
    jt.set_global_seed(8100 + rank)
    np.random.seed(7100 + rank)
    random.seed(6100 + rank)
    module = model(jt, native=True, dropout=0.25 if a.mode != "parity" else 0.0)
    kwargs = {}
    if a.dtype == "bfloat16":
        from jittor.distributed.fsdp import MixedPrecisionPolicy

        kwargs["mixed_precision"] = MixedPrecisionPolicy(
            param_dtype="bfloat16", reduce_dtype="float32", output_dtype="float32"
        )
    module = FullyShardedDataParallel(module, **kwargs)
    opt = optimizer(jt, module, a.kind)
    state = module.module._fsdp_state
    arrays = {}
    readback_tag = object()
    checks = {"all_gather": 0, "reduce_scatter": 0}
    compute_dtypes = []
    collective_dtypes = {"all_gather": [], "reduce_scatter": []}
    all_gather = dist.all_gather
    reduce_scatter = dist.reduce_scatter

    def observe(value, storage=False):
        value.sync()
        assert jt.core.dispatch_context([value]) == ("cuda", 0)
        if value.location() != "device":
            assert (
                storage
                and value.location() == "cpu"
                and getattr(value, "_native_fsdp_trace_readback", None) is readback_tag
            ), (
                "unexplained CUDA residency",
                str(value.location()),
                tuple(value.shape),
            )
            print("AG_INPUT_EXPLICIT_NUMPY_READBACK", step, tuple(value.shape), flush=True)
        device = ctypes.c_int(-1)
        result = ctypes.CDLL("libcuda.so.1").cuPointerGetAttribute(
            ctypes.byref(device), 9, ctypes.c_void_p(value.device_raw_ptr)
        )
        assert result == 0 and device.value == 0
        assert value.location() == "device"
        getattr(value, "__dict__", {}).pop("_native_fsdp_trace_readback", None)

    def gather(value, **kw):
        checks["all_gather"] += 1
        collective_dtypes["all_gather"].append(str(value.dtype))
        observe(value, storage=True)
        result = all_gather(value, **kw)
        observe(result)
        return result

    def scatter(value, **kw):
        checks["reduce_scatter"] += 1
        collective_dtypes["reduce_scatter"].append(str(value.dtype))
        observe(value)
        result = reduce_scatter(value, **kw)
        observe(result)
        return result

    def rng(prefix):
        arrays[prefix + "/python"] = np.array([random.random() for _ in range(5)])
        arrays[prefix + "/numpy"] = np.random.random(7)
        with jt.flag_scope(use_cuda=0):
            arrays[prefix + "/cpu"] = jt.rand(13).numpy().copy()
        gpu = jt.rand(15)
        observe(gpu)
        arrays[prefix + "/cuda"] = gpu.numpy().copy()

    def record(prefix, value):
        if isinstance(value, jt.Var):
            observe(value)
            arrays[prefix] = value.numpy().copy()
            object.__setattr__(value, "_native_fsdp_trace_readback", readback_tag)
        else:
            arrays[prefix] = np.asarray(value)

    # Observe the actual Linear call while preserving its native owner unchanged.
    linear = module.module.fc1
    linear_execute = linear.execute

    def observed_linear(sample):
        compute_dtypes.append({"input": str(sample.dtype), "weight": str(linear.weight.dtype)})
        assert str(sample.dtype) == a.dtype and str(linear.weight.dtype) == a.dtype
        return linear_execute(sample)

    object.__setattr__(linear, "execute", observed_linear)

    dropout_observations = []
    if a.mode != "parity":
        dropout = module.module.dropout
        dropout_execute = dropout.execute

        def observed_dropout(sample):
            output = dropout_execute(sample)
            before_value = sample.cast("float32")
            after_value = output.cast("float32")
            record(f"step/{step}/dropout/input", before_value)
            record(f"step/{step}/dropout/output", after_value)
            before_array = arrays[f"step/{step}/dropout/input"]
            after_array = arrays[f"step/{step}/dropout/output"]
            active = after_array != 0
            np.testing.assert_allclose(
                after_array[active], before_array[active] / 0.75, rtol=0.015, atol=2e-4
            )
            dropout_observations.append(int(np.count_nonzero((before_array != 0) & ~active)))
            return output

        object.__setattr__(dropout, "execute", observed_dropout)

    before = jt.core.backend_fallback_count()
    assert before == 0
    with forbid_backend_fallbacks():
        if a.mode == "resume":
            assert a.checkpoint is not None
            extra = checkpoint.load_training_state(module, opt, a.checkpoint)
            assert extra == {"completed_steps": 3, "dtype": a.dtype, "kind": a.kind}
            start = 3
        else:
            start = 0
        dist.all_gather = gather
        dist.reduce_scatter = scatter
        for step in range(start, 5):
            x, y = samples(step, rank)
            x = jt.array(x).start_grad()
            y = jt.array(y).stop_grad()
            prediction = module(x)
            loss = ((prediction - y) ** 2).mean()
            record(f"step/{step}/prediction", prediction)
            record(f"step/{step}/loss", loss)
            record(f"step/{step}/input_gradient", jt.grad(loss, x, retain_graph=True))
            opt.backward(loss)
            for group in opt.param_groups:
                for parameter, gradient in zip(group["params"], group["grads"]):
                    assert str(gradient.dtype) == "float32"
                    entry = parameter._jittor_fsdp2_entry
                    record(f"step/{step}/gradient/{entry.name}", gradient)
            assert all(entry.full_param is None for entry in state.true_fsdp_params)
            opt.step()
            for entry in state.true_fsdp_params:
                record(f"step/{step}/parameter/{entry.name}", entry.shard)
                assert getattr(entry.owner, entry.attr) is entry.shard
                assert int(entry.shard.numel()) == entry.shard_numel
            for group in opt.param_groups:
                for field in ("m", "values"):
                    if field not in group:
                        continue
                    for parameter, value in zip(group["params"], group[field]):
                        assert str(value.dtype) == "float32"
                        record(
                            f"step/{step}/state/{parameter._jittor_fsdp2_entry.name}/{field}", value
                        )
            if a.mode != "parity":
                rng(f"step/{step}/rng")
                if step == 2:
                    assert a.checkpoint is not None
                    checkpoint.save_training_state(
                        module,
                        opt,
                        a.checkpoint,
                        extra={"completed_steps": 3, "dtype": a.dtype, "kind": a.kind},
                    )
        dist.all_gather = all_gather
        dist.reduce_scatter = reduce_scatter
    assert jt.core.backend_fallback_count() == before
    assert checks["all_gather"] == checks["reduce_scatter"] == 4 * (5 - start)
    assert len(compute_dtypes) == 5 - start
    assert set(collective_dtypes["all_gather"]) == {a.dtype}
    assert set(collective_dtypes["reduce_scatter"]) == {"float32"}
    assert all(str(e.shard.dtype) == "float32" for e in state.true_fsdp_params)
    if a.mode != "parity":
        assert len(dropout_observations) == 5 - start and sum(dropout_observations) > 0
    assert "torch" not in sys.modules and not any(
        n.startswith("jittor.compat") for n in sys.modules
    )
    a.output.mkdir(parents=True, exist_ok=True)
    np.savez(a.output / f"rank{rank}.npz", **arrays)
    metadata = {
        "mode": a.mode,
        "kind": a.kind,
        "dtype": a.dtype,
        "rank": rank,
        "world": 2,
        "steps": 5 - start,
        "source": jt.__file__,
        "shim_loaded": False,
        "fallback_delta": 0,
        "float32_precision": precision,
        "collectives": checks,
        "local_elements": sum(int(e.shard.numel()) for e in state.true_fsdp_params),
        "global_elements": sum(e.numel for e in state.true_fsdp_params),
        "actual_device": "cuda:0",
        "physical_visible_device": __import__("os").environ["CUDA_VISIBLE_DEVICES"],
        "arrays": len(arrays),
        "dropout_zero_counts": dropout_observations,
        "compute_dtypes": compute_dtypes,
        "collective_dtypes": collective_dtypes,
    }
    (a.output / f"rank{rank}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("NATIVE_FSDP_WORKER " + json.dumps(metadata, sort_keys=True), flush=True)
    return arrays, metadata


def run_step(module, opt, index, accumulate=1):
    import jittor as jt

    arrays = {}
    for micro in range(accumulate):
        x, y = samples(index * accumulate + micro, 0)
        x, y = jt.array(x).start_grad(), jt.array(y).stop_grad()
        output = module(x)
        loss = ((output - y) ** 2).mean() / accumulate
        arrays["input/" + str(micro)] = jt.grad(loss, x, retain_graph=True).numpy().copy()
        arrays["output/" + str(micro)] = output.numpy().copy()
        opt.backward(loss)
    for gi, group in enumerate(opt.param_groups):
        for pi, value in enumerate(group["grads"]):
            parameter = group["params"][pi]
            entry = getattr(parameter, "_jittor_fsdp2_entry", None)
            shape = entry.shape if entry is not None else tuple(parameter.shape)
            arrays[f"gradient/{gi}/{pi}"] = value.numpy().copy().reshape(shape)
    opt.step()
    for name, value in module.named_parameters():
        entry = getattr(value, "_jittor_fsdp2_entry", None)
        arrays["parameter/" + (name[7:] if name.startswith("module.") else name)] = (
            value.numpy().copy().reshape(entry.shape if entry is not None else tuple(value.shape))
        )
    for gi, group in enumerate(opt.param_groups):
        for field in ("m", "values"):
            for pi, value in enumerate(group.get(field, ())):
                parameter = group["params"][pi]
                entry = getattr(parameter, "_jittor_fsdp2_entry", None)
                arrays[f"{field}/{gi}/{pi}"] = (
                    value.numpy()
                    .copy()
                    .reshape(entry.shape if entry is not None else tuple(parameter.shape))
                )
    return arrays


def equal(actual, expected):
    assert set(actual) == set(expected)
    for key in actual:
        assert actual[key].shape == expected[key].shape
        np.testing.assert_allclose(actual[key], expected[key], rtol=2e-6, atol=2e-7, err_msg=key)


def assert_oracle_tensor(actual, expected, key, dtype):
    """Fixed elementwise oracle protocol, with shape and finite checks."""
    assert actual.shape == expected.shape, (key, actual.shape, expected.shape)
    assert np.isfinite(actual).all() and np.isfinite(expected).all(), key
    atol, rtol = (2e-6, 2e-5) if dtype == "float32" else (4e-4, 3e-2)
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=key)


def assert_oracle_update(actual, expected, key, dtype):
    """The complete per-parameter step must track the oracle, including small updates.

    The BF16 norm budget is fixed at 3% plus 1e-6, independently of the
    elementwise parameter tolerance. An update below the absolute floor is
    rejected as uninformative, rather than granting a no-op a pass.
    """
    assert actual.shape == expected.shape, (key, actual.shape, expected.shape)
    assert np.isfinite(actual).all() and np.isfinite(expected).all(), key
    floor, relative = (2e-6, 2e-5) if dtype == "float32" else (1e-6, 0.03)
    norm = float(np.linalg.norm(expected))
    error = float(np.linalg.norm(actual - expected))
    budget = floor + relative * norm
    assert norm > 10 * floor, (key, "oracle update cannot be distinguished from the floor", norm)
    assert error <= budget, (key, "parameter update", error, budget)
    return {
        "oracle_norm": norm,
        "error_norm": error,
        "absolute_floor": floor,
        "relative_budget": relative,
        "error_budget": budget,
    }
