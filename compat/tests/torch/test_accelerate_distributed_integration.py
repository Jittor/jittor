"""Real two-rank Accelerate gate; run with tools/run_accelerate_gate.py.

The existing NCCL launcher gives each rank one visible device. Its native rank
variables are translated before Accelerate initializes its PartialState.
"""

import contextlib
import ctypes
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import pytest

for _source, _target in (("JT_NCCL_RANK", "RANK"),
                         ("JT_NCCL_WORLD_SIZE", "WORLD_SIZE")):
    if _source in os.environ:
        os.environ.setdefault(_target, os.environ[_source])
if "JT_NCCL_RANK" in os.environ:
    os.environ.setdefault("LOCAL_RANK", "0")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from _accelerate_protocol import Capture, hyperparameters  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402


def _values(value):
    if isinstance(value, torch.Tensor):
        if value.dtype == torch.bfloat16:
            value = value.float()
        return value.detach().clone().cpu().numpy().tolist()
    if isinstance(value, dict):
        return {str(key): _values(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_values(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _assert_tree(actual, expected):
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            _assert_tree(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_tree(left, right)
    elif isinstance(expected, (int, float)):
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
    else:
        assert actual == expected


def _progress(accelerator, stage):
    if os.environ.get("JITTOR_ACCELERATE_INTEGRATION_PROGRESS") == "1":
        print("ACCELERATE_STAGE rank=%d %s" % (accelerator.process_index, stage), flush=True)


class _ObservedLinear(torch.nn.Linear):
    """Observe the actual FSDP compute tensors before Accelerate casts outputs."""

    def forward(self, inputs):
        output = super().forward(inputs)
        observation = {}
        driver = ctypes.CDLL("libcuda.so.1")
        for name, value in (("input", inputs), ("weight", self.weight),
                            ("bias", self.bias), ("output", output)):
            ordinal = ctypes.c_int(-1)
            # CUDA driver attribute 9 independently verifies actual allocation.
            rc = driver.cuPointerGetAttribute(ctypes.byref(ordinal), 9,
                                              ctypes.c_void_p(value.data_ptr()))
            assert rc == 0 and ordinal.value == value.device.index
            observation[name] = {"dtype": str(value.dtype), "device": value.device.type,
                                 "pointer_device": ordinal.value}
        assert all(item["device"] == "cuda" for item in observation.values())
        assert all(item["dtype"] == "torch.bfloat16" for item in observation.values())
        self.compute_observations.append(observation)
        return output


def _model_optimizer(device, observe_bf16=False):
    model = (_ObservedLinear if observe_bf16 else torch.nn.Linear)(4, 3).to(device)
    if observe_bf16:
        model.compute_observations = []
    with torch.no_grad():
        model.weight.copy_(torch.tensor(np.arange(12, dtype=np.float32).reshape(3, 4) / 50, device=device))
        model.bias.copy_(torch.tensor([.03, -.02, .01], device=device))
    optimizer = torch.optim.AdamW([
        {"params": [model.weight], "lr": .01},
        {"params": [model.bias], "lr": .015},
    ], betas=(.7, .93), eps=1e-4, weight_decay=.02)
    return model, optimizer


@pytest.fixture
def distributed_scope():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world != 2:
        if os.environ.get("JITTOR_ACCELERATE_DISTRIBUTED_REQUIRED") == "1":
            pytest.fail("Accelerate integration gate requires two real ranks")
        pytest.skip("requires the real two-rank Accelerate NCCL gate")
    # A required integration gate cannot pass through an optional-import skip.
    from accelerate.state import AcceleratorState, GradientState

    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._shared_state.clear()
    previous_precision = (torch.backends.cuda.matmul.allow_tf32,
                          torch.backends.cudnn.allow_tf32)
    jt = sys.modules.get("jittor")
    fallback = jt.core.backend_fallback_count() if jt is not None else None
    if jt is not None:
        from jittor._runtime.fallback import forbid_backend_fallbacks
        scope = forbid_backend_fallbacks()
    else:
        scope = contextlib.nullcontext()
    try:
        with scope:
            yield
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        AcceleratorState._reset_state(reset_partial_state=True)
        GradientState._shared_state.clear()
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous_precision
        gc.collect()
        if jt is not None:
            assert jt.core.backend_fallback_count() == fallback


def _report(accelerator, case, model, **records):
    metadata = importlib.import_module("importlib.metadata")
    torch.cuda.synchronize()
    raw = accelerator.unwrap_model(model)
    global_parameters = dict(raw.named_parameters())
    parameters = {name: value.to_local() if callable(getattr(value, "to_local", None)) else value
                  for name, value in global_parameters.items()}
    repo = Path(__file__).resolve().parents[3]
    source_paths = ("compat/torch/installers/distributed.py", "compat/fsdp2/_state_dict.py",
                    "compat/torch/optimizer_api.py", "compat/torch/grad.py",
                    "src/runtime/init.cc", "src/core/var.cc", "src/core/node.h",
                    "compat/tests/torch/test_accelerate_distributed_integration.py")
    assert all(parameter.device.type == "cuda" for parameter in parameters.values())
    report = {
        "case": case, "rank": accelerator.process_index,
        "world_size": accelerator.num_processes, "device": str(accelerator.device),
        "backend": dist.get_backend(), "pid": os.getpid(),
        "torch_module": getattr(torch, "__file__", None),
        "torch_distribution_version": metadata.version("torch"),
        "torch_native_version": getattr(torch, "__version__", None),
        "torch_declared_api_version": getattr(torch, "__torch_version__", None),
        "torch_version_namespace": getattr(torch.version, "__version__", None),
        "accelerate_version": metadata.version("accelerate"),
        "torch_type": str(type(torch)), "jittor_loaded": "jittor" in sys.modules,
        "torch_compat_active": hasattr(torch, "_torch_compat_install_context"),
        "source_hashes": {path: hashlib.sha256((repo / path).read_bytes()).hexdigest()
                          for path in source_paths},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "parameter_devices": {name: str(value.device) for name, value in parameters.items()},
        "master_parameter_dtypes": {name: str(value.dtype) for name, value in parameters.items()},
        "parameter_pointers": {name: value.data_ptr() for name, value in parameters.items()},
        "parameter_shapes": {name: list(value.shape) for name, value in parameters.items()},
        "global_parameter_shapes": {name: list(value.shape) for name, value in global_parameters.items()},
        **records,
    }
    assert all(report["parameter_pointers"][name] for name, value in parameters.items() if value.numel())
    assert any(value.numel() for value in parameters.values())
    output = os.environ.get("JITTOR_ACCELERATE_INTEGRATION_OUTPUT")
    if output:
        root = Path(output)
        root.mkdir(parents=True, exist_ok=True)
        (root / (case + "_rank%d.json" % accelerator.process_index)).write_text(json.dumps(report, indent=2))
    accelerator.wait_for_everyone()
    return report


def test_accumulate_real_ddp_tail_and_gather_for_metrics(distributed_scope):
    from accelerate import Accelerator

    accelerator = Accelerator(gradient_accumulation_steps=2, mixed_precision="no")
    _progress(accelerator, "prepare_ddp_tail")
    assert accelerator.num_processes == 2 and accelerator.device.type == "cuda"
    model, optimizer = _model_optimizer("cpu")
    ids = torch.tensor(np.arange(17), dtype=torch.int64, device="cpu")
    inputs = torch.tensor(np.arange(68, dtype=np.float32).reshape(17, 4) / 100, device="cpu")
    targets = torch.tensor(np.arange(51, dtype=np.float32).reshape(17, 3) / 80, device="cpu")
    loader = DataLoader(TensorDataset(inputs, targets, ids), batch_size=4, shuffle=False)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    assert isinstance(model, torch.nn.parallel.DistributedDataParallel)
    raw = accelerator.unwrap_model(model)
    records, metric_ids = [], []
    initial = _values(dict(raw.named_parameters()))
    for index, (x, y, sample_ids) in enumerate(loader):
        _progress(accelerator, "ddp_tail_micro_%d_begin" % index)
        assert x.device.type == y.device.type == sample_ids.device.type == "cuda"
        with accelerator.accumulate(model):
            loss = ((model(x) - y) ** 2).mean()
            accelerator.backward(loss)
            _progress(accelerator, "ddp_tail_micro_%d_backward_done" % index)
            assert all(p.grad is not None and p.grad.device.type == "cuda" for p in raw.parameters())
            grads = _values({name: p.grad for name, p in raw.named_parameters()})
            optimizer.step()
            optimizer.zero_grad()
            gathered_ids = _values(accelerator.gather_for_metrics(sample_ids))
            _progress(accelerator, "ddp_tail_micro_%d_metrics_done" % index)
            metric_ids.extend(gathered_ids)
            records.append({"local_ids": _values(sample_ids), "metrics_ids": gathered_ids,
                            "sync_gradients": accelerator.sync_gradients, "loss": _values(loss),
                            "grads": grads, "parameters": _values(dict(raw.named_parameters())),
                            "optimizer": _values(optimizer.state_dict())})
    assert metric_ids == list(range(17))
    assert [record["sync_gradients"] for record in records] == [False, True, True]
    _report(accelerator, "ddp_tail", model, initial=initial, records=records, all_metric_ids=metric_ids)


def test_real_ddp_no_sync_then_closing_backward(distributed_scope):
    from accelerate import Accelerator

    accelerator = Accelerator(gradient_accumulation_steps=2, mixed_precision="no")
    _progress(accelerator, "prepare_ddp_no_sync")
    model, optimizer = _model_optimizer("cpu")
    model, optimizer = accelerator.prepare(model, optimizer)
    assert isinstance(model, torch.nn.parallel.DistributedDataParallel)
    raw = accelerator.unwrap_model(model)
    initial = _values(dict(raw.named_parameters()))
    records = []
    for index in range(2):
        _progress(accelerator, "no_sync_micro_%d_begin" % index)
        x = torch.tensor(np.full((2, 4), .1 * (accelerator.process_index + 1) * (index + 1), dtype=np.float32), device=accelerator.device)
        scope = accelerator.no_sync(model) if index == 0 else contextlib.nullcontext()
        with scope:
            accelerator.backward(model(x).square().mean())
        _progress(accelerator, "no_sync_micro_%d_backward_done" % index)
        grads = [p.grad.detach().clone() for p in raw.parameters()]
        peers = [torch.zeros_like(grads[0]) for _ in range(2)]
        dist.all_gather(peers, grads[0])
        _progress(accelerator, "no_sync_micro_%d_gather_done" % index)
        if index == 0:
            assert not torch.allclose(peers[0], peers[1])
        else:
            assert torch.allclose(peers[0], peers[1], atol=2e-6, rtol=2e-5)
        records.append({"grads": _values(grads), "peer_weight_grads": _values(peers)})
    optimizer.step()
    _report(accelerator, "no_sync", model, initial=initial, records=records,
            final_parameters=_values(dict(raw.named_parameters())), optimizer=_values(optimizer.state_dict()))


@pytest.mark.parametrize("mixed_precision", ["no", "bf16"])
def test_fsdp2_full_model_optimizer_checkpoint_resume(distributed_scope, tmp_path, mixed_precision):
    from accelerate import Accelerator, FullyShardedDataParallelPlugin, InitProcessGroupKwargs
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict
    from torch.distributed.fsdp import FullOptimStateDictConfig, FullStateDictConfig

    plugin = FullyShardedDataParallelPlugin(
        fsdp_version=2, state_dict_type="FULL_STATE_DICT", auto_wrap_policy="NO_WRAP",
        cpu_ram_efficient_loading=False,
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False))
    # Torch 2.6's FULL loader broadcasts CPU scalars; its oracle needs Gloo
    # alongside NCCL. The Jittor owner places FULL inputs on the target device.
    handlers = [] if hasattr(torch, "_torch_compat_install_context") else [InitProcessGroupKwargs(backend="cpu:gloo,cuda:nccl")]
    accelerator = Accelerator(fsdp_plugin=plugin, mixed_precision=mixed_precision, kwargs_handlers=handlers)
    assert accelerator.is_fsdp2 and accelerator.num_processes == 2
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    bf16 = mixed_precision == "bf16"
    model, optimizer = _model_optimizer(accelerator.device, observe_bf16=bf16)
    model, optimizer = accelerator.prepare(model, optimizer)
    raw = accelerator.unwrap_model(model)
    capture = Capture("fsdp2_full", torch, "cuda", accelerator.num_processes,
                      accelerator.process_index, mixed_precision)
    input_seed = int(os.environ.get("JITTOR_ACCELERATE_INPUT_SEED", "0"))
    input_values = np.arange(16, dtype=np.float32).reshape(4, 4) / 100 + accelerator.process_index / 10
    if input_seed:
        input_values += np.random.RandomState(input_seed).normal(0, .01, input_values.shape).astype(np.float32)
    x = torch.tensor(input_values, device=accelerator.device, requires_grad=True)
    y = torch.tensor(np.arange(12, dtype=np.float32).reshape(4, 3) / 80, device=accelerator.device)
    options = StateDictOptions(full_state_dict=True, cpu_offload=False)

    capture.add("initial/model", dict(raw.named_parameters()))
    update_index = [0]

    def update():
        optimizer.zero_grad()
        x.grad = None
        prefix = "step/%d" % update_index[0]
        update_index[0] += 1
        output = model(x)
        assert output.dtype == (torch.bfloat16 if bf16 else torch.float32)
        loss = ((output.float() - y) ** 2).mean()
        assert loss.dtype == torch.float32
        accelerator.backward(loss)
        assert x.grad is not None and x.grad.device.type == "cuda"
        capture.add(prefix + "/input", x)
        capture.add(prefix + "/target", y)
        capture.add(prefix + "/output", output)
        capture.add(prefix + "/loss", loss)
        capture.add(prefix + "/input_gradient", x.grad)
        capture.add(prefix + "/parameter_gradients", {name: parameter.grad for name, parameter in raw.named_parameters()})
        gradient_dtypes, gradients = {}, {}
        for name, parameter in raw.named_parameters():
            local = parameter.to_local() if callable(getattr(parameter, "to_local", None)) else parameter
            assert local.dtype == torch.float32 and local.device.type == "cuda"
            gradient = parameter.grad
            assert gradient is not None
            local_grad = gradient.to_local() if callable(getattr(gradient, "to_local", None)) else gradient
            assert local_grad.dtype == torch.float32 and local_grad.device.type == "cuda"
            gradient_dtypes[name] = str(local_grad.dtype)
            full_grad = gradient.full_tensor() if callable(getattr(gradient, "full_tensor", None)) else gradient
            gradients[name] = _values(full_grad)
        optimizer.step()
        torch.cuda.synchronize()
        state = get_state_dict(model, optimizer, options=options)
        assert all(isinstance(value, torch.Tensor) for value in state[0].values())
        assert all(isinstance(value, torch.Tensor) for entry in state[1]["state"].values()
                   for value in entry.values())
        assert all(value.dtype == torch.float32 and value.device.type == "cuda"
                   for value in state[0].values())
        capture.add(prefix + "/model", state[0])
        capture.add(prefix + "/optimizer", state[1]["state"])
        capture.add(prefix + "/hyperparameters", hyperparameters(optimizer))
        moment_dtypes = {}
        for name, entry in state[1]["state"].items():
            for key in ("exp_avg", "exp_avg_sq"):
                assert entry[key].dtype == torch.float32 and entry[key].device.type == "cuda"
                moment_dtypes[name + "." + key] = str(entry[key].dtype)
        return {"loss": _values(loss), "model": _values(state[0]), "optimizer": _values(state[1]),
                "gradients": gradients, "input_gradient": _values(x.grad),
                "gradient_dtypes": gradient_dtypes, "moment_dtypes": moment_dtypes}

    def random_sample():
        return {"python": random.random(), "numpy": np.random.rand(3).tolist(),
                "torch_cpu": _values(torch.rand(3, device="cpu")),
                "torch_cuda": _values(torch.rand(3, device=accelerator.device))}

    path = [str(tmp_path / "saved") if accelerator.process_index == 0 else None]
    _progress(accelerator, "checkpoint_path_broadcast")
    dist.broadcast_object_list(path, src=0)
    _progress(accelerator, "first_update")
    random.seed(1729 + accelerator.process_index)
    np.random.seed(1729 + accelerator.process_index)
    torch.manual_seed(1729 + accelerator.process_index)
    torch.cuda.manual_seed_all(1729 + accelerator.process_index)
    random_sample()  # Exercise an advanced stream, not just seed restoration.
    before_save = [update() for _ in range(3 if bf16 or capture.enabled else 1)]
    saved = before_save[-1]
    for entry in saved["optimizer"]["state"].values():
        assert entry["step"] == len(before_save)
        assert np.any(np.asarray(entry["exp_avg"]) != 0)
    _progress(accelerator, "accelerator_save_state")
    accelerator.save_state(path[0], safe_serialization=False)
    expected_random = random_sample()
    expected = [update() for _ in range(2 if bf16 or capture.enabled else 1)]
    _progress(accelerator, "accelerator_load_state")
    accelerator.load_state(path[0])
    restored = get_state_dict(model, optimizer, options=options)
    restored = {"model": _values(restored[0]), "optimizer": _values(restored[1])}
    assert restored == {key: saved[key] for key in ("model", "optimizer")}
    resumed_random = random_sample()
    assert resumed_random == expected_random
    resumed = [update() for _ in range(2 if bf16 or capture.enabled else 1)]
    # The fixed tiny model uses deterministic kernels in both runtimes. FULL
    # restoration must reproduce the same trajectory without a tolerance.
    assert resumed == expected
    jt = sys.modules.get("jittor")
    if jt is not None:
        assert jt.core.backend_fallback_count() == 0
    observations = raw.compute_observations if bf16 else []
    if bf16:
        assert len(observations) == 7
    observation = _report(accelerator, "full_checkpoint_bf16" if bf16 else "full_checkpoint", model,
            mixed_precision=mixed_precision, before_save=before_save,
            compute_observations=observations, fallback_count=0 if jt is not None else None,
            fallback_policy="error" if jt is not None else None,
            global_parameter_shapes={name: list(np.asarray(value).shape)
                                     for name, value in saved["model"].items()},
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            expected_random=expected_random, resumed_random=resumed_random,
            saved=saved, after_restore=restored,
            expected_resume=expected, actual_resume=resumed,
            files=sorted(file.name for file in Path(path[0]).iterdir()))
    capture.finish({"before_save": before_save, "expected_resume": expected, "actual_resume": resumed,
                    "saved": saved, "after_restore": restored,
                    "expected_random": expected_random, "resumed_random": resumed_random,
                    "compute_observations": observations, "distributed_observation": observation}, backend=dist.get_backend())
