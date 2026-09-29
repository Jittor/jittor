"""Real two-rank Accelerate gate; run with tools/run_accelerate_gate.py.

The existing NCCL launcher gives each rank one visible device. Its native rank
variables are translated before Accelerate initializes its PartialState.
"""

import contextlib
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

for _source, _target in (("JT_NCCL_RANK", "RANK"),
                         ("JT_NCCL_WORLD_SIZE", "WORLD_SIZE")):
    if _source in os.environ:
        os.environ.setdefault(_target, os.environ[_source])
if "JT_NCCL_RANK" in os.environ:
    os.environ.setdefault("LOCAL_RANK", "0")

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset


def _values(value):
    if isinstance(value, torch.Tensor):
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


def _model_optimizer(device):
    model = torch.nn.Linear(4, 3).to(device)
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
                    "src/runtime/init.cc", "src/core/var.cc")
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


def test_fsdp2_full_model_optimizer_checkpoint_resume(distributed_scope, tmp_path):
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
    accelerator = Accelerator(fsdp_plugin=plugin, mixed_precision="no", kwargs_handlers=handlers)
    assert accelerator.is_fsdp2 and accelerator.num_processes == 2
    model, optimizer = _model_optimizer(accelerator.device)
    model, optimizer = accelerator.prepare(model, optimizer)
    x = torch.tensor(np.arange(16, dtype=np.float32).reshape(4, 4) / 100 + accelerator.process_index / 10, device=accelerator.device)
    y = torch.tensor(np.arange(12, dtype=np.float32).reshape(4, 3) / 80, device=accelerator.device)
    options = StateDictOptions(full_state_dict=True, cpu_offload=False)

    def update():
        optimizer.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        accelerator.backward(loss)
        optimizer.step()
        torch.cuda.synchronize()
        state = get_state_dict(model, optimizer, options=options)
        assert all(isinstance(value, torch.Tensor) for value in state[0].values())
        assert all(isinstance(value, torch.Tensor) for entry in state[1]["state"].values()
                   for value in entry.values())
        return {"loss": _values(loss), "model": _values(state[0]), "optimizer": _values(state[1])}

    path = [str(tmp_path / "saved") if accelerator.process_index == 0 else None]
    _progress(accelerator, "checkpoint_path_broadcast")
    dist.broadcast_object_list(path, src=0)
    _progress(accelerator, "first_update")
    saved = update()
    _progress(accelerator, "accelerator_save_state")
    accelerator.save_state(path[0], safe_serialization=False)
    expected = update()
    _progress(accelerator, "accelerator_load_state")
    accelerator.load_state(path[0])
    restored = get_state_dict(model, optimizer, options=options)
    restored = {"model": _values(restored[0]), "optimizer": _values(restored[1])}
    _assert_tree(restored, {key: saved[key] for key in ("model", "optimizer")})
    resumed = update()
    _assert_tree(resumed, expected)
    _report(accelerator, "full_checkpoint", model, saved=saved, after_restore=restored,
            expected_resume=expected, actual_resume=resumed,
            files=sorted(file.name for file in Path(path[0]).iterdir()))
