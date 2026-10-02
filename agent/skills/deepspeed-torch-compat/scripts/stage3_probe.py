import json
import os
from pathlib import Path

import numpy as np

runtime = os.environ["DS_RUNTIME"]
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
assert world_size == 2
contiguous_gradients = "default"
result_case = "stage3-formal"
assert "DS_CONTIGUOUS_GRADIENTS" not in os.environ
if runtime == "shim":
    from jittor_adapters.deepspeed import activate
    activate(device="npu")

import torch
shim = hasattr(torch, "_torch_compat_install_context")
assert shim == (runtime == "shim")
if runtime == "oracle":
    assert hasattr(torch, "_C")
    assert not shim
    import torch_npu
    assert torch.npu.is_available()
    torch.npu.set_device(0)
else:
    import jittor as jt
    jt.flags.use_acl = 1
    from jittor.distributed import get_hccl_world_info
    assert get_hccl_world_info() == {"initialized": True, "rank": rank, "world_size": 2}
    fallback_before = jt.core.backend_fallback_count()

import deepspeed
assert deepspeed.__version__ == "0.17.6"
device = "npu:0"
model = torch.nn.Sequential(
    torch.nn.Linear(4, 8),
    torch.nn.Tanh(),
    torch.nn.Linear(8, 2),
).to(device)
with torch.no_grad():
    for index, (_, parameter) in enumerate(model.named_parameters()):
        values = np.arange(int(np.prod(tuple(parameter.shape))), dtype=np.float32)
        values = ((values.reshape(tuple(parameter.shape)) % 13) - 6 + index) / 32
        parameter.copy_(torch.tensor(values, dtype=torch.float32, device=device))
optimizer = torch.optim.AdamW(
    model.parameters(), lr=0.001, weight_decay=0.01, betas=(0.9, 0.99), eps=1e-8
)
zero_config = {"stage": 3}
config = {
    "train_batch_size": 4,
    "train_micro_batch_size_per_gpu": 2,
    "gradient_accumulation_steps": 1,
    "gradient_clipping": 0,
    "zero_optimization": zero_config,
    "fp16": {"enabled": False},
    "bf16": {"enabled": False},
    "steps_per_print": 1000,
    "wall_clock_breakdown": False,
    "memory_breakdown": False,
}

engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=config)
assert torch.distributed.is_initialized()
assert str(torch.distributed.get_backend()) == "hccl"
arrays = {}
def save(name, value):
    data = value.detach().cpu().numpy().copy()
    assert np.isfinite(data).all(), name
    arrays[name] = data

current_step = [-1]
_original_optimizer_step = engine.optimizer._optimizer_step
def _capture_optimizer_partition(group_no):
    save("partition_grad/%d/%d" % (current_step[0], group_no),
         engine.optimizer.fp32_partitioned_groups_flat[group_no].grad)
    return _original_optimizer_step(group_no)
engine.optimizer._optimizer_step = _capture_optimizer_partition

for step in range(3):
    current_step[0] = step
    base = np.arange(8, dtype=np.float32).reshape(2, 4)
    x_data = (base - 3 + step) / 10 + rank * 0.05
    target_data = np.array([[0.1, -0.2], [-0.1, 0.3]], dtype=np.float32)
    target_data = target_data + step / 100 + rank * 0.02
    x = torch.tensor(x_data, dtype=torch.float32, device=device, requires_grad=True)
    target = torch.tensor(target_data, dtype=torch.float32, device=device)
    prediction = engine(x)
    loss = (prediction - target).square().mean()
    engine.backward(loss)
    save("output/%d" % step, prediction)
    save("loss/%d" % step, loss)
    save("input_grad/%d" % step, x.grad)
    engine.step()
    for name, parameter in model.named_parameters():
        save("updated/%d/%s" % (step, name), parameter)

if runtime == "shim":
    jt.sync_all()
    assert jt.core.backend_fallback_count() == fallback_before
out = Path(os.environ["DS_MULTI_OUT"]) / result_case / runtime
out.mkdir(parents=True, exist_ok=True)
stem = out / ("rank%d" % rank)
np.savez(stem.with_suffix(".npz"), **arrays)
report = {
    "status": "passed",
    "runtime": runtime,
    "rank": rank,
    "world_size": world_size,
    "logical_device": 0,
    "physical_visibility": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
    "backend": str(torch.distributed.get_backend()),
    "deepspeed": deepspeed.__version__,
    "steps": 3,
    "zero_stage": 3,
    "contiguous_gradients": contiguous_gradients,
    "parameter_count": len(list(model.parameters())),
    "fallback_delta": 0 if runtime == "shim" else None,
}
stem.with_suffix(".json").write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
torch.distributed.destroy_process_group()
