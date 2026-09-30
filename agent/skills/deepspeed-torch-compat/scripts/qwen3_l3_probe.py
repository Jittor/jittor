import hashlib
import json
import os
from pathlib import Path

import numpy as np

runtime = os.environ["DS_RUNTIME"]
rank = int(os.environ["RANK"])
assert int(os.environ["WORLD_SIZE"]) == 2
model_path = os.environ["DS_MODEL_PATH"]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


model_root = Path(model_path)
source_files = [model_root / "config.json"] + sorted(model_root.glob("*.safetensors"))
assert all(path.is_file() for path in source_files), source_files
source_checkpoint = [
    {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    for path in source_files
]

if runtime == "shim":
    from jittor_adapters.deepspeed import activate
    activate(device="npu")

import torch
shim = hasattr(torch, "_torch_compat_install_context")
assert shim == (runtime == "shim")
if runtime == "oracle":
    assert hasattr(torch, "_C")
    import torch_npu
    assert torch.npu.is_available()
    torch.npu.set_device(0)
else:
    import jittor as jt
    jt.flags.use_acl = 1
    from jittor.distributed import get_hccl_world_info
    assert get_hccl_world_info() == {
        "initialized": True, "rank": rank, "world_size": 2
    }
    fallback_before = jt.core.backend_fallback_count()

from transformers import AutoModelForCausalLM
import deepspeed

assert deepspeed.__version__ == "0.17.6"
print(json.dumps({"phase": "load-start", "runtime": runtime, "rank": rank}), flush=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    dtype=torch.float32,
    attn_implementation="eager",
)
model.config.use_cache = False
model.eval()
parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())
assert parameter_count > 500_000_000, parameter_count
model = model.to("npu:0")
if runtime == "shim":
    jt.sync_all(True)
    jt.clean_graph()
else:
    torch.npu.synchronize()
print(json.dumps({
    "phase": "load-done", "runtime": runtime, "rank": rank,
    "parameter_count": parameter_count,
}), flush=True)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-5,
    weight_decay=0.0,
    betas=(0.9, 0.99),
    eps=1e-8,
)
config = {
    "train_batch_size": 2,
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 1,
    "gradient_clipping": 0,
    "zero_optimization": {"stage": 0},
    "fp16": {"enabled": False},
    "bf16": {"enabled": False},
    "steps_per_print": 1000,
    "wall_clock_breakdown": False,
    "memory_breakdown": False,
}
engine, _, _, _ = deepspeed.initialize(
    model=model, optimizer=optimizer, config=config
)
assert str(torch.distributed.get_backend()) == "hccl"
if runtime == "shim":
    jt.sync_all(True)
    jt.clean_graph()
else:
    torch.npu.synchronize()
print(json.dumps({"phase": "engine-ready", "runtime": runtime, "rank": rank}), flush=True)

tokens = np.array([[1, 42, 314, 2718, 99, 7, 1234, 151643]], dtype=np.int64)
tokens[:, 1:7] += rank
input_ids = torch.tensor(tokens, dtype=torch.long, device="npu:0")
labels = input_ids.clone()
with torch.no_grad():
    outputs = engine(input_ids=input_ids, labels=labels, use_cache=False)
loss = float(outputs.loss.detach().cpu().item())
logits = outputs.logits[0, -1, :8].float().detach().cpu().numpy().astype(np.float32)
assert np.isfinite(loss)
assert np.isfinite(logits).all()

reference_loss = loss
reference_logits = logits.copy()
save_dir = Path(os.environ["DS_L3_OUT"]) / runtime / ("saved-rank%d" % rank)
save_dir.mkdir(parents=True, exist_ok=True)
engine.module.save_pretrained(save_dir, safe_serialization=True)
weight_files = sorted(save_dir.glob("*.safetensors"))
assert weight_files, list(save_dir.iterdir())
saved_weight_bytes = sum(path.stat().st_size for path in weight_files)
saved_weight_sha256 = [
    {"name": path.name, "sha256": sha256_file(path)} for path in weight_files
]

del outputs, engine, optimizer, model
import gc
gc.collect()
if runtime == "shim":
    jt.sync_all(True)
    jt.clean_graph()
else:
    torch.npu.synchronize()
reloaded = AutoModelForCausalLM.from_pretrained(
    save_dir,
    local_files_only=True,
    dtype=torch.float32,
    attn_implementation="eager",
)
reloaded.config.use_cache = False
reloaded.eval()
reloaded = reloaded.to("npu:0")
if runtime == "shim":
    jt.sync_all(True)
    jt.clean_graph()
else:
    torch.npu.synchronize()
with torch.no_grad():
    roundtrip_outputs = reloaded(input_ids=input_ids, labels=labels, use_cache=False)
roundtrip_loss = float(roundtrip_outputs.loss.detach().cpu().item())
roundtrip_logits = roundtrip_outputs.logits[0, -1, :8].float().detach().cpu().numpy().astype(np.float32)
roundtrip_loss_abs = abs(roundtrip_loss - reference_loss)
roundtrip_logits_max_abs = float(np.max(np.abs(roundtrip_logits - reference_logits)))
assert roundtrip_loss_abs <= 1e-5, roundtrip_loss_abs
assert roundtrip_logits_max_abs <= 1e-5, roundtrip_logits_max_abs

if runtime == "shim":
    jt.sync_all()
    fallback_delta = int(jt.core.backend_fallback_count() - fallback_before)
    assert fallback_delta == 0
else:
    fallback_delta = None

report = {
    "status": "l3-roundtrip-passed",
    "runtime": runtime,
    "rank": rank,
    "world_size": 2,
    "physical_visibility": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
    "backend": str(torch.distributed.get_backend()),
    "deepspeed": deepspeed.__version__,
    "transformers": __import__("transformers").__version__,
    "model_path": model_path,
    "source_checkpoint": source_checkpoint,
    "parameter_count": parameter_count,
    "zero_stage": 0,
    "loss": loss,
    "logits_last_first8": logits.tolist(),
    "fallback_delta": fallback_delta,
    "roundtrip_loss_abs": roundtrip_loss_abs,
    "roundtrip_logits_max_abs": roundtrip_logits_max_abs,
    "saved_weight_bytes": saved_weight_bytes,
    "saved_weight_sha256": saved_weight_sha256,
}
out = Path(os.environ["DS_L3_OUT"]) / runtime
out.mkdir(parents=True, exist_ok=True)
(out / ("rank%d.json" % rank)).write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
torch.distributed.destroy_process_group()
