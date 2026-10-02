"""Qwen3 ZeRO-1/2/3 checkpoint and optimizer-resume probe on two Ascend NPUs."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

runtime = os.environ["DS_RUNTIME"]
rank = int(os.environ["RANK"])
zero_stage = int(os.environ.get("DS_ZERO_STAGE", "1"))
assert zero_stage in (1, 2, 3)
assert runtime in ("oracle", "shim") and int(os.environ["WORLD_SIZE"]) == 2
if runtime == "shim":
    from jittor_adapters.deepspeed import activate
    activate(device="npu")
import torch
assert hasattr(torch, "_torch_compat_install_context") == (runtime == "shim")
if runtime == "oracle":
    assert hasattr(torch, "_C")
    import torch_npu
    torch.npu.set_device(0)
else:
    import jittor as jt
    fallback_before = jt.core.backend_fallback_count()

import deepspeed
from deepspeed.utils import safe_get_full_fp32_param
from transformers import AutoModelForCausalLM
assert deepspeed.__version__ == "0.17.6"
root = Path(os.environ["DS_RESUME_OUT"])
out = root / runtime / ("rank%d" % rank)
out.mkdir(parents=True, exist_ok=True)
assert not (out / "report.json").exists()
ckpt = root / runtime / "checkpoint"
pretrained = Path(os.environ["DS_MODEL_PATH"])
if os.environ.get("DS_TRACE_HANG") == "1":
    import faulthandler
    import time
    faulthandler.dump_traceback_later(90, repeat=True)

def trace_phase(event):
    if os.environ.get("DS_TRACE_HANG") == "1":
        print(json.dumps(dict(runtime=runtime, rank=rank, event=event,
                              time_ns=time.time_ns(),
                              rootinfo=os.environ.get("JT_HCCL_ROOTINFO_FILE"),
                              rendezvous_timeout_s=os.environ.get(
                                  "JT_RENDEZVOUS_TIMEOUT_S", "120"))), flush=True)

trace_phase("load-start")
model = AutoModelForCausalLM.from_pretrained(
    str(pretrained), local_files_only=True, torch_dtype=torch.float32,
    attn_implementation="eager").to("npu:0")
trace_phase("load-end")
model.config.use_cache = False
model.train()
parameters = list(model.named_parameters())
assert len(parameters) == 310
assert sum(p.numel() for _, p in parameters) == 596049920
optimizer = torch.optim.AdamW(
    model.parameters(), lr=1e-5, weight_decay=0.0,
    betas=(0.9, 0.99), eps=1e-6)
config = dict(
    train_batch_size=2, train_micro_batch_size_per_gpu=1,
    gradient_accumulation_steps=1, gradient_clipping=0,
    zero_optimization={"stage": zero_stage},
    fp16={"enabled": False}, bf16={"enabled": False},
    steps_per_print=1000, wall_clock_breakdown=False,
    memory_breakdown=False)
trace_phase("initialize-start")
engine, _, _, _ = deepspeed.initialize(
    model=model, optimizer=optimizer, config=config)
if os.environ.get("DS_TRACE_HANG") == "1":
    faulthandler.cancel_dump_traceback_later()
assert str(torch.distributed.get_backend()) == "hccl"
print(json.dumps({"runtime": runtime, "rank": rank, "event": "initialized"}), flush=True)

def save(name, value):
    assert value is not None and value.device.type == "npu", name
    array = value.detach().cpu().numpy()
    assert np.isfinite(array).all(), name
    path = out / (name + ".npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array, allow_pickle=False)
    return array

def weight_hashes():
    result = {}
    for name, parameter in parameters:
        full = safe_get_full_fp32_param(parameter) if zero_stage == 3 else parameter
        assert full is not None and full.device.type == "npu"
        array = full.detach().cpu().numpy()
        assert array.size > 0
        result[name] = hashlib.sha256(array.tobytes()).hexdigest()
        del full
    return result

def step_once(step, phase=None):
    engine.zero_grad()
    ids = ((np.arange(12, dtype=np.int64).reshape(1, 12) * 17
            + 31 + rank * 7 + step * 13) % model.config.vocab_size)
    input_ids = torch.tensor(ids, dtype=torch.int64, device="npu:0")
    embeddings = model.get_input_embeddings()(input_ids)
    embeddings.retain_grad()
    outputs = engine(inputs_embeds=embeddings,
                     labels=input_ids.clone(), use_cache=False)
    if phase is not None:
        save(phase + "/input_ids", input_ids)
        save(phase + "/logits", outputs.logits)
        save(phase + "/loss", outputs.loss)
    engine.backward(outputs.loss)
    if phase is not None:
        save(phase + "/input_grad", embeddings.grad)
    engine.step()
    torch.npu.synchronize()
    if phase is not None:
        for name, parameter in parameters:
            full = safe_get_full_fp32_param(parameter) if zero_stage == 3 else parameter
            assert full is not None
            save(phase + "/updated/" + name, full)
            del full
    if runtime == "shim":
        assert jt.core.backend_fallback_count() == fallback_before
    print(json.dumps({"runtime": runtime, "rank": rank,
                      "event": "step", "step": step, "phase": phase}), flush=True)
    del outputs, embeddings, input_ids

for step in range(3):
    step_once(step)
saved_hashes = weight_hashes()
save_result = engine.save_checkpoint(
    str(ckpt), tag="step3",
    client_state={"marker": "qwen3-zero%d-resume" % zero_stage, "completed_steps": 3})
assert save_result is not False
print(json.dumps({"runtime": runtime, "rank": rank,
                  "event": "checkpoint-saved"}), flush=True)
step_once(3, "first")
with torch.no_grad():
    load_path, client_state = engine.load_checkpoint(
        str(ckpt), tag="step3", load_module_strict=True,
        load_optimizer_states=True, load_lr_scheduler_states=True)
assert load_path and client_state["marker"] == "qwen3-zero%d-resume" % zero_stage
assert client_state["completed_steps"] == 3
restored_hashes = weight_hashes()
assert saved_hashes == restored_hashes, "model weights did not restore exactly"
print(json.dumps({"runtime": runtime, "rank": rank,
                  "event": "checkpoint-restored"}), flush=True)
step_once(3, "resumed")
if runtime == "shim":
    assert jt.core.backend_fallback_count() == fallback_before
report = dict(
    status="executed", runtime=runtime, rank=rank, device="npu",
    backend="hccl", dtype="float32", zero_stage=zero_stage, steps_before_save=3,
    resumed_steps=1, parameters=len(parameters), checkpoint_tag="step3",
    checkpoint_load_path=str(load_path), client_state=client_state,
    checkpoint_sha256=hashlib.sha256(
        (pretrained / "config.json").read_bytes()).hexdigest(),
    model_restored_exact=saved_hashes == restored_hashes,
    fallback_delta=0 if runtime == "shim" else None,
    probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(out / "report.json").write_text(json.dumps(report, indent=2))
print(json.dumps({"runtime": runtime, "rank": rank,
                  "event": "finished", "status": "executed"}), flush=True)
torch.distributed.destroy_process_group()
