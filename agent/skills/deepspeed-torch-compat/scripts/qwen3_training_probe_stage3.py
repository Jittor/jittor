"""Full Qwen3 ZeRO-3 training records; run under a two-rank site launcher.

Every trainable parameter gradient and updated value is saved, not sampled.
Raw arrays belong in DS_QWEN_FULL_OUT outside the checkout. Both runtimes use
this same probe and fixed local checkpoint. Comparison is a separate step.
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np

runtime = os.environ['DS_RUNTIME']
rank = int(os.environ['RANK'])
assert runtime in ('oracle', 'shim') and int(os.environ['WORLD_SIZE']) == 2
steps = int(os.environ.get('DS_FULL_STEPS', '3'))
zero_stage = int(os.environ.get('DS_ZERO_STAGE', '3'))
assert steps > 0 and zero_stage == 3
out = Path(os.environ['DS_QWEN_FULL_OUT']) / runtime / ('rank%d' % rank)
out.mkdir(parents=True, exist_ok=True)
assert not (out / 'report.json').exists(), 'use a fresh result directory'


def event(name, **fields):
    print(json.dumps(dict(event=name, runtime=runtime, rank=rank, **fields)), flush=True)


if runtime == 'shim':
    from jittor_adapters.deepspeed import activate
    activate(device='npu')
import torch
assert hasattr(torch, '_torch_compat_install_context') == (runtime == 'shim')
if runtime == 'oracle':
    assert hasattr(torch, '_C')
    import torch_npu
    assert torch.npu.is_available()
    torch.npu.set_device(0)
else:
    import jittor as jt
    from jittor.distributed import get_hccl_world_info
    assert get_hccl_world_info() == dict(initialized=True, rank=rank, world_size=2)
    fallback_before = jt.core.backend_fallback_count()

import deepspeed
from deepspeed.utils import safe_get_full_grad, safe_get_full_fp32_param
from transformers import AutoModelForCausalLM
assert deepspeed.__version__ == '0.17.6'
device = 'npu:0'
checkpoint = Path(os.environ['DS_MODEL_PATH'])
checkpoint_digest = hashlib.sha256()
with (checkpoint / 'model.safetensors').open('rb') as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
        checkpoint_digest.update(block)
checkpoint_sha256 = checkpoint_digest.hexdigest()
assert checkpoint_sha256 == 'f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b'
model = AutoModelForCausalLM.from_pretrained(
    str(checkpoint), local_files_only=True, torch_dtype=torch.float32,
    attn_implementation='eager').to(device)
model.config.use_cache = False
model.train()
parameters = list(model.named_parameters())
assert len(parameters) == 310
assert sum(p.numel() for _, p in parameters) == 596049920
assert all(p.requires_grad and p.dtype == torch.float32 for _, p in parameters)
original_shapes = {name: list(p.shape) for name, p in parameters}
optimizer_config = dict(lr=1e-5, weight_decay=0.0, betas=(0.9, 0.99), eps=1e-6)
optimizer = torch.optim.AdamW(model.parameters(), **optimizer_config)
config = dict(train_batch_size=2, train_micro_batch_size_per_gpu=1,
    gradient_accumulation_steps=1, gradient_clipping=0,
    zero_optimization={'stage': zero_stage}, fp16={'enabled': False}, bf16={'enabled': False},
    steps_per_print=1000, wall_clock_breakdown=False, memory_breakdown=False)
engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=config)
assert str(torch.distributed.get_backend()) == 'hccl'
event('initialized', parameters=len(parameters))
records = {}
grad_none = {}


def save(key, value):
    assert value is not None, 'missing tensor: ' + key
    assert value.device.type == 'npu', (key, str(value.device))
    array = value.detach().cpu().numpy()
    assert np.isfinite(array).all(), 'non-finite: ' + key
    path = out / (key + '.npy')
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array, allow_pickle=False)
    records[key] = dict(shape=list(array.shape), dtype=str(array.dtype),
        device='npu', elements=int(array.size), path=str(path.relative_to(out)))


for step in range(steps):
    engine.zero_grad()
    ids = ((np.arange(12, dtype=np.int64).reshape(1, 12) * 17
            + 31 + rank * 7 + step * 13) % model.config.vocab_size)
    input_ids = torch.tensor(ids, dtype=torch.int64, device=device)
    embeddings = model.get_input_embeddings()(input_ids)
    embeddings.retain_grad()
    outputs = engine(inputs_embeds=embeddings, labels=input_ids.clone(), use_cache=False)
    save('step%d/input_ids' % step, input_ids)
    save('step%d/logits' % step, outputs.logits)
    save('step%d/loss' % step, outputs.loss)
    event('forward', step=step)
    engine.backward(outputs.loss)
    save('step%d/input_grad' % step, embeddings.grad)
    grad_none[str(step)] = {name: p.grad is None for name, p in parameters}
    event('backward', step=step, none_gradients=sum(grad_none[str(step)].values()))
    # All ranks must call the public accessor in identical parameter order:
    # ZeRO-1 assembles a full gradient using a collective for each parameter.
    for index, (name, parameter) in enumerate(parameters):
        gradient = safe_get_full_grad(parameter)
        assert gradient is not None, 'missing parameter gradient: ' + name
        assert tuple(gradient.shape) == tuple(original_shapes[name]), name
        save('step%d/grad/%s' % (step, name), gradient)
        del gradient
        if index % 50 == 0:
            event('gradient-records', step=step, count=index + 1)
    engine.step()
    torch.npu.synchronize()
    for name, parameter in parameters:
        full_parameter = safe_get_full_fp32_param(parameter)
        assert full_parameter is not None and tuple(full_parameter.shape) == tuple(original_shapes[name]), name
        save('step%d/updated/%s' % (step, name), full_parameter)
        del full_parameter
    if runtime == 'shim':
        assert jt.core.backend_fallback_count() == fallback_before
    (out / ('step%d-records.json' % step)).write_text(json.dumps(records, indent=2))
    event('step-complete', step=step)
    del outputs, embeddings, input_ids

versions = {name: importlib.metadata.version(name)
            for name in ('deepspeed', 'transformers', 'numpy', 'safetensors')}
report = dict(status='passed', runtime=runtime, rank=rank, world_size=2,
    zero_stage=zero_stage, steps=steps, parameter_count=len(parameters),
    parameters=original_shapes,
    versions=versions, dtype='float32', device='npu', backend='hccl',
    torch_is_shim=hasattr(torch, '_torch_compat_install_context'),
    torch_has_c_extension=hasattr(torch, '_C'),
    torch_version=str(torch.__version__), torch_file=(str(torch.__file__) if runtime == 'oracle' else str(jt.__file__)),
    training_config=config, optimizer_config=optimizer_config,
    checkpoint_sha256=checkpoint_sha256, grad_none=grad_none,
    input_path='inputs_embeds from trainable embedding, without detach',
    config_sha256=hashlib.sha256((checkpoint / 'config.json').read_bytes()).hexdigest(),
    probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    fallback_delta=0 if runtime == 'shim' else None, records=records)
(out / 'report.json').write_text(json.dumps(report, indent=2))
event('finished', steps=steps)
torch.distributed.destroy_process_group()
