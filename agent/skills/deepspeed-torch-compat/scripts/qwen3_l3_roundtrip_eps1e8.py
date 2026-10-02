"""Round-trip the real Qwen3 model after the recorded three-step ZeRO-1 run.

This tests model save_pretrained/from_pretrained and tokenizer, not optimizer-state
checkpoint/resume. Both runtimes run this same probe on allocated NPU devices.
"""
import hashlib, json, os
from pathlib import Path
import numpy as np
runtime=os.environ['DS_RUNTIME']; rank=int(os.environ['RANK'])
assert runtime in ('oracle','shim') and int(os.environ['WORLD_SIZE'])==2
if runtime=='shim':
    from jittor_adapters.deepspeed import activate
    activate(device='npu')
import torch
assert hasattr(torch,'_torch_compat_install_context')==(runtime=='shim')
if runtime=='oracle':
    assert hasattr(torch,'_C')
    import torch_npu
    torch.npu.set_device(0)
else:
    import jittor as jt
    fallback_before=jt.core.backend_fallback_count()
from transformers import AutoModelForCausalLM, AutoTokenizer
source=Path(os.environ['DS_L3_SOURCE'])/runtime/('rank%d'%rank)
initial=Path(os.environ['DS_MODEL_PATH'])
out=Path(os.environ['DS_L3_OUT'])/runtime/('rank%d'%rank)
out.mkdir(parents=True,exist_ok=True)
assert not (out/'report.json').exists(), 'fresh output required'
training=json.loads((source/'report.json').read_text())
zero_stage=int(os.environ.get('DS_ZERO_STAGE','1'))
assert zero_stage in (1,2,3)
assert training['status']=='passed' and training['steps']==3 and training['zero_stage']==zero_stage
assert training['optimizer_config']['eps']==1e-8 and training['fallback_delta'] in (None,0)
assert training['checkpoint_sha256']=='f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b'
model=AutoModelForCausalLM.from_pretrained(str(initial),local_files_only=True,
    torch_dtype=torch.float32,attn_implementation='eager').to('npu:0')
parameters=list(model.named_parameters()); assert len(parameters)==310
with torch.no_grad():
    for name,p in parameters:
        a=np.load(source/'step2/updated'/(name+'.npy'),allow_pickle=False)
        assert a.shape==tuple(p.shape) and a.dtype==np.float32
        p.copy_(torch.tensor(a,device='npu:0',dtype=torch.float32))
model.eval()
ids=np.load(source/'step2/input_ids.npy',allow_pickle=False)
ids=torch.tensor(ids,device='npu:0',dtype=torch.int64)
def run_forward(m):
    with torch.no_grad():
        y=m(input_ids=ids,labels=ids.clone(),use_cache=False)
    loss=y.loss.detach().cpu().numpy()
    logits=y.logits.detach().cpu().numpy()
    assert np.isfinite(loss).all() and np.isfinite(logits).all()
    return loss,logits
before_loss,before_logits=run_forward(model)
saved=out/'saved-model'
model.save_pretrained(str(saved),safe_serialization=True)
assert list(saved.glob('*.safetensors')), 'no saved model weights'
reloaded=AutoModelForCausalLM.from_pretrained(str(saved),local_files_only=True,
    torch_dtype=torch.float32,attn_implementation='eager').to('npu:0')
reloaded.eval()
loaded=list(reloaded.named_parameters());assert [n for n,_ in loaded]==[n for n,_ in parameters]
max_param_abs=0.0; bad_params=[]
for (name,p),(_,q) in zip(parameters,loaded):
    a=p.detach().cpu().numpy();b=q.detach().cpu().numpy()
    assert a.shape==b.shape and a.dtype==b.dtype
    diff=float(np.max(np.abs(a-b)))
    max_param_abs=max(max_param_abs,diff)
    if diff: bad_params.append(name)
after_loss,after_logits=run_forward(reloaded)
loss_abs=float(np.max(np.abs(before_loss-after_loss)))
logits_abs=float(np.max(np.abs(before_logits-after_logits)))
loss_scale=float(np.max(np.abs(before_loss)))
logits_scale=float(np.max(np.abs(before_logits)))
loss_tol=5e-5+5e-5*loss_scale
logits_tol=5e-5+5e-5*logits_scale
diagnostic=dict(parameter_worst_abs=max_param_abs,changed_parameter_names=bad_params,
    loss_worst_abs=loss_abs,logits_worst_abs=logits_abs,
    loss_tolerance=loss_tol,logits_tolerance=logits_tol)
(out/'roundtrip-diagnostic.json').write_text(json.dumps(diagnostic,indent=2))
assert max_param_abs==0 and loss_abs<=loss_tol and logits_abs<=logits_tol, 'round-trip changed values beyond contract'
tokenizer=AutoTokenizer.from_pretrained(str(initial),local_files_only=True)
text='计图兼容层测试'
encoded=tokenizer.encode(text,add_special_tokens=False)
assert tokenizer.decode(encoded)==text
if runtime=='shim':
    assert jt.core.backend_fallback_count()==fallback_before
else:
    torch.npu.synchronize()
report=dict(status='passed',runtime=runtime,rank=rank,world_size=2,zero_stage=zero_stage,device='npu',
    source_training_report_sha256=hashlib.sha256((source/'report.json').read_bytes()).hexdigest(),
    parameter_count=len(parameters),parameter_worst_abs=max_param_abs,
    changed_parameter_names=bad_params,loss_worst_abs=loss_abs,
    logits_worst_abs=logits_abs,tokenizer_text=text,token_ids=encoded,
    saved_files=[p.name for p in saved.iterdir()],
    fallback_delta=0 if runtime=='shim' else None,
    probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
print(json.dumps(dict(runtime=runtime,rank=rank,event='roundtrip-passed')),flush=True)
