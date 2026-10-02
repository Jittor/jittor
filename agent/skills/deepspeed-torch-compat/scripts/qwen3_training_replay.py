"""Diagnostic replay: common weights vs each runtime's saved training weights."""
import json, os
from pathlib import Path
import numpy as np
runtime=os.environ['DS_RUNTIME']; rank=int(os.environ['RANK'])
if runtime=='shim':
    from jittor_adapters.deepspeed import activate
    activate(device='npu')
import torch
assert hasattr(torch, '_torch_compat_install_context') == (runtime=='shim')
if runtime=='oracle':
    import torch_npu
    torch.npu.set_device(0)
else:
    import jittor as jt
    before=jt.core.backend_fallback_count()
from transformers import AutoModelForCausalLM
model=AutoModelForCausalLM.from_pretrained(os.environ['DS_MODEL_PATH'],local_files_only=True,
    torch_dtype=torch.float32,attn_implementation='eager').to('npu:0')
model.train(); model.config.use_cache=False
source=Path(os.environ['DS_REPLAY_SOURCE'])
out=Path(os.environ['DS_REPLAY_OUT'])/runtime/('rank%d'%rank)
out.mkdir(parents=True,exist_ok=True)
assert not (out/'report.json').exists()
reports=[]
hidden={}
def make_hook(index):
    def hook(module, args, output):
        value=output[0] if isinstance(output,(tuple,list)) else output
        value.retain_grad()
        hidden[index]=value
    return hook
handles=[layer.register_forward_hook(make_hook(i)) for i,layer in enumerate(model.model.layers)]
for case,weight_runtime in [('common','oracle'),('own',runtime)]:
    hidden.clear()
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        for name,p in model.named_parameters():
            a=np.load(source/weight_runtime/('rank%d'%rank)/'step1/updated'/(name+'.npy'))
            p.copy_(torch.tensor(a,device='npu:0',dtype=torch.float32))
    ids=np.load(source/'oracle'/('rank%d'%rank)/'step2/input_ids.npy')
    ids=torch.tensor(ids,device='npu:0',dtype=torch.int64)
    x=model.get_input_embeddings()(ids); x.retain_grad()
    y=model(inputs_embeds=x,labels=ids.clone(),use_cache=False)
    y.loss.backward()
    assert x.grad is not None and x.grad.device.type=='npu'
    for name,v in [('input',x),('input_grad',x.grad),('loss',y.loss),('logits',y.logits)]:
        a=v.detach().cpu().numpy(); assert np.isfinite(a).all()
        np.save(out/(case+'-'+name+'.npy'),a,allow_pickle=False)
    for i,v in hidden.items():
        assert v.grad is not None
        for suffix,t in [('value',v),('grad',v.grad)]:
            a=t.detach().cpu().numpy(); assert np.isfinite(a).all()
            np.save(out/(case+'-layer%02d-'%i+suffix+'.npy'),a,allow_pickle=False)
    torch.npu.synchronize()
    fallback=jt.core.backend_fallback_count()-before if runtime=='shim' else None
    assert fallback in (None,0)
    reports.append(dict(case=case,weight_runtime=weight_runtime,fallback=fallback))
    print(json.dumps(dict(runtime=runtime,rank=rank,case=case,status='recorded')),flush=True)
    del x,y,ids
(out/'report.json').write_text(json.dumps(dict(runtime=runtime,rank=rank,cases=reports),indent=2))
