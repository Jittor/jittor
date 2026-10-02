"""Trace Qwen3 first training backward before any optimizer step."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
runtime=os.environ['DS_RUNTIME']
rank=int(os.environ['RANK'])
assert runtime in ('oracle','shim') and int(os.environ['WORLD_SIZE'])==2
if runtime=='shim':
    from jittor_adapters.deepspeed import activate
    activate(device='npu')
import torch
assert hasattr(torch,'_torch_compat_install_context') == (runtime=='shim')
if runtime=='oracle':
    assert hasattr(torch,'_C')
    import torch_npu
    torch.npu.set_device(0)
else:
    import jittor as jt
    fallback_before=jt.core.backend_fallback_count()
from transformers import AutoModelForCausalLM
model_path=Path(os.environ['DS_MODEL_PATH'])
source=Path(os.environ['DS_TRACE_SOURCE'])
out=Path(os.environ['DS_TRACE_OUT'])/runtime/('rank%d'%rank)
out.mkdir(parents=True,exist_ok=True)
assert not (out/'report.json').exists(), 'fresh trace output required'
model=AutoModelForCausalLM.from_pretrained(str(model_path),local_files_only=True,
    torch_dtype=torch.float32,attn_implementation='eager').to('npu:0')
model.train(); model.config.use_cache=False
assert len(list(model.named_parameters()))==310
seen={}
names={'model.layers.0.input_layernorm', 'model.layers.0.self_attn.q_proj',
       'model.layers.0.self_attn.k_proj','model.layers.0.self_attn.v_proj',
       'model.layers.0.self_attn.o_proj','model.layers.0.post_attention_layernorm',
       'model.layers.0.mlp.gate_proj','model.layers.0.mlp.up_proj',
       'model.layers.0.mlp.down_proj','model.layers.0', 'model.layers.27'}
def hook_for(name):
    def hook(_module,_inputs,output):
        tensor=output[0] if isinstance(output,(tuple,list)) else output
        assert tensor.device.type=='npu' and tensor.dtype==torch.float32
        tensor.retain_grad(); seen[name]=tensor
    return hook
handles=[]
for name,module in model.named_modules():
    if name in names: handles.append(module.register_forward_hook(hook_for(name)))
assert len(handles)==len(names)
ids=np.load(source/'oracle'/('rank%d'%rank)/'step0/input_ids.npy')
ids=torch.tensor(ids,dtype=torch.int64,device='npu:0')
x=model.get_input_embeddings()(ids); x.retain_grad()
y=model(inputs_embeds=x,labels=ids.clone(),use_cache=False)
y.loss.backward()

def save(name,tensor):
    assert tensor is not None and tensor.device.type=='npu'
    array=tensor.detach().cpu().numpy()
    assert np.isfinite(array).all()
    np.save(out/(name+'.npy'),array,allow_pickle=False)
    return {'shape':list(array.shape),'dtype':str(array.dtype),'max_abs':float(np.max(np.abs(array))) if array.size else 0.0}
records={}
for name,value in [('input',x),('input_grad',x.grad),('logits',y.logits),('loss',y.loss),
                   ('v_proj_weight_grad',model.model.layers[0].self_attn.v_proj.weight.grad)]:
    records[name]=save(name,value)
for name,tensor in seen.items():
    key=name.replace('.','_')
    records[key]=save(key,tensor)
    records[key+'_grad']=save(key+'_grad',tensor.grad)
assert set(seen)==names
fallback=jt.core.backend_fallback_count()-fallback_before if runtime=='shim' else None
assert fallback in (None,0)
torch.npu.synchronize()
report=dict(runtime=runtime,rank=rank,world_size=2,device='npu',dtype='float32',
    checkpoint_sha256='f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b',
    source=str(source),records=records,fallback_delta=fallback,
    torch_is_shim=hasattr(torch,'_torch_compat_install_context'),
    probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(out/'report.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'runtime':runtime,'rank':rank,'event':'finished','records':len(records)}),flush=True)
