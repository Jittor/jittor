"""RMSNorm intermediates on the same recorded Qwen3 input and real NPU."""
import hashlib,json,os
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
 before=jt.core.backend_fallback_count()
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
state=Path(os.environ['DS_RMS_STATE'])
out=Path(os.environ['DS_RMS_OUT'])/runtime/('rank%d'%rank)
out.mkdir(parents=True,exist_ok=True)
assert not (out/'report.json').exists()
source=state/'qwen-trace-7d20-v1'/runtime/('rank%d'%rank)
weight_root=state/'qwen-rms-weights'
records={}
def save(key,value):
 assert value.device.type=='npu'
 array=value.detach().cpu().numpy();assert np.isfinite(array).all()
 np.save(out/(key+'.npy'),array,allow_pickle=False)
 records[key]=dict(shape=list(array.shape),dtype=str(array.dtype))
for case in ('input','post'):
 if case=='input':
  data=np.load(source/'input.npy')
  weight=np.load(weight_root/'model.layers.0.input_layernorm.weight.npy')
 else:
  data=np.load(source/'input.npy')+np.load(source/'model_layers_0_self_attn_o_proj.npy')
  weight=np.load(weight_root/'model.layers.0.post_attention_layernorm.weight.npy')
 x=torch.tensor(data,device='npu:0',dtype=torch.float32)
 w=torch.tensor(weight,device='npu:0',dtype=torch.float32)
 module=Qwen3RMSNorm(1024,eps=1e-6).to('npu:0')
 with torch.no_grad(): module.weight.copy_(w)
 variance=x.float().pow(2).mean(-1,keepdim=True)
 inv=torch.rsqrt(variance+1e-6)
 normalized=x.float()*inv
 formula=module.weight*normalized.to(x.dtype)
 direct=module(x)
 for name,tensor in [('input',x),('variance',variance),('inverse',inv),
                     ('normalized',normalized),('formula',formula),('module',direct)]:
  save(case+'_'+name,tensor)
assert jt.core.backend_fallback_count()==before if runtime=='shim' else True
(out/'report.json').write_text(json.dumps(dict(runtime=runtime,rank=rank,records=records,
 probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
print(json.dumps(dict(runtime=runtime,rank=rank,event='finished')),flush=True)
