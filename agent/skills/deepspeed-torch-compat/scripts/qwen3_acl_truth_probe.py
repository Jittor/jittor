"""True torch vs Jittor ACL truth reductions and isin on an allocated NPU."""
import json,os,hashlib
from pathlib import Path
import numpy as np
runtime=os.environ['DS_RUNTIME'];rank=int(os.environ['RANK'])
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
x=torch.tensor([[True,False,True],[False,False,False]],device='npu:0')
cases={'any_last':x.any(dim=-1),'any_last_keep':x.any(dim=-1,keepdim=True),
       'all_first':x.all(dim=0),'all_first_keep':x.all(dim=0,keepdim=True),
       'any_multi_keep':x.any(dim=(0,1),keepdim=True),
       'isin':torch.isin(torch.tensor([1,2,3],device='npu:0'),
                         torch.tensor([2],device='npu:0'))}
records={}
for name,v in cases.items():
 assert v.device.type=='npu'
 a=v.detach().cpu().numpy()
 records[name]=dict(shape=list(a.shape),dtype=str(a.dtype),values=a.tolist())
if runtime=='shim':assert jt.core.backend_fallback_count()==before
else:torch.npu.synchronize()
out=Path(os.environ['DS_TRUTH_OUT'])/runtime/('rank%d'%rank)
out.mkdir(parents=True,exist_ok=True)
assert not (out/'report.json').exists()
(out/'report.json').write_text(json.dumps(dict(runtime=runtime,rank=rank,cases=records,
 probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
print(json.dumps(dict(runtime=runtime,rank=rank,event='truth-passed')),flush=True)
