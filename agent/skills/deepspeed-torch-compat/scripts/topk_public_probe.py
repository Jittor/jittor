import json,os
from pathlib import Path
runtime=os.environ['DS_RUNTIME'];rank=int(os.environ['RANK'])
if runtime=='shim':
 from jittor_adapters.deepspeed import activate
 activate(device='npu')
import torch
assert hasattr(torch,'_torch_compat_install_context')==(runtime=='shim')
if runtime=='oracle':
 import torch_npu
 torch.npu.set_device(0)
else:
 import jittor as jt
 before=jt.core.backend_fallback_count()
cases={'unique':[[1.,4.,3.,2.]],
       'ties':[[3.,3.,2.,float('-inf')]],
       'all_neginf':[[float('-inf')]*4],
       'smallest':[[1.,4.,3.,2.]]}
result={}
for name,values in cases.items():
 x=torch.tensor(values,device='npu:0',dtype=torch.float32)
 got=torch.topk(x,3,dim=1,largest=(name!='smallest'))
 v=got.values.detach().cpu().numpy()
 i=got.indices.detach().cpu().numpy()
 assert len(set(i[0].tolist()))==3
 result[name]={'values':v.tolist(),'indices':i.tolist(),
               'shape':list(v.shape),'dtype':str(v.dtype)}
if runtime=='shim':assert jt.core.backend_fallback_count()==before
out=Path(os.environ['DS_TOPK_OUT'])/runtime/f'rank{rank}'
out.mkdir(parents=True,exist_ok=True)
(out/'report.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'runtime':runtime,'rank':rank,'status':'passed'}),flush=True)
