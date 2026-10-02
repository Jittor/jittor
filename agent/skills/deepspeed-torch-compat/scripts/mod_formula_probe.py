import json,os
from pathlib import Path
rt=os.environ['DS_RUNTIME'];rank=int(os.environ['RANK'])
if rt=='shim':
 from jittor_adapters.deepspeed import activate
 activate(device='npu')
import torch
assert hasattr(torch,'_torch_compat_install_context')==(rt=='shim')
if rt=='oracle':
 import torch_npu
 torch.npu.set_device(0)
else:
 import jittor as jt
 before=jt.core.backend_fallback_count()
x=torch.tensor([-7,-5,-1,0,1,5,7],device='npu:0',dtype=torch.int64)
records={}
for d in (3,-3):
 q=x//d
 r=x-q*d
 records[str(d)]={'q':q.detach().cpu().numpy().tolist(),'r':r.detach().cpu().numpy().tolist()}
 if rt=='oracle':
  records[str(d)]['native_mod']=(x%d).detach().cpu().numpy().tolist()
if rt=='shim':assert jt.core.backend_fallback_count()==before
out=Path(os.environ['DS_MOD_OUT'])/rt/f'rank{rank}'
out.mkdir(parents=True,exist_ok=True)
(out/'report.json').write_text(json.dumps(records,indent=2))
print(json.dumps({'runtime':rt,'rank':rank,'status':'passed'}),flush=True)
