import json, os
from jittor_adapters.deepspeed import activate
activate(device='npu')
import torch
import jittor as jt
import numpy as np
x=torch.tensor([[1,2,3,4]],device='npu:0',dtype=torch.int64)
method=os.environ['REPEAT_METHOD']
if method=='api':
 y=x.repeat_interleave(2,dim=0)
elif method=='broadcast':
 y=x.unsqueeze(1).broadcast((1,2,4)).reshape((2,4))
elif method=='arange':
 y=x[jt.arange(2).int64()//2,:]
else:
 y=x[jt.array([0,0]).int64(),:]
a=y.detach().cpu().numpy()
assert np.array_equal(a,np.array([[1,2,3,4],[1,2,3,4]])),a
assert jt.core.backend_fallback_count()==0
print(json.dumps({'method':method,'shape':list(a.shape),'values':a.tolist(),'fallback':0}),flush=True)
