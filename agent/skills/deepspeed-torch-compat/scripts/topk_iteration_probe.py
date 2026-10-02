import json
from jittor_adapters.deepspeed import activate
activate(device='npu')
import torch
import jittor as jt
from jittor.compat.torch.installers import tensor as owner
def small_topk(x,k):
    taken=jt.zeros(x.shape,dtype='float32')
    vals=[];idxs=[]
    for _ in range(k):
        candidates=jt.where(taken>0,float('-inf'),x)
        idx,maxv=owner._NATIVE_ARGMAX(candidates,dim=1,keepdims=True)
        alternate,_=owner._NATIVE_ARGMAX(1-taken,dim=1,keepdims=True)
        idx=jt.where(maxv==float('-inf'),alternate,idx)
        vals.append(jt.gather(x,1,idx));idxs.append(idx.int64())
        taken=jt.scatter(taken,1,idx,jt.ones(idx.shape,dtype='float32'))
    return jt.concat(vals,dim=1),jt.concat(idxs,dim=1)
cases=[[[1.,4.,3.,2.]],[[float('-inf')]*4],[[3.,3.,2.,float('-inf')]]]
for case in cases:
    x=torch.tensor(case,device='npu:0',dtype=torch.float32)
    v,i=small_topk(x,3)
    values=v.numpy();indices=i.numpy()
    assert len(set(indices[0].tolist()))==3,indices
    assert all(float(values[0,j])==float(x.numpy()[0,indices[0,j]]) for j in range(3))
    print(json.dumps({'values':values.tolist(),'indices':indices.tolist()}),flush=True)
assert jt.core.backend_fallback_count()==0
