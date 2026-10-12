"""Analytic multi-rank native gradients, bucketing and fail-stop contracts."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import jittor as jt
from jittor import distributed as dist, nn
from jittor.nn.parallel import DistributedDataParallel
import jittor.nn.parallel.distributed_data_parallel as reducer


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--failure',default='')
    p.add_argument('--reference',action='store_true')
    p.add_argument('--steps',type=int,default=40)
    a=p.parse_args()
    out=Path(a.output)
    out.mkdir(parents=True,exist_ok=True)
    dist.init_process_group('nccl')
    jt.flags.use_cuda=1
    rank,world=dist.get_rank(),dist.get_world_size()
    reducer._BUCKET_BYTES=512
    reducer._BUCKET_INPUTS=4

    class Analytic(nn.Module):
        def __init__(self):
            self.a=jt.array(np.linspace(.1,.8,33,dtype=np.float32)).cuda(0)
            self.b=jt.array(np.linspace(.2,.9,47,dtype=np.float32)).cuda(0)
            self.c=jt.array(np.linspace(.3,1,73,dtype=np.float64),dtype='float64').cuda(0)
            self.d=jt.array(np.linspace(.4,1.1,41,dtype=np.float64),dtype='float64').cuda(0)
            self.unused=jt.ones(3,dtype='float32').cuda(0)
            self.one_rank=jt.ones(5,dtype='float32').cuda(0)
            self.empty=jt.empty((0,),dtype='float32').cuda(0)

        def execute(self, step):
            result=jt.array(0.,dtype='float64').cuda(0)
            for index,name in enumerate(['a','b','c','d']):
                param=getattr(self,name,None)
                if param is None: continue
                g=(np.arange(param.numel(),dtype='float64')%7+index+1+step*.01)*(rank+1)
                result=result+(param*jt.array(g,dtype=str(param.dtype)).cuda(0)).sum().float64()
            if rank==0:
                result=result+self.one_rank.sum().float64()*3
            return result

    try:
        raw=Analytic()
        model=DistributedDataParallel(raw, device_ids=[0], output_device=0)
        if a.reference:
            # The ungrouped, unbatched path is an independent NCCL ordering
            # reference; it still checks the same model lifecycle.
            model.process_group._get_backend_name=lambda:'reference'
        parameters=list(raw.parameters())
        ordered=parameters if rank%2==0 else parameters[::-1]
        optimizer=jt.optim.SGD(ordered,lr=.001,momentum=.9)
        expected={name:p.numpy().astype('float64') for name,p in raw.named_parameters()}
        momentum={name:np.zeros_like(value) for name,value in expected.items()}
        errors=[]
        for step in range(a.steps):
            optimizer.zero_grad()
            if a.failure and step==2 and rank==0:
                if a.failure=='replace': raw.a=jt.ones_like(raw.a)
                elif a.failure=='add': raw.new=jt.ones(3).cuda(0)
                elif a.failure=='remove': del raw.b
                elif a.failure=='trainable': raw.a.stop_grad()
                elif a.failure=='missing': optimizer.param_groups[0]['params']=[p for p in optimizer.param_groups[0]['params'] if p is not raw.a]
                elif a.failure=='allmissing': optimizer.param_groups[0]['params']=[]
                elif a.failure=='duplicate': optimizer.add_param_group({'params':[raw.a]})
                else: raise ValueError(a.failure)
            # Accumulate one unsynchronised microbatch then one synchronised
            # microbatch; all rank gradients differ and one leaf is rank-only.
            with model.no_sync():
                optimizer.backward(model(step)/2)
            optimizer.backward(model(step)/2)
            grads={id(param):grad for group in optimizer.param_groups
                   for param,grad in zip(group['params'],group['grads'])}
            for index,name in enumerate(['a','b','c','d']):
                param=getattr(raw,name)
                wanted=(np.arange(param.numel())%7+index+1+step*.01)*(world+1)/2
                got=grads[id(param)].numpy().reshape(-1)
                np.testing.assert_allclose(got,wanted,rtol=2e-6,atol=2e-6)
                errors.append(float(np.max(np.abs(got-wanted))))
                momentum[name]=.9*momentum[name]+wanted.reshape(param.shape)
                expected[name]-=.001*momentum[name]
            assert np.all(grads[id(raw.unused)].numpy()==0)
            np.testing.assert_allclose(grads[id(raw.one_rank)].numpy(),3/world,rtol=1e-6)
            momentum['one_rank']=.9*momentum['one_rank']+3/world
            expected['one_rank']-=.001*momentum['one_rank']
            optimizer.step()
            jt.sync_all(True)
            for name,param in raw.named_parameters():
                np.testing.assert_allclose(param.numpy(),expected[name],rtol=1e-5,atol=5e-6)
            assert not jt.core._cuda_stream_join_pending(1,jt.current_device())
        np.savez(out/('rank%d.npz'%rank),**{name:p.numpy() for name,p in raw.named_parameters()})
        (out/('result-rank%d.json'%rank)).write_text(json.dumps(
            {'rank':rank,'steps':a.steps,'max_gradient_abs_error':max(errors),'buckets':len(model._jittor_ddp_state.buckets)}))
        print('ANALYTIC_DDP_OK rank=%d steps=%d max_error=%g'%(rank,a.steps,max(errors)),flush=True)
    finally:
        dist.destroy_process_group()


if __name__=='__main__': main()
