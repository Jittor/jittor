"""Same public collective math, separate binary Torch / native MPI or NCCL."""


def main():
    import argparse
    from contextlib import nullcontext
    from datetime import timedelta
    import json
    import os
    from pathlib import Path

    p=argparse.ArgumentParser()
    p.add_argument('--runtime',choices=['jittor','torch'],required=True)
    p.add_argument('--device',choices=['cpu','cuda'],required=True)
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args()
    import numpy as np
    rank=int(os.environ.get('OMPI_COMM_WORLD_RANK',os.environ.get('RANK','-1')))
    if rank not in (0, 1):
        raise ValueError('Expected an actual two-rank launch with RANK or OMPI_COMM_WORLD_RANK')
    a.root.mkdir(parents=True, exist_ok=True)
    # Reserve each rank before initialization, preserving partial/failed runs.
    (a.root / ('rank%d.claim' % rank)).open('x').close()
    if a.runtime=='jittor':
        import jittor as jt
        jt.flags.use_cuda=int(a.device=='cuda')
        from jittor._runtime import forbid_backend_fallbacks
        before=jt.core.backend_fallback_count()
    import torch
    import torch.distributed as dist
    from torch.distributed.nn import all_reduce as differentiable_all_reduce
    if a.runtime=='torch':
        assert not hasattr(torch,'_torch_compat_install_context')
        assert torch._C.__file__.endswith('.so')
    else:
        assert hasattr(torch,'_torch_compat_install_context')
    backend='nccl' if a.device=='cuda' else ('mpi' if a.runtime=='jittor' else 'gloo')
    if a.device=='cuda':
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    dist.init_process_group(backend=backend,rank=rank,world_size=2,
        init_method=(a.root/'store').resolve().as_uri(),timeout=timedelta(seconds=90))
    assert dist.get_rank() == rank and dist.get_world_size() == 2
    arrays={}
    seen=[]
    def tensor(data,dtype):
        out=torch.tensor(data,dtype=dtype,device=a.device)
        seen.append(out)
        return out

    def snap(key,value,bits=False):
        assert value.device.type==a.device
        if a.runtime=='jittor':
            value.sync()
            assert value.location()==('device' if a.device=='cuda' else 'cpu')
        if bits:
            value=value.detach().view(torch.int16)
        else:
            value=value.detach().float()
        arrays[key]=np.array(value.cpu().numpy(),copy=True)

    with forbid_backend_fallbacks() if a.runtime=='jittor' else nullcontext():
        group=dist.new_group([0,1],backend=backend)
        patterns=np.asarray([0x0000,0x8000,0x0001,0x007f,0x0080,0x3f80,0x7f7f,0x7f80,0xff80,0x7fc1,0xffc2,0x7fa1],dtype='uint16')
        local=tensor(np.roll(patterns,rank).view('int16'),torch.int16).view(torch.bfloat16)
        for root in (0,1):
            value=local.clone()
            alias=value.detach()
            pointer=value.data_ptr()
            dist.broadcast(value,src=root,group=group)
            assert value.data_ptr()==alias.data_ptr()==pointer
            snap('broadcast%d'%root,value,True)
            snap('broadcast%d_alias'%root,alias,True)
        # This is the flattened data/Parameter alias used by ZeRO synchronization.
        out=torch.nn.Parameter(torch.zeros((24,),dtype=torch.bfloat16,device=a.device))
        alias=out.data
        detached=out.detach()
        view=out.detach()[3:18]
        old=out.detach().clone()
        pointer=out.data_ptr()
        with torch.no_grad():
            dist.all_gather_into_tensor(alias,local,group=group)
        assert out.data_ptr()==alias.data_ptr()==detached.data_ptr()==pointer
        assert out.is_leaf and out.requires_grad
        for key,value in [('gather',out),('gather_alias',alias),('gather_detach',detached),('gather_view',view),('gather_old',old)]:
            snap(key,value,True)
        for dtype in (torch.uint8,torch.float32,torch.bfloat16):
            pair=([200,250,3],[3,2,200]) if dtype==torch.uint8 else ([-2.,1.5,-.25],[-.5,-3.,2.])
            for name,op in [('sum',dist.ReduceOp.SUM),('max',dist.ReduceOp.MAX),('min',dist.ReduceOp.MIN),('product',dist.ReduceOp.PRODUCT)]:
                value=tensor(pair[rank],dtype)
                alias=value.detach()
                pointer=value.data_ptr()
                dist.all_reduce(value,op=op,group=group)
                assert value.data_ptr()==alias.data_ptr()==pointer
                snap(str(dtype)+'_'+name,value)
                snap(str(dtype)+'_'+name+'_alias',alias)
        ties=([0x3f80,0x3f81,0x0001],[0x3b80,0x3b80,0x0002])
        value=tensor(np.asarray(ties[rank],dtype='uint16').view('int16'),torch.int16).view(torch.bfloat16)
        dist.all_reduce(value,group=group)
        snap('bf16_sum_rounding_bits',value,True)
        value=tensor([.5,1.25],torch.bfloat16).requires_grad_()
        output=differentiable_all_reduce(value,group=group)
        output.float().sum().mul(rank+1).backward()
        snap('sum_autograd_forward',output)
        snap('sum_autograd_gradient',value.grad)
        if a.runtime=='jittor':
            assert jt.core.backend_fallback_count()==before
        metadata={'runtime':a.runtime,'transport':backend,'rank':rank,'world_size':2,'device':a.device,'torch_origin':torch.__file__,'torch_version':torch.__version__,'binary_extension':getattr(getattr(torch,'_C',None),'__file__',None),'fallback_delta':0 if a.runtime=='jittor' else None,'numeric_fields':len(arrays)}
        arrays['metadata']=np.asarray(json.dumps(metadata,sort_keys=True))
        path=a.root/('rank%d.npz'%rank)
        assert not path.exists()
        np.savez(path,**arrays)
        print(json.dumps(metadata,sort_keys=True),flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
