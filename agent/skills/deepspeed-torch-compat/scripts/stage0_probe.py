import os as _os, sys as _sys
if _os.environ.get('DS_EXPERIMENTAL_SOURCE'):
    _sys.path.insert(0, _os.environ['DS_EXPERIMENTAL_SOURCE'])
"""Fixed-fixture DeepSpeed stage-0 trajectory probe; experimental, not a support claim."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def make_fixture(path):
    shapes={'0.weight':(8,4),'0.bias':(8,), '2.weight':(2,8),'2.bias':(2,)}
    values={name:((np.arange(np.prod(shape),dtype=np.float32).reshape(shape)%13)-6)/32 for name,shape in shapes.items()}
    for step in range(3):
        values['input/'+str(step)]=(np.arange(8,dtype=np.float32).reshape(2,4)-3+step)/10
        values['target/'+str(step)]=np.array([[.1,-.2],[-.1,.3]],dtype=np.float32)+step/100
    np.savez(path,**values)


def run(args):
    import os
    import socket
    import torch
    shim=hasattr(torch,'_torch_compat_install_context')
    assert shim == (args.runtime=='shim'), 'Wrong torch runtime'
    if args.runtime=='oracle':
        assert not hasattr(torch,'_torch_compat_install_context')
        assert hasattr(torch._C, '_c10d_init'), 'Oracle must be binary PyTorch'
        if args.device=='npu':
            import torch_npu
            assert torch.npu.is_available()
            torch.npu.set_device(0)
    else:
        import jittor as jt
        jt.flags.use_cuda=int(args.device=='npu')
        assert bool(jt.flags.use_cuda)==(args.device=='npu')
        if args.device == 'npu':
            assert jt.compiler.has_acl and jt.flags.use_cuda, 'Real ACL execution required'
        fallback_start=jt.core.backend_fallback_count()
    import deepspeed
    assert deepspeed.__version__=='0.17.6'
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        port=sock.getsockname()[1]
    os.environ.update(RANK='0',WORLD_SIZE='1',LOCAL_RANK='0',MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port))
    device='npu:0' if args.device=='npu' else 'cpu'
    arrays=np.load(args.fixture,allow_pickle=False)
    model=torch.nn.Sequential(torch.nn.Linear(4,8),torch.nn.Tanh(),torch.nn.Linear(8,2)).to(device)
    output={}
    def save(name,tensor):
        assert tensor is not None, 'Missing tensor: '+name
        data=tensor.detach().cpu().numpy().copy()
        assert np.isfinite(data).all(), name
        output[name]=data
    with torch.no_grad():
        for name,param in model.named_parameters():
            param.copy_(torch.tensor(arrays[name],dtype=torch.float32,device=device))
            assert param.device.type==args.device
            save('initial/'+name,param)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01,betas=(.9,.99),eps=1e-8)
    engine=None
    try:
        engine,_,_,_=deepspeed.initialize(model=model,optimizer=optimizer,config={
            'train_micro_batch_size_per_gpu':2,'gradient_accumulation_steps':1,
            'zero_optimization':{'stage':0},'fp16':{'enabled':False},'bf16':{'enabled':False},
            'steps_per_print':1000})
        def describe_named(named):
            result = []
            for name, value in named:
                assert value.device.type == args.device, name
                data = value.detach().cpu().numpy().copy()
                result.append({"name": name, "shape": list(data.shape),
                               "dtype": str(data.dtype),
                               "requires_grad": bool(value.requires_grad),
                               "sha256": hashlib.sha256(data.tobytes()).hexdigest()})
            return result
        from deepspeed.accelerator import get_accelerator
        import deepspeed.runtime.utils as ds_utils
        accelerator = get_accelerator()
        assert accelerator.device_name() == args.device
        assert ds_utils.torch_memory_reserved.__self__ is accelerator, "stale import-time accelerator binding"
        constructed = {"status": "constructed", "runtime": args.runtime,
                       "device": args.device, "deepspeed": deepspeed.__version__,
                       "engine_type": type(engine).__module__ + "." + type(engine).__name__,
                       "parameters": describe_named(engine.module.named_parameters()),
                       "buffers": describe_named(engine.module.named_buffers()),
                       "reported_communication_backend": str(torch.distributed.get_backend()),
                       "fixture_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
                       "engine_sha256": hashlib.sha256(Path(deepspeed.runtime.engine.__file__).read_bytes()).hexdigest()}
        args.out.with_suffix(".construction.json").write_text(json.dumps(constructed, indent=2))
        print("DEEPSPEED_CONSTRUCTED=" + json.dumps(constructed), flush=True)
        engine.eval()
        assert not engine.module.training
        engine.train()
        assert engine.module.training
        for step in range(3):
            x=torch.tensor(arrays['input/'+str(step)],dtype=torch.float32,device=device,requires_grad=True)
            target=torch.tensor(arrays['target/'+str(step)],dtype=torch.float32,device=device)
            prediction=engine(x)
            assert prediction.device.type==args.device
            loss=(prediction-target).square().mean()
            engine.backward(loss)
            save('output/'+str(step),prediction)
            save('loss/'+str(step),loss)
            assert x.grad is not None and x.grad.device.type == args.device
            save('input_grad/'+str(step),x.grad)
            for name,param in model.named_parameters():
                assert param.grad is not None,name
                assert param.grad.device.type==args.device,name
                save('param_grad/'+str(step)+'/'+name,param.grad)
            engine.step()
            for name,param in model.named_parameters():
                save('updated/'+str(step)+'/'+name,param)
        if shim:
            assert jt.core.backend_fallback_count()==fallback_start
        np.savez(args.out.with_suffix('.npz'),**output)
        report={'status':'passed','runtime':args.runtime,'torch':str(torch.__version__),
                'torch_origin':getattr(torch,'__file__',None),'deepspeed':deepspeed.__version__,
                'deepspeed_origin':str(Path(deepspeed.__file__).resolve()),
                'engine_sha256':hashlib.sha256(Path(deepspeed.runtime.engine.__file__).read_bytes()).hexdigest(),
                'device':args.device,'dtype':'float32','steps':3,'zero_stage':0,
                'reported_communication_backend': constructed['reported_communication_backend'],
                'scope': 'single-process numeric trajectory only; not a library support claim',
                'parameters':list(dict(model.named_parameters())),
                'fallback_delta':0 if shim else None,'fixture':str(args.fixture),'fixture_sha256':hashlib.sha256(args.fixture.read_bytes()).hexdigest()}
        args.out.with_suffix('.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report))
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def compare(args):
    ref=np.load(args.oracle.with_suffix('.npz'),allow_pickle=False)
    got=np.load(args.candidate.with_suffix('.npz'),allow_pickle=False)
    rm=json.loads(args.oracle.with_suffix('.json').read_text())
    gm=json.loads(args.candidate.with_suffix('.json').read_text())
    assert rm['runtime']=='oracle' and gm['runtime']=='shim'
    for key in ('device','dtype','steps','zero_stage','parameters','deepspeed','fixture_sha256','engine_sha256'):
        assert rm[key]==gm[key],key
    assert set(ref.files)==set(got.files)
    summary={}
    for field in sorted({key.split('/')[0] for key in ref.files}):
        keys=[k for k in ref.files if k.split('/')[0]==field]
        scale=max(float(np.max(np.abs(ref[k]))) for k in keys)
        worst=0.
        for key in keys:
            assert ref[key].shape==got[key].shape and ref[key].dtype==got[key].dtype,key
            assert np.isfinite(got[key]).all(),key
            error=float(np.max(np.abs(ref[key]-got[key])))
            worst=max(worst,error)
            assert error<=2e-5+2e-4*scale,(key,error,scale)
        summary[field]={'max_abs':worst,'field_scaled_rel':worst/max(scale,1e-12),'reference_field_scale':scale}
    args.out.write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('fixture');p.add_argument('--out',type=Path,required=True)
    p=sub.add_parser('run');p.add_argument('--fixture',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--runtime',choices=('oracle','shim'),required=True);p.add_argument('--device',choices=('cpu','npu'),required=True)
    p=sub.add_parser('compare');p.add_argument('--oracle',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.mode=='fixture':make_fixture(args.out)
    elif args.mode=='run':run(args)
    else:compare(args)

if __name__=='__main__':main()
