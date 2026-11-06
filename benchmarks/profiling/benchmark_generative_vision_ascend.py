"""Two-process, same-checkpoint model inference probe; outputs in external state."""
import argparse, contextlib, hashlib, json, os, statistics, subprocess, time
from pathlib import Path
import numpy as np


def mapped(value, fn):
    if isinstance(value,dict): return {k:mapped(v,fn) for k,v in value.items()}
    if isinstance(value,tuple): return tuple(mapped(v,fn) for v in value)
    if isinstance(value,list): return [mapped(v,fn) for v in value]
    return fn(value)


def graph_replay_metadata(model):
    """Read replay counters after timing without executing tensors."""
    state = vars(model).get("_auto_graph_replay")
    if state is None:
        return None
    result = {name: getattr(state, name, None)
              for name in ("seen", "give_up", "unstable", "recaptures")}
    replay = getattr(state, "replay", None)
    if replay is not None:
        result["replay"] = dict(
            stats=dict(replay.stats), refused=replay.refused,
            device_graph_active=bool(replay._cuda_graph),
            device_graph_host_prefix=bool(getattr(replay, "_graph_has_host_prefix", False)),
            host_prefix_retained_bytes=getattr(replay, "_graph_host_prefix_bytes", 0),
            device_graph_refused=replay._graph_refused,
            estimated_retained_bytes=replay._graph_bytes,
            retained_bytes_limit=replay._max_retained_bytes)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--backend',choices=['torch','jittor'],required=True)
    p.add_argument('--device',choices=['cpu','npu'],required=True)
    from _generative_vision_cases import CASES
    p.add_argument('--case',choices=CASES,required=True)
    p.add_argument('--model')
    p.add_argument('--input-npz',help='Shared preprocessed real-image inputs for stock vision cases')
    p.add_argument('--seed',type=int,default=20261020)
    p.add_argument('--tiny',action='store_true')
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--profile',default='default')
    p.add_argument('--dtype',default='float16')
    p.add_argument('--attention',default='eager')
    p.add_argument('--pack',action='store_true')
    p.add_argument('--warmups',type=int,default=3)
    p.add_argument('--samples',type=int,default=5)
    p.add_argument('--slots',type=int,default=2)
    p.add_argument('--output',required=True)
    p.add_argument('--source-root',help='Source checkout for commit metadata; defaults to this script checkout')
    args=p.parse_args()
    from _generative_vision_cases import create_model,make_inputs,make_inputs_from_npz,invoke,extract_output
    from _paths import lab_output_path, resolve_source_root
    if args.samples < 1 or args.slots < 1 or args.warmups < 0:
        p.error('samples/slots must be positive and warmups must be nonnegative')
    if args.prepare and (args.backend != 'torch' or args.device != 'cpu' or not args.tiny):
        p.error('--prepare requires --backend torch --device cpu --tiny')
    if args.pack and args.device != 'npu':
        p.error('--pack requires --device npu')
    if args.input_npz and (args.tiny or args.prepare):
        p.error('--input-npz is for complete pretrained models, not tiny fixtures')
    if args.input_npz and args.case not in ('convnextv2','swinv2','depth_anything_v2','rtdetr_v2'):
        p.error('--input-npz currently supports only the four stock vision prediction models')
    try:
        repo=resolve_source_root(args.source_root)
        out=lab_output_path(args.output,repo)
        if args.prepare:
            if not args.model: p.error('--prepare requires a --model fixture directory')
            args.model=str(lab_output_path(args.model,repo))
        if args.input_npz:
            args.input_npz=str(lab_output_path(args.input_npz,repo))
            if not Path(args.input_npz).is_file():
                p.error('--input-npz must name an existing archive')
    except ValueError as error:
        p.error(str(error))
    args.source_root=str(repo)
    args.output=str(out)
    input_sha256=(hashlib.sha256(Path(args.input_npz).read_bytes()).hexdigest()
                  if args.input_npz else None)
    out.parent.mkdir(parents=True,exist_ok=True)
    with contextlib.ExitStack() as stack:
        jt=None
        if args.backend=='jittor':
            import jittor as jt
            stack.enter_context(jt.runtime.scope(use_cuda=0,use_acl=0,backend_fallback='error',use_parallel_op_compiler=0,acl_allow_hf32=1))
            import torch
            assert hasattr(torch,'_torch_compat_install_context')
            sync=lambda:jt.sync_all(True)
        else:
            import torch
            assert hasattr(torch,'_C') and not hasattr(torch,'_torch_compat_install_context')
            if args.device=='npu':
                import torch_npu
                torch.npu.set_device(0)
                torch.npu.set_compile_mode(jit_compile=False)
                torch.npu.matmul.allow_hf32=True
                sync=torch.npu.synchronize
            else: sync=lambda:None
        torch.set_num_threads(4)
        torch.manual_seed(42)
        dtype=getattr(torch,'float32' if args.tiny else args.dtype)
        print('PHASE load',args.case,flush=True)
        started=time.perf_counter()
        model=create_model(torch,args.case,model_path=None if args.prepare else args.model,tiny=args.tiny,dtype=dtype,attn_implementation=args.attention)
        model.eval()
        if args.prepare:
            assert args.backend=='torch' and args.tiny
            model.save_pretrained(args.model,safe_serialization=True)
            print('PREPARED',args.model,flush=True)
        parameters=sum(p.numel() for p in model.parameters())
        rng=np.random.default_rng(args.seed)
        slots=[]
        for index in range(args.slots):
            kwargs,spec=(make_inputs_from_npz(torch,args.case,model,args.input_npz)
                         if args.input_npz else
                         make_inputs(torch,args.case,model,rng,tiny=args.tiny,profile=args.profile))
            if args.input_npz:
                spec["input_sha256"]=input_sha256
            else:
                spec["input_seed"]=args.seed
            slots.append(kwargs)
        if jt is not None and args.device=='npu':
            stack.enter_context(jt.runtime.scope(use_cuda=1,use_acl=1))
        device=('cuda' if jt is not None else 'npu:0') if args.device=='npu' else 'cpu'
        model.to(device=device)
        slots=[mapped(x,lambda v:v.to(device=device) if torch.is_tensor(v) else v) for x in slots]
        if jt is not None:
            from jittor._runtime.fallback import forbid_backend_fallbacks
            stack.enter_context(forbid_backend_fallbacks())
            fallback_before=jt.core.backend_fallback_count()
        pack_count=pack_skipped=0
        pack_seconds=0.0
        with torch.no_grad():
            if jt is not None:
                runtime_policy={name:int(getattr(jt.flags,name)) for name in (
                    'no_grad','amp_reg','acl_allow_hf32','auto_graph_replay',
                    'auto_graph_replay_bytes','auto_graph_replay_retain_bytes')}
            elif args.device=='npu':
                runtime_policy=dict(jit_compile=False,
                    matmul_allow_hf32=bool(torch.npu.matmul.allow_hf32),
                    conv_allow_hf32=bool(torch.npu.conv.allow_hf32))
            else:
                runtime_policy=dict(num_threads=int(torch.get_num_threads()))
            if args.pack and args.device=='npu':
                from benchmark_diffusion_ascend import pack_fp16_linears
                pack_started=time.perf_counter()
                model,pack_count,pack_skipped=pack_fp16_linears(model,torch,jt=jt,torch_npu=(torch_npu if jt is None else None))
                sync()
                pack_seconds=time.perf_counter()-pack_started
            sync()
            load_seconds=time.perf_counter()-started
            def call(kw):
                result=extract_output(args.case,invoke(args.case,model,kw))
                for key,value in result.items():
                    assert torch.is_tensor(value),key
                    if args.device=='npu': assert value.device.type in ('cuda','npu'),(key,value.device)
                    else: assert value.device.type=='cpu',(key,value.device)
                sync()
                return result
            print('PHASE cold_forward',flush=True)
            cold_started=time.perf_counter()
            result=call(slots[0]); cold=time.perf_counter()-cold_started
            print('COLD_SECONDS',cold,flush=True)
            for i in range(args.warmups):
                call(slots[i%len(slots)])
                print('WARMUP',i,flush=True)
            timings=[]
            for i in range(args.samples):
                sync(); started=time.perf_counter(); result=call(slots[i%len(slots)])
                timings.append(time.perf_counter()-started)
                print('SAMPLE',i,timings[-1],flush=True)
            arrays={}
            for i,kw in enumerate(slots):
                result=call(kw)
                for key,value in result.items():
                    array=value.detach().float().cpu().numpy()
                    if not np.isfinite(array).all(): raise RuntimeError('nonfinite '+key)
                    arrays[f'slot{i}::{key}']=array
            np.savez(str(out)+'.npz',**arrays)
        import transformers,diffusers
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
        diff=subprocess.check_output(['git','diff','HEAD'],cwd=repo)
        report=dict(vars(args),parameters=parameters,spec=spec,torch_file=torch.__file__,torch_version=torch.__version__,transformers=transformers.__version__,diffusers=diffusers.__version__,load_seconds=load_seconds,cold_seconds=cold,seconds=timings,median_seconds=statistics.median(timings),p95_seconds=float(np.percentile(timings,95)),pack_count=pack_count,pack_skipped=pack_skipped,pack_seconds=pack_seconds,input_sha256=input_sha256,runtime_flags=({'use_cuda':int(jt.flags.use_cuda),'use_acl':int(jt.flags.use_acl)} if jt is not None else None),commit=commit,dirty_sha256=hashlib.sha256(diff).hexdigest(),fallback_count=(jt.core.backend_fallback_count()-fallback_before if jt is not None else None),output_shapes={k:list(v.shape) for k,v in arrays.items()})
        report['inference_mode']='explicit_fp16_linear_snapshot' if args.pack else 'stock_model'
        report['shared_input_slots']=bool(args.input_npz)
        report['runtime_policy']=runtime_policy
        report['harness_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        report['case_helper_sha256']={name:hashlib.sha256(
            Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('_generative_vision_cases.py','_stock_vision_cases.py')}
        report['graph_replay'] = graph_replay_metadata(model) if jt is not None else None
        if jt is not None and report['fallback_count']!=0: raise RuntimeError('backend fallback attempted')
        out.write_text(json.dumps(report,indent=2))
        print('PASS',args.case,report['median_seconds'],flush=True)
if __name__=='__main__': main()
