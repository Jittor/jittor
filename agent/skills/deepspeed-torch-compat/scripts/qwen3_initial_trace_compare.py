"""Compare first Qwen3 backward traces and ZeRO-1 reduction, preserving failures."""
import argparse,json
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser()
parser.add_argument('--trace-root',required=True)
parser.add_argument('--training-root',required=True)
parser.add_argument('--out',required=True)
a=parser.parse_args()
root=Path(a.trace_root); train=Path(a.training_root)
reports={(runtime,rank):json.loads((root/runtime/('rank%d'%rank)/'report.json').read_text()) for runtime in ('oracle','shim') for rank in range(2)}
assert all(r['device']=='npu' and r['world_size']==2 and r['fallback_delta'] in (None,0) for r in reports.values())
keys=set(reports['oracle',0]['records'])
assert all(set(r['records'])==keys for r in reports.values())
rows=[]
for name in sorted(keys):
    reference=[np.load(root/'oracle'/('rank%d'%rank)/(name+'.npy')) for rank in range(2)]
    scale=max(float(np.max(np.abs(x))) for x in reference)
    for rank,ref in enumerate(reference):
        cur=np.load(root/'shim'/('rank%d'%rank)/(name+'.npy'))
        assert ref.shape==cur.shape and ref.dtype==cur.dtype
        delta=np.abs(ref.astype(np.float64)-cur.astype(np.float64))
        tol=5e-5+5e-5*scale
        idx=np.unravel_index(int(np.argmax(delta)),delta.shape) if delta.ndim else ()
        rows.append(dict(name=name,rank=rank,scale=scale,worst=float(np.max(delta)),
            tolerance=tol,bad=int(np.count_nonzero(delta>tol)),index=[int(v) for v in idx],
            oracle_at_worst=float(ref[idx]),shim_at_worst=float(cur[idx])))
name='v_proj_weight_grad'
reduction={}
for runtime in ('oracle','shim'):
    x=[np.load(root/runtime/('rank%d'%r)/(name+'.npy')) for r in (0,1)]
    full=np.load(train/runtime/'rank0/step0/grad/model.layers.0.self_attn.v_proj.weight.npy')
    assert x[0].shape==x[1].shape==full.shape
    mean=(x[0].astype(np.float64)+x[1].astype(np.float64))/2
    difference=np.abs(mean-full)
    idx=(521,1000)
    reduction[runtime]=dict(raw_rank0=float(x[0][idx]),raw_rank1=float(x[1][idx]),
      raw_mean=float(mean[idx]),saved_full=float(full[idx]),
      worst_assembly_abs=float(difference.max()),example_assembly_abs=float(difference[idx]))
result=dict(rows=rows,reduction=reduction)
Path(a.out).write_text(json.dumps(result,indent=2))
print(json.dumps(dict(failures=[r for r in rows if r['bad']],reduction=reduction),indent=2))
