from pathlib import Path
import json,numpy as np
import argparse
parser=argparse.ArgumentParser(description='Compare same-weight and own-weight diagnostic replays; not a maturity gate.')
parser.add_argument('--replay-root',required=True)
parser.add_argument('--training-root',required=True)
args=parser.parse_args()
p=Path(args.replay_root); old=Path(args.training_root)
for runtime in ('oracle','shim'):
 for rank in range(2):
  assert (p/runtime/('rank%d'%rank)/'report.json').exists()
rows=[]
for case in ('common','own'):
 for field in ['input','input_grad','loss','logits']+['layer%02d-%s'%(i,k) for i in range(28) for k in ('value','grad')]:
  aa=[np.load(p/'oracle'/('rank%d'%k)/(case+'-'+field+'.npy')) for k in range(2)]
  scale=max(float(np.max(np.abs(a))) for a in aa)
  for rank,a in enumerate(aa):
   b=np.load(p/'shim'/('rank%d'%rank)/(case+'-'+field+'.npy'))
   diff=np.abs(a.astype('float64')-b.astype('float64'))
   rows.append(dict(case=case,field=field,rank=rank,worst=float(diff.max()),scale=scale,tolerance=5e-5+5e-5*scale,bad=int((diff>5e-5+5e-5*scale).sum())))
oldmatch=[]
for runtime in ('oracle','shim'):
 for rank in range(2):
  for field in ('input_grad','loss','logits'):
   a=np.load(old/runtime/('rank%d'%rank)/'step2'/(field+'.npy'))
   b=np.load(p/runtime/('rank%d'%rank)/('own-'+field+'.npy'))
   oldmatch.append(dict(runtime=runtime,rank=rank,field=field,worst=float(np.max(np.abs(a-b)))))
result=dict(rows=rows,old_match=oldmatch)
(p/'comparison.json').write_text(json.dumps(result,indent=2))
print(json.dumps(dict(summary=[r for r in rows if r['field'] in ('input','input_grad','loss','logits')],old_match=oldmatch),indent=2))
