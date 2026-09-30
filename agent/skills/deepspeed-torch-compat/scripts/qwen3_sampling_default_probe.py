"""Qwen3 generation/cache probe on real trained checkpoint, one NPU per rank."""
import hashlib,json,os
from pathlib import Path
import numpy as np
runtime=os.environ['DS_RUNTIME'];rank=int(os.environ['RANK'])
assert runtime in ('oracle','shim') and int(os.environ['WORLD_SIZE'])==2
if runtime=='shim':
 from jittor_adapters.deepspeed import activate
 activate(device='npu')
import torch
assert hasattr(torch,'_torch_compat_install_context')==(runtime=='shim')
if runtime=='oracle':
 assert hasattr(torch,'_C')
 import torch_npu
 torch.npu.set_device(0)
else:
 import jittor as jt
 fallback_before=jt.core.backend_fallback_count()
from transformers import AutoModelForCausalLM,AutoTokenizer
model_dir=Path(os.environ['DS_GEN_MODEL'])/runtime/('rank%d'%rank)/'saved-model'
initial=Path(os.environ['DS_MODEL_PATH'])
out=Path(os.environ['DS_SAMPLE_OUT'])/runtime/('rank%d'%rank)
out.mkdir(parents=True,exist_ok=True)
assert not (out/'report.json').exists()
model=AutoModelForCausalLM.from_pretrained(str(model_dir),local_files_only=True,
 torch_dtype=torch.float32,attn_implementation='eager').to('npu:0')
model.eval()
tokenizer=AutoTokenizer.from_pretrained(str(initial),local_files_only=True)
prompt='你好，计图'
input_ids=torch.tensor([tokenizer.encode(prompt,add_special_tokens=False)],device='npu:0',dtype=torch.int64)
assert input_ids.shape[0]==1
attention_mask=torch.ones(input_ids.shape,device='npu:0',dtype=torch.int64)
decode_attention_mask=torch.ones((1,input_ids.shape[1]+1),device='npu:0',dtype=torch.int64)
records={}
def save(name,t):
 assert t.device.type=='npu'
 a=t.detach().cpu().numpy()
 assert np.isfinite(a).all(),name
 np.save(out/(name+'.npy'),a,allow_pickle=False)
 records[name]=a.tolist()
 return a
save('input_ids',input_ids)
with torch.no_grad():
 prefill=model(input_ids=input_ids,attention_mask=attention_mask,use_cache=False)
 top2=torch.topk(prefill.logits[:,-1,:],2,dim=-1)
 probabilities=torch.softmax(top2.values/0.7,dim=-1)
save('first_step_top2_indices',top2.indices)
save('first_step_top2_values',top2.values)
save('first_step_top2_probabilities',probabilities)
with torch.no_grad():
 scores=prefill.logits[:,-1,:]
 sorted_values,sorted_indices=torch.sort(scores,dim=-1,descending=True)
 top20=torch.topk(scores,20,dim=-1)
save('full_sorted_values',sorted_values)
save('full_sorted_indices',sorted_indices)
save('top20_values',top20.values)
save('top20_indices',top20.indices)
with torch.no_grad():
 torch.manual_seed(20260925)
 sampled=model.generate(input_ids=input_ids,attention_mask=attention_mask,max_new_tokens=4,
                        do_sample=True,use_cache=True)
 torch.manual_seed(20260925)
 repeated=model.generate(input_ids=input_ids,attention_mask=attention_mask,max_new_tokens=4,
                         do_sample=True,use_cache=True)
a=save('sample_default',sampled)
b=save('sample_default_repeat',repeated)
assert np.array_equal(a,b),'same-runtime default sampling seed did not reproduce'
if runtime=='shim':assert jt.core.backend_fallback_count()==fallback_before
else:torch.npu.synchronize()
report=dict(status='passed',runtime=runtime,rank=rank,prompt=prompt,seed=20260925,
 dtype='float32',device='npu',fallback_delta=0 if runtime=='shim' else None,
 records={k:v for k,v in records.items() if not k.startswith('full_sorted')},
 probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
print(json.dumps(dict(runtime=runtime,rank=rank,event='sampling-default-finished',status='passed')),flush=True)
