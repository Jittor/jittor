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
out=Path(os.environ['DS_GEN_OUT'])/runtime/('rank%d'%rank)
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
 a=t.detach().cpu().numpy(); assert np.isfinite(a).all()
 np.save(out/(name+'.npy'),a,allow_pickle=False)
 records[name]=dict(shape=list(a.shape),dtype=str(a.dtype),device='npu')
 return a
save('input_ids',input_ids)
with torch.no_grad():
 prefill=model(input_ids=input_ids,attention_mask=attention_mask,use_cache=True)
cache=prefill.past_key_values
assert cache is not None
save('prefill_last_logits',prefill.logits[:,-1,:])
cache_type=type(cache).__name__
cache_length=int(cache.get_seq_length()) if hasattr(cache,'get_seq_length') else None
first=prefill.logits[:,-1,:].argmax(dim=-1,keepdim=True)
with torch.no_grad():
 decode=model(input_ids=first,attention_mask=decode_attention_mask,past_key_values=cache,use_cache=True)
 full=model(input_ids=torch.cat((input_ids,first),dim=-1),attention_mask=decode_attention_mask,use_cache=False)
a=save('decode_last_logits',decode.logits[:,-1,:])
b=save('recomputed_last_logits',full.logits[:,-1,:])
scale=max(float(np.max(np.abs(b))),float(np.max(np.abs(a))))
cache_worst=float(np.max(np.abs(a-b)))
cache_tolerance=5e-5+5e-5*scale
assert cache_worst<=cache_tolerance,(cache_worst,cache_tolerance)
with torch.no_grad():
 greedy_cached=model.generate(input_ids=input_ids,attention_mask=attention_mask,max_new_tokens=4,do_sample=False,num_beams=1,use_cache=True)
 greedy_uncached=model.generate(input_ids=input_ids,attention_mask=attention_mask,max_new_tokens=4,do_sample=False,num_beams=1,use_cache=False)
 beam=model.generate(input_ids=input_ids,attention_mask=attention_mask,max_new_tokens=4,do_sample=False,num_beams=2,use_cache=True)
greedy_a=save('greedy_cached',greedy_cached)
greedy_b=save('greedy_uncached',greedy_uncached)
beam_a=save('beam_cached',beam)
assert np.array_equal(greedy_a,greedy_b), 'greedy cached and uncached disagree'
assert int(greedy_a[0,input_ids.shape[1]])==int(first.detach().cpu().numpy()[0,0])
if runtime=='shim':assert jt.core.backend_fallback_count()==fallback_before
else:torch.npu.synchronize()
report=dict(status='passed',runtime=runtime,rank=rank,prompt=prompt,
 tokenizer_ids=tokenizer.encode(prompt,add_special_tokens=False),
 cache_type=cache_type,cache_length=cache_length,cache_worst_abs=cache_worst,
 cache_tolerance=cache_tolerance,greedy_cached=greedy_a.tolist(),
 greedy_uncached=greedy_b.tolist(),beam=beam_a.tolist(),
 fallback_delta=0 if runtime=='shim' else None,
 model_dir=str(model_dir),records=records,
 probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
print(json.dumps(dict(runtime=runtime,rank=rank,event='finished',report=report['status'])),flush=True)
