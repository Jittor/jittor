"""Qwen3 L0 parameter/buffer manifest on the declared device."""
import hashlib,json,os
import importlib.metadata
from pathlib import Path
runtime=os.environ['DS_RUNTIME'];rank=int(os.environ['RANK'])
zero_stage=int(os.environ.get('DS_ZERO_STAGE','1'));assert zero_stage in (1,2,3)
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
 before=jt.core.backend_fallback_count()
from transformers import AutoConfig,AutoModelForCausalLM
path=Path(os.environ['DS_MODEL_PATH'])
out=Path(os.environ['DS_L0_OUT'])/runtime/('rank%d'%rank)
out.mkdir(parents=True,exist_ok=True)
assert not (out/'manifest.json').exists()
config=AutoConfig.from_pretrained(str(path),local_files_only=True)
model=AutoModelForCausalLM.from_pretrained(str(path),local_files_only=True,
 torch_dtype=torch.float32,attn_implementation='eager').to('npu:0')
model.train()
def details(p):
 assert p.device.type=='npu'
 return dict(shape=list(p.shape),dtype=str(p.dtype),device=p.device.type)
parameters={name:details(p) for name,p in model.named_parameters()}
buffers={name:details(b) for name,b in model.named_buffers()}
assert len(parameters)==310 and sum(int(p.numel()) for p in model.parameters())==596049920
assert all(v['dtype']=='torch.float32' for v in parameters.values())
placement_checked=0
if runtime=='shim':
 for _,p in list(model.named_parameters())+list(model.named_buffers()):
  p.sync()
  assert p.location()=='device'
  assert p.placement_backend in (-1,2)
  assert p.device_id>=0
  placement_checked+=1
else:
 torch.npu.synchronize()
 placement_checked=len(parameters)+len(buffers)
assert placement_checked==311
after=jt.core.backend_fallback_count() if runtime=='shim' else None
if runtime=='shim':assert after==before
report=dict(runtime=runtime,rank=rank,requested_zero_stage=zero_stage,model_class=type(model).__name__,
 versions={name:importlib.metadata.version(name) for name in ('deepspeed','transformers','numpy','safetensors')},
 torch_is_shim=hasattr(torch,'_torch_compat_install_context'),
 torch_has_c_extension=hasattr(torch,'_C'),
 torch_file=str(torch.__file__) if runtime=='oracle' else str(jt.__file__),
 torch_version=str(getattr(torch,'__version__','unavailable')),
 torch_npu_version=str(getattr(torch_npu,'__version__','unavailable')) if runtime=='oracle' else None,
 fallback_before=before if runtime=='shim' else None, fallback_after=after,
 fallback_delta=after-before if runtime=='shim' else None,
 placement_checked=placement_checked,placement_synchronized=True,
 config={key:getattr(config,key) for key in ('model_type','hidden_size','num_hidden_layers',
 'num_attention_heads','num_key_value_heads','vocab_size','rms_norm_eps')},
 parameter_count=len(parameters),parameters=parameters,
 buffer_count=len(buffers),buffers=buffers,device='npu',dtype='float32',
 probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(out/'manifest.json').write_text(json.dumps(report,indent=2))
print(json.dumps(dict(runtime=runtime,rank=rank,event='manifest',buffers=len(buffers))),flush=True)
