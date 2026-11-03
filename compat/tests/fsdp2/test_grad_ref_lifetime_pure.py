"""Torch oracle + CPU control-flow test for FSDP gradient method lifetime.

The tested resolver class is extracted unchanged from the actual source; no
claim is made about Jittor/NCCL runtime behavior in these CPU-only tests.
"""
import ast,gc,types,unittest,weakref
from pathlib import Path
import torch
class Var:
 def __init__(self,value):self.value=value

def load_method():
 root=Path(__file__).resolve().parents[3]
 source=root/'compat/fsdp2/shard.py'
 node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='_ShardTensorMethod')
 ns={'weakref':weakref,'jt':types.SimpleNamespace(Var=Var)}
 exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source),'exec'),ns)
 return ns['_ShardTensorMethod']

class Box:pass
class TestReferences(unittest.TestCase):
 def setUp(self):self.method=load_method()
 def test_set_to_none_torch_oracle_and_weak_lifetime(self):
  p=torch.nn.Parameter(torch.tensor([1.]));opt=torch.optim.SGD([p],lr=.1)
  (p*2).sum().backward();old=p.grad;before=old.clone()
  opt.zero_grad(set_to_none=True);assert p.grad is None
  (p*3).sum().backward();assert old is not p.grad and torch.equal(old,before)
  state=Box();entry=Box();entry.shard=Box();state.true_fsdp_params=[entry]
  state.torch_dim0_per_parameter=True;first=Var(2);second=Var(3)
  state.true_fsdp_last_grads=[first];get=self.method(lambda t:t,state,entry,'grad_shard',first)
  state.true_fsdp_last_grads=[second];self.assertIs(get(),first)
  first_ref=weakref.ref(first);del first;gc.collect();self.assertIsNone(first_ref())
  with self.assertRaises(ReferenceError):get()
 def test_zero_grad_false_and_accumulation_preserve_live_value(self):
  p=torch.nn.Parameter(torch.tensor([1.]));opt=torch.optim.SGD([p],lr=.1)
  (p*2).sum().backward();old=p.grad
  opt.zero_grad(set_to_none=False);self.assertIs(p.grad,old);self.assertEqual(float(old),0.)
  (p*3).sum().backward();self.assertIs(p.grad,old);self.assertEqual(float(old),3.)
  (p*4).sum().backward();self.assertIs(p.grad,old);self.assertEqual(float(old),7.)
  state=Box();entry=Box();entry.shard=Box();state.true_fsdp_params=[entry];state.torch_dim0_per_parameter=True
  local=Var(2);state.true_fsdp_last_grads=[local];get=self.method(lambda t:t,state,entry,'grad_shard',local)
  local.value=0;local.value+=3;local.value+=4;self.assertEqual(get().value,7)
 def test_legacy_branch_remains_dynamic(self):
  state=Box();entry=Box();entry.shard=Box();state.true_fsdp_params=[entry];state.torch_dim0_per_parameter=False
  first=Var(1);second=Var(2);state.true_fsdp_last_grads=[first]
  get=self.method(lambda t:t,state,entry,'grad_shard',first);state.true_fsdp_last_grads=[second]
  self.assertIs(get(),second)
if __name__=='__main__':unittest.main()
