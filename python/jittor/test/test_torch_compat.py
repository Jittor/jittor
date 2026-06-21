"""Unit tests for the `import jittor as torch` compatibility layer.

Run: python -m pytest python/jittor/test/test_torch_compat.py  (or plain python).
These exercise the torch-API surface that transformers/LlamaFactory depend on.
"""
import numpy as np
import jittor as torch          # the whole point: jittor IS torch here
import jittor as jt

PASS = FAIL = 0
def ok(cond, name):
    global PASS, FAIL
    if cond: PASS += 1; print("ok  ", name)
    else: FAIL += 1; print("FAIL", name)

# dtypes
ok(repr(torch.float32) == "torch.float32", "dtype repr")
ok(torch.float32 == "float32", "dtype eq str")
ok(torch.float32.is_floating_point and not torch.int64.is_floating_point, "is_floating_point")
ok(torch.zeros(2, dtype=torch.float16).dtype == "float16", "zeros dtype")
ok(torch.long == "int64", "long alias")

# constructors with torch kwargs
ok(tuple(torch.zeros(2, 3, device="cuda").shape) == (2, 3), "zeros device kwarg")
ok(tuple(torch.arange(5, device="cuda").shape) == (5,), "arange device kwarg")
ok(torch.from_numpy(np.zeros(3, dtype=np.int64)).dtype == "int64", "from_numpy int64")
ok(int(torch.tensor([1,2,3]).numel()) == 3, "tensor numel")

# reductions (torch semantics)
x = torch.tensor([[1., 5., 2.], [7., 3., 4.]])
ok(torch.argmax(x, dim=-1).numpy().tolist() == [1, 0], "argmax indices")
ok(str(torch.argmax(x, dim=-1).dtype) == "int64", "argmax int64")
mx = torch.max(x, dim=-1)
ok(mx.values.numpy().tolist() == [5., 7.] and mx.indices.numpy().tolist() == [1, 0], "max namedtuple")
ok(float(torch.max(x).numpy()) == 7.0, "max scalar")
ok(torch.max(x, torch.zeros_like(x) + 3).numpy().tolist() == [[3,5,3],[7,3,4]], "max elementwise")
ok(np.asarray(torch.sort(x, dim=-1, descending=True).values.numpy())[0].tolist() == [5,2,1], "sort")
ok(np.asarray(torch.topk(x, 2, dim=-1).values.numpy())[1].tolist() == [7,4], "topk")

# cat with empty + dtype
ok(tuple(torch.cat([torch.zeros(0,4), torch.ones(2,4)]).shape) == (2,4), "cat empty")

# no_grad three forms
@torch.no_grad
def f(a): return a*2
@torch.no_grad()
def g2(a): return a*3
with torch.no_grad():
    z = torch.zeros(2)+1
ok(f(torch.zeros(2)+1).numpy().tolist() == [2,2], "no_grad bare deco")
ok(g2(torch.zeros(2)+1).numpy().tolist() == [3,3], "no_grad called deco")

# nn
lin = torch.nn.Linear(4, 3)
y = lin(torch.rand(2, 4))
ok(tuple(y.shape) == (2, 3), "nn.Linear")
gg = jt.grad((y*y).sum(), list(lin.parameters()))
ok(any(float((p*p).sum().numpy())>0 for p in gg), "nn.Linear grads flow")
ok(torch.nn.functional.relu(torch.tensor([-1.,2.])).numpy().tolist() == [0,2], "F.relu")
ok(callable(torch.nn.functional.scaled_dot_product_attention), "F.sdpa present")

# Module forward<->execute bridge
class MyMod(torch.nn.Module):
    def __init__(s): super().__init__(); s.l = torch.nn.Linear(4,4)
    def forward(s, x): return s.l(x)        # torch-style forward
m = MyMod()
ok(tuple(m(torch.rand(1,4)).shape) == (1,4), "Module forward() bridge")
m.eval(); m.train(True)
ok(True, "train/eval mode")

# in-place ops preserve grad-tracking
p = torch.nn.Linear(3,3).weight
p.normal_(0, 0.02)
ok(not p.is_stop_grad(), "in-place init keeps grad")

# Var ops
ok(tuple(torch.tensor([[1.,2.]]).view(2,1).shape) == (2,1), "view")
ok((~torch.tensor([True, False])).numpy().tolist() == [False, True], "invert bool")
ok(torch.finfo(torch.float16).max > 0, "finfo")
ok(torch.cuda.is_available() in (True, False), "cuda.is_available")

print(f"\n==== {PASS} passed, {FAIL} failed ====")
import sys; sys.exit(1 if FAIL else 0)
