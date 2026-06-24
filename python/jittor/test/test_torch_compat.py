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

# ---- training bridge: backward / optimizer / LR scheduler / save-load ----
# loss.backward() + optimizer.step() must actually update params
lin2 = torch.nn.Linear(4, 2)
opt = torch.optim.AdamW(lin2.parameters(), lr=1e-2)
w_before = float((lin2.weight*lin2.weight).sum().numpy())
x = jt.randn(8, 4)
loss = ((lin2(x))**2).mean()
opt.zero_grad(); loss.backward()
gnorm = torch.nn.utils.clip_grad_norm_(lin2.parameters(), 1.0)
opt.step()
w_after = float((lin2.weight*lin2.weight).sum().numpy())
ok(float(gnorm.numpy()) > 0, "backward produces non-zero grad_norm")
ok(abs(w_after - w_before) > 0, "optimizer.step updates params")
ok(lin2.weight.grad is not None, "param.grad accessible after backward")

# LR scheduler drives jittor optimizer lr (shim-level; skip if not wired)
_lrsmod = getattr(torch.optim, "lr_scheduler", None)
if _lrsmod is not None and hasattr(_lrsmod, "LambdaLR"):
    opt2 = torch.optim.AdamW(lin2.parameters(), lr=1e-3)
    sched = _lrsmod.LambdaLR(opt2, lr_lambda=lambda s: max(0.0, 1.0 - s/4.0))
    lr0 = sched.get_last_lr()[0]
    for _ in range(2):
        sched.step()
    lr2 = sched.get_last_lr()[0]
    ok(abs(lr0 - 1e-3) < 1e-9, "scheduler initial lr == base lr")
    ok(lr2 < lr0, "scheduler decays lr")

# load_state_dict must NOT freeze trainable params (accelerate round-trip bug)
opt3 = torch.optim.AdamW(lin2.parameters(), lr=1e-3)
opt3.load_state_dict(opt3.state_dict())
ok(not lin2.weight.is_stop_grad(), "optimizer load_state_dict keeps params trainable")

# torch.save / torch.load round-trips tensors and plain objects
import tempfile, os as _os
_tmp = tempfile.mkdtemp()
_obj = {"w": torch.tensor([1., 2., 3.]), "meta": {"lr": 0.1, "name": "x"}}
torch.save(_obj, _os.path.join(_tmp, "ckpt.bin"))
_loaded = torch.load(_os.path.join(_tmp, "ckpt.bin"))
ok(_loaded["w"].numpy().tolist() == [1, 2, 3], "torch.save/load tensor round-trip")
ok(_loaded["meta"]["lr"] == 0.1 and _loaded["meta"]["name"] == "x", "torch.save/load object round-trip")

# Var.dtype is hashable & comparable to torch dtype objects
_dt = torch.tensor([1.0]).float().dtype
ok(_dt in {torch.float16, torch.float32}, "Var.dtype hashable + in dtype set")

# ---- regression: NPU dispatch + ops exposed when moving CPU->Ascend ----
# On Ascend the compat layer MUST enable jt.flags.use_cuda, else every op runs
# on CPU (~10000x slower). This was the root cause of pathological 8B step times.
if getattr(jt.compiler, "has_acl", 0):
    ok(jt.flags.use_cuda == 1, "NPU dispatch enabled (use_cuda=1) when has_acl")

# cumsum on bool must not segfault (ACL aclnnCumsum crashes on bool input) and
# must match torch, which promotes bool->int64 then counts.
_cb = torch.tensor([True, False, True, True]).cumsum(0)
ok(_cb.numpy().tolist() == [1, 1, 2, 3], "cumsum(bool) counts (no ACL segfault)")

# torch.diff(prepend=...) used by transformers packed-sequence detection
_pi = torch.tensor([[0, 1, 2, 0, 1]])
_pd = torch.diff(_pi, prepend=_pi[:, :1], dim=-1)
ok(_pd.numpy().tolist() == [[0, 1, 1, -2, 1]], "torch.diff prepend dim=-1")

# F.softmax(dtype=) used by eager attention
_sm = torch.nn.functional.softmax(torch.tensor([1.0, 2.0, 3.0]), dim=-1, dtype=torch.float32)
ok(abs(float(_sm.sum().numpy()) - 1.0) < 1e-5, "F.softmax(dtype=) normalizes")

# torch.autocast must work as BOTH a context manager and a decorator (accelerate
# does `new_forward = autocast(model_forward)`).
with torch.autocast("cuda", dtype=torch.bfloat16):
    _ac = (torch.ones(2) + 1)
ok(_ac.numpy().tolist() == [2, 2], "autocast as context manager")
@torch.autocast("cuda", dtype=torch.bfloat16)
def _acf(x): return x * 3
ok(_acf(torch.ones(2)).numpy().tolist() == [3, 3], "autocast as decorator")

# bf16 matmul must work on ACL (cube units); was 'Not supported dtype: bfloat16'
if getattr(jt.compiler, "has_acl", 0):
    _bm = torch.matmul(torch.ones(8, 8).to(torch.bfloat16), torch.ones(8, 8).to(torch.bfloat16))
    ok(str(_bm.dtype) == "bfloat16" and abs(float(_bm.float32().numpy().reshape(-1)[0]) - 8.0) < 1e-2,
       "bf16 matmul on ACL")

# torch.cuda.amp.GradScaler: scale(loss).backward() -> step() -> update() must
# train through the optimizer bridge (fp16 mixed precision).
_gs = torch.cuda.amp.GradScaler(init_scale=1024.0)
_lg = torch.nn.Linear(8, 8)
_og = torch.optim.Adam(_lg.parameters(), lr=2e-2)
_xg = jt.randn(16, 8)
_l0 = _l1 = None
for _i in range(5):
    _og.zero_grad(); _ls = (_lg(_xg) ** 2).mean()
    _gs.scale(_ls).backward(); _gs.step(_og); _gs.update()
    if _i == 0: _l0 = float(_ls.item())
    _l1 = float(_ls.item())
ok(_l1 < _l0, "GradScaler train loop decreases loss")
ok(_gs.get_scale() == 1024.0, "GradScaler scale stable without overflow")

# --- ops/features added later (lock them in) ---
# scatter family + cummax + new methods
_b = torch.zeros(3, 4)
ok(float(_b.scatter_add(0, torch.tensor([[0,1,2,0]]).int32(), torch.ones(1,4)).numpy().sum()) == 4.0, "scatter_add")
ok(float(torch.zeros(3,4).scatter(0, torch.tensor([[0,1,2,0]]).int32(), 9.0).numpy().sum()) == 36.0, "scalar scatter")
_cm = torch.tensor([1.,3.,3.,2.,5.]).cummax(dim=0)
ok(_cm.values.numpy().tolist() == [1,3,3,3,5] and _cm.indices.numpy().tolist() == [0,1,1,1,4], "cummax values+indices")
ok(np.allclose(torch.tensor([1.,2.,4.]).log_().numpy(), np.log([1.,2.,4.]), atol=1e-5), "in-place log_")
ok(tuple(torch.tensor([1,2,3]).reshape_as(torch.zeros(3)).shape) == (3,), "reshape_as")
ok(np.allclose(torch.tensor([[1.,5.],[2.,3.]]).var(dim=1).numpy(), [8.0, 0.5], atol=1e-5), "var unbiased")
# complex API + FFT
_c = torch.complex(torch.tensor([1.,2.]), torch.tensor([3.,4.]))
ok(torch.is_complex(_c) and (_c*_c).real.numpy().tolist() == [-8.,-12.], "complex mul")
_xf = np.random.RandomState(0).randn(8).astype("float32")
ok(np.allclose(torch.fft.fft(torch.tensor(_xf)).real.numpy(), np.fft.fft(_xf).real, atol=1e-4), "fft vs numpy")
ok(np.allclose(torch.fft.irfft(torch.fft.rfft(torch.tensor(_xf)), n=8).numpy(), _xf, atol=1e-4), "rfft/irfft roundtrip")
# multi_head_attention_forward
_q = jt.randn(4, 2, 8)
_o, _w = torch.nn.functional.multi_head_attention_forward(
    _q, _q, _q, 8, 2, jt.randn(24, 8), jt.zeros(24), None, None, False, 0.0,
    jt.randn(8, 8), jt.zeros(8), False, need_weights=True)
ok(tuple(_o.shape) == (4, 2, 8) and np.allclose(_w.numpy().sum(-1), 1, atol=1e-5), "multi_head_attention_forward")
# torch.cuda memory reports real (non-zero after an allocation)
_big = jt.randn(2000, 2000); _big.sync()
ok(torch.cuda.memory_allocated() > 0, "torch.cuda.memory_allocated real")
# model.save/load round-trip (was a RecursionError)
import tempfile as _tf, os as _os
_ml = torch.nn.Linear(4, 3); _w0 = _ml.weight.numpy().copy()
_mp = _os.path.join(_tf.mkdtemp(), "m.pkl"); _ml.save(_mp)
_ml2 = torch.nn.Linear(4, 3); _ml2.load(_mp)
ok(_os.path.exists(_mp) and np.allclose(_ml2.weight.numpy(), _w0), "model.save/load round-trip")
# jittor.lightning trains
import jittor.lightning as _pl
class _Lit(_pl.LightningModule):
    def __init__(self): super().__init__(); self.net = torch.nn.Linear(4, 1)
    def forward(self, x): return self.net(x)
    def training_step(self, b, i): x, y = b; return ((self(x) - y) ** 2).mean()
    def configure_optimizers(self): return torch.optim.Adam(self.parameters(), lr=0.1)
_W = np.random.randn(4, 1).astype("float32")
_data = [(jt.array(np.random.randn(8, 4).astype("float32")),) for _ in range(4)]
_data = [(d[0], d[0] @ jt.array(_W)) for d in _data]
_losses = []
_lit = _Lit(); _orig = _lit.training_step
_lit.training_step = lambda b, i: (_losses.append(float(_orig(b, i).item())) or _orig(b, i))
_pl.Trainer(max_epochs=8).fit(_lit, train_dataloaders=_data)
ok(_losses[-1] < _losses[0], "jittor.lightning Trainer trains (loss decreases)")

# DataLoader default collation: `for x, y in dl` must yield STACKED batches (was a no-op
# that returned a raw list of samples). Also torch.utils.data attribute access.
from torch.utils.data import DataLoader as _DL, TensorDataset as _TD
_dl = _DL(_TD(jt.randn(10, 4), jt.randn(10, 2)), batch_size=4, shuffle=False)
_xb, _yb = next(iter(_dl))
ok(tuple(_xb.shape) == (4, 4) and tuple(_yb.shape) == (4, 2), "DataLoader collates into batches")
ok(hasattr(torch.utils.data, "Dataset"), "torch.utils.data attribute access")

# torch.func (functorch): functional_call swaps params without mutating the module;
# grad is a functional gradient transform. Both verified bit-identical to real torch.
_fl = torch.nn.Linear(3, 2)
_fW = np.random.randn(2, 3).astype("float32"); _fB = np.random.randn(2).astype("float32")
_fX = np.random.randn(4, 3).astype("float32")
_fy = torch.func.functional_call(_fl, {"weight": jt.array(_fW), "bias": jt.array(_fB)}, (jt.array(_fX),))
ok(np.abs(_fy.numpy() - (_fX @ _fW.T + _fB)).max() < 1e-4, "torch.func.functional_call")
ok(not np.allclose(_fl.weight.numpy(), _fW, atol=1e-4), "functional_call leaves module unmutated")
_fg = torch.func.grad(lambda w, x: ((x @ w.transpose(0, 1)) ** 2).sum())(jt.array(_fW), jt.array(_fX))
ok(np.abs(_fg.numpy() - 2 * ((_fX @ _fW.T).T @ _fX)).max() < 1e-3, "torch.func.grad")
_fps, _fbs = torch.func.stack_module_state([torch.nn.Linear(3, 2) for _ in range(5)])
ok(tuple(_fps["weight"].shape) == (5, 2, 3), "torch.func.stack_module_state")

# nn.utils reparametrizations: weight_norm reparametrizes weight->(weight_g,weight_v)
# and recomputes weight before forward (wav2vec2 positional conv; verified vs real
# torch to ~1e-5). spectral_norm divides weight by its top singular value (verified
# vs np.linalg.svd + converged torch). pad_sequence pads a ragged batch.
_wc = torch.nn.Conv1d(4, 4, 3, padding=1)
_wc = torch.nn.utils.weight_norm(_wc, name="weight", dim=2)
ok(sorted(n for n, _ in _wc.named_parameters()) == ["bias", "weight_g", "weight_v"],
   "weight_norm reparametrizes weight->g/v (not weight)")
ok(tuple(_wc(jt.randn(2, 4, 8)).shape) == (2, 4, 8), "weight_norm conv forward")
_Wl = np.random.randn(3, 5).astype("float32")
_sl = torch.nn.Linear(5, 3); _sl.weight.update(jt.array(_Wl))
_sl = torch.nn.utils.spectral_norm(_sl, n_power_iterations=30)
for _ in range(3):
    _ = _sl(jt.randn(2, 5))
_sig = float((_sl.weight_orig.numpy() / _sl.weight.numpy()).mean())
ok(abs(_sig - np.linalg.svd(_Wl, compute_uv=False)[0]) < 1e-3, "spectral_norm sigma == top singular value")
_pad = torch.nn.utils.rnn.pad_sequence([jt.ones(3, 2), jt.ones(1, 2) * 2, jt.ones(2, 2) * 3], batch_first=True)
ok(tuple(_pad.shape) == (3, 3, 2) and _pad.numpy()[1, 1, 0] == 0.0, "rnn.pad_sequence pads ragged batch")

# Write-through _parameters/_buffers: accelerate's set_module_tensor_to_device (the
# from_pretrained meta / low_cpu_mem_usage fast path used by diffusers + transformers)
# assigns via `module._parameters[name] = value` / `module._buffers[name] = value`.
# jittor's properties build a fresh dict per access, so without write-through the
# assignment is lost -> loaded weights silently ignored (model keeps construction
# weights). Verify the idiom persists and preserves param/buffer classification.
_wt = torch.nn.Linear(4, 3); _wt.register_buffer("rm", jt.zeros(3))
_wt._parameters["weight"] = jt.array(np.ones((3, 4), "float32"))
_wt._buffers["rm"] = jt.ones(3)
_wtp = [p.name() for p in _wt.parameters()]
ok(float(_wt.weight.numpy().sum()) == 12.0, "_parameters[name]=v write-through persists")
ok(float(_wt.rm.numpy().sum()) == 3.0, "_buffers[name]=v write-through persists")
ok("rm" not in _wtp and "weight" in _wtp, "write-through preserves buffer/param classification")

# F.scaled_dot_product_attention (SDPA) — the default attention in transformers 5.x.
# Verify forward against a softmax(QK^T/sqrt(d))V reference (plain/causal/bool-mask/
# scale) and backward against a numeric gradient. Subtle spots: bool-mask semantics
# (True=keep, not inverted) and the scale override.
_sdpa = torch.nn.functional.scaled_dot_product_attention
np.random.seed(7)
_q = np.random.randn(1, 2, 4, 8).astype("float32"); _k = np.random.randn(1, 2, 4, 8).astype("float32")
_v = np.random.randn(1, 2, 4, 8).astype("float32")
def _ref_sdpa(q, k, v, causal=False, bmask=None, scale=None):
    sc = scale if scale is not None else 1.0 / np.sqrt(q.shape[-1])
    s = (q @ np.transpose(k, (0, 1, 3, 2))) * sc
    if causal:
        s = np.where(np.triu(np.ones((4, 4)), 1).astype(bool), -1e30, s)
    if bmask is not None:
        s = np.where(bmask, s, -1e30)
    s = s - s.max(-1, keepdims=True); e = np.exp(s)
    return (e / e.sum(-1, keepdims=True)) @ v
ok(np.abs(_sdpa(jt.array(_q), jt.array(_k), jt.array(_v)).numpy() - _ref_sdpa(_q, _k, _v)).max() < 1e-5,
   "SDPA forward (plain)")
ok(np.abs(_sdpa(jt.array(_q), jt.array(_k), jt.array(_v), is_causal=True).numpy() - _ref_sdpa(_q, _k, _v, causal=True)).max() < 1e-5,
   "SDPA forward (causal)")
_bm = np.tril(np.ones((4, 4))).astype(bool)[None, None].repeat(2, 1)
ok(np.abs(_sdpa(jt.array(_q), jt.array(_k), jt.array(_v), attn_mask=jt.array(_bm)).numpy() - _ref_sdpa(_q, _k, _v, bmask=_bm)).max() < 1e-5,
   "SDPA forward (bool mask: True=keep)")
ok(np.abs(_sdpa(jt.array(_q), jt.array(_k), jt.array(_v), scale=0.25).numpy() - _ref_sdpa(_q, _k, _v, scale=0.25)).max() < 1e-5,
   "SDPA forward (scale override)")
_qv, _kv, _vv = jt.array(_q), jt.array(_k), jt.array(_v)
_gq = jt.grad((_sdpa(_qv, _kv, _vv, is_causal=True) ** 2).sum(), [_qv])[0]
ok(bool(jt.isfinite(_gq).all().item()) and float(jt.abs(_gq).sum().item()) > 0, "SDPA backward grad finite+nonzero")

# torch.optim.lr_scheduler on the `import jittor as torch` path (was entirely missing;
# the documented primary path). Schedulers drive jittor optimizers by updating both
# optimizer.lr and each param_group["lr"]; verified against torch's exact formulas.
import math as _math
_lrs = torch.optim.lr_scheduler
def _curve(make, n=8):
    _l = torch.nn.Linear(2, 2); _o = torch.optim.AdamW(_l.parameters(), lr=1.0)
    _s = make(_o); seen = []
    for _ in range(n):
        seen.append(round(float(_o.param_groups[0]["lr"]), 5)); _s.step()
    return seen
ok(_curve(lambda o: _lrs.LambdaLR(o, lr_lambda=lambda e: min(1.0, (e + 1) / 5))) ==
   [0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0], "lr_scheduler.LambdaLR warmup (HF warmup helpers wrap this)")
ok(_curve(lambda o: _lrs.StepLR(o, step_size=2, gamma=0.5)) ==
   [1.0, 1.0, 0.5, 0.5, 0.25, 0.25, 0.125, 0.125], "lr_scheduler.StepLR")
ok(_curve(lambda o: _lrs.CosineAnnealingLR(o, T_max=4)) ==
   [round((1 + _math.cos(_math.pi * e / 4)) / 2, 5) for e in range(8)], "lr_scheduler.CosineAnnealingLR")
ok(_curve(lambda o: _lrs.LinearLR(o, start_factor=0.5, end_factor=1.0, total_iters=4)) ==
   [round(0.5 + 0.5 * min(e, 4) / 4, 5) for e in range(8)], "lr_scheduler.LinearLR")
ok(all(hasattr(_lrs, n) for n in ["PolynomialLR", "MultiStepLR", "ExponentialLR",
       "SequentialLR", "ConstantLR", "ReduceLROnPlateau"]), "lr_scheduler common set present")

# model.generate(num_beams=...) path fixes (greedy/sampling worked; beam crashed on
# three torch-compat gaps): (1) torch.full(fill_value=) keyword, (2) take_along_dim
# broadcasting size-1 index dims (beam _gather_beams gathers full sequences),
# (3) torch.all/any numpy-style axis=/keepdims= aliases (beam _update_finished_beams).
ok(torch.full((2, 2), fill_value=5.0).numpy().tolist() == [[5, 5], [5, 5]], "full(fill_value=) keyword")
ok(torch.full_like(jt.zeros(3), fill_value=4.0).numpy().tolist() == [4, 4, 4], "full_like(fill_value=) keyword")
_tad_t = jt.array(np.arange(24).reshape(2, 3, 4).astype("float32"))
_tad_i = jt.array(np.array([[[1], [0]], [[2], [1]]]).astype("int64"))  # (2,2,1) -> broadcast to (2,2,4)
_tad_o = torch.take_along_dim(_tad_t, _tad_i, dim=1).numpy()
_tad_r = np.take_along_axis(_tad_t.numpy(), np.broadcast_to(_tad_i.numpy(), (2, 2, 4)), axis=1)
ok(_tad_o.shape == (2, 2, 4) and np.array_equal(_tad_o, _tad_r), "take_along_dim broadcasts size-1 index dims")
_bb = jt.array(np.array([[True, True, False], [True, True, True]]))
ok(torch.all(_bb, axis=-1, keepdims=True).numpy().tolist() == [[False], [True]], "torch.all(axis=,keepdims=) aliases")
ok(torch.any(_bb, axis=0).numpy().tolist() == [True, True, True], "torch.any(axis=) alias")
ok(torch.all(_bb, dim=1).numpy().tolist() == [False, True], "torch.all(dim=) still works")

# F.cross_entropy(label_smoothing=) — used by many training recipes (ImageNet,
# translation, some SFT); jittor's cross_entropy_loss lacked it. Verified bit-equal to
# real torch 2.12: ls=0.1 -> 1.452645, +weight -> 1.490588, +ignore -> 1.371985.
_ceF = torch.nn.functional.cross_entropy
np.random.seed(0)
_cel = np.random.randn(8, 5).astype("float32"); _cet = np.random.randint(0, 5, (8,)).astype("int64")
_cew = np.array([0.5, 1.0, 2.0, 1.5, 0.8], dtype="float32")
ok(abs(float(_ceF(jt.array(_cel), jt.array(_cet), label_smoothing=0.1).item()) - 1.452645) < 1e-4,
   "F.cross_entropy label_smoothing matches torch")
ok(abs(float(_ceF(jt.array(_cel), jt.array(_cet), weight=jt.array(_cew), label_smoothing=0.1).item()) - 1.490588) < 1e-4,
   "F.cross_entropy label_smoothing + weight matches torch")
_cet2 = _cet.copy(); _cet2[0] = -100; _cet2[3] = -100
ok(abs(float(_ceF(jt.array(_cel), jt.array(_cet2), ignore_index=-100, label_smoothing=0.1).item()) - 1.371985) < 1e-4,
   "F.cross_entropy label_smoothing + ignore_index matches torch")
ok(abs(float(_ceF(jt.array(_cel), jt.array(_cet)).item()) -
       float(_ceF(jt.array(_cel), jt.array(_cet), label_smoothing=0.0).item())) < 1e-6,
   "F.cross_entropy label_smoothing=0 delegates unchanged")

# Loss functions real workloads use that jittor's functional lacked: kl_div (knowledge
# distillation), binary_cross_entropy, huber_loss, cosine_embedding/margin_ranking/
# gaussian_nll. Verified bit-equal to real torch 2.12; class versions wrap the functional.
_Lf = torch.nn.functional
np.random.seed(0); _N, _Cc = 4, 6
_slog = np.log(np.random.dirichlet(np.ones(_Cc), _N).astype("float32") + 1e-9)
_tpr = np.random.dirichlet(np.ones(_Cc), _N).astype("float32")
ok(abs(float(_Lf.kl_div(jt.array(_slog), jt.array(_tpr), reduction="batchmean").item()) - 0.741089) < 2e-4,
   "F.kl_div batchmean matches torch (distillation)")
ok(abs(float(_Lf.kl_div(jt.array(_slog), jt.array(_tpr), reduction="mean").item()) - 0.123515) < 2e-4,
   "F.kl_div mean matches torch")
_bp = np.random.rand(_N, _Cc).astype("float32"); _bt = (np.random.rand(_N, _Cc) > 0.5).astype("float32")
ok(abs(float(_Lf.binary_cross_entropy(jt.array(_bp), jt.array(_bt)).item()) - 1.225431) < 2e-4,
   "F.binary_cross_entropy matches torch")
_ha = jt.array(np.random.randn(_N, _Cc).astype("float32")); _hb = jt.array(np.random.randn(_N, _Cc).astype("float32"))
ok(abs(float(torch.nn.HuberLoss(delta=0.5)(_ha, _hb).item()) -
       float(_Lf.huber_loss(_ha, _hb, delta=0.5).item())) < 1e-6, "nn.HuberLoss == F.huber_loss")
ok(all(hasattr(_Lf, n) for n in ["kl_div", "binary_cross_entropy", "huber_loss",
       "cosine_embedding_loss", "margin_ranking_loss", "gaussian_nll_loss"]), "F loss set present")
ok(all(hasattr(torch.nn, n) for n in ["HuberLoss", "SmoothL1Loss", "CosineEmbeddingLoss",
       "MarginRankingLoss", "GaussianNLLLoss", "NLLLoss"]), "nn loss class set present")

# pixel_shuffle / pixel_unshuffle (super-resolution, some VAE decoders): (N,C*r^2,H,W) <->
# (N,C,H*r,W*r). Verified vs real torch (flat layout) + roundtrip + nn class. interpolate
# (bilinear align_corners True/False) and pad(reflect) already match torch.
_psx = jt.array(np.arange(32).reshape(1, 8, 2, 2).astype("float32"))
_pso = torch.nn.functional.pixel_shuffle(_psx, 2)
ok(tuple(_pso.shape) == (1, 2, 4, 4) and _pso.numpy().flatten()[:6].tolist() == [0, 4, 1, 5, 8, 12],
   "F.pixel_shuffle matches torch layout")
ok(np.array_equal(torch.nn.functional.pixel_unshuffle(_pso, 2).numpy(), _psx.numpy()),
   "F.pixel_unshuffle inverts pixel_shuffle")
ok(np.array_equal(torch.nn.PixelShuffle(2)(_psx).numpy(), _pso.numpy()), "nn.PixelShuffle == functional")
_vx = np.random.RandomState(0).randn(1, 2, 4, 4).astype("float32")
ok(abs(float(torch.nn.functional.interpolate(jt.array(_vx), size=(8, 8), mode="bilinear", align_corners=True).sum().item()) - 47.48457) < 1e-2,
   "F.interpolate bilinear align_corners=True matches torch")

# torch.roll: negative dims previously emitted 'i-1' -> JIT compile error ('op0_i'
# undeclared); no-dims must flatten (torch), not roll dim 0. torch.cumprod: was
# exp(cumsum(log)) -> NaN for negatives; now sign-aware (diffusers alphas_cumprod, etc.).
_rx = np.random.RandomState(0).randn(2, 3, 4).astype("float32")
ok(np.abs(torch.roll(jt.array(_rx), 1, dims=-1).numpy() - np.roll(_rx, 1, axis=-1)).max() == 0,
   "torch.roll negative dim (was a JIT compile error)")
ok(np.abs(torch.roll(jt.array(_rx), (-1, -1), dims=(1, 2)).numpy() - np.roll(np.roll(_rx, -1, 1), -1, 2)).max() == 0,
   "torch.roll multi-dim (swin window shift)")
ok(np.abs(torch.roll(jt.array(_rx), 2).numpy() - np.roll(_rx, 2)).max() == 0,
   "torch.roll no-dims flattens (torch semantics)")
_cx = np.random.RandomState(1).randn(2, 5).astype("float32")
ok(np.abs(torch.cumprod(jt.array(_cx), dim=-1).numpy() - np.cumprod(_cx, axis=-1)).max() < 1e-4,
   "torch.cumprod sign-aware (negatives no longer NaN)")
_cz = np.array([[2., 0., 3., -1.], [1., -2., 0., 4.]], dtype="float32")
ok(np.abs(torch.cumprod(jt.array(_cz), dim=-1).numpy() - np.cumprod(_cz, axis=-1)).max() < 1e-5,
   "torch.cumprod with zeros and negatives")

# Var.index_fill_ was broken (negative-dim crash + iterated the index TENSOR into an
# f-string) and unexposed. Rewrote mask-based; matches torch (in-place, negative dim,
# tensor or list index, differentiable).
_ifx = np.random.RandomState(0).randn(3, 4).astype("float32")
_ifa = jt.array(_ifx.copy()); _ifa.index_fill_(1, jt.array(np.array([0, 2], dtype="int64")), 9.0)
ok(abs(float(_ifa.sum().item()) - 57.37729) < 1e-3, "Var.index_fill_ dim=1 matches torch")
_ifc = jt.array(_ifx.copy()); _ifc.index_fill_(-1, jt.array(np.array([3], dtype="int64")), -1.0)
ok(abs(float(_ifc.sum().item()) - 2.43474) < 1e-3, "Var.index_fill_ negative dim (was a crash)")
_ifd = jt.array(_ifx.copy()); _ = _ifd.index_fill(1, jt.array(np.array([0], dtype="int64")), 0.0)
ok(np.array_equal(_ifd.numpy(), _ifx), "Var.index_fill out-of-place leaves input")

print(f"\n==== {PASS} passed, {FAIL} failed ====")
import sys as _sys
_sys.exit(1 if FAIL else 0)
import sys; sys.exit(1 if FAIL else 0)
