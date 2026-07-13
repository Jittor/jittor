"""Regression coverage for the torch operator surface used by mmdetection 3.3.0.

`import torch` IS the jittor torch-shim here (see jittor/torch_shim), so this runs
exactly the way mmdet's own modules would. It checks two things:

  1. EXISTENCE  -- every torch.* / F.* / nn.* / Tensor-method symbol that
     mmdetection's source references resolves on the shim (the surface was
     extracted from a full grep of the mmdet/ tree).
  2. CORRECTNESS -- the high-risk / heavily-used ops (interpolate, grid_sample,
     the losses, conv, sparse_coo_tensor->dense, einsum, ...) produce results
     that match torch semantics, verified against numpy / mathematical identities.

NOT covered (and intentionally so): mmcv.ops native custom kernels (NMS, RoIAlign,
DeformConv, MultiScaleDeformableAttention, point_sample, CARAFE, ...). Those are
not torch operators -- they are C++/CUDA kernels mmcv compiles against libtorch
and must be ported to jittor separately.

Run:  PYTHONPATH=python python -m jittor.test.test_mmdet_ops
"""
import importlib
import numpy as np
import torch                       # the jittor torch-shim
import torch.nn as nn
import torch.nn.functional as F

PASS = FAIL = 0
def ok(cond, name):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print("FAIL", name)

def close(a, b, name, atol=1e-4, rtol=1e-4):
    # value-level check: compare flattened elements so jittor's (1,)-shaped scalar
    # reductions match numpy's 0-d scalars. Shape is asserted separately where it
    # matters (interpolate / pad / split / attention).
    a = a.numpy() if hasattr(a, "numpy") else np.asarray(a)
    b = b.numpy() if hasattr(b, "numpy") else np.asarray(b)
    af, bf = np.ravel(a), np.ravel(b)
    ok(af.size == bf.size and np.allclose(af, bf, atol=atol, rtol=rtol),
       name + (f" [size {af.size} vs {bf.size}]" if af.size != bf.size else " [value mismatch]"))

# --------------------------------------------------------------------------- #
# 1. EXISTENCE SWEEP: the operator surface extracted from the mmdet/ tree.     #
# --------------------------------------------------------------------------- #
# torch.* functions/classes mmdet calls (submodules + non-ops checked separately).
TORCH_SYMS = """
abs acos arange argsort as_tensor asin atan bernoulli bmm bool cat cdist ceil
chunk clamp clone cos cumsum device div einsum empty empty_like exp eye finfo
flatten flip float float16 float32 float64 floor from_numpy full full_like gather
Generator half histc index_select int int32 int64 isfinite isinf isnan kthvalue
linspace log log2 logical_xor logsumexp long masked_select matmul max mean meshgrid
min mm multinomial mul nan_to_num nan_to_num_ no_grad nonzero norm ones ones_like
permute pow rand rand_like randint randint_like randn randperm repeat_interleave
roll round scatter set_grad_enabled sigmoid sign size softmax sort sparse_coo_tensor
split split_with_sizes sqrt squeeze stack std sum tensor topk transpose uint8 unique
where zeros zeros_like is_grad_enabled _shape_as_tensor
LongTensor FloatTensor BoolTensor ByteTensor Size Tensor
""".split()
# torch tensor "type" constructors (LongTensor etc.) are callables on the shim.
for s in TORCH_SYMS:
    ok(hasattr(torch, s), f"torch.{s} exists")

# torch.* submodules / namespaces mmdet imports (resolved lazily via sys.modules).
TORCH_MODS = ["torch.nn", "torch.optim", "torch.utils", "torch.utils.checkpoint",
              "torch.autograd", "torch.distributed", "torch.multiprocessing",
              "torch.onnx", "torch.fx", "torch.hub", "torch.backends",
              "torch.backends.cudnn", "torch._utils", "torch.cuda", "torch.sparse"]
for m in TORCH_MODS:
    try:
        importlib.import_module(m); ok(True, f"import {m}")
    except Exception as e:
        ok(False, f"import {m}: {type(e).__name__}: {e}")

# torch.nn.functional.* used by mmdet.
FUNC_SYMS = """
interpolate relu relu_ softmax pad binary_cross_entropy_with_logits max_pool2d
binary_cross_entropy one_hot cross_entropy adaptive_avg_pool2d adaptive_max_pool2d
normalize batch_norm logsigmoid linear dropout avg_pool2d log_softmax kl_div
grid_sample unfold mse_loss upsample_bilinear _Reduction
""".split()
for s in FUNC_SYMS:
    ok(hasattr(F, s), f"F.{s} exists")

# torch.nn.* modules used by mmdet.
NN_SYMS = """
ModuleList Conv2d Module ReLU Linear Sequential Embedding Parameter
MultiheadAttention LayerNorm AdaptiveAvgPool2d AvgPool2d Identity Dropout Upsample
MaxPool2d ConvTranspose2d BatchNorm2d Hardsigmoid BatchNorm1d Unfold ModuleDict
GroupNorm Softmax ConvTranspose1d Conv1d Tanh SyncBatchNorm Sigmoid init functional
""".split()
for s in NN_SYMS:
    ok(hasattr(nn, s), f"nn.{s} exists")

# Tensor methods used by mmdet (probed on a real Var).
METHOD_SYMS = """
new_tensor new_zeros new_ones new_full clamp clamp_ clamp_min clamp_max sigmoid
softmax scatter scatter_ gather split unbind sort topk flip roll repeat_interleave
index_select permute expand expand_as reshape view masked_fill masked_fill_
masked_select sign bmm matmul mm t transpose contiguous flatten chunk narrow
squeeze unsqueeze type_as detach clone numpy item argmax argmin max min sum mean
cumsum any all nonzero unique floor ceil round exp log log2 sqrt pow abs neg
reciprocal eq ne gt ge lt le isnan isinf relu diff fliplr flipud fmod remainder
new_empty nan_to_num softplus
""".split()
_v = torch.zeros(2, 3)
for s in METHOD_SYMS:
    ok(hasattr(_v, s), f"Tensor.{s} exists")

# internal submodule import paths mmdet uses at module load.
def _imp(stmt_mod, names):
    try:
        mod = importlib.import_module(stmt_mod)
        for n in names:
            getattr(mod, n)
        ok(True, f"from {stmt_mod} import {','.join(names)}")
    except Exception as e:
        ok(False, f"from {stmt_mod} import {','.join(names)}: {type(e).__name__}: {e}")
_imp("torch.nn.modules.utils", ["_pair", "_single", "_triple", "_ntuple"])
_imp("torch.nn.modules.batchnorm", ["_BatchNorm", "BatchNorm2d", "SyncBatchNorm"])
_imp("torch.nn.modules.normalization", ["GroupNorm"])
_imp("torch.nn.modules.activation", ["SiLU", "ReLU"])
_imp("torch._utils", ["_flatten_dense_tensors", "_unflatten_dense_tensors", "_take_tensors"])

# --------------------------------------------------------------------------- #
# 2. CORRECTNESS: high-risk / heavily-used ops vs numpy / identities.          #
# --------------------------------------------------------------------------- #
rng = np.random.RandomState(0)

# interpolate: a constant image stays constant under every mode (and shape grows).
const = torch.ones(1, 2, 4, 4) * 3.0
for mode, ac in [("bilinear", False), ("bilinear", True), ("nearest", None), ("bicubic", False)]:
    kw = {} if ac is None else {"align_corners": ac}
    out = F.interpolate(const, size=(8, 8), mode=mode, **kw)
    ok(tuple(out.shape) == (1, 2, 8, 8), f"interpolate {mode} ac={ac} shape")
    close(out, np.full((1, 2, 8, 8), 3.0), f"interpolate {mode} ac={ac} constant", atol=1e-3)
# nearest upsample x2 replicates each pixel into a 2x2 block.
small = torch.from_numpy(rng.rand(1, 1, 2, 2).astype("float32"))
up = F.interpolate(small, scale_factor=2, mode="nearest")
sn = small.numpy()[0, 0]
exp = np.kron(sn, np.ones((2, 2), dtype="float32"))
close(up[0, 0], exp, "interpolate nearest x2 replication")

# grid_sample with the identity grid (align_corners=True) recovers the input.
img = torch.from_numpy(rng.rand(1, 3, 5, 7).astype("float32"))
ys = np.linspace(-1, 1, 5, dtype="float32")
xs = np.linspace(-1, 1, 7, dtype="float32")
gx, gy = np.meshgrid(xs, ys)                       # (5,7) each
grid = torch.from_numpy(np.stack([gx, gy], axis=-1)[None].astype("float32"))
samp = F.grid_sample(img, grid, mode="bilinear", align_corners=True)
close(samp, img, "grid_sample identity grid == input", atol=1e-3)

# einsum batched matmul == matmul.
a = torch.from_numpy(rng.rand(2, 3, 4).astype("float32"))
b = torch.from_numpy(rng.rand(2, 4, 5).astype("float32"))
close(torch.einsum("bij,bjk->bik", a, b), torch.matmul(a, b), "einsum bmm == matmul")

# torch.mm == numpy matmul (2-D).
m1 = torch.from_numpy(rng.rand(3, 4).astype("float32"))
m2 = torch.from_numpy(rng.rand(4, 6).astype("float32"))
close(torch.mm(m1, m2), m1.numpy() @ m2.numpy(), "torch.mm == numpy @")

# one_hot.
oh = F.one_hot(torch.from_numpy(np.array([0, 2, 1], dtype="int64")), num_classes=3)
close(oh, np.eye(3, dtype="int64")[[0, 2, 1]], "one_hot")

# softmax sums to 1 along dim.
sm = F.softmax(torch.from_numpy(rng.rand(4, 5).astype("float32")), dim=-1)
close(sm.sum(-1), np.ones(4, dtype="float32"), "softmax sums to 1")

# bce_with_logits == -[y*log s + (1-y)*log(1-s)] mean.
logit = rng.rand(6).astype("float32") * 4 - 2
tgt = (rng.rand(6) > 0.5).astype("float32")
s = 1 / (1 + np.exp(-logit))
exp_bce = -(tgt * np.log(s) + (1 - tgt) * np.log(1 - s)).mean()
got_bce = F.binary_cross_entropy_with_logits(torch.from_numpy(logit), torch.from_numpy(tgt))
close(got_bce, np.array(exp_bce, dtype="float32"), "bce_with_logits formula", atol=1e-3)

# cross_entropy == mean(-log_softmax[gather target]).
ce_in = rng.rand(4, 5).astype("float32")
ce_tg = np.array([0, 4, 2, 1], dtype="int64")
logp = ce_in - np.log(np.exp(ce_in).sum(1, keepdims=True))
exp_ce = -logp[np.arange(4), ce_tg].mean()
got_ce = F.cross_entropy(torch.from_numpy(ce_in), torch.from_numpy(ce_tg))
close(got_ce, np.array(exp_ce, dtype="float32"), "cross_entropy formula", atol=1e-3)

# F._Reduction.get_enum mapping used by mmdet loss utils.
ok(F._Reduction.get_enum("none") == 0 and F._Reduction.get_enum("mean") == 1 and
   F._Reduction.get_enum("sum") == 2, "F._Reduction.get_enum none/mean/sum")

# adaptive_max_pool2d global == channelwise max.
amp_in = torch.from_numpy(rng.rand(1, 3, 6, 6).astype("float32"))
close(F.adaptive_max_pool2d(amp_in, (1, 1)).reshape(3),
      amp_in.numpy().reshape(3, -1).max(1), "adaptive_max_pool2d global == max")

# adaptive_avg_pool2d of a constant == that constant.
close(F.adaptive_avg_pool2d(const, (1, 1)).reshape(2), np.full(2, 3.0, "float32"),
      "adaptive_avg_pool2d constant")

# split_with_sizes shapes.
parts = torch.split_with_sizes(torch.zeros(10, 4), [2, 3, 5], dim=0)
ok([tuple(p.shape) for p in parts] == [(2, 4), (3, 4), (5, 4)], "split_with_sizes shapes")

# masked_select == numpy boolean index.
ms_x = rng.rand(3, 4).astype("float32")
ms_m = ms_x > 0.5
close(torch.masked_select(torch.from_numpy(ms_x), torch.from_numpy(ms_m)),
      ms_x[ms_m], "masked_select == numpy[mask]")

# meshgrid indexing semantics.
mi, mj = torch.meshgrid(torch.arange(3), torch.arange(4), indexing="ij")
ok(tuple(mi.shape) == (3, 4) and int(mi[2, 0]) == 2 and int(mj[0, 3]) == 3, "meshgrid ij")
xi, xj = torch.meshgrid(torch.arange(3), torch.arange(4), indexing="xy")
ok(tuple(xi.shape) == (4, 3), "meshgrid xy shape")

# sparse_coo_tensor (hybrid) -> sum over sparse dim -> dense, vs manual scatter.
# (mirrors mmdet free_anchor_retina_head's box-prob construction.)
idx = torch.from_numpy(np.array([[0, 1, 2, 0], [1, 0, 1, 1]], dtype="int64"))  # (2,nnz)
vals = torch.from_numpy(rng.rand(4, 3).astype("float32"))                       # hybrid: tail=3
sp = torch.sparse_coo_tensor(idx, vals)                       # shape (3, 2, 3)
dense = sp.to_dense()
manual = np.zeros((3, 2, 3), dtype="float32")
vn = vals.numpy()
for k in range(4):
    manual[idx.numpy()[0, k], idx.numpy()[1, k]] += vn[k]
close(dense, manual, "sparse_coo_tensor -> to_dense")
summed = torch.sparse.sum(sp, dim=0).to_dense()              # sum over first sparse dim
close(summed, manual.sum(0), "sparse.sum(dim=0).to_dense")
# with explicit size + .to_dense() one-shot (the other free_anchor call site).
sp2 = torch.sparse_coo_tensor(idx, torch.from_numpy(rng.rand(4).astype("float32")),
                              size=(3, 2)).to_dense()
ok(tuple(sp2.shape) == (3, 2), "sparse_coo_tensor explicit size -> dense shape")

# nn.SyncBatchNorm forward == BatchNorm2d forward; convert_sync_batchnorm round-trips.
bn = nn.BatchNorm2d(4); sbn = nn.SyncBatchNorm(4)
xb = torch.from_numpy(rng.rand(2, 4, 5, 5).astype("float32"))
bn.eval(); sbn.eval()
close(sbn(xb), bn(xb), "SyncBatchNorm fwd == BatchNorm2d fwd")
conv = nn.Conv2d(3, 4, 3)
ok(nn.SyncBatchNorm.convert_sync_batchnorm(conv) is conv, "convert_sync_batchnorm returns model")

# MultiheadAttention forward shape (batch_first).
mha = nn.MultiheadAttention(8, 2, batch_first=True)
q = torch.from_numpy(rng.rand(2, 5, 8).astype("float32"))
att_out, _ = mha(q, q, q)
ok(tuple(att_out.shape) == (2, 5, 8), "MultiheadAttention out shape")

# Tensor method correctness: relu / clamp_min / eq / diff / fliplr.
rv = torch.from_numpy(np.array([-1.0, 0.0, 2.0], dtype="float32"))
close(rv.relu(), np.array([0, 0, 2], "float32"), "Tensor.relu")
close(rv.clamp_min(0.5), np.array([0.5, 0.5, 2.0], "float32"), "Tensor.clamp_min")
ok(rv.eq(0.0).numpy().tolist() == [False, True, False], "Tensor.eq")
dv = torch.from_numpy(np.array([1.0, 4.0, 9.0, 16.0], dtype="float32"))
close(dv.diff(), np.array([3, 5, 7], "float32"), "Tensor.diff")
flr = torch.from_numpy(rng.rand(2, 3).astype("float32"))
close(flr.fliplr(), flr.numpy()[:, ::-1], "Tensor.fliplr")

# pad (constant + replicate) shapes.
pin = torch.from_numpy(rng.rand(1, 1, 3, 3).astype("float32"))
ok(tuple(F.pad(pin, (1, 1, 1, 1)).shape) == (1, 1, 5, 5), "F.pad constant shape")
ok(tuple(F.pad(pin, (1, 1, 1, 1), mode="replicate").shape) == (1, 1, 5, 5), "F.pad replicate shape")

print(f"\n[test_mmdet_ops] {PASS} passed, {FAIL} failed")
if FAIL and __name__ == "__main__":
    raise SystemExit(1)


# pytest entry point
def test_mmdet_ops():
    assert FAIL == 0, f"{FAIL} mmdet-op checks failed"
