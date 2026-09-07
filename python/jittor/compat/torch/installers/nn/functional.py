import jittor as jt
from jittor import nn
from ...types import _dtype_to_str

def _pixel_shuffle(input, upscale_factor):
    r = upscale_factor
    N, Cr2, H, W = input.shape
    C = Cr2 // (r * r)
    return input.reshape((N, C, r, r, H, W)).permute(0, 1, 4, 2, 5, 3).reshape((N, C, H * r, W * r))

def _pixel_unshuffle(input, downscale_factor):
    r = downscale_factor
    N, C, H, W = input.shape
    return input.reshape((N, C, H // r, r, W // r, r)).permute(0, 1, 3, 5, 2, 4).reshape((N, C * r * r, H // r, W // r))

def _gumbel_softmax(logits, tau=1.0, hard=False, eps=1e-10, dim=-1):
    u = jt.rand(logits.shape)
    g = -jt.log(-jt.log(u + eps) + eps)             # Gumbel(0,1) noise
    y = nn.softmax((logits + g) / tau, dim=dim)
    if hard:
        m = y.max(dim, keepdims=True)
        y_hard = (y >= m).float32()
        y = (y_hard - y).stop_grad() + y            # straight-through estimator
    return y

def _rms_norm(input, normalized_shape, weight=None, eps=None):
    if eps is None:
        eps = 1.1920929e-07                          # finfo(float32).eps, torch default
    ndn = len(normalized_shape) if hasattr(normalized_shape, "__len__") else 1
    dims = list(range(input.ndim - ndn, input.ndim))
    out = input * (1.0 / jt.sqrt((input * input).mean(dims, keepdims=True) + eps))
    return out * weight if weight is not None else out

def _selu(input, inplace=False):
    a = 1.6732632423543772848170429916717
    s = 1.0507009873554804934193349852946
    return s * (jt.maximum(input, 0.0) + jt.minimum(0.0, a * (jt.exp(input) - 1)))

def _threshold(input, threshold, value, inplace=False):
    m = (input > threshold).float32()
    return m * input + (1 - m) * value

def _triplet(anchor, positive, negative, margin=1.0, p=2.0, eps=1e-6,
             swap=False, size_average=None, reduce=None, reduction="mean"):
    def _d(a, b):
        return ((jt.abs(a - b) ** p).sum(-1) + eps) ** (1.0 / p)
    dp, dn = _d(anchor, positive), _d(anchor, negative)
    if swap:
        dn = jt.minimum(dn, _d(positive, negative))
    loss = jt.maximum(dp - dn + margin, 0.0)
    return loss.mean() if reduction == "mean" else (loss.sum() if reduction == "sum" else loss)

def _poisson_nll(input, target, log_input=True, full=False, size_average=None,
                 eps=1e-8, reduce=None, reduction="mean"):
    loss = (jt.exp(input) - target * input) if log_input else (input - target * jt.log(input + eps))
    if full:
        import math as _mp
        stir = target * jt.log(jt.maximum(target, eps)) - target + 0.5 * jt.log(2 * _mp.pi * jt.maximum(target, eps))
        loss = loss + jt.ternary(target > 1, stir, jt.zeros_like(target))
    return loss.mean() if reduction == "mean" else (loss.sum() if reduction == "sum" else loss)

def _install_functional(ctx):
    _modules = ctx.registry.module_map
    g = ctx.jittor_module
    Var = ctx.state["Var"]
    _DTYPE_OBJS = ctx.state["dtypes"]
    if not hasattr(nn, "functional"):
        import types as _types
        F = _types.ModuleType("jittor.nn.functional")
        for fname in dir(nn):
            fobj = getattr(nn, fname)
            if callable(fobj) and not isinstance(fobj, type):
                setattr(F, fname, fobj)
    else:
        F = nn.functional
    if hasattr(nn, "relu"): F.relu = nn.relu
    if hasattr(nn, "gelu"): F.gelu = nn.gelu
    if hasattr(nn, "softmax"):
        # torch: F.softmax(input, dim=None, _stacklevel=3, dtype=None).
        # When dtype is given, input is cast to it before softmax (used by
        # transformers' eager attention: F.softmax(scores, dim=-1, dtype=fp32)).
        _jt_softmax = nn.softmax
        def _softmax(input, dim=-1, _stacklevel=3, dtype=None):
            if dtype is not None:
                input = input.cast(_dtype_to_str(dtype))
            return _jt_softmax(input, dim=dim)
        F.softmax = _softmax
    if hasattr(nn, "linear"): F.linear = nn.linear
    if hasattr(nn, "interpolate"):
        # torch.nn.functional.interpolate defaults to mode='nearest', but
        # jittor.nn.interpolate defaults to 'bilinear'. Code that omits the
        # mode (e.g. YOLOV3Neck: F.interpolate(x, scale_factor=2)) silently
        # gets the wrong upsampling. Wrap so the torch-shim functional matches
        # torch's default and accepts torch's arg name / extra kwargs. Only
        # this shim copy is affected, not jittor's native nn.interpolate.
        _jt_interpolate = nn.interpolate
        def _interpolate(input=None, size=None, scale_factor=None,
                         mode="nearest", align_corners=None,
                         recompute_scale_factor=None, antialias=False,
                         **_kw):
            if input is None:
                input = _kw.pop("X")
            ac = False if align_corners is None else align_corners
            return _jt_interpolate(input, size=size,
                                   scale_factor=scale_factor, mode=mode,
                                   align_corners=ac)
        F.interpolate = _interpolate
    if hasattr(nn, "cross_entropy_loss"):
        _jt_ce = nn.cross_entropy_loss
        # torch.nn.functional.cross_entropy(..., label_smoothing=): jittor's
        # cross_entropy_loss has no label_smoothing (used by many training recipes:
        # ImageNet, translation, some SFT). Delegate to jittor for ls=0 (verified
        # correct incl. weight/ignore_index); implement smoothing to match torch:
        #   loss_i = (1-ls)*nll_i + (ls/C)*smooth_i,  nll_i = -w[t]*logp[i,t],
        #   smooth_i = -sum_c(w_c*logp[i,c]);  mean divides by sum(w[t]) (or count).
        def _cross_entropy(input, target, weight=None, size_average=None,
                           ignore_index=-100, reduce=None, reduction="mean",
                           label_smoothing=0.0):
            # torch: a floating-point target with the SAME shape as input is a
            # class-probability ("soft label") target (mixup / distillation / soft
            # label-smoothing). jittor's cross_entropy_loss only understands integer
            # class-index targets, so handle the soft case here.
            if (isinstance(target, jt.Var) and target.ndim == input.ndim
                    and "int" not in str(target.dtype)):
                Cc = int(input.shape[1]) if input.ndim >= 2 else int(input.shape[-1])
                cdim = 1 if input.ndim >= 2 else -1
                logp = nn.log_softmax(input, dim=cdim)
                tgt = target
                if label_smoothing:
                    tgt = (1.0 - label_smoothing) * tgt + label_smoothing / Cc
                if weight is not None:
                    wsh = [1] * input.ndim; wsh[cdim] = Cc
                    wloss = -(tgt * logp * weight.reshape(wsh)).sum(dim=cdim)
                else:
                    wloss = -(tgt * logp).sum(dim=cdim)
                if reduction == "sum":
                    return wloss.sum()
                if reduction == "none":
                    return wloss
                return wloss.mean()        # torch divides the soft-target loss by N
            if not label_smoothing:
                ii = -100 if ignore_index is None else ignore_index
                return _jt_ce(input, target, weight=weight, ignore_index=ii,
                              reduction=reduction)
            C = int(input.shape[1]) if input.ndim >= 2 else int(input.shape[-1])
            if input.ndim > 2:                  # (N,C,d...) -> (M,C)
                perm = [0] + list(range(2, input.ndim)) + [1]
                x = input.transpose(perm).reshape((-1, C))
            else:
                x = input
            t = target.reshape((-1,))
            logp = nn.log_softmax(x, dim=-1)
            ig = None if ignore_index is None else ignore_index
            t_safe = t if ig is None else jt.ternary(t == ig, jt.zeros_like(t), t)
            nll = -logp.gather(1, t_safe.reshape((-1, 1))).reshape((-1,))
            if weight is not None:
                wt = weight[t_safe]
                nll = nll * wt
                smooth = -(logp * weight.reshape((1, -1))).sum(dim=-1)
            else:
                wt = None
                smooth = -logp.sum(dim=-1)
            loss = (1.0 - label_smoothing) * nll + (label_smoothing / C) * smooth
            if ig is not None:
                keep = (t != ig).float32()
                loss = loss * keep
                norm = (wt * keep).sum() if wt is not None else keep.sum()
            else:
                norm = wt.sum() if wt is not None else jt.array(float(t.shape[0]))
            if reduction == "sum":
                return loss.sum()
            if reduction == "none":
                return loss.reshape(target.shape) if input.ndim > 2 else loss
            return loss.sum() / norm
        F.cross_entropy = _cross_entropy
    # These losses are native functional implementations.  Torch mode only
    # publishes the canonical objects; keeping a second fallback body here
    # would make signatures and fixes diverge between the two entry points.
    from jittor.nn.functional.loss import (
        binary_cross_entropy as _native_bce,
        cosine_embedding_loss as _native_cosine_embedding,
        gaussian_nll_loss as _native_gaussian_nll,
        huber_loss as _native_huber,
        kl_div as _native_kl_div,
        margin_ranking_loss as _native_margin_ranking,
    )
    for _name, _fn in (
        ("binary_cross_entropy", _native_bce),
        ("cosine_embedding_loss", _native_cosine_embedding),
        ("gaussian_nll_loss", _native_gaussian_nll),
        ("huber_loss", _native_huber),
        ("kl_div", _native_kl_div),
        ("margin_ranking_loss", _native_margin_ranking),
    ):
        setattr(F, _name, _fn)
    # nn.*Loss class versions (criterion = nn.HuberLoss()): thin wrappers over the
    # functional. KLDivLoss/BCELoss/BCEWithLogitsLoss/CrossEntropyLoss/MSELoss/L1Loss
    # already exist on jittor.nn (verified correct); add the rest.
    _Mod = nn.Module
    def _add_loss_class(cname, fn, defaults, arg_order):
        if hasattr(nn, cname):
            return
        class _L(_Mod):
            def __init__(self, *a, **k):
                super().__init__()
                self._kw = dict(defaults); self._kw.update(k)
                for nm, val in zip(arg_order, a):
                    self._kw[nm] = val
            def execute(self, *inputs):
                return fn(*inputs, **self._kw)
        _L.__name__ = cname
        setattr(nn, cname, _L)
    _add_loss_class("HuberLoss", F.huber_loss, dict(reduction="mean", delta=1.0), ("reduction", "delta"))
    _add_loss_class("SmoothL1Loss", F.smooth_l1_loss, dict(reduction="mean"), ("reduction",))
    _add_loss_class("MarginRankingLoss", F.margin_ranking_loss, dict(margin=0.0, reduction="mean"), ("margin", "reduction"))
    _add_loss_class("CosineEmbeddingLoss", F.cosine_embedding_loss, dict(margin=0.0, reduction="mean"), ("margin", "reduction"))
    _add_loss_class("GaussianNLLLoss", F.gaussian_nll_loss, dict(full=False, eps=1e-6, reduction="mean"), ("full", "eps", "reduction"))
    _add_loss_class("NLLLoss", F.nll_loss, dict(reduction="mean"), ("weight", "size_average", "ignore_index"))
    # pixel_shuffle / pixel_unshuffle (super-resolution, some VAE decoders): jittor's
    # functional lacks them. (N, C*r^2, H, W) <-> (N, C, H*r, W*r). Verified vs torch.
    if not hasattr(F, "pixel_shuffle"):
        F.pixel_shuffle = _pixel_shuffle
        g.pixel_shuffle = _pixel_shuffle
    if not hasattr(F, "pixel_unshuffle"):
        F.pixel_unshuffle = _pixel_unshuffle
        g.pixel_unshuffle = _pixel_unshuffle
    for _pscn, _psfn in (("PixelShuffle", "pixel_shuffle"), ("PixelUnshuffle", "pixel_unshuffle")):
        if not hasattr(nn, _pscn):
            def _mk(fn):
                class _PS(nn.Module):
                    def __init__(self, factor): super().__init__(); self._f = factor
                    def execute(self, x): return getattr(F, fn)(x, self._f)
                return _PS
            _cls = _mk(_psfn); _cls.__name__ = _pscn; setattr(nn, _pscn, _cls)
    # F.logsigmoid (DPO/preference losses), F.gumbel_softmax (discrete/MoE sampling).
    if not hasattr(F, "logsigmoid"):
        # stable: log(sigmoid(x)) = min(x,0) - log(1+exp(-|x|))
        F.logsigmoid = lambda input: jt.minimum(input, 0.0) - jt.log(1.0 + jt.exp(-jt.abs(input)))
    if not hasattr(F, "gumbel_softmax"):
        F.gumbel_softmax = _gumbel_softmax
    if not hasattr(F, "rms_norm"):
        # F.rms_norm (torch 2.4+): x / sqrt(mean(x^2, over last len(normalized_shape)
        # dims) + eps) * weight. The norm modern LLMs (Llama/Qwen/Gemma) use.
        F.rms_norm = _rms_norm
    # Activations / losses jittor's functional lacked (verified vs real torch 2.12).
    if not hasattr(F, "softmin"):
        F.softmin = lambda input, dim=-1, _stacklevel=3, dtype=None: nn.softmax(-input, dim=dim)
    if not hasattr(F, "tanhshrink"):
        F.tanhshrink = lambda input: input - jt.tanh(input)
    if not hasattr(F, "celu"):
        F.celu = lambda input, alpha=1.0, inplace=False: \
            jt.maximum(input, 0.0) + jt.minimum(0.0, alpha * (jt.exp(input / alpha) - 1))
    if not hasattr(F, "selu"):
        F.selu = _selu
    if not hasattr(F, "threshold"):
        F.threshold = _threshold
    if not hasattr(F, "triplet_margin_loss"):
        F.triplet_margin_loss = _triplet
    if not hasattr(F, "poisson_nll_loss"):
        F.poisson_nll_loss = _poisson_nll
    if not hasattr(F, "ctc_loss"):
        # F.ctc_loss (wav2vec2 / speech ASR): the CTC forward (alpha) DP in log space.
        # log_probs (T,N,C) log-softmax; targets (N,S) padded or 1-D concatenated.
        # Differentiable (grad flows to log_probs). Verified bit-equal to real torch.
        import numpy as _np_ctc
        _CNEG = -1e30
        def _ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0,
                      reduction="mean", zero_infinity=False):
            def _ints(v):
                return [int(x) for x in (v.numpy().reshape(-1) if isinstance(v, jt.Var) else _np_ctc.asarray(v).reshape(-1))]
            in_lens, tgt_lens = _ints(input_lengths), _ints(target_lengths)
            tnp = targets.numpy() if isinstance(targets, jt.Var) else _np_ctc.asarray(targets)
            flat = (tnp.ndim == 1)
            def _shift(v, k):
                return jt.concat([jt.full((k,), _CNEG), v[:int(v.shape[0]) - k]]) if k > 0 else v
            def _lse(mats):
                m = mats[0]
                for x in mats[1:]:
                    m = jt.maximum(m, x)
                return m + jt.safe_log(sum(jt.exp(x - m) for x in mats))
            N = log_probs.shape[1]
            losses, offset = [], 0
            for n in range(N):
                Tn, Sn = in_lens[n], tgt_lens[n]
                if flat:
                    seq = [int(x) for x in tnp[offset:offset + Sn]]; offset += Sn
                else:
                    seq = [int(x) for x in tnp[n, :Sn]]
                ext = [blank]
                for lab in seq:
                    ext += [lab, blank]
                L = len(ext)
                ext_idx = jt.array(_np_ctc.array(ext, dtype="int64"))
                skip = _np_ctc.zeros(L, dtype="float32")
                for s in range(2, L):
                    if ext[s] != blank and ext[s] != ext[s - 2]:
                        skip[s] = 1.0
                skip_v = jt.array(skip)
                start = _np_ctc.full(L, _CNEG, dtype="float32"); start[0] = 0.0
                if L > 1:
                    start[1] = 0.0
                lp_n = log_probs[:Tn, n, :]
                alpha = lp_n[0][ext_idx] + jt.array(start)
                for t in range(1, Tn):
                    a2 = _shift(alpha, 2) * skip_v + (1 - skip_v) * _CNEG
                    alpha = lp_n[t][ext_idx] + _lse([alpha, _shift(alpha, 1), a2])
                losses.append(-(_lse([alpha[L - 1], alpha[L - 2]]) if L > 1 else alpha[L - 1]))
            out = jt.stack(losses).reshape((N,))   # (N,1)->(N,): jittor has no 0-d scalar
            if zero_infinity:
                out = jt.ternary(jt.isfinite(out), out, jt.zeros_like(out))
            if reduction == "none":
                return out
            if reduction == "sum":
                return out.sum()
            tl = jt.array(_np_ctc.array([max(s, 1) for s in tgt_lens], dtype="float32"))
            return (out / tl).mean()
        F.ctc_loss = _ctc_loss
    if hasattr(nn, "layer_norm"): F.layer_norm = nn.layer_norm
    if hasattr(nn, "embedding"): F.embedding = nn.embedding
    nn.functional = F
    g.nn.functional = nn.functional
    if not hasattr(nn.functional, "cosine_similarity") and hasattr(nn, "cosine_similarity"):
        nn.functional.cosine_similarity = nn.cosine_similarity
    if not hasattr(nn.functional, "pairwise_distance") and hasattr(nn, "pairwise_distance"):
        nn.functional.pairwise_distance = nn.pairwise_distance
