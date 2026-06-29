"""Stub torchvision for jittor-as-torch text path (not functional).

A meta-path finder fabricates ANY torchvision.* (sub)module on demand as a
permissive package, so arbitrary nested imports
(`import torchvision.transforms.v2.functional`, `from torchvision.io import X`)
all succeed. Attribute access returns auto-created stub classes.
"""
import sys, types, importlib.abc, importlib.machinery
__version__ = "2.11.0"


class _IMMeta(type):
    def __getattr__(cls, name):
        return name.lower()
class InterpolationMode(metaclass=_IMMeta):
    BILINEAR = "bilinear"; BICUBIC = "bicubic"; NEAREST = "nearest"
    NEAREST_EXACT = "nearest_exact"; LANCZOS = "lanczos"; HAMMING = "hamming"; BOX = "box"

class _IRMMeta(type):
    def __getattr__(cls, name): return 3
class ImageReadMode(metaclass=_IRMMeta):
    UNCHANGED = 0; GRAY = 1; GRAY_ALPHA = 2; RGB = 3; RGB_ALPHA = 4
# Real tensor transforms (mmrotate h2rbox_v2 self-supervised branch uses
# transforms.functional.vflip). Implemented on jittor Vars.
import jittor as _jt
def _vflip(img): return _jt.flip(img, -2)            # flip height
def _hflip(img): return _jt.flip(img, -1)            # flip width
def _rot90(img, k=1, dims=(-2, -1)):
    out = img
    for _ in range(k % 4):
        out = _jt.flip(out, dims[1]).transpose(dims[0], dims[1])
    return out
def _rotate(img, angle, *a, **k):                    # 90-multiple fast path; else identity-ish
    a90 = int(round(angle / 90.0)) % 4
    return _rot90(img, a90) if a90 else img

# ---- real functional transforms for the multimodal (VL) image path ----
# transformers' "fast" image processor (Qwen2/2.5-VL etc.) drives preprocessing
# through torchvision.transforms.functional on tensors. The stub made tvF.X a
# dummy class -> `pil_to_tensor(img)` returned a class with no .ndim and crashed.
# Implement the handful of ops the VL image pipeline uses, on jittor Vars / PIL.
def _pil_to_tensor(pic):
    # PIL image -> uint8 CHW Var (torchvision.pil_to_tensor semantics)
    import numpy as _np
    arr = _np.asarray(pic)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    t = _jt.array(arr.copy())           # HWC
    return t.permute(2, 0, 1)           # CHW

def _resize(img, size, interpolation="bilinear", antialias=True, **k):
    # img: CHW (or NCHW) Var. size: int or (h,w). Route through jittor interpolate.
    interp = getattr(interpolation, "value", interpolation)
    interp = str(interp).lower()
    mode = {"bilinear": "bilinear", "bicubic": "bicubic", "nearest": "nearest",
            "nearest_exact": "nearest", "box": "area"}.get(interp, "bilinear")
    if isinstance(size, int):
        # torchvision: scale shorter side to `size`, keep aspect
        c, h, w = img.shape[-3], img.shape[-2], img.shape[-1]
        if h <= w: nh, nw = size, int(round(size * w / h))
        else:      nh, nw = int(round(size * h / w)), size
    else:
        nh, nw = int(size[0]), int(size[1])
    x = img if img.ndim == 4 else img.unsqueeze(0)
    was_uint8 = "uint8" in str(x.dtype)
    xf = x.float32()
    align = False if mode in ("bilinear", "bicubic") else None
    try:
        y = _nn.interpolate(xf, size=(nh, nw), mode=mode,
                            align_corners=align) if align is not None \
            else _nn.interpolate(xf, size=(nh, nw), mode=mode)
    except Exception:
        y = _nn.interpolate(xf, size=(nh, nw), mode="bilinear", align_corners=False)
    if was_uint8:
        y = y.round().clamp(0, 255).uint8()
    return y if img.ndim == 4 else y.squeeze(0)

def _normalize(img, mean, std, inplace=False, **k):
    m = _jt.array(mean).reshape((-1, 1, 1)).float32()
    s = _jt.array(std).reshape((-1, 1, 1)).float32()
    return (img.float32() - m) / s

def _pad(img, padding, fill=0, padding_mode="constant", **k):
    # padding: int or [l,t,r,b] (torchvision) ; jittor F.pad wants (l,r,t,b)
    if isinstance(padding, int):
        l = r = t = b = padding
    elif len(padding) == 2:
        l = r = padding[0]; t = b = padding[1]
    else:
        l, t, r, b = padding
    return _nn.pad(img, [l, r, t, b], mode="constant", value=fill)

def _crop(img, top, left, height, width, **k):
    return img[..., top:top + height, left:left + width]

import jittor.nn as _nn
_CONCRETE = {"InterpolationMode": InterpolationMode, "ImageReadMode": ImageReadMode,
             "decode_image": (lambda *a, **k: None), "read_image": (lambda *a, **k: None),
             "encode_jpeg": (lambda *a, **k: None), "decode_jpeg": (lambda *a, **k: None),
             "vflip": _vflip, "hflip": _hflip, "rot90": _rot90, "rotate": _rotate,
             "pil_to_tensor": _pil_to_tensor, "to_tensor": _pil_to_tensor,
             "resize": _resize, "normalize": _normalize, "pad": _pad, "crop": _crop}


class _AnyModule(types.ModuleType):
    __path__ = []   # makes it a package so submodule imports descend
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        if name in _CONCRETE:
            return _CONCRETE[name]
        if name in ("functional", "v2", "transforms", "F"):   # real submodules
            return _AnyModule(getattr(self, "__name__", "torchvision") + "." + name)
        return type(name, (), {"__init__": lambda self, *a, **k: None})


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname == "torchvision" or fullname.startswith("torchvision."):
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None
    def create_module(self, spec):
        return _AnyModule(spec.name)
    def exec_module(self, module):
        pass

sys.meta_path.insert(0, _Finder())
