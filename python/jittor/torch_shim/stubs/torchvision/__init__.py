"""Minimal torchvision facade for jittor-as-torch.

The module still fabricates unknown torchvision submodules on demand so import
probes succeed, but common image transforms, image saving and classification
models are backed by real Jittor implementations.
"""
import sys, types, importlib.abc, importlib.machinery
import functools
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

def _to_tensor(pic):
    # torchvision.transforms.functional.to_tensor semantics:
    # PIL/uint8 ndarray -> float CHW in [0,1].
    import numpy as _np
    from PIL import Image as _Im
    if isinstance(pic, _jt.Var):
        return pic.float32() / 255.0 if "uint8" in str(pic.dtype) else pic.float32()
    if isinstance(pic, _Im.Image):
        arr = _np.asarray(pic)
    else:
        arr = _np.asarray(pic)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.dtype == _np.uint8:
        arr = arr.astype(_np.float32) / 255.0
    else:
        arr = arr.astype(_np.float32)
    if arr.ndim == 3:
        arr = arr.transpose(2, 0, 1)
    return _jt.array(arr.copy())

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

# ---- real class-based transforms (torchvision.transforms.{Compose,Normalize,
# Resize,ToTensor,ToPILImage,...}) ----
# The meta-path stub used to fabricate these as no-op dummy classes (a Compose
# instance was not callable -> TypeError). TRELLIS.2's DinoV3FeatureExtractor
# does `Compose([Normalize(mean,std)])(tensor)`, and rembg.BiRefNet uses
# Resize/ToTensor/Normalize/ToPILImage. Implement them on jittor Vars / PIL so
# the default image pipeline runs under `import jittor as torch` (no adapter).
def _pil_resample(interp):
    # map torchvision InterpolationMode / string -> PIL.Image.Resampling
    from PIL import Image as _Im
    s = str(getattr(interp, "value", interp)).lower() if interp is not None else "bilinear"
    return {"bilinear": _Im.Resampling.BILINEAR, "bicubic": _Im.Resampling.BICUBIC,
            "nearest": _Im.Resampling.NEAREST, "nearest_exact": _Im.Resampling.NEAREST,
            "lanczos": _Im.Resampling.LANCZOS, "box": _Im.Resampling.BOX,
            "hamming": _Im.Resampling.HAMMING}.get(s, _Im.Resampling.BILINEAR)


class Compose:
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self):
        return "Compose(" + ", ".join(repr(t) for t in self.transforms) + ")"


class Normalize:
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        # tensor: (C,H,W) or (B,C,H,W) jittor Var / array.
        import numpy as _np
        t = tensor if isinstance(tensor, _jt.Var) else _jt.array(tensor)
        t = t.float32()
        c = t.shape[-3]
        m = _jt.array(_np.asarray(self.mean, dtype=_np.float32)).reshape((c, 1, 1))
        s = _jt.array(_np.asarray(self.std, dtype=_np.float32)).reshape((c, 1, 1))
        return (t - m) / s

    def __repr__(self):
        return f"Normalize(mean={self.mean}, std={self.std})"


class ToTensor:
    def __call__(self, pic):
        return _to_tensor(pic)

    def __repr__(self):
        return "ToTensor()"


class Resize:
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR,
                 max_size=None, antialias=True, **kw):
        self.size = size
        self.interpolation = interpolation
        self.max_size = max_size
        self.antialias = antialias

    def _hw(self, h, w):
        size = self.size
        if isinstance(size, int):
            # torchvision: scale shorter side to `size`, keep aspect ratio.
            if h <= w:
                nh, nw = size, int(round(size * w / h))
            else:
                nh, nw = int(round(size * h / w)), size
            if self.max_size is not None and max(nh, nw) > self.max_size:
                scale = self.max_size / max(nh, nw)
                nh, nw = int(round(nh * scale)), int(round(nw * scale))
            return nh, nw
        return int(size[0]), int(size[1])

    def __call__(self, pic):
        from PIL import Image as _Im
        if isinstance(pic, _Im.Image):
            nh, nw = self._hw(pic.height, pic.width)
            return pic.resize((nw, nh), _pil_resample(self.interpolation))
        # tensor (C,H,W) or (B,C,H,W): bilinear/bicubic/nearest via jittor.
        t = pic if isinstance(pic, _jt.Var) else _jt.array(pic)
        nh, nw = self._hw(t.shape[-2], t.shape[-1])
        s = str(getattr(self.interpolation, "value", self.interpolation)).lower()
        mode = {"bilinear": "bilinear", "bicubic": "bicubic", "nearest": "nearest",
                "nearest_exact": "nearest"}.get(s, "bilinear")
        x = t.float32() if t.ndim == 4 else t.float32().unsqueeze(0)
        if mode in ("bilinear", "bicubic"):
            y = _nn.interpolate(x, size=(nh, nw), mode=mode, align_corners=False)
        else:
            y = _nn.interpolate(x, size=(nh, nw), mode=mode)
        return y if t.ndim == 4 else y.squeeze(0)

    def __repr__(self):
        return f"Resize(size={self.size}, interpolation={self.interpolation})"


class ToPILImage:
    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, pic):
        from PIL import Image as _Im
        import numpy as _np
        if isinstance(pic, _Im.Image):
            return pic
        t = pic.numpy() if isinstance(pic, _jt.Var) else _np.asarray(pic)
        t = _np.asarray(t, dtype=_np.float32)
        if t.ndim == 3 and t.shape[0] in (1, 3, 4):
            t = t.transpose(1, 2, 0)
        if t.ndim == 3 and t.shape[2] == 1:
            t = t[:, :, 0]
        if t.size and t.max() <= 1.0 + 1e-4 and t.min() >= -1e-4:
            t = t * 255.0
        t = _np.clip(t, 0, 255).astype(_np.uint8)
        return _Im.fromarray(t, mode=self.mode)

    def __repr__(self):
        return "ToPILImage()"


class Lambda:
    def __init__(self, lambd):
        self.lambd = lambd

    def __call__(self, x):
        return self.lambd(x)

    def __repr__(self):
        return "Lambda()"


class CenterCrop:
    def __init__(self, size):
        self.size = (size, size) if isinstance(size, int) else tuple(size)

    def __call__(self, pic):
        from PIL import Image as _Im
        th, tw = self.size
        if isinstance(pic, _Im.Image):
            w, h = pic.width, pic.height
            l = int(round((w - tw) / 2.0)); t = int(round((h - th) / 2.0))
            return pic.crop((l, t, l + tw, t + th))
        x = pic
        h, w = x.shape[-2], x.shape[-1]
        t = int(round((h - th) / 2.0)); l = int(round((w - tw) / 2.0))
        return x[..., t:t + th, l:l + tw]

    def __repr__(self):
        return f"CenterCrop(size={self.size})"


def _make_grid(tensor, nrow=8, padding=2, normalize=False, value_range=None,
               scale_each=False, pad_value=0, **kw):
    if value_range is None and "range" in kw:
        value_range = kw["range"]
    return _jt.make_grid(tensor, nrow=nrow, padding=padding,
                         normalize=normalize, range=value_range,
                         scale_each=scale_each, pad_value=pad_value)


def _save_image(tensor, fp, format=None, **kwargs):
    grid = _make_grid(tensor, **kwargs)
    _jt.save_image(grid, fp, format=format)


def _model_features(model):
    return getattr(model, "features", model)


def _drop_classifier_kwargs(kwargs):
    kwargs = dict(kwargs)
    kwargs.pop("weights", None)
    kwargs.pop("progress", None)
    kwargs.pop("init_weights", None)
    return kwargs


def _vgg16(pretrained=False, weights=None, progress=True, **kwargs):
    from jittor.models import vgg16
    return vgg16(pretrained=bool(weights is not None or pretrained),
                 **_drop_classifier_kwargs(kwargs))


def _alexnet(pretrained=False, weights=None, progress=True, **kwargs):
    from jittor.models import alexnet
    return alexnet(pretrained=bool(weights is not None or pretrained),
                   **_drop_classifier_kwargs(kwargs))


def _squeezenet1_1(pretrained=False, weights=None, progress=True, **kwargs):
    from jittor.models import squeezenet1_1
    return squeezenet1_1(pretrained=bool(weights is not None or pretrained),
                         **_drop_classifier_kwargs(kwargs))


def _resnet50(pretrained=False, weights=None, progress=True, **kwargs):
    from jittor.models import resnet50
    return resnet50(pretrained=bool(weights is not None or pretrained),
                    **_drop_classifier_kwargs(kwargs))


class _WeightEnum(metaclass=_IRMMeta):
    DEFAULT = "default"
    IMAGENET1K_V1 = "imagenet1k_v1"


class _ModelsModule(types.ModuleType):
    VGG16_Weights = type("VGG16_Weights", (_WeightEnum,), {})
    AlexNet_Weights = type("AlexNet_Weights", (_WeightEnum,), {})
    SqueezeNet1_1_Weights = type("SqueezeNet1_1_Weights", (_WeightEnum,), {})
    ResNet50_Weights = type("ResNet50_Weights", (_WeightEnum,), {})
    vgg16 = staticmethod(_vgg16)
    alexnet = staticmethod(_alexnet)
    squeezenet1_1 = staticmethod(_squeezenet1_1)
    resnet50 = staticmethod(_resnet50)


class _UtilsModule(types.ModuleType):
    make_grid = staticmethod(_make_grid)
    save_image = staticmethod(_save_image)


_CONCRETE = {"InterpolationMode": InterpolationMode, "ImageReadMode": ImageReadMode,
             "decode_image": (lambda *a, **k: None), "read_image": (lambda *a, **k: None),
             "encode_jpeg": (lambda *a, **k: None), "decode_jpeg": (lambda *a, **k: None),
             "vflip": _vflip, "hflip": _hflip, "rot90": _rot90, "rotate": _rotate,
             "pil_to_tensor": _pil_to_tensor, "to_tensor": _to_tensor,
             "resize": _resize, "normalize": _normalize, "pad": _pad, "crop": _crop,
             "make_grid": _make_grid, "save_image": _save_image,
             # real class-based transforms (torchvision.transforms.*)
             "Compose": Compose, "Normalize": Normalize, "ToTensor": ToTensor,
             "Resize": Resize, "ToPILImage": ToPILImage, "Lambda": Lambda,
             "CenterCrop": CenterCrop}


class _AnyModule(types.ModuleType):
    __path__ = []   # makes it a package so submodule imports descend
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        if name in _CONCRETE:
            return _CONCRETE[name]
        if name == "models":
            return _models
        if name == "utils":
            return _utils
        if name in ("functional", "v2", "transforms", "F"):   # real submodules
            return _module_for_name(getattr(self, "__name__", "torchvision") + "." + name)
        return type(name, (), {"__init__": lambda self, *a, **k: None})


@functools.lru_cache(maxsize=None)
def _module_for_name(name):
    if name == "torchvision.models":
        return _models
    if name == "torchvision.utils":
        return _utils
    return _AnyModule(name)


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname == "torchvision" or fullname.startswith("torchvision."):
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None
    def create_module(self, spec):
        return _module_for_name(spec.name)
    def exec_module(self, module):
        pass

sys.meta_path.insert(0, _Finder())
_models = _ModelsModule("torchvision.models")
_utils = _UtilsModule("torchvision.utils")
models = _models
utils = _utils
transforms = _module_for_name("torchvision.transforms")
sys.modules.setdefault("torchvision.models", _models)
sys.modules.setdefault("torchvision.utils", _utils)
sys.modules.setdefault("torchvision.transforms", transforms)
sys.modules.setdefault("torchvision.transforms.functional",
                       _module_for_name("torchvision.transforms.functional"))


def __getattr__(name):
    if name in _CONCRETE:
        return _CONCRETE[name]
    if name == "models":
        return _models
    if name == "utils":
        return _utils
    if name in ("transforms", "functional", "v2", "F"):
        return _module_for_name("torchvision." + name)
    raise AttributeError(name)
