"""Minimal torchvision facade for jittor-as-torch.

The module still fabricates unknown torchvision submodules on demand so import
probes succeed, but common image transforms, image saving and classification
models are backed by real Jittor implementations.
"""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name
import sys, types, importlib.abc, importlib.machinery
import functools
__version__ = "2.11.0"


import enum as _enum


class InterpolationMode(_enum.Enum):
    """torchvision.transforms.InterpolationMode (enum values, constructible)."""
    NEAREST = "nearest"
    NEAREST_EXACT = "nearest-exact"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    BOX = "box"
    HAMMING = "hamming"
    LANCZOS = "lanczos"

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            normalized = value.lower().replace("_", "-")
            for member in cls:
                if member.value == normalized:
                    return member
        return None

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
def _as_frontend(t):
    """Return a torch.Tensor (shim frontend) so isinstance(x, torch.Tensor) holds."""
    try:
        import torch as _torch
        return _torch.as_tensor(t)
    except Exception:
        return t


def _pil_to_tensor(pic):
    # PIL image -> uint8 CHW Var (torchvision.pil_to_tensor semantics)
    import numpy as _np
    arr = _np.asarray(pic)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    t = _jt.array(arr.copy())           # HWC
    return _as_frontend(t.permute(2, 0, 1))           # CHW

def _to_tensor(pic):
    # torchvision.transforms.functional.to_tensor semantics:
    # PIL/uint8 ndarray -> float CHW in [0,1].
    import numpy as _np
    from PIL import Image as _Im
    if isinstance(pic, _jt.Var):
        return pic.float32() / 255.0 if "uint8" in _jittor_dtype_name(pic.dtype) else pic.float32()
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
    return _as_frontend(_jt.array(arr.copy()))

def _get_image_size(img):
    """torchvision.transforms.functional.get_image_size -> (width, height)."""
    if hasattr(img, "shape"):
        return (int(img.shape[-1]), int(img.shape[-2]))
    from PIL import Image as _Im
    if isinstance(img, _Im.Image):
        return img.size
    raise TypeError("get_image_size expects a tensor or PIL image")


def _get_dimensions(img):
    """torchvision.transforms.functional.get_dimensions -> [C, H, W]."""
    if hasattr(img, "shape"):
        return [int(img.shape[-3]), int(img.shape[-2]), int(img.shape[-1])]
    from PIL import Image as _Im
    if isinstance(img, _Im.Image):
        return [len(img.getbands()), img.size[1], img.size[0]]
    raise TypeError("get_dimensions expects a tensor or PIL image")


def _resize(img, size, interpolation="bilinear", antialias=True, **k):
    # img: CHW (or NCHW) Var or PIL image. size: int or (h,w).
    from PIL import Image as _Im
    if isinstance(img, _Im.Image):
        w, h = img.size
        if isinstance(size, int):
            if h <= w:
                nh, nw = size, int(round(size * w / h))
            else:
                nh, nw = int(round(size * h / w)), size
        else:
            nh, nw = int(size[0]), int(size[1])
        return img.resize((nw, nh), _pil_resample(interpolation))
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
    original_shape = tuple(img.shape)
    if img.ndim == 3:
        x = img.unsqueeze(0)
    else:
        x = img.reshape((-1, original_shape[-3], original_shape[-2], original_shape[-1]))
    was_uint8 = "uint8" in _jittor_dtype_name(x.dtype)
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
    if img.ndim == 3:
        return y.squeeze(0)
    return y.reshape(original_shape[:-2] + (nh, nw))

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
    elif len(padding) == 4:
        l, t, r, b = padding
    else:
        return _nn.pad(img, list(padding), mode="constant", value=fill)
    return _nn.pad(img, [l, r, t, b], mode="constant", value=fill)

def _crop(img, top, left, height, width, **k):
    from PIL import Image as _Im
    if isinstance(img, _Im.Image):
        return img.crop((left, top, left + width, top + height))
    return img[..., top:top + height, left:left + width]


def _resized_crop(img, top, left, height, width, size, interpolation="bilinear",
                  antialias=True, **k):
    return _resize(_crop(img, top, left, height, width), size, interpolation,
                   antialias=antialias)

def _center_crop(img, output_size, **k):
    th, tw = (output_size, output_size) if isinstance(output_size, int) else output_size
    h, w = img.shape[-2:]
    return _crop(img, int(round((h - th) / 2.0)), int(round((w - tw) / 2.0)), th, tw)

def _rgb_to_grayscale(img, num_output_channels=1):
    from PIL import Image as _Im
    if num_output_channels not in (1, 3):
        raise ValueError("num_output_channels must be 1 or 3")
    if isinstance(img, _Im.Image):
        return img.convert("L" if num_output_channels == 1 else "RGB")
    if img.ndim < 3:
        raise TypeError("image tensor must have at least 3 dimensions")
    channels = img.shape[-3]
    if channels == 1:
        gray = img.clone()
    elif channels >= 3:
        gray = (img[..., 0:1, :, :].float32() * 0.2989
                + img[..., 1:2, :, :].float32() * 0.587
                + img[..., 2:3, :, :].float32() * 0.114).cast(img.dtype)
    else:
        raise ValueError("image tensor must have 1, 3, or 4 channels")
    return _jt.concat([gray, gray, gray], dim=-3) if num_output_channels == 3 else gray

def _grayscale_to_rgb(img):
    from PIL import Image as _Im
    if isinstance(img, _Im.Image):
        return img.convert("RGB")
    if img.ndim < 3:
        raise TypeError("image tensor must have at least 3 dimensions")
    if img.shape[-3] >= 3:
        return img
    return _rgb_to_grayscale(img, num_output_channels=3)

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
        mean = _np.asarray(self.mean, dtype=_np.float32)
        std = _np.asarray(self.std, dtype=_np.float32)
        # torchvision accepts per-channel sequences and scalars; a length-1
        # sequence broadcasts over the channel axis, as torch does.
        m = _jt.array(mean).reshape((-1, 1, 1) if mean.ndim else (1, 1, 1))
        s = _jt.array(std).reshape((-1, 1, 1) if std.ndim else (1, 1, 1))
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


def _torchify(model):
    """Give a native jittor model the torch Module surface (to/train/state_dict).

    jittor.models classes inherit the native Module base, not the compat
    torch.nn.Module subclass, so torch references could not call
    ``get_model(...).to(device)``. Mixing the compat base in front keeps the
    model's own execute/forward while exposing the torch methods.
    """
    import torch
    cls = type(model)
    base = torch.nn.Module
    if issubclass(cls, base):
        return model
    attributes = {"__module__": cls.__module__}
    execute = cls.__dict__.get("execute") or getattr(cls, "execute", None)
    if execute is not None:
        attributes["execute"] = execute
    forward = getattr(cls, "forward", None)
    if forward is not None:
        attributes["forward"] = forward
    model.__class__ = type("Torch" + cls.__name__, (base, cls), attributes)
    return model


def _vgg16(pretrained=False, weights=None, progress=True, **kwargs):
    from jittor.models import vgg16
    return _torchify(vgg16(pretrained=bool(weights is not None or pretrained),
                           **_drop_classifier_kwargs(kwargs)))


def _alexnet(pretrained=False, weights=None, progress=True, **kwargs):
    from jittor.models import alexnet
    return _torchify(alexnet(pretrained=bool(weights is not None or pretrained),
                             **_drop_classifier_kwargs(kwargs)))


def _squeezenet1_1(pretrained=False, weights=None, progress=True, **kwargs):
    from jittor.models import squeezenet1_1
    return _torchify(squeezenet1_1(pretrained=bool(weights is not None or pretrained),
                                   **_drop_classifier_kwargs(kwargs)))


def _resnet_builder(name):
    def build(pretrained=False, weights=None, progress=True, **kwargs):
        import jittor.models as _jtm
        return _torchify(getattr(_jtm, name)(
            pretrained=bool(weights is not None or pretrained),
            **_drop_classifier_kwargs(kwargs)))
    build.__name__ = name
    return build


_resnet18 = _resnet_builder("resnet18")
_resnet34 = _resnet_builder("resnet34")
_resnet50 = _resnet_builder("resnet50")
_resnet101 = _resnet_builder("resnet101")
_resnet152 = _resnet_builder("resnet152")


_MODEL_BUILDERS = {
    "resnet18": _resnet18, "resnet34": _resnet34, "resnet50": _resnet50,
    "resnet101": _resnet101, "resnet152": _resnet152,
    "vgg16": _vgg16, "alexnet": _alexnet, "squeezenet1_1": _squeezenet1_1,
}


def _get_model(name, weights=None, progress=True, **kwargs):
    builder = _MODEL_BUILDERS.get(name)
    if builder is None:
        raise ValueError(
            "torchvision.models.get_model(%r) is not implemented by the jittor "
            "shim; available: %s" % (name, ", ".join(sorted(_MODEL_BUILDERS))))
    return builder(weights=weights, progress=progress, **kwargs)


def _list_models(weights=None):
    return sorted(_MODEL_BUILDERS)


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
    resnet18 = staticmethod(_resnet18)
    resnet34 = staticmethod(_resnet34)
    resnet50 = staticmethod(_resnet50)
    resnet101 = staticmethod(_resnet101)
    resnet152 = staticmethod(_resnet152)
    get_model = staticmethod(_get_model)
    list_models = staticmethod(_list_models)


class _UtilsModule(types.ModuleType):
    make_grid = staticmethod(_make_grid)
    save_image = staticmethod(_save_image)


_CONCRETE = {"InterpolationMode": InterpolationMode, "ImageReadMode": ImageReadMode,
             "decode_image": (lambda *a, **k: None), "read_image": (lambda *a, **k: None),
             "encode_jpeg": (lambda *a, **k: None), "decode_jpeg": (lambda *a, **k: None),
             "vflip": _vflip, "hflip": _hflip, "rot90": _rot90, "rotate": _rotate,
             "pil_to_tensor": _pil_to_tensor, "to_tensor": _to_tensor,
             "resize": _resize, "normalize": _normalize, "pad": _pad, "crop": _crop,
             "center_crop": _center_crop, "resized_crop": _resized_crop,
             "get_image_size": _get_image_size, "get_dimensions": _get_dimensions,
             "rgb_to_grayscale": _rgb_to_grayscale,
             "grayscale_to_rgb": _grayscale_to_rgb,
             "make_grid": _make_grid, "save_image": _save_image,
             # real class-based transforms (torchvision.transforms.*)
             "Compose": Compose, "Normalize": Normalize, "ToTensor": ToTensor,
             "Resize": Resize, "ToPILImage": ToPILImage, "Lambda": Lambda,
             "CenterCrop": CenterCrop}


class _Passthrough:
    """Base for fabricated transform classes: identity unless implemented."""
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        return x

    def __repr__(self):
        return type(self).__name__ + "()"


def _random_hflip(img, p=0.5):
    import random
    if random.random() >= p:
        return img
    from PIL import Image as _Im
    if isinstance(img, _Im.Image):
        return img.transpose(_Im.FLIP_LEFT_RIGHT)
    return _hflip(img)


def _random_vflip(img, p=0.5):
    import random
    if random.random() >= p:
        return img
    from PIL import Image as _Im
    if isinstance(img, _Im.Image):
        return img.transpose(_Im.FLIP_TOP_BOTTOM)
    return _vflip(img)


class RandomHorizontalFlip(_Passthrough):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        return _random_hflip(img, self.p)


class RandomVerticalFlip(_Passthrough):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        return _random_vflip(img, self.p)


class RandomCrop(_Passthrough):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0,
                 padding_mode="constant"):
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        from PIL import Image as _Im, ImageOps as _ImOps
        import random
        if self.padding is not None:
            if isinstance(img, _Im.Image):
                pad = self.padding
                img = _ImOps.expand(img, pad, fill=self.fill)
            else:
                img = _pad(img, self.padding, fill=self.fill,
                           padding_mode=self.padding_mode)
        h, w = (img.size[1], img.size[0]) if isinstance(img, _Im.Image) else img.shape[-2:]
        th, tw = self.size
        if self.pad_if_needed and (h < th or w < tw):
            ph, pw = max(0, th - h), max(0, tw - w)
            if isinstance(img, _Im.Image):
                img = _ImOps.expand(img, (pw, ph))
            else:
                img = _pad(img, (pw, ph))
            h, w = (img.size[1], img.size[0]) if isinstance(img, _Im.Image) else img.shape[-2:]
        top = random.randint(0, max(0, h - th))
        left = random.randint(0, max(0, w - tw))
        return _crop(img, top, left, th, tw)


class RandomResizedCrop(_Passthrough):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0),
                 interpolation=InterpolationMode.BILINEAR, antialias=True):
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.antialias = antialias

    def __call__(self, img):
        import random, math
        from PIL import Image as _Im
        h, w = (img.size[1], img.size[0]) if isinstance(img, _Im.Image) else img.shape[-2:]
        area = h * w
        log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
        for _ in range(10):
            target = area * random.uniform(*self.scale)
            aspect = math.exp(random.uniform(*log_ratio))
            cw = int(round(math.sqrt(target * aspect)))
            ch = int(round(math.sqrt(target / aspect)))
            if 0 < cw <= w and 0 < ch <= h:
                left = random.randint(0, w - cw)
                top = random.randint(0, h - ch)
                break
        else:
            in_ratio = w / h
            if in_ratio < min(self.ratio):
                cw, ch = w, int(round(w / min(self.ratio)))
            elif in_ratio > max(self.ratio):
                cw, ch = int(round(h * max(self.ratio))), h
            else:
                cw, ch = w, h
            left = (w - cw) // 2
            top = (h - ch) // 2
        return _resized_crop(img, top, left, ch, cw, self.size,
                             self.interpolation, antialias=self.antialias)


class RandomRotation(_Passthrough):
    def __init__(self, degrees, interpolation=InterpolationMode.NEAREST,
                 expand=False, center=None, fill=0):
        if isinstance(degrees, (tuple, list)):
            self.degrees = degrees
        else:
            self.degrees = (-degrees, degrees)
        self.interpolation = interpolation
        self.expand = expand
        self.fill = fill

    def __call__(self, img):
        import random
        from PIL import Image as _Im
        angle = random.uniform(*self.degrees)
        if isinstance(img, _Im.Image):
            return img.rotate(angle, resample=_pil_resample(self.interpolation),
                              expand=self.expand)
        return _rotate(img, angle)


class Grayscale(_Passthrough):
    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels

    def __call__(self, img):
        return _rgb_to_grayscale(img, self.num_output_channels)


class RandomGrayscale(_Passthrough):
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, img):
        import random
        if random.random() >= self.p:
            return img
        return _rgb_to_grayscale(img, 3)


class ColorJitter(_Passthrough):
    """Brightness/contrast/saturation/hue jitter on PIL or float tensors."""
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.factors = (brightness, contrast, saturation, hue)

    def __call__(self, img):
        import random
        from PIL import Image as _Im, ImageEnhance as _Enh
        is_pil = isinstance(img, _Im.Image)
        if not is_pil:
            from PIL import Image as _Im2
            import numpy as _np
            arr = img.numpy()
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = _np.transpose(arr, (1, 2, 0))
            if arr.dtype != _np.uint8:
                arr = _np.clip(arr * 255.0, 0, 255).astype(_np.uint8)
            work = _Im2.fromarray(arr)
        else:
            work = img
        b, c, s, h = self.factors
        def _factor(value):
            if isinstance(value, (tuple, list)):
                return random.uniform(value[0], value[1])
            return random.uniform(max(0.0, 1 - value), 1 + value) if value else 1.0
        if b:
            work = _Enh.Brightness(work).enhance(_factor(b))
        if c:
            work = _Enh.Contrast(work).enhance(_factor(c))
        if s:
            work = _Enh.Color(work).enhance(_factor(s))
        if h:
            import numpy as _np
            arr = _np.asarray(work).astype(_np.float32) / 255.0
            import colorsys
            shift = _factor(h) - 1.0
            hsv = _np.empty_like(arr)
            flat = arr.reshape(-1, 3)
            out = _np.empty_like(flat)
            for i, (r, g, b) in enumerate(flat):
                hh, ss, vv = colorsys.rgb_to_hsv(float(r), float(g), float(b))
                hh = (hh + shift) % 1.0
                out[i] = colorsys.hsv_to_rgb(hh, ss, vv)
            hsv = out.reshape(arr.shape)
            work = _Im.fromarray((hsv * 255.0).clip(0, 255).astype(_np.uint8))
        if is_pil:
            return work
        import numpy as _np
        out = _np.asarray(work).astype(_np.float32) / 255.0
        return _jt.array(out.transpose(2, 0, 1).copy())


class PILToTensor:
    """torchvision.transforms.PILToTensor: PIL -> uint8 CHW tensor."""

    def __call__(self, pic):
        return _pil_to_tensor(pic)

    def __repr__(self):
        return "PILToTensor()"


class ConvertImageDtype:
    """torchvision.transforms.ConvertImageDtype for tensor backends."""

    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, tensor):
        return tensor.to(self.dtype) if hasattr(tensor, "to") else tensor

    def __repr__(self):
        return "ConvertImageDtype(%s)" % (self.dtype,)


_CONCRETE.update({
    "PILToTensor": PILToTensor,
    "ConvertImageDtype": ConvertImageDtype,
    "RandomHorizontalFlip": RandomHorizontalFlip,
    "RandomVerticalFlip": RandomVerticalFlip,
    "RandomCrop": RandomCrop,
    "RandomResizedCrop": RandomResizedCrop,
    "RandomRotation": RandomRotation,
    "Grayscale": Grayscale,
    "RandomGrayscale": RandomGrayscale,
    "ColorJitter": ColorJitter,
})


class DatasetFolder:
    """torchvision.datasets.DatasetFolder over a directory tree."""

    def __init__(self, root, loader, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None):
        import os
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.extensions = extensions
        classes = sorted(entry.name for entry in os.scandir(root) if entry.is_dir())
        if not classes:
            raise FileNotFoundError("Couldn't find any class folder in %s." % root)
        self.classes = classes
        self.class_to_idx = {name: index for index, name in enumerate(classes)}
        samples = []
        for name in classes:
            class_index = self.class_to_idx[name]
            directory = os.path.join(root, name)
            for entry in sorted(os.scandir(directory), key=lambda e: e.name):
                path = entry.path
                if is_valid_file is not None:
                    if not is_valid_file(path):
                        continue
                elif extensions is not None and os.path.splitext(path)[1].lower() not in extensions:
                    continue
                samples.append((path, class_index))
        if not samples:
            raise FileNotFoundError("Found 0 files in subfolders of: %s" % root)
        self.samples = samples
        self.imgs = samples
        self.targets = [target for _, target in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __repr__(self):
        return "%s(%d items)" % (type(self).__name__, len(self))


_IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


def _pil_loader(path):
    from PIL import Image as _Im
    with open(path, "rb") as handle:
        img = _Im.open(handle)
        return img.convert("RGB")


class ImageFolder(DatasetFolder):
    """Standard ImageFolder: class-per-directory, PIL images, RGB."""

    def __init__(self, root, transform=None, target_transform=None,
                 loader=_pil_loader, is_valid_file=None, allow_empty=False):
        super().__init__(root, loader, _IMG_EXTENSIONS, transform=transform,
                         target_transform=target_transform,
                         is_valid_file=is_valid_file)
        if not allow_empty and len(self.samples) == 0:
            raise FileNotFoundError("Found 0 files in subfolders of: %s" % root)


class FashionMNIST:
    """FashionMNIST IDX dataset with the three standard splits."""

    mirrors = ["http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"]
    resources = [
        ("train-images-idx3-ubyte.gz", "train-images-idx3-ubyte", 60000),
        ("train-labels-idx1-ubyte.gz", "train-labels-idx1-ubyte", 60000),
        ("t10k-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte", 10000),
        ("t10k-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte", 10000),
    ]
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        import os
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        os.makedirs(root, exist_ok=True)
        missing = [raw for _, raw, _ in self.resources
                   if not os.path.exists(os.path.join(root, raw))]
        if missing and download:
            self._download()
            missing = [raw for _, raw, _ in self.resources
                       if not os.path.exists(os.path.join(root, raw))]
        if missing:
            raise RuntimeError(
                "FashionMNIST data not found under %s (missing %s); pass "
                "download=True to fetch it" % (root, ", ".join(missing)))
        prefix = "train" if train else "t10k"
        self.data = self._read_images(os.path.join(root, prefix + "-images-idx3-ubyte"))
        _targets = self._read_labels(os.path.join(root, prefix + "-labels-idx1-ubyte"))
        import torch as _torch
        # torchvision returns `targets` as a long tensor, not a Python list.
        self.targets = _torch.as_tensor(_targets.astype("int64"))

    def _download(self):
        import os, urllib.request, gzip
        for filename, raw, _ in self.resources:
            target = os.path.join(self.root, raw)
            if os.path.exists(target):
                continue
            last = None
            for mirror in self.mirrors:
                try:
                    with urllib.request.urlopen(mirror + filename, timeout=120) as response:
                        payload = response.read()
                    with open(os.path.join(self.root, filename), "wb") as handle:
                        handle.write(payload)
                    with gzip.open(os.path.join(self.root, filename), "rb") as source:
                        with open(target, "wb") as sink:
                            sink.write(source.read())
                    last = None
                    break
                except Exception as exc:
                    last = exc
            if last is not None:
                raise RuntimeError("could not download %s: %s" % (filename, last))

    @staticmethod
    def _read_images(path):
        import numpy as _np
        with open(path, "rb") as handle:
            magic = int.from_bytes(handle.read(4), "big")
            if magic != 2051:
                raise ValueError("invalid FashionMNIST image file: %s" % path)
            count = int.from_bytes(handle.read(4), "big")
            rows = int.from_bytes(handle.read(4), "big")
            cols = int.from_bytes(handle.read(4), "big")
            payload = _np.frombuffer(handle.read(), dtype=_np.uint8)
        return payload.reshape(count, rows, cols)

    @staticmethod
    def _read_labels(path):
        import numpy as _np
        with open(path, "rb") as handle:
            magic = int.from_bytes(handle.read(4), "big")
            if magic != 2049:
                raise ValueError("invalid FashionMNIST label file: %s" % path)
            count = int.from_bytes(handle.read(4), "big")
            payload = _np.frombuffer(handle.read(), dtype=_np.uint8)
        return payload[:count]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        from PIL import Image as _Im
        image = _Im.fromarray(self.data[index], mode="L")
        target = int(self.targets[index])
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target


class _DatasetsModule(types.ModuleType):
    __path__ = []
    ImageFolder = ImageFolder
    DatasetFolder = DatasetFolder
    FashionMNIST = FashionMNIST

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        raise AttributeError(
            "torchvision.datasets.%s is not implemented by the jittor shim; "
            "implement it in compat/shim/resources/stubs/torchvision" % name)


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
        if name == "datasets":
            return _datasets
        if name in ("functional", "v2", "transforms", "F"):   # real submodules
            return _module_for_name(getattr(self, "__name__", "torchvision") + "." + name)
        return type(name, (_Passthrough,), {})


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
_datasets = _DatasetsModule("torchvision.datasets")
models = _models
utils = _utils
datasets = _datasets
transforms = _module_for_name("torchvision.transforms")
sys.modules.setdefault("torchvision.models", _models)
sys.modules.setdefault("torchvision.utils", _utils)
sys.modules.setdefault("torchvision.datasets", _datasets)
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
    if name == "datasets":
        return _datasets
    if name in ("transforms", "functional", "v2", "F"):
        return _module_for_name("torchvision." + name)
    raise AttributeError(name)
