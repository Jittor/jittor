"""Minimal torchvision facade for jittor-as-torch.

The module still fabricates unknown torchvision submodules on demand so import
probes succeed, but common image transforms, image saving and classification
models are backed by real Jittor implementations.
"""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name
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
def _frontend_tensor(value):
    # Return the torch-facing Tensor type when jittor-torch-shim is deployed.
    # A bare native Var shares the storage but is not an instance of
    # torch.Tensor, and datasets' torch formatting (and callers like
    # Dataset.set_transform) expect the facade type.
    try:
        import torch as _torch
        if hasattr(_torch, "_torch_compat_install_context"):
            return _torch.tensor(value)
    except Exception:
        pass
    return value


def _pil_to_tensor(pic):
    # PIL image -> uint8 CHW Var (torchvision.pil_to_tensor semantics)
    import numpy as _np
    arr = _np.asarray(pic)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    t = _jt.array(arr.copy())           # HWC
    return _frontend_tensor(t.permute(2, 0, 1))   # CHW

def _to_tensor(pic):
    # torchvision.transforms.functional.to_tensor semantics:
    # PIL/uint8 ndarray -> float CHW in [0,1].
    import numpy as _np
    from PIL import Image as _Im
    if isinstance(pic, _jt.Var):
        t = pic.float32() / 255.0 if "uint8" in _jittor_dtype_name(pic.dtype) else pic.float32()
        return _frontend_tensor(t)
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
    return _frontend_tensor(_jt.array(arr.copy()))

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
    return img[..., top:top + height, left:left + width]

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


class RandomResizedCrop:
    """torchvision.transforms.RandomResizedCrop for PIL images and Vars."""

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0),
                 interpolation=InterpolationMode.BILINEAR, antialias=True, **kw):
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        self.scale = tuple(scale)
        self.ratio = tuple(ratio)
        self.interpolation = interpolation
        self.antialias = antialias

    def _crop_box(self, h, w):
        import random
        area = h * w
        for _ in range(10):
            target = random.uniform(*self.scale) * area
            aspect = random.uniform(*self.ratio)
            cw = int(round((target * aspect) ** 0.5))
            ch = int(round((target / aspect) ** 0.5))
            if 0 < cw <= w and 0 < ch <= h:
                return random.randint(0, h - ch), random.randint(0, w - cw), ch, cw
        side = min(h, w)
        return (h - side) // 2, (w - side) // 2, side, side

    def __call__(self, pic):
        from PIL import Image as _Im
        if isinstance(pic, _Im.Image):
            i, j, ch, cw = self._crop_box(pic.height, pic.width)
            cropped = pic.crop((j, i, j + cw, i + ch))
            return cropped.resize((self.size[1], self.size[0]), _pil_resample(self.interpolation))
        i, j, ch, cw = self._crop_box(pic.shape[-2], pic.shape[-1])
        return Resize(self.size, self.interpolation, antialias=self.antialias)(pic[..., i:i + ch, j:j + cw])

    def __repr__(self):
        return f"RandomResizedCrop(size={self.size}, scale={self.scale}, ratio={self.ratio})"


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


class ColorJitter:
    """torchvision.transforms.ColorJitter on PIL images and jittor Vars.

    Scalar parameters expand to torchvision's random factor ranges; a tuple is
    sampled directly.  PIL images use ImageEnhance (hue as an HSV H-channel
    shift); tensors are adjusted in float RGB space with the torchvision YIQ
    hue rotation.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(hue, "hue")

    @staticmethod
    def _check_input(value, name):
        if isinstance(value, bool):
            raise TypeError(f"{name} must be a number or a sequence of two numbers")
        if isinstance(value, (int, float)):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            if name == "hue":
                if value > 0.5:
                    raise ValueError("hue values must be in [-0.5, 0.5]")
                bounds = (-float(value), float(value))
            else:
                bounds = (max(0.0, 1.0 - float(value)), 1.0 + float(value))
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            bounds = (float(value[0]), float(value[1]))
            if bounds[0] < 0 or bounds[1] < 0:
                raise ValueError(f"{name} values must be non-negative")
            if bounds[0] > bounds[1]:
                raise ValueError(f"{name} values must be in increasing order")
            if name == "hue" and (bounds[0] < -0.5 or bounds[1] > 0.5):
                raise ValueError("hue values must be in [-0.5, 0.5]")
        else:
            raise TypeError(f"{name} must be a number or a sequence of two numbers")
        center = 0.0 if name == "hue" else 1.0
        return None if bounds[0] == bounds[1] == center else bounds

    def __call__(self, img):
        import random
        import numpy as _np
        from PIL import Image as _Im

        operations = []
        for name in ("brightness", "contrast", "saturation", "hue"):
            bounds = getattr(self, name)
            if bounds is not None:
                operations.append((name, random.uniform(*bounds)))
        random.shuffle(operations)
        if not operations:
            return img  # identity, as in torchvision
        if isinstance(img, _Im.Image):
            from PIL import ImageEnhance
            for name, factor in operations:
                if name == "brightness":
                    img = ImageEnhance.Brightness(img).enhance(factor)
                elif name == "contrast":
                    img = ImageEnhance.Contrast(img).enhance(factor)
                elif name == "saturation":
                    img = ImageEnhance.Color(img).enhance(factor)
                else:
                    mode = img.mode
                    bands = list(img.convert("HSV").split())
                    shift = int(round(factor * 255))
                    bands[0] = bands[0].point(lambda pixel: (pixel + shift) % 256)
                    img = _Im.merge("HSV", bands).convert(mode)
            return img
        t = img if isinstance(img, _jt.Var) else _jt.array(_np.asarray(img))
        t = t.float32()
        squeezed = []
        while t.ndim < 4:
            t = t.unsqueeze(0)
            squeezed.append(0)
        for name, factor in operations:
            if name == "brightness":
                t = t * float(factor)
            elif name == "contrast":
                mean = t.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
                t = (t - mean) * float(factor) + mean
            elif name == "saturation":
                gray = t.mean(dim=1, keepdim=True)
                t = gray + (t - gray) * float(factor)
            else:
                t = self._adjust_hue(t, float(factor))
        for _ in squeezed:
            t = t.squeeze(0)
        return t

    @staticmethod
    def _adjust_hue(t, factor):
        import math
        r, g, b = t[:, 0:1], t[:, 1:2], t[:, 2:3]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        i = 0.596 * r - 0.274 * g - 0.322 * b
        q = 0.211 * r - 0.523 * g + 0.312 * b
        angle = factor * 2.0 * math.pi
        cosine, sine = math.cos(angle), math.sin(angle)
        i_rot = i * cosine - q * sine
        q_rot = i * sine + q * cosine
        r_new = y + 0.956 * i_rot + 0.621 * q_rot
        g_new = y - 0.272 * i_rot - 0.647 * q_rot
        b_new = y - 1.106 * i_rot + 1.703 * q_rot
        return _jt.stack([r_new, g_new, b_new], dim=1)

    def __repr__(self):
        return (f"ColorJitter(brightness={self.brightness}, "
                f"contrast={self.contrast}, saturation={self.saturation}, "
                f"hue={self.hue})")


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


def _box_convert(boxes, in_fmt="xyxy", out_fmt="xyxy"):
    """torchvision.ops.box_convert on jittor Vars."""
    import numpy as _np
    t = boxes if isinstance(boxes, _jt.Var) else _jt.array(_np.asarray(boxes, dtype=_np.float32))
    if t.ndim == 1:
        t = t.unsqueeze(0)
    t = t.float32()
    a, b, c, d = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
    src, dst = str(in_fmt).lower(), str(out_fmt).lower()
    if src == "xyxy":
        x1, y1, x2, y2 = a, b, c, d
    elif src == "xywh":
        x1, y1, x2, y2 = a, b, a + c, b + d
    elif src == "cxcywh":
        x1, y1, x2, y2 = a - c / 2, b - d / 2, a + c / 2, b + d / 2
    else:
        raise ValueError(f"Unsupported box format: {in_fmt}")
    if dst == "xyxy":
        return _jt.stack([x1, y1, x2, y2], dim=1)
    if dst == "xywh":
        return _jt.stack([x1, y1, x2 - x1, y2 - y1], dim=1)
    if dst == "cxcywh":
        return _jt.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=1)
    raise ValueError(f"Unsupported box format: {out_fmt}")


def _draw_bounding_boxes(image, boxes, labels=None, colors=None, fill=False,
                         width=1, font=None, font_size=None):
    """torchvision.utils.draw_bounding_boxes on a uint8 CHW tensor."""
    import numpy as _np
    from PIL import Image as _Im, ImageDraw as _ImageDraw
    arr = image.numpy() if isinstance(image, _jt.Var) else _np.asarray(image)
    if arr.ndim == 2:
        arr = _np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = _np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = _np.repeat(arr, 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        arr = arr[:, :, :3]
    if arr.dtype != _np.uint8:
        arr = _np.clip(arr, 0, 255).astype(_np.uint8)
    pil = _Im.fromarray(arr)
    draw = _ImageDraw.Draw(pil)
    b = boxes.numpy() if isinstance(boxes, _jt.Var) else _np.asarray(boxes)
    color_list = list(colors) if isinstance(colors, (list, tuple)) else None
    for index, box in enumerate(_np.asarray(b)):
        x1, y1, x2, y2 = (float(v) for v in box[:4])
        color = color_list[index % len(color_list)] if color_list else (colors or "red")
        draw.rectangle([x1, y1, x2, y2], outline=color, width=int(width),
                       fill=color if fill else None)
        if labels is not None and index < len(labels):
            draw.text((x1 + 2, y1 + 2), str(labels[index]), fill=color)
    out = _np.transpose(_np.asarray(pil), (2, 0, 1)).astype(_np.uint8)
    return _frontend_tensor(_jt.array(out))


class _OpsModule(types.ModuleType):
    box_convert = staticmethod(_box_convert)


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
    draw_bounding_boxes = staticmethod(_draw_bounding_boxes)


_CONCRETE = {"InterpolationMode": InterpolationMode, "ImageReadMode": ImageReadMode,
             "decode_image": (lambda *a, **k: None), "read_image": (lambda *a, **k: None),
             "encode_jpeg": (lambda *a, **k: None), "decode_jpeg": (lambda *a, **k: None),
             "vflip": _vflip, "hflip": _hflip, "rot90": _rot90, "rotate": _rotate,
             "pil_to_tensor": _pil_to_tensor, "to_tensor": _to_tensor,
             "resize": _resize, "normalize": _normalize, "pad": _pad, "crop": _crop,
             "center_crop": _center_crop, "rgb_to_grayscale": _rgb_to_grayscale,
             "grayscale_to_rgb": _grayscale_to_rgb,
             "make_grid": _make_grid, "save_image": _save_image,
             "draw_bounding_boxes": _draw_bounding_boxes,
             # real class-based transforms (torchvision.transforms.*)
             "Compose": Compose, "Normalize": Normalize, "ToTensor": ToTensor,
             "Resize": Resize, "ToPILImage": ToPILImage, "Lambda": Lambda,
             "CenterCrop": CenterCrop, "ColorJitter": ColorJitter,
             "RandomResizedCrop": RandomResizedCrop}


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
_ops = _OpsModule("torchvision.ops")
models = _models
utils = _utils
transforms = _module_for_name("torchvision.transforms")
sys.modules.setdefault("torchvision.models", _models)
sys.modules.setdefault("torchvision.utils", _utils)
sys.modules.setdefault("torchvision.ops", _ops)
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
    if name == "ops":
        return _ops
    if name in ("transforms", "functional", "v2", "F"):
        return _module_for_name("torchvision." + name)
    raise AttributeError(name)
