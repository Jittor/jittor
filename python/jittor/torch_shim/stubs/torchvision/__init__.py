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
_CONCRETE = {"InterpolationMode": InterpolationMode, "ImageReadMode": ImageReadMode,
             "decode_image": (lambda *a, **k: None), "read_image": (lambda *a, **k: None),
             "encode_jpeg": (lambda *a, **k: None), "decode_jpeg": (lambda *a, **k: None)}


class _AnyModule(types.ModuleType):
    __path__ = []   # makes it a package so submodule imports descend
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        if name in _CONCRETE:
            return _CONCRETE[name]
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
