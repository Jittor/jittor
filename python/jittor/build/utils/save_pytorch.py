"""Legacy archive writer names, supplied after ``import jittor``."""

from .runtime_services import module_compatibility as _module_compatibility

__getattr__, __dir__ = _module_compatibility("save_pytorch")
