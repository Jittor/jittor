"""Stub torchaudio for jittor-as-torch text path (not functional)."""
import sys, types
__version__ = "2.11.0"
class _AnyModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"): raise AttributeError(name)
        return type(name, (), {})
def __getattr__(name):
    m=_AnyModule(f"torchaudio.{name}"); sys.modules[f"torchaudio.{name}"]=m; return m
