"""Stub torchaudio for jittor-as-torch text path (not functional).

Import-only: the shim ships no audio DSP, so every attribute resolves to an
empty class that raises when it is called. Two spellings of "get at a
torchaudio name" have to work, because downstream uses both:

* ``from torchaudio import functional`` -- answered by ``__getattr__`` below;
* ``from torchaudio.functional import melscale_fbanks`` -- this one imports the
  *submodule* first. vLLM-Omni's ``vllm_omni/utils/audio.py`` does it at module
  level, and a module with no ``__path__`` offers the import system no submodule
  to find, so it failed with ``No module named 'torchaudio.functional'`` before
  it reached the name lookup that ``__getattr__`` would have answered.

``_AnyFinder`` fabricates an empty module for anything under ``torchaudio.``
rather than enumerating a real torchaudio's submodules, which would go stale.
Its attributes are the same loudly-failing classes as the top-level ones: a
call site that wants real audio DSP still raises instead of computing something
plausible and wrong.
"""
import importlib.abc
import importlib.machinery
import sys
import types

__version__ = "2.11.0"


class _AnyModule(types.ModuleType):
    # Every fabricated module is a package, so a deeper name under it reaches
    # this module's `__getattr__` (or `_AnyFinder`) instead of dying with
    # "'torchaudio.functional' is not a package".
    __path__ = []

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (), {})


class _AnyFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Answer any ``torchaudio.<...>`` import with an empty module."""

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("torchaudio."):
            return None
        return importlib.machinery.ModuleSpec(fullname, self, is_package=True)

    def create_module(self, spec):
        return _AnyModule(spec.name)

    def exec_module(self, module):
        pass


def __getattr__(name):
    # An already-imported submodule wins over a fresh dummy, so that
    # ``from torchaudio import functional`` and ``from torchaudio.functional
    # import x`` hand back the same module instead of one object per access.
    existing = sys.modules.get("%s.%s" % (__name__, name))
    if existing is not None:
        return existing
    m = _AnyModule(f"{__name__}.{name}")
    sys.modules[m.__name__] = m
    return m


if not any(isinstance(finder, _AnyFinder) for finder in sys.meta_path):
    sys.meta_path.append(_AnyFinder())
