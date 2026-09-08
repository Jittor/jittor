"""CUDA facade owner: stable implementations plus installation bindings."""

from .bindings import install, _install_cuda, _install_version, _install_accelerator
from .api import _TF32_FLAGS, _TF32_FALLBACK

__all__ = ["install"]
