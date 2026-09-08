"""CUDA facade owner: stable implementations plus installation bindings."""

from .bindings import install, _install_cuda, _install_version, _install_accelerator

__all__ = ["install"]
