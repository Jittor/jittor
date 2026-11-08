"""Compatibility import for the native generator owner.

The public type is owned by jittor_core; Python must not replace its
handle with an independent seed/counter implementation.
"""
from jittor_core import Generator

__all__ = ("Generator",)
