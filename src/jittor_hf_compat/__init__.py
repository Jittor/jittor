"""Transformers version-compatibility registration for Jittor."""

from __future__ import annotations


def register_patches(register):
    from .patches import register_patches as _register_patches

    return _register_patches(register)


def install():
    from .patches import install as _install

    return _install()


__all__ = ["install", "register_patches"]
