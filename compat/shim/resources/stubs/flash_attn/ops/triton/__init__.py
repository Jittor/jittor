"""Triton ops subpackage for the flash_attn compatibility shim."""
from .rotary import apply_rotary

__all__ = ["apply_rotary"]
