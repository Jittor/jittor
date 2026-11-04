"""Shared physical device checks for Torch compatibility tests."""
import torch


def require_residency(value, device):
    assert value.device.type == device
    if hasattr(torch, '_torch_compat_install_context') or getattr(torch, '__name__', None) == 'jittor':
        import jittor as jt
        value.sync()
        assert value.location() == ('device' if device == 'cuda' else 'cpu')
        assert jt.core.dispatch_context([value])[0] == device


