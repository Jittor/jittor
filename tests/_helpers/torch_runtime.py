"""Delay PyTorch imports until pytest starts executing a test module."""

import importlib
import importlib.util


def modules_available(*module_names):
    """Return whether top-level optional dependencies are discoverable without importing."""
    top_level_names = {module_name.partition(".")[0] for module_name in module_names}
    for module_name in top_level_names:
        try:
            if importlib.util.find_spec(module_name) is None:
                return False
        except (ImportError, ValueError):
            return False
    return True


def import_torch_modules(*module_names):
    """Apply Jittor's runtime workaround and import optional Torch modules."""
    import jittor as jt

    jt.dirty_fix_pytorch_runtime_error()
    return tuple(importlib.import_module(name) for name in module_names)
