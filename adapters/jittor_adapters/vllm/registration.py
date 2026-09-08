"""Registrar for the public jittor.module_patches entry-point contract."""


def register(register_module_patch):
    """Arm before-import setup and after-import patches; return no registrar."""
    from jittor.compat.transaction import active_transaction
    from .bootstrap import arm
    arm(transaction=active_transaction(), register_callback=register_module_patch)
