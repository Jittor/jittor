"""Optional, source-pinned DeepSpeed eager-training integration."""
from .bootstrap import arm


def register(register_module_patch):
    from jittor.compat.transaction import active_transaction
    arm(transaction=active_transaction())


__all__ = ["arm", "register"]
