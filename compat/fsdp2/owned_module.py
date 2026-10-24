"""Installation-owned physical FSDP2 mixin, without native class mutation."""


def _new_unsharded_module(cls, *args, **kwargs):
    # Indexing/reconstruction through FSDP<Model> must create the original
    # Module, not an uninitialized FSDP instance without sharding state.
    original = cls.__mro__[2]
    result = original.__new__(original, *args, **kwargs)
    result.__init__(*args, **kwargs)
    return result


def owned_fsdp_module_type(context):
    current = context.state.get("owned_fsdp_module_type")
    if current is not None:
        return current
    from . import api

    # These are the actual implementations, not a virtual isinstance marker.
    # An implementation base would add a private class to the public MRO.
    # None of these methods closes over __class__ or uses zero-argument super.
    members = {name: getattr(api.FSDPModule, name) for name in api._FSDP_METHODS}
    members.update({
        "__module__": "torch.distributed.fsdp",
        "__doc__": api.FSDPModule.__doc__,
        "__new__": _new_unsharded_module,
    })
    current = type("FSDPModule", (), members)
    context.state["owned_fsdp_module_type"] = current
    context.state["nn_frontend_owner"].fsdp_module_type = current
    return current
