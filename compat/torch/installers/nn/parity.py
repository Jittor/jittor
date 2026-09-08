import abc
from ...fidelity import Fidelity, register_fidelity

def _functional_call(target, parameters, args=(), kwargs=None):
    return target(*args, **(kwargs or {}))

register_fidelity("torch.nn.utils.stateless.functional_call", _functional_call,
    Fidelity.UNIMPLEMENTED, "Calls the target with its existing parameters; supplied replacements are ignored")

def abc_base(name, *concrete):
    base = abc.ABCMeta(name, (object,), {})
    for item in concrete:
        if isinstance(item, type):
            base.register(item)
    return base



def install_parity(ctx):
    import abc
    import importlib
    g = ctx.jittor_module
    registry = ctx.registry
    nn = g.nn
    modules = registry.get("torch.nn.modules")


    conv = registry.ensure("torch.nn.modules.conv")
    conv._ConvNd = getattr(
        conv,
        "_ConvNd",
        abc_base(
            "_ConvNd",
            getattr(nn, "Conv", None),
            getattr(nn, "Conv1d", None),
            getattr(nn, "Conv2d", None),
            getattr(nn, "Conv3d", None),
        ),
    )
    conv._ConvTransposeNd = getattr(
        conv,
        "_ConvTransposeNd",
        abc_base(
            "_ConvTransposeNd",
            getattr(nn, "ConvTranspose", None),
            getattr(nn, "ConvTranspose1d", None),
            getattr(nn, "ConvTranspose2d", None),
            getattr(nn, "ConvTranspose3d", None),
        ),
    )
    conv._ConvTransposeMixin = conv._ConvTransposeNd
    for name in ("Conv1d", "Conv2d", "Conv3d", "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d"):
        value = getattr(nn, name, None)
        if value is not None:
            setattr(conv, name, value)
    modules.conv = conv

    if ctx.target_namespace is ctx.native_backend:
        pooling = importlib.import_module("jittor.nn.modules.pooling")
    else:
        # The NN frontend already owns this module and its layer adapters.
        # Publishing the native owner here would undo namespace isolation.
        pooling = nn.modules.pooling
    registry.publish("torch.nn.modules.pooling", pooling)
    pooling._MaxPoolNd = getattr(
        pooling,
        "_MaxPoolNd",
        abc_base(
            "_MaxPoolNd",
            getattr(nn, "Pool", None),
            getattr(nn, "MaxPool1d", None),
            getattr(nn, "MaxPool2d", None),
            getattr(nn, "MaxPool3d", None),
        ),
    )
    pooling._AvgPoolNd = getattr(
        pooling,
        "_AvgPoolNd",
        abc_base(
            "_AvgPoolNd",
            getattr(nn, "AvgPool1d", None),
            getattr(nn, "AvgPool2d", None),
            getattr(nn, "AvgPool3d", None),
        ),
    )
    pooling._AdaptiveAvgPoolNd = getattr(
        pooling,
        "_AdaptiveAvgPoolNd",
        abc_base(
            "_AdaptiveAvgPoolNd",
            getattr(nn, "AdaptiveAvgPool1d", None),
            getattr(nn, "AdaptiveAvgPool2d", None),
            getattr(nn, "AdaptiveAvgPool3d", None),
        ),
    )
    pooling._AdaptiveMaxPoolNd = getattr(
        pooling,
        "_AdaptiveMaxPoolNd",
        abc_base(
            "_AdaptiveMaxPoolNd",
            getattr(nn, "AdaptiveMaxPool1d", None),
            getattr(nn, "AdaptiveMaxPool2d", None),
            getattr(nn, "AdaptiveMaxPool3d", None),
        ),
    )
    modules.pooling = pooling

    instancenorm = registry.ensure("torch.nn.modules.instancenorm")
    instancenorm._InstanceNorm = getattr(
        instancenorm,
        "_InstanceNorm",
        abc_base(
            "_InstanceNorm",
            getattr(nn, "InstanceNorm", None),
            getattr(nn, "InstanceNorm1d", None),
            getattr(nn, "InstanceNorm2d", None),
            getattr(nn, "InstanceNorm3d", None),
        ),
    )
    modules.instancenorm = instancenorm

    stateless = registry.ensure("torch.nn.utils.stateless")
    stateless.functional_call = getattr(
        getattr(g, "func", None),
        "functional_call",
        _functional_call,
    )
    nn.utils.stateless = stateless
