"""Canonical parameter names shared by the jittor and torch benchmark models.

The two model files in ``bench/`` describe the same networks but name their
submodules differently, so a weight transfer needs one agreed spelling. Neither
function imports a framework: both walk attributes and read ``type(...).__name__``,
so this module can be imported from either side.
"""


def _linear(out, prefix, module):
    out[prefix + ".weight"] = module.weight
    out[prefix + ".bias"] = module.bias


def jittor_parameters(name, model):
    """Canonical name -> Var, for a model built by ``bench_jittor.build``."""
    out = {}
    if name == "mlp":
        for index, layer in enumerate([model.l1, model.l2, model.l3, model.l4], 1):
            _linear(out, "l%d" % index, layer)
    elif name == "cnn":
        for index, conv in enumerate(model.convs):
            _linear(out, "conv%d" % index, conv)
        _linear(out, "h1", model.h1)
        _linear(out, "h2", model.h2)
    elif name == "transformer":
        out["embed.weight"] = model.embed.weight
        for index, block in enumerate(model.blocks):
            for tag in ("n1", "n2", "qkv", "proj", "f1", "f2"):
                _linear(out, "blocks.%d.%s" % (index, tag), getattr(block, tag))
        _linear(out, "norm", model.norm)
        _linear(out, "head", model.head)
    else:
        raise SystemExit("unknown model " + name)
    return out


def torch_parameters(name, model):
    """Canonical name -> Parameter, for a model built by ``bench_torch.build``."""
    out = {}
    if name == "mlp":
        # nn.Sequential(Linear, ReLU, Linear, ReLU, Linear, ReLU, Linear)
        for index, position in enumerate([0, 2, 4, 6], 1):
            _linear(out, "l%d" % index, model.net[position])
    elif name == "cnn":
        # `features` interleaves ReLU and MaxPool between the convolutions, so
        # pick the convolutions by type rather than by a hand-counted index.
        convs = [m for m in model.features if type(m).__name__ == "Conv2d"]
        if len(convs) != 6:
            raise SystemExit("expected 6 convolutions, found %d" % len(convs))
        for index, conv in enumerate(convs):
            _linear(out, "conv%d" % index, conv)
        _linear(out, "h1", model.head[0])
        _linear(out, "h2", model.head[2])
    elif name == "transformer":
        out["embed.weight"] = model.embed.weight
        for index, block in enumerate(model.blocks):
            for tag in ("n1", "n2", "qkv", "proj", "f1", "f2"):
                _linear(out, "blocks.%d.%s" % (index, tag), getattr(block, tag))
        _linear(out, "norm", model.norm)
        _linear(out, "head", model.head)
    else:
        raise SystemExit("unknown model " + name)
    return out


def check_complete(name, params, trainable):
    """Every trainable parameter must be named, or the transfer is partial."""
    named = {id(value) for value in params.values()}
    missing = [p for p in trainable if id(p) not in named]
    if missing:
        raise SystemExit("%s: %d parameters are outside the canonical map"
                         % (name, len(missing)))
