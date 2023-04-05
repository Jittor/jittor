import jittor as jt
from jittor import nn

def _weight_norm(v, g, dim):
    return v * (g / jt.norm(v, 2, dim, keepdim=True))

class WeightNorm(object):
    def __init__(self, name: str, dim: int) -> None:
        if dim is None:
            dim = -1
        self.name = name
        self.dim = dim

    # TODO Make return type more specific
    def compute_weight(self, module: nn.Module):
        g = getattr(module, self.name + '_g')
        v = getattr(module, self.name + '_v')
        return _weight_norm(v, g, self.dim)

    @staticmethod
    def apply(module, name: str, dim: int):
        if hasattr(module, '__fhook2__') and isinstance(module.__fhook2__, WeightNorm):
            raise RuntimeError("Cannot register two weight_norm hooks on "
                                "the same parameter {}".format(name))

        if dim is None:
            dim = -1

        fn = WeightNorm(name, dim)

        weight = getattr(module, name)
        # todo: add check
        # remove w from parameter list
        # del module._parameters[name]
        delattr(module, name)

        # add g and v as new parameters and express w as g/||v|| * v
        module.__setattr__(name + '_g', jt.norm(weight, 2, dim, keepdim=True).detach())
        module.__setattr__(name + '_v', weight.detach())
        setattr(module, name, fn.compute_weight(module))

        # recompute weight before every forward()
        # todo: support multiple hook in a module
        module.register_pre_forward_hook(fn)
        return fn

    def remove(self, module: nn.Module) -> None:
        weight = self.compute_weight(module)
        delattr(module, self.name)
        delattr(module, self.name + '_g')
        delattr(module, self.name + '_v')
        setattr(module, self.name, weight.detach())

    def __call__(self, module: nn.Module, inputs) -> None:
        setattr(module, self.name, self.compute_weight(module))

def weight_norm(module, name, dim):
    ''' Add a module weight normalization.

    :param module: input model.
    :param name: name of the assigned parameter.
    :param dim: which dim to carry out weightnorm.

    Example::

    class jt_module(jt.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.linear = jt.array(weight)

        def execute(self, x):
            return jt.matmul(self.linear, x)
    
    jm = jt_module(weight)
    weight_norm(jm, 'linear', -1)
    
    '''
    WeightNorm.apply(module, name, dim)
    return module

def remove_weight_norm(module, name: str = 'weight'):
    if hasattr(module, "__fhook2__") and isinstance(module.__fhook2__, WeightNorm):
        delattr(module, "__fhook2__")
        return module
    raise ValueError("weight_norm of '{}' not found in {}"
                     .format(name, module))