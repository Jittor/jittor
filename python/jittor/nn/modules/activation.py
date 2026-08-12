"""Stateful activation modules exposed through :mod:`jittor.nn`."""

import jittor as jt


class RReLU(jt.Module):
    """Apply randomized leaky ReLU with a training-dependent slope."""

    def __init__(self, lower=1.0 / 8, upper=1.0 / 3):
        self.lower = lower
        self.upper = upper
        self.is_train = True

    def execute(self, x):
        return jt.nn.rrelu(x, self.lower, self.upper, getattr(self, "is_train", True))


class Hardswish(jt.Module):
    def execute(self, x):
        return jt.nn.hardswish(x)


class Hardsigmoid(jt.Module):
    def execute(self, x):
        return jt.nn.hardsigmoid(x)


class ELU(jt.Module):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def execute(self, x):
        return jt.nn.elu(x, self.alpha)


class PReLU(jt.Module):
    def __init__(self, num_parameters=1, init_=0.25):
        self.num_parameters = num_parameters
        self.weight = jt.init.constant((num_parameters,), "float32", init_)

    def execute(self, x):
        return jt.nn.prelu(x, self.weight)


class GLU(jt.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def execute(self, x):
        return jt.nn.glu(x, self.dim)


class Softsign(jt.Module):
    def __init__(self):
        super().__init__()

    def execute(self, x):
        return jt.nn.softsign(x)


class Tanh(jt.Module):
    def __init__(self):
        super().__init__()

    def execute(self, x):
        return x.tanh()


class Sigmoid(jt.Module):
    def __init__(self):
        super().__init__()

    def execute(self, x):
        return x.sigmoid()


class Softplus(jt.Module):
    def __init__(self, beta=1, threshold=20):
        self.beta = beta
        self.threshold = threshold

    def execute(self, x):
        return jt.nn.softplus(x, self.beta, self.threshold)


class Mish(jt.Module):
    def __init__(self, inplace=False):
        pass

    def execute(self, x):
        return jt.nn.mish(x)


class _FunctionModule(jt.Module):
    _function_name = None

    def __init__(self, *args, **kw):
        self.args = args
        self.kw = kw

    def execute(self, *args):
        function = getattr(jt.nn, self._function_name)
        return function(*args, *self.args, **self.kw)

    def __str__(self):
        return "{}({})".format(self._function_name, self.extra_repr())

    def extra_repr(self):
        return ",".join(map(str, self.args))


class ReLU(_FunctionModule):
    _function_name = "relu"


Relu = ReLU


class LeakyReLU(_FunctionModule):
    _function_name = "leaky_relu"


Leaky_relu = LeakyReLU


class ReLU6(_FunctionModule):
    _function_name = "relu6"


class Softmax(_FunctionModule):
    _function_name = "softmax"


class GELU(_FunctionModule):
    _function_name = "gelu"


class SiLU(_FunctionModule):
    _function_name = "silu"


__all__ = [
    "ELU",
    "GELU",
    "GLU",
    "Hardsigmoid",
    "Hardswish",
    "LeakyReLU",
    "Leaky_relu",
    "Mish",
    "PReLU",
    "RReLU",
    "ReLU",
    "ReLU6",
    "Relu",
    "SiLU",
    "Sigmoid",
    "Softmax",
    "Softplus",
    "Softsign",
    "Tanh",
]
