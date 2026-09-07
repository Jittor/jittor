# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers:
#     Haoyang Peng <2247838039@qq.com>
#     Dun Liang <randonlang@gmail.com>.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************


import types


class _Constraint:
    def __init__(self, *args, **kwargs):
        pass

    def check(self, value):
        import jittor as jt
        if isinstance(value, jt.Var):
            return jt.ones(value.shape, dtype="bool")
        return True

class _Real(_Constraint):
    pass

class _Interval(_Constraint):
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def check(self, value):
        return (value >= self.lower_bound) & (value <= self.upper_bound)

class _GreaterThan(_Constraint):
    def __init__(self, lower_bound):
        self.lower_bound = lower_bound

    def check(self, value):
        return value > self.lower_bound

class _GreaterThanEq(_Constraint):
    def __init__(self, lower_bound):
        self.lower_bound = lower_bound

    def check(self, value):
        return value >= self.lower_bound

class _LessThan(_Constraint):
    def __init__(self, upper_bound):
        self.upper_bound = upper_bound

    def check(self, value):
        return value < self.upper_bound

class _DependentProperty(property):
    def __init__(self, fn=None, is_discrete=False, event_dim=None):
        self.is_discrete = is_discrete
        self.event_dim = event_dim
        super().__init__(fn) if fn is not None else super().__init__()

    def __call__(self, fn):
        return type(self)(fn, self.is_discrete, self.event_dim)

def _dependent_property(fn=None, *, is_discrete=False, event_dim=None):
    prop = _DependentProperty(is_discrete=is_discrete, event_dim=event_dim)
    return prop(fn) if fn is not None else prop

class _ConstraintsModule(types.ModuleType):
    Constraint = _Constraint
    _Real = _Real
    real = _Real()
    positive = _GreaterThan(0)
    nonnegative = _GreaterThanEq(0)
    nonnegative_integer = _GreaterThanEq(0)
    positive_integer = _GreaterThan(0)
    unit_interval = _Interval(0, 1)
    simplex = _Constraint()
    lower_cholesky = _Constraint()
    positive_definite = _Constraint()
    boolean = _Constraint()
    real_vector = _Constraint()
    dependent = _Constraint()
    independent = _Constraint()
    dependent_property = staticmethod(_dependent_property)
    greater_than = staticmethod(lambda lower_bound: _GreaterThan(lower_bound))
    greater_than_eq = staticmethod(lambda lower_bound: _GreaterThanEq(lower_bound))
    less_than = staticmethod(lambda upper_bound: _LessThan(upper_bound))
    interval = staticmethod(lambda lower_bound, upper_bound: _Interval(lower_bound, upper_bound))
    half_open_interval = staticmethod(lambda lower_bound, upper_bound: _Interval(lower_bound, upper_bound))
    integer_interval = staticmethod(lambda lower_bound, upper_bound: _Interval(lower_bound, upper_bound))
    cat = staticmethod(lambda constraints, dim=0: _Constraint())
    stack = staticmethod(lambda constraints, dim=0: _Constraint())

constraints = _ConstraintsModule("torch.distributions.constraints")
