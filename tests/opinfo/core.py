# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""``OpInfo`` and friends -- declarative per-operator test metadata.

A faithful, jittor-flavored port of ``torch.testing._internal.opinfo.core``. An
:class:`OpInfo` bundles everything the generic test templates need to test one
operator: how to build sample inputs, an independent numpy ``ref`` for the forward,
the dtypes it supports, and whether/how to gradcheck it. ``test_ops.py`` iterates
the registry (``op_db``) and, for every op × dtype × device, generates the same
battery PyTorch runs -- forward-vs-reference, variant consistency, and gradcheck.

Why declarative: the audit found the same forward/backward/device scaffolding
copy-pasted across ~150 files with divergent tolerances. Collapsing each op to one
metadata row means adding an op (or a missing dtype/backward) is a few lines, and
every op automatically gets the full battery -- closing the "forward-only" holes by
construction rather than by remembering to write a backward test.
"""
from _helpers import common as cu


# ----------------------------------------------------------------- sample inputs

class SampleInput:
    """One concrete call to an operator: ``op(input, *args, **kwargs)``.

    ``input`` is the primary (and usually differentiated) operand, a jittor Var.
    ``args`` may hold further Vars (also differentiated if floating) or plain
    python values; ``kwargs`` are non-tensor options (dims, eps, reduction...).
    Mirrors torch's SampleInput so sample-input functions read the same.
    """

    __slots__ = ["input", "args", "kwargs", "name", "output_process_fn_grad"]

    def __init__(self, input, *var_args, args=None, kwargs=None, name="",
                 output_process_fn_grad=None, **var_kwargs):
        self.input = input
        if args is not None or kwargs is not None:
            assert not (var_args or var_kwargs), \
                "use SampleInput(input, *args, **kwargs) OR args=/kwargs=, not both"
            self.args = tuple(args) if args is not None else ()
            self.kwargs = dict(kwargs) if kwargs is not None else {}
        else:
            self.args = var_args
            self.kwargs = var_kwargs
        self.name = name
        self.output_process_fn_grad = output_process_fn_grad or (lambda x: x)

    def __repr__(self):
        return f"SampleInput(input={_shape_of(self.input)}, args={self.args}, kwargs={self.kwargs})"


def _shape_of(x):
    return tuple(x.shape) if hasattr(x, "shape") else x


# ---------------------------------------------------------------- decorate / skip

class DecorateInfo:
    """Attach a decorator (or skip/xfail) to specific generated tests.

    Mirrors torch's DecorateInfo: apply ``decorator`` only to tests whose name is in
    ``test_name`` (or all), on ``device_type``/``dtypes`` when given. Used to encode
    the audit's must-preserve ``@skip``-locked semantic divergences verbatim.
    """

    def __init__(self, decorator=None, test_name=None, *, device_type=None,
                 dtypes=None, active_if=True):
        self.decorator = decorator
        self.test_name = test_name
        self.device_type = device_type
        self.dtypes = dtypes
        self.active_if = active_if

    def is_active(self, test_name, device_type, dtype):
        if not self.active_if:
            return False
        if self.test_name is not None and self.test_name != test_name:
            return False
        if self.device_type is not None and self.device_type != device_type:
            return False
        if self.dtypes is not None and dtype not in self.dtypes:
            return False
        return True


def skip(test_name=None, *, device_type=None, dtypes=None, reason="skipped"):
    """A DecorateInfo that skips matching tests (records the reason verbatim)."""
    import unittest
    return DecorateInfo(unittest.skip(reason), test_name,
                        device_type=device_type, dtypes=dtypes)


def xfail(test_name=None, *, device_type=None, dtypes=None, reason="expected failure"):
    """A DecorateInfo that marks matching tests as expected-to-fail."""
    import unittest
    return DecorateInfo(unittest.expectedFailure, test_name,
                        device_type=device_type, dtypes=dtypes)


# --------------------------------------------------------------------- OpInfo

class OpInfo:
    """Metadata for one operator under test.

    Args (the subset jittor uses; names mirror torch):
      name:                 op slug; also resolves ``op`` from ``jittor`` if not given.
      op:                   the callable under test (default: ``getattr(jt, name)``).
      ref:                  numpy reference forward, ``ref(*np_inputs, **kwargs)``. The
                            INDEPENDENT oracle that makes a green forward meaningful.
      sample_inputs_func:   ``f(opinfo, device, dtype, requires_grad) -> [SampleInput]``.
      dtypes / dtypesIfCUDA: supported dtype sets (``cu.floating_types()`` etc.).
      supports_autograd:    whether to run gradcheck (default True).
      supports_gradgrad:    whether to run gradgradcheck (default = supports_autograd).
      gradcheck_nondet_tol: extra absolute slack for nondeterministic ops.
      gradcheck_wrapper:    optional ``f(op, *inputs, **kwargs)`` if the op needs
                            massaging before differentiation.
      decorators / skips:   sequences of DecorateInfo applied to generated tests.
      variant_test_name:    disambiguating suffix when one op has several OpInfos.
    """

    def __init__(self, name, *, op=None, ref=None, sample_inputs_func=None,
                 dtypes=None, dtypesIfCUDA=None,
                 supports_autograd=True, supports_gradgrad=None,
                 gradcheck_nondet_tol=0.0, gradcheck_wrapper=None,
                 reference_tol=None,
                 assert_autodiffed=False, decorators=(), skips=(),
                 variant_test_name="", op_name=None):
        self.name = name
        self.variant_test_name = variant_test_name
        self.op_name = op_name or name
        self._op = op
        self.ref = ref
        # optional (atol, rtol) override for the forward-vs-reference check; ops that
        # accumulate many float32 terms (conv) legitimately need a looser tolerance than
        # the per-dtype default. None -> use the default dtype tolerance.
        self.reference_tol = reference_tol
        self.sample_inputs_func = sample_inputs_func
        self.dtypes = tuple(dtypes) if dtypes is not None else cu.floating_types()
        self.dtypesIfCUDA = tuple(dtypesIfCUDA) if dtypesIfCUDA is not None else self.dtypes
        self.supports_autograd = supports_autograd
        self.supports_gradgrad = (supports_autograd if supports_gradgrad is None
                                  else supports_gradgrad)
        self.gradcheck_nondet_tol = gradcheck_nondet_tol
        self.gradcheck_wrapper = gradcheck_wrapper
        self.assert_autodiffed = assert_autodiffed
        self.decorators = tuple(decorators) + tuple(skips)

    # -- the callable under test -------------------------------------------
    @property
    def op(self):
        if self._op is not None:
            return self._op
        import jittor as jt
        fn = getattr(jt, self.op_name, None)
        if fn is None:
            fn = getattr(jt.nn, self.op_name, None)
        if fn is None and hasattr(jt.nn, "functional"):
            fn = getattr(jt.nn.functional, self.op_name, None)
        if fn is None:
            raise AttributeError(f"OpInfo '{self.name}': cannot resolve op '{self.op_name}'")
        return fn

    def __call__(self, *args, **kwargs):
        return self.op(*args, **kwargs)

    # -- sample inputs ------------------------------------------------------
    def supported_dtypes(self, device):
        return self.dtypesIfCUDA if device in ("cuda", "npu") else self.dtypes

    def supports_dtype(self, dtype, device):
        return dtype in self.supported_dtypes(device)

    def sample_inputs(self, device, dtype, requires_grad=False):
        assert self.sample_inputs_func is not None, \
            f"OpInfo '{self.name}' has no sample_inputs_func"
        out = self.sample_inputs_func(self, device, dtype, requires_grad)
        return list(out)

    @property
    def full_name(self):
        return self.name + (("_" + self.variant_test_name) if self.variant_test_name else "")

    def __repr__(self):
        return f"OpInfo({self.full_name})"


# ----------------------------------------------------- specialized OpInfo classes

class UnaryUfuncInfo(OpInfo):
    """Elementwise unary op (exp, log, sin, relu, ...).

    Provides a default elementwise ``sample_inputs_func`` over a few shapes; the
    numpy ``ref`` is applied elementwise. Mirrors torch's UnaryUfuncInfo, which is
    the subclass torch lets actually test forward values (not just gradients).
    """

    def __init__(self, name, *, ref=None, domain=(None, None), **kwargs):
        self.domain = domain
        kwargs.setdefault("sample_inputs_func", sample_inputs_unary)
        super().__init__(name, ref=ref, **kwargs)


class BinaryUfuncInfo(OpInfo):
    """Elementwise binary op (add, mul, div, maximum, ...), with broadcasting samples."""

    def __init__(self, name, *, ref=None, **kwargs):
        kwargs.setdefault("sample_inputs_func", sample_inputs_binary)
        super().__init__(name, ref=ref, **kwargs)


class ReductionOpInfo(OpInfo):
    """Reduction op (sum, mean, max, ...). Samples sweep dim / keepdim.

    Reductions were the audit's biggest *backward* hole (broadcast-back gradient,
    keepdims, negative dims). The default samples exercise those axes so the generic
    gradcheck covers them automatically.
    """

    def __init__(self, name, *, ref=None, **kwargs):
        kwargs.setdefault("sample_inputs_func", sample_inputs_reduction)
        super().__init__(name, ref=ref, **kwargs)


# --------------------------------------------------------- default sample funcs

def sample_inputs_unary(op_info, device, dtype, requires_grad):
    lo, hi = getattr(op_info, "domain", (None, None))
    shapes = [(5,), (3, 4), (2, 3, 4)]
    return [SampleInput(cu.make_tensor(*s, dtype=dtype, low=lo, high=hi,
                                       requires_grad=requires_grad, seed=100 + i))
            for i, s in enumerate(shapes)]


def sample_inputs_binary(op_info, device, dtype, requires_grad):
    pairs = [((3, 4), (3, 4)), ((3, 4), (4,)), ((2, 1, 4), (3, 4))]  # incl. broadcast
    samples = []
    for i, (sa, sb) in enumerate(pairs):
        a = cu.make_tensor(*sa, dtype=dtype, requires_grad=requires_grad, seed=200 + i)
        b = cu.make_tensor(*sb, dtype=dtype, requires_grad=requires_grad, seed=250 + i,
                           low=0.5, high=2.0)  # keep >0 so div/pow refs stay finite
        samples.append(SampleInput(a, b))
    return samples


def sample_inputs_reduction(op_info, device, dtype, requires_grad):
    samples = []
    base = cu.make_tensor(3, 4, 5, dtype=dtype, requires_grad=requires_grad, seed=300)
    samples.append(SampleInput(base))                              # full reduce
    for i, dim in enumerate([0, 1, 2, -1]):
        for keepdims in (False, True):
            samples.append(SampleInput(
                cu.make_tensor(3, 4, 5, dtype=dtype, requires_grad=requires_grad,
                               seed=310 + i),
                dim=dim, keepdims=keepdims))
    return samples
