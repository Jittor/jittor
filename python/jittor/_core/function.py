"""One-call autograd contexts and their taped gradient nodes."""
from collections.abc import Sequence
from jittor_core import Var, tape_together
from .flags import flags
from .module import Module


class Function(Module):
    ''' Function Module for customized backward operations

Example 1 (Function can have multiple input and multiple output, and user
can store value for backward computation)::

    import jittor as jt
    from jittor import Function

    class MyFunc(Function):
        def execute(self, x, y):
            self.x = x
            self.y = y
            return x*y, x/y

        def grad(self, grad0, grad1):
            return grad0 * self.y, grad1 * self.x
    a = jt.array(3.0)
    b = jt.array(4.0)
    func = MyFunc.apply
    c,d = func(a, b)
    da, db = jt.grad(c+d*3, [a, b])
    assert da.data == 4
    assert db.data == 9

Example 2(Function can return None for no gradiant, and gradiant
can also be None)::

    import jittor as jt
    from jittor import Function

    class MyFunc(Function):
        def execute(self, x, y):
            self.x = x
            self.y = y
            return x*y, x/y

        def grad(self, grad0, grad1):
            assert grad1 is None
            return grad0 * self.y, None
    a = jt.array(3.0)
    b = jt.array(4.0)
    func = MyFunc.apply
    c,d = func(a, b)
    d.stop_grad()
    da, db = jt.grad(c+d*3, [a, b])
    assert da.data == 4
    assert db.data == 0

    '''
    def _new_call_context(self):
        """A one-shot object to run this call's ``execute``/``grad`` against.

        Everything a Function saves for its backward -- the user's
        ``self.x = x`` and the framework's own input/output masks -- used to
        live on the Function INSTANCE. Calling one instance twice therefore
        overwrote the first call's saved state, and the first call's backward
        then ran against the second call's tensors: a **wrong gradient with no
        warning**::

            f = Mul(); o1 = f(a, b); o2 = f(a, c)
            jt.grad(o1, a)      # used to give dc, not db

        ``MyFunc.apply(...)`` happened to be safe because it builds a new
        instance per call, and the examples above only show that spelling --
        but ``f = MyFunc(); f(x); f(y)`` is just as natural, and 50+ Function
        subclasses in this tree save state this way.

        The context starts as a shallow copy of the instance ``__dict__``, so
        anything ``__init__`` configured is visible to ``execute``, while
        writes made during the call land on the context and leave the shared
        instance alone. Binding ``ctx._grad`` into the tape keeps the context
        alive exactly as long as its backward might run.
        """
        ctx = object.__new__(type(self))
        ctx.__dict__.update(self.__dict__)
        return ctx

    @staticmethod
    def _reject_var_keywords(owner, kw):
        # Only positional arguments are taped, so a Var passed by keyword would
        # silently come back with no gradient. Say so instead.
        for k, v in kw.items():
            if isinstance(v, Var) and not v.is_stop_grad():
                raise TypeError(
                    f"{owner}: pass differentiable Var arguments positionally, "
                    f"not as the keyword {k!r}. Keyword arguments are not taped, "
                    "so this Var would silently receive no gradient.")

    def __call__(self, *args, **kw):
        # One context per call. `self` is only a factory from here on.
        return self._new_call_context()._run_call(*args, **kw)

    def _run_call(self, *args, **kw):
        """Run one call. ``self`` here is a one-shot context, not the instance.

        Split out of ``__call__`` so that a wrapper which needs per-call
        bookkeeping of its own can build the context, write onto it, and run
        the call against that same object::

            ctx = fn._new_call_context()
            ctx.my_state = ...          # visible to execute() and to grad()
            out = ctx._run_call(*args)
            ctx.more_state = ...        # still visible to grad()

        The torch compatibility layer does exactly this (it records
        ``needs_input_grad`` and the forward input/output shapes). Writing that
        bookkeeping onto the Function INSTANCE instead does not work, and fails
        in two different directions: whatever is written before the call is
        overwritten by the next call of the same instance, and whatever is
        written after the call never reaches the backward at all, because the
        context was copied from the instance when the call started.
        """
        self._reject_var_keywords(type(self).__name__, kw)
        if flags.no_grad:
            return self.execute(*args, **kw)
        backup = args
        args = list(args)
        taped_inputs = []
        taped_outputs = []
        input_mask = [-1] * len(args)
        for i,v in enumerate(args):
            if isinstance(v, Var):
                if v.is_stop_grad():
                    # -2 in input_mask represents it is stop_grad
                    input_mask[i] = -2
                    continue
                v = v.tape()
                input_mask[i] = len(taped_inputs)
                args[i] = v
                taped_inputs.append(v)
        ori_res = self.execute(*args, **kw)
        if not isinstance(ori_res, Sequence):
            res = [ori_res]
        else:
            res = list(ori_res)
        output_mask = [-1] * len(res)
        for i,v in enumerate(res):
            if isinstance(v, Var):
                v = v.tape()
                output_mask[i] = len(taped_outputs)
                res[i] = v
                taped_outputs.append(v)
        self.input_mask = input_mask
        self.output_mask = output_mask
        # tape output and input together so
        # backward treat them as one operator
        tape_together(taped_inputs, taped_outputs, self._grad)
        if isinstance(ori_res, Sequence):
            return res
        else:
            return res[0]

    def _grad(self, *args):
        new_args = ( (args[i] if i>=0 else None) for i in self.output_mask )
        ret = self.grad(*new_args)
        if not isinstance(ret, Sequence):
            ret = (ret,)
        new_ret = []
        for i, r in enumerate(ret):
            j = self.input_mask[i]
            if j<0:
                # -2 in input_mask represents it is stop_grad
                assert r is None or j==-2, f"{type(self)}'s {i}-th returned grad should be None, "\
                    "because the input value is not jittor variable."
            else:
                new_ret.append(r)
        return new_ret

    def dfs(self, parents, k, callback, callback_leave=None, recurse=True):
        pass

    @classmethod
    def apply(cls, *args, **kw):
        # Same contract as __call__, which now also accepts **kw (it used to
        # reject it outright, so `apply` could not actually forward keywords).
        func = cls()
        return func(*args, **kw)


class GradHooker(Function):
    def __init__(self, hook):
        self.hook = hook

    def execute(self, *args):
        return args

    def grad(self, *grad_input):
        ret = self.hook(grad_input)
        if ret: grad_input = ret
        return grad_input
