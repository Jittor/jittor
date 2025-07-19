import jittor as jt
from jittor import Var
from collections.abc import Sequence, Mapping

Variable = Var

class FunctionContext:
    def save_for_backward(self, *args):
        self.saved_tensors = args

class Function:
    ''' Function Module for customized backward operations

Example 1 (Function can have multiple input and multiple output, and user
can store value for backward computation)::

    import jtorch
    from jtorch import Function

    class MyFunc(Function):
        @staticmethod
        def forward(self, x, y):
            self.x = x
            self.y = y
            return x*y, x/y

        @staticmethod
        def backward(self, grad0, grad1):
            return grad0 * self.y, grad1 * self.x

    a = jtorch.array(3.0)
    a.requires_grad = True
    b = jtorch.array(4.0)
    b.requires_grad = True
    func = MyFunc.apply
    c,d = func(a, b)
    (c+d*3).backward()
    assert a.grad.data == 4
    assert b.grad.data == 9

Example 2(Function can return None for no gradiant, and gradiant
can also be None)::

    import jtorch
    from jtorch import Function
    
    class MyFunc(Function):
        @staticmethod
        def forward(self, x, y):
            self.x = x
            self.y = y
            return x*y, x/y

        @staticmethod
        def backward(self, grad0, grad1):
            assert grad1 is None
            return grad0 * self.y, None
    a = jt.array(3.0)
    a.requires_grad = True
    b = jt.array(4.0)
    b.requires_grad = True
    func = MyFunc.apply
    c,d = func(a, b)
    d.stop_grad()
    da, db = jt.grad(c+d*3, [a, b])
    assert da.data == 4
    assert db.data == 0

    '''
    def __call__(self, *args):
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
        ctx = FunctionContext()
        ori_res = self.forward(ctx, *args)
        # ori_res = self.execute(*args)
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
        ctx.input_mask = input_mask
        ctx.output_mask = output_mask
        # tape output and input together so
        # backward treat them as one operator
        jt.tape_together(taped_inputs, taped_outputs, 
            lambda *args: self._grad(ctx, self, *args))
        if isinstance(ori_res, Sequence):
            return res
        else:
            return res[0]

    @staticmethod
    def _grad(ctx, func, *args):
        new_args = ( (args[i] if i>=0 else None) for i in ctx.output_mask )
        ret = func.backward(ctx, *new_args)
        if not isinstance(ret, Sequence):
            ret = (ret,)
        new_ret = []
        for i, r in enumerate(ret):
            j = ctx.input_mask[i]
            if j<0:
                # -2 in input_mask represents it is stop_grad
                assert r is None or j==-2, f"{type(self)}'s {i}-th returned grad should be None, "\
                    "because the input value is not jittor variable."
            else:
                new_ret.append(r)
        return new_ret

    def dfs(self, parents, k, callback, callback_leave=None):
        pass

    @classmethod
    def apply(cls, *args, **kw):
        func = cls()
        return func(*args, **kw)
