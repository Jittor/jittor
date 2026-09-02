# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers:
#     Guowei Yang <471184555@qq.com>
#     Guoye Yang <498731903@qq.com>
#     Wenyang Zhou <576825820@qq.com>
#     Meng-Hao Guo <guomenghao1997@gmail.com>
#     Dun Liang <randonlang@gmail.com>.
#
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
import numpy as np
from copy import deepcopy

def _grad_matches_param(p, g):
    return isinstance(g, jt.Var) and list(g.shape) == list(p.shape)

def _param_requires_grad(p):
    return bool(p.requires_grad)

def _update_preserve_dtype(target, value):
    if str(value.dtype) != str(target.dtype):
        value = value.to(target.dtype)
    target.update(value)

class Optimizer(object):
    """ Basic class of Optimizer.

    Example::

        optimizer = nn.SGD(model.parameters(), lr)
        optimizer.step(loss)
    """
    def __init__(self, params, lr, param_sync_iter=10000):
        self.param_groups = []
        self.lr = lr
        self.param_sync_iter = param_sync_iter

        assert len(params) > 0, "Length of parameters should not be zero"
        if not isinstance(params[0], dict):
            params = [{'params': params}]
        for pg in params:
            assert isinstance(pg, dict)
            self.param_groups.append(pg)
        self.n_step = 0
        # __zero_grad is a value for fast determ the grad is zero or not
        # so we can omit 0+x
        self.__zero_grad = True
        self._grad_map = {}
        self.__input_params = []

    def add_param_group(self, group):
        self.param_groups.append(group)

    def _advance_step_count(self, pg):
        """Number of optimizer steps this param group has taken, 1-based.

        Bias corrections must use this, not ``self.n_step``: ``n_step`` counts
        ``backward`` calls, and the gradient-accumulation loop documented on
        ``backward`` calls it once per micro-batch, so a correction keyed on it
        runs ahead by the accumulation factor. Kept inside the param group so
        it rides along in ``state_dict``/``load_state_dict`` and so a group
        added mid-training starts its own correction at step 1.
        """
        n = int(pg.get("n_step", 0)) + 1
        pg["n_step"] = n
        return n

    def set_input_into_param_group(self, inputs):
        """ This function adds inputs to the optimizer as variables that need tuning.
            This is to enforce the calculation of gradients from the output to the input,
            ensuring that the backward hook is called correctly.

        Args:
            inputs: List of the input
        """
        self.__input_params = []
        if isinstance(inputs, jt.Var):
            self.__input_params.append(inputs)
        elif isinstance(inputs, (list, tuple)):
            for v in inputs:
                if isinstance(v, jt.Var):
                    self.__input_params.append(v)
        else:
            raise NotImplementedError

    def clip_grad_norm(self, max_norm:float, norm_type:int=2):
        r"""Clips gradient norm of this optimizer.
        The norm is computed over all gradients together.

        Args:
            max_norm (float or int): max norm of the gradients
            norm_type (int): 1-norm or 2-norm

        Example::

            a = jt.ones(2)
            opt = jt.optim.SGD([a], 0.1)

            loss = a*a
            opt.zero_grad()
            opt.backward(loss)

            print(opt.param_groups[0]['grads'][0].norm()) # output: 2.83
            opt.clip_grad_norm(0.01, 2)
            print(opt.param_groups[0]['grads'][0].norm()) # output: 0.01

            opt.step()

        """
        if self.__zero_grad: return
        grads = []
        for pg in self.param_groups:
            for p, g in zip(pg["params"], pg["grads"]):
                if not _param_requires_grad(p) or not _grad_matches_param(p, g): continue
                grads.append(g.flatten())
        if len(grads) == 0: return
        total_norm = jt.norm(jt.concat(grads), norm_type)
        clip_coef = jt.minimum(max_norm / (total_norm + 1e-6), 1.0)
        for pg in self.param_groups:
            for p, g in zip(pg["params"], pg["grads"]):
                if not _param_requires_grad(p) or not _grad_matches_param(p, g): continue
                g.update(g*clip_coef)

    @property
    def defaults(self):
        exclude = set(("defaults", "pre_step", "step"))
        return { k:v for k, v in self.__dict__.items()
            if k[0] != '_' and k not in exclude and not callable(v) }

    def state_dict(self):
        state = {"defaults": self.defaults}
        return state

    def load_state_dict(self, state):

        def dfs(x):
            if isinstance(x, list):
                return [dfs(value) for value in x]
            if isinstance(x, tuple):
                return tuple(dfs(value) for value in x)
            if isinstance(x, dict):
                return {key: dfs(value) for key, value in x.items()}
            if isinstance(x, np.ndarray):
                return jt.array(x).stop_grad()
            if isinstance(x, jt.Var):
                return x.clone().stop_grad()
            return x

        exclude = set(("param_groups", "params"))
        for k, v in state["defaults"].items():
            if k not in exclude:
                setattr(self, k, dfs(v))
        param_groups = state["defaults"].get('param_groups', None)
        if param_groups is not None:
            for i in range(len(param_groups)):
                for k, v in param_groups[i].items():
                    if k != "params":
                        self.param_groups[i][k] = dfs(v)



    def zero_grad(self):
        ''' Reset the accumulated gradients of every param group to zero.

        The buffers in ``pg["grads"]`` are cleared, not just marked: they are
        public (``opt_grad``, ``clip_grad_norm`` and user code all read them),
        and ``post_step`` calls this after every step, so leaving the consumed
        gradients in place made every later reader silently work on stale data.

        The write is skipped when the gradients are already known to be zero,
        and in the ordinary training loop the zeros are overwritten by the next
        ``backward`` before anything can observe them, so jittor's lazy graph
        drops them without ever running the fill.
        '''
        if not self.__zero_grad:
            for pg in self.param_groups:
                for g in pg.get("grads", ()):
                    g.update(jt.zeros_like(g).stop_grad())
        self.__zero_grad = True

    def backward(self, loss, retain_graph=False):
        '''
        optimize.backward(loss) is used for accumulate multiple step,
        it can be used as following:

        Origin source code ::

        n_iter = 10000
        batch_size = 100
        ...
        for i in range(n_iter):
            ...
            loss = calc_loss()
            optimizer.step(loss)

        Accumulation version ::

        n_iter = 10000
        batch_size = 100
        accumulation_steps = 10
        n_iter *= accumulation_steps
        batch_size //= accumulation_steps
        ...
        for i in range(n_iter):
            ...
            loss = calc_loss()
            # if loss is a mean across batch, we need to divide accumulation_steps
            optimizer.backward(loss / accumulation_steps)
            if (i+1) % accumulation_steps == 0:
                optimizer.step()


        '''
        # clean prev grads
        params = []
        params_has_grad = []
        for pg in self.param_groups:
            for p in pg['params']:
                params.append(p)
                if _param_requires_grad(p):
                    params_has_grad.append(p)
        for p in self.__input_params:
            if _param_requires_grad(p):
                params_has_grad.append(p)

        # sync prev params
        jt.sync(params_has_grad)

        # get gradient
        grads = jt.grad(loss, params_has_grad, retain_graph)

        # sync grads and model if in mpi
        if jt.in_mpi:
            dep = []
            def add_dep(v):
                nonlocal dep
                v._add_dependency(dep)
                dep = [v]

            for g in grads:
                g.assign(g.mpi_all_reduce("mean"))
                add_dep(g._input(0))
            if self.n_step % self.param_sync_iter == 0:
                for p in params:
                    p.assign(p.mpi_broadcast())
                    add_dep(p)
        self.n_step += 1

        # set up grads in param_groups
        pid = 0
        for pg in self.param_groups:
            if "grads" not in pg:
                pg["grads"] = [ jt.zeros_like(p).stop_grad().stop_fuse() for p in pg['params'] ]
            pg_grads = pg["grads"]
            for i, p in enumerate(pg['params']):
                if _param_requires_grad(p):
                    # accumulate grad and stop grad of grad
                    g = grads[pid].stop_grad()
                    if not self.__zero_grad:
                        g = g + pg_grads[i]
                    pg_grads[i].update(g)
                    pid += 1
        self.__zero_grad = False

    def pre_step(self, loss, retain_graph=False):
        """ something should be done before step, such as calc gradients, mpi sync, and so on.

        Example::

            class MyOptimizer(Optimizer):
                def step(self, loss):
                    self.pre_step(loss)
                    ...
                    self.post_step()
        """
        if loss is not None:
            self.backward(loss, retain_graph)
        jt.flags.node_order = 1

    def post_step(self):
        """ something should be done before step, such as zero grad, and so on.

        Example::

            class MyOptimizer(Optimizer):
                def step(self, loss):
                    self.pre_step(loss)
                    ...
                    self.post_step()
        """
        jt.flags.node_order = 0
        self.zero_grad()


    def step(self, loss=None, retain_graph=False):
        self.pre_step(loss, retain_graph)
        for pg in self.param_groups:
            lr = pg.get("lr", self.lr)
            for p, g in zip(pg["params"], pg["grads"]):
                if not _param_requires_grad(p) or not _grad_matches_param(p, g): continue
                _update_preserve_dtype(p, p - g * lr)
        self.post_step()

    def _build_grad_map(self):
        _grad_map = {}
        for pg in self.param_groups:
            for p, g in zip(pg["params"], pg["grads"]):
                if not _grad_matches_param(p, g):
                    continue
                _grad_map[id(p)] = g
        self._grad_map = _grad_map

    def find_grad(self, v:jt.Var) -> jt.Var:
        if id(v) not in self._grad_map:
            self._build_grad_map()
            if id(v) not in self._grad_map:
                raise RuntimeError("This variable is not managed by this optimizer")
        return self._grad_map[id(v)]

def opt_grad(v:jt.Var, opt:Optimizer):
    ''' Get grad of certain variable in optimizer, Example::


    model = Model()
    optimizer = SGD(model.parameters())
    ...
    optimizer.backward(loss)

    for p in model.parameters():
        grad = p.opt_grad(optimizer)
    '''
    return opt.find_grad(v)


jt.Var.opt_grad = opt_grad
