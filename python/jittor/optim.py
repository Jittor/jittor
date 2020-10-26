# ***************************************************************
# Copyright (c) 2020 Jittor. Authors:
#     Guowei Yang <471184555@qq.com>
#     Guoye Yang <498731903@qq.com>
#     Wenyang Zhou <576825820@qq.com>
#     Meng-Hao Guo <guomenghao1997@gmail.com>
#     Dun Liang <randonlang@gmail.com>.
#
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
import numpy as np

def add_dep(a, b):
    ''' add dependency from a to b '''
    b.swap(b + a.sum() * 0)
    # b.swap(b.clone())
    # b._add_input(a)

class AllReduceFuser:
    sum_limit = 1048576
    num_limit = 10

    vars = []
    shapes = []
    cnts = []
    sum = 0
    
    def push(self, g):
        s = 1
        for ss in g.shape:
            s = s * ss
        self.vars.append(g)
        self.shapes.append(g.shape)
        self.cnts.append(s)
        self.sum += s
        # self.send()
        if (self.sum >= self.sum_limit or len(self.vars) >= self.num_limit):
            self.send()

    def send(self):
        if (len(self.vars) == 0):
            return
        tmp = []
        for v in self.vars:
            tmp.append(v.reshape(-1))
        t = jt.ops.concat(tmp, 0)
        t.compile_options = {'optim_all_reduce':1}
        temp = t.mpi_all_reduce("mean")
        ssum = 0
        for i in range(len(self.vars)):
            # aft_v = temp[ssum:ssum + self.cnts[i]]
            aft_v = temp[0:0 + self.cnts[i]]
            aft_v = aft_v.reshape(*self.shapes[i])
            self.vars[i].assign(aft_v)
            ssum += self.cnts[i]
        self.sum = 0
        self.vars = []
        self.shapes = []
        self.cnts = []

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
        self.fuser = AllReduceFuser()

    def pre_step(self, loss):
        """ something should be done before step, such as calc gradients, mpi sync, and so on.

        Example::

            class MyOptimizer(Optimizer):
                def step(self, loss):
                    self.post_step(loss)
                    ...
        """
        # clean prev grads
        params = []
        params_has_grad = []
        names = []
        for pg in self.param_groups:
            pg["grads"] = [None] * len(pg['params'])
            for p in pg['params']:
                params.append(p)
                if not p.is_stop_grad():
                    params_has_grad.append(p)
                    names.append(p.name())
        
        # sync params, reduce computing graph size
        jt.sync(params)

        # get gradient
        grads = jt.grad(loss, params_has_grad)
        cnt = 0
        # sync grads and model if in mpi
        if jt.mpi:
            last = None
            cnt = 0

            for g in grads:
                cnt += 1
                if (cnt <= -1 and last is not None):
                    g.swap(g + last.sum() * 0)
                g.compile_options = {'optim_all_reduce':1}   
                g.name(f"cnt_{cnt}_step_{self.n_step}_{names[cnt-1]}")
                temp = g.mpi_all_reduce("mean")
                g.swap(temp)
                last = g

            # for g in grads:
            #     g.compile_options = {'optim_all_reduce':1}
            #     temp = g.mpi_all_reduce("mean")
            #     g.assign(temp)

            # for g in grads:
            #     self.fuser.push(g)
            #     self.fuser.send()
            # self.fuser.send()

            # if self.n_step % self.param_sync_iter == 0:
            #     for p in params:
            #         p.assign(p.mpi_all_reduce("mean"))
        self.n_step += 1

        # set up grads in param_groups
        pid = 0
        for pg in self.param_groups:
            pg_grads = pg["grads"]
            for i, p in enumerate(pg['params']):
                if not p.is_stop_grad():
                    pg_grads[i] = grads[pid]
                    pid += 1
        
    def step(self, loss):
        self.pre_step(loss)
        for pg in self.param_groups:
            lr = pg.get("lr", self.lr)
            for p, g in zip(pg["params"], pg["grads"]):
                if p.is_stop_grad(): continue
                p -= g * lr
                # detach with the prev graph to reduce memory consumption
                p.detach_inplace()


class SGD(Optimizer):
    """ SGD Optimizer.

    Example::

        optimizer = nn.SGD(model.parameters(), lr, momentum=0.9)
        optimizer.step(loss)
    """
    def __init__(self, params, lr, momentum=0, weight_decay=0, dampening=0, nesterov=False):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dampening = dampening
        self.nesterov = nesterov

        # initialize required arguments
        for pg in self.param_groups:
            values = pg["values"] = []
            for p in pg["params"]:
                values.append(jt.zeros(p.shape, p.dtype).stop_fuse().stop_grad())

    def step(self, loss):
        self.pre_step(loss)
        for pg in self.param_groups:
            # get arguments from each param_groups
            lr = pg.get("lr", self.lr)
            momentum = pg.get("momentum", self.momentum)
            weight_decay = pg.get("weight_decay", self.weight_decay)
            dampening = pg.get("dampening", self.dampening)
            nesterov = pg.get("nesterov", self.nesterov)

            # optimize main body
            for p, g, v in zip(pg["params"], pg["grads"], pg["values"]):
                if p.is_stop_grad(): continue
                dp = p * weight_decay + g
                v.assign(momentum * v + dp * (1 - dampening))
                if nesterov:
                    p -= (dp + momentum * v) * lr
                else:
                    p -= v * lr
                p.detach_inplace()

class RMSprop(Optimizer):
    """ RMSprop Optimizer.
    Args:
        params(list): parameters of model.
        lr(float): learning rate.
        eps(float): term added to the denominator to avoid division by zero, default 1e-8.
        alpha(float): smoothing constant, default 0.99.

    Example:
        optimizer = nn.RMSprop(model.parameters(), lr)
        optimizer.step(loss)
    """
    def __init__(self, params, lr=1e-2, eps=1e-8, alpha=0.99):
        super().__init__(params, lr)
        self.eps = eps
        self.alpha = alpha
        
        # initialize required arguments for each param_groups
        for pg in self.param_groups:
            values = pg["values"] = []
            for p in pg["params"]:
                values.append(jt.zeros(p.shape, p.dtype).stop_fuse().stop_grad())

    def step(self, loss):
        self.pre_step(loss)
        for pg in self.param_groups:
            # get arguments from each param_groups
            lr = pg.get("lr", self.lr)
            eps = pg.get("eps", self.eps)
            alpha = pg.get("alpha", self.alpha)
            for p, g, v in zip(pg["params"], pg["grads"], pg["values"]):
                if p.is_stop_grad(): continue
                v.assign(alpha * v + (1-alpha) * g * g)
                p -= lr * g / (jt.sqrt(v) + eps)
                p.detach_inplace()

class Adam(Optimizer):
    """ Adam Optimizer.
    
    Example::

        optimizer = nn.Adam(model.parameters(), lr, eps=1e-8, betas=(0.9, 0.999))
        optimizer.step(loss)
    """
    def __init__(self, params, lr, eps=1e-8, betas=(0.9, 0.999), weight_decay=0):
        super().__init__(params, lr)
        self.eps = eps
        self.betas = betas
        # self.weight_decay = weight_decay
        assert weight_decay==0, "weight_decay is not supported yet"
        
        # initialize required arguments for each param_groups
        for pg in self.param_groups:
            values = pg["values"] = []
            m = pg["m"] = []
            for p in pg["params"]:
                values.append(jt.zeros(p.shape, p.dtype).stop_fuse().stop_grad())
                m.append(jt.zeros(p.shape, p.dtype).stop_fuse().stop_grad())

    def step(self, loss):
        self.pre_step(loss)
        n = float(self.n_step)
        for pg in self.param_groups:
            # get arguments from each param_groups
            lr = pg.get("lr", self.lr)
            eps = pg.get("eps", self.eps)
            b0, b1 = pg.get("betas", self.betas)
            for p, g, v, m in zip(pg["params"], pg["grads"], pg["values"], pg["m"]):
                if p.is_stop_grad(): continue
                m.assign(b0 * m + (1-b0) * g)
                v.assign(b1 * v + (1-b1) * g * g)
                step_size = lr * jt.sqrt(1-b1**n) / (1-b0 ** n)
                p -= m * step_size / (jt.sqrt(v) + eps)
                p.detach_inplace()
