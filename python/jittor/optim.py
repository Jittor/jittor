# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
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

    def add_param_group(self, group):
        self.param_groups.append(group)

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
                if p.is_stop_grad(): continue
                grads.append(g.flatten())
        if len(grads) == 0: return
        total_norm = jt.norm(jt.concat(grads), norm_type)
        clip_coef = jt.minimum(max_norm / (total_norm + 1e-6), 1.0)
        for pg in self.param_groups:
            for p, g in zip(pg["params"], pg["grads"]):
                if p.is_stop_grad(): continue
                g.update(g*clip_coef)

    
    @property
    def defaults(self):
        exclude = set(("defaults", "param_groups", "n_step", "pre_step", "step"))
        return { k:v for k, v in self.__dict__.items()
            if k[0] != '_' and k not in exclude and not callable(v) }

    def state_dict(self):
        state = {"defaults": self.defaults}
        return state

    def load_state_dict(self, state):
        for k,v in state["defaults"].items():
            setattr(self, k, v)

    def zero_grad(self):
        self.__zero_grad = True

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
        for pg in self.param_groups:
            for p in pg['params']:
                params.append(p)
                if not p.is_stop_grad():
                    params_has_grad.append(p)

        # get gradient
        grads = jt.grad(loss, params_has_grad)

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
                if not p.is_stop_grad():
                    # accumulate grad and stop grad of grad
                    g = grads[pid].stop_grad()
                    if not self.__zero_grad:
                        g = g + pg_grads[i]
                    pg_grads[i].update(g)
                    pid += 1
        self.__zero_grad = False
        
    def backward(self, loss):
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
        self.pre_step(loss)

    def step(self, loss=None):
        if loss is not None:
            self.pre_step(loss)
        for pg in self.param_groups:
            lr = pg.get("lr", self.lr)
            for p, g in zip(pg["params"], pg["grads"]):
                if p.is_stop_grad(): continue
                p.update(p - g * lr)
        self.zero_grad()

    def _build_grad_map(self):
        _grad_map = {}
        for pg in self.param_groups:
            for p, g in zip(pg["params"], pg["grads"]):
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
                values.append(jt.zeros(p.shape, p.dtype).stop_grad())

    def add_param_group(self, group):
        values = group["values"] = []
        for p in group["params"]:
            values.append(jt.zeros(p.shape, p.dtype).stop_grad())
        self.param_groups.append(group)

    def step(self, loss=None):
        if loss is not None:
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
                v.update(momentum * v + dp * (1 - dampening))
                if nesterov:
                    p.update(p - (dp + momentum * v) * lr)
                else:
                    p.update(p - v * lr)
        self.zero_grad()

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
                values.append(jt.zeros(p.shape, p.dtype).stop_grad())

    def add_param_group(self, group):
        values = group["values"] = []
        for p in group["params"]:
            values.append(jt.zeros(p.shape, p.dtype).stop_grad())
        self.param_groups.append(group)

    def step(self, loss=None):
        if loss is not None:
            self.pre_step(loss)
        for pg in self.param_groups:
            # get arguments from each param_groups
            lr = pg.get("lr", self.lr)
            eps = pg.get("eps", self.eps)
            alpha = pg.get("alpha", self.alpha)
            for p, g, v in zip(pg["params"], pg["grads"], pg["values"]):
                if p.is_stop_grad(): continue
                v.update(alpha * v + (1-alpha) * g * g)
                p.update(p - lr * g / (jt.sqrt(v) + eps))
        self.zero_grad()

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
        self.weight_decay = weight_decay
        # assert weight_decay==0, "weight_decay is not supported yet"
        
        # initialize required arguments for each param_groups
        for pg in self.param_groups:
            values = pg["values"] = []
            m = pg["m"] = []
            for p in pg["params"]:
                values.append(jt.zeros(p.shape, p.dtype).stop_grad())
                m.append(jt.zeros(p.shape, p.dtype).stop_grad())

    def add_param_group(self, group):
        values = group["values"] = []
        m = group["m"] = []
        for p in group["params"]:
            values.append(jt.zeros(p.shape, p.dtype).stop_grad())
            m.append(jt.zeros(p.shape, p.dtype).stop_grad())
        self.param_groups.append(group)

    def step(self, loss=None):
        if loss is not None:
            self.pre_step(loss)
        n = float(self.n_step)
        for pg in self.param_groups:
            # get arguments from each param_groups
            lr = pg.get("lr", self.lr)
            eps = pg.get("eps", self.eps)
            weight_decay = pg.get("weight_decay", self.weight_decay)
            b0, b1 = pg.get("betas", self.betas)
            for p, g, v, m in zip(pg["params"], pg["grads"], pg["values"], pg["m"]):
                if p.is_stop_grad(): continue
                g = p * weight_decay + g
                m.update(b0 * m + (1-b0) * g)
                v.update(b1 * v + (1-b1) * g * g)
                step_size = lr * jt.sqrt(1-b1**n) / (1-b0 ** n)
                p.update(p - m * step_size / (jt.sqrt(v) + eps))
        self.zero_grad()


class AdamW(Optimizer):
    """ AdamW Optimizer.
    
    Example::

        optimizer = nn.AdamW(model.parameters(), lr, eps=1e-8, betas=(0.9, 0.999))
        optimizer.step(loss)
    """
    def __init__(self, params, lr, eps=1e-8, betas=(0.9, 0.999), weight_decay=0):
        super().__init__(params, lr)
        self.eps = eps
        self.betas = betas
        self.weight_decay = weight_decay
        # assert weight_decay==0, "weight_decay is not supported yet"
        
        # initialize required arguments for each param_groups
        for pg in self.param_groups:
            values = pg["values"] = []
            m = pg["m"] = []
            for p in pg["params"]:
                values.append(jt.zeros(p.shape, p.dtype).stop_grad())
                m.append(jt.zeros(p.shape, p.dtype).stop_grad())

    def add_param_group(self, group):
        values = group["values"] = []
        m = group["m"] = []
        for p in group["params"]:
            values.append(jt.zeros(p.shape, p.dtype).stop_grad())
            m.append(jt.zeros(p.shape, p.dtype).stop_grad())
        self.param_groups.append(group)

    def step(self, loss=None):
        if loss is not None:
            self.pre_step(loss)
        n = float(self.n_step)
        for pg in self.param_groups:
            # get arguments from each param_groups
            lr = pg.get("lr", self.lr)
            eps = pg.get("eps", self.eps)
            weight_decay = pg.get("weight_decay", self.weight_decay)
            b0, b1 = pg.get("betas", self.betas)
            for p, g, v, m in zip(pg["params"], pg["grads"], pg["values"], pg["m"]):
                if p.is_stop_grad(): continue
                p.update(p * (1 - lr * weight_decay))
                bias_correction1 = 1 - b0 ** n
                bias_correction2 = 1 - b1 ** n
                m.update(b0 * m + (1-b0) * g) #exp_avg
                v.update(b1 * v + (1-b1) * g * g) #exp_avg_sq
                denom = jt.sqrt(v) / jt.sqrt(bias_correction2) + eps
                step_size = lr / bias_correction1
                p.update(p - step_size * m / denom)
        self.zero_grad()


class LRScheduler:
    def __init__(self,optimizer, last_epoch=-1):
        assert isinstance(optimizer,Optimizer)
        self.optimizer = optimizer
        
        if last_epoch==-1:
            for gp in optimizer.param_groups:
                gp.setdefault('initial_lr',gp.get('lr',optimizer.lr))
        else:
            for gp in optimizer.param_groups:
                assert 'initial_lr' in gp 
        
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_epoch = last_epoch
        self.optimizer._step_count = 0
        self._step_count = 0
        self.step()

    def get_lr(self):
        raise NotImplementedError 
    
    def get_last_lr(self):
        return self._last_lr

    def step(self,epoch=None):
        self._step_count += 1

        if epoch is None:
            self.last_epoch += 1
            values = self.get_lr()
        else:
            self.last_epoch = epoch
            values = self.get_lr()

        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, lr = data
            param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


class LambdaLR(LRScheduler):

    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        if not isinstance(lr_lambda, list) and not isinstance(lr_lambda, tuple):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            if len(lr_lambda) != len(optimizer.param_groups):
                raise ValueError("Expected {} lr_lambdas, but got {}".format(len(optimizer.param_groups), len(lr_lambda)))

            self.lr_lambdas = list(lr_lambda)
            
        super(LambdaLR, self).__init__(optimizer, last_epoch)
        
        

    def get_lr(self):
        return [base_lr * lmbda(self.last_epoch)
                for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]
