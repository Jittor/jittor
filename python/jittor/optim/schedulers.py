"""Learning-rate schedulers bundled with the optimizer facade."""

from .base import Optimizer

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
