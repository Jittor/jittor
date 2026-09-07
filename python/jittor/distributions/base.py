# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers:
#     Haoyang Peng <2247838039@qq.com>
#     Dun Liang <randonlang@gmail.com>.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************


from ._utils import _full_shape


class Distribution:
    ''' Base class for every distribution in this module, matching
    torch.distributions.Distribution closely enough for ``isinstance`` checks
    and for the shared ``sample`` contract.

    ``sample()`` is implemented **here, once**, and always detaches: torch's
    ``sample`` is ``with torch.no_grad(): return self.rsample(...)``, so a draw
    never carries a gradient path back to the parameters. Subclasses provide
    ``_sample_impl`` (the drawing itself) and never call ``stop_grad``
    themselves. Half the distributions here used to override ``sample``
    without detaching, which silently reconnected policy/variational
    parameters to their own samples.
    '''
    has_rsample = False
    arg_constraints = {}
    batch_shape = ()
    event_shape = ()

    def __init__(self, batch_shape=(), event_shape=(), validate_args=None):
        self.batch_shape = tuple(batch_shape)
        self.event_shape = tuple(event_shape)
        self._validate_args = validate_args

    def _extended_shape(self, sample_shape=None):
        return _full_shape(sample_shape, self.batch_shape, self.event_shape)

    def _validate_sample(self, value):
        return None

    def _sample_impl(self, sample_shape=None):
        ''' Draw a sample. Subclasses override this, never :meth:`sample`. '''
        if type(self).rsample is Distribution.rsample:
            raise NotImplementedError(
                "%s implements neither _sample_impl nor rsample"
                % type(self).__name__)
        return self.rsample(sample_shape)

    def sample(self, sample_shape=None):
        ''' A detached draw: no gradient flows back to the parameters. '''
        import jittor as jt
        result = self._sample_impl(sample_shape)
        return result.stop_grad() if isinstance(result, jt.Var) else result

    def rsample(self, sample_shape=None):
        ''' The reparameterised (pathwise-differentiable) draw.

        Defaults to the undetached ``_sample_impl``, which is the pathwise
        sample for the location-scale and inverse-CDF distributions here. For a
        discrete distribution there is no reparameterisation and the draw
        simply carries no gradient -- torch raises in that case, jittor has
        always returned the draw, and callers rely on that.
        '''
        if type(self)._sample_impl is Distribution._sample_impl:
            raise NotImplementedError(
                "%s implements neither _sample_impl nor rsample"
                % type(self).__name__)
        return self._sample_impl(sample_shape)

    def log_prob(self, value):
        raise NotImplementedError
    def entropy(self):
        raise NotImplementedError

class Independent(Distribution):
    ''' torch.distributions.Independent: reinterpret the last
    `reinterpreted_batch_ndims` batch dims of `base_distribution` as event dims, i.e.
    sum log_prob/entropy over them. Common in RL continuous control:
    Independent(Normal(mu, sigma), 1). '''
    def __init__(self, base_distribution, reinterpreted_batch_ndims):
        self.base_dist = base_distribution
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims

    def _sample_impl(self, sample_shape=None):
        return self.base_dist.sample(sample_shape)

    def rsample(self, sample_shape=None):
        return self.base_dist.rsample(sample_shape) if hasattr(self.base_dist, "rsample") \
            else self.base_dist.sample(sample_shape)

    def log_prob(self, x):
        lp = self.base_dist.log_prob(x)
        for _ in range(self.reinterpreted_batch_ndims):
            lp = lp.sum(-1)
        return lp

    def entropy(self):
        ent = self.base_dist.entropy()
        for _ in range(self.reinterpreted_batch_ndims):
            ent = ent.sum(-1)
        return ent
