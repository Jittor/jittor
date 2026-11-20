"""Wishart matrix distributions."""

import math
import warnings

from jittor._core.dtypes import dtype_name
from jittor._core.var import device_scope_like

from ._constraints import constraints
from ._utils import _broadcast_two, _digamma, _lgamma
from .base import Distribution


def _transpose_matrix(x):
    import jittor as jt
    axes = tuple(range(x.ndim - 2)) + (x.ndim - 1, x.ndim - 2)
    return jt.transpose(x, axes)


def _diagonal(x):
    import jittor as jt
    return jt.diagonal(x, dim1=-2, dim2=-1)


class Wishart(Distribution):
    """Wishart distribution with a covariance, precision or Cholesky factor."""

    has_rsample = True
    support = constraints.positive_definite

    def __init__(self, df, covariance_matrix=None, precision_matrix=None,
                 scale_tril=None, validate_args=None):
        import jittor as jt

        matrices = [x for x in (covariance_matrix, precision_matrix, scale_tril) if x is not None]
        if len(matrices) != 1:
            raise ValueError("Specify exactly one covariance_matrix, precision_matrix or scale_tril")
        matrix = matrices[0]
        if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2] or not matrix.shape[-1]:
            raise ValueError("Wishart requires a nonempty square matrix")
        if dtype_name(matrix.dtype) not in ("float32", "float64"):
            raise TypeError("Wishart requires a float32 or float64 matrix")
        rank = matrix.shape[-1]
        with device_scope_like(matrix):
            if not isinstance(df, jt.Var):
                df = jt.full((), df, dtype=matrix.dtype)
            self._offsets = jt.arange(rank, dtype=matrix.dtype)
        if not bool((df > rank - 1).all().item()):
            raise ValueError("Wishart degrees of freedom must be greater than the matrix size minus one")
        batch_shape = _broadcast_two(tuple(matrix.shape[:-2]), tuple(df.shape))
        super().__init__(batch_shape, (rank, rank), validate_args)
        self.df = df.broadcast(batch_shape)
        if scale_tril is None:
            cov = matrix if covariance_matrix is not None else jt.linalg.inv(matrix)
            scale_tril = jt.linalg.cholesky(cov)
        elif validate_args and not bool(((matrix == jt.tril(matrix)).all() &
                                        (_diagonal(matrix) > 0).all()).item()):
            raise ValueError("Wishart scale_tril must be lower triangular with a positive diagonal")
        self.scale_tril = scale_tril.broadcast(batch_shape + self.event_shape)
        self._needs_correction = bool((self.df < rank).any().item())
        if self._needs_correction:
            warnings.warn("Low degrees of freedom can produce numerically singular Wishart samples.",
                          stacklevel=2)

    @property
    def arg_constraints(self):
        return {
            "df": constraints.greater_than(self.event_shape[-1] - 1),
            "covariance_matrix": constraints.positive_definite,
            "precision_matrix": constraints.positive_definite,
            "scale_tril": constraints.lower_cholesky,
        }

    @property
    def covariance_matrix(self):
        return self.scale_tril @ _transpose_matrix(self.scale_tril)

    @property
    def precision_matrix(self):
        import jittor as jt
        return jt.linalg.inv(self.covariance_matrix)

    @property
    def mean(self):
        return self.df.reshape(self.batch_shape + (1, 1)) * self.covariance_matrix

    @property
    def mode(self):
        import jittor as jt
        scale = self.df - self.event_shape[-1] - 1
        scale = jt.where(scale > 0, scale, float("nan"))
        return scale.reshape(self.batch_shape + (1, 1)) * self.covariance_matrix

    @property
    def variance(self):
        covariance = self.covariance_matrix
        diagonal = _diagonal(covariance)
        return self.df.reshape(self.batch_shape + (1, 1)) * (
            covariance.square() + diagonal.unsqueeze(-1) * diagonal.unsqueeze(-2)
        )

    def _gamma_concentration(self):
        return (self.df.unsqueeze(-1) - self._offsets) * 0.5

    def _bartlett_sampling(self, sample_shape=None):
        import jittor as jt
        from jittor.math_util.gamma import sample_gamma

        shape = self._extended_shape(sample_shape)
        scale = self.scale_tril
        with device_scope_like(scale):
            gamma = sample_gamma(self._gamma_concentration(), shape[:-1], dtype=scale.dtype)
            diagonal = jt.maximum(jt.sqrt(2 * gamma), jt.finfo(scale.dtype).eps)
            identity = jt.init.eye(self.event_shape[-1], dtype=scale.dtype)
            noise = jt.tril(jt.randn(shape, dtype=scale.dtype), -1)
        factor = scale @ (noise + diagonal.unsqueeze(-1) * identity)
        return factor @ _transpose_matrix(factor)

    def rsample(self, sample_shape=None, max_try_correction=None):
        import jittor as jt
        sample = self._bartlett_sampling(sample_shape)
        if not self._needs_correction:
            return sample
        tries = 10 if max_try_correction is None else max_try_correction
        for _ in range(tries):
            failed = (jt.linalg.eigvalsh(sample) <= 0).any(-1)
            if not bool(failed.any().item()):
                break
            sample = jt.where(failed.unsqueeze(-1).unsqueeze(-1),
                              self._bartlett_sampling(sample_shape), sample)
        return sample

    def expand(self, batch_shape):
        batch_shape = tuple(batch_shape)
        return type(self)(self.df.broadcast(batch_shape),
                          scale_tril=self.scale_tril.broadcast(batch_shape + self.event_shape),
                          validate_args=self._validate_args)

    def _normalization(self):
        import jittor as jt
        rank = self.event_shape[-1]
        half_log_determinant = jt.log(_diagonal(self.scale_tril)).sum(-1)
        multivariate_lgamma = _lgamma(self._gamma_concentration()).sum(-1)
        multivariate_lgamma += rank * (rank - 1) * math.log(math.pi) / 4
        return rank * math.log(2) / 2 + half_log_determinant, multivariate_lgamma

    def log_prob(self, value):
        import jittor as jt
        if tuple(value.shape[-2:]) != self.event_shape:
            raise ValueError("Wishart log_prob requires a matrix with the distribution's event shape")
        _, logdet = jt.linalg.slogdet(value)
        logdet = logdet.reshape(value.shape[:-2])
        trace = _diagonal(self.precision_matrix @ value).sum(-1)
        scale, normalizer = self._normalization()
        power = (self.df - self.event_shape[-1] - 1) / 2
        return power * logdet - trace / 2 - self.df * scale - normalizer

    def entropy(self):
        rank = self.event_shape[-1]
        scale, normalizer = self._normalization()
        digamma_sum = _digamma(self._gamma_concentration()).sum(-1)
        return ((rank + 1) * scale + normalizer + self.df * rank / 2
                - (self.df - rank - 1) * digamma_sum / 2)
