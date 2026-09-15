# Author: Vrinda12-tech <your-github-email>
#
# Total coding rate loss for self-supervised learning, ported from EMP-SSL.

import jittor as jt
import jittor.nn as nn

__all__ = ["coding_rate", "TotalCodingRate"]


def coding_rate(z, eps=0.01, normalize=False):
    """Compute the coding rate R(Z) for a batch of embeddings.

    R(Z) = 0.5 * logdet(I_D + (D / (N * eps)) * Z^T Z)

    R(Z) is non-negative and increases with batch diversity. To obtain
    the trainable loss, use :class:`TotalCodingRate` (which returns
    ``-R(Z)``) or negate this value yourself.

    :param z: batch of embeddings, shape (N, D), N >= 2
    :type z: jt.Var
    :param eps: rate-distortion parameter, used directly (not squared)
        to match the EMP-SSL reference implementation. Default: 0.01
    :type eps: float, optional
    :param normalize: if True, L2-normalize rows before computing.
        This is an extension not present in the reference; default False.
    :type normalize: bool, optional

    :return: scalar, R(Z)
    :rtype: jt.Var

    Example:

    >>> import jittor as jt
    >>> from jittor.loss_ssl import coding_rate
    >>> jt.flags.use_cuda = 0
    >>> z = jt.randn(64, 32)
    >>> rate = coding_rate(z)
    """
    if z.ndim != 2:
        raise ValueError(f"coding_rate expects a 2-D (N, D) batch, got shape {tuple(z.shape)}")
    n, d = z.shape[0], z.shape[1]
    if n < 2:
        raise ValueError(f"coding_rate needs at least 2 samples in the batch, got N={n}")
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}")

    if normalize:
        z = z / (jt.norm(z, dim=1, keepdim=True) + 1e-8)

    identity = jt.init.eye(d)
    scalar = d / (n * eps)
    m = identity + scalar * jt.matmul(z.transpose(1, 0), z)
    _sign, logdet = jt.linalg.slogdet(m)
    return 0.5 * logdet


class TotalCodingRate(nn.Module):
    """Total coding rate loss: a batch-level anti-collapse regularizer.

    Ported from EMP-SSL (Tong, Chen, Ma & LeCun, 2023) which in turn
    builds on the coding-rate function of MCR^2 (Yu et al., NeurIPS 2020).
    Intended to be added to a point-wise loss, not to replace it:

        total_loss = pointwise_loss + lambda * TotalCodingRate()(z)

    At default settings (``eps=0.01``, no row-normalization), this matches
    the reference PyTorch implementation exactly. See the PR description
    for the numerical fidelity check and for the ``normalize=True``
    extension, which is off by default.

    :param eps: forwarded to :func:`coding_rate`. Default: 0.01
    :type eps: float, optional
    :param normalize: forwarded to :func:`coding_rate`. Default: False
    :type normalize: bool, optional

    Example:

    >>> import jittor as jt
    >>> from jittor.loss_ssl import TotalCodingRate
    >>> jt.flags.use_cuda = 0
    >>> criterion = TotalCodingRate()
    >>> z = jt.randn(64, 32)
    >>> loss = criterion(z)
    """

    def __init__(self, eps=0.01, normalize=False):
        super().__init__()
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")
        self.eps = eps
        self.normalize = normalize

    def execute(self, z):
        return -coding_rate(z, eps=self.eps, normalize=self.normalize)