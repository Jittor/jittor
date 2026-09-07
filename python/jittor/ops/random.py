"""Random tensor operations."""

import numpy as np
import time
from jittor_core import Var

def bernoulli(input):
    import jittor as jt
    return (input>jt.rand_like(input)).cast(input.dtype)


def arange(start=0, end=None, step=1,dtype=None):
    import jittor as jt
    if isinstance(start, Var): start = start.item()
    if isinstance(end, Var): end = end.item()
    if isinstance(step, Var): step = step.item()
    if end is None:
        end,start = start,0
    l = round((end-start)//step)+1
    if (l-1)*step+start>=end:
        l-=1
    x = jt.index((l,),0)
    x = x*step+start
    if dtype is not None:
        x= x.cast(dtype)
    return x


def linspace(start, end, steps):
    import jittor as jt
    if steps > 1:
        res = jt.index((steps,))[0]
        res = res*float((end-start)/(steps-1))+start
    else:
        res = jt.array([start])
    return res


def randperm(n, dtype="int64"):
    ''' A random permutation of ``range(n)``, int64 like ``torch.randperm``.

    int64 because a permutation *is* an index: int32 stops naming an element at
    2**31, and Jittor promotes by byte width, so an int32 permutation kept any
    arithmetic done with it (a flat offset, say) in int32 too.
    '''
    import jittor as jt
    key = jt.random((n,))
    res = jt.argsort(key)
    # jt.argsort may return either the index Var directly or a
    # (index, value) tuple depending on the build; handle both.
    index = res[0] if isinstance(res, (tuple, list)) else res
    return index.cast(dtype)


def set_global_seed(seed, different_seed_for_mpi=True):
    ''' Sets the seeds of the random number generators of Python, numpy and jittor,
    simultaneously.

    This reaches outside jittor on purpose -- it reseeds Python's ``random``,
    numpy's global RNG and (when installed) cupy's. Call it deliberately; it is
    NOT called for you at import. Use :func:`jittor.set_seed` to seed jittor
    alone.

    .. note::
    Jittor also gurantees each worker of jittor.dataset.Dataset to hold a different seed,
    also gurantees each process hold a different seed which using mpi,
    which is (global_seed ^ (worker_id*1167)) ^ 1234 + jt.rank * 2591
    '''
    import jittor as jt
    if (different_seed_for_mpi):
        seed = seed + jt.rank * 2591
    import random
    random.seed(seed)
    jt.set_seed(seed)
    np.random.seed(seed)
    try:
        import cupy
        cupy.random.seed(seed)
    except:
        pass


def _seed_jittor_at_import():
    """Give jittor's own RNG a per-process seed, and touch nothing else.

    This line used to call ``set_global_seed``, so merely importing jittor
    reseeded Python's ``random``, numpy's global RNG and cupy's::

        import numpy as np
        np.random.seed(0)          # the caller's reproducible stream
        import jittor              # ...silently thrown away here

    Nothing was printed and nothing failed; the caller's numpy stream just
    stopped being the one they asked for, and differed on every run because
    the seed jittor substituted came from the wall clock. It also runs at
    ``import jittor`` -- not at first use -- so a library that imports jittor
    somewhere down its own import chain reseeded its user's numpy too.

    jittor does need a per-process seed of its own; it has no business
    reseeding three libraries it does not own. Callers who want all four
    seeded together still say so, with ``jt.set_global_seed(...)``.
    """
    import jittor as jt
    jt.set_seed(int(time.time() * 1000000) % 100000007 + jt.rank * 2591)


def multinomial(weights: Var, num_samples: int, replacement: bool=False) -> Var:
    ''' Returns a var where each row contains num_samples indices sampled from the multinomial probability distribution located in the corresponding row of input weights.

    :param weights: the input probability.
    :param num_samples: number of samples.
    :param replacement: whether to draw with replacement or not.


    Example::

        weights = jt.float32([0, 10, 3, 0])
        x = jt.multinomial(weights, 2)
        assert jt.all_equal(x, [1, 2]) or jt.all_equal(x, [2, 1])
        x = jt.multinomial(weights, 4, replacement=True)
        assert x.shape == (4, )

        weights = jt.float32([[0,0,2],[0,1,0], [0.5,0,0]])
        x = jt.multinomial(weights, 1)
        assert jt.all_equal(x, [[2],[1],[0]])

    '''
    import jittor as jt
    if replacement:
        cum_probs = jt.cumsum(weights)[..., None, :]
        cum_probs_l = cum_probs[..., :-1]
        cum_probs_r = cum_probs[..., 1:]
        shape = weights.shape[:-1] + (num_samples, 1)
        rand = jt.rand(shape) * cum_probs[..., :1, -1:]
        one_hot = jt.logical_and(cum_probs_l < rand, rand <= cum_probs_r)
        index = one_hot.index(one_hot.ndim - 1) + 1
        return (one_hot * index).sum(-1)
    else:
        # A-Res algorithm
        # Pavlos S. Efraimidis and Paul G. Spirakis, 2006, Weighted random sampling with a reservoir
        assert num_samples <= weights.shape[-1], "num_samples larger than the input"
        # Use a strictly positive denominator and mask zero-probability entries
        # after the exponentiation.  The old ``1 / weights`` expression made
        # zero weights produce ``0 ** inf``/NaN keys, allowing an impossible
        # category to win ``topk``.  Positive keys are in (0, 1), so -1 is a
        # stable sentinel for masked entries.
        safe_weights = weights.maximum(1e-20)
        a = jt.rand(weights.shape).minimum(0.999999).maximum(1e-7)
        rand = a ** (1 / safe_weights)
        rand = jt.ternary(weights > 0, rand, jt.full_like(rand, -1.0))
        _, indices = jt.topk(rand, num_samples)
        return indices


def histc(input, bins, min=0., max=0.):
    ''' Return the histogram of the input N-d array.

    :param input: the input array.
    :param bins: number of bins.
    :param min: min of the range.
    :param max: max of the range.

    Example::

        inputs = jt.randn((40,40))
        joup = jt.histc(x, bins=10)

    '''
    import jittor as jt
    if min == 0 and max == 0:
        min, max = input.min(), input.max()
    assert min < max
    if bins <= 0:
        raise RuntimeError(f"bins must be > 0, but got {bins}")
    bin_length = (max - min) / bins
    histc = jt.floor((input[jt.logical_and(input >= min, input < max)] - min) / bin_length).int().reshape(-1)
    hist = jt.ones_like(histc).float().reindex_reduce("add", [bins,], ["@e0(i0)"], extras=[histc])
    hist[-1] += input[input == max].shape[0]
    return hist
