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
    """``steps`` values evenly spaced from ``start`` to ``end``, endpoint included.

    The last value is ``end`` exactly, which is what numpy's and torch's
    ``linspace`` promise and what callers compare against. The arithmetic
    series alone does not give it: ``i * (end - start) / (steps - 1) + start``
    accumulates one rounding step on the final point, and on the CPU placement
    ``linspace(1, 0, 4)`` ended at ``-2.98e-08`` -- *below* ``end``.

    MiniMax-H3 builds its sigma schedule as
    ``linspace(1.0, 0.0, num_inference_steps)`` and validates that no sigma is
    negative, so every step count from 4 up died with ``sigma_next must be
    non-negative``. It only showed on the CPU placement because the CUDA path
    happened to round that point to exactly 0.
    """
    import jittor as jt
    if steps > 1:
        res = jt.index((steps,))[0]
        res = res*float((end-start)/(steps-1))+start
        # Pin the endpoint instead of trusting the accumulated rounding.
        res = jt.cat([res[:-1], jt.array([end], dtype=res.dtype)])
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
        if num_samples > weights.shape[-1]:
            raise ValueError("multinomial: num_samples larger than the input")
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
    if min >= max:
        raise ValueError("histc: min must be less than max")
    if bins <= 0:
        raise RuntimeError(f"bins must be > 0, but got {bins}")
    bin_length = (max - min) / bins
    histc = jt.floor((input[jt.logical_and(input >= min, input < max)] - min) / bin_length).int().reshape(-1)
    hist = jt.ones_like(histc).float().reindex_reduce("add", [bins,], ["@e0(i0)"], extras=[histc])
    hist[-1] += input[input == max].shape[0]
    return hist


class CounterGenerator:
    """Independent Philox4x32-10 stream evaluated on the tensor's device.

    Each nonempty call reserves four counter positions; tensor elements occupy
    independent subsequences. Reserving at construction (not execution) makes
    lazy draws independent of evaluation order. The offset is in 32-bit words
    and can be restored to replay a whole draw. CPU and CUDA are supported;
    neither backend draws tensor values on the host for the other backend.

    This is a Jittor stream, not PyTorch's launch-geometry-dependent RNG stream.
    """

    def __init__(self, seed=0):
        import threading
        self._lock = threading.Lock()
        self.manual_seed(seed)

    def __getstate__(self):
        return self.get_state()

    def __setstate__(self, state):
        import threading
        self._lock = threading.Lock()
        self.set_state(state)

    def manual_seed(self, seed):
        self.set_state((seed, 0))
        return self

    def get_state(self):
        with self._lock:
            return self._seed, self._offset

    def set_state(self, state):
        seed, offset = map(int, state)
        if not 0 <= seed < 2**64:
            raise ValueError("seed must be an unsigned 64-bit integer")
        if not 0 <= offset < 2**64 or offset % 4:
            raise ValueError("offset must be an unsigned multiple of four")
        with self._lock:
            self._seed, self._offset = seed, offset
        return self

    def get_offset(self):
        return self.get_state()[1]

    def set_offset(self, offset):
        with self._lock:
            offset = int(offset)
            if not 0 <= offset < 2**64 or offset % 4:
                raise ValueError("offset must be an unsigned multiple of four")
            self._offset = offset
        return self

    def uniform_like(self, like, dtype="float32"):
        """Return device-native uniform samples with ``like``'s shape/device."""
        import jittor as jt
        from .._core.var import device_scope_like
        from .._core.dtypes import dtype_name
        dtype = dtype_name(dtype)
        if dtype not in ("float32", "float64"):
            raise ValueError("CounterGenerator uniform requires float32 or float64")
        if int(like.placement_backend) not in (-1, 0, 1):
            raise NotImplementedError("CounterGenerator supports CPU and CUDA")
        shape = tuple(int(v) for v in like.shape)
        count = int(np.prod(shape))
        if count > 2**31 - 1:
            raise ValueError("CounterGenerator draw exceeds the kernel index range")
        with self._lock:
            seed, offset = self._seed, self._offset
            if count:
                if offset > 2**64 - 8:
                    raise OverflowError("CounterGenerator stream exhausted")
                self._offset += 4
        with device_scope_like(like):
            if not count:
                return jt.empty(shape, dtype=dtype)
            # Only four scalar metadata words cross to the selected device.
            # Capturing them as an immutable input also prevents a later seed
            # reset from changing an already-queued lazy draw.
            params = jt.array([seed & 0xffffffff, seed >> 32,
                               (offset // 4) & 0xffffffff, (offset // 4) >> 32], dtype="int64")
            is_double = dtype == "float64"
            conversion = (
                "uint64 bits = ((uint64)c[(i&1)*2] << 21) | (c[(i&1)*2+1] >> 11);\n"
                "double value = ((double)bits + 0.5) * 1.1102230246251565404236316680908203125e-16;\n"
                "out[i] = value < 1.0 ? value : 0.99999999999999988897769753748434595763683319091796875;"
                if is_double else
                "float value = (float)(((double)c[i&3] + 0.5) * 2.3283064365386962890625e-10);\n"
                "out[i] = value < 1.0f ? value : 0.999999940395355224609375f;"
            )
            header = """
namespace jittor {
@python.jittor.auto_parallel(1)
inline static void counter_uniform(int n, int i, const int64* state, OUT_TYPE* out) {
    uint32 k0 = (uint32)state[0], k1 = (uint32)state[1];
    uint32 c[4] = {(uint32)state[2], (uint32)state[3], (uint32)(i / GROUP), 0};
    for (int round=0; round<10; ++round) {
        uint64 p0 = (uint64)0xD2511F53u * c[0];
        uint64 p1 = (uint64)0xCD9E8D57u * c[2];
        uint32 next0 = (uint32)(p1 >> 32) ^ c[1] ^ k0;
        uint32 next2 = (uint32)(p0 >> 32) ^ c[3] ^ k1;
        c[0] = next0; c[1] = (uint32)p1;
        c[2] = next2; c[3] = (uint32)p0;
        k0 += 0x9E3779B9u; k1 += 0xBB67AE85u;
    }
    CONVERSION
}
}
""".replace("OUT_TYPE", "double" if is_double else "float").replace(
                "GROUP", "2" if is_double else "4").replace("CONVERSION", conversion)
            source = "counter_uniform(out0->num, 0, in0_p, out0_p);"
            return jt.code([count], dtype, [params], cpu_header=header, cpu_src=source,
                           cuda_header=header, cuda_src=source).reshape(shape).stop_grad()
