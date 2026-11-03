"""Array API indexing regressions moved from the vendored runtime module.

These four original cases execute on NumPy's CPU Array API implementation.
NumPy 2 removed that experimental namespace; report that prerequisite as a skip
rather than pretending these cases exercise Jittor devices.
"""
import pytest

np = pytest.importorskip("numpy.array_api", reason="requires NumPy 1.x Array API")

from jittor.contrib.einops.experimental.indexing import (
    CompositionDecomposition, arange_at_position, einindex,
)

def test_composition_and_decomposition():
    x = np.arange(2 * 3 * 5 * 7)
    x = np.reshape(x, (2, 3, 5, 7))
    comp = CompositionDecomposition(
        decomposed_shape=['a', 'b', 'c', 'd'],
        composed_shape=[['a', 'b'], ['c', 'd']],
    )
    assert comp.compose(x, known_axes_lengths={}).shape == (2 * 3, 5 * 7)

    y = CompositionDecomposition(
        decomposed_shape=['a', 'b', 'c', 'd'],
        composed_shape=[['a', 'b'], [], ['c', 'd']],
    ).compose(x, {})
    assert y.shape == (2 * 3, 1, 5 * 7)
    assert np.all(np.reshape(x, (-1,)) == np.reshape(y, (-1,)))

    comp = CompositionDecomposition(
        decomposed_shape=['a', 'b', 'e', 'c', 'd'],
        composed_shape=[['e', 'c'], ['b'], ['a', 'd']],
    )
    x = np.arange(2 * 3 * 5 * 7 * 3)
    x = np.reshape(x, (2, 3, 5, 7, 3))

    axes = {}
    y = comp.compose(x, axes)
    x2 = comp.decompose(y, axes)
    assert np.all(x == x2)


def test_simple_indexing():

    # simple 2d test
    arr = np.reshape(np.arange(5 * 7), (5, 7))
    ind = np.arange(7) % 5
    x = einindex('j <- i j, [i] j', arr, [ind])
    for j, i in enumerate(ind):
        assert arr[i, j] == x[j]

    y = einindex('j <- j i, [i] j', np.permute_dims(arr, (1, 0)), [ind])
    for j, i in enumerate(ind):
        assert arr[i, j] == y[j]


def test_multidimensional_indexing():

    embedding_bhwc = (
            + arange_at_position(np, 4, 0, 2) * 1000
            + arange_at_position(np, 4, 1, 3) * 100
            + arange_at_position(np, 4, 2, 5) * 10
            + arange_at_position(np, 4, 3, 7) * 1
    )

    hindices_bt = np.reshape(np.arange(6), (2, 3)) % 3
    windices_bt = np.reshape(np.arange(6), (2, 3)) % 5

    # imagine that you have pairs of image <> sentence
    # your goal is to get most suitable token from image for every token in sentence
    # thus for every token in sentence you compute best k and v

    result = einindex('c t b <- b h w c, [h, w] b t', embedding_bhwc, [hindices_bt, windices_bt])
    # example of using a single array for indexing multiple axes
    hw_indices_bt = np.stack([hindices_bt, windices_bt])
    result2 = einindex('c t b <- b h w c, [h, w] b t', embedding_bhwc, hw_indices_bt)
    assert np.all(result == result2)

    # check vs manual element computation
    result_manual = result * 0
    for b in range(2):
        for t in range(3):
            for c in range(7):
                h = hindices_bt[b, t]
                w = windices_bt[b, t]
                result_manual[c, t, b] = embedding_bhwc[b, h, w, c]

    assert np.all(result == result_manual)


def test_reverse_indexing():

    C, T, B = 2, 3, 5
    # G = GPU, batch-like varaible
    G = 4
    H = 7
    W = 9

    arr_gtbc = (
            + arange_at_position(np, 4, 0, G) * 1000
            + arange_at_position(np, 4, 1, T) * 100
            + arange_at_position(np, 4, 2, B) * 10
            + arange_at_position(np, 4, 3, C) * 1
    )

    t_indices_gbhw = np.reshape(np.arange(G * B * H * W), (G, B, H, W)) % T

    result = einindex('g b c h w <- g t b c, [t] g b h w', arr_gtbc, [t_indices_gbhw])

    result_manual = result * 0
    for g in range(G):
        for b in range(B):
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        t = t_indices_gbhw[g, b, h, w]
                        result_manual[g, b, c, h, w] = arr_gtbc[g, t, b, c]

    assert np.all(result == result_manual)
