"""

Indexing one array with the other(s).

Concept for discussion.

Notation targets hard cases, not simple ones, like indexing of 1d-array with another 1d-array
(notation supports that, but you can't simplify arr[ind], and there is no reason to)

Examples

1. query for every token in sequence a token in the image. Images and sequences are paired
   einindex('b t c <- b h w c, [h, w] b t', arr_bhwc, [h_indices_bt, w_indices_bt])

   this is equivalent, so you can pass indexers idependently or together
   einindex('b t c <- b h w c, [h, w] b t', arr_bhwc, np.asarray([h_indices_bt, w_indices_bt]))

   after some thinking I decided that having first axis for indexing variable is not too restrictive,
   but should simplify mapping of such cases.
   For this reason [...] part should always go first in indexer.

   This makes the largest difference with einindex https://github.com/malmaud/einindex,
   which has almost identical grammar, but puts special dimension last, while we put it first.
   This trick allows naturally decomposing multiindex into individual dimensions or visa versa.


2. query for every token in the video the most suitable word in a (matching) sentence
   einindex('b t h w <- seq b, [seq] t b h w', arr_tbc, [t_indices_bhw])

   note, that only one indexer is used, but still it has to be enclosed in the list.
   That's a price for being generic. Alternatively leading singleton dimension can be added.


3. (not supported now, future planning)
   for every timeframe in a video, find the token with the highest norm (across h and w), and compose a new stack of them
   indices_2bt = argmax(x_bthwc.norm(dim=-1), 'b t h w -> [h, w] b t')
   selected_embeddings_btc = einindex('b t c <- b t h w c, [h, w] b t', x_bthwc, indices_2bt)

   while currently question is around 'how do we index',
   it is important to pre-align that with a question 'what are natural ways to get indices'.
   Most common are min/max. less common options: topk (works here), random sampling.



Some important properties of this notation:
- support for multiple indexers, including using a single tensor to keep multiple indexers
- 'batch' indexing, when some axes of indexer and array should be matched
- universal (one-indexing-to-rule-them-all)
- extensible for (named) ellipses, including variadic number of indexers
- extensible for einops-style compositions and decompositions
- extensible for outer indexing when indexers are not aligned

Current implementation based on python array api and uses loops,
because no appropriate indexing available in the standard.

"""

from __future__ import annotations

from typing import List, Union, TypeVar, Tuple

from jittor.contrib.einops._errors import EinopsError

T = TypeVar('T')


class CompositionDecomposition:
    def __init__(
            self,
            decomposed_shape: List[str],
            composed_shape: List[List[str]],
    ):
        flat_shape = []
        for x in composed_shape:
            flat_shape.extend(x)

        self.compose_transposition: Tuple[int] = tuple([decomposed_shape.index(x) for x in flat_shape])
        self.decompose_transposition: Tuple[int] = tuple([flat_shape.index(x) for x in decomposed_shape])
        self.composed_shape = composed_shape
        self.decomposed_shape = decomposed_shape

    def decompose(self, x, known_axes_lengths: dict[str, int]):
        xp = x.__array_namespace__()
        shape = x.shape

        flat_shape = []

        for i, axis_group in enumerate(self.composed_shape):
            unknown_axis_name = None
            known_sizes_prod = 1
            for axis_name in axis_group:
                if axis_name in known_axes_lengths:
                    known_sizes_prod *= known_axes_lengths[axis_name]
                else:
                    if unknown_axis_name is None:
                        unknown_axis_name = axis_name
                    else:
                        raise EinopsError("Can't infer the size")

            if unknown_axis_name is None:
                assert shape[i] == known_sizes_prod
            else:
                known_axes_lengths[unknown_axis_name] = shape[i] // known_sizes_prod

            for axis in axis_group:
                flat_shape.append(known_axes_lengths[axis])

        x = xp.reshape(x, flat_shape)
        return xp.permute_dims(x, self.decompose_transposition)

    def compose(self, x, known_axes_lengths: dict[str, int]):
        xp = x.__array_namespace__()

        for axis_len, axis_name in zip(x.shape, self.decomposed_shape):
            if axis_name in known_axes_lengths:
                assert known_axes_lengths[axis_name] == axis_len
            else:
                known_axes_lengths[axis_name] = axis_len

        x = xp.permute_dims(x, self.compose_transposition)
        new_shape = []
        for axis_group in self.composed_shape:
            composed_axis_size = 1
            for axis_name in axis_group:
                composed_axis_size *= known_axes_lengths[axis_name]
            new_shape.append(composed_axis_size)

        return xp.reshape(x, tuple(new_shape))


def arange_at_position(xp, n_axes, axis, axis_len, device=None):
    x = xp.arange(axis_len, dtype=xp.int64, device=device)
    shape = [1] * n_axes
    shape[axis] = axis_len
    x = xp.reshape(x, shape)
    return x


class IndexingFormula:

    def __init__(self, pattern: str):
        """
        :param pattern: example 'b t c <- b hsel wsel c, [hsel, wsel] b t'
        """
        self.pattern = pattern
        left, right = pattern.split('<-')
        arg_split = right.index(',')
        arr_pattern, ind_pattern = right[:arg_split], right[arg_split + 1:]
        ind_pattern = ind_pattern.strip()
        # print(
        #     arr_pattern, '\n',
        #     ind_pattern,
        # )
        assert ind_pattern.startswith('['), 'composition axis should go first in indexer (second argument) [h w] i j k'
        composition_start = ind_pattern.index('[')
        composition_end = ind_pattern.index(']')
        composition = ind_pattern[composition_start + 1: composition_end]
        ind_other_axes = ind_pattern[composition_end + 1:]

        self.result_axes_names = left.split()
        self.array_axes_names = arr_pattern.split()
        self.indexing_axes_names = [x.strip() for x in composition.split(',')]
        self.indexer_other_axes_names = ind_other_axes.split()

        for group_name, group in [
            ('result', self.result_axes_names),
            ('array', self.array_axes_names),
            ('indexer', self.indexing_axes_names + self.indexer_other_axes_names),
        ]:
            if len(set(group)) != len(group):
                # need more verbosity, which axis, raise
                raise EinopsError(f'{group_name} pattern ({group}) contains a duplicated axis')

        axis_groups = [
            self.result_axes_names,
            self.array_axes_names,
            self.indexing_axes_names,
            self.indexer_other_axes_names,
        ]

        all_axes = set()
        for group in axis_groups:
            all_axes.update(group)

        self.indexer_axes = []
        self.batch_axes = []
        self.result_and_index_axes = []
        self.result_and_array_axes = []

        for axis in all_axes:
            presence = tuple(axis in g for g in axis_groups)
            # want match-case here. sweet dreams
            if presence == (False, True, True, False):
                self.indexer_axes.append(axis)
            elif presence[2]:
                raise EinopsError(f'Wrong usage of indexer variable {axis}')
            elif presence == (True, True, False, True):
                self.batch_axes.append(axis)
            elif presence == (True, False, False, True):
                self.result_and_index_axes.append(axis)
            elif presence == (True, True, False, False):
                self.result_and_array_axes.append(axis)
            else:
                # TODO better categorization of wrong usage patterns
                raise EinopsError(f'{axis} is used incorrectly in {pattern}')

        assert set(self.indexer_axes) == set(self.indexing_axes_names)
        # order of these variables matters, since we can't lose mapping here
        self.indexer_axes = self.indexing_axes_names

        self.array_composition = CompositionDecomposition(
            decomposed_shape=self.array_axes_names,
            composed_shape=[self.batch_axes + self.indexer_axes, self.result_and_array_axes],
        )

        self.index_composition = CompositionDecomposition(
            decomposed_shape=self.indexer_other_axes_names,
            # single axis after composition
            composed_shape=[self.batch_axes + self.result_and_index_axes],
        )

        self.result_composition = CompositionDecomposition(
            decomposed_shape=self.result_axes_names,
            composed_shape=[self.batch_axes + self.result_and_index_axes, self.result_and_array_axes],
        )

    def apply_to_array_api(self, arr: T, ind: Union[T, List[T]]):
        known_axes_sizes: dict[str, int] = {}
        xp = arr.__array_namespace__()

        if not isinstance(ind, list):
            ind = [ind[i, ...] for i in range(ind.shape[0])]

        for indexer in ind:
            assert len(indexer.shape) == len(self.indexer_other_axes_names)

        # step 1. transpose, reshapes of arr; learn its dimensions
        arr_2d = self.array_composition.compose(arr, known_axes_sizes)

        # step 2. compute shifts and create an actual indexing array
        shift = 1
        full_index = xp.zeros([1] * len(ind[0].shape), dtype=xp.int64, device=arr.device)

        # original order: [*batch-like axes, *indexing_axes,]
        # now we need to traverse them in the opposite direction

        for axis_name, indexer in list(zip(self.indexing_axes_names, ind))[::-1]:
            full_index = full_index + shift * (indexer % known_axes_sizes[axis_name])
            shift *= known_axes_sizes[axis_name]

        for axis_name in self.batch_axes[::-1]:
            axis_id = self.indexer_other_axes_names.index(axis_name)
            full_index = full_index + arange_at_position(
                xp, len(self.indexer_other_axes_names), axis=axis_id, axis_len=known_axes_sizes[axis_name],
                device=arr.device,
            ) * shift
            shift *= known_axes_sizes[axis_name]

        assert shift == arr_2d.shape[0]

        # step 3. Flatten index
        full_index = self.index_composition.compose(full_index, known_axes_sizes)

        # step 4. indexing
        # python array api lacks any integer indexing, so... I use loops.
        # did you know that there is conceptual programming ... just like art?
        # result_2d = arr_2d[full_index]
        result_2d = xp.stack([arr_2d[full_index[i], :] for i in range(full_index.shape[0])])

        # step 5. doing resulting
        result = self.result_composition.decompose(result_2d, known_axes_sizes)
        return result


def einindex(pattern: str, arr: T, ind: Union[T, List[T]]):
    """
    Demonstrates how einindex should work.
    Supports data-api compliant arrays.
    """
    formula = IndexingFormula(pattern)
    return formula.apply_to_array_api(arr, ind)
