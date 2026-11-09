"""Behavior of the small helpers published by torch.nn.modules.utils."""

from collections import OrderedDict

import pytest
import torch.nn.modules.utils as nn_utils


def test_reverse_repeat_tuple_reverses_and_repeats():
    assert nn_utils._reverse_repeat_tuple((1, 2, 3), 2) == (3, 3, 2, 2, 1, 1)


def test_list_with_default_fills_only_none_entries():
    assert nn_utils._list_with_default((None, 5, None), (10, 20, 2, 3, 4)) == [2, 5, 4]


def test_list_with_default_preserves_integer_and_empty_sizes():
    assert nn_utils._list_with_default(4, (1, 2, 3)) == 4
    assert nn_utils._list_with_default((), (1, 2)) == []


@pytest.mark.parametrize("defaults", [(3,), (3, 4)])
def test_list_with_default_rejects_too_many_dimensions(defaults):
    with pytest.raises(ValueError):
        nn_utils._list_with_default((1, 2), defaults)


def test_consume_prefix_updates_keys_and_metadata_in_place():
    state_dict = OrderedDict((
        ("module.weight", 1),
        ("module.layer.bias", 2),
        ("other", 3),
    ))
    state_dict._metadata = OrderedDict((
        ("", {"version": 1}),
        ("module", {"version": 2}),
        ("module.layer", {"version": 3}),
    ))

    result = nn_utils.consume_prefix_in_state_dict_if_present(state_dict, "module.")

    assert result is None
    assert state_dict == OrderedDict((
        ("other", 3),
        ("weight", 1),
        ("layer.bias", 2),
    ))
    assert state_dict._metadata == OrderedDict((
        ("", {"version": 2}),
        ("layer", {"version": 3}),
    ))


def test_consume_prefix_without_trailing_dot_updates_metadata():
    state_dict = OrderedDict((("moduleweight", 1),))
    state_dict._metadata = OrderedDict((("", 1), ("modul", 4), ("module", 2), ("modulelayer", 3)))

    nn_utils.consume_prefix_in_state_dict_if_present(state_dict, "module")

    assert state_dict == OrderedDict((("weight", 1),))
    assert state_dict._metadata == OrderedDict((("", 2), ("modul", 4), ("layer", 3)))
