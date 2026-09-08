"""Native parameter identity and module-owned registration roles."""

import jittor as jt
import numpy as np
import copy
import pickle


def test_native_parameter_is_a_real_subtype_with_live_python_members():
    class Weight(jt.nn.Parameter):
        def __init__(self, data, requires_grad=True):
            super().__init__(data, requires_grad)
            self.tag = 3

        @property
        def scaled_tag(self):
            return self.tag * 2

    source = jt.array([1.0, 2.0])
    weight = Weight(source)
    assert type(weight) is Weight
    assert isinstance(weight, jt.Var)
    assert weight.scaled_tag == 6
    weight.tag = 5
    assert weight.scaled_tag == 10
    assert weight.requires_grad
    assert weight._storage_address == source._storage_address
    np.testing.assert_array_equal((weight + 1).numpy(), [2.0, 3.0])
    assert type(weight + 1) is jt.Var
    duplicate = copy.deepcopy(weight)
    assert type(duplicate) is Weight and duplicate.scaled_tag == 10
    assert duplicate._storage_address != weight._storage_address


def test_native_parameter_pickle_preserves_type_dtype_and_python_state():
    weight = jt.nn.Parameter(jt.array([1.25], dtype="float64"), requires_grad=False)
    weight.tag = "restored"
    restored = pickle.loads(pickle.dumps(weight))
    assert type(restored) is jt.nn.Parameter
    assert restored.tag == "restored" and not restored.requires_grad
    assert str(restored.dtype) == "float64"
    np.testing.assert_array_equal(restored.numpy(), [1.25])


def test_native_parameter_and_buffer_roles_do_not_mutate_tensor_tags():
    module = jt.Module()
    weight = jt.array([1.0])
    module.weight = weight
    assert module.parameters() == [weight]
    assert "_is_torch_parameter" not in vars(weight)
    module.register_buffer("same_storage", weight)
    assert list(module.named_parameters()) == [("weight", weight)]
    assert list(module.named_buffers()) == [("same_storage", weight)]
    assert "_is_torch_parameter" not in vars(weight)
    buffer = jt.array([2.0])
    module.register_buffer("running", buffer)
    module.running_alias = buffer
    assert len(module.parameters()) == 1
    assert "is_buffer" not in vars(buffer)
