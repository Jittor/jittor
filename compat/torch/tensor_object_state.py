"""Per-tensor compatibility state, owned by the actual Python Tensor holder."""


class TensorObjectState:
    __slots__ = ("grad", "data_owner", "data_path", "scalar_marker")

    def __init__(self):
        self.grad = None
        self.data_owner = None
        self.data_path = ()
        self.scalar_marker = False


_FIELDS = (
    ("_torch_grad", "grad", None),
    ("_torch_data_owner", "data_owner", None),
    ("_torch_data_path", "data_path", ()),
    ("_torch_0d", "scalar_marker", False),
)
_STATE_KEY = "_torch_object_state"


def get_tensor_object_state(tensor, create=False):
    """Return this holder's state; reads do not allocate or retain another Var."""
    attributes = vars(tensor)
    state = attributes.get(_STATE_KEY)
    if state is not None and not isinstance(state, TensorObjectState):
        raise TypeError("Tensor compatibility state must be TensorObjectState")
    if state is None and create:
        state = TensorObjectState()
        # Old pickles carry the individual attributes in the holder dict.
        # Adopt them once, then remove the duplicate references.
        for legacy, field, default in _FIELDS:
            setattr(state, field, attributes.pop(legacy, default))
        attributes[_STATE_KEY] = state
    return state


def _state_property(legacy, field, default):
    def get(tensor):
        state = get_tensor_object_state(tensor)
        if state is None:
            return vars(tensor).get(legacy, default)
        return getattr(state, field)

    def set(tensor, value):
        setattr(get_tensor_object_state(tensor, create=True), field, value)

    def delete(tensor):
        state = get_tensor_object_state(tensor)
        if state is None:
            vars(tensor).pop(legacy, None)
        else:
            setattr(state, field, default)

    return property(get, set, delete)


def tensor_object_properties():
    """Descriptors installed only on the independent Tensor type."""
    return {legacy: _state_property(legacy, field, default)
            for legacy, field, default in _FIELDS}
