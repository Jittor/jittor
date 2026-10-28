# Copyright (c) Meta Platforms, Inc. and affiliates.
# Exact helper from TensorDict 0.10.0 (MIT), tensordict/utils.py.
from typing import Type


def _rebuild_njt_from_njt(x, values, offsets, lengths):
    from torch._subclasses.fake_tensor import FakeTensor
    from torch._subclasses.functional_tensor import FunctionalTensor
    from torch.nested._internal.nested_tensor import (
        _tensor_symint_registry,
        NestedTensor,
    )
    from torch.nested._internal.ops import extract_kwargs

    kwargs = extract_kwargs(x)
    kwargs["offsets"] = offsets
    if x._lengths is not None:
        kwargs["lengths"] = lengths
        ragged_source = x._lengths
    else:
        ragged_source = x._offsets
    new_thing = kwargs.get("lengths", kwargs.get("offsets"))
    if isinstance(new_thing, (FakeTensor, FunctionalTensor)):
        from torch._subclasses.functional_tensor import mb_unwrap_functional_tensor

        # Temporary hack until we have the union find
        tgt = mb_unwrap_functional_tensor(new_thing)
        src = mb_unwrap_functional_tensor(ragged_source)
        tgt.nested_int_memo = src.nested_int_memo
    elif new_thing is not None:
        _tensor_symint_registry[new_thing] = _tensor_symint_registry[ragged_source]

    return NestedTensor(
        values,
        **kwargs,
    )


# Exact predicate from the same release, tensordict/base.py.
def _is_leaf_nontensor(cls: Type) -> bool:
    """Returns ``True`` if a type is not a tensor collection (tensordict or tensorclass) or is a non-tensor.

    Examples:
        >>> from tensordict import TensorDict, default_is_leaf
        >>> import torch
        >>> td = TensorDict(a={}, b="a string!", c=torch.randn(()))
        >>> print(td.keys(leaves_only=True, is_leaf=default_is_leaf))
        _TensorDictKeysView(['b', 'c'],
            include_nested=False,
            leaves_only=True)

    .. seealso:: :meth:`~tensordict.default_is_leaf`.
    """
    if _is_tensor_collection(cls):
        return _pass_through_cls(cls)
    return issubclass(cls, torch.Tensor)
