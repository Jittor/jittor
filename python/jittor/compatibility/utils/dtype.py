from typing import Callable, Union
Dtype = Union[Callable, str]

def get_string_dtype(dtype):
    if callable(dtype):
        dtype = dtype.__name__
    if not isinstance(dtype, str):
        raise ValueError(f"dtype is expected to be str, python type function, or jittor type function, but got {dtype}.")
    return dtype