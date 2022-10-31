"""
Backends in `einops` are organized to meet the following requirements
- backends are not imported unless those are actually needed, because
    - backends may not be installed
    - importing all available backends will drive to significant memory footprint
    - backends may by present but installed with errors (but never used),
      importing may drive to crashes
- backend should be either symbolic or imperative (tensorflow is for both, but that causes problems)
    - this determines which methods (from_numpy/to_numpy or create_symbol/eval_symbol) should be defined
- if backend can't (temporarily) provide symbols for shape dimensions, UnknownSize objects are used
"""

import sys
import warnings

__author__ = 'Alex Rogozhnikov, RuiYang Liu'

_backends = {}
_debug_importing = False


def get_backend(tensor) -> 'AbstractBackend':
    """
    Takes a correct backend (e.g. numpy backend if tensor is numpy.ndarray) for a tensor.
    If needed, imports package and creates backend
    """
    for framework_name, backend in _backends.items():
        if backend.is_appropriate_type(tensor):
            return backend

    # Find backend subclasses recursively
    backend_subclasses = []
    backends = AbstractBackend.__subclasses__()
    while backends:
        backend = backends.pop()
        backends += backend.__subclasses__()
        backend_subclasses.append(backend)

    for BackendSubclass in backend_subclasses:
        if _debug_importing:
            print('Testing for subclass of ', BackendSubclass)
        if BackendSubclass.framework_name not in _backends:
            # check that module was already imported. Otherwise it can't be imported
            if BackendSubclass.framework_name in sys.modules:
                if _debug_importing:
                    print('Imported backend for ', BackendSubclass.framework_name)
                backend = BackendSubclass()
                _backends[backend.framework_name] = backend
                if backend.is_appropriate_type(tensor):
                    return backend

    raise RuntimeError('Tensor type unknown to einops {}'.format(type(tensor)))


class AbstractBackend:
    """ Base backend class, major part of methods are only for debugging purposes. """
    framework_name = None

    def is_appropriate_type(self, tensor):
        """ helper method should recognize tensors it can handle """
        raise NotImplementedError()

    def from_numpy(self, x):
        raise NotImplementedError("framework doesn't support imperative execution")

    def to_numpy(self, x):
        raise NotImplementedError("framework doesn't support imperative execution")

    def create_symbol(self, shape):
        raise NotImplementedError("framework doesn't support symbolic computations")

    def eval_symbol(self, symbol, input_dict):
        raise NotImplementedError("framework doesn't support symbolic computations")

    def arange(self, start, stop):
        # supplementary method used only in testing, so should implement CPU version
        raise NotImplementedError("framework doesn't implement arange")

    def shape(self, x):
        """shape should return a tuple with integers or "shape symbols" (which will evaluate to actual size)"""
        return x.shape

    def reshape(self, x, shape):
        return x.reshape(shape)

    def transpose(self, x, axes):
        return x.transpose(axes)

    def reduce(self, x, operation, axes):
        return getattr(x, operation)(axis=axes)

    def stack_on_zeroth_dimension(self, tensors: list):
        raise NotImplementedError()

    def add_axis(self, x, new_position):
        raise NotImplementedError()

    def add_axes(self, x, n_axes, pos2len):
        repeats = [1] * n_axes
        for axis_position, axis_length in pos2len.items():
            x = self.add_axis(x, axis_position)
            repeats[axis_position] = axis_length
        return self.tile(x, tuple(repeats))

    def tile(self, x, repeats):
        """repeats is a number of  """
        raise NotImplementedError()

    def is_float_type(self, x):
        # Decided to drop average for all backends if type is not floating
        raise NotImplementedError()

    def layers(self):
        raise NotImplementedError("backend does not provide layers")

    def __repr__(self):
        return "<einops backend for {}>".format(self.framework_name)

    def einsum(self, pattern, *x):
        raise NotImplementedError("backend does not support einsum")


class UnknownSize:
    """ pseudo-symbol for symbolic frameworks which do not provide symbols for shape elements """

    def __floordiv__(self, other):
        return self

    def __eq__(self, other):
        return True  # we don't know actual size

    def __mul__(self, other):
        return self

    def __rmul__(self, other):
        return self

    def __hash__(self):
        return None.__hash__()


class NumpyBackend(AbstractBackend):
    framework_name = 'numpy'

    def __init__(self):
        import numpy
        self.np = numpy

    def is_appropriate_type(self, tensor):
        return isinstance(tensor, self.np.ndarray)

    def from_numpy(self, x):
        return x

    def to_numpy(self, x):
        return x

    def arange(self, start, stop):
        return self.np.arange(start, stop)

    def stack_on_zeroth_dimension(self, tensors: list):
        return self.np.stack(tensors)

    def tile(self, x, repeats):
        return self.np.tile(x, repeats)

    def is_float_type(self, x):
        return x.dtype in ('float16', 'float32', 'float64', 'float128', 'bfloat16')

    def add_axis(self, x, new_position):
        return self.np.expand_dims(x, new_position)
    
    def einsum(self, pattern, *x):
        return self.np.einsum(pattern, *x)


class HashableTuple:
    """Overcomes non-hashability of symbolic elements"""

    def __init__(self, elements: tuple):
        self.elements = elements

    def __iter__(self):
        for x in self.elements:
            yield x

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, item):
        return self.elements[item]

class JittorBackend(AbstractBackend):
    framework_name = 'jittor'

    def __init__(self):
        import jittor
        self.jittor = jittor

    def is_appropriate_type(self, tensor):
        return isinstance(tensor, self.jittor.Var)

    def from_numpy(self, x):
        variable = self.jittor.array(x)
        return variable

    def to_numpy(self, x):
        return x.detach().numpy()

    def arange(self, start, stop):
        return self.jittor.arange(start, stop, dtype='int64')

    def shape(self, x):
        return tuple(x.shape)

    def reshape(self, x, shape):
        if len(shape) == 0:
            return x
        return self.jittor.reshape(x, shape)
    
    def reduce(self, x, operation, reduced_axes):
        
        if operation == 'prod':
            #avoid overflow
            return x.prod(reduced_axes)
        for axis in sorted(reduced_axes, reverse=True):
            if operation == 'min':
                x = x.min(dim=axis)
            elif operation == 'max':
                x = x.max(dim=axis)
            elif operation in ['sum', 'mean']:
                x = getattr(x, operation)(dim=axis)
            else:
                raise NotImplementedError('Unknown reduction ', operation)
        return x

    def transpose(self, x, axes):
        return x.permute(axes)

    def stack_on_zeroth_dimension(self, tensors: list):
        return self.jittor.stack(tensors)

    def add_axes(self, x, n_axes, pos2len):
        repeats = [-1] * n_axes
        for axis_position, axis_length in pos2len.items():
            x = self.add_axis(x, axis_position)
            repeats[axis_position] = axis_length
        return x.expand(repeats)

    def tile(self, x, repeats):
        return x.repeat(repeats)

    def add_axis(self, x, new_position):
        return self.jittor.unsqueeze(x, new_position)

    def is_float_type(self, x):
        return x.dtype in ["float16", "float32", "float64"]

    def layers(self):
        from jittor.einops.layers import jittor
        return jittor
    
    def einsum(self, pattern, *x):
        return self.jittor.linalg.einsum(pattern, *x)