class EinopsError(RuntimeError):
    """ Runtime error thrown by einops """
    pass


__all__ = ['rearrange', 'reduce', 'repeat', 'parse_shape', 'asnumpy', 'EinopsError']

from jittor.einops.einops import rearrange, reduce, repeat, parse_shape, asnumpy
