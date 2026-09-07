"""Shared errors independent of parsing, backends and the public facade."""


class EinopsError(RuntimeError):
    """ Runtime error thrown by einops """
    pass
