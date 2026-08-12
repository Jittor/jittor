"""Functional shape operations exposed through :mod:`jittor.nn`."""

from jittor import flatten


def identity(input):
    return input


__all__ = ["flatten", "identity"]
