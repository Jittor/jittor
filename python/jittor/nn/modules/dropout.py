"""Stateful dropout modules exposed through :mod:`jittor.nn`."""

import jittor as jt

from ..functional.dropout import _check_probability


class Dropout(jt.Module):
    def __init__(self, p=0.5, is_train=False):
        _check_probability(p)
        self.p = p
        self.is_train = is_train

    def execute(self, input):
        return jt.nn.dropout(input, self.p, self.is_train)


class Dropout2d(jt.Module):
    def __init__(self, p=0.5, is_train=False):
        _check_probability(p)
        self.p = p
        self.is_train = is_train

    def execute(self, input):
        return jt.nn.dropout2d(input, self.p, self.is_train)


class DropPath(jt.Module):
    def __init__(self, p=0.5, is_train=False):
        self.p = p
        self.is_train = is_train

    def execute(self, x):
        return jt.nn.droppath(x, self.p, self.is_train)


__all__ = ["Dropout", "Dropout2d", "DropPath"]
