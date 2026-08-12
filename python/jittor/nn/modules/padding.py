"""Stateful padding modules."""

import jittor as jt

from ..functional.padding import pad as pad


class ReflectionPad2d(jt.Module):
    def __init__(self, padding):
        # torch.nn.ReflectionPad2d accepts an int OR a 4-tuple (left,right,top,bottom).
        # The scalar `< 0` guard must not run for tuples (tuple < int -> TypeError);
        # per-side non-negativity is checked below after unpacking.
        if isinstance(padding, int) and padding < 0:
            raise RuntimeError(f"padding must be > 0, but got {padding}")
        self.padding = padding
        if isinstance(self.padding, int):
            self.pl = self.padding
            self.pr = self.padding
            self.pt = self.padding
            self.pb = self.padding
        elif isinstance(self.padding, tuple):
            self.pl, self.pr, self.pt, self.pb = self.padding
        else:
            raise TypeError(f"ReflectionPad2d padding just support int or tuple, but found {type(padding)}")
        if self.pl < 0 or self.pr < 0 or self.pt < 0 or self.pb < 0:
            raise ValueError(f"padding must be non-negative")

    def execute(self, x):
        n,c,h,w = x.shape
        assert (self.pl < w and self.pr < w), f"padding_left and padding_right should be smaller than input width"
        assert (self.pt < h and self.pb < h), f"padding_top and padding_bottom should be smaller than input height"
        oh=h+self.pt+self.pb
        ow=w+self.pl+self.pr
        l = self.pl
        r = self.pl + w - 1
        t = self.pt
        b = self.pt + h - 1
        return x.reindex([n,c,oh,ow], ["i0","i1",
            f"i2<{t} ? {t}-i2 : i2 > {b} ? {h-1+b}-i2 : i2-{t}",
            f"i3<{l} ? {l}-i3 : i3 > {r} ? {w-1+r}-i3 : i3-{l}",
        ])


class ZeroPad2d(jt.Module):
    def __init__(self, padding):
        self.padding = padding
        if isinstance(self.padding, int):
            self.pl = self.padding
            self.pr = self.padding
            self.pt = self.padding
            self.pb = self.padding
        elif isinstance(self.padding, (tuple,list)):
            self.pl, self.pr, self.pt, self.pb = self.padding
        else:
            raise TypeError(f"ZeroPad2d padding just support int or tuple, but found {type(padding)}")
        if self.pl < 0 or self.pr < 0 or self.pt < 0 or self.pb < 0:
            raise ValueError(f"padding must be non-negative")

    def execute(self, x):
        if x.dim() != 4:
            raise RuntimeError("Input shape must be `(N, C, H, W)`!")
        n,c,h,w = x.shape
        return x.reindex([n,c,h+self.pt+self.pb,w+self.pl+self.pr], ["i0","i1",f"i2-{self.pt}",f"i3-{self.pl}"])


class ConstantPad2d(jt.Module):
    def __init__(self, padding, value):
        self.padding = padding
        if isinstance(self.padding, int):
            self.pl = self.padding
            self.pr = self.padding
            self.pt = self.padding
            self.pb = self.padding
        elif isinstance(self.padding, tuple):
            self.pl, self.pr, self.pt, self.pb = self.padding
        else:
            raise TypeError(f"ConstantPad2d padding just support int or tuple, but found {type(padding)}")
        self.value = value
        if self.pl < 0 or self.pr < 0 or self.pt < 0 or self.pb < 0:
            raise ValueError(f"padding must be non-negative")

    def execute(self, x):
        assert len(x.shape) >= 2
        shape = x.shape
        tar_shape = shape[0:-2] + [shape[-2]+self.pt+self.pb,shape[-1]+self.pl+self.pr]
        tar_dims = []
        for i in range(len(shape)-2):
            tar_dims.append(f"i{i}")
        tar_dims.append(f"i{i+1}-{self.pt}")
        tar_dims.append(f"i{i+2}-{self.pl}")
        return x.reindex(tar_shape, tar_dims, overflow_value=self.value)


class ConstantPad1d(jt.Module):
    '''Pads the last dim with a constant. torch: ConstantPad1d((left, right), value)
    (canine downsamples char->molecule sequences with this).'''
    def __init__(self, padding, value):
        if isinstance(padding, int):
            self.pl = self.pr = padding
        elif isinstance(padding, (tuple, list)):
            self.pl, self.pr = padding
        else:
            raise TypeError(f"ConstantPad1d padding just support int or tuple, but found {type(padding)}")
        self.value = value
        if self.pl < 0 or self.pr < 0:
            raise ValueError("padding must be non-negative")

    def execute(self, x):
        assert len(x.shape) >= 1
        shape = x.shape
        n = len(shape)
        tar_shape = shape[:-1] + [shape[-1] + self.pl + self.pr]
        tar_dims = [f"i{i}" for i in range(n - 1)]
        tar_dims.append(f"i{n-1}-{self.pl}")
        return x.reindex(tar_shape, tar_dims, overflow_value=self.value)


class ConstantPad3d(jt.Module):
    '''Pads the last 3 dims with a constant. torch:
    ConstantPad3d((left, right, top, bottom, front, back), value).'''
    def __init__(self, padding, value):
        if isinstance(padding, int):
            self.pl = self.pr = self.pt = self.pb = self.pf = self.pba = padding
        elif isinstance(padding, (tuple, list)):
            self.pl, self.pr, self.pt, self.pb, self.pf, self.pba = padding
        else:
            raise TypeError(f"ConstantPad3d padding just support int or tuple, but found {type(padding)}")
        self.value = value
        if min(self.pl, self.pr, self.pt, self.pb, self.pf, self.pba) < 0:
            raise ValueError("padding must be non-negative")

    def execute(self, x):
        assert len(x.shape) >= 3
        shape = x.shape
        n = len(shape)
        tar_shape = shape[:-3] + [shape[-3] + self.pf + self.pba,
                                  shape[-2] + self.pt + self.pb,
                                  shape[-1] + self.pl + self.pr]
        tar_dims = [f"i{i}" for i in range(n - 3)]
        tar_dims.append(f"i{n-3}-{self.pf}")
        tar_dims.append(f"i{n-2}-{self.pt}")
        tar_dims.append(f"i{n-1}-{self.pl}")
        return x.reindex(tar_shape, tar_dims, overflow_value=self.value)


class ReplicationPad2d(jt.Module):
    def __init__(self, padding):
        # torch.nn.ReplicationPad2d accepts an int OR a 4-tuple (left,right,top,bottom).
        # The scalar `< 0` guard must not run for tuples (tuple < int -> TypeError);
        # per-side non-negativity is checked below after unpacking.
        if isinstance(padding, int) and padding < 0:
            raise RuntimeError(f"padding must be > 0, but got {padding}")
        self.padding = padding
        if isinstance(self.padding, int):
            self.pl = self.padding
            self.pr = self.padding
            self.pt = self.padding
            self.pb = self.padding
        elif isinstance(self.padding, tuple):
            self.pl, self.pr, self.pt, self.pb = self.padding
        else:
            raise TypeError(f"ReplicationPad2d padding just support int or tuple, but found {type(padding)}")
        if self.pl < 0 or self.pr < 0 or self.pt < 0 or self.pb < 0:
            raise ValueError(f"padding must be non-negative")

    def execute(self, x):
        if x.dim() != 4:
            raise RuntimeError("Input shape must be `(N, C, H, W)`!")
        n,c,h,w = x.shape
        oh=h+self.pt+self.pb
        ow=w+self.pl+self.pr
        l = self.pl
        r = self.pl + w - 1
        t = self.pt
        b = self.pt + h - 1
        return x.reindex([n,c,oh,ow], ["i0","i1",
            f"i2<{t} ? 0 : i2 > {b} ? {h-1} : i2-{t}",
            f"i3<{l} ? 0 : i3 > {r} ? {w-1} : i3-{l}"
        ])
