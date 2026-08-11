"""Legacy adaptive pooling modules for two and three dimensions."""

from .runtime import jt, preserve_facade_origins


class AdaptiveAvgPool2d(jt.Module):
    def __init__(self, output_size):
        self.output_size = output_size

    def execute(self, x):
        if isinstance(self.output_size, int):
            oh = self.output_size
            ow = self.output_size
        elif hasattr(self.output_size, "__len__") and not isinstance(self.output_size, str):
            # tuple / list / jittor NanoVector (e.g. x.shape[2:] from a semantic head)
            oh = x.shape[2] if self.output_size[0] is None else int(self.output_size[0])
            ow = x.shape[3] if self.output_size[1] is None else int(self.output_size[1])
        else:
            raise TypeError(f"AdaptiveAvgPool2d only support int, tuple or list input. Not support {type(self.output_size)} yet.")
        if oh == 1 and ow == 1:
            return x.reduce("mean", [2,3], keepdims=True)
        N,C,H,W = x.shape
        self.sh = jt.pool.math.floor(H / oh)
        self.sw = jt.pool.math.floor(W / ow)
        self.ksh = H - (oh - 1) * self.sh
        self.ksw = W - (ow - 1) * self.sw
        h = (H-self.ksh)//self.sh+1
        w = (W-self.ksw)//self.sw+1
        xx = x.reindex([N,C,h,w,self.ksh,self.ksw], [
            "i0", # Nid
            "i1", # Cid
            f"i2*{self.sh}+i4", # Hid
            f"i3*{self.sw}+i5", # Wid
        ])
        return xx.reduce("mean", [4,5])


class AdaptiveMaxPool2d(jt.Module):
    def __init__(self, output_size, return_indices=False):
        self.output_size = output_size
        self.return_indices = return_indices

    def execute(self, x):
        if isinstance(self.output_size, int):
            oh = self.output_size
            ow = self.output_size
        elif hasattr(self.output_size, "__len__") and not isinstance(self.output_size, str):
            # tuple / list / jittor NanoVector (e.g. x.shape[2:] from a semantic head)
            oh = x.shape[2] if self.output_size[0] is None else int(self.output_size[0])
            ow = x.shape[3] if self.output_size[1] is None else int(self.output_size[1])
        else:
            raise TypeError(f"AdaptiveMaxPool2d only support int, tuple or list input. Not support {type(self.output_size)} yet.")
        if oh == 1 and ow == 1:
            return x.reduce("maximum", [2,3], keepdims=True)
        N,C,H,W = x.shape
        self.sh = jt.pool.math.floor(H / oh)
        self.sw = jt.pool.math.floor(W / ow)
        self.ksh = H - (oh - 1) * self.sh
        self.ksw = W - (ow - 1) * self.sw
        if self.return_indices:
            return jt.pool.MaxPool2d(
                kernel_size=(self.ksh, self.ksw),
                stride=(self.sh, self.sw), return_indices=True)(x)
        h = (H-self.ksh)//self.sh+1
        w = (W-self.ksw)//self.sw+1
        xx = x.reindex([N,C,h,w,self.ksh,self.ksw], [
            "i0", # Nid
            "i1", # Cid
            f"i2*{self.sh}+i4", # Hid
            f"i3*{self.sw}+i5", # Wid
        ])
        return xx.reduce("maximum", [4,5])


class AdaptiveAvgPool3d(jt.Module):
    def __init__(self, output_size):
        self.output_size = jt.pool._triple(output_size)

    def execute(self, x):
        od, oh, ow = self.output_size
        if od == 1 and oh == 1 and ow == 1:
            return x.reduce("mean", [2,3,4], keepdims=True)
        N,C,D,H,W = x.shape
        self.sd = jt.pool.math.floor(D / od)
        self.sh = jt.pool.math.floor(H / oh)
        self.sw = jt.pool.math.floor(W / ow)
        self.ksd = D - (od - 1) * self.sd
        self.ksh = H - (oh - 1) * self.sh
        self.ksw = W - (ow - 1) * self.sw
        d = (D-self.ksd)//self.sd+1
        h = (H-self.ksh)//self.sh+1
        w = (W-self.ksw)//self.sw+1
        xx = x.reindex([N,C,d,h,w,self.ksd,self.ksh,self.ksw], [
            "i0", # Nid
            "i1", # Cid
            f"i2*{self.sd}+i5", # Did
            f"i3*{self.sh}+i6", # Hid
            f"i4*{self.sw}+i7", # Wid
        ])
        return xx.reduce("mean", [5,6,7])


class AdaptiveMaxPool3d(jt.Module):
    def __init__(self, output_size, return_indices=False):
        self.output_size = jt.pool._triple(output_size)
        self.return_indices = return_indices

    def execute(self, x):
        od, oh, ow = self.output_size
        if od == 1 and oh == 1 and ow == 1 and not self.return_indices:
            return x.reduce("maximum", [2,3,4], keepdims=True)
        N,C,D,H,W = x.shape
        self.sd = jt.pool.math.floor(D / od)
        self.sh = jt.pool.math.floor(H / oh)
        self.sw = jt.pool.math.floor(W / ow)
        self.ksd = D - (od - 1) * self.sd
        self.ksh = H - (oh - 1) * self.sh
        self.ksw = W - (ow - 1) * self.sw
        if self.return_indices:
            return jt.pool.MaxPool3d(
                kernel_size=(self.ksd, self.ksh, self.ksw),
                stride=(self.sd, self.sh, self.sw), return_indices=True)(x)
        d = (D-self.ksd)//self.sd+1
        h = (H-self.ksh)//self.sh+1
        w = (W-self.ksw)//self.sw+1
        xx = x.reindex([N,C,d,h,w,self.ksd,self.ksh,self.ksw], [
            "i0", # Nid
            "i1", # Cid
            f"i2*{self.sd}+i5", # Did
            f"i3*{self.sh}+i6", # Hid
            f"i4*{self.sw}+i7", # Wid
        ])
        return xx.reduce("maximum", [5,6,7])


_FACADE_SYMBOLS = (
    AdaptiveAvgPool2d, AdaptiveMaxPool2d, AdaptiveAvgPool3d,
    AdaptiveMaxPool3d,
)
preserve_facade_origins(_FACADE_SYMBOLS)
