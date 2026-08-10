"""Transposed convolution layer implementations."""

import math

from .runtime import jt, preserve_facade_origins


class ConvTranspose(jt.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, \
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1):
        self.in_channels = in_channels
        self.out_channels = out_channels

        # added
        self.dilation = dilation
        self.groups = groups

        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        # added
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.real_padding = (self.dilation[0] * (self.kernel_size[0] - 1) - self.padding[0],
            self.dilation[1] * (self.kernel_size[1] - 1) - self.padding[1])
        self.output_padding = output_padding if isinstance (output_padding, tuple) else (output_padding, output_padding)
        assert self.stride[0] > 0 and self.stride[1] > 0,"stride must be positive"
        assert self.padding[0] >= 0 and self.padding[1] >= 0,"padding must be non-negative"
        assert self.output_padding[0] < max(self.stride[0], self.dilation[0]) and \
            self.output_padding[1] < max(self.stride[1], self.dilation[1]), \
            "output padding must be smaller than max(stride, dilation)"
        assert in_channels % groups == 0, 'in_channels must be divisible by groups'
        assert out_channels % groups == 0, 'out_channels must be divisible by groups'

        self.weight = jt.nn.init.invariant_uniform((in_channels, out_channels//groups) + self.kernel_size, dtype="float")
        if bias:
            fan=1
            for i in self.weight.shape[1:]:
                fan *= i
            bound = 1 / math.sqrt(fan)
            self.bias = jt.nn.init.uniform([out_channels], dtype="float", low=-bound, high=bound)
        else:
            self.bias = None

    def execute(self, x):
        if x.dim() != 4:
            raise RuntimeError(f'Expected 4D (batched) input to conv_transpose2d, but got input of size: {x.shape}')
        if self.groups == 1:
            N,C,H,W = x.shape
            i,o,h,w = self.weight.shape
            assert C==i
            stride_h, stride_w = self.stride
            padding_h, padding_w = self.padding
            dilation_h, dilation_w = self.dilation

            h_out = (H-1) * stride_h + self.output_padding[0] - 2*padding_h + 1 + (h-1)*dilation_h
            w_out = (W-1) * stride_w + self.output_padding[1] - 2*padding_w + 1 + (w-1)*dilation_w
            out_shape = (N, o, h_out, w_out)
            shape = (N, i, o, H, W, h, w)
            xx = x.broadcast(shape, (2, 5, 6)) # i,h,w
            ww = self.weight.broadcast(shape, (0, 3, 4)) # N,H,W
            y = (ww*xx).reindex_reduce("add", out_shape, [
                'i0', # N
                'i2', # o
                f'i3*{stride_h}-{padding_h}+i5*{dilation_h}', # Hid+Khid
                f'i4*{stride_w}-{padding_w}+i6*{dilation_w}', # Wid+KWid
            ])
            if self.bias is not None:
                b = self.bias.broadcast(y.shape, [0,2,3])
                y = y + b
            return y
        else:
            N,C,H,W = x.shape
            Kh, Kw = self.kernel_size
            i,o,h,w = self.weight.shape
            oc = self.out_channels
            G = self.groups
            CpG = C // G # channels per group
            assert C==self.in_channels
            stride_h, stride_w = self.stride
            padding_h, padding_w = self.padding
            dilation_h, dilation_w = self.dilation

            oh = (H-1) * stride_h + self.output_padding[0] - 2*padding_h + 1 + (h-1)*dilation_h
            ow = (W-1) * stride_w + self.output_padding[1] - 2*padding_w + 1 + (w-1)*dilation_w
            out_shape = (N, oc, oh, ow)
            shape = [N,G,oc//G,CpG,oh,ow,Kh,Kw]
            xx = x.reindex(shape, [
                'i0',
                f'i1*{oc//G}+i2',
                'i4',
                'i5'
            ])
            ww = self.weight.reindex(shape, [
                f'i1*{oc//G}+i2',
                'i3',
                'i6',
                'i7'
            ])
            ww.compile_options = xx.compile_options = {"G":G,"C":C}
            y = (ww*xx).reindex_reduce("add", out_shape, [
                'i0', # Nid
                f'i1*{CpG}+i3', # Gid
                f'i4*{self.stride[0]}-{self.padding[0]}+i6*{self.dilation[0]}', # Hid+Khid
                f'i5*{self.stride[1]}-{self.padding[1]}+i7*{self.dilation[1]}', # Wid+KWid
            ])
            if self.bias is not None:
                b = self.bias.broadcast(y.shape, [0,2,3])
                y = y + b
            return y


class ConvTranspose3d(jt.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, \
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1):
        self.in_channels = in_channels
        self.out_channels = out_channels

        # added
        self.dilation = dilation
        self.group = groups
        assert groups==1, "Group conv not supported yet."

        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
        # added
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.real_padding = (
            self.dilation[0] * (self.kernel_size[0] - 1) - self.padding[0],
            self.dilation[1] * (self.kernel_size[1] - 1) - self.padding[1],
            self.dilation[2] * (self.kernel_size[2] - 1) - self.padding[2])
        self.output_padding = output_padding if isinstance (output_padding, tuple) else (output_padding, output_padding, output_padding)
        assert self.output_padding[0] < max(self.stride[0], self.dilation[0]) and \
            self.output_padding[1] < max(self.stride[1], self.dilation[1]) and \
            self.output_padding[2] < max(self.stride[2], self.dilation[2]), \
            "output padding must be smaller than max(stride, dilation)"

        self.weight = jt.nn.init.invariant_uniform((in_channels, out_channels) + self.kernel_size, dtype="float")
        if bias:
            fan=1
            for i in self.weight.shape[1:]:
                fan *= i
            bound = 1 / math.sqrt(fan)
            self.bias = jt.nn.init.uniform([out_channels], dtype="float", low=-bound, high=bound)
        else:
            self.bias = None

    def execute(self, x):
        return jt.nn.conv_transpose3d(x, self.weight, self.bias, self.stride, self.padding, self.output_padding, self.group, self.dilation)


_FACADE_SYMBOLS = (ConvTranspose, ConvTranspose3d)
preserve_facade_origins(_FACADE_SYMBOLS)
