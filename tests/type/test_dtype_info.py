import unittest
import jittor as jt
import numpy as np


class TestFinfo(unittest.TestCase):
    def test(self):
        for dtype in ['float16', 'float32', 'float64']:
            finfo = jt.finfo(dtype)
            np_finfo = np.finfo(dtype)
            assert finfo.bits == np_finfo.bits
            assert finfo.eps == np_finfo.eps
            assert finfo.max == np_finfo.max
            assert finfo.min == np_finfo.min
            assert finfo.nexp == np_finfo.nexp
            assert finfo.nmant == np_finfo.nmant
            assert finfo.iexp == np_finfo.iexp
            assert finfo.precision == np_finfo.precision
            assert finfo.resolution == np_finfo.resolution
            assert finfo.tiny == np_finfo.tiny
        for dtype_jt, dtype in [
            (jt.float16, 'float16'),
            (jt.float32, 'float32'),
            (jt.float64, 'float64'),
        ]:
            finfo = jt.finfo(dtype_jt)
            np_finfo = np.finfo(dtype)
            assert finfo.bits == np_finfo.bits
            assert finfo.eps == np_finfo.eps
            assert finfo.max == np_finfo.max
            assert finfo.min == np_finfo.min
            assert finfo.nexp == np_finfo.nexp
            assert finfo.nmant == np_finfo.nmant
            assert finfo.iexp == np_finfo.iexp
            assert finfo.precision == np_finfo.precision
            assert finfo.resolution == np_finfo.resolution
            assert finfo.tiny == np_finfo.tiny


class TestIinfo(unittest.TestCase):
    def test(self):
        for dtype in ['int16', 'int32', 'int64']:
            iinfo = jt.iinfo(dtype)
            np_iinfo = np.iinfo(dtype)
            assert iinfo.bits == np_iinfo.bits
            assert iinfo.max == np_iinfo.max
            assert iinfo.min == np_iinfo.min
            assert iinfo.dtype == np.dtype(dtype)
        for dtype_jt, dtype in [
            (jt.int16, 'int16'),
            (jt.int32, 'int32'),
            (jt.int64, 'int64'),
        ]:
            iinfo = jt.iinfo(dtype_jt)
            np_iinfo = np.iinfo(dtype)
            assert iinfo.bits == np_iinfo.bits
            assert iinfo.max == np_iinfo.max
            assert iinfo.min == np_iinfo.min
            assert iinfo.dtype == np.dtype(dtype)


if __name__ == "__main__":
    unittest.main()
