import jittor as jt
import numpy as np
import math
from collections.abc import Sequence,Iterable
from ctypes import *

def test_call_c(x, path):
    tso = cdll.LoadLibrary(path)
    tso.PyShowImage(c_void_p(id(x)))