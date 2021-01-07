# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers:
#   Dun Liang <randonlang@gmail.com>.
#   Xiangli Li <190569238@qq.com>
#
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************

import jittor as jt
import numpy as np

class SparseVar:
    def __init__(self,indices,values,shape):
        assert isinstance(indices,jt.Var) and isinstance(values,jt.Var) and isinstance(shape,jt.NanoVector)
        self.indices = indices
        self.values = values
        self.shape = shape
        self.ndim = len(shape)
        
    def _indices(self):
        return self.indices
    
    def _values(self):
        return self.values
    
    def t(self):
        indices = list(self.indices.split(1,dim=0))
        indices[-1],indices[-2] = indices[-2],indices[-1]
        indices = jt.contrib.concat(indices,dim=0)
        shape = list(self.shape)
        shape[-1],shape[-2] = shape[-2],shape[-1]
        shape = jt.NanoVector(shape)
        return SparseVar(indices,self.values,shape)
        
    def to_dense(self):
        ret = jt.zeros(self.shape,self.values.dtype)
        indices  = tuple(self.indices.split(1,dim=0))
        ret[indices]=self.values
        return ret

def sparse_array(indices,values,shape):
    return SparseVar(indices,values,shape)

def spmm(spase_x,y):
    assert isinstance(spase_x,SparseVar) and isinstance(y,jt.Var)
    assert spase_x.ndim==2 and y.ndim==2 and spase_x.shape[-1]==y.shape[0]
    
    # TODO
    x = spase_x.to_dense()
    return jt.matmul(x,y)
    