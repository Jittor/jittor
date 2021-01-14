# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
from jittor import Module
from jittor.models import resnet
import pickle

f32 = jt.float32

def matmul(a, b):
    (n, m), k = a.shape, b.shape[-1]
    a = a.broadcast([n,m,k], dims=[2])
    b = b.broadcast([n,m,k], dims=[0])
    return (a*b).sum(dim=1)


def relu(x):
    return jt.maximum(x, 0.0)
Relu = jt.make_module(relu)

class Model(Module):
    def __init__(self, input_size):
        self.linear1 = Linear(input_size, 10)
        self.relu1 = Relu()
        self.linear2 = Linear(10, 1)
    def execute(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        return self.linear2(x)

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        self.w = (jt.random((in_features, out_features))-0.5) / in_features**0.5
        self.b = jt.random((out_features,))-0.5 if bias else None
    def execute(self, x):
        x = matmul(x, self.w)
        if self.b is not None: 
            return x+self.b
        return x


class TestTraceVar(unittest.TestCase):
    def test_simple_model(self):
        with jt.flag_scope(trace_py_var=2):

            model = Model(input_size=1)
            batch_size = 10
            x = jt.float32(np.random.rand(batch_size, 1))
            y = model(x)
            y.sync()


            data = jt.dump_trace_data()
            jt.clear_trace_data()
            # with open("/tmp/simple_model.pkl", "wb") as f:
            #     pickle.dump(data, f)

    def test_simple_model_train(self):
        with jt.flag_scope(trace_py_var=2):
            
            model = Model(input_size=1)
            opt = jt.optim.SGD(model.parameters(), 0.1)

            batch_size = 10
            x = jt.float32(np.random.rand(batch_size, 1))
            y = model(x)
            opt.step(y**2)
            jt.sync_all()

            data = jt.dump_trace_data()
            jt.clear_trace_data()
            # with open("/tmp/simple_model_train.pkl", "wb") as f:
            #     pickle.dump(data, f)

    def test_resnet(self):
        with jt.flag_scope(trace_py_var=2):

            resnet18 = resnet.Resnet18()
            x = jt.float32(np.random.rand(2, 3, 224, 224))
            y = resnet18(x)
            y.sync()

            data = jt.dump_trace_data()
            jt.clear_trace_data()
            # with open("/tmp/resnet.pkl", "wb") as f:
            #     pickle.dump(data, f)

    def test_resnet_train(self):
        with jt.flag_scope(trace_py_var=2):

            resnet18 = resnet.Resnet18()
            opt = jt.optim.SGD(resnet18.parameters(), 0.1)
            x = jt.float32(np.random.rand(2, 3, 224, 224))
            y = resnet18(x)

            opt.step(y**2)
            jt.sync_all()

            data = jt.dump_trace_data()
            jt.clear_trace_data()
            # with open("/tmp/resnet_train.pkl", "wb") as f:
            #     pickle.dump(data, f)

    def test_resnet_train_profile(self):
        with jt.profile_scope(trace_py_var=1):

            resnet18 = resnet.Resnet18()
            opt = jt.optim.SGD(resnet18.parameters(), 0.1)
            x = jt.float32(np.random.rand(2, 3, 224, 224))
            y = resnet18(x)

            opt.step(y**2)
            jt.sync_all()


if __name__ == "__main__":
    unittest.main()