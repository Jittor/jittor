# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
from jittor import init, Module
import numpy as np
from jittor.optim import Optimizer
f32 = jt.float32

def matmul(a, b):
    (n, m), k = a.shape, b.shape[-1]
    a = a.broadcast([n,m,k], dims=[2])
    b = b.broadcast([n,m,k], dims=[0])
    return (a*b).sum(dim=1)

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        self.w = (jt.random((in_features, out_features))-0.5) / in_features**0.5
        self.b = jt.random((out_features,))-0.5 if bias else None
    def execute(self, x):
        x = matmul(x, self.w)
        if self.b is not None: 
            return x+self.b
        return x

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

class TestExample(unittest.TestCase):
    def test1(self):
        np.random.seed(0)
        jt.set_seed(3)
        n = 1000
        batch_size = 50
        base_lr = 0.05
        # tune accumulation_steps for step and batch_size
        accumulation_steps = 10
        n *= accumulation_steps
        batch_size //= accumulation_steps
        # we need to stop grad of global value to prevent memory leak
        lr = f32(base_lr).name("lr").stop_grad()

        def get_data(n):
            for i in range(n):
                x = np.random.rand(batch_size, 1)
                y = x*x
                yield jt.float32(x), jt.float32(y)
        
        model = Model(input_size=1)
        ps = model.parameters()
        for p in reversed(ps): p.sync(0,0)
        opt = Optimizer(ps, lr)
        all_loss = 0

        for i,(x,y) in enumerate(get_data(n)):
            pred_y = model(x).name("pred_y")
            loss = ((pred_y - y)**f32(2)).name("loss")
            loss_mean = loss.mean()  / accumulation_steps
            all_loss += loss_mean.item()
            
            opt.backward(loss_mean)
            if (i+1) % accumulation_steps == 0:
                opt.step()

                if i>50:
                    assert prev == jt.liveness_info(), f"memory leak {prev} {jt.liveness_info()}"
                prev = jt.liveness_info()
                print(f"step {i}, loss = {loss_mean.data.sum()} {jt.liveness_info()}")

        print(all_loss)
        possible_results = [19.8639366890402, 8.207454475712439]
        assert any(abs(all_loss - r) < 1e-3 for r in possible_results)
        jt.clean()

if __name__ == "__main__":
    unittest.main()