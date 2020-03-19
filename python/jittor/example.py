import jittor as jt
import numpy as np
import sys, os
f32 = jt.float32

@jt.var_scope('linear')
def linear(x, n):
    w = jt.make_var([x.shape[-1], n], init=lambda *a:
            (jt.random(*a)-f32(0.5)) / f32(x.shape[-1])**f32(0.5))
    b = jt.make_var([n], init=lambda *a: jt.random(*a)-f32(0.5))
    return jt.matmul(x, w) + b
    
def relu(x): return jt.maximum(x, f32(0))

@jt.var_scope('model', unique=True)
def model(x):
    x = linear(x, 10)
    x = relu(x)
    x = linear(x, 1)
    return x

np.random.seed(0)
jt.set_seed(3)
n = 1000
batch_size = 50
base_lr = 0.05
# we need to stop grad of global value to prevent memory leak
lr = f32(base_lr).name("lr").stop_grad()

def get_data(n):
    for i in range(n):
        x = np.random.rand(batch_size, 1)
        y = x*x
        yield np.float32(x), np.float32(y)

for i,(x,y) in enumerate(get_data(n)):
    pred_y = model(x).name("pred_y")
    loss = ((pred_y - y)**f32(2)).name("loss")
    loss_mean = loss.mean()
    
    ps = jt.find_vars('model')
    gs = jt.grad(loss_mean, ps)
    for p,g in zip(ps, gs):
        p -= g * lr
    if i>2:
        assert prev == jt.liveness_info(), f"memory leak {prev} {jt.liveness_info()}"
    prev = jt.liveness_info()
    print(f"step {i}, loss = {loss_mean().sum()}")

# result is 0.0009948202641680837
result = 0.0009948202641680837
assert abs(loss_mean.data - result) < 1e-6