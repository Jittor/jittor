import jittor as jt

for k,v in jt.nn.init.__dict__.items():
    if callable(v):
        globals()[k] = v


normal = gauss
normal_ = gauss_
xavier_normal = xavier_gauss
xavier_normal_ = xavier_gauss_
zeros_ = zero_


jt.Var.normal_ = normal_

