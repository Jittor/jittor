import jittor as jt
from jittor import nn
from jittor.models import resnet50
import time

jt.flags.use_cuda = 1

net = resnet50()
x = jt.ones(2, 3, 224, 224)
y = net(x)
y.sync()
start = time.time()
for i in range(100):
    y = net(x)
    y.sync()
print(time.time() - start)
