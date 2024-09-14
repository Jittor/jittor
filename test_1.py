import jittor as jt
jt.flags.use_acl =1

weight = jt.zeros([200,])
# b = weight[0,0]
# g = jt.grad(b,weight)
# print(g)
# print(weight)
weight[0]=1.2
print(weight)