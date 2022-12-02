import jittor as jt

jt.flags.use_device = 1
n = 100000

with jt.profile_scope(10, 10) as rep:
    jt.code([2], "float32", [],
    cuda_header='''__global__ void kernel(float* a) {}''',
    cuda_src=f'''
    for (int i=0; i<{n}; i++) kernel<<<1,32>>>(out0_p);
    ''').sync()

avg_ns = float(rep[1][4]) / n
print("kernel launch overhead(ns):", avg_ns)
