"""CUDA thread mapping and launch generation for shared auto_parallel bodies."""


def auto_parallel_cuda(n, src, block_num, func_name, pargs, oargs, pnargs, pnargs2, oargs2, cpu_source):
    entry_func_args_def = ",".join(["int tn"+str(i) for i in range(n)]
        + pnargs + oargs)
    entry_func_args = ",".join(["tn"+str(i) for i in range(n)]
        + pnargs2 + oargs2)
    tid_def = ""
    tid_loop = ""
    call_args = []
    for i in reversed(range(n)):
        tid_def += f"\nauto tid{i} = tid & ((1<<tn{i})-1);"
        tid_def += f"\nauto tnum{i} = 1<<tn{i};"
        tid_def += f"\ntid = tid>>tn{i};"
    for i in range(n):
        tid_loop += f"\nfor (int i{i}=tid{i}; i{i}<{pnargs2[i]}; i{i}+=tnum{i})"
        call_args.append(pnargs2[i])
        call_args.append(f"i{i}")
    call_args += oargs2
    call_args = ",".join(call_args)
    xn = '\n'
    new_src = f"""
#ifdef JIT_cuda
__device__
{src.replace(func_name, func_name+"_inner", 1)}

__global__ static void {func_name}_entry({entry_func_args_def}) {{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    {tid_def}
    {tid_loop}
    {func_name}_inner({call_args});
}}

inline static void {func_name}({",".join(pargs+oargs)}) {{
    int thread_num = 256*{block_num};
    {xn.join([f"int tn{i} = NanoVector::get_nbits(std::min(thread_num, {pnargs2[i]})) - 2;thread_num >>= tn{i};" for i in reversed(range(n))])}
    thread_num = 1<<({"+".join([f"tn{i}" for i in range(n)])});
    int p1 = std::max(thread_num/{block_num}, 1);
    int p2 = std::min(thread_num, {block_num});
    {func_name}_entry<<<p1,p2>>>({entry_func_args});
}}
#else
{cpu_source}
#endif
"""
    return new_src
