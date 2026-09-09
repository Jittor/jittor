"""Codegen tensor operations."""

from typing import List


def python_pass_wrapper(mod_func, args, kw):
    import importlib
    mod, func = mod_func.rsplit(".", 1)
    mod = importlib.import_module(mod)
    func = getattr(mod, func)
    args = args + ("**kw",)
    args = ",".join(args)
    return eval(f"func({args})")


def auto_parallel(n, src, block_num=1024, **kw):
    """
    auto parallel(CPU and GPU) n-d for loop function like below:

    Before:

    void inner_func(int n0, int i0, int n1, int i1) {
        ...
    }

    for (int i0=0; i0<n0; i0++)
        for (int i1=0; i1<n1; i1++)
            inner_func(n0, i0, n1, i1, ...);

    After:

    @python.jittor.auto_parallel(2)
    void inner_func(int n0, int i0, int n1, int i1) {
        ...
    }

    inner_func(n0, 0, n1, 0, ...);


    """
    from jittor.backends.cuda.kernels.misc import codegen as _cuda_codegen
    # src = prev_func func_name(args)code
    a, b = src.split('(', 1)
    prev_func, func_name = a.rsplit(None, 1)
    args, code = b.split(')', 1)
    args = args.split(',')
    if len(args) < n * 2:
        raise ValueError(
            "codegen: expected at least {} argument descriptors, got {}".format(
                n * 2, len(args)
            )
        )
    oargs = args[n*2:]
    pargs = args[:n*2]
    pnargs = pargs[0::2]
    pnargs2 = [ a.split()[-1] for a in pnargs ]
    oargs2 = [ a.split()[-1] for a in oargs ]
    call_args: List[str] = []
    for i in range(n):
        call_args.extend((pnargs2[i], f"i{i}"))
    call_args += oargs2
    loops = "\n".join(f"for (int i{i}=0; i{i}<{pnargs2[i]}; i{i}++)" for i in range(n))
    cpu_source = f"""
{src.replace(func_name, func_name+"_inner", 1)}
inline static void {func_name}({",".join(pargs+oargs)}) {{
    {loops}
    {func_name}_inner({",".join(call_args)});
}}
"""
    return _cuda_codegen.auto_parallel_cuda(
        n, src, block_num, func_name, pargs, oargs, pnargs, pnargs2, oargs2, cpu_source)
