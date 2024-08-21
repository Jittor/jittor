import jittor as jt
import os
import jittor_utils
from jittor_utils import lock
import jittor.compiler as compiler
import numpy as np

dirname = os.path.dirname(__file__)
cc_files = [ dirname + "/vcompiler.cc"]
with open(dirname + "/vcompiler.h") as f:
    h_src = f.read()

with lock.lock_scope():
    mod = jittor_utils.compile_module(h_src, compiler.cc_flags + "-I" + dirname + " " + " ".join(cc_files))

for k, v in mod.__dict__.items():
    if k.startswith("_"):
        continue
    globals()[k] = v

def dfs(obj, path=""):
    if isinstance(obj, jt.Var):
        return [((path, len(obj.shape), str(obj.dtype)), obj)]
    if isinstance(obj, (list, tuple)):
        ret = []
        for i, v in enumerate(obj):
            ret += dfs(v, path + "[%d]" % i)
        return ret
    if isinstance(obj, dict):
        ret = []
        for k, v in obj.items():
            ret += dfs(v, path + "[%r]" % k)
        return ret
    return []

def dfs_config(obj):
    if isinstance(obj, jt.Var):
        return "Var"
    if isinstance(obj, (int, float, bool, str, type(None))):
        return obj
    if isinstance(obj, (list, tuple)):
        return [ dfs_config(v) for v in obj ]
    if isinstance(obj, dict):
        return { k:dfs_config(v) for k, v in obj.items() }
    raise ValueError(f"Unknown type {type(obj)}")

def dfs_clone_var(obj):
    if isinstance(obj, jt.Var):
        return obj.clone()
    if isinstance(obj, (int, float, bool, str, type(None))):
        return obj
    if isinstance(obj, (list, tuple)):
        return [ dfs_clone_var(v) for v in obj ]
    if isinstance(obj, dict):
        return { k:dfs_clone_var(v) for k, v in obj.items() }
    raise ValueError(f"Unknown type {type(obj)}")

def dfs_fill(obj, vars):
    i = 0
    def dfs_fill_var(obj):
        nonlocal i
        if isinstance(obj, jt.Var):
            v = vars[i]
            i += 1
            return v
        if isinstance(obj, (int, float, bool, str, type(None))):
            return obj
        if isinstance(obj, (list, tuple)):
            return [ dfs_fill_var(v) for v in obj ]
        if isinstance(obj, dict):
            ret = { k:dfs_fill_var(v) for k, v in obj.items() }
            return obj.__class__(ret)
        raise ValueError(f"Unknown type {type(obj)}")
    return dfs_fill_var(obj)


class CachedGraph:
    def __init__(self, func, args, kw):
        args = dfs_clone_var(args)
        kw = dfs_clone_var(kw)
        self.func = func
        self.inputs = (args, kw)
        jt.sync_all()
        exec_called = jt.flags.exec_called
        self.outputs = func(*args, **kw)
        import gc; gc.collect()
        assert exec_called == jt.flags.exec_called, (exec_called, jt.flags.exec_called)
        self.outputs_parsed = dfs(self.outputs)
        self.outputs_var = [ v for _, v in self.outputs_parsed ]
        self.inputs_parsed = dfs(self.inputs)
        self.inputs_var = [ v for _, v in self.inputs_parsed ]
        self.inputs_key = str([ key for key, _ in self.inputs_parsed ])
        for v in self.outputs_var:
            v.release_from_holders()
        for v in self.inputs_var:
            v.release_from_holders()
        self.sgraph = mod.build_sgraph(self.outputs_var, self.inputs_var)

# a function decorator
# build new graph:
# 1. shape dim changed
# 2. dtype changed
# 3. var path changed
# graph key:
# (args, kw), [ (var_path, shape dim, dtype), var ]
def build(func, debug=False, fallback_func=None):
    cache = {}
    def func_wrapper(*args, **kw):
        if fallback_func and fallback_func(*args, **kw):
            return func(*args, **kw)
        inputs = (args, kw)
        config_key = str(dfs_config(inputs))
        inputs_parsed = dfs(inputs)
        inputs_key = str([ key for key, _ in inputs_parsed ])
        inputs_var = [ v for _, v in inputs_parsed ]
        jt.sync(inputs_var)
        all_key = config_key + inputs_key
        if all_key not in cache:
            # print(f"create graph with key '{all_key[:30]}'...")
            cache[all_key] = CachedGraph(func, args, kw)
        graph = cache[all_key]
        if not mod.prob_sgraph(graph.sgraph, inputs_var):
            # print(f"merge graph with key '{all_key[:30]}'...")
            graph2 = CachedGraph(func, args, kw)
            mod.merge_sgraph(graph.sgraph, graph2.sgraph)
        outputs = mod.exec_sgraph(graph.sgraph, inputs_var)
        if debug:
            graph2 = CachedGraph(func, args, kw)
            outputs2 = mod.exec_sgraph(graph2.sgraph, inputs_var)
            for v1, v2 in zip(outputs, outputs2):
                np.testing.assert_allclose(v1.data, v2.data, rtol=0.01, atol=0.05)
        return dfs_fill(graph.outputs, outputs)
    return func_wrapper

# c interface
# build_sgraph -> sgraph
# merge_sgraph
# exec_sgraph
# prob_sgraph: check sgraph can exec
# if prob_sgraph failed
#   build_sgraph
#   merge_sgraph
# exec_sgraph



# overall code:
# 1. get input_key from (args, kw)
# 2. if input_key not in cache
#       graph.outputs = func(*args, **kw)
#       graph.outputs_var = var_parser(graph.outputs)
#       graph.sgraph = build_sgraph(outputs_var)
#       graph.inputs = (args, kw)
#       graph.inputs_var = var_parser(args, kw)
