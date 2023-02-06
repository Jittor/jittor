from jittor_core import *
from jittor_core.ops import *
from .misc import *
from . import attention as attention, contrib as contrib, dataset as dataset, init as init, linalg as linalg, lr_scheduler as lr_scheduler, numpy2cupy as numpy2cupy, optim as optim, sparse as sparse
from .compile_extern import cublas as cublas, cudnn as cudnn, cufft as cufft, curand as curand, mkl_ops as mkl_ops, mpi_ops as mpi_ops, world_size as world_size
from .compiler import compile_custom_op as compile_custom_op, compile_custom_ops as compile_custom_ops
from .contrib import concat as concat
from .nn import bmm as bmm, bmm_transpose as bmm_transpose, matmul as matmul
from collections import OrderedDict as OrderedDict
from collections.abc import Mapping as Mapping
from typing import Any, List, Tuple


def safepickle(obj, path) -> None: ...
def safeunpickle(path): ...

class _call_no_record_scope:
    def __enter__(self) -> None: ...
    def __exit__(self, *exc) -> None: ...
    def __call__(self, func): ...

class flag_scope(_call_no_record_scope):
    jt_flags: Any
    def __init__(self, **jt_flags) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, *exc) -> None: ...

class no_grad(flag_scope):
    jt_flags: Any
    def __init__(self, **jt_flags) -> None: ...

class enable_grad(flag_scope):
    jt_flags: Any
    def __init__(self, **jt_flags) -> None: ...

single_log_capture: Any

class log_capture_scope(_call_no_record_scope):
    fs: Any
    def __init__(self, **jt_flags) -> None: ...
    logs: Any
    def __enter__(self): ...
    def __exit__(self, *exc) -> None: ...

class profile_scope(_call_no_record_scope):
    fs: Any
    warmup: Any
    rerun: Any
    def __init__(self, warmup: int = ..., rerun: int = ..., **jt_flags) -> None: ...
    report: Any
    def __enter__(self): ...
    def __exit__(self, *exc) -> None: ...

class __single_process_scope:
    rank: Any
    def __init__(self, rank: int = ...) -> None: ...
    bk_in_mpi: Any
    bk_mpi_state: Any
    def __enter__(self): ...
    def __exit__(self, *exc) -> None: ...

def single_process_scope(rank: int = ...): ...
def clean() -> None: ...
cast = unary

def array(data, dtype: Any | None = ...): ...
def random(shape, dtype: str = ..., type: str = ...): ...
def float_auto(x): ...
def array64(data, dtype: Any | None = ...): ...
def grad(loss, targets, retain_graph: bool = ...): ...
def liveness_info(): ...
def ones(shape, dtype: str = ...): ...
def ones_like(x): ...
def zeros(shape, dtype: str = ...): ...
def full(shape, val, dtype: str = ...): ...
def full_like(x, val, dtype: Any | None = ...) -> Var: ...
def zeros_like(x, dtype: Any | None = ...) -> Var: ...

def var(x, dim: Any | None = ..., dims: Any | None = ..., unbiased: bool = ..., keepdims: bool = ...): ...
def std(x): ...
def norm(x, p: int = ..., dim: int = ..., keepdims: bool = ..., eps: float = ..., keepdim: bool = ...): ...
origin_reshape = reshape

def reshape(x, *shape): ...
view = reshape
origin_transpose = transpose

def transpose(x, *dim): ...
permute = transpose
def flatten(input, start_dim: int = ..., end_dim: int = ...): ...
def detach(x): ...
def unsqueeze(x, dim): ...
def squeeze(x, dim): ...
def clamp(x, min_v: Any | None = ..., max_v: Any | None = ...): ...
def type_as(a, b): ...
def masked_fill(x, mask, value): ...
def sqr(x): ...
def pow(x, y): ...
def argmax(x, dim, keepdims: bool = ...): ...
def argmin(x, dim, keepdims: bool = ...): ...
def randn(*size, dtype: str = ..., requires_grad: bool = ...) -> Var: ...
def rand(*size, dtype: str = ..., requires_grad: bool = ...) -> Var: ...
def rand_like(x, dtype: Any | None = ...) -> Var: ...
def randn_like(x, dtype: Any | None = ...) -> Var: ...
def randint(low, high: Any | None = ..., shape=..., dtype: str = ...) -> Var: ...
def randint_like(x, low, high: Any | None = ...) -> Var: ...
def normal(mean, std, size: Any | None = ..., dtype: str = ...) -> Var: ...
def attrs(var): ...
def fetch(*args) -> None: ...
def display_memory_info() -> None: ...
def load(path: str): ...
def save(params_dict, path: str): ...

class Module:
    def __init__(self, *args, **kw) -> None: ...
    def execute(self, *args, **kw) -> None: ...
    def __call__(self, *args, **kw): ...
    def __name__(self) -> None: ...
    def dfs(self, parents, k, callback, callback_leave: Any | None = ...) -> None: ...
    def parameters(self) -> List: ...
    def state_dict(self, to: Any | None = ...): ...
    def named_parameters(self) -> List[Tuple[str, Var]]: ...
    def load_state_dict(self, params) -> None: ...
    def modules(self) -> List: ...
    def named_modules(self): ...
    def requires_grad_(self, requires_grad: bool = ...): ...
    def __hooked_call__(self, *args, **kw): ...
    __fhook__: Any
    def register_forward_hook(self, func) -> None: ...
    def remove_forward_hook(self) -> None: ...
    __fhook2__: Any
    def register_pre_forward_hook(self, func) -> None: ...
    def remove_pre_forward_hook(self) -> None: ...
    __bihook__: Any
    def register_input_backward_hook(self, func) -> None: ...
    def remove_input_backward_hook(self) -> None: ...
    __bohook__: Any
    def register_output_backward_hook(self, func) -> None: ...
    def remove_output_backward_hook(self) -> None: ...
    def register_backward_hook(self, func): ...
    def remove_backward_hook(self) -> None: ...
    def children(self) -> List: ...
    def extra_repr(self): ...
    def apply(self, func) -> None: ...
    def load_parameters(self, params) -> None: ...
    def save(self, path: str): ...
    def load(self, path: str): ...
    backup_grad_state: Any
    def eval(self) -> None: ...
    def train(self) -> None: ...
    is_train: bool
    def is_training(self) -> bool: ...
    def mpi_param_broadcast(self, root: int = ...) -> None: ...
    def __setattr__(self, key, value) -> None: ...
    def __getattr__(self, key): ...
    def float64(self): ...
    def float16(self): ...
    def half(self): ...
    def float_auto(self): ...

class Function(Module):
    input_mask: Any
    output_mask: Any
    def __call__(self, *args): ...
    def dfs(self, parents, k, callback, callback_leave: Any | None = ...) -> None: ...
    @classmethod
    def apply(cls, *args, **kw): ...

class GradHooker(Function):
    hook: Any
    def __init__(self, hook) -> None: ...
    def execute(self, *args): ...
    def grad(self, *grad_input): ...

def grad_hooker(args, hook): ...
def register_hook(v, hook): ...
def make_module(func, exec_n_args: int = ...): ...
def dirty_fix_pytorch_runtime_error() -> None: ...

class ExitHooks:
    exit_code: Any
    exception: Any
    def __init__(self) -> None: ...
    def hook(self) -> None: ...
    def exit(self, code: int = ...) -> None: ...
    def exc_handler(self, exc_type, exc, *args) -> None: ...

hooks: Any

def jittor_exit() -> None: ...
def vtos(v): ...
def size(v, dim: Any | None = ...): ...
def to_int(v): ...
def to_float(v): ...
def to_bool(v): ...
def format(v, spec): ...
def get_len(var): ...
half = float16

def is_var(v): ...
from typing import List, Tuple, Callable, overload
import numpy
def ternary(cond: Var, x: Var, y: Var)-> Var:
 ...
@overload
def reindex(x: Var, shape: Tuple[int], indexes: List[str], overflow_value: float=0, overflow_conditions: List[str]={}, extras: List[Var]={})-> Var:
	'''Document:
	* 
	    Reindex Operator is a one-to-many map operator.
	    It performs equivalent Python-pseudo implementation below::
	
	        # input is x, output is y
	        n = len(shape)-1
	        m = len(x.shape)-1
	        k = len(overflow_conditions)-1
	        y = np.zeros(shape, x.dtype)
	        for i0 in range(shape[0]): # 1-st loop
	            for i1 in range(shape[1]): # 2-nd loop
	                ...... # many loops
	                for in in range(shape[n]) # n+1 -th loop
	                    if is_overflow(i0,i1,...,in):
	                        y[i0,i1,...,in] = overflow_value
	                    else:
	                        # indexes[i] is a c++ style integer expression consisting of i0,i1,...,in
	                        y[i0,i1,...,in] = x[indexes[0],indexes[1],...,indexes[m]]
	
	        # is_overflow is defined as following
	        def is_overflow(i0,i1,...,in):
	            return (
	                indexes[0] < 0 || indexes[0] >= x.shape[0] ||
	                indexes[1] < 0 || indexes[1] >= x.shape[1] ||
	                ......
	                indexes[m] < 0 || indexes[m] >= x.shape[m] ||
	
	                # overflow_conditions[i] is a c++ style boolean expression consisting of i0,i1,...,in
	                overflow_conditions[0] ||
	                overflow_conditions[1] ||
	                ......
	                overflow_conditions[k]
	            )
	    ----------------
	    * [in] x:	A input jittor Var
		
	    * [in] shape:	the output shape, a integer array
		
	    * [in] indexes:	array of c++ style integer expression, its length should be the same with the number of dimension of x, some buildin variables it can use are::
	        
	             XDIM, xshape0, ..., xshapen, xstride0, ..., xstriden
	             YDIM, yshape0, ..., yshapem, ystride0, ..., ystridem
	             i0, i1, ..., in
	             @e0(...), @e1(...) for extras input index
	             e0p, e1p , ... for extras input pointer
				 
	    * [in] overflow_value:	overflow value
		
	    * [in] overflow_conditions:	array of c++ style boolean expression, it length can be vary. the buildin variables it can use are the same with indexes
			
	    * [in] extras: extra var used for index
		
	    ----------------
	    Example
	    Convolution implemented by reindex operation::
	
	        def conv(x, w):
	            N,H,W,C = x.shape
	            Kh, Kw, _C, Kc = w.shape
	            assert C==_C
	            xx = x.reindex([N,H-Kh+1,W-Kw+1,Kh,Kw,C,Kc], [
	                'i0', # Nid
	                'i1+i3', # Hid+Khid
	                'i2+i4', # Wid+KWid
	                'i5', # Cid
	            ])
	            ww = w.broadcast_var(xx)
	            yy = xx*ww
	            y = yy.sum([3,4,5]) # Kh, Kw, C
	            return y, yy'''
	...
@overload
def reindex(x: Var, indexes: List[Var], overflow_value: float=0, overflow_conditions: List[str]={})-> Var:
	'''Document:
	* Alias x.reindex([i,j,k]) -> 
	        x.reindex(i.shape, ['@e0(...)','@e1(...)','@e2(...)',], extras=[i,j,k])'''
	...
def reindex_var(x: Var, indexes: List[Var], overflow_value: float=0, overflow_conditions: List[str]={})-> Var:
	'''Document:
	* Alias x.reindex([i,j,k]) -> 
	        x.reindex(i.shape, ['@e0(...)','@e1(...)','@e2(...)',], extras=[i,j,k])'''
	...
@overload
def index(shape: Tuple[int], dim: int, dtype: str="int32")-> Var:
	'''Document:
	* 
	    Index Operator generate index of shape.
	    
	    It performs equivalent Python-pseudo implementation below::
	    
	        n = len(shape)-1
	        x = np.zeros(shape, dtype)
	        for i0 in range(shape[0]): # 1-st loop
	            for i1 in range(shape[1]): # 2-nd loop
	                ...... # many loops
	                for in in range(shape[n]) # n+1 -th loop
	                    x[i0,i1,...,in] = i@dim
	    
	    * [in] shape:   the output shape, a integer array
	    * [in] dim: the dim of the index.
	    * [in] dtype:   the data type string, default int32
	
	    Example::
	
	        print(jt.index([2,2], 0)())
	        # output: [[0,0],[1,1]]
	        print(jt.index([2,2], 1)())
	        # output: [[0,1],[0,1]]'''
	...
@overload
def index(shape: Tuple[int], dtype: str="int32")-> Tuple[Var]:
	'''Document:
	* 
	    Index Operator generate index of shape.
	    
	    It performs equivalent Python-pseudo implementation below::
	    
	        n = len(shape)-1
	        x = np.zeros(shape, dtype)
	        for i0 in range(shape[0]): # 1-st loop
	            for i1 in range(shape[1]): # 2-nd loop
	                ...... # many loops
	                for in in range(shape[n]) # n+1 -th loop
	                    x[i0,i1,...,in] = i@dim
	    
	    * [in] shape:   the output shape, a integer array
	    * [in] dim: the dim of the index.
	    * [in] dtype:   the data type string, default int32
	
	    Example::
	
	        print(jt.index([2,2], 0)())
	        # output: [[0,0],[1,1]]
	        print(jt.index([2,2], 1)())
	        # output: [[0,1],[0,1]]'''
	...
@overload
def index(a: Var, dim: int, dtype: str="int32")-> Var:
	'''Document:
	* shape dependency version of index op
	        jt.index_var(a, 1) similar with jt.index(a.shape, 1)'''
	...
@overload
def index(a: Var, dtype: str="int32")-> Tuple[Var]:
	'''Document:
	* shape dependency version of index op
	        jt.index_var(a) similar with jt.index(a.shape)'''
	...
@overload
def index_var(a: Var, dim: int, dtype: str="int32")-> Var:
	'''Document:
	* shape dependency version of index op
	        jt.index_var(a, 1) similar with jt.index(a.shape, 1)'''
	...
@overload
def index_var(a: Var, dtype: str="int32")-> Tuple[Var]:
	'''Document:
	* shape dependency version of index op
	        jt.index_var(a) similar with jt.index(a.shape)'''
	...
def binary(x: Var, y: Var, p: str)-> Var:
 ...
def pow(x: Var, y: Var)-> Var:
	'''Document:
	*
	    Computes ``x^y``, element-wise. 
	
	    ----------------
	
	    * [in] x: the first input,  a python number or jt.Var.
	
	    * [in] y: the second input, a python number or jt.Var.'''
	...
def maximum(x: Var, y: Var)-> Var:
	'''Document:
	*
	    Returns the element-wise maximum of ``x`` and ``y``. 
	
	    ----------------
	
	    * [in] x: the first input,  a python number or jt.Var.
	
	    * [in] y: the second input, a python number or jt.Var.'''
	...
def minimum(x: Var, y: Var)-> Var:
	'''Document:
	*
	    Returns the element-wise minimum of ``x`` and ``y``. 
	
	    ----------------
	
	    * [in] x: the first input,  a python number or jt.Var.
	
	    * [in] y: the second input, a python number or jt.Var.'''
	...
def add(x: Var, y: Var)-> Var:
	'''Document:
	*
	    Element-wise adds ``x`` and ``y`` and returns a new Var. 
	    
	    This operation is equivalent to ``x + y``.
	
	    ----------------
	
	    * [in] x: the first input,  a python number or jt.Var.
	
	    * [in] y: the second input, a python number or jt.Var.'''
	...
def subtract(x: Var, y: Var)-> Var:
	'''Document:
	*
	    Element-wise subtract ``y`` from ``x`` and returns a new Var.
	
	    This operation is equivalent to ``x - y``.
	
	    ----------------
	
	    * [in] x: the first input,  a python number or jt.Var.
	
	    * [in] y: the second input, a python number or jt.Var.'''
	...
def multiply(x: Var, y: Var)-> Var:
	'''Document:
	*
	    Element-wise muliplies ``x`` with ``y`` and returns a new Var.
	
	    This operation is equivalent to ``x * y``.
	
	    ----------------
	
	    * [in] x: the first input,  a python number or jt.Var.
	
	    * [in] y: the second input, a python number or jt.Var.'''
	...
def divide(x: Var, y: Var)-> Var:
	'''Document:
	*
	    Element-wise divide ``x`` by ``y`` and returns a new Var.
	
	    This operation is equivalent to ``x / y``.
	
	    ----------------
	
	    * [in] x: the first input,  a python number or jt.Var.
	
	    * [in] y: the second input, a python number or jt.Var.
	
	    ----------------
	
	    Example-1::
	        >>> a = jt.empty((3,), dtype=jt.int32)
	        >>> a
	        jt.Var([707406378 707406378 707406378], dtype=int32)
	        >>> b = jt.empty((3,), dtype=jt.int32)
	        >>> b
	        jt.Var([674510453 171649398 538976288], dtype=int32)
	        >>> jt.divide(a, b)
	        jt.Var([1.0487701 4.1212287 1.3125001], dtype=float32)
	        >>> a / b
	        jt.Var([1.0487701 4.1212287 1.3125001], dtype=float32)
	
	    .. note ::
	    returns float value even if the dtype of input Vars are both integers.
	    @see jt.ops.floor_divide() for floor division.'''
	...
def floor_divide(x: Var, y: Var)-> Var:
	'''Document:
	*
	    Element-wise divide ``x`` by ``y`` and returns the floor of the result.
	
	    This operation is equivalent to ``x // y``.
	
	    ----------------
	
	    * [in] x: the first input,  a python number or jt.Var.
	
	    * [in] y: the second input, a python number or jt.Var.
	
	    ----------------
	
	    Example-1::
	        >>> a = jt.randint(1, 10, (3,), dtype=jt.int32)
	        >>> a
	        jt.Var([9 2 7], dtype=int32)
	        >>> b = jt.randint(1, 10, (3,), dtype=jt.int32)
	        >>> b
	        jt.Var([6 4 6], dtype=int32)
	        >>> jt.floor_divide(a, b)
	        jt.Var([1 0 1], dtype=int32)
	        >>> a // b
	        jt.Var([1 0 1], dtype=int32)'''
	...
def mod(x: Var, y: Var)-> Var:
	'''Document:
	*
	    Returns the element-wise remainder of division.
	
	    This operation is equivalent to ``x % y``.
	
	    ----------------
	
	    * [in] x: the first input,  a python number or jt.Var.
	
	    * [in] y: the second input, a python number or jt.Var.
	
	    ----------------
	
	    Example-1::
	        >>> a = jt.rand(3)
	        >>> a
	        jt.Var([0.3989529  0.20159635 0.22973768], dtype=float32)
	        >>> b = jt.rand(3)
	        >>> b
	        jt.Var([0.20121202 0.7704864  0.5654395 ], dtype=float32)
	        >>> jt.mod(a, b)
	        jt.Var([0.19774088 0.20159635 0.22973768], dtype=float32)
	        >>> a % b
	        jt.Var([0.19774088 0.20159635 0.22973768], dtype=float32)'''
	...
def less(x: Var, y: Var)-> Var:
	'''Document:
	*
	    Returns ``x < y`` element-wise.
	
	    This operation is equivalent to ``x < y``.
	
	    ----------------
	
	    * [in] x: the first input,  a python number or jt.Var.
	
	    * [in] y: the second input, a python number or jt.Var.'''
	...
def less_equal(x: Var, y: Var)-> Var:
	'''Document:
	*
	    Returns ``x <= y`` element-wise.
	
	    This operation is equivalent to ``x <= y``.
	
	    ----------------
	
	    * [in] x: the first input,  a python number or jt.Var.
	
	    * [in] y: the second input, a python number or jt.Var.'''
	...
def greater(x: Var, y: Var)-> Var:
	'''Document:
	*
	    Returns ``x > y`` element-wise.
	
	    This operation is equivalent to ``x > y``.
	
	    ----------------
	
	    * [in] x: the first input,  a python number or jt.Var.
	
	    * [in] y: the second input, a python number or jt.Var.'''
	...
def greater_equal(x: Var, y: Var)-> Var:
	'''Document:
	*
	    Returns ``x >= y`` element-wise.
	    
	    This operation is equivalent to ``x >= y``.
	
	    ----------------
	
	    * [in] x: the first input,  a python number or jt.Var.
	
	    * [in] y: the second input, a python number or jt.Var.'''
	...
def equal(x: Var, y: Var)-> Var:
	'''Document:
	*
	    Returns ``x == y`` element-wise.
	
	    This operation is equivalent to ``x == y``.
	
	    ----------------
	
	    * [in] x: the first input,  a python number or jt.Var.
	
	    * [in] y: the second input, a python number or jt.Var.'''
	...
def not_equal(x: Var, y: Var)-> Var:
	'''Document:
	*
	    Returns ``x != y`` element-wise.
	
	    This operation is equivalent to ``x != y``.
	
	    ----------------
	
	    * [in] x: the first input,  a python number or jt.Var.
	
	    * [in] y: the second input, a python number or jt.Var.'''
	...
def left_shift(x: Var, y: Var)-> Var:
	'''Document:
	*
	    Shifts the bits of ``x`` to the left by ``y``. 
	
	    Bits are shifted to the left by appending ``y`` 0s at the right of ``x``.
	    This operation is equivalent to ``x << y``.
	
	    ----------------
	
	    * [in] x: the first input,  a python number or jt.Var (int32 or int64).
	
	    * [in] y: the second input, a python number or jt.Var (int32 or int64).
	
	    ----------------
	
	    Example-1::
	        >>> a = jt.randint(0, 10, shape=(3,))
	        >>> a
	        jt.Var([7 6 7], dtype=int32)
	        >>> b = jt.randint(0, 10, shape=(3,))
	        >>> b
	        jt.Var([3 9 8], dtype=int32)
	        >>> jt.left_shift(a, b)
	        jt.Var([  56 3072 1792], dtype=int32)
	        >>> a << b
	        jt.Var([  56 3072 1792], dtype=int32)'''
	...
def right_shift(x: Var, y: Var)-> Var:
	'''Document:
	*
	    Shifts the bits of ``x`` to the right by ``y``. 
	
	    This operation is equivalent to ``x >> y``.
	
	    ----------------
	
	    * [in] x: the first input,  a python number or jt.Var (int32 or int64).
	
	    * [in] y: the second input, a python number or jt.Var (int32 or int64).
	
	    ----------------
	
	    Example-1::
	        >>> a = jt.randint(0, 1024, shape=(3,))
	        >>> a
	        jt.Var([439 113  92], dtype=int32)
	        >>> b = jt.randint(0, 10, shape=(3,))
	        >>> b
	        jt.Var([6 8 4], dtype=int32)
	        >>> jt.right_shift(a, b)
	        jt.Var([6 0 5], dtype=int32)'''
	...
def logical_and(x: Var, y: Var)-> Var:
	'''Document:
	*
	    Returns the element-wise logical AND of the inputs. 
	
	    ----------------
	
	    * [in] x: the first input, jt.Var.
	
	    * [in] y: the second input, jt.Var.'''
	...
def logical_or(x: Var, y: Var)-> Var:
	'''Document:
	*
	    Returns the element-wise logical OR of the inputs. 
	
	    ----------------
	
	    * [in] x: the first input, jt.Var.
	
	    * [in] y: the second input, jt.Var.'''
	...
def logical_xor(x: Var, y: Var)-> Var:
	'''Document:
	*
	    Returns the element-wise logical XOR of the inputs. 
	
	    ----------------
	
	    * [in] x: the first input, jt.Var.
	
	    * [in] y: the second input, jt.Var.'''
	...
def bitwise_and(x: Var, y: Var)-> Var:
	'''Document:
	*
	    Computes the bitwise AND of x and y.
	
	    ----------------
	
	    * [in] x: the first input, jt.Var (integal or boolean).
	
	    * [in] y: the second input, jt.Var (integal or boolean).'''
	...
def bitwise_or(x: Var, y: Var)-> Var:
	'''Document:
	*
	    Computes the bitwise OR of x and y.
	
	    ----------------
	
	    * [in] x: the first input, jt.Var (integal or boolean).
	
	    * [in] y: the second input, jt.Var (integal or boolean).'''
	...
def bitwise_xor(x: Var, y: Var)-> Var:
	'''Document:
	*
	    Computes the bitwise XOR of x and y.
	
	    ----------------
	
	    * [in] x: the first input, jt.Var (integal or boolean).
	
	    * [in] y: the second input, jt.Var (integal or boolean).'''
	...
def tape(x: Var)-> Var:
 ...
@overload
def where(cond: Var, dtype: str="int32")-> Tuple[Var]:
	'''Document:
	*
	    Where Operator generate index of true condition.
	
	    * [in] cond:    condition for index generation
	
	    * [in] dtype:   type of return indexes
	    
	    * [out] out:  return an array of indexes, same length with number of dims of cond 
	    
	    Example::
	
	        jt.where([[0,0,1],[1,0,0]])
	        # return [jt.Var([0 1], dtype=int32), jt.Var([2 0], dtype=int32)]'''
	...
@overload
def where(cond: Var, x: Var, y: Var)-> Var:
	'''Document:
	*
	     * Condition operator, perform cond ? x : y
	     *'''
	...
def argsort(x: Var, dim: int=-1, descending: bool=False, dtype: str="int32")-> Tuple[Var]:
	'''Document:
	* 
	    Argsort Operator Perform an indirect sort by given key or compare function.
	
	    x is input, y is output index, satisfy:
	
	        x[y[0]] <= x[y[1]] <= x[y[2]] <= ... <= x[y[n]]
	
	    or
	
	        key(y[0]) <= key(y[1]) <= key(y[2]) <= ... <= key(y[n])
	
	    or
	
	        compare(y[0], y[1]) && compare(y[1], y[2]) && ...
	
	    * [in] x: input var for sort
	
	    * [in] dim: sort alone which dim
	
	    * [in] descending:  the elements are sorted in descending order or not(default False).
	
	    * [in] dtype: type of return indexes
	
	    * [out] index: index have the same size with sorted dim
	
	    * [out] value: sorted value
	
	    
	    Example::
	
	            index, value = jt.argsort([11,13,12])
	            # return [0 2 1], [11 12 13]
	            index, value = jt.argsort([11,13,12], descending=True)
	            # return [1 2 0], [13 12 11]
	            index, value = jt.argsort([[11,13,12], [12,11,13]])
	            # return [[0 2 1],[1 0 2]],  [[11 12 13],[11 12 13]]
	            index, value = jt.argsort([[11,13,12], [12,11,13]], dim=0)
	            # return [[0 1 0],[1 0 1]],  [[11 11 12],[12 13 13]]'''
	...
def fetch(inputs: List[Var], func: Callable)-> Var:
 ...
def arg_reduce(x: Var, op: str, dim: int, keepdims: bool)-> Tuple[Var]:
	'''Document:
	*
	    Returns the indices of the maximum / minimum of the input across a dimension.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] op:      "max" or "min". 
	
	    * [in] dim:     int. Specifies which dimension to be reduced.
	
	    * [in] keepdims: bool. Whether the output has ``dim`` retained or not.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randint(0, 10, shape=(2, 3))
	        >>> x
	        jt.Var([[4 2 5]
	         [6 7 1]], dtype=int32)
	        >>> jt.arg_reduce(x, 'max', dim=1, keepdims=False)
	        [jt.Var([2 1], dtype=int32), jt.Var([5 7], dtype=int32)]
	        >>> jt.arg_reduce(x, 'min', dim=1, keepdims=False)
	        [jt.Var([1 2], dtype=int32), jt.Var([2 1], dtype=int32)]'''
	...
def random(shape: Tuple[int], dtype: str="float32", type: str="uniform")-> Var:
 ...
@overload
def reduce(x: Var, op: str, dim: int, keepdims: bool=False)-> Var:
 ...
@overload
def reduce(x: Var, op: str, dims: Tuple[int]=(), keepdims: bool=False)-> Var:
 ...
@overload
def max(x: Var, dim: int, keepdims: bool=False)-> Var:
	'''Document:
	*
	    Returns the maximum elements in the input.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
	
	    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randint(10, shape=(2, 3))
	        >>> x
	        jt.Var([[4 1 2]
	         [0 2 4]], dtype=int32)
	        >>> jt.max(x)
	        jt.Var([4], dtype=int32)
	        >>> x.max()
	        jt.Var([4], dtype=int32)
	        >>> x.max(dim=1)
	        jt.Var([4 4], dtype=int32)
	        >>> x.max(dim=1, keepdims=True)
	        jt.Var([[4]
	         [4]], dtype=int32)'''
	...
@overload
def max(x: Var, dims: Tuple[int]=(), keepdims: bool=False)-> Var:
	'''Document:
	*
	    Returns the maximum elements in the input.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
	
	    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randint(10, shape=(2, 3))
	        >>> x
	        jt.Var([[4 1 2]
	         [0 2 4]], dtype=int32)
	        >>> jt.max(x)
	        jt.Var([4], dtype=int32)
	        >>> x.max()
	        jt.Var([4], dtype=int32)
	        >>> x.max(dim=1)
	        jt.Var([4 4], dtype=int32)
	        >>> x.max(dim=1, keepdims=True)
	        jt.Var([[4]
	         [4]], dtype=int32)'''
	...
@overload
def max(x: Var, dims_mask: int, keepdims_mask: int)-> Var:
	'''Document:
	*
	    Returns the maximum elements in the input.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
	
	    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randint(10, shape=(2, 3))
	        >>> x
	        jt.Var([[4 1 2]
	         [0 2 4]], dtype=int32)
	        >>> jt.max(x)
	        jt.Var([4], dtype=int32)
	        >>> x.max()
	        jt.Var([4], dtype=int32)
	        >>> x.max(dim=1)
	        jt.Var([4 4], dtype=int32)
	        >>> x.max(dim=1, keepdims=True)
	        jt.Var([[4]
	         [4]], dtype=int32)'''
	...
@overload
def reduce_maximum(x: Var, dim: int, keepdims: bool=False)-> Var:
	'''Document:
	*
	    Returns the maximum elements in the input.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
	
	    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randint(10, shape=(2, 3))
	        >>> x
	        jt.Var([[4 1 2]
	         [0 2 4]], dtype=int32)
	        >>> jt.max(x)
	        jt.Var([4], dtype=int32)
	        >>> x.max()
	        jt.Var([4], dtype=int32)
	        >>> x.max(dim=1)
	        jt.Var([4 4], dtype=int32)
	        >>> x.max(dim=1, keepdims=True)
	        jt.Var([[4]
	         [4]], dtype=int32)'''
	...
@overload
def reduce_maximum(x: Var, dims: Tuple[int]=(), keepdims: bool=False)-> Var:
	'''Document:
	*
	    Returns the maximum elements in the input.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
	
	    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randint(10, shape=(2, 3))
	        >>> x
	        jt.Var([[4 1 2]
	         [0 2 4]], dtype=int32)
	        >>> jt.max(x)
	        jt.Var([4], dtype=int32)
	        >>> x.max()
	        jt.Var([4], dtype=int32)
	        >>> x.max(dim=1)
	        jt.Var([4 4], dtype=int32)
	        >>> x.max(dim=1, keepdims=True)
	        jt.Var([[4]
	         [4]], dtype=int32)'''
	...
@overload
def reduce_maximum(x: Var, dims_mask: int, keepdims_mask: int)-> Var:
	'''Document:
	*
	    Returns the maximum elements in the input.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
	
	    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randint(10, shape=(2, 3))
	        >>> x
	        jt.Var([[4 1 2]
	         [0 2 4]], dtype=int32)
	        >>> jt.max(x)
	        jt.Var([4], dtype=int32)
	        >>> x.max()
	        jt.Var([4], dtype=int32)
	        >>> x.max(dim=1)
	        jt.Var([4 4], dtype=int32)
	        >>> x.max(dim=1, keepdims=True)
	        jt.Var([[4]
	         [4]], dtype=int32)'''
	...
@overload
def min(x: Var, dim: int, keepdims: bool=False)-> Var:
	'''Document:
	*
	    Returns the minimum elements in the input.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
	
	    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randint(10, shape=(2, 3))
	        >>> x
	        jt.Var([[4 1 2]
	         [0 2 4]], dtype=int32)
	        >>> jt.min(x)
	        jt.Var([0], dtype=int32)
	        >>> x.min()
	        jt.Var([0], dtype=int32)
	        >>> x.min(dim=1)
	        jt.Var([1 0], dtype=int32)
	        >>> x.min(dim=1, keepdims=True)
	        jt.Var([[1]
	         [0]], dtype=int32)'''
	...
@overload
def min(x: Var, dims: Tuple[int]=(), keepdims: bool=False)-> Var:
	'''Document:
	*
	    Returns the minimum elements in the input.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
	
	    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randint(10, shape=(2, 3))
	        >>> x
	        jt.Var([[4 1 2]
	         [0 2 4]], dtype=int32)
	        >>> jt.min(x)
	        jt.Var([0], dtype=int32)
	        >>> x.min()
	        jt.Var([0], dtype=int32)
	        >>> x.min(dim=1)
	        jt.Var([1 0], dtype=int32)
	        >>> x.min(dim=1, keepdims=True)
	        jt.Var([[1]
	         [0]], dtype=int32)'''
	...
@overload
def min(x: Var, dims_mask: int, keepdims_mask: int)-> Var:
	'''Document:
	*
	    Returns the minimum elements in the input.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
	
	    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randint(10, shape=(2, 3))
	        >>> x
	        jt.Var([[4 1 2]
	         [0 2 4]], dtype=int32)
	        >>> jt.min(x)
	        jt.Var([0], dtype=int32)
	        >>> x.min()
	        jt.Var([0], dtype=int32)
	        >>> x.min(dim=1)
	        jt.Var([1 0], dtype=int32)
	        >>> x.min(dim=1, keepdims=True)
	        jt.Var([[1]
	         [0]], dtype=int32)'''
	...
@overload
def reduce_minimum(x: Var, dim: int, keepdims: bool=False)-> Var:
	'''Document:
	*
	    Returns the minimum elements in the input.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
	
	    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randint(10, shape=(2, 3))
	        >>> x
	        jt.Var([[4 1 2]
	         [0 2 4]], dtype=int32)
	        >>> jt.min(x)
	        jt.Var([0], dtype=int32)
	        >>> x.min()
	        jt.Var([0], dtype=int32)
	        >>> x.min(dim=1)
	        jt.Var([1 0], dtype=int32)
	        >>> x.min(dim=1, keepdims=True)
	        jt.Var([[1]
	         [0]], dtype=int32)'''
	...
@overload
def reduce_minimum(x: Var, dims: Tuple[int]=(), keepdims: bool=False)-> Var:
	'''Document:
	*
	    Returns the minimum elements in the input.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
	
	    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randint(10, shape=(2, 3))
	        >>> x
	        jt.Var([[4 1 2]
	         [0 2 4]], dtype=int32)
	        >>> jt.min(x)
	        jt.Var([0], dtype=int32)
	        >>> x.min()
	        jt.Var([0], dtype=int32)
	        >>> x.min(dim=1)
	        jt.Var([1 0], dtype=int32)
	        >>> x.min(dim=1, keepdims=True)
	        jt.Var([[1]
	         [0]], dtype=int32)'''
	...
@overload
def reduce_minimum(x: Var, dims_mask: int, keepdims_mask: int)-> Var:
	'''Document:
	*
	    Returns the minimum elements in the input.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
	
	    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randint(10, shape=(2, 3))
	        >>> x
	        jt.Var([[4 1 2]
	         [0 2 4]], dtype=int32)
	        >>> jt.min(x)
	        jt.Var([0], dtype=int32)
	        >>> x.min()
	        jt.Var([0], dtype=int32)
	        >>> x.min(dim=1)
	        jt.Var([1 0], dtype=int32)
	        >>> x.min(dim=1, keepdims=True)
	        jt.Var([[1]
	         [0]], dtype=int32)'''
	...
@overload
def sum(x: Var, dim: int, keepdims: bool=False)-> Var:
	'''Document:
	*
	    Returns the sum of the input.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
	
	    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randint(10, shape=(2, 3))
	        >>> x
	        jt.Var([[4 1 2]
	         [0 2 4]], dtype=int32)
	        >>> jt.sum(x)
	        jt.Var([13], dtype=int32)
	        >>> x.sum()
	        jt.Var([13], dtype=int32)
	        >>> x.sum(dim=1)
	        jt.Var([7 6], dtype=int32)
	        >>> x.sum(dim=1, keepdims=True)
	        jt.Var([[7]
	         [6]], dtype=int32)'''
	...
@overload
def sum(x: Var, dims: Tuple[int]=(), keepdims: bool=False)-> Var:
	'''Document:
	*
	    Returns the sum of the input.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
	
	    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randint(10, shape=(2, 3))
	        >>> x
	        jt.Var([[4 1 2]
	         [0 2 4]], dtype=int32)
	        >>> jt.sum(x)
	        jt.Var([13], dtype=int32)
	        >>> x.sum()
	        jt.Var([13], dtype=int32)
	        >>> x.sum(dim=1)
	        jt.Var([7 6], dtype=int32)
	        >>> x.sum(dim=1, keepdims=True)
	        jt.Var([[7]
	         [6]], dtype=int32)'''
	...
@overload
def sum(x: Var, dims_mask: int, keepdims_mask: int)-> Var:
	'''Document:
	*
	    Returns the sum of the input.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
	
	    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randint(10, shape=(2, 3))
	        >>> x
	        jt.Var([[4 1 2]
	         [0 2 4]], dtype=int32)
	        >>> jt.sum(x)
	        jt.Var([13], dtype=int32)
	        >>> x.sum()
	        jt.Var([13], dtype=int32)
	        >>> x.sum(dim=1)
	        jt.Var([7 6], dtype=int32)
	        >>> x.sum(dim=1, keepdims=True)
	        jt.Var([[7]
	         [6]], dtype=int32)'''
	...
@overload
def reduce_add(x: Var, dim: int, keepdims: bool=False)-> Var:
	'''Document:
	*
	    Returns the sum of the input.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
	
	    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randint(10, shape=(2, 3))
	        >>> x
	        jt.Var([[4 1 2]
	         [0 2 4]], dtype=int32)
	        >>> jt.sum(x)
	        jt.Var([13], dtype=int32)
	        >>> x.sum()
	        jt.Var([13], dtype=int32)
	        >>> x.sum(dim=1)
	        jt.Var([7 6], dtype=int32)
	        >>> x.sum(dim=1, keepdims=True)
	        jt.Var([[7]
	         [6]], dtype=int32)'''
	...
@overload
def reduce_add(x: Var, dims: Tuple[int]=(), keepdims: bool=False)-> Var:
	'''Document:
	*
	    Returns the sum of the input.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
	
	    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randint(10, shape=(2, 3))
	        >>> x
	        jt.Var([[4 1 2]
	         [0 2 4]], dtype=int32)
	        >>> jt.sum(x)
	        jt.Var([13], dtype=int32)
	        >>> x.sum()
	        jt.Var([13], dtype=int32)
	        >>> x.sum(dim=1)
	        jt.Var([7 6], dtype=int32)
	        >>> x.sum(dim=1, keepdims=True)
	        jt.Var([[7]
	         [6]], dtype=int32)'''
	...
@overload
def reduce_add(x: Var, dims_mask: int, keepdims_mask: int)-> Var:
	'''Document:
	*
	    Returns the sum of the input.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
	
	    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randint(10, shape=(2, 3))
	        >>> x
	        jt.Var([[4 1 2]
	         [0 2 4]], dtype=int32)
	        >>> jt.sum(x)
	        jt.Var([13], dtype=int32)
	        >>> x.sum()
	        jt.Var([13], dtype=int32)
	        >>> x.sum(dim=1)
	        jt.Var([7 6], dtype=int32)
	        >>> x.sum(dim=1, keepdims=True)
	        jt.Var([[7]
	         [6]], dtype=int32)'''
	...
@overload
def prod(x: Var, dim: int, keepdims: bool=False)-> Var:
	'''Document:
	*
	    Returns the product of all the elements in the input.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
	
	    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randint(10, shape=(2, 3))
	        >>> x
	        jt.Var([[7 5 5]
	         [5 7 5]], dtype=int32)
	        >>> jt.prod(x)
	        jt.Var([30625], dtype=int32)
	        >>> x.prod()
	        jt.Var([30625], dtype=int32)
	        >>> x.prod(dim=1)
	        jt.Var([175 175], dtype=int32)
	        >>> x.prod(dim=1, keepdims=True)
	        jt.Var([[175]
	         [175]], dtype=int32)'''
	...
@overload
def prod(x: Var, dims: Tuple[int]=(), keepdims: bool=False)-> Var:
	'''Document:
	*
	    Returns the product of all the elements in the input.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
	
	    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randint(10, shape=(2, 3))
	        >>> x
	        jt.Var([[7 5 5]
	         [5 7 5]], dtype=int32)
	        >>> jt.prod(x)
	        jt.Var([30625], dtype=int32)
	        >>> x.prod()
	        jt.Var([30625], dtype=int32)
	        >>> x.prod(dim=1)
	        jt.Var([175 175], dtype=int32)
	        >>> x.prod(dim=1, keepdims=True)
	        jt.Var([[175]
	         [175]], dtype=int32)'''
	...
@overload
def prod(x: Var, dims_mask: int, keepdims_mask: int)-> Var:
	'''Document:
	*
	    Returns the product of all the elements in the input.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
	
	    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randint(10, shape=(2, 3))
	        >>> x
	        jt.Var([[7 5 5]
	         [5 7 5]], dtype=int32)
	        >>> jt.prod(x)
	        jt.Var([30625], dtype=int32)
	        >>> x.prod()
	        jt.Var([30625], dtype=int32)
	        >>> x.prod(dim=1)
	        jt.Var([175 175], dtype=int32)
	        >>> x.prod(dim=1, keepdims=True)
	        jt.Var([[175]
	         [175]], dtype=int32)'''
	...
@overload
def product(x: Var, dim: int, keepdims: bool=False)-> Var:
	'''Document:
	*
	    Returns the product of all the elements in the input.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
	
	    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randint(10, shape=(2, 3))
	        >>> x
	        jt.Var([[7 5 5]
	         [5 7 5]], dtype=int32)
	        >>> jt.prod(x)
	        jt.Var([30625], dtype=int32)
	        >>> x.prod()
	        jt.Var([30625], dtype=int32)
	        >>> x.prod(dim=1)
	        jt.Var([175 175], dtype=int32)
	        >>> x.prod(dim=1, keepdims=True)
	        jt.Var([[175]
	         [175]], dtype=int32)'''
	...
@overload
def product(x: Var, dims: Tuple[int]=(), keepdims: bool=False)-> Var:
	'''Document:
	*
	    Returns the product of all the elements in the input.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
	
	    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randint(10, shape=(2, 3))
	        >>> x
	        jt.Var([[7 5 5]
	         [5 7 5]], dtype=int32)
	        >>> jt.prod(x)
	        jt.Var([30625], dtype=int32)
	        >>> x.prod()
	        jt.Var([30625], dtype=int32)
	        >>> x.prod(dim=1)
	        jt.Var([175 175], dtype=int32)
	        >>> x.prod(dim=1, keepdims=True)
	        jt.Var([[175]
	         [175]], dtype=int32)'''
	...
@overload
def product(x: Var, dims_mask: int, keepdims_mask: int)-> Var:
	'''Document:
	*
	    Returns the product of all the elements in the input.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
	
	    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randint(10, shape=(2, 3))
	        >>> x
	        jt.Var([[7 5 5]
	         [5 7 5]], dtype=int32)
	        >>> jt.prod(x)
	        jt.Var([30625], dtype=int32)
	        >>> x.prod()
	        jt.Var([30625], dtype=int32)
	        >>> x.prod(dim=1)
	        jt.Var([175 175], dtype=int32)
	        >>> x.prod(dim=1, keepdims=True)
	        jt.Var([[175]
	         [175]], dtype=int32)'''
	...
@overload
def reduce_multiply(x: Var, dim: int, keepdims: bool=False)-> Var:
	'''Document:
	*
	    Returns the product of all the elements in the input.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
	
	    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randint(10, shape=(2, 3))
	        >>> x
	        jt.Var([[7 5 5]
	         [5 7 5]], dtype=int32)
	        >>> jt.prod(x)
	        jt.Var([30625], dtype=int32)
	        >>> x.prod()
	        jt.Var([30625], dtype=int32)
	        >>> x.prod(dim=1)
	        jt.Var([175 175], dtype=int32)
	        >>> x.prod(dim=1, keepdims=True)
	        jt.Var([[175]
	         [175]], dtype=int32)'''
	...
@overload
def reduce_multiply(x: Var, dims: Tuple[int]=(), keepdims: bool=False)-> Var:
	'''Document:
	*
	    Returns the product of all the elements in the input.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
	
	    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randint(10, shape=(2, 3))
	        >>> x
	        jt.Var([[7 5 5]
	         [5 7 5]], dtype=int32)
	        >>> jt.prod(x)
	        jt.Var([30625], dtype=int32)
	        >>> x.prod()
	        jt.Var([30625], dtype=int32)
	        >>> x.prod(dim=1)
	        jt.Var([175 175], dtype=int32)
	        >>> x.prod(dim=1, keepdims=True)
	        jt.Var([[175]
	         [175]], dtype=int32)'''
	...
@overload
def reduce_multiply(x: Var, dims_mask: int, keepdims_mask: int)-> Var:
	'''Document:
	*
	    Returns the product of all the elements in the input.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
	
	    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randint(10, shape=(2, 3))
	        >>> x
	        jt.Var([[7 5 5]
	         [5 7 5]], dtype=int32)
	        >>> jt.prod(x)
	        jt.Var([30625], dtype=int32)
	        >>> x.prod()
	        jt.Var([30625], dtype=int32)
	        >>> x.prod(dim=1)
	        jt.Var([175 175], dtype=int32)
	        >>> x.prod(dim=1, keepdims=True)
	        jt.Var([[175]
	         [175]], dtype=int32)'''
	...
@overload
def reduce_logical_and(x: Var, dim: int, keepdims: bool=False)-> Var:
	'''Document:
	*
	    Tests if all elements in input evaluate to True.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
	
	    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randint(2, shape=(2, 3))
	        >>> x
	        jt.Var([[1 1 1]
	         [0 1 0]], dtype=int32)
	        >>> jt.all_(x)
	        jt.Var([False], dtype=int32)
	        >>> x.all_()
	        jt.Var([False], dtype=int32)
	        >>> x.all_(dim=1)
	        jt.Var([True False], dtype=int32)
	        >>> x.all_(dim=1, keepdims=True)
	        jt.Var([[True]
	         [False]], dtype=int32)'''
	...
@overload
def reduce_logical_and(x: Var, dims: Tuple[int]=(), keepdims: bool=False)-> Var:
	'''Document:
	*
	    Tests if all elements in input evaluate to True.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
	
	    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randint(2, shape=(2, 3))
	        >>> x
	        jt.Var([[1 1 1]
	         [0 1 0]], dtype=int32)
	        >>> jt.all_(x)
	        jt.Var([False], dtype=int32)
	        >>> x.all_()
	        jt.Var([False], dtype=int32)
	        >>> x.all_(dim=1)
	        jt.Var([True False], dtype=int32)
	        >>> x.all_(dim=1, keepdims=True)
	        jt.Var([[True]
	         [False]], dtype=int32)'''
	...
@overload
def reduce_logical_and(x: Var, dims_mask: int, keepdims_mask: int)-> Var:
	'''Document:
	*
	    Tests if all elements in input evaluate to True.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
	
	    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randint(2, shape=(2, 3))
	        >>> x
	        jt.Var([[1 1 1]
	         [0 1 0]], dtype=int32)
	        >>> jt.all_(x)
	        jt.Var([False], dtype=int32)
	        >>> x.all_()
	        jt.Var([False], dtype=int32)
	        >>> x.all_(dim=1)
	        jt.Var([True False], dtype=int32)
	        >>> x.all_(dim=1, keepdims=True)
	        jt.Var([[True]
	         [False]], dtype=int32)'''
	...
@overload
def all_(x: Var, dim: int, keepdims: bool=False)-> Var:
	'''Document:
	*
	    Tests if all elements in input evaluate to True.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
	
	    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randint(2, shape=(2, 3))
	        >>> x
	        jt.Var([[1 1 1]
	         [0 1 0]], dtype=int32)
	        >>> jt.all_(x)
	        jt.Var([False], dtype=int32)
	        >>> x.all_()
	        jt.Var([False], dtype=int32)
	        >>> x.all_(dim=1)
	        jt.Var([True False], dtype=int32)
	        >>> x.all_(dim=1, keepdims=True)
	        jt.Var([[True]
	         [False]], dtype=int32)'''
	...
@overload
def all_(x: Var, dims: Tuple[int]=(), keepdims: bool=False)-> Var:
	'''Document:
	*
	    Tests if all elements in input evaluate to True.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
	
	    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randint(2, shape=(2, 3))
	        >>> x
	        jt.Var([[1 1 1]
	         [0 1 0]], dtype=int32)
	        >>> jt.all_(x)
	        jt.Var([False], dtype=int32)
	        >>> x.all_()
	        jt.Var([False], dtype=int32)
	        >>> x.all_(dim=1)
	        jt.Var([True False], dtype=int32)
	        >>> x.all_(dim=1, keepdims=True)
	        jt.Var([[True]
	         [False]], dtype=int32)'''
	...
@overload
def all_(x: Var, dims_mask: int, keepdims_mask: int)-> Var:
	'''Document:
	*
	    Tests if all elements in input evaluate to True.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
	
	    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randint(2, shape=(2, 3))
	        >>> x
	        jt.Var([[1 1 1]
	         [0 1 0]], dtype=int32)
	        >>> jt.all_(x)
	        jt.Var([False], dtype=int32)
	        >>> x.all_()
	        jt.Var([False], dtype=int32)
	        >>> x.all_(dim=1)
	        jt.Var([True False], dtype=int32)
	        >>> x.all_(dim=1, keepdims=True)
	        jt.Var([[True]
	         [False]], dtype=int32)'''
	...
@overload
def reduce_logical_or(x: Var, dim: int, keepdims: bool=False)-> Var:
	'''Document:
	*
	    Tests if any elements in input evaluate to True.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
	
	    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randint(2, shape=(2, 3))
	        >>> x
	        jt.Var([[1 0 1]
	         [0 0 0]], dtype=int32)
	        >>> jt.any_(x)
	        jt.Var([True], dtype=int32)
	        >>> x.any_()
	        jt.Var([True], dtype=int32)
	        >>> x.any_(dim=1)
	        jt.Var([True False], dtype=int32)
	        >>> x.any_(dim=1, keepdims=True)
	        jt.Var([[True]
	         [False]], dtype=int32)'''
	...
@overload
def reduce_logical_or(x: Var, dims: Tuple[int]=(), keepdims: bool=False)-> Var:
	'''Document:
	*
	    Tests if any elements in input evaluate to True.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
	
	    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randint(2, shape=(2, 3))
	        >>> x
	        jt.Var([[1 0 1]
	         [0 0 0]], dtype=int32)
	        >>> jt.any_(x)
	        jt.Var([True], dtype=int32)
	        >>> x.any_()
	        jt.Var([True], dtype=int32)
	        >>> x.any_(dim=1)
	        jt.Var([True False], dtype=int32)
	        >>> x.any_(dim=1, keepdims=True)
	        jt.Var([[True]
	         [False]], dtype=int32)'''
	...
@overload
def reduce_logical_or(x: Var, dims_mask: int, keepdims_mask: int)-> Var:
	'''Document:
	*
	    Tests if any elements in input evaluate to True.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
	
	    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randint(2, shape=(2, 3))
	        >>> x
	        jt.Var([[1 0 1]
	         [0 0 0]], dtype=int32)
	        >>> jt.any_(x)
	        jt.Var([True], dtype=int32)
	        >>> x.any_()
	        jt.Var([True], dtype=int32)
	        >>> x.any_(dim=1)
	        jt.Var([True False], dtype=int32)
	        >>> x.any_(dim=1, keepdims=True)
	        jt.Var([[True]
	         [False]], dtype=int32)'''
	...
@overload
def any_(x: Var, dim: int, keepdims: bool=False)-> Var:
	'''Document:
	*
	    Tests if any elements in input evaluate to True.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
	
	    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randint(2, shape=(2, 3))
	        >>> x
	        jt.Var([[1 0 1]
	         [0 0 0]], dtype=int32)
	        >>> jt.any_(x)
	        jt.Var([True], dtype=int32)
	        >>> x.any_()
	        jt.Var([True], dtype=int32)
	        >>> x.any_(dim=1)
	        jt.Var([True False], dtype=int32)
	        >>> x.any_(dim=1, keepdims=True)
	        jt.Var([[True]
	         [False]], dtype=int32)'''
	...
@overload
def any_(x: Var, dims: Tuple[int]=(), keepdims: bool=False)-> Var:
	'''Document:
	*
	    Tests if any elements in input evaluate to True.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
	
	    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randint(2, shape=(2, 3))
	        >>> x
	        jt.Var([[1 0 1]
	         [0 0 0]], dtype=int32)
	        >>> jt.any_(x)
	        jt.Var([True], dtype=int32)
	        >>> x.any_()
	        jt.Var([True], dtype=int32)
	        >>> x.any_(dim=1)
	        jt.Var([True False], dtype=int32)
	        >>> x.any_(dim=1, keepdims=True)
	        jt.Var([[True]
	         [False]], dtype=int32)'''
	...
@overload
def any_(x: Var, dims_mask: int, keepdims_mask: int)-> Var:
	'''Document:
	*
	    Tests if any elements in input evaluate to True.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
	
	    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randint(2, shape=(2, 3))
	        >>> x
	        jt.Var([[1 0 1]
	         [0 0 0]], dtype=int32)
	        >>> jt.any_(x)
	        jt.Var([True], dtype=int32)
	        >>> x.any_()
	        jt.Var([True], dtype=int32)
	        >>> x.any_(dim=1)
	        jt.Var([True False], dtype=int32)
	        >>> x.any_(dim=1, keepdims=True)
	        jt.Var([[True]
	         [False]], dtype=int32)'''
	...
@overload
def reduce_logical_xor(x: Var, dim: int, keepdims: bool=False)-> Var:
 ...
@overload
def reduce_logical_xor(x: Var, dims: Tuple[int]=(), keepdims: bool=False)-> Var:
 ...
@overload
def reduce_logical_xor(x: Var, dims_mask: int, keepdims_mask: int)-> Var:
 ...
@overload
def reduce_bitwise_and(x: Var, dim: int, keepdims: bool=False)-> Var:
 ...
@overload
def reduce_bitwise_and(x: Var, dims: Tuple[int]=(), keepdims: bool=False)-> Var:
 ...
@overload
def reduce_bitwise_and(x: Var, dims_mask: int, keepdims_mask: int)-> Var:
 ...
@overload
def reduce_bitwise_or(x: Var, dim: int, keepdims: bool=False)-> Var:
 ...
@overload
def reduce_bitwise_or(x: Var, dims: Tuple[int]=(), keepdims: bool=False)-> Var:
 ...
@overload
def reduce_bitwise_or(x: Var, dims_mask: int, keepdims_mask: int)-> Var:
 ...
@overload
def reduce_bitwise_xor(x: Var, dim: int, keepdims: bool=False)-> Var:
 ...
@overload
def reduce_bitwise_xor(x: Var, dims: Tuple[int]=(), keepdims: bool=False)-> Var:
 ...
@overload
def reduce_bitwise_xor(x: Var, dims_mask: int, keepdims_mask: int)-> Var:
 ...
@overload
def mean(x: Var, dim: int, keepdims: bool=False)-> Var:
	'''Document:
	*
	    Returns the mean value of the input.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
	
	    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randint(10, shape=(2, 3))
	        >>> x
	        jt.Var([[9 4 4]
	         [1 9 6]], dtype=int32)
	        >>> jt.mean(x)
	        jt.Var([5.5000005], dtype=float32)
	        >>> x.mean()
	        jt.Var([5.5000005], dtype=float32)
	        >>> x.mean(dim=1)
	        jt.Var([5.666667  5.3333335], dtype=float32)
	        >>> x.mean(dim=1, keepdims=True)
	        jt.Var([[5.666667 ]
	         [5.3333335]], dtype=float32)'''
	...
@overload
def mean(x: Var, dims: Tuple[int]=(), keepdims: bool=False)-> Var:
	'''Document:
	*
	    Returns the mean value of the input.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
	
	    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randint(10, shape=(2, 3))
	        >>> x
	        jt.Var([[9 4 4]
	         [1 9 6]], dtype=int32)
	        >>> jt.mean(x)
	        jt.Var([5.5000005], dtype=float32)
	        >>> x.mean()
	        jt.Var([5.5000005], dtype=float32)
	        >>> x.mean(dim=1)
	        jt.Var([5.666667  5.3333335], dtype=float32)
	        >>> x.mean(dim=1, keepdims=True)
	        jt.Var([[5.666667 ]
	         [5.3333335]], dtype=float32)'''
	...
@overload
def mean(x: Var, dims_mask: int, keepdims_mask: int)-> Var:
	'''Document:
	*
	    Returns the mean value of the input.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
	
	    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randint(10, shape=(2, 3))
	        >>> x
	        jt.Var([[9 4 4]
	         [1 9 6]], dtype=int32)
	        >>> jt.mean(x)
	        jt.Var([5.5000005], dtype=float32)
	        >>> x.mean()
	        jt.Var([5.5000005], dtype=float32)
	        >>> x.mean(dim=1)
	        jt.Var([5.666667  5.3333335], dtype=float32)
	        >>> x.mean(dim=1, keepdims=True)
	        jt.Var([[5.666667 ]
	         [5.3333335]], dtype=float32)'''
	...
def clone(x: Var)-> Var:
 ...
def unary(x: Var, op: str)-> Var:
 ...
def cast(x: Var, op: str)-> Var:
 ...
def int8(x: Var)-> Var:
	'''Document:
	*
	    Returns a copy of the input var, casted to int8.
	
	    ----------------
	
	    * [in] x:   the input jt.Var
	
	    ----------------
	    
	    Example-1::
	        >>> x = jt.rand(3) * 10 
	        >>> x
	        jt.Var([4.093273  2.0086648 8.474352 ], dtype=float32)
	        >>> x.int8()
	        jt.Var([4 2 8], dtype=int8)
	        >>> jt.int8(x)
	        jt.Var([4 2 8], dtype=int8)'''
	...
def int16(x: Var)-> Var:
	'''Document:
	*
	    Returns a copy of the input var, casted to int16.
	
	    ----------------
	
	    * [in] x:   the input jt.Var
	
	    ----------------
	    
	    Example-1::
	        >>> x = jt.rand(3) * 10 
	        >>> x
	        jt.Var([4.093273  2.0086648 8.474352 ], dtype=float32)
	        >>> x.int16()
	        jt.Var([4 2 8], dtype=int16)
	        >>> jt.int16(x)
	        jt.Var([4 2 8], dtype=int16)'''
	...
def int32(x: Var)-> Var:
	'''Document:
	*
	    Returns a copy of the input var, casted to int32.
	
	    ----------------
	
	    * [in] x:   the input jt.Var
	
	    ----------------
	    
	    Example-1::
	        >>> x = jt.rand(3) * 10 
	        >>> x
	        jt.Var([4.093273  2.0086648 8.474352 ], dtype=float32)
	        >>> x.int()
	        jt.Var([4 2 8], dtype=int32)
	        >>> jt.int(x)
	        jt.Var([4 2 8], dtype=int32)
	        >>> x.int32()
	        jt.Var([4 2 8], dtype=int32)
	        >>> jt.int32(x)
	        jt.Var([4 2 8], dtype=int32)
	        >>> x.long()
	        jt.Var([4 2 8], dtype=int32)
	        >>> jt.long(x)
	        jt.Var([4 2 8], dtype=int32)'''
	...
def int64(x: Var)-> Var:
	'''Document:
	*
	    Returns a copy of the input var, casted to int64.
	
	    ----------------
	
	    * [in] x:   the input jt.Var
	
	    ----------------
	    
	    Example-1::
	        >>> x = jt.rand(3) * 10 
	        >>> x
	        jt.Var([4.093273  2.0086648 8.474352 ], dtype=float32)
	        >>> x.int64()
	        jt.Var([4 2 8], dtype=int64)
	        >>> jt.int64(x)
	        jt.Var([4 2 8], dtype=int64)'''
	...
def uint8(x: Var)-> Var:
	'''Document:
	*
	    Returns a copy of the input var, casted to unsigned int8.
	
	    ----------------
	
	    * [in] x:   the input jt.Var
	
	    ----------------
	    
	    Example-1::
	        >>> x = jt.rand(3) * 10 
	        >>> x
	        jt.Var([4.093273  2.0086648 8.474352 ], dtype=float32)
	        >>> x.uint8()
	        jt.Var([4 2 8], dtype=uint8)
	        >>> jt.uint8(x)
	        jt.Var([4 2 8], dtype=uint8)'''
	...
def uint16(x: Var)-> Var:
	'''Document:
	*
	    Returns a copy of the input var, casted to unsigned int16.
	
	    ----------------
	
	    * [in] x:   the input jt.Var
	
	    ----------------
	    
	    Example-1::
	        >>> x = jt.rand(3) * 10 
	        >>> x
	        jt.Var([4.093273  2.0086648 8.474352 ], dtype=float32)
	        >>> x.uint16()
	        jt.Var([4 2 8], dtype=uint16)
	        >>> jt.uint16(x)
	        jt.Var([4 2 8], dtype=uint16)'''
	...
def uint32(x: Var)-> Var:
	'''Document:
	*
	    Returns a copy of the input var, casted to unsigned int32.
	
	    ----------------
	
	    * [in] x:   the input jt.Var
	
	    ----------------
	    
	    Example-1::
	        >>> x = jt.rand(3) * 10 
	        >>> x
	        jt.Var([4.093273  2.0086648 8.474352 ], dtype=float32)
	        >>> x.uint32()
	        jt.Var([4 2 8], dtype=uint32)
	        >>> jt.uint32(x)
	        jt.Var([4 2 8], dtype=uint32)'''
	...
def uint64(x: Var)-> Var:
	'''Document:
	*
	    Returns a copy of the input var, casted to unsigned int64.
	
	    ----------------
	
	    * [in] x:   the input jt.Var
	
	    ----------------
	    
	    Example-1::
	        >>> x = jt.rand(3) * 10 
	        >>> x
	        jt.Var([4.093273  2.0086648 8.474352 ], dtype=float32)
	        >>> x.uint64()
	        jt.Var([4 2 8], dtype=uint64)
	        >>> jt.uint64(x)
	        jt.Var([4 2 8], dtype=uint64)'''
	...
def float16(x: Var)-> Var:
	'''Document:
	*
	    Returns a copy of the input var, casted to float16 (half-precision float).
	
	    ----------------
	
	    * [in] x:   the input jt.Var
	
	    ----------------
	    
	    Example-1::
	        >>> x = jt.rand(3) * 10 
	        >>> x
	        jt.Var([4.093273  2.0086648 8.474352 ], dtype=float32)
	        >>> x.half()
	        jt.Var([4.094 2.008 8.48 ], dtype=float16)
	        >>> jt.half(x)
	        jt.Var([4.094 2.008 8.48 ], dtype=float16)
	        >>> x.float16()
	        jt.Var([4.094 2.008 8.48 ], dtype=float16)
	        >>> jt.float16(x)
	        jt.Var([4.094 2.008 8.48 ], dtype=float16)'''
	...
def float32(x: Var)-> Var:
	'''Document:
	*
	    Returns a copy of the input var, casted to float32.
	
	    ----------------
	
	    * [in] x:   the input jt.Var
	
	    ----------------
	    
	    Example-1::
	        >>> x = jt.arange(3)
	        >>> x
	        jt.Var([0 1 2], dtype=int32)
	        >>> x.float()
	        jt.Var([0. 1. 2.], dtype=float32)
	        >>> jt.float(x) 
	        jt.Var([0. 1. 2.], dtype=float32)
	        >>> x.float32()
	        jt.Var([0. 1. 2.], dtype=float32)
	        >>> jt.float32(x) 
	        jt.Var([0. 1. 2.], dtype=float32)'''
	...
def float64(x: Var)-> Var:
	'''Document:
	*
	    Returns a copy of the input var, casted to float64 (double-precision float).
	
	    ----------------
	
	    * [in] x:   the input jt.Var
	
	    ----------------
	    
	    Example-1::
	        >>> x = jt.arange(3)
	        >>> x
	        jt.Var([0 1 2], dtype=int32)
	        >>> x.double()
	        jt.Var([0. 1. 2.], dtype=float64)
	        >>> jt.double(x) 
	        jt.Var([0. 1. 2.], dtype=float64)
	        >>> x.float64()
	        jt.Var([0. 1. 2.], dtype=float64)
	        >>> jt.float64(x) 
	        jt.Var([0. 1. 2.], dtype=float64)'''
	...
def abs(x: Var)-> Var:
	'''Document:
	*
	    Returns the absolute value of the input ``x``. 
	
	    ----------------
	
	    * [in] x:   the input jt.Var
	
	    ----------------
	    
	    Example-1::
	        >>> jt.abs(jt.float32([-1, 0, 1]))
	        jt.Var([1. 0. 1.], dtype=float32)'''
	...
def negative(x: Var)-> Var:
	'''Document:
	*
	    Returns the negative value of the input ``x``. 
	
	    This operator is equavilant to ``-x``.
	
	    ----------------
	
	    * [in] x:   the input jt.Var.
	
	    ----------------
	    
	    Example-1::
	        >>> jt.negative(jt.float32([-1, 0, 1]))
	        jt.Var([ 1. -0. -1.], dtype=float32)'''
	...
def logical_not(x: Var)-> Var:
	'''Document:
	*
	    Returns the logical NOT of the input ``x``. 
	     
	    ----------------
	
	    * [in] x: the input jt.Var, integal or boolean.
	
	    ----------------
	
	    Example-1::
	        >>> jt.logical_not(jt.int32([-1, 0, 1]))
	        jt.Var([False  True False], dtype=bool)'''
	...
def bitwise_not(x: Var)-> Var:
	'''Document:
	*
	    Returns the bitwise NOT of the input ``x``. 
	
	    ----------------
	
	    * [in] x: the input jt.Var, integal or boolean.
	
	    ----------------
	
	    Example-1::
	        >>> jt.bitwise_not(jt.int32([1, 2, -3]))
	        jt.Var([-2 -3  2], dtype=int32)'''
	...
def log(x: Var)-> Var:
	'''Document:
	*
	    Returns the natural logarithm of the input ``x``. 
	
	    ----------------
	
	    * [in] x: the input jt.Var.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.rand(4) * 2
	        >>> x
	        jt.Var([0.02863695 1.30122    1.6048753  1.140261  ], dtype=float32)
	        >>> jt.log(x)
	        jt.Var([-3.5530574   0.26330233  0.47304606  0.13125724], dtype=float32)
	        >>> x.log()
	        jt.Var([-3.5530574   0.26330233  0.47304606  0.13125724], dtype=float32)'''
	...
def exp(x: Var)-> Var:
	'''Document:
	*
	     Returns the exponential of the input ``x``. 
	
	    ----------------
	
	    * [in] x: the input jt.Var.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.rand(4) * 2
	        >>> x
	        jt.Var([1.9841381 1.4103996 0.5855549 1.4212812], dtype=float32)
	        >>> jt.exp(x)
	        jt.Var([7.2727766 4.0975924 1.7959872 4.1424246], dtype=float32)
	        >>> x.exp()
	        jt.Var([7.2727766 4.0975924 1.7959872 4.1424246], dtype=float32)'''
	...
def sqrt(x: Var)-> Var:
	'''Document:
	*
	    Returns the square root of the input ``x``. 
	
	    ----------------
	
	    * [in] x: the input jt.Var.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.rand(4) * 2
	        >>> x
	        jt.Var([0.81957287 0.5609612  0.07435933 1.7571875 ], dtype=float32)
	        >>> jt.sqrt(x)
	        jt.Var([0.90530264 0.7489734  0.27268907 1.3255895 ], dtype=float32)
	        >>> x.sqrt()
	        jt.Var([0.90530264 0.7489734  0.27268907 1.3255895 ], dtype=float32)'''
	...
def round(x: Var)-> Var:
	'''Document:
	*
	    Returns the closest integer of the input ``x``. 
	
	    ----------------
	
	    * [in] x: the input jt.Var.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randn(4)
	        >>> x
	        jt.Var([ 2.101595    0.33055413 -0.44147047 -0.7720668 ], dtype=float32)
	        >>> jt.round(x)
	        jt.Var([ 2.0  0.0  0.0 -1.0], dtype=float32)
	        >>> x.round()
	        jt.Var([ 2.0  0.0  0.0 -1.0], dtype=float32)'''
	...
def floor(x: Var)-> Var:
	'''Document:
	*
	     Returns the largest integer less than or equal to the input ``x``. 
	
	    ----------------
	
	    * [in] x: the input jt.Var.
	
	    ----------------
	    Example-1::
	        >>> x = jt.randn(4)
	        >>> x
	        jt.Var([-1.0339162 -0.7259972 -0.9220003 -0.8449701], dtype=float32)
	        >>> jt.floor(x)
	        jt.Var([-2.0 -1.0 -1.0 -1.0], dtype=float32)
	        >>> x.floor
	        jt.Var([-2.0 -1.0 -1.0 -1.0], dtype=float32)'''
	...
def ceil(x: Var)-> Var:
	'''Document:
	*
	    Returns the smallest integer greater than or equal to the input ``x``. 
	
	    ----------------
	
	    * [in] x: the input jt.Var.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randn(4)
	        >>> x
	        jt.Var([-1.0339162 -0.7259972 -0.9220003 -0.8449701], dtype=float32)
	        >>> jt.ceil(x)
	        jt.Var([-1.0  0.0  0.0  0.0], dtype=float32)
	        >>> x.ceil()
	        jt.Var([-1.0  0.0  0.0  0.0], dtype=float32)'''
	...
def round_int(x: Var)-> Var:
	'''Document:
	*
	    Returns the closest integer of the input ``x``. 
	
	    ----------------
	
	    * [in] x: the input jt.Var.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randn(4)
	        >>> x
	        jt.Var([ 2.101595    0.33055413 -0.44147047 -0.7720668 ], dtype=float32)
	        >>> jt.round_int(x)
	        jt.Var([ 2  0  0 -1], dtype=int32)
	        >>> x.round_int
	        jt.Var([ 2  0  0 -1], dtype=int32)'''
	...
def floor_int(x: Var)-> Var:
	'''Document:
	*
	     Returns the largest integer less than or equal to the input ``x``. 
	
	    ----------------
	
	    * [in] x: the input jt.Var.
	
	    ----------------
	    Example-1::
	        >>> x = jt.randn(4)
	        >>> x
	        jt.Var([-1.0339162 -0.7259972 -0.9220003 -0.8449701], dtype=float32)
	        >>> jt.floor_int(x)
	        jt.Var([-2 -1 -1 -1], dtype=int32)
	        >>> x.floor_int
	        jt.Var([-2 -1 -1 -1], dtype=int32)'''
	...
def ceil_int(x: Var)-> Var:
	'''Document:
	*
	    Returns the smallest integer greater than or equal to the input ``x``. 
	
	    ----------------
	
	    * [in] x: the input jt.Var.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randn(4)
	        >>> x
	        jt.Var([-1.0339162 -0.7259972 -0.9220003 -0.8449701], dtype=float32)
	        >>> jt.ceil_int(x)
	        jt.Var([-1  0  0  0], dtype=int32)
	        >>> x.ceil_int()
	        jt.Var([-1  0  0  0], dtype=int32)'''
	...
def sin(x: Var)-> Var:
	'''Document:
	*
	    Returns the sine of the input ``x``. 
	
	    ----------------
	
	    * [in] x: the input jt.Var.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randn(4)
	        >>> x
	        jt.Var([ 0.32893723 -0.7112559  -0.872391    1.8001337 ], dtype=float32)
	        >>> jt.sin(x)
	        jt.Var([ 0.32303742 -0.6527857  -0.76586854  0.9738172 ], dtype=float32)
	        >>> x.sin()
	        jt.Var([ 0.32303742 -0.6527857  -0.76586854  0.9738172 ], dtype=float32)'''
	...
def asin(x: Var)-> Var:
	'''Document:
	*
	    Returns the arcsine of the input ``x``. 
	
	    ----------------
	
	    * [in] x: the input jt.Var.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randn(4)
	        >>> x
	        jt.Var([ 0.09342023 -0.42522037  0.9264933  -0.785264  ], dtype=float32)
	        >>> jt.asin(x)
	        jt.Var([ 0.09355665 -0.43920535  1.1849847  -0.9031224 ], dtype=float32)
	        >>> x.asin()
	        jt.Var([ 0.09355665 -0.43920535  1.1849847  -0.9031224 ], dtype=float32)'''
	...
def arcsin(x: Var)-> Var:
	'''Document:
	*
	    Returns the arcsine of the input ``x``. 
	
	    ----------------
	
	    * [in] x: the input jt.Var.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randn(4)
	        >>> x
	        jt.Var([ 0.09342023 -0.42522037  0.9264933  -0.785264  ], dtype=float32)
	        >>> jt.asin(x)
	        jt.Var([ 0.09355665 -0.43920535  1.1849847  -0.9031224 ], dtype=float32)
	        >>> x.asin()
	        jt.Var([ 0.09355665 -0.43920535  1.1849847  -0.9031224 ], dtype=float32)'''
	...
def sinh(x: Var)-> Var:
	'''Document:
	*
	    Returns the hyperbolic sine of the input ``x``. 
	
	    ----------------
	
	    * [in] x: the input jt.Var.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randn(4)
	        >>> x
	        jt.Var([ 0.32893723 -0.7112559  -0.872391    1.8001337 ], dtype=float32)
	        >>> jt.sinh(x)
	        jt.Var([ 0.3349012  -0.77276015 -0.9873369   2.9425898 ], dtype=float32)
	        >>> x.sinh
	        jt.Var([ 0.3349012  -0.77276015 -0.9873369   2.9425898 ], dtype=float32)'''
	...
def asinh(x: Var)-> Var:
	'''Document:
	*
	    Returns the inverse hyperbolic sine of the input ``x``. 
	
	    ----------------
	
	    * [in] x: the input jt.Var.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randn(4)
	        >>> x
	        jt.Var([-1.9749726  -0.52341473  0.8906148   1.0338128 ], dtype=float32)
	        >>> jt.asinh(x)
	        jt.Var([-1.4323865  -0.5020559   0.8018747   0.90508187], dtype=float32)
	        >>> x.asinh()
	        jt.Var([-1.4323865  -0.5020559   0.8018747   0.90508187], dtype=float32)'''
	...
def arcsinh(x: Var)-> Var:
	'''Document:
	*
	    Returns the inverse hyperbolic sine of the input ``x``. 
	
	    ----------------
	
	    * [in] x: the input jt.Var.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randn(4)
	        >>> x
	        jt.Var([-1.9749726  -0.52341473  0.8906148   1.0338128 ], dtype=float32)
	        >>> jt.asinh(x)
	        jt.Var([-1.4323865  -0.5020559   0.8018747   0.90508187], dtype=float32)
	        >>> x.asinh()
	        jt.Var([-1.4323865  -0.5020559   0.8018747   0.90508187], dtype=float32)'''
	...
def tan(x: Var)-> Var:
	'''Document:
	*
	    Returns the tangent of the input ``x``. 
	
	    ----------------
	
	    * [in] x: the input jt.Var.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randn(4)
	        >>> x
	        jt.Var([ 0.32893723 -0.7112559  -0.872391    1.8001337 ], dtype=float32)
	        >>> jt.tan(x)
	        jt.Var([ 0.34133783 -0.8617148  -1.1910915  -4.283673  ], dtype=float32)
	        >>> x.tan()
	        jt.Var([ 0.34133783 -0.8617148  -1.1910915  -4.283673  ], dtype=float32)'''
	...
def atan(x: Var)-> Var:
	'''Document:
	*
	    Returns the inverse tangent of the input ``x``. 
	
	    ----------------
	
	    * [in] x: the input jt.Var.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randn(4)
	        >>> x
	        jt.Var([-0.85885596  1.187804    0.47249675  0.95933187], dtype=float32)
	        >>> jt.atan(x)
	        jt.Var([-0.70961297  0.87102956  0.44140393  0.76464504], dtype=float32)
	        >>> x.atan()
	        jt.Var([-0.70961297  0.87102956  0.44140393  0.76464504], dtype=float32)'''
	...
def arctan(x: Var)-> Var:
	'''Document:
	*
	    Returns the inverse tangent of the input ``x``. 
	
	    ----------------
	
	    * [in] x: the input jt.Var.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randn(4)
	        >>> x
	        jt.Var([-0.85885596  1.187804    0.47249675  0.95933187], dtype=float32)
	        >>> jt.atan(x)
	        jt.Var([-0.70961297  0.87102956  0.44140393  0.76464504], dtype=float32)
	        >>> x.atan()
	        jt.Var([-0.70961297  0.87102956  0.44140393  0.76464504], dtype=float32)'''
	...
def tanh(x: Var)-> Var:
	'''Document:
	*
	    Returns the hyperbolic tangent of the input ``x``. 
	
	    ----------------
	
	    * [in] x: the input jt.Var.
	
	    ----------------
	    
	    Example-1::
	        >>> x = jt.randn(4)
	        >>> x
	        jt.Var([-0.85885596  1.187804    0.47249675  0.95933187], dtype=float32)
	        >>> jt.tanh(x)
	        jt.Var([-0.6956678   0.82989657  0.4402144   0.7439787 ], dtype=float32)
	        >>> x.tanh()
	        jt.Var([-0.6956678   0.82989657  0.4402144   0.7439787 ], dtype=float32)'''
	...
def atanh(x: Var)-> Var:
	'''Document:
	*
	    Returns the inverse hyperbolic tangent of the input ``x``. 
	
	    ----------------
	
	    * [in] x: the input jt.Var.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.rand(4) * 2 - 1
	        >>> x
	        jt.Var([ 0.9062414  -0.799802   -0.27219176 -0.7274077 ], dtype=float32)
	        >>> jt.atanh(x)
	        jt.Var([ 1.5060828  -1.0980625  -0.27922946 -0.9231999 ], dtype=float32)
	        >>> x.atanh()
	        jt.Var([ 1.5060828  -1.0980625  -0.27922946 -0.9231999 ], dtype=float32)'''
	...
def arctanh(x: Var)-> Var:
	'''Document:
	*
	    Returns the inverse hyperbolic tangent of the input ``x``. 
	
	    ----------------
	
	    * [in] x: the input jt.Var.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.rand(4) * 2 - 1
	        >>> x
	        jt.Var([ 0.9062414  -0.799802   -0.27219176 -0.7274077 ], dtype=float32)
	        >>> jt.atanh(x)
	        jt.Var([ 1.5060828  -1.0980625  -0.27922946 -0.9231999 ], dtype=float32)
	        >>> x.atanh()
	        jt.Var([ 1.5060828  -1.0980625  -0.27922946 -0.9231999 ], dtype=float32)'''
	...
def cos(x: Var)-> Var:
	'''Document:
	*
	    Returns the cosine of the input ``x``. 
	
	    ----------------
	
	    * [in] x: the input jt.Var.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randn(4)
	        >>> x
	        jt.Var([ 0.32893723 -0.7112559  -0.872391    1.8001337 ], dtype=float32)
	        >>> jt.cos(x)
	        jt.Var([ 0.9463862  0.7575426  0.6429972 -0.2273323], dtype=float32)
	        >>> x.cos()
	        jt.Var([ 0.9463862  0.7575426  0.6429972 -0.2273323], dtype=float32)'''
	...
def acos(x: Var)-> Var:
	'''Document:
	*
	    Returns the inverse cosine of the input ``x``. 
	
	    ----------------
	
	    * [in] x: the input jt.Var.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.rand(4) * 2 - 1
	        >>> x
	        jt.Var([ 0.5876564  0.740723  -0.667666   0.5371753], dtype=float32)
	        >>> jt.acos(x)
	        jt.Var([0.9426371 0.7366504 2.3018656 1.0037117], dtype=float32)
	        >>> x.acos()
	        jt.Var([0.9426371 0.7366504 2.3018656 1.0037117], dtype=float32)'''
	...
def arccos(x: Var)-> Var:
	'''Document:
	*
	    Returns the inverse cosine of the input ``x``. 
	
	    ----------------
	
	    * [in] x: the input jt.Var.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.rand(4) * 2 - 1
	        >>> x
	        jt.Var([ 0.5876564  0.740723  -0.667666   0.5371753], dtype=float32)
	        >>> jt.acos(x)
	        jt.Var([0.9426371 0.7366504 2.3018656 1.0037117], dtype=float32)
	        >>> x.acos()
	        jt.Var([0.9426371 0.7366504 2.3018656 1.0037117], dtype=float32)'''
	...
def cosh(x: Var)-> Var:
	'''Document:
	*
	    Returns the hyperbolic cosine of the input ``x``. 
	
	    ----------------
	
	    * [in] x: the input jt.Var.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randn(4)
	        >>> x
	        jt.Var([ 0.32893723 -0.7112559  -0.872391    1.8001337 ], dtype=float32)
	        >>> jt.cosh(x)
	        jt.Var([1.0545894 1.2637873 1.405288  3.1078668], dtype=float32)
	        >>> x.cosh()
	        jt.Var([1.0545894 1.2637873 1.405288  3.1078668], dtype=float32)'''
	...
def acosh(x: Var)-> Var:
	'''Document:
	*
	    Returns the inverse hyperbolic cosine of the input ``x``. 
	
	    ----------------
	
	    * [in] x: the input jt.Var.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.rand(4) + 1
	        >>> x
	        jt.Var([1.3609099 1.8137748 1.1146184 1.3911307], dtype=float32)
	        >>> jt.acosh(x)
	        jt.Var([0.8259237  1.2020639  0.47432774 0.8579033 ], dtype=float32)
	        >>> x.acosh()
	        jt.Var([0.8259237  1.2020639  0.47432774 0.8579033 ], dtype=float32)'''
	...
def arccosh(x: Var)-> Var:
	'''Document:
	*
	    Returns the inverse hyperbolic cosine of the input ``x``. 
	
	    ----------------
	
	    * [in] x: the input jt.Var.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.rand(4) + 1
	        >>> x
	        jt.Var([1.3609099 1.8137748 1.1146184 1.3911307], dtype=float32)
	        >>> jt.acosh(x)
	        jt.Var([0.8259237  1.2020639  0.47432774 0.8579033 ], dtype=float32)
	        >>> x.acosh()
	        jt.Var([0.8259237  1.2020639  0.47432774 0.8579033 ], dtype=float32)'''
	...
def sigmoid(x: Var)-> Var:
	'''Document:
	*
	    Returns the sigmoid of the input ``x``. 
	    
	    .. math::
	       out_i = \frac{1}{1 + e^{x_i}}
	
	    ----------------
	
	    * [in] x: the input jt.Var.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randn(4)
	        >>> x
	        jt.Var([ 0.49443012  0.4305426  -1.0364404  -1.2628382 ], dtype=float32)
	        >>> jt.sigmoid(x)
	        jt.Var([0.62114954 0.6060032  0.2618374  0.2204857 ], dtype=float32)
	        >>> x.sigmoid()
	        jt.Var([0.62114954 0.6060032  0.2618374  0.2204857 ], dtype=float32)'''
	...
def erf(x: Var)-> Var:
	'''Document:
	*
	    Computes the error function of each element. The error function is defined as follows:
	
	    .. math::
	        erf(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} dt
	
	    ----------------
	
	    * [in] x: the input jt.Var.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randn(4)
	        >>> x
	        jt.Var([ 0.49443012  0.4305426  -1.0364404  -1.2628382 ], dtype=float32)
	        >>> jt.erf(x)
	        jt.Var([ 0.51559156  0.45739546 -0.85728306 -0.9258883 ], dtype=float32)
	        >>> x.erf()
	        jt.Var([ 0.51559156  0.45739546 -0.85728306 -0.9258883 ], dtype=float32)'''
	...
def erfinv(x: Var)-> Var:
	'''Document:
	*
	    Computes the inverse error function of each element. 
	
	    * [in] x: the input jt.Var.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.rand(4) * 2 - 1 
	        >>> x
	        jt.Var([ 0.00277209 -0.26642472  0.7869792   0.5415418 ], dtype=float32)
	        >>> jt.erfinv(x)
	        jt.Var([ 0.00245671 -0.24068035  0.8805613   0.5242405 ], dtype=float32)
	        >>> x.erfinv()
	        jt.Var([ 0.00245671 -0.24068035  0.8805613   0.5242405 ], dtype=float32)'''
	...
def transpose(x: Var, axes: Tuple[int]=())-> Var:
 ...
def fuse_transpose(x: Var, axes: Tuple[int]=())-> Var:
 ...
def safe_clip(x: Var, left: float, right: float)-> Var:
	'''Document:
	* Safe clip value to a range, and keep 
	 the gradient pass thought.
	 
	    * [in] x:   input value
	    * [in] left: float64 clip min value.
	    * [in] right: float64 clip max value.'''
	...
def array_(args: numpy.ndarray)-> Var:
 ...
def array(obj: float | int | numpy.ndarray | Var)-> Var:
 ...
@overload
def getitem(x: Var, slices: slice)-> Var:
 ...
@overload
def getitem(x: Var, slices: slice, _: int)-> Tuple[Var]:
 ...
def candidate(x: Var, fail_cond: str, dtype: str="int32")-> Var:
	'''Document:
	*
	    Candidate Operator Perform an indirect candidate filter by given a fail condition.
	    
	    x is input, y is output index, satisfy::
	
	        not fail_cond(y[0], y[1]) and
	        not fail_cond(y[0], y[2]) and not fail_cond(y[1], y[2]) and
	        ...
	        ... and not fail_cond(y[m-2], y[m-1])
	
	    Where m is number of selected candidates.
	
	    Pseudo code::
	    
	        y = []
	        for i in range(n):
	            pass = True
	            for j in y:
	                if (@fail_cond):
	                    pass = false
	                    break
	            if (pass):
	                y.append(i)
	        return y
	
	    * [in] x:   input var for filter
	
	    * [in] fail_cond:   code for fail condition
	
	    * [in] dtype:   type of return indexes
	
	    * [out] index: .
	
	    Example::
	
	        jt.candidate(jt.random(100,2), '(@x(j,0)>@x(i,0))or(@x(j,1)>@x(i,1))')
	        # return y satisfy:
	        #    x[y[0], 0] <= x[y[1], 0] and x[y[1], 0] <= x[y[2], 0] and ... and x[y[m-2], 0] <= x[y[m-1], 0] and
	        #    x[y[0], 1] <= x[y[1], 1] and x[y[1], 1] <= x[y[2], 1] and ... and x[y[m-2], 1] <= x[y[m-1], 1]'''
	...
@overload
def numpy_code(shape: Tuple[int], dtype: str, inputs: List[Var], forward: Callable, backward: List[Callable])-> Var:
	'''Document:
	*
	    Numpy Code Operator for easily customized op.
	
	    ----------------
	
	    * [in] shape:    the output shape, a integer array
	    
	    * [in] dtype:    the output data type
	    
	    * [in] inputs:   A list of input jittor Vars
	
	    * [in] forward:  function, represents forward python function
	
	    * [in] backward: A list of function, represents gradiant for each input
	
	    ----------------
	    
	    Example-1::
	
	        def forward_code(np, data):
	            a = data["inputs"][0]
	            b = data["outputs"][0]
	            np.add(a,a,out=b)
	
	        def backward_code(np, data):
	            dout = data["dout"]
	            out = data["outputs"][0]
	            np.copyto(out, dout*2.0)
	
	        a = jt.random((5,1))
	        b = jt.numpy_code(
	            a.shape,
	            a.dtype,
	            [a],
	            forward_code,
	            [backward_code],
	        )
	
	    Example-2::
	    
	        def forward_code(np, data):
	            a,b = data["inputs"]
	            c,d = data["outputs"]
	            np.add(a,b,out=c)
	            np.subtract(a,b,out=d)
	
	        def backward_code1(np, data):
	            dout = data["dout"]
	            out = data["outputs"][0]
	            np.copyto(out, dout)
	
	        def backward_code2(np, data):
	            dout = data["dout"]
	            out_index = data["out_index"]
	            out = data["outputs"][0]
	            if out_index==0:
	                np.copyto(out, dout)
	            else:
	                np.negative(dout, out)
	
	        a = jt.random((5,1))
	        b = jt.random((5,1))
	        c, d = jt.numpy_code(
	            [a.shape, a.shape],
	            [a.dtype, a.dtype],
	            [a, b],
	            forward_code,
	            [backward_code1,backward_code2],
	        )'''
	...
@overload
def numpy_code(shapes: List[Tuple[int]], dtypes: List[str], inputs: List[Var], forward: Callable, backward: List[Callable])-> Tuple[Var]:
	'''Document:
	*
	    Numpy Code Operator for easily customized op.
	
	    ----------------
	
	    * [in] shape:    the output shape, a integer array
	    
	    * [in] dtype:    the output data type
	    
	    * [in] inputs:   A list of input jittor Vars
	
	    * [in] forward:  function, represents forward python function
	
	    * [in] backward: A list of function, represents gradiant for each input
	
	    ----------------
	    
	    Example-1::
	
	        def forward_code(np, data):
	            a = data["inputs"][0]
	            b = data["outputs"][0]
	            np.add(a,a,out=b)
	
	        def backward_code(np, data):
	            dout = data["dout"]
	            out = data["outputs"][0]
	            np.copyto(out, dout*2.0)
	
	        a = jt.random((5,1))
	        b = jt.numpy_code(
	            a.shape,
	            a.dtype,
	            [a],
	            forward_code,
	            [backward_code],
	        )
	
	    Example-2::
	    
	        def forward_code(np, data):
	            a,b = data["inputs"]
	            c,d = data["outputs"]
	            np.add(a,b,out=c)
	            np.subtract(a,b,out=d)
	
	        def backward_code1(np, data):
	            dout = data["dout"]
	            out = data["outputs"][0]
	            np.copyto(out, dout)
	
	        def backward_code2(np, data):
	            dout = data["dout"]
	            out_index = data["out_index"]
	            out = data["outputs"][0]
	            if out_index==0:
	                np.copyto(out, dout)
	            else:
	                np.negative(dout, out)
	
	        a = jt.random((5,1))
	        b = jt.random((5,1))
	        c, d = jt.numpy_code(
	            [a.shape, a.shape],
	            [a.dtype, a.dtype],
	            [a, b],
	            forward_code,
	            [backward_code1,backward_code2],
	        )'''
	...
@overload
def numpy_code(shape: Tuple[int], dtype: str, inputs: List[Var], forward: Callable)-> Var:
	'''Document:
	*
	    Numpy Code Operator for easily customized op.
	
	    ----------------
	
	    * [in] shape:    the output shape, a integer array
	    
	    * [in] dtype:    the output data type
	    
	    * [in] inputs:   A list of input jittor Vars
	
	    * [in] forward:  function, represents forward python function
	
	    * [in] backward: A list of function, represents gradiant for each input
	
	    ----------------
	    
	    Example-1::
	
	        def forward_code(np, data):
	            a = data["inputs"][0]
	            b = data["outputs"][0]
	            np.add(a,a,out=b)
	
	        def backward_code(np, data):
	            dout = data["dout"]
	            out = data["outputs"][0]
	            np.copyto(out, dout*2.0)
	
	        a = jt.random((5,1))
	        b = jt.numpy_code(
	            a.shape,
	            a.dtype,
	            [a],
	            forward_code,
	            [backward_code],
	        )
	
	    Example-2::
	    
	        def forward_code(np, data):
	            a,b = data["inputs"]
	            c,d = data["outputs"]
	            np.add(a,b,out=c)
	            np.subtract(a,b,out=d)
	
	        def backward_code1(np, data):
	            dout = data["dout"]
	            out = data["outputs"][0]
	            np.copyto(out, dout)
	
	        def backward_code2(np, data):
	            dout = data["dout"]
	            out_index = data["out_index"]
	            out = data["outputs"][0]
	            if out_index==0:
	                np.copyto(out, dout)
	            else:
	                np.negative(dout, out)
	
	        a = jt.random((5,1))
	        b = jt.random((5,1))
	        c, d = jt.numpy_code(
	            [a.shape, a.shape],
	            [a.dtype, a.dtype],
	            [a, b],
	            forward_code,
	            [backward_code1,backward_code2],
	        )'''
	...
@overload
def numpy_code(shapes: List[Tuple[int]], dtypes: List[str], inputs: List[Var], forward: Callable)-> Tuple[Var]:
	'''Document:
	*
	    Numpy Code Operator for easily customized op.
	
	    ----------------
	
	    * [in] shape:    the output shape, a integer array
	    
	    * [in] dtype:    the output data type
	    
	    * [in] inputs:   A list of input jittor Vars
	
	    * [in] forward:  function, represents forward python function
	
	    * [in] backward: A list of function, represents gradiant for each input
	
	    ----------------
	    
	    Example-1::
	
	        def forward_code(np, data):
	            a = data["inputs"][0]
	            b = data["outputs"][0]
	            np.add(a,a,out=b)
	
	        def backward_code(np, data):
	            dout = data["dout"]
	            out = data["outputs"][0]
	            np.copyto(out, dout*2.0)
	
	        a = jt.random((5,1))
	        b = jt.numpy_code(
	            a.shape,
	            a.dtype,
	            [a],
	            forward_code,
	            [backward_code],
	        )
	
	    Example-2::
	    
	        def forward_code(np, data):
	            a,b = data["inputs"]
	            c,d = data["outputs"]
	            np.add(a,b,out=c)
	            np.subtract(a,b,out=d)
	
	        def backward_code1(np, data):
	            dout = data["dout"]
	            out = data["outputs"][0]
	            np.copyto(out, dout)
	
	        def backward_code2(np, data):
	            dout = data["dout"]
	            out_index = data["out_index"]
	            out = data["outputs"][0]
	            if out_index==0:
	                np.copyto(out, dout)
	            else:
	                np.negative(dout, out)
	
	        a = jt.random((5,1))
	        b = jt.random((5,1))
	        c, d = jt.numpy_code(
	            [a.shape, a.shape],
	            [a.dtype, a.dtype],
	            [a, b],
	            forward_code,
	            [backward_code1,backward_code2],
	        )'''
	...
@overload
def code(shape: Tuple[int], dtype: str, inputs: List[Var]={}, cpu_src: str="", cpu_grad_src: List[str]={}, cpu_header: str="", cuda_src: str="", cuda_grad_src: List[str]={}, cuda_header: str="")-> Var:
	'''Document:
	*
	    Code Operator for easily customized op.
	
	    ----------------
	
	    * [in] shape:   the output shape, a integer array
	    
	    * [in] dtype:   the output data type
	    
	    * [in] inputs:  A list of input jittor Vars
	    
	    * [in] cpu_src: cpu source code string, buildin value:
	
	            *   in{x}, in{x}_shape{y}, in{x}_stride{y}, in{x}_type, in{x}_p, @in0(...)
	            *   out{x}, out{x}_shape{y}, out{x}_stride{y}, out{x}_type, out{x}_p, @out0(...)
	            *   out, out_shape{y}, out_stride{y}, out_type, out_p, @out(...)
	    
	    * [in] cpu_header: cpu header code string.
	
	    * [in] cuda_src: cuda source code string.
	
	    * [in] cuda_header: cuda header code string.
	
	    ----------------
	    
	    Example-1::
	
	        from jittor import Function
	        import jittor as jt
	
	        class Func(Function):
	            def execute(self, x):
	                self.save_vars = x
	                return jt.code(x.shape, x.dtype, [x],
	                    cpu_src="""
	                        for (int i=0; i<in0_shape0; i++)
	                            @out(i) = @in0(i)*@in0(i)*2;
	                    """)
	
	            def grad(self, grad_x):
	                x = self.save_vars
	                return jt.code(x.shape, x.dtype, [x, grad_x],
	                    cpu_src="""
	                        for (int i=0; i<in0_shape0; i++)
	                            @out(i) = @in1(i)*@in0(i)*4;
	                    """)
	
	        a = jt.random([10])
	        func = Func()
	        b = func(a)
	        print(b)
	        print(jt.grad(b,a))
	
	    Example-2::
	
	        a = jt.array([3,2,1])
	        b = jt.code(a.shape, a.dtype, [a],
	            cpu_header="""
	                #include <algorithm>
	                @alias(a, in0)
	                @alias(b, out)
	            """,
	            cpu_src="""
	                for (int i=0; i<a_shape0; i++)
	                    @b(i) = @a(i);
	                std::sort(&@b(0), &@b(in0_shape0));
	            """
	        )
	        assert (b.data==[1,2,3]).all()
	
	    Example-3::
	
	        #This example shows how to set multiple outputs in code op.
	        a = jt.array([3,2,1])
	        b,c = jt.code([(1,), (1,)], [a.dtype, a.dtype], [a],
	            cpu_header="""
	                #include <iostream>
	                using namespace std;
	            """,
	            cpu_src="""
	                @alias(a, in0)
	                @alias(b, out0)
	                @alias(c, out1)
	                @b(0) = @c(0) = @a(0);
	                for (int i=0; i<a_shape0; i++) {
	                    @b(0) = std::min(@b(0), @a(i));
	                    @c(0) = std::max(@c(0), @a(i));
	                }
	                cout << "min:" << @b(0) << " max:" << @c(0) << endl;
	            """
	        )
	        assert b.data == 1, b
	        assert c.data == 3, c
	
	    Example-4::
	
	        #This example shows how to use dynamic shape of jittor variables.
	        a = jt.array([5,-4,3,-2,1])
	        
	        # negtive shape for max size of vary dimension
	        b,c = jt.code([(-5,), (-5,)], [a.dtype, a.dtype], [a],
	            cpu_src="""
	                @alias(a, in0)
	                @alias(b, out0)
	                @alias(c, out1)
	                int num_b=0, num_c=0;
	                for (int i=0; i<a_shape0; i++) {
	                    if (@a(i)>0)
	                        @b(num_b++) = @a(i);
	                    else
	                        @c(num_c++) = @a(i);
	                }
	                b->set_shape({num_b});
	                c->set_shape({num_c});
	            """
	        )
	        assert (b.data == [5,3,1]).all()
	        assert (c.data == [-4,-2]).all()
	
	    Example-5::
	
	        # This example shows how to customize code op
	        # compilation flags, such as add include search
	        # path, add definitions, or any command line options
	
	        a = jt.random([10])
	        b = jt.code(a.shape, a.dtype, [a],
	            cpu_src="""
	                @out0(0) = HAHAHA;
	            """)
	        # HAHAHA is defined in flags below
	        # /any/include/path can be change to any path you want to include
	        b.compile_options = {"FLAGS: -DHAHAHA=233 -I/any/include/path ": 1}
	        print(b[0])
	        # will output 233
	
	
	    CUDA Example-1::
	
	        #This example shows how to use CUDA in code op.
	        import jittor as jt
	        from jittor import Function
	        jt.flags.use_cuda = 1
	
	        class Func(Function):
	            def execute(self, a, b):
	                self.save_vars = a, b
	                return jt.code(a.shape, a.dtype, [a,b],
	                    cuda_src="""
	                        __global__ static void kernel1(@ARGS_DEF) {
	                            @PRECALC
	                            int i = threadIdx.x + blockIdx.x * blockDim.x;
	                            int stride = blockDim.x * gridDim.x;
	                            for (; i<in0_shape0; i+=stride)
	                                @out(i) = @in0(i)*@in1(i);
	                        }
	                        kernel1<<<(in0_shape0-1)/1024+1, 1024>>>(@ARGS);
	                    """)
	
	            def grad(self, grad):
	                a, b = self.save_vars
	                return jt.code([a.shape, b.shape], [a.dtype, b.dtype], [a, b, grad],
	                    cuda_src="""
	                        __global__ static void kernel2(@ARGS_DEF) {
	                            @PRECALC
	                            int i = threadIdx.x + blockIdx.x * blockDim.x;
	                            int stride = blockDim.x * gridDim.x;
	                            for (; i<in0_shape0; i+=stride) {
	                                @out0(i) = @in2(i)*@in1(i);
	                                @out1(i) = @in2(i)*@in0(i);
	                            }
	                        }
	                        kernel2<<<(in0_shape0-1)/1024+1, 1024>>>(@ARGS);
	                    """)
	                
	        a = jt.random([100000])
	        b = jt.random([100000])
	        func = Func()
	        c = func(a,b)
	        print(c)
	        print(jt.grad(c, [a, b]))
	
	    CUDA Example-2::
	    
	        #This example shows how to use multi dimension data with CUDA.
	        import jittor as jt
	        from jittor import Function
	        jt.flags.use_cuda = 1
	
	        class Func(Function):
	            def execute(self, a, b):
	                self.save_vars = a, b
	                return jt.code(a.shape, a.dtype, [a,b],
	                    cuda_src="""
	                        __global__ static void kernel1(@ARGS_DEF) {
	                            @PRECALC
	                            for (int i=blockIdx.x; i<in0_shape0; i+=gridDim.x)
	                            for (int j=threadIdx.x; j<in0_shape1; j+=blockDim.x)
	                                @out(i,j) = @in0(i,j)*@in1(i,j);
	                        }
	                        kernel1<<<32, 32>>>(@ARGS);
	                    """)
	
	            def grad(self, grad):
	                a, b = self.save_vars
	                return jt.code([a.shape, b.shape], [a.dtype, b.dtype], [a, b, grad],
	                    cuda_src="""
	                        __global__ static void kernel2(@ARGS_DEF) {
	                            @PRECALC
	                            for (int i=blockIdx.x; i<in0_shape0; i+=gridDim.x)
	                            for (int j=threadIdx.x; j<in0_shape1; j+=blockDim.x) {
	                                @out0(i,j) = @in2(i,j)*@in1(i,j);
	                                @out1(i,j) = @in2(i,j)*@in0(i,j);
	                            }
	                        }
	                        kernel2<<<32, 32>>>(@ARGS);
	                    """)
	                
	        a = jt.random((100,100))
	        b = jt.random((100,100))
	        func = Func()
	        c = func(a,b)
	        print(c)
	        print(jt.grad(c, [a, b]))'''
	...
@overload
def code(shapes: List[Tuple[int]], dtypes: List[str], inputs: List[Var]={}, cpu_src: str="", cpu_grad_src: List[str]={}, cpu_header: str="", cuda_src: str="", cuda_grad_src: List[str]={}, cuda_header: str="")-> Tuple[Var]:
	'''Document:
	*
	    Code Operator for easily customized op.
	
	    ----------------
	
	    * [in] shape:   the output shape, a integer array
	    
	    * [in] dtype:   the output data type
	    
	    * [in] inputs:  A list of input jittor Vars
	    
	    * [in] cpu_src: cpu source code string, buildin value:
	
	            *   in{x}, in{x}_shape{y}, in{x}_stride{y}, in{x}_type, in{x}_p, @in0(...)
	            *   out{x}, out{x}_shape{y}, out{x}_stride{y}, out{x}_type, out{x}_p, @out0(...)
	            *   out, out_shape{y}, out_stride{y}, out_type, out_p, @out(...)
	    
	    * [in] cpu_header: cpu header code string.
	
	    * [in] cuda_src: cuda source code string.
	
	    * [in] cuda_header: cuda header code string.
	
	    ----------------
	    
	    Example-1::
	
	        from jittor import Function
	        import jittor as jt
	
	        class Func(Function):
	            def execute(self, x):
	                self.save_vars = x
	                return jt.code(x.shape, x.dtype, [x],
	                    cpu_src="""
	                        for (int i=0; i<in0_shape0; i++)
	                            @out(i) = @in0(i)*@in0(i)*2;
	                    """)
	
	            def grad(self, grad_x):
	                x = self.save_vars
	                return jt.code(x.shape, x.dtype, [x, grad_x],
	                    cpu_src="""
	                        for (int i=0; i<in0_shape0; i++)
	                            @out(i) = @in1(i)*@in0(i)*4;
	                    """)
	
	        a = jt.random([10])
	        func = Func()
	        b = func(a)
	        print(b)
	        print(jt.grad(b,a))
	
	    Example-2::
	
	        a = jt.array([3,2,1])
	        b = jt.code(a.shape, a.dtype, [a],
	            cpu_header="""
	                #include <algorithm>
	                @alias(a, in0)
	                @alias(b, out)
	            """,
	            cpu_src="""
	                for (int i=0; i<a_shape0; i++)
	                    @b(i) = @a(i);
	                std::sort(&@b(0), &@b(in0_shape0));
	            """
	        )
	        assert (b.data==[1,2,3]).all()
	
	    Example-3::
	
	        #This example shows how to set multiple outputs in code op.
	        a = jt.array([3,2,1])
	        b,c = jt.code([(1,), (1,)], [a.dtype, a.dtype], [a],
	            cpu_header="""
	                #include <iostream>
	                using namespace std;
	            """,
	            cpu_src="""
	                @alias(a, in0)
	                @alias(b, out0)
	                @alias(c, out1)
	                @b(0) = @c(0) = @a(0);
	                for (int i=0; i<a_shape0; i++) {
	                    @b(0) = std::min(@b(0), @a(i));
	                    @c(0) = std::max(@c(0), @a(i));
	                }
	                cout << "min:" << @b(0) << " max:" << @c(0) << endl;
	            """
	        )
	        assert b.data == 1, b
	        assert c.data == 3, c
	
	    Example-4::
	
	        #This example shows how to use dynamic shape of jittor variables.
	        a = jt.array([5,-4,3,-2,1])
	        
	        # negtive shape for max size of vary dimension
	        b,c = jt.code([(-5,), (-5,)], [a.dtype, a.dtype], [a],
	            cpu_src="""
	                @alias(a, in0)
	                @alias(b, out0)
	                @alias(c, out1)
	                int num_b=0, num_c=0;
	                for (int i=0; i<a_shape0; i++) {
	                    if (@a(i)>0)
	                        @b(num_b++) = @a(i);
	                    else
	                        @c(num_c++) = @a(i);
	                }
	                b->set_shape({num_b});
	                c->set_shape({num_c});
	            """
	        )
	        assert (b.data == [5,3,1]).all()
	        assert (c.data == [-4,-2]).all()
	
	    Example-5::
	
	        # This example shows how to customize code op
	        # compilation flags, such as add include search
	        # path, add definitions, or any command line options
	
	        a = jt.random([10])
	        b = jt.code(a.shape, a.dtype, [a],
	            cpu_src="""
	                @out0(0) = HAHAHA;
	            """)
	        # HAHAHA is defined in flags below
	        # /any/include/path can be change to any path you want to include
	        b.compile_options = {"FLAGS: -DHAHAHA=233 -I/any/include/path ": 1}
	        print(b[0])
	        # will output 233
	
	
	    CUDA Example-1::
	
	        #This example shows how to use CUDA in code op.
	        import jittor as jt
	        from jittor import Function
	        jt.flags.use_cuda = 1
	
	        class Func(Function):
	            def execute(self, a, b):
	                self.save_vars = a, b
	                return jt.code(a.shape, a.dtype, [a,b],
	                    cuda_src="""
	                        __global__ static void kernel1(@ARGS_DEF) {
	                            @PRECALC
	                            int i = threadIdx.x + blockIdx.x * blockDim.x;
	                            int stride = blockDim.x * gridDim.x;
	                            for (; i<in0_shape0; i+=stride)
	                                @out(i) = @in0(i)*@in1(i);
	                        }
	                        kernel1<<<(in0_shape0-1)/1024+1, 1024>>>(@ARGS);
	                    """)
	
	            def grad(self, grad):
	                a, b = self.save_vars
	                return jt.code([a.shape, b.shape], [a.dtype, b.dtype], [a, b, grad],
	                    cuda_src="""
	                        __global__ static void kernel2(@ARGS_DEF) {
	                            @PRECALC
	                            int i = threadIdx.x + blockIdx.x * blockDim.x;
	                            int stride = blockDim.x * gridDim.x;
	                            for (; i<in0_shape0; i+=stride) {
	                                @out0(i) = @in2(i)*@in1(i);
	                                @out1(i) = @in2(i)*@in0(i);
	                            }
	                        }
	                        kernel2<<<(in0_shape0-1)/1024+1, 1024>>>(@ARGS);
	                    """)
	                
	        a = jt.random([100000])
	        b = jt.random([100000])
	        func = Func()
	        c = func(a,b)
	        print(c)
	        print(jt.grad(c, [a, b]))
	
	    CUDA Example-2::
	    
	        #This example shows how to use multi dimension data with CUDA.
	        import jittor as jt
	        from jittor import Function
	        jt.flags.use_cuda = 1
	
	        class Func(Function):
	            def execute(self, a, b):
	                self.save_vars = a, b
	                return jt.code(a.shape, a.dtype, [a,b],
	                    cuda_src="""
	                        __global__ static void kernel1(@ARGS_DEF) {
	                            @PRECALC
	                            for (int i=blockIdx.x; i<in0_shape0; i+=gridDim.x)
	                            for (int j=threadIdx.x; j<in0_shape1; j+=blockDim.x)
	                                @out(i,j) = @in0(i,j)*@in1(i,j);
	                        }
	                        kernel1<<<32, 32>>>(@ARGS);
	                    """)
	
	            def grad(self, grad):
	                a, b = self.save_vars
	                return jt.code([a.shape, b.shape], [a.dtype, b.dtype], [a, b, grad],
	                    cuda_src="""
	                        __global__ static void kernel2(@ARGS_DEF) {
	                            @PRECALC
	                            for (int i=blockIdx.x; i<in0_shape0; i+=gridDim.x)
	                            for (int j=threadIdx.x; j<in0_shape1; j+=blockDim.x) {
	                                @out0(i,j) = @in2(i,j)*@in1(i,j);
	                                @out1(i,j) = @in2(i,j)*@in0(i,j);
	                            }
	                        }
	                        kernel2<<<32, 32>>>(@ARGS);
	                    """)
	                
	        a = jt.random((100,100))
	        b = jt.random((100,100))
	        func = Func()
	        c = func(a,b)
	        print(c)
	        print(jt.grad(c, [a, b]))'''
	...
@overload
def code(inputs: List[Var], outputs: List[Var], cpu_src: str="", cpu_grad_src: List[str]={}, cpu_header: str="", cuda_src: str="", cuda_grad_src: List[str]={}, cuda_header: str="")-> Tuple[Var]:
	'''Document:
	*
	    Code Operator for easily customized op.
	
	    ----------------
	
	    * [in] shape:   the output shape, a integer array
	    
	    * [in] dtype:   the output data type
	    
	    * [in] inputs:  A list of input jittor Vars
	    
	    * [in] cpu_src: cpu source code string, buildin value:
	
	            *   in{x}, in{x}_shape{y}, in{x}_stride{y}, in{x}_type, in{x}_p, @in0(...)
	            *   out{x}, out{x}_shape{y}, out{x}_stride{y}, out{x}_type, out{x}_p, @out0(...)
	            *   out, out_shape{y}, out_stride{y}, out_type, out_p, @out(...)
	    
	    * [in] cpu_header: cpu header code string.
	
	    * [in] cuda_src: cuda source code string.
	
	    * [in] cuda_header: cuda header code string.
	
	    ----------------
	    
	    Example-1::
	
	        from jittor import Function
	        import jittor as jt
	
	        class Func(Function):
	            def execute(self, x):
	                self.save_vars = x
	                return jt.code(x.shape, x.dtype, [x],
	                    cpu_src="""
	                        for (int i=0; i<in0_shape0; i++)
	                            @out(i) = @in0(i)*@in0(i)*2;
	                    """)
	
	            def grad(self, grad_x):
	                x = self.save_vars
	                return jt.code(x.shape, x.dtype, [x, grad_x],
	                    cpu_src="""
	                        for (int i=0; i<in0_shape0; i++)
	                            @out(i) = @in1(i)*@in0(i)*4;
	                    """)
	
	        a = jt.random([10])
	        func = Func()
	        b = func(a)
	        print(b)
	        print(jt.grad(b,a))
	
	    Example-2::
	
	        a = jt.array([3,2,1])
	        b = jt.code(a.shape, a.dtype, [a],
	            cpu_header="""
	                #include <algorithm>
	                @alias(a, in0)
	                @alias(b, out)
	            """,
	            cpu_src="""
	                for (int i=0; i<a_shape0; i++)
	                    @b(i) = @a(i);
	                std::sort(&@b(0), &@b(in0_shape0));
	            """
	        )
	        assert (b.data==[1,2,3]).all()
	
	    Example-3::
	
	        #This example shows how to set multiple outputs in code op.
	        a = jt.array([3,2,1])
	        b,c = jt.code([(1,), (1,)], [a.dtype, a.dtype], [a],
	            cpu_header="""
	                #include <iostream>
	                using namespace std;
	            """,
	            cpu_src="""
	                @alias(a, in0)
	                @alias(b, out0)
	                @alias(c, out1)
	                @b(0) = @c(0) = @a(0);
	                for (int i=0; i<a_shape0; i++) {
	                    @b(0) = std::min(@b(0), @a(i));
	                    @c(0) = std::max(@c(0), @a(i));
	                }
	                cout << "min:" << @b(0) << " max:" << @c(0) << endl;
	            """
	        )
	        assert b.data == 1, b
	        assert c.data == 3, c
	
	    Example-4::
	
	        #This example shows how to use dynamic shape of jittor variables.
	        a = jt.array([5,-4,3,-2,1])
	        
	        # negtive shape for max size of vary dimension
	        b,c = jt.code([(-5,), (-5,)], [a.dtype, a.dtype], [a],
	            cpu_src="""
	                @alias(a, in0)
	                @alias(b, out0)
	                @alias(c, out1)
	                int num_b=0, num_c=0;
	                for (int i=0; i<a_shape0; i++) {
	                    if (@a(i)>0)
	                        @b(num_b++) = @a(i);
	                    else
	                        @c(num_c++) = @a(i);
	                }
	                b->set_shape({num_b});
	                c->set_shape({num_c});
	            """
	        )
	        assert (b.data == [5,3,1]).all()
	        assert (c.data == [-4,-2]).all()
	
	    Example-5::
	
	        # This example shows how to customize code op
	        # compilation flags, such as add include search
	        # path, add definitions, or any command line options
	
	        a = jt.random([10])
	        b = jt.code(a.shape, a.dtype, [a],
	            cpu_src="""
	                @out0(0) = HAHAHA;
	            """)
	        # HAHAHA is defined in flags below
	        # /any/include/path can be change to any path you want to include
	        b.compile_options = {"FLAGS: -DHAHAHA=233 -I/any/include/path ": 1}
	        print(b[0])
	        # will output 233
	
	
	    CUDA Example-1::
	
	        #This example shows how to use CUDA in code op.
	        import jittor as jt
	        from jittor import Function
	        jt.flags.use_cuda = 1
	
	        class Func(Function):
	            def execute(self, a, b):
	                self.save_vars = a, b
	                return jt.code(a.shape, a.dtype, [a,b],
	                    cuda_src="""
	                        __global__ static void kernel1(@ARGS_DEF) {
	                            @PRECALC
	                            int i = threadIdx.x + blockIdx.x * blockDim.x;
	                            int stride = blockDim.x * gridDim.x;
	                            for (; i<in0_shape0; i+=stride)
	                                @out(i) = @in0(i)*@in1(i);
	                        }
	                        kernel1<<<(in0_shape0-1)/1024+1, 1024>>>(@ARGS);
	                    """)
	
	            def grad(self, grad):
	                a, b = self.save_vars
	                return jt.code([a.shape, b.shape], [a.dtype, b.dtype], [a, b, grad],
	                    cuda_src="""
	                        __global__ static void kernel2(@ARGS_DEF) {
	                            @PRECALC
	                            int i = threadIdx.x + blockIdx.x * blockDim.x;
	                            int stride = blockDim.x * gridDim.x;
	                            for (; i<in0_shape0; i+=stride) {
	                                @out0(i) = @in2(i)*@in1(i);
	                                @out1(i) = @in2(i)*@in0(i);
	                            }
	                        }
	                        kernel2<<<(in0_shape0-1)/1024+1, 1024>>>(@ARGS);
	                    """)
	                
	        a = jt.random([100000])
	        b = jt.random([100000])
	        func = Func()
	        c = func(a,b)
	        print(c)
	        print(jt.grad(c, [a, b]))
	
	    CUDA Example-2::
	    
	        #This example shows how to use multi dimension data with CUDA.
	        import jittor as jt
	        from jittor import Function
	        jt.flags.use_cuda = 1
	
	        class Func(Function):
	            def execute(self, a, b):
	                self.save_vars = a, b
	                return jt.code(a.shape, a.dtype, [a,b],
	                    cuda_src="""
	                        __global__ static void kernel1(@ARGS_DEF) {
	                            @PRECALC
	                            for (int i=blockIdx.x; i<in0_shape0; i+=gridDim.x)
	                            for (int j=threadIdx.x; j<in0_shape1; j+=blockDim.x)
	                                @out(i,j) = @in0(i,j)*@in1(i,j);
	                        }
	                        kernel1<<<32, 32>>>(@ARGS);
	                    """)
	
	            def grad(self, grad):
	                a, b = self.save_vars
	                return jt.code([a.shape, b.shape], [a.dtype, b.dtype], [a, b, grad],
	                    cuda_src="""
	                        __global__ static void kernel2(@ARGS_DEF) {
	                            @PRECALC
	                            for (int i=blockIdx.x; i<in0_shape0; i+=gridDim.x)
	                            for (int j=threadIdx.x; j<in0_shape1; j+=blockDim.x) {
	                                @out0(i,j) = @in2(i,j)*@in1(i,j);
	                                @out1(i,j) = @in2(i,j)*@in0(i,j);
	                            }
	                        }
	                        kernel2<<<32, 32>>>(@ARGS);
	                    """)
	                
	        a = jt.random((100,100))
	        b = jt.random((100,100))
	        func = Func()
	        c = func(a,b)
	        print(c)
	        print(jt.grad(c, [a, b]))'''
	...
def copy(x: Var)-> Var:
 ...
def setitem(x: Var, slices: slice, y: Var, op: str="void")-> Var:
 ...
@overload
def broadcast(x: Var, shape: Tuple[int], dims: Tuple[int]=())-> Var:
	'''Document:
	*
	    Broadcast ``x`` to a given shape.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] shape:   the output shape.
	
	    * [in] dims:    specifies the new dimension in the output shape, an integer array.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randint(0, 10, shape=(2, 2))
	        >>> x
	        jt.Var([[8 1]
	         [7 6]], dtype=int32)
	        >>> jt.broadcast(x, shape=(2, 3, 2), dims=[1])
	        jt.Var([[[8 1]
	          [8 1]
	          [8 1]],
	         [[7 6]
	          [7 6]
	          [7 6]]], dtype=int32)'''
	...
@overload
def broadcast(x: Var, y: Var, dims: Tuple[int]=())-> Var:
	'''Document:
	*
	    Broadcast ``x`` to the same shape as ``y``.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] y:       the reference jt.Var.
	
	    * [in] dims:    specifies the new dimension in the output shape, an integer array.
	
	    ----------------
	
	    .. note::
	      jt.broadcast_var(x, y, dims) is an alias of jt.broadcast(x, y, dims)
	
	    Example-1::
	        >>> x = jt.randint(0, 10, shape=(2, 2))
	        >>> x
	        jt.Var([[8 1]
	         [7 6]], dtype=int32)
	        >>> y = jt.randint(0, 10, shape=(2, 3, 2))
	        >>> jt.broadcast(x, y, dims=[1])
	        jt.Var([[[8 1]
	          [8 1]
	          [8 1]],
	         [[7 6]
	          [7 6]
	          [7 6]]], dtype=int32)
	        >>> jt.broadcast_var(x, y, dims=[1])
	        jt.Var([[[8 1]
	          [8 1]
	          [8 1]],
	         [[7 6]
	          [7 6]
	          [7 6]]], dtype=int32)'''
	...
def broadcast_var(x: Var, y: Var, dims: Tuple[int]=())-> Var:
	'''Document:
	*
	    Broadcast ``x`` to the same shape as ``y``.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] y:       the reference jt.Var.
	
	    * [in] dims:    specifies the new dimension in the output shape, an integer array.
	
	    ----------------
	
	    .. note::
	      jt.broadcast_var(x, y, dims) is an alias of jt.broadcast(x, y, dims)
	
	    Example-1::
	        >>> x = jt.randint(0, 10, shape=(2, 2))
	        >>> x
	        jt.Var([[8 1]
	         [7 6]], dtype=int32)
	        >>> y = jt.randint(0, 10, shape=(2, 3, 2))
	        >>> jt.broadcast(x, y, dims=[1])
	        jt.Var([[[8 1]
	          [8 1]
	          [8 1]],
	         [[7 6]
	          [7 6]
	          [7 6]]], dtype=int32)
	        >>> jt.broadcast_var(x, y, dims=[1])
	        jt.Var([[[8 1]
	          [8 1]
	          [8 1]],
	         [[7 6]
	          [7 6]
	          [7 6]]], dtype=int32)'''
	...
def reshape(x: Var, shape: Tuple[int])-> Var:
	'''Document:
	*
	    Returns a tensor with the same data and number of elements as input, but with the specified shape. 
	
	    A single dimension may be -1, in which case it's inferred from the remaining dimensions and the number of elements in input.
	
	    ----------------
	
	    * [in] x:       the input jt.Var
	
	    * [in] shape:   the output shape, an integer array
	
	    ----------------
	
	    Example-1::
	        >>> a = jt.randint(0, 10, shape=(12,))
	        >>> a
	        jt.Var([4 0 8 4 6 3 1 8 1 1 2 2], dtype=int32)
	        >>> jt.reshape(a, (3, 4))
	        jt.Var([[4 0 8 4]
	         [6 3 1 8]
	         [1 1 2 2]], dtype=int32)
	        >>> jt.reshape(a, (-1, 6))
	        jt.Var([[4 0 8 4 6 3]
	         [1 8 1 1 2 2]], dtype=int32)'''
	...
def empty(shape: Tuple[int], dtype: str="float32")-> Var:
 ...
def reindex_reduce(y: Var, op: str, shape: Tuple[int], indexes: List[str], overflow_conditions: List[str]={}, extras: List[Var]={})-> Var:
	'''Document:
	*
	    Reindex Reduce Operator is a many-to-one map operator.
	    It performs equivalent Python-pseudo implementation below::
	
	        # input is y, output is x
	        n = len(y.shape)-1
	        m = len(shape)-1
	        k = len(overflow_conditions)-1
	        x = np.zeros(shape, y.dtype)
	        x[:] = initial_value(op)
	        for i0 in range(y.shape[0]): # 1-st loop
	            for i1 in range(y.shape[1]): # 2-nd loop
	                ...... # many loops
	                for in in range(y.shape[n]) # n+1 -th loop
	                    # indexes[i] is a c++ style integer expression consisting of i0,i1,...,in
	                    xi0,xi1,...,xim = indexes[0],indexes[1],...,indexes[m]
	                    if not is_overflow(xi0,xi1,...,xim):
	                        x[xi0,xi1,...,xim] = op(x[xi0,xi1,...,xim], y[i0,i1,...,in])
	
	        # is_overflow is defined as following
	        def is_overflow(xi0,xi1,...,xim):
	            return (
	                xi0 < 0 || xi0 >= shape[0] ||
	                xi1 < 0 || xi1 >= shape[1] ||
	                ......
	                xim < 0 || xim >= shape[m] ||
	
	                # overflow_conditions[i] is a c++ style boolean expression consisting of i0,i1,...,in
	                overflow_conditions[0] ||
	                overflow_conditions[1] ||
	                ......
	                overflow_conditions[k]
	            )
	
	    * [in] y:   A input jittor Var
	    
	    * [in] op:  a string represent the reduce operation type
	    
	    * [in] shape:   the output shape, a integer array
	    
	    * [in] indexes: array of c++ style integer expression, its length should be the same with length of output shape, some buildin variables it can use are::
	    
	             XDIM, xshape0, ..., xshapem, xstride0, ..., xstridem
	             YDIM, yshape0, ..., yshapen, ystride0, ..., ystriden
	             i0, i1, ..., in
	             @e0(...), @e1(...) for extras input index
	             e0p, e1p , ... for extras input pointer
	    
	    * [in] overflow_conditions: array of c++ style boolean expression, it length can be vary. the buildin variables it can use are the same with indexes.
	    
	    * [in] extras:  extra var used for index
	    
	    Example 
	
	    Pooling implemented by reindex operation::
	
	        def pool(x, size, op):
	            N,H,W,C = x.shape
	            h = (H+size-1)//size
	            w = (W+size-1)//size
	            return x.reindex_reduce(op, [N,h,w,C], [
	                "i0", # Nid
	                f"i1/{size}", # Hid
	                f"i2/{size}", # Wid
	                "i3", # Cid
	            ])'''
	...
class Var:
	'''Variable that stores multi-dimensional data.'''
	def ternary(self, x: Var, y: Var)-> Var: ...
	@overload
	def reindex(self, shape: Tuple[int], indexes: List[str], overflow_value: float=0, overflow_conditions: List[str]={}, extras: List[Var]={})-> Var:		
		'''Document:
		* 
		    Reindex Operator is a one-to-many map operator.
		    It performs equivalent Python-pseudo implementation below::
		
		        # input is x, output is y
		        n = len(shape)-1
		        m = len(x.shape)-1
		        k = len(overflow_conditions)-1
		        y = np.zeros(shape, x.dtype)
		        for i0 in range(shape[0]): # 1-st loop
		            for i1 in range(shape[1]): # 2-nd loop
		                ...... # many loops
		                for in in range(shape[n]) # n+1 -th loop
		                    if is_overflow(i0,i1,...,in):
		                        y[i0,i1,...,in] = overflow_value
		                    else:
		                        # indexes[i] is a c++ style integer expression consisting of i0,i1,...,in
		                        y[i0,i1,...,in] = x[indexes[0],indexes[1],...,indexes[m]]
		
		        # is_overflow is defined as following
		        def is_overflow(i0,i1,...,in):
		            return (
		                indexes[0] < 0 || indexes[0] >= x.shape[0] ||
		                indexes[1] < 0 || indexes[1] >= x.shape[1] ||
		                ......
		                indexes[m] < 0 || indexes[m] >= x.shape[m] ||
		
		                # overflow_conditions[i] is a c++ style boolean expression consisting of i0,i1,...,in
		                overflow_conditions[0] ||
		                overflow_conditions[1] ||
		                ......
		                overflow_conditions[k]
		            )
		    ----------------
		    * [in] x:	A input jittor Var
			
		    * [in] shape:	the output shape, a integer array
			
		    * [in] indexes:	array of c++ style integer expression, its length should be the same with the number of dimension of x, some buildin variables it can use are::
		        
		             XDIM, xshape0, ..., xshapen, xstride0, ..., xstriden
		             YDIM, yshape0, ..., yshapem, ystride0, ..., ystridem
		             i0, i1, ..., in
		             @e0(...), @e1(...) for extras input index
		             e0p, e1p , ... for extras input pointer
					 
		    * [in] overflow_value:	overflow value
			
		    * [in] overflow_conditions:	array of c++ style boolean expression, it length can be vary. the buildin variables it can use are the same with indexes
				
		    * [in] extras: extra var used for index
			
		    ----------------
		    Example
		    Convolution implemented by reindex operation::
		
		        def conv(x, w):
		            N,H,W,C = x.shape
		            Kh, Kw, _C, Kc = w.shape
		            assert C==_C
		            xx = x.reindex([N,H-Kh+1,W-Kw+1,Kh,Kw,C,Kc], [
		                'i0', # Nid
		                'i1+i3', # Hid+Khid
		                'i2+i4', # Wid+KWid
		                'i5', # Cid
		            ])
		            ww = w.broadcast_var(xx)
		            yy = xx*ww
		            y = yy.sum([3,4,5]) # Kh, Kw, C
		            return y, yy'''
		...
	@overload
	def reindex(self, indexes: List[Var], overflow_value: float=0, overflow_conditions: List[str]={})-> Var:		
		'''Document:
		* Alias x.reindex([i,j,k]) -> 
		        x.reindex(i.shape, ['@e0(...)','@e1(...)','@e2(...)',], extras=[i,j,k])'''
		...
	def reindex_var(self, indexes: List[Var], overflow_value: float=0, overflow_conditions: List[str]={})-> Var:		
		'''Document:
		* Alias x.reindex([i,j,k]) -> 
		        x.reindex(i.shape, ['@e0(...)','@e1(...)','@e2(...)',], extras=[i,j,k])'''
		...
	@overload
	def index(self, dim: int, dtype: str="int32")-> Var:		
		'''Document:
		* shape dependency version of index op
		        jt.index_var(a, 1) similar with jt.index(a.shape, 1)'''
		...
	@overload
	def index(self, dtype: str="int32")-> Tuple[Var]:		
		'''Document:
		* shape dependency version of index op
		        jt.index_var(a) similar with jt.index(a.shape)'''
		...
	@overload
	def index_var(self, dim: int, dtype: str="int32")-> Var:		
		'''Document:
		* shape dependency version of index op
		        jt.index_var(a, 1) similar with jt.index(a.shape, 1)'''
		...
	@overload
	def index_var(self, dtype: str="int32")-> Tuple[Var]:		
		'''Document:
		* shape dependency version of index op
		        jt.index_var(a) similar with jt.index(a.shape)'''
		...
	def binary(self, y: Var, p: str)-> Var: ...
	def pow(self, y: Var)-> Var:		
		'''Document:
		*
		    Computes ``x^y``, element-wise. 
		
		    ----------------
		
		    * [in] x: the first input,  a python number or jt.Var.
		
		    * [in] y: the second input, a python number or jt.Var.'''
		...
	def maximum(self, y: Var)-> Var:		
		'''Document:
		*
		    Returns the element-wise maximum of ``x`` and ``y``. 
		
		    ----------------
		
		    * [in] x: the first input,  a python number or jt.Var.
		
		    * [in] y: the second input, a python number or jt.Var.'''
		...
	def minimum(self, y: Var)-> Var:		
		'''Document:
		*
		    Returns the element-wise minimum of ``x`` and ``y``. 
		
		    ----------------
		
		    * [in] x: the first input,  a python number or jt.Var.
		
		    * [in] y: the second input, a python number or jt.Var.'''
		...
	def add(self, y: Var)-> Var:		
		'''Document:
		*
		    Element-wise adds ``x`` and ``y`` and returns a new Var. 
		    
		    This operation is equivalent to ``x + y``.
		
		    ----------------
		
		    * [in] x: the first input,  a python number or jt.Var.
		
		    * [in] y: the second input, a python number or jt.Var.'''
		...
	def subtract(self, y: Var)-> Var:		
		'''Document:
		*
		    Element-wise subtract ``y`` from ``x`` and returns a new Var.
		
		    This operation is equivalent to ``x - y``.
		
		    ----------------
		
		    * [in] x: the first input,  a python number or jt.Var.
		
		    * [in] y: the second input, a python number or jt.Var.'''
		...
	def multiply(self, y: Var)-> Var:		
		'''Document:
		*
		    Element-wise muliplies ``x`` with ``y`` and returns a new Var.
		
		    This operation is equivalent to ``x * y``.
		
		    ----------------
		
		    * [in] x: the first input,  a python number or jt.Var.
		
		    * [in] y: the second input, a python number or jt.Var.'''
		...
	def divide(self, y: Var)-> Var:		
		'''Document:
		*
		    Element-wise divide ``x`` by ``y`` and returns a new Var.
		
		    This operation is equivalent to ``x / y``.
		
		    ----------------
		
		    * [in] x: the first input,  a python number or jt.Var.
		
		    * [in] y: the second input, a python number or jt.Var.
		
		    ----------------
		
		    Example-1::
		        >>> a = jt.empty((3,), dtype=jt.int32)
		        >>> a
		        jt.Var([707406378 707406378 707406378], dtype=int32)
		        >>> b = jt.empty((3,), dtype=jt.int32)
		        >>> b
		        jt.Var([674510453 171649398 538976288], dtype=int32)
		        >>> jt.divide(a, b)
		        jt.Var([1.0487701 4.1212287 1.3125001], dtype=float32)
		        >>> a / b
		        jt.Var([1.0487701 4.1212287 1.3125001], dtype=float32)
		
		    .. note ::
		    returns float value even if the dtype of input Vars are both integers.
		    @see jt.ops.floor_divide() for floor division.'''
		...
	def floor_divide(self, y: Var)-> Var:		
		'''Document:
		*
		    Element-wise divide ``x`` by ``y`` and returns the floor of the result.
		
		    This operation is equivalent to ``x // y``.
		
		    ----------------
		
		    * [in] x: the first input,  a python number or jt.Var.
		
		    * [in] y: the second input, a python number or jt.Var.
		
		    ----------------
		
		    Example-1::
		        >>> a = jt.randint(1, 10, (3,), dtype=jt.int32)
		        >>> a
		        jt.Var([9 2 7], dtype=int32)
		        >>> b = jt.randint(1, 10, (3,), dtype=jt.int32)
		        >>> b
		        jt.Var([6 4 6], dtype=int32)
		        >>> jt.floor_divide(a, b)
		        jt.Var([1 0 1], dtype=int32)
		        >>> a // b
		        jt.Var([1 0 1], dtype=int32)'''
		...
	def mod(self, y: Var)-> Var:		
		'''Document:
		*
		    Returns the element-wise remainder of division.
		
		    This operation is equivalent to ``x % y``.
		
		    ----------------
		
		    * [in] x: the first input,  a python number or jt.Var.
		
		    * [in] y: the second input, a python number or jt.Var.
		
		    ----------------
		
		    Example-1::
		        >>> a = jt.rand(3)
		        >>> a
		        jt.Var([0.3989529  0.20159635 0.22973768], dtype=float32)
		        >>> b = jt.rand(3)
		        >>> b
		        jt.Var([0.20121202 0.7704864  0.5654395 ], dtype=float32)
		        >>> jt.mod(a, b)
		        jt.Var([0.19774088 0.20159635 0.22973768], dtype=float32)
		        >>> a % b
		        jt.Var([0.19774088 0.20159635 0.22973768], dtype=float32)'''
		...
	def less(self, y: Var)-> Var:		
		'''Document:
		*
		    Returns ``x < y`` element-wise.
		
		    This operation is equivalent to ``x < y``.
		
		    ----------------
		
		    * [in] x: the first input,  a python number or jt.Var.
		
		    * [in] y: the second input, a python number or jt.Var.'''
		...
	def less_equal(self, y: Var)-> Var:		
		'''Document:
		*
		    Returns ``x <= y`` element-wise.
		
		    This operation is equivalent to ``x <= y``.
		
		    ----------------
		
		    * [in] x: the first input,  a python number or jt.Var.
		
		    * [in] y: the second input, a python number or jt.Var.'''
		...
	def greater(self, y: Var)-> Var:		
		'''Document:
		*
		    Returns ``x > y`` element-wise.
		
		    This operation is equivalent to ``x > y``.
		
		    ----------------
		
		    * [in] x: the first input,  a python number or jt.Var.
		
		    * [in] y: the second input, a python number or jt.Var.'''
		...
	def greater_equal(self, y: Var)-> Var:		
		'''Document:
		*
		    Returns ``x >= y`` element-wise.
		    
		    This operation is equivalent to ``x >= y``.
		
		    ----------------
		
		    * [in] x: the first input,  a python number or jt.Var.
		
		    * [in] y: the second input, a python number or jt.Var.'''
		...
	def equal(self, y: Var)-> Var:		
		'''Document:
		*
		    Returns ``x == y`` element-wise.
		
		    This operation is equivalent to ``x == y``.
		
		    ----------------
		
		    * [in] x: the first input,  a python number or jt.Var.
		
		    * [in] y: the second input, a python number or jt.Var.'''
		...
	def not_equal(self, y: Var)-> Var:		
		'''Document:
		*
		    Returns ``x != y`` element-wise.
		
		    This operation is equivalent to ``x != y``.
		
		    ----------------
		
		    * [in] x: the first input,  a python number or jt.Var.
		
		    * [in] y: the second input, a python number or jt.Var.'''
		...
	def left_shift(self, y: Var)-> Var:		
		'''Document:
		*
		    Shifts the bits of ``x`` to the left by ``y``. 
		
		    Bits are shifted to the left by appending ``y`` 0s at the right of ``x``.
		    This operation is equivalent to ``x << y``.
		
		    ----------------
		
		    * [in] x: the first input,  a python number or jt.Var (int32 or int64).
		
		    * [in] y: the second input, a python number or jt.Var (int32 or int64).
		
		    ----------------
		
		    Example-1::
		        >>> a = jt.randint(0, 10, shape=(3,))
		        >>> a
		        jt.Var([7 6 7], dtype=int32)
		        >>> b = jt.randint(0, 10, shape=(3,))
		        >>> b
		        jt.Var([3 9 8], dtype=int32)
		        >>> jt.left_shift(a, b)
		        jt.Var([  56 3072 1792], dtype=int32)
		        >>> a << b
		        jt.Var([  56 3072 1792], dtype=int32)'''
		...
	def right_shift(self, y: Var)-> Var:		
		'''Document:
		*
		    Shifts the bits of ``x`` to the right by ``y``. 
		
		    This operation is equivalent to ``x >> y``.
		
		    ----------------
		
		    * [in] x: the first input,  a python number or jt.Var (int32 or int64).
		
		    * [in] y: the second input, a python number or jt.Var (int32 or int64).
		
		    ----------------
		
		    Example-1::
		        >>> a = jt.randint(0, 1024, shape=(3,))
		        >>> a
		        jt.Var([439 113  92], dtype=int32)
		        >>> b = jt.randint(0, 10, shape=(3,))
		        >>> b
		        jt.Var([6 8 4], dtype=int32)
		        >>> jt.right_shift(a, b)
		        jt.Var([6 0 5], dtype=int32)'''
		...
	def logical_and(self, y: Var)-> Var:		
		'''Document:
		*
		    Returns the element-wise logical AND of the inputs. 
		
		    ----------------
		
		    * [in] x: the first input, jt.Var.
		
		    * [in] y: the second input, jt.Var.'''
		...
	def logical_or(self, y: Var)-> Var:		
		'''Document:
		*
		    Returns the element-wise logical OR of the inputs. 
		
		    ----------------
		
		    * [in] x: the first input, jt.Var.
		
		    * [in] y: the second input, jt.Var.'''
		...
	def logical_xor(self, y: Var)-> Var:		
		'''Document:
		*
		    Returns the element-wise logical XOR of the inputs. 
		
		    ----------------
		
		    * [in] x: the first input, jt.Var.
		
		    * [in] y: the second input, jt.Var.'''
		...
	def bitwise_and(self, y: Var)-> Var:		
		'''Document:
		*
		    Computes the bitwise AND of x and y.
		
		    ----------------
		
		    * [in] x: the first input, jt.Var (integal or boolean).
		
		    * [in] y: the second input, jt.Var (integal or boolean).'''
		...
	def bitwise_or(self, y: Var)-> Var:		
		'''Document:
		*
		    Computes the bitwise OR of x and y.
		
		    ----------------
		
		    * [in] x: the first input, jt.Var (integal or boolean).
		
		    * [in] y: the second input, jt.Var (integal or boolean).'''
		...
	def bitwise_xor(self, y: Var)-> Var:		
		'''Document:
		*
		    Computes the bitwise XOR of x and y.
		
		    ----------------
		
		    * [in] x: the first input, jt.Var (integal or boolean).
		
		    * [in] y: the second input, jt.Var (integal or boolean).'''
		...
	def tape(self)-> Var: ...
	@overload
	def where(self, dtype: str="int32")-> Tuple[Var]:		
		'''Document:
		*
		    Where Operator generate index of true condition.
		
		    * [in] cond:    condition for index generation
		
		    * [in] dtype:   type of return indexes
		    
		    * [out] out:  return an array of indexes, same length with number of dims of cond 
		    
		    Example::
		
		        jt.where([[0,0,1],[1,0,0]])
		        # return [jt.Var([0 1], dtype=int32), jt.Var([2 0], dtype=int32)]'''
		...
	@overload
	def where(self, x: Var, y: Var)-> Var:		
		'''Document:
		*
		     * Condition operator, perform cond ? x : y
		     *'''
		...
	def argsort(self, dim: int=-1, descending: bool=False, dtype: str="int32")-> Tuple[Var]:		
		'''Document:
		* 
		    Argsort Operator Perform an indirect sort by given key or compare function.
		
		    x is input, y is output index, satisfy:
		
		        x[y[0]] <= x[y[1]] <= x[y[2]] <= ... <= x[y[n]]
		
		    or
		
		        key(y[0]) <= key(y[1]) <= key(y[2]) <= ... <= key(y[n])
		
		    or
		
		        compare(y[0], y[1]) && compare(y[1], y[2]) && ...
		
		    * [in] x: input var for sort
		
		    * [in] dim: sort alone which dim
		
		    * [in] descending:  the elements are sorted in descending order or not(default False).
		
		    * [in] dtype: type of return indexes
		
		    * [out] index: index have the same size with sorted dim
		
		    * [out] value: sorted value
		
		    
		    Example::
		
		            index, value = jt.argsort([11,13,12])
		            # return [0 2 1], [11 12 13]
		            index, value = jt.argsort([11,13,12], descending=True)
		            # return [1 2 0], [13 12 11]
		            index, value = jt.argsort([[11,13,12], [12,11,13]])
		            # return [[0 2 1],[1 0 2]],  [[11 12 13],[11 12 13]]
		            index, value = jt.argsort([[11,13,12], [12,11,13]], dim=0)
		            # return [[0 1 0],[1 0 1]],  [[11 11 12],[12 13 13]]'''
		...
	def fetch(self, func: Callable)-> Var: ...
	def arg_reduce(self, op: str, dim: int, keepdims: bool)-> Tuple[Var]:		
		'''Document:
		*
		    Returns the indices of the maximum / minimum of the input across a dimension.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] op:      "max" or "min". 
		
		    * [in] dim:     int. Specifies which dimension to be reduced.
		
		    * [in] keepdims: bool. Whether the output has ``dim`` retained or not.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randint(0, 10, shape=(2, 3))
		        >>> x
		        jt.Var([[4 2 5]
		         [6 7 1]], dtype=int32)
		        >>> jt.arg_reduce(x, 'max', dim=1, keepdims=False)
		        [jt.Var([2 1], dtype=int32), jt.Var([5 7], dtype=int32)]
		        >>> jt.arg_reduce(x, 'min', dim=1, keepdims=False)
		        [jt.Var([1 2], dtype=int32), jt.Var([2 1], dtype=int32)]'''
		...
	@overload
	def reduce(self, op: str, dim: int, keepdims: bool=False)-> Var: ...
	@overload
	def reduce(self, op: str, dims: Tuple[int]=(), keepdims: bool=False)-> Var: ...
	@overload
	def max(self, dim: int, keepdims: bool=False)-> Var:		
		'''Document:
		*
		    Returns the maximum elements in the input.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
		
		    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randint(10, shape=(2, 3))
		        >>> x
		        jt.Var([[4 1 2]
		         [0 2 4]], dtype=int32)
		        >>> jt.max(x)
		        jt.Var([4], dtype=int32)
		        >>> x.max()
		        jt.Var([4], dtype=int32)
		        >>> x.max(dim=1)
		        jt.Var([4 4], dtype=int32)
		        >>> x.max(dim=1, keepdims=True)
		        jt.Var([[4]
		         [4]], dtype=int32)'''
		...
	@overload
	def max(self, dims: Tuple[int]=(), keepdims: bool=False)-> Var:		
		'''Document:
		*
		    Returns the maximum elements in the input.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
		
		    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randint(10, shape=(2, 3))
		        >>> x
		        jt.Var([[4 1 2]
		         [0 2 4]], dtype=int32)
		        >>> jt.max(x)
		        jt.Var([4], dtype=int32)
		        >>> x.max()
		        jt.Var([4], dtype=int32)
		        >>> x.max(dim=1)
		        jt.Var([4 4], dtype=int32)
		        >>> x.max(dim=1, keepdims=True)
		        jt.Var([[4]
		         [4]], dtype=int32)'''
		...
	@overload
	def max(self, dims_mask: int, keepdims_mask: int)-> Var:		
		'''Document:
		*
		    Returns the maximum elements in the input.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
		
		    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randint(10, shape=(2, 3))
		        >>> x
		        jt.Var([[4 1 2]
		         [0 2 4]], dtype=int32)
		        >>> jt.max(x)
		        jt.Var([4], dtype=int32)
		        >>> x.max()
		        jt.Var([4], dtype=int32)
		        >>> x.max(dim=1)
		        jt.Var([4 4], dtype=int32)
		        >>> x.max(dim=1, keepdims=True)
		        jt.Var([[4]
		         [4]], dtype=int32)'''
		...
	@overload
	def reduce_maximum(self, dim: int, keepdims: bool=False)-> Var:		
		'''Document:
		*
		    Returns the maximum elements in the input.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
		
		    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randint(10, shape=(2, 3))
		        >>> x
		        jt.Var([[4 1 2]
		         [0 2 4]], dtype=int32)
		        >>> jt.max(x)
		        jt.Var([4], dtype=int32)
		        >>> x.max()
		        jt.Var([4], dtype=int32)
		        >>> x.max(dim=1)
		        jt.Var([4 4], dtype=int32)
		        >>> x.max(dim=1, keepdims=True)
		        jt.Var([[4]
		         [4]], dtype=int32)'''
		...
	@overload
	def reduce_maximum(self, dims: Tuple[int]=(), keepdims: bool=False)-> Var:		
		'''Document:
		*
		    Returns the maximum elements in the input.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
		
		    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randint(10, shape=(2, 3))
		        >>> x
		        jt.Var([[4 1 2]
		         [0 2 4]], dtype=int32)
		        >>> jt.max(x)
		        jt.Var([4], dtype=int32)
		        >>> x.max()
		        jt.Var([4], dtype=int32)
		        >>> x.max(dim=1)
		        jt.Var([4 4], dtype=int32)
		        >>> x.max(dim=1, keepdims=True)
		        jt.Var([[4]
		         [4]], dtype=int32)'''
		...
	@overload
	def reduce_maximum(self, dims_mask: int, keepdims_mask: int)-> Var:		
		'''Document:
		*
		    Returns the maximum elements in the input.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
		
		    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randint(10, shape=(2, 3))
		        >>> x
		        jt.Var([[4 1 2]
		         [0 2 4]], dtype=int32)
		        >>> jt.max(x)
		        jt.Var([4], dtype=int32)
		        >>> x.max()
		        jt.Var([4], dtype=int32)
		        >>> x.max(dim=1)
		        jt.Var([4 4], dtype=int32)
		        >>> x.max(dim=1, keepdims=True)
		        jt.Var([[4]
		         [4]], dtype=int32)'''
		...
	@overload
	def min(self, dim: int, keepdims: bool=False)-> Var:		
		'''Document:
		*
		    Returns the minimum elements in the input.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
		
		    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randint(10, shape=(2, 3))
		        >>> x
		        jt.Var([[4 1 2]
		         [0 2 4]], dtype=int32)
		        >>> jt.min(x)
		        jt.Var([0], dtype=int32)
		        >>> x.min()
		        jt.Var([0], dtype=int32)
		        >>> x.min(dim=1)
		        jt.Var([1 0], dtype=int32)
		        >>> x.min(dim=1, keepdims=True)
		        jt.Var([[1]
		         [0]], dtype=int32)'''
		...
	@overload
	def min(self, dims: Tuple[int]=(), keepdims: bool=False)-> Var:		
		'''Document:
		*
		    Returns the minimum elements in the input.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
		
		    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randint(10, shape=(2, 3))
		        >>> x
		        jt.Var([[4 1 2]
		         [0 2 4]], dtype=int32)
		        >>> jt.min(x)
		        jt.Var([0], dtype=int32)
		        >>> x.min()
		        jt.Var([0], dtype=int32)
		        >>> x.min(dim=1)
		        jt.Var([1 0], dtype=int32)
		        >>> x.min(dim=1, keepdims=True)
		        jt.Var([[1]
		         [0]], dtype=int32)'''
		...
	@overload
	def min(self, dims_mask: int, keepdims_mask: int)-> Var:		
		'''Document:
		*
		    Returns the minimum elements in the input.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
		
		    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randint(10, shape=(2, 3))
		        >>> x
		        jt.Var([[4 1 2]
		         [0 2 4]], dtype=int32)
		        >>> jt.min(x)
		        jt.Var([0], dtype=int32)
		        >>> x.min()
		        jt.Var([0], dtype=int32)
		        >>> x.min(dim=1)
		        jt.Var([1 0], dtype=int32)
		        >>> x.min(dim=1, keepdims=True)
		        jt.Var([[1]
		         [0]], dtype=int32)'''
		...
	@overload
	def reduce_minimum(self, dim: int, keepdims: bool=False)-> Var:		
		'''Document:
		*
		    Returns the minimum elements in the input.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
		
		    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randint(10, shape=(2, 3))
		        >>> x
		        jt.Var([[4 1 2]
		         [0 2 4]], dtype=int32)
		        >>> jt.min(x)
		        jt.Var([0], dtype=int32)
		        >>> x.min()
		        jt.Var([0], dtype=int32)
		        >>> x.min(dim=1)
		        jt.Var([1 0], dtype=int32)
		        >>> x.min(dim=1, keepdims=True)
		        jt.Var([[1]
		         [0]], dtype=int32)'''
		...
	@overload
	def reduce_minimum(self, dims: Tuple[int]=(), keepdims: bool=False)-> Var:		
		'''Document:
		*
		    Returns the minimum elements in the input.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
		
		    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randint(10, shape=(2, 3))
		        >>> x
		        jt.Var([[4 1 2]
		         [0 2 4]], dtype=int32)
		        >>> jt.min(x)
		        jt.Var([0], dtype=int32)
		        >>> x.min()
		        jt.Var([0], dtype=int32)
		        >>> x.min(dim=1)
		        jt.Var([1 0], dtype=int32)
		        >>> x.min(dim=1, keepdims=True)
		        jt.Var([[1]
		         [0]], dtype=int32)'''
		...
	@overload
	def reduce_minimum(self, dims_mask: int, keepdims_mask: int)-> Var:		
		'''Document:
		*
		    Returns the minimum elements in the input.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
		
		    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randint(10, shape=(2, 3))
		        >>> x
		        jt.Var([[4 1 2]
		         [0 2 4]], dtype=int32)
		        >>> jt.min(x)
		        jt.Var([0], dtype=int32)
		        >>> x.min()
		        jt.Var([0], dtype=int32)
		        >>> x.min(dim=1)
		        jt.Var([1 0], dtype=int32)
		        >>> x.min(dim=1, keepdims=True)
		        jt.Var([[1]
		         [0]], dtype=int32)'''
		...
	@overload
	def sum(self, dim: int, keepdims: bool=False)-> Var:		
		'''Document:
		*
		    Returns the sum of the input.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
		
		    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randint(10, shape=(2, 3))
		        >>> x
		        jt.Var([[4 1 2]
		         [0 2 4]], dtype=int32)
		        >>> jt.sum(x)
		        jt.Var([13], dtype=int32)
		        >>> x.sum()
		        jt.Var([13], dtype=int32)
		        >>> x.sum(dim=1)
		        jt.Var([7 6], dtype=int32)
		        >>> x.sum(dim=1, keepdims=True)
		        jt.Var([[7]
		         [6]], dtype=int32)'''
		...
	@overload
	def sum(self, dims: Tuple[int]=(), keepdims: bool=False)-> Var:		
		'''Document:
		*
		    Returns the sum of the input.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
		
		    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randint(10, shape=(2, 3))
		        >>> x
		        jt.Var([[4 1 2]
		         [0 2 4]], dtype=int32)
		        >>> jt.sum(x)
		        jt.Var([13], dtype=int32)
		        >>> x.sum()
		        jt.Var([13], dtype=int32)
		        >>> x.sum(dim=1)
		        jt.Var([7 6], dtype=int32)
		        >>> x.sum(dim=1, keepdims=True)
		        jt.Var([[7]
		         [6]], dtype=int32)'''
		...
	@overload
	def sum(self, dims_mask: int, keepdims_mask: int)-> Var:		
		'''Document:
		*
		    Returns the sum of the input.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
		
		    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randint(10, shape=(2, 3))
		        >>> x
		        jt.Var([[4 1 2]
		         [0 2 4]], dtype=int32)
		        >>> jt.sum(x)
		        jt.Var([13], dtype=int32)
		        >>> x.sum()
		        jt.Var([13], dtype=int32)
		        >>> x.sum(dim=1)
		        jt.Var([7 6], dtype=int32)
		        >>> x.sum(dim=1, keepdims=True)
		        jt.Var([[7]
		         [6]], dtype=int32)'''
		...
	@overload
	def reduce_add(self, dim: int, keepdims: bool=False)-> Var:		
		'''Document:
		*
		    Returns the sum of the input.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
		
		    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randint(10, shape=(2, 3))
		        >>> x
		        jt.Var([[4 1 2]
		         [0 2 4]], dtype=int32)
		        >>> jt.sum(x)
		        jt.Var([13], dtype=int32)
		        >>> x.sum()
		        jt.Var([13], dtype=int32)
		        >>> x.sum(dim=1)
		        jt.Var([7 6], dtype=int32)
		        >>> x.sum(dim=1, keepdims=True)
		        jt.Var([[7]
		         [6]], dtype=int32)'''
		...
	@overload
	def reduce_add(self, dims: Tuple[int]=(), keepdims: bool=False)-> Var:		
		'''Document:
		*
		    Returns the sum of the input.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
		
		    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randint(10, shape=(2, 3))
		        >>> x
		        jt.Var([[4 1 2]
		         [0 2 4]], dtype=int32)
		        >>> jt.sum(x)
		        jt.Var([13], dtype=int32)
		        >>> x.sum()
		        jt.Var([13], dtype=int32)
		        >>> x.sum(dim=1)
		        jt.Var([7 6], dtype=int32)
		        >>> x.sum(dim=1, keepdims=True)
		        jt.Var([[7]
		         [6]], dtype=int32)'''
		...
	@overload
	def reduce_add(self, dims_mask: int, keepdims_mask: int)-> Var:		
		'''Document:
		*
		    Returns the sum of the input.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
		
		    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randint(10, shape=(2, 3))
		        >>> x
		        jt.Var([[4 1 2]
		         [0 2 4]], dtype=int32)
		        >>> jt.sum(x)
		        jt.Var([13], dtype=int32)
		        >>> x.sum()
		        jt.Var([13], dtype=int32)
		        >>> x.sum(dim=1)
		        jt.Var([7 6], dtype=int32)
		        >>> x.sum(dim=1, keepdims=True)
		        jt.Var([[7]
		         [6]], dtype=int32)'''
		...
	@overload
	def prod(self, dim: int, keepdims: bool=False)-> Var:		
		'''Document:
		*
		    Returns the product of all the elements in the input.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
		
		    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randint(10, shape=(2, 3))
		        >>> x
		        jt.Var([[7 5 5]
		         [5 7 5]], dtype=int32)
		        >>> jt.prod(x)
		        jt.Var([30625], dtype=int32)
		        >>> x.prod()
		        jt.Var([30625], dtype=int32)
		        >>> x.prod(dim=1)
		        jt.Var([175 175], dtype=int32)
		        >>> x.prod(dim=1, keepdims=True)
		        jt.Var([[175]
		         [175]], dtype=int32)'''
		...
	@overload
	def prod(self, dims: Tuple[int]=(), keepdims: bool=False)-> Var:		
		'''Document:
		*
		    Returns the product of all the elements in the input.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
		
		    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randint(10, shape=(2, 3))
		        >>> x
		        jt.Var([[7 5 5]
		         [5 7 5]], dtype=int32)
		        >>> jt.prod(x)
		        jt.Var([30625], dtype=int32)
		        >>> x.prod()
		        jt.Var([30625], dtype=int32)
		        >>> x.prod(dim=1)
		        jt.Var([175 175], dtype=int32)
		        >>> x.prod(dim=1, keepdims=True)
		        jt.Var([[175]
		         [175]], dtype=int32)'''
		...
	@overload
	def prod(self, dims_mask: int, keepdims_mask: int)-> Var:		
		'''Document:
		*
		    Returns the product of all the elements in the input.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
		
		    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randint(10, shape=(2, 3))
		        >>> x
		        jt.Var([[7 5 5]
		         [5 7 5]], dtype=int32)
		        >>> jt.prod(x)
		        jt.Var([30625], dtype=int32)
		        >>> x.prod()
		        jt.Var([30625], dtype=int32)
		        >>> x.prod(dim=1)
		        jt.Var([175 175], dtype=int32)
		        >>> x.prod(dim=1, keepdims=True)
		        jt.Var([[175]
		         [175]], dtype=int32)'''
		...
	@overload
	def product(self, dim: int, keepdims: bool=False)-> Var:		
		'''Document:
		*
		    Returns the product of all the elements in the input.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
		
		    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randint(10, shape=(2, 3))
		        >>> x
		        jt.Var([[7 5 5]
		         [5 7 5]], dtype=int32)
		        >>> jt.prod(x)
		        jt.Var([30625], dtype=int32)
		        >>> x.prod()
		        jt.Var([30625], dtype=int32)
		        >>> x.prod(dim=1)
		        jt.Var([175 175], dtype=int32)
		        >>> x.prod(dim=1, keepdims=True)
		        jt.Var([[175]
		         [175]], dtype=int32)'''
		...
	@overload
	def product(self, dims: Tuple[int]=(), keepdims: bool=False)-> Var:		
		'''Document:
		*
		    Returns the product of all the elements in the input.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
		
		    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randint(10, shape=(2, 3))
		        >>> x
		        jt.Var([[7 5 5]
		         [5 7 5]], dtype=int32)
		        >>> jt.prod(x)
		        jt.Var([30625], dtype=int32)
		        >>> x.prod()
		        jt.Var([30625], dtype=int32)
		        >>> x.prod(dim=1)
		        jt.Var([175 175], dtype=int32)
		        >>> x.prod(dim=1, keepdims=True)
		        jt.Var([[175]
		         [175]], dtype=int32)'''
		...
	@overload
	def product(self, dims_mask: int, keepdims_mask: int)-> Var:		
		'''Document:
		*
		    Returns the product of all the elements in the input.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
		
		    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randint(10, shape=(2, 3))
		        >>> x
		        jt.Var([[7 5 5]
		         [5 7 5]], dtype=int32)
		        >>> jt.prod(x)
		        jt.Var([30625], dtype=int32)
		        >>> x.prod()
		        jt.Var([30625], dtype=int32)
		        >>> x.prod(dim=1)
		        jt.Var([175 175], dtype=int32)
		        >>> x.prod(dim=1, keepdims=True)
		        jt.Var([[175]
		         [175]], dtype=int32)'''
		...
	@overload
	def reduce_multiply(self, dim: int, keepdims: bool=False)-> Var:		
		'''Document:
		*
		    Returns the product of all the elements in the input.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
		
		    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randint(10, shape=(2, 3))
		        >>> x
		        jt.Var([[7 5 5]
		         [5 7 5]], dtype=int32)
		        >>> jt.prod(x)
		        jt.Var([30625], dtype=int32)
		        >>> x.prod()
		        jt.Var([30625], dtype=int32)
		        >>> x.prod(dim=1)
		        jt.Var([175 175], dtype=int32)
		        >>> x.prod(dim=1, keepdims=True)
		        jt.Var([[175]
		         [175]], dtype=int32)'''
		...
	@overload
	def reduce_multiply(self, dims: Tuple[int]=(), keepdims: bool=False)-> Var:		
		'''Document:
		*
		    Returns the product of all the elements in the input.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
		
		    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randint(10, shape=(2, 3))
		        >>> x
		        jt.Var([[7 5 5]
		         [5 7 5]], dtype=int32)
		        >>> jt.prod(x)
		        jt.Var([30625], dtype=int32)
		        >>> x.prod()
		        jt.Var([30625], dtype=int32)
		        >>> x.prod(dim=1)
		        jt.Var([175 175], dtype=int32)
		        >>> x.prod(dim=1, keepdims=True)
		        jt.Var([[175]
		         [175]], dtype=int32)'''
		...
	@overload
	def reduce_multiply(self, dims_mask: int, keepdims_mask: int)-> Var:		
		'''Document:
		*
		    Returns the product of all the elements in the input.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
		
		    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randint(10, shape=(2, 3))
		        >>> x
		        jt.Var([[7 5 5]
		         [5 7 5]], dtype=int32)
		        >>> jt.prod(x)
		        jt.Var([30625], dtype=int32)
		        >>> x.prod()
		        jt.Var([30625], dtype=int32)
		        >>> x.prod(dim=1)
		        jt.Var([175 175], dtype=int32)
		        >>> x.prod(dim=1, keepdims=True)
		        jt.Var([[175]
		         [175]], dtype=int32)'''
		...
	@overload
	def reduce_logical_and(self, dim: int, keepdims: bool=False)-> Var:		
		'''Document:
		*
		    Tests if all elements in input evaluate to True.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
		
		    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randint(2, shape=(2, 3))
		        >>> x
		        jt.Var([[1 1 1]
		         [0 1 0]], dtype=int32)
		        >>> jt.all_(x)
		        jt.Var([False], dtype=int32)
		        >>> x.all_()
		        jt.Var([False], dtype=int32)
		        >>> x.all_(dim=1)
		        jt.Var([True False], dtype=int32)
		        >>> x.all_(dim=1, keepdims=True)
		        jt.Var([[True]
		         [False]], dtype=int32)'''
		...
	@overload
	def reduce_logical_and(self, dims: Tuple[int]=(), keepdims: bool=False)-> Var:		
		'''Document:
		*
		    Tests if all elements in input evaluate to True.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
		
		    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randint(2, shape=(2, 3))
		        >>> x
		        jt.Var([[1 1 1]
		         [0 1 0]], dtype=int32)
		        >>> jt.all_(x)
		        jt.Var([False], dtype=int32)
		        >>> x.all_()
		        jt.Var([False], dtype=int32)
		        >>> x.all_(dim=1)
		        jt.Var([True False], dtype=int32)
		        >>> x.all_(dim=1, keepdims=True)
		        jt.Var([[True]
		         [False]], dtype=int32)'''
		...
	@overload
	def reduce_logical_and(self, dims_mask: int, keepdims_mask: int)-> Var:		
		'''Document:
		*
		    Tests if all elements in input evaluate to True.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
		
		    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randint(2, shape=(2, 3))
		        >>> x
		        jt.Var([[1 1 1]
		         [0 1 0]], dtype=int32)
		        >>> jt.all_(x)
		        jt.Var([False], dtype=int32)
		        >>> x.all_()
		        jt.Var([False], dtype=int32)
		        >>> x.all_(dim=1)
		        jt.Var([True False], dtype=int32)
		        >>> x.all_(dim=1, keepdims=True)
		        jt.Var([[True]
		         [False]], dtype=int32)'''
		...
	@overload
	def all_(self, dim: int, keepdims: bool=False)-> Var:		
		'''Document:
		*
		    Tests if all elements in input evaluate to True.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
		
		    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randint(2, shape=(2, 3))
		        >>> x
		        jt.Var([[1 1 1]
		         [0 1 0]], dtype=int32)
		        >>> jt.all_(x)
		        jt.Var([False], dtype=int32)
		        >>> x.all_()
		        jt.Var([False], dtype=int32)
		        >>> x.all_(dim=1)
		        jt.Var([True False], dtype=int32)
		        >>> x.all_(dim=1, keepdims=True)
		        jt.Var([[True]
		         [False]], dtype=int32)'''
		...
	@overload
	def all_(self, dims: Tuple[int]=(), keepdims: bool=False)-> Var:		
		'''Document:
		*
		    Tests if all elements in input evaluate to True.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
		
		    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randint(2, shape=(2, 3))
		        >>> x
		        jt.Var([[1 1 1]
		         [0 1 0]], dtype=int32)
		        >>> jt.all_(x)
		        jt.Var([False], dtype=int32)
		        >>> x.all_()
		        jt.Var([False], dtype=int32)
		        >>> x.all_(dim=1)
		        jt.Var([True False], dtype=int32)
		        >>> x.all_(dim=1, keepdims=True)
		        jt.Var([[True]
		         [False]], dtype=int32)'''
		...
	@overload
	def all_(self, dims_mask: int, keepdims_mask: int)-> Var:		
		'''Document:
		*
		    Tests if all elements in input evaluate to True.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
		
		    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randint(2, shape=(2, 3))
		        >>> x
		        jt.Var([[1 1 1]
		         [0 1 0]], dtype=int32)
		        >>> jt.all_(x)
		        jt.Var([False], dtype=int32)
		        >>> x.all_()
		        jt.Var([False], dtype=int32)
		        >>> x.all_(dim=1)
		        jt.Var([True False], dtype=int32)
		        >>> x.all_(dim=1, keepdims=True)
		        jt.Var([[True]
		         [False]], dtype=int32)'''
		...
	@overload
	def reduce_logical_or(self, dim: int, keepdims: bool=False)-> Var:		
		'''Document:
		*
		    Tests if any elements in input evaluate to True.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
		
		    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randint(2, shape=(2, 3))
		        >>> x
		        jt.Var([[1 0 1]
		         [0 0 0]], dtype=int32)
		        >>> jt.any_(x)
		        jt.Var([True], dtype=int32)
		        >>> x.any_()
		        jt.Var([True], dtype=int32)
		        >>> x.any_(dim=1)
		        jt.Var([True False], dtype=int32)
		        >>> x.any_(dim=1, keepdims=True)
		        jt.Var([[True]
		         [False]], dtype=int32)'''
		...
	@overload
	def reduce_logical_or(self, dims: Tuple[int]=(), keepdims: bool=False)-> Var:		
		'''Document:
		*
		    Tests if any elements in input evaluate to True.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
		
		    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randint(2, shape=(2, 3))
		        >>> x
		        jt.Var([[1 0 1]
		         [0 0 0]], dtype=int32)
		        >>> jt.any_(x)
		        jt.Var([True], dtype=int32)
		        >>> x.any_()
		        jt.Var([True], dtype=int32)
		        >>> x.any_(dim=1)
		        jt.Var([True False], dtype=int32)
		        >>> x.any_(dim=1, keepdims=True)
		        jt.Var([[True]
		         [False]], dtype=int32)'''
		...
	@overload
	def reduce_logical_or(self, dims_mask: int, keepdims_mask: int)-> Var:		
		'''Document:
		*
		    Tests if any elements in input evaluate to True.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
		
		    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randint(2, shape=(2, 3))
		        >>> x
		        jt.Var([[1 0 1]
		         [0 0 0]], dtype=int32)
		        >>> jt.any_(x)
		        jt.Var([True], dtype=int32)
		        >>> x.any_()
		        jt.Var([True], dtype=int32)
		        >>> x.any_(dim=1)
		        jt.Var([True False], dtype=int32)
		        >>> x.any_(dim=1, keepdims=True)
		        jt.Var([[True]
		         [False]], dtype=int32)'''
		...
	@overload
	def any_(self, dim: int, keepdims: bool=False)-> Var:		
		'''Document:
		*
		    Tests if any elements in input evaluate to True.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
		
		    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randint(2, shape=(2, 3))
		        >>> x
		        jt.Var([[1 0 1]
		         [0 0 0]], dtype=int32)
		        >>> jt.any_(x)
		        jt.Var([True], dtype=int32)
		        >>> x.any_()
		        jt.Var([True], dtype=int32)
		        >>> x.any_(dim=1)
		        jt.Var([True False], dtype=int32)
		        >>> x.any_(dim=1, keepdims=True)
		        jt.Var([[True]
		         [False]], dtype=int32)'''
		...
	@overload
	def any_(self, dims: Tuple[int]=(), keepdims: bool=False)-> Var:		
		'''Document:
		*
		    Tests if any elements in input evaluate to True.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
		
		    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randint(2, shape=(2, 3))
		        >>> x
		        jt.Var([[1 0 1]
		         [0 0 0]], dtype=int32)
		        >>> jt.any_(x)
		        jt.Var([True], dtype=int32)
		        >>> x.any_()
		        jt.Var([True], dtype=int32)
		        >>> x.any_(dim=1)
		        jt.Var([True False], dtype=int32)
		        >>> x.any_(dim=1, keepdims=True)
		        jt.Var([[True]
		         [False]], dtype=int32)'''
		...
	@overload
	def any_(self, dims_mask: int, keepdims_mask: int)-> Var:		
		'''Document:
		*
		    Tests if any elements in input evaluate to True.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
		
		    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randint(2, shape=(2, 3))
		        >>> x
		        jt.Var([[1 0 1]
		         [0 0 0]], dtype=int32)
		        >>> jt.any_(x)
		        jt.Var([True], dtype=int32)
		        >>> x.any_()
		        jt.Var([True], dtype=int32)
		        >>> x.any_(dim=1)
		        jt.Var([True False], dtype=int32)
		        >>> x.any_(dim=1, keepdims=True)
		        jt.Var([[True]
		         [False]], dtype=int32)'''
		...
	@overload
	def reduce_logical_xor(self, dim: int, keepdims: bool=False)-> Var: ...
	@overload
	def reduce_logical_xor(self, dims: Tuple[int]=(), keepdims: bool=False)-> Var: ...
	@overload
	def reduce_logical_xor(self, dims_mask: int, keepdims_mask: int)-> Var: ...
	@overload
	def reduce_bitwise_and(self, dim: int, keepdims: bool=False)-> Var: ...
	@overload
	def reduce_bitwise_and(self, dims: Tuple[int]=(), keepdims: bool=False)-> Var: ...
	@overload
	def reduce_bitwise_and(self, dims_mask: int, keepdims_mask: int)-> Var: ...
	@overload
	def reduce_bitwise_or(self, dim: int, keepdims: bool=False)-> Var: ...
	@overload
	def reduce_bitwise_or(self, dims: Tuple[int]=(), keepdims: bool=False)-> Var: ...
	@overload
	def reduce_bitwise_or(self, dims_mask: int, keepdims_mask: int)-> Var: ...
	@overload
	def reduce_bitwise_xor(self, dim: int, keepdims: bool=False)-> Var: ...
	@overload
	def reduce_bitwise_xor(self, dims: Tuple[int]=(), keepdims: bool=False)-> Var: ...
	@overload
	def reduce_bitwise_xor(self, dims_mask: int, keepdims_mask: int)-> Var: ...
	@overload
	def mean(self, dim: int, keepdims: bool=False)-> Var:		
		'''Document:
		*
		    Returns the mean value of the input.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
		
		    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randint(10, shape=(2, 3))
		        >>> x
		        jt.Var([[9 4 4]
		         [1 9 6]], dtype=int32)
		        >>> jt.mean(x)
		        jt.Var([5.5000005], dtype=float32)
		        >>> x.mean()
		        jt.Var([5.5000005], dtype=float32)
		        >>> x.mean(dim=1)
		        jt.Var([5.666667  5.3333335], dtype=float32)
		        >>> x.mean(dim=1, keepdims=True)
		        jt.Var([[5.666667 ]
		         [5.3333335]], dtype=float32)'''
		...
	@overload
	def mean(self, dims: Tuple[int]=(), keepdims: bool=False)-> Var:		
		'''Document:
		*
		    Returns the mean value of the input.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
		
		    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randint(10, shape=(2, 3))
		        >>> x
		        jt.Var([[9 4 4]
		         [1 9 6]], dtype=int32)
		        >>> jt.mean(x)
		        jt.Var([5.5000005], dtype=float32)
		        >>> x.mean()
		        jt.Var([5.5000005], dtype=float32)
		        >>> x.mean(dim=1)
		        jt.Var([5.666667  5.3333335], dtype=float32)
		        >>> x.mean(dim=1, keepdims=True)
		        jt.Var([[5.666667 ]
		         [5.3333335]], dtype=float32)'''
		...
	@overload
	def mean(self, dims_mask: int, keepdims_mask: int)-> Var:		
		'''Document:
		*
		    Returns the mean value of the input.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).
		
		    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randint(10, shape=(2, 3))
		        >>> x
		        jt.Var([[9 4 4]
		         [1 9 6]], dtype=int32)
		        >>> jt.mean(x)
		        jt.Var([5.5000005], dtype=float32)
		        >>> x.mean()
		        jt.Var([5.5000005], dtype=float32)
		        >>> x.mean(dim=1)
		        jt.Var([5.666667  5.3333335], dtype=float32)
		        >>> x.mean(dim=1, keepdims=True)
		        jt.Var([[5.666667 ]
		         [5.3333335]], dtype=float32)'''
		...
	def clone(self)-> Var: ...
	def unary(self, op: str)-> Var: ...
	def cast(self, op: str)-> Var: ...
	def int8(self)-> Var:		
		'''Document:
		*
		    Returns a copy of the input var, casted to int8.
		
		    ----------------
		
		    * [in] x:   the input jt.Var
		
		    ----------------
		    
		    Example-1::
		        >>> x = jt.rand(3) * 10 
		        >>> x
		        jt.Var([4.093273  2.0086648 8.474352 ], dtype=float32)
		        >>> x.int8()
		        jt.Var([4 2 8], dtype=int8)
		        >>> jt.int8(x)
		        jt.Var([4 2 8], dtype=int8)'''
		...
	def int16(self)-> Var:		
		'''Document:
		*
		    Returns a copy of the input var, casted to int16.
		
		    ----------------
		
		    * [in] x:   the input jt.Var
		
		    ----------------
		    
		    Example-1::
		        >>> x = jt.rand(3) * 10 
		        >>> x
		        jt.Var([4.093273  2.0086648 8.474352 ], dtype=float32)
		        >>> x.int16()
		        jt.Var([4 2 8], dtype=int16)
		        >>> jt.int16(x)
		        jt.Var([4 2 8], dtype=int16)'''
		...
	def int32(self)-> Var:		
		'''Document:
		*
		    Returns a copy of the input var, casted to int32.
		
		    ----------------
		
		    * [in] x:   the input jt.Var
		
		    ----------------
		    
		    Example-1::
		        >>> x = jt.rand(3) * 10 
		        >>> x
		        jt.Var([4.093273  2.0086648 8.474352 ], dtype=float32)
		        >>> x.int()
		        jt.Var([4 2 8], dtype=int32)
		        >>> jt.int(x)
		        jt.Var([4 2 8], dtype=int32)
		        >>> x.int32()
		        jt.Var([4 2 8], dtype=int32)
		        >>> jt.int32(x)
		        jt.Var([4 2 8], dtype=int32)
		        >>> x.long()
		        jt.Var([4 2 8], dtype=int32)
		        >>> jt.long(x)
		        jt.Var([4 2 8], dtype=int32)'''
		...
	def int64(self)-> Var:		
		'''Document:
		*
		    Returns a copy of the input var, casted to int64.
		
		    ----------------
		
		    * [in] x:   the input jt.Var
		
		    ----------------
		    
		    Example-1::
		        >>> x = jt.rand(3) * 10 
		        >>> x
		        jt.Var([4.093273  2.0086648 8.474352 ], dtype=float32)
		        >>> x.int64()
		        jt.Var([4 2 8], dtype=int64)
		        >>> jt.int64(x)
		        jt.Var([4 2 8], dtype=int64)'''
		...
	def uint8(self)-> Var:		
		'''Document:
		*
		    Returns a copy of the input var, casted to unsigned int8.
		
		    ----------------
		
		    * [in] x:   the input jt.Var
		
		    ----------------
		    
		    Example-1::
		        >>> x = jt.rand(3) * 10 
		        >>> x
		        jt.Var([4.093273  2.0086648 8.474352 ], dtype=float32)
		        >>> x.uint8()
		        jt.Var([4 2 8], dtype=uint8)
		        >>> jt.uint8(x)
		        jt.Var([4 2 8], dtype=uint8)'''
		...
	def uint16(self)-> Var:		
		'''Document:
		*
		    Returns a copy of the input var, casted to unsigned int16.
		
		    ----------------
		
		    * [in] x:   the input jt.Var
		
		    ----------------
		    
		    Example-1::
		        >>> x = jt.rand(3) * 10 
		        >>> x
		        jt.Var([4.093273  2.0086648 8.474352 ], dtype=float32)
		        >>> x.uint16()
		        jt.Var([4 2 8], dtype=uint16)
		        >>> jt.uint16(x)
		        jt.Var([4 2 8], dtype=uint16)'''
		...
	def uint32(self)-> Var:		
		'''Document:
		*
		    Returns a copy of the input var, casted to unsigned int32.
		
		    ----------------
		
		    * [in] x:   the input jt.Var
		
		    ----------------
		    
		    Example-1::
		        >>> x = jt.rand(3) * 10 
		        >>> x
		        jt.Var([4.093273  2.0086648 8.474352 ], dtype=float32)
		        >>> x.uint32()
		        jt.Var([4 2 8], dtype=uint32)
		        >>> jt.uint32(x)
		        jt.Var([4 2 8], dtype=uint32)'''
		...
	def uint64(self)-> Var:		
		'''Document:
		*
		    Returns a copy of the input var, casted to unsigned int64.
		
		    ----------------
		
		    * [in] x:   the input jt.Var
		
		    ----------------
		    
		    Example-1::
		        >>> x = jt.rand(3) * 10 
		        >>> x
		        jt.Var([4.093273  2.0086648 8.474352 ], dtype=float32)
		        >>> x.uint64()
		        jt.Var([4 2 8], dtype=uint64)
		        >>> jt.uint64(x)
		        jt.Var([4 2 8], dtype=uint64)'''
		...
	def float16(self)-> Var:		
		'''Document:
		*
		    Returns a copy of the input var, casted to float16 (half-precision float).
		
		    ----------------
		
		    * [in] x:   the input jt.Var
		
		    ----------------
		    
		    Example-1::
		        >>> x = jt.rand(3) * 10 
		        >>> x
		        jt.Var([4.093273  2.0086648 8.474352 ], dtype=float32)
		        >>> x.half()
		        jt.Var([4.094 2.008 8.48 ], dtype=float16)
		        >>> jt.half(x)
		        jt.Var([4.094 2.008 8.48 ], dtype=float16)
		        >>> x.float16()
		        jt.Var([4.094 2.008 8.48 ], dtype=float16)
		        >>> jt.float16(x)
		        jt.Var([4.094 2.008 8.48 ], dtype=float16)'''
		...
	def float32(self)-> Var:		
		'''Document:
		*
		    Returns a copy of the input var, casted to float32.
		
		    ----------------
		
		    * [in] x:   the input jt.Var
		
		    ----------------
		    
		    Example-1::
		        >>> x = jt.arange(3)
		        >>> x
		        jt.Var([0 1 2], dtype=int32)
		        >>> x.float()
		        jt.Var([0. 1. 2.], dtype=float32)
		        >>> jt.float(x) 
		        jt.Var([0. 1. 2.], dtype=float32)
		        >>> x.float32()
		        jt.Var([0. 1. 2.], dtype=float32)
		        >>> jt.float32(x) 
		        jt.Var([0. 1. 2.], dtype=float32)'''
		...
	def float64(self)-> Var:		
		'''Document:
		*
		    Returns a copy of the input var, casted to float64 (double-precision float).
		
		    ----------------
		
		    * [in] x:   the input jt.Var
		
		    ----------------
		    
		    Example-1::
		        >>> x = jt.arange(3)
		        >>> x
		        jt.Var([0 1 2], dtype=int32)
		        >>> x.double()
		        jt.Var([0. 1. 2.], dtype=float64)
		        >>> jt.double(x) 
		        jt.Var([0. 1. 2.], dtype=float64)
		        >>> x.float64()
		        jt.Var([0. 1. 2.], dtype=float64)
		        >>> jt.float64(x) 
		        jt.Var([0. 1. 2.], dtype=float64)'''
		...
	def abs(self)-> Var:		
		'''Document:
		*
		    Returns the absolute value of the input ``x``. 
		
		    ----------------
		
		    * [in] x:   the input jt.Var
		
		    ----------------
		    
		    Example-1::
		        >>> jt.abs(jt.float32([-1, 0, 1]))
		        jt.Var([1. 0. 1.], dtype=float32)'''
		...
	def negative(self)-> Var:		
		'''Document:
		*
		    Returns the negative value of the input ``x``. 
		
		    This operator is equavilant to ``-x``.
		
		    ----------------
		
		    * [in] x:   the input jt.Var.
		
		    ----------------
		    
		    Example-1::
		        >>> jt.negative(jt.float32([-1, 0, 1]))
		        jt.Var([ 1. -0. -1.], dtype=float32)'''
		...
	def logical_not(self)-> Var:		
		'''Document:
		*
		    Returns the logical NOT of the input ``x``. 
		     
		    ----------------
		
		    * [in] x: the input jt.Var, integal or boolean.
		
		    ----------------
		
		    Example-1::
		        >>> jt.logical_not(jt.int32([-1, 0, 1]))
		        jt.Var([False  True False], dtype=bool)'''
		...
	def bitwise_not(self)-> Var:		
		'''Document:
		*
		    Returns the bitwise NOT of the input ``x``. 
		
		    ----------------
		
		    * [in] x: the input jt.Var, integal or boolean.
		
		    ----------------
		
		    Example-1::
		        >>> jt.bitwise_not(jt.int32([1, 2, -3]))
		        jt.Var([-2 -3  2], dtype=int32)'''
		...
	def log(self)-> Var:		
		'''Document:
		*
		    Returns the natural logarithm of the input ``x``. 
		
		    ----------------
		
		    * [in] x: the input jt.Var.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.rand(4) * 2
		        >>> x
		        jt.Var([0.02863695 1.30122    1.6048753  1.140261  ], dtype=float32)
		        >>> jt.log(x)
		        jt.Var([-3.5530574   0.26330233  0.47304606  0.13125724], dtype=float32)
		        >>> x.log()
		        jt.Var([-3.5530574   0.26330233  0.47304606  0.13125724], dtype=float32)'''
		...
	def exp(self)-> Var:		
		'''Document:
		*
		     Returns the exponential of the input ``x``. 
		
		    ----------------
		
		    * [in] x: the input jt.Var.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.rand(4) * 2
		        >>> x
		        jt.Var([1.9841381 1.4103996 0.5855549 1.4212812], dtype=float32)
		        >>> jt.exp(x)
		        jt.Var([7.2727766 4.0975924 1.7959872 4.1424246], dtype=float32)
		        >>> x.exp()
		        jt.Var([7.2727766 4.0975924 1.7959872 4.1424246], dtype=float32)'''
		...
	def sqrt(self)-> Var:		
		'''Document:
		*
		    Returns the square root of the input ``x``. 
		
		    ----------------
		
		    * [in] x: the input jt.Var.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.rand(4) * 2
		        >>> x
		        jt.Var([0.81957287 0.5609612  0.07435933 1.7571875 ], dtype=float32)
		        >>> jt.sqrt(x)
		        jt.Var([0.90530264 0.7489734  0.27268907 1.3255895 ], dtype=float32)
		        >>> x.sqrt()
		        jt.Var([0.90530264 0.7489734  0.27268907 1.3255895 ], dtype=float32)'''
		...
	def round(self)-> Var:		
		'''Document:
		*
		    Returns the closest integer of the input ``x``. 
		
		    ----------------
		
		    * [in] x: the input jt.Var.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randn(4)
		        >>> x
		        jt.Var([ 2.101595    0.33055413 -0.44147047 -0.7720668 ], dtype=float32)
		        >>> jt.round(x)
		        jt.Var([ 2.0  0.0  0.0 -1.0], dtype=float32)
		        >>> x.round()
		        jt.Var([ 2.0  0.0  0.0 -1.0], dtype=float32)'''
		...
	def floor(self)-> Var:		
		'''Document:
		*
		     Returns the largest integer less than or equal to the input ``x``. 
		
		    ----------------
		
		    * [in] x: the input jt.Var.
		
		    ----------------
		    Example-1::
		        >>> x = jt.randn(4)
		        >>> x
		        jt.Var([-1.0339162 -0.7259972 -0.9220003 -0.8449701], dtype=float32)
		        >>> jt.floor(x)
		        jt.Var([-2.0 -1.0 -1.0 -1.0], dtype=float32)
		        >>> x.floor
		        jt.Var([-2.0 -1.0 -1.0 -1.0], dtype=float32)'''
		...
	def ceil(self)-> Var:		
		'''Document:
		*
		    Returns the smallest integer greater than or equal to the input ``x``. 
		
		    ----------------
		
		    * [in] x: the input jt.Var.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randn(4)
		        >>> x
		        jt.Var([-1.0339162 -0.7259972 -0.9220003 -0.8449701], dtype=float32)
		        >>> jt.ceil(x)
		        jt.Var([-1.0  0.0  0.0  0.0], dtype=float32)
		        >>> x.ceil()
		        jt.Var([-1.0  0.0  0.0  0.0], dtype=float32)'''
		...
	def round_int(self)-> Var:		
		'''Document:
		*
		    Returns the closest integer of the input ``x``. 
		
		    ----------------
		
		    * [in] x: the input jt.Var.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randn(4)
		        >>> x
		        jt.Var([ 2.101595    0.33055413 -0.44147047 -0.7720668 ], dtype=float32)
		        >>> jt.round_int(x)
		        jt.Var([ 2  0  0 -1], dtype=int32)
		        >>> x.round_int
		        jt.Var([ 2  0  0 -1], dtype=int32)'''
		...
	def floor_int(self)-> Var:		
		'''Document:
		*
		     Returns the largest integer less than or equal to the input ``x``. 
		
		    ----------------
		
		    * [in] x: the input jt.Var.
		
		    ----------------
		    Example-1::
		        >>> x = jt.randn(4)
		        >>> x
		        jt.Var([-1.0339162 -0.7259972 -0.9220003 -0.8449701], dtype=float32)
		        >>> jt.floor_int(x)
		        jt.Var([-2 -1 -1 -1], dtype=int32)
		        >>> x.floor_int
		        jt.Var([-2 -1 -1 -1], dtype=int32)'''
		...
	def ceil_int(self)-> Var:		
		'''Document:
		*
		    Returns the smallest integer greater than or equal to the input ``x``. 
		
		    ----------------
		
		    * [in] x: the input jt.Var.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randn(4)
		        >>> x
		        jt.Var([-1.0339162 -0.7259972 -0.9220003 -0.8449701], dtype=float32)
		        >>> jt.ceil_int(x)
		        jt.Var([-1  0  0  0], dtype=int32)
		        >>> x.ceil_int()
		        jt.Var([-1  0  0  0], dtype=int32)'''
		...
	def sin(self)-> Var:		
		'''Document:
		*
		    Returns the sine of the input ``x``. 
		
		    ----------------
		
		    * [in] x: the input jt.Var.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randn(4)
		        >>> x
		        jt.Var([ 0.32893723 -0.7112559  -0.872391    1.8001337 ], dtype=float32)
		        >>> jt.sin(x)
		        jt.Var([ 0.32303742 -0.6527857  -0.76586854  0.9738172 ], dtype=float32)
		        >>> x.sin()
		        jt.Var([ 0.32303742 -0.6527857  -0.76586854  0.9738172 ], dtype=float32)'''
		...
	def asin(self)-> Var:		
		'''Document:
		*
		    Returns the arcsine of the input ``x``. 
		
		    ----------------
		
		    * [in] x: the input jt.Var.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randn(4)
		        >>> x
		        jt.Var([ 0.09342023 -0.42522037  0.9264933  -0.785264  ], dtype=float32)
		        >>> jt.asin(x)
		        jt.Var([ 0.09355665 -0.43920535  1.1849847  -0.9031224 ], dtype=float32)
		        >>> x.asin()
		        jt.Var([ 0.09355665 -0.43920535  1.1849847  -0.9031224 ], dtype=float32)'''
		...
	def arcsin(self)-> Var:		
		'''Document:
		*
		    Returns the arcsine of the input ``x``. 
		
		    ----------------
		
		    * [in] x: the input jt.Var.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randn(4)
		        >>> x
		        jt.Var([ 0.09342023 -0.42522037  0.9264933  -0.785264  ], dtype=float32)
		        >>> jt.asin(x)
		        jt.Var([ 0.09355665 -0.43920535  1.1849847  -0.9031224 ], dtype=float32)
		        >>> x.asin()
		        jt.Var([ 0.09355665 -0.43920535  1.1849847  -0.9031224 ], dtype=float32)'''
		...
	def sinh(self)-> Var:		
		'''Document:
		*
		    Returns the hyperbolic sine of the input ``x``. 
		
		    ----------------
		
		    * [in] x: the input jt.Var.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randn(4)
		        >>> x
		        jt.Var([ 0.32893723 -0.7112559  -0.872391    1.8001337 ], dtype=float32)
		        >>> jt.sinh(x)
		        jt.Var([ 0.3349012  -0.77276015 -0.9873369   2.9425898 ], dtype=float32)
		        >>> x.sinh
		        jt.Var([ 0.3349012  -0.77276015 -0.9873369   2.9425898 ], dtype=float32)'''
		...
	def asinh(self)-> Var:		
		'''Document:
		*
		    Returns the inverse hyperbolic sine of the input ``x``. 
		
		    ----------------
		
		    * [in] x: the input jt.Var.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randn(4)
		        >>> x
		        jt.Var([-1.9749726  -0.52341473  0.8906148   1.0338128 ], dtype=float32)
		        >>> jt.asinh(x)
		        jt.Var([-1.4323865  -0.5020559   0.8018747   0.90508187], dtype=float32)
		        >>> x.asinh()
		        jt.Var([-1.4323865  -0.5020559   0.8018747   0.90508187], dtype=float32)'''
		...
	def arcsinh(self)-> Var:		
		'''Document:
		*
		    Returns the inverse hyperbolic sine of the input ``x``. 
		
		    ----------------
		
		    * [in] x: the input jt.Var.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randn(4)
		        >>> x
		        jt.Var([-1.9749726  -0.52341473  0.8906148   1.0338128 ], dtype=float32)
		        >>> jt.asinh(x)
		        jt.Var([-1.4323865  -0.5020559   0.8018747   0.90508187], dtype=float32)
		        >>> x.asinh()
		        jt.Var([-1.4323865  -0.5020559   0.8018747   0.90508187], dtype=float32)'''
		...
	def tan(self)-> Var:		
		'''Document:
		*
		    Returns the tangent of the input ``x``. 
		
		    ----------------
		
		    * [in] x: the input jt.Var.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randn(4)
		        >>> x
		        jt.Var([ 0.32893723 -0.7112559  -0.872391    1.8001337 ], dtype=float32)
		        >>> jt.tan(x)
		        jt.Var([ 0.34133783 -0.8617148  -1.1910915  -4.283673  ], dtype=float32)
		        >>> x.tan()
		        jt.Var([ 0.34133783 -0.8617148  -1.1910915  -4.283673  ], dtype=float32)'''
		...
	def atan(self)-> Var:		
		'''Document:
		*
		    Returns the inverse tangent of the input ``x``. 
		
		    ----------------
		
		    * [in] x: the input jt.Var.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randn(4)
		        >>> x
		        jt.Var([-0.85885596  1.187804    0.47249675  0.95933187], dtype=float32)
		        >>> jt.atan(x)
		        jt.Var([-0.70961297  0.87102956  0.44140393  0.76464504], dtype=float32)
		        >>> x.atan()
		        jt.Var([-0.70961297  0.87102956  0.44140393  0.76464504], dtype=float32)'''
		...
	def arctan(self)-> Var:		
		'''Document:
		*
		    Returns the inverse tangent of the input ``x``. 
		
		    ----------------
		
		    * [in] x: the input jt.Var.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randn(4)
		        >>> x
		        jt.Var([-0.85885596  1.187804    0.47249675  0.95933187], dtype=float32)
		        >>> jt.atan(x)
		        jt.Var([-0.70961297  0.87102956  0.44140393  0.76464504], dtype=float32)
		        >>> x.atan()
		        jt.Var([-0.70961297  0.87102956  0.44140393  0.76464504], dtype=float32)'''
		...
	def tanh(self)-> Var:		
		'''Document:
		*
		    Returns the hyperbolic tangent of the input ``x``. 
		
		    ----------------
		
		    * [in] x: the input jt.Var.
		
		    ----------------
		    
		    Example-1::
		        >>> x = jt.randn(4)
		        >>> x
		        jt.Var([-0.85885596  1.187804    0.47249675  0.95933187], dtype=float32)
		        >>> jt.tanh(x)
		        jt.Var([-0.6956678   0.82989657  0.4402144   0.7439787 ], dtype=float32)
		        >>> x.tanh()
		        jt.Var([-0.6956678   0.82989657  0.4402144   0.7439787 ], dtype=float32)'''
		...
	def atanh(self)-> Var:		
		'''Document:
		*
		    Returns the inverse hyperbolic tangent of the input ``x``. 
		
		    ----------------
		
		    * [in] x: the input jt.Var.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.rand(4) * 2 - 1
		        >>> x
		        jt.Var([ 0.9062414  -0.799802   -0.27219176 -0.7274077 ], dtype=float32)
		        >>> jt.atanh(x)
		        jt.Var([ 1.5060828  -1.0980625  -0.27922946 -0.9231999 ], dtype=float32)
		        >>> x.atanh()
		        jt.Var([ 1.5060828  -1.0980625  -0.27922946 -0.9231999 ], dtype=float32)'''
		...
	def arctanh(self)-> Var:		
		'''Document:
		*
		    Returns the inverse hyperbolic tangent of the input ``x``. 
		
		    ----------------
		
		    * [in] x: the input jt.Var.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.rand(4) * 2 - 1
		        >>> x
		        jt.Var([ 0.9062414  -0.799802   -0.27219176 -0.7274077 ], dtype=float32)
		        >>> jt.atanh(x)
		        jt.Var([ 1.5060828  -1.0980625  -0.27922946 -0.9231999 ], dtype=float32)
		        >>> x.atanh()
		        jt.Var([ 1.5060828  -1.0980625  -0.27922946 -0.9231999 ], dtype=float32)'''
		...
	def cos(self)-> Var:		
		'''Document:
		*
		    Returns the cosine of the input ``x``. 
		
		    ----------------
		
		    * [in] x: the input jt.Var.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randn(4)
		        >>> x
		        jt.Var([ 0.32893723 -0.7112559  -0.872391    1.8001337 ], dtype=float32)
		        >>> jt.cos(x)
		        jt.Var([ 0.9463862  0.7575426  0.6429972 -0.2273323], dtype=float32)
		        >>> x.cos()
		        jt.Var([ 0.9463862  0.7575426  0.6429972 -0.2273323], dtype=float32)'''
		...
	def acos(self)-> Var:		
		'''Document:
		*
		    Returns the inverse cosine of the input ``x``. 
		
		    ----------------
		
		    * [in] x: the input jt.Var.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.rand(4) * 2 - 1
		        >>> x
		        jt.Var([ 0.5876564  0.740723  -0.667666   0.5371753], dtype=float32)
		        >>> jt.acos(x)
		        jt.Var([0.9426371 0.7366504 2.3018656 1.0037117], dtype=float32)
		        >>> x.acos()
		        jt.Var([0.9426371 0.7366504 2.3018656 1.0037117], dtype=float32)'''
		...
	def arccos(self)-> Var:		
		'''Document:
		*
		    Returns the inverse cosine of the input ``x``. 
		
		    ----------------
		
		    * [in] x: the input jt.Var.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.rand(4) * 2 - 1
		        >>> x
		        jt.Var([ 0.5876564  0.740723  -0.667666   0.5371753], dtype=float32)
		        >>> jt.acos(x)
		        jt.Var([0.9426371 0.7366504 2.3018656 1.0037117], dtype=float32)
		        >>> x.acos()
		        jt.Var([0.9426371 0.7366504 2.3018656 1.0037117], dtype=float32)'''
		...
	def cosh(self)-> Var:		
		'''Document:
		*
		    Returns the hyperbolic cosine of the input ``x``. 
		
		    ----------------
		
		    * [in] x: the input jt.Var.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randn(4)
		        >>> x
		        jt.Var([ 0.32893723 -0.7112559  -0.872391    1.8001337 ], dtype=float32)
		        >>> jt.cosh(x)
		        jt.Var([1.0545894 1.2637873 1.405288  3.1078668], dtype=float32)
		        >>> x.cosh()
		        jt.Var([1.0545894 1.2637873 1.405288  3.1078668], dtype=float32)'''
		...
	def acosh(self)-> Var:		
		'''Document:
		*
		    Returns the inverse hyperbolic cosine of the input ``x``. 
		
		    ----------------
		
		    * [in] x: the input jt.Var.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.rand(4) + 1
		        >>> x
		        jt.Var([1.3609099 1.8137748 1.1146184 1.3911307], dtype=float32)
		        >>> jt.acosh(x)
		        jt.Var([0.8259237  1.2020639  0.47432774 0.8579033 ], dtype=float32)
		        >>> x.acosh()
		        jt.Var([0.8259237  1.2020639  0.47432774 0.8579033 ], dtype=float32)'''
		...
	def arccosh(self)-> Var:		
		'''Document:
		*
		    Returns the inverse hyperbolic cosine of the input ``x``. 
		
		    ----------------
		
		    * [in] x: the input jt.Var.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.rand(4) + 1
		        >>> x
		        jt.Var([1.3609099 1.8137748 1.1146184 1.3911307], dtype=float32)
		        >>> jt.acosh(x)
		        jt.Var([0.8259237  1.2020639  0.47432774 0.8579033 ], dtype=float32)
		        >>> x.acosh()
		        jt.Var([0.8259237  1.2020639  0.47432774 0.8579033 ], dtype=float32)'''
		...
	def sigmoid(self)-> Var:		
		'''Document:
		*
		    Returns the sigmoid of the input ``x``. 
		    
		    .. math::
		       out_i = \frac{1}{1 + e^{x_i}}
		
		    ----------------
		
		    * [in] x: the input jt.Var.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randn(4)
		        >>> x
		        jt.Var([ 0.49443012  0.4305426  -1.0364404  -1.2628382 ], dtype=float32)
		        >>> jt.sigmoid(x)
		        jt.Var([0.62114954 0.6060032  0.2618374  0.2204857 ], dtype=float32)
		        >>> x.sigmoid()
		        jt.Var([0.62114954 0.6060032  0.2618374  0.2204857 ], dtype=float32)'''
		...
	def erf(self)-> Var:		
		'''Document:
		*
		    Computes the error function of each element. The error function is defined as follows:
		
		    .. math::
		        erf(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} dt
		
		    ----------------
		
		    * [in] x: the input jt.Var.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randn(4)
		        >>> x
		        jt.Var([ 0.49443012  0.4305426  -1.0364404  -1.2628382 ], dtype=float32)
		        >>> jt.erf(x)
		        jt.Var([ 0.51559156  0.45739546 -0.85728306 -0.9258883 ], dtype=float32)
		        >>> x.erf()
		        jt.Var([ 0.51559156  0.45739546 -0.85728306 -0.9258883 ], dtype=float32)'''
		...
	def erfinv(self)-> Var:		
		'''Document:
		*
		    Computes the inverse error function of each element. 
		
		    * [in] x: the input jt.Var.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.rand(4) * 2 - 1 
		        >>> x
		        jt.Var([ 0.00277209 -0.26642472  0.7869792   0.5415418 ], dtype=float32)
		        >>> jt.erfinv(x)
		        jt.Var([ 0.00245671 -0.24068035  0.8805613   0.5242405 ], dtype=float32)
		        >>> x.erfinv()
		        jt.Var([ 0.00245671 -0.24068035  0.8805613   0.5242405 ], dtype=float32)'''
		...
	def transpose(self, axes: Tuple[int]=())-> Var: ...
	def fuse_transpose(self, axes: Tuple[int]=())-> Var: ...
	def safe_clip(self, left: float, right: float)-> Var:		
		'''Document:
		* Safe clip value to a range, and keep 
		 the gradient pass thought.
		 
		    * [in] x:   input value
		    * [in] left: float64 clip min value.
		    * [in] right: float64 clip max value.'''
		...
	def array(self)-> Var: ...
	@overload
	def getitem(self, slices: slice)-> Var: ...
	@overload
	def getitem(self, slices: slice, _: int)-> Tuple[Var]: ...
	def candidate(self, fail_cond: str, dtype: str="int32")-> Var:		
		'''Document:
		*
		    Candidate Operator Perform an indirect candidate filter by given a fail condition.
		    
		    x is input, y is output index, satisfy::
		
		        not fail_cond(y[0], y[1]) and
		        not fail_cond(y[0], y[2]) and not fail_cond(y[1], y[2]) and
		        ...
		        ... and not fail_cond(y[m-2], y[m-1])
		
		    Where m is number of selected candidates.
		
		    Pseudo code::
		    
		        y = []
		        for i in range(n):
		            pass = True
		            for j in y:
		                if (@fail_cond):
		                    pass = false
		                    break
		            if (pass):
		                y.append(i)
		        return y
		
		    * [in] x:   input var for filter
		
		    * [in] fail_cond:   code for fail condition
		
		    * [in] dtype:   type of return indexes
		
		    * [out] index: .
		
		    Example::
		
		        jt.candidate(jt.random(100,2), '(@x(j,0)>@x(i,0))or(@x(j,1)>@x(i,1))')
		        # return y satisfy:
		        #    x[y[0], 0] <= x[y[1], 0] and x[y[1], 0] <= x[y[2], 0] and ... and x[y[m-2], 0] <= x[y[m-1], 0] and
		        #    x[y[0], 1] <= x[y[1], 1] and x[y[1], 1] <= x[y[2], 1] and ... and x[y[m-2], 1] <= x[y[m-1], 1]'''
		...
	@overload
	def code(self, outputs: List[Var], cpu_src: str="", cpu_grad_src: List[str]={}, cpu_header: str="", cuda_src: str="", cuda_grad_src: List[str]={}, cuda_header: str="")-> Tuple[Var]:		
		'''Document:
		*
		    Code Operator for easily customized op.
		
		    ----------------
		
		    * [in] shape:   the output shape, a integer array
		    
		    * [in] dtype:   the output data type
		    
		    * [in] inputs:  A list of input jittor Vars
		    
		    * [in] cpu_src: cpu source code string, buildin value:
		
		            *   in{x}, in{x}_shape{y}, in{x}_stride{y}, in{x}_type, in{x}_p, @in0(...)
		            *   out{x}, out{x}_shape{y}, out{x}_stride{y}, out{x}_type, out{x}_p, @out0(...)
		            *   out, out_shape{y}, out_stride{y}, out_type, out_p, @out(...)
		    
		    * [in] cpu_header: cpu header code string.
		
		    * [in] cuda_src: cuda source code string.
		
		    * [in] cuda_header: cuda header code string.
		
		    ----------------
		    
		    Example-1::
		
		        from jittor import Function
		        import jittor as jt
		
		        class Func(Function):
		            def execute(self, x):
		                self.save_vars = x
		                return jt.code(x.shape, x.dtype, [x],
		                    cpu_src="""
		                        for (int i=0; i<in0_shape0; i++)
		                            @out(i) = @in0(i)*@in0(i)*2;
		                    """)
		
		            def grad(self, grad_x):
		                x = self.save_vars
		                return jt.code(x.shape, x.dtype, [x, grad_x],
		                    cpu_src="""
		                        for (int i=0; i<in0_shape0; i++)
		                            @out(i) = @in1(i)*@in0(i)*4;
		                    """)
		
		        a = jt.random([10])
		        func = Func()
		        b = func(a)
		        print(b)
		        print(jt.grad(b,a))
		
		    Example-2::
		
		        a = jt.array([3,2,1])
		        b = jt.code(a.shape, a.dtype, [a],
		            cpu_header="""
		                #include <algorithm>
		                @alias(a, in0)
		                @alias(b, out)
		            """,
		            cpu_src="""
		                for (int i=0; i<a_shape0; i++)
		                    @b(i) = @a(i);
		                std::sort(&@b(0), &@b(in0_shape0));
		            """
		        )
		        assert (b.data==[1,2,3]).all()
		
		    Example-3::
		
		        #This example shows how to set multiple outputs in code op.
		        a = jt.array([3,2,1])
		        b,c = jt.code([(1,), (1,)], [a.dtype, a.dtype], [a],
		            cpu_header="""
		                #include <iostream>
		                using namespace std;
		            """,
		            cpu_src="""
		                @alias(a, in0)
		                @alias(b, out0)
		                @alias(c, out1)
		                @b(0) = @c(0) = @a(0);
		                for (int i=0; i<a_shape0; i++) {
		                    @b(0) = std::min(@b(0), @a(i));
		                    @c(0) = std::max(@c(0), @a(i));
		                }
		                cout << "min:" << @b(0) << " max:" << @c(0) << endl;
		            """
		        )
		        assert b.data == 1, b
		        assert c.data == 3, c
		
		    Example-4::
		
		        #This example shows how to use dynamic shape of jittor variables.
		        a = jt.array([5,-4,3,-2,1])
		        
		        # negtive shape for max size of vary dimension
		        b,c = jt.code([(-5,), (-5,)], [a.dtype, a.dtype], [a],
		            cpu_src="""
		                @alias(a, in0)
		                @alias(b, out0)
		                @alias(c, out1)
		                int num_b=0, num_c=0;
		                for (int i=0; i<a_shape0; i++) {
		                    if (@a(i)>0)
		                        @b(num_b++) = @a(i);
		                    else
		                        @c(num_c++) = @a(i);
		                }
		                b->set_shape({num_b});
		                c->set_shape({num_c});
		            """
		        )
		        assert (b.data == [5,3,1]).all()
		        assert (c.data == [-4,-2]).all()
		
		    Example-5::
		
		        # This example shows how to customize code op
		        # compilation flags, such as add include search
		        # path, add definitions, or any command line options
		
		        a = jt.random([10])
		        b = jt.code(a.shape, a.dtype, [a],
		            cpu_src="""
		                @out0(0) = HAHAHA;
		            """)
		        # HAHAHA is defined in flags below
		        # /any/include/path can be change to any path you want to include
		        b.compile_options = {"FLAGS: -DHAHAHA=233 -I/any/include/path ": 1}
		        print(b[0])
		        # will output 233
		
		
		    CUDA Example-1::
		
		        #This example shows how to use CUDA in code op.
		        import jittor as jt
		        from jittor import Function
		        jt.flags.use_cuda = 1
		
		        class Func(Function):
		            def execute(self, a, b):
		                self.save_vars = a, b
		                return jt.code(a.shape, a.dtype, [a,b],
		                    cuda_src="""
		                        __global__ static void kernel1(@ARGS_DEF) {
		                            @PRECALC
		                            int i = threadIdx.x + blockIdx.x * blockDim.x;
		                            int stride = blockDim.x * gridDim.x;
		                            for (; i<in0_shape0; i+=stride)
		                                @out(i) = @in0(i)*@in1(i);
		                        }
		                        kernel1<<<(in0_shape0-1)/1024+1, 1024>>>(@ARGS);
		                    """)
		
		            def grad(self, grad):
		                a, b = self.save_vars
		                return jt.code([a.shape, b.shape], [a.dtype, b.dtype], [a, b, grad],
		                    cuda_src="""
		                        __global__ static void kernel2(@ARGS_DEF) {
		                            @PRECALC
		                            int i = threadIdx.x + blockIdx.x * blockDim.x;
		                            int stride = blockDim.x * gridDim.x;
		                            for (; i<in0_shape0; i+=stride) {
		                                @out0(i) = @in2(i)*@in1(i);
		                                @out1(i) = @in2(i)*@in0(i);
		                            }
		                        }
		                        kernel2<<<(in0_shape0-1)/1024+1, 1024>>>(@ARGS);
		                    """)
		                
		        a = jt.random([100000])
		        b = jt.random([100000])
		        func = Func()
		        c = func(a,b)
		        print(c)
		        print(jt.grad(c, [a, b]))
		
		    CUDA Example-2::
		    
		        #This example shows how to use multi dimension data with CUDA.
		        import jittor as jt
		        from jittor import Function
		        jt.flags.use_cuda = 1
		
		        class Func(Function):
		            def execute(self, a, b):
		                self.save_vars = a, b
		                return jt.code(a.shape, a.dtype, [a,b],
		                    cuda_src="""
		                        __global__ static void kernel1(@ARGS_DEF) {
		                            @PRECALC
		                            for (int i=blockIdx.x; i<in0_shape0; i+=gridDim.x)
		                            for (int j=threadIdx.x; j<in0_shape1; j+=blockDim.x)
		                                @out(i,j) = @in0(i,j)*@in1(i,j);
		                        }
		                        kernel1<<<32, 32>>>(@ARGS);
		                    """)
		
		            def grad(self, grad):
		                a, b = self.save_vars
		                return jt.code([a.shape, b.shape], [a.dtype, b.dtype], [a, b, grad],
		                    cuda_src="""
		                        __global__ static void kernel2(@ARGS_DEF) {
		                            @PRECALC
		                            for (int i=blockIdx.x; i<in0_shape0; i+=gridDim.x)
		                            for (int j=threadIdx.x; j<in0_shape1; j+=blockDim.x) {
		                                @out0(i,j) = @in2(i,j)*@in1(i,j);
		                                @out1(i,j) = @in2(i,j)*@in0(i,j);
		                            }
		                        }
		                        kernel2<<<32, 32>>>(@ARGS);
		                    """)
		                
		        a = jt.random((100,100))
		        b = jt.random((100,100))
		        func = Func()
		        c = func(a,b)
		        print(c)
		        print(jt.grad(c, [a, b]))'''
		...
	def copy(self)-> Var: ...
	def setitem(self, slices: slice, y: Var, op: str="void")-> Var: ...
	@overload
	def broadcast(self, shape: Tuple[int], dims: Tuple[int]=())-> Var:		
		'''Document:
		*
		    Broadcast ``x`` to a given shape.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] shape:   the output shape.
		
		    * [in] dims:    specifies the new dimension in the output shape, an integer array.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randint(0, 10, shape=(2, 2))
		        >>> x
		        jt.Var([[8 1]
		         [7 6]], dtype=int32)
		        >>> jt.broadcast(x, shape=(2, 3, 2), dims=[1])
		        jt.Var([[[8 1]
		          [8 1]
		          [8 1]],
		         [[7 6]
		          [7 6]
		          [7 6]]], dtype=int32)'''
		...
	@overload
	def broadcast(self, y: Var, dims: Tuple[int]=())-> Var:		
		'''Document:
		*
		    Broadcast ``x`` to the same shape as ``y``.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] y:       the reference jt.Var.
		
		    * [in] dims:    specifies the new dimension in the output shape, an integer array.
		
		    ----------------
		
		    .. note::
		      jt.broadcast_var(x, y, dims) is an alias of jt.broadcast(x, y, dims)
		
		    Example-1::
		        >>> x = jt.randint(0, 10, shape=(2, 2))
		        >>> x
		        jt.Var([[8 1]
		         [7 6]], dtype=int32)
		        >>> y = jt.randint(0, 10, shape=(2, 3, 2))
		        >>> jt.broadcast(x, y, dims=[1])
		        jt.Var([[[8 1]
		          [8 1]
		          [8 1]],
		         [[7 6]
		          [7 6]
		          [7 6]]], dtype=int32)
		        >>> jt.broadcast_var(x, y, dims=[1])
		        jt.Var([[[8 1]
		          [8 1]
		          [8 1]],
		         [[7 6]
		          [7 6]
		          [7 6]]], dtype=int32)'''
		...
	def broadcast_var(self, y: Var, dims: Tuple[int]=())-> Var:		
		'''Document:
		*
		    Broadcast ``x`` to the same shape as ``y``.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] y:       the reference jt.Var.
		
		    * [in] dims:    specifies the new dimension in the output shape, an integer array.
		
		    ----------------
		
		    .. note::
		      jt.broadcast_var(x, y, dims) is an alias of jt.broadcast(x, y, dims)
		
		    Example-1::
		        >>> x = jt.randint(0, 10, shape=(2, 2))
		        >>> x
		        jt.Var([[8 1]
		         [7 6]], dtype=int32)
		        >>> y = jt.randint(0, 10, shape=(2, 3, 2))
		        >>> jt.broadcast(x, y, dims=[1])
		        jt.Var([[[8 1]
		          [8 1]
		          [8 1]],
		         [[7 6]
		          [7 6]
		          [7 6]]], dtype=int32)
		        >>> jt.broadcast_var(x, y, dims=[1])
		        jt.Var([[[8 1]
		          [8 1]
		          [8 1]],
		         [[7 6]
		          [7 6]
		          [7 6]]], dtype=int32)'''
		...
	def reshape(self, shape: Tuple[int])-> Var:		
		'''Document:
		*
		    Returns a tensor with the same data and number of elements as input, but with the specified shape. 
		
		    A single dimension may be -1, in which case it's inferred from the remaining dimensions and the number of elements in input.
		
		    ----------------
		
		    * [in] x:       the input jt.Var
		
		    * [in] shape:   the output shape, an integer array
		
		    ----------------
		
		    Example-1::
		        >>> a = jt.randint(0, 10, shape=(12,))
		        >>> a
		        jt.Var([4 0 8 4 6 3 1 8 1 1 2 2], dtype=int32)
		        >>> jt.reshape(a, (3, 4))
		        jt.Var([[4 0 8 4]
		         [6 3 1 8]
		         [1 1 2 2]], dtype=int32)
		        >>> jt.reshape(a, (-1, 6))
		        jt.Var([[4 0 8 4 6 3]
		         [1 8 1 1 2 2]], dtype=int32)'''
		...
	def reindex_reduce(self, op: str, shape: Tuple[int], indexes: List[str], overflow_conditions: List[str]={}, extras: List[Var]={})-> Var:		
		'''Document:
		*
		    Reindex Reduce Operator is a many-to-one map operator.
		    It performs equivalent Python-pseudo implementation below::
		
		        # input is y, output is x
		        n = len(y.shape)-1
		        m = len(shape)-1
		        k = len(overflow_conditions)-1
		        x = np.zeros(shape, y.dtype)
		        x[:] = initial_value(op)
		        for i0 in range(y.shape[0]): # 1-st loop
		            for i1 in range(y.shape[1]): # 2-nd loop
		                ...... # many loops
		                for in in range(y.shape[n]) # n+1 -th loop
		                    # indexes[i] is a c++ style integer expression consisting of i0,i1,...,in
		                    xi0,xi1,...,xim = indexes[0],indexes[1],...,indexes[m]
		                    if not is_overflow(xi0,xi1,...,xim):
		                        x[xi0,xi1,...,xim] = op(x[xi0,xi1,...,xim], y[i0,i1,...,in])
		
		        # is_overflow is defined as following
		        def is_overflow(xi0,xi1,...,xim):
		            return (
		                xi0 < 0 || xi0 >= shape[0] ||
		                xi1 < 0 || xi1 >= shape[1] ||
		                ......
		                xim < 0 || xim >= shape[m] ||
		
		                # overflow_conditions[i] is a c++ style boolean expression consisting of i0,i1,...,in
		                overflow_conditions[0] ||
		                overflow_conditions[1] ||
		                ......
		                overflow_conditions[k]
		            )
		
		    * [in] y:   A input jittor Var
		    
		    * [in] op:  a string represent the reduce operation type
		    
		    * [in] shape:   the output shape, a integer array
		    
		    * [in] indexes: array of c++ style integer expression, its length should be the same with length of output shape, some buildin variables it can use are::
		    
		             XDIM, xshape0, ..., xshapem, xstride0, ..., xstridem
		             YDIM, yshape0, ..., yshapen, ystride0, ..., ystriden
		             i0, i1, ..., in
		             @e0(...), @e1(...) for extras input index
		             e0p, e1p , ... for extras input pointer
		    
		    * [in] overflow_conditions: array of c++ style boolean expression, it length can be vary. the buildin variables it can use are the same with indexes.
		    
		    * [in] extras:  extra var used for index
		    
		    Example 
		
		    Pooling implemented by reindex operation::
		
		        def pool(x, size, op):
		            N,H,W,C = x.shape
		            h = (H+size-1)//size
		            w = (W+size-1)//size
		            return x.reindex_reduce(op, [N,h,w,C], [
		                "i0", # Nid
		                f"i1/{size}", # Hid
		                f"i2/{size}", # Wid
		                "i3", # Cid
		            ])'''
		...
	def sync(self, device_sync: bool=False, weak_sync: bool=True): ...
	def fetch_sync(self)-> numpy.ndarray:		
		'''Document:
		*
		     * Returns a numpy array copy of the Var.'''
		...
	def numpy(self)-> numpy.ndarray:		
		'''Document:
		*
		     * Returns a numpy array copy of the Var.'''
		...
	def assign(self, v: Var)-> Var:		
		'''Document:
		*
		     * assign the data from another Var.'''
		...
	def update(self, v: Var)-> Var:		
		'''Document:
		*
		     * update parameter and global variable,
		     * different from assign, it will
		     * stop grad between origin var and assigned var, and
		     * will update in the background'''
		...
	def _update(self, v: Var)-> Var:		
		'''Document:
		*
		     * update parameter without set attribute.'''
		...
	def swap(self, v: Var)-> Var:		
		'''Document:
		*
		     * swap the data with another Var.'''
		...
	@overload
	def name(self, s: str)-> Var:		
		'''Document:
		* 
		     * set the name of the Var.'''
		...
	@overload
	def name(self)-> str:		
		'''Document:
		* 
		     * set the name of the Var.'''
		...
	def numel(self)-> int:		
		'''Document:
		* 
		     * return the number of elements in the Var.'''
		...
	def stop_grad(self)-> Var:		
		'''Document:
		* 
		     * disable the gradient calculation for the Var.'''
		...
	def is_stop_grad(self)-> bool:		
		'''Document:
		*
		     * return True if the gradient is stopped.'''
		...
	def detach(self)-> Var:		
		'''Document:
		 detach the grad'''
		...
	def stop_fuse(self)-> Var:		
		'''Document:
		*
		     * stop operator fusion.'''
		...
	def is_stop_fuse(self)-> bool:		
		'''Document:
		*
		     * return True if operator fusion is stopped.'''
		...
	def out_hint(self)-> Var:		
		'''Document:
		*
		     * output hint for training optimization'''
		...
	def start_grad(self)-> Var:		
		'''Document:
		* 
		     * enable the gradient calculation for the Var.'''
		...
	def item(self)-> float | int | bool:		
		'''Document:
		*
		     * returns the Python number if the Var contains only one element.
		     * For other cases, see data().'''
		...
	def share_with(self, other: Var)-> Var: ...
	def debug_msg(self)-> str:		
		'''Document:
		*
		     * print the information of the Var to debug.'''
		...
	def _input(self, i: int)-> Var: ...
	def _add_dependency(self, vars: List[Var])-> Var:		
		'''Document:
		 Add dependency, make var computed after vars'''
		...
	def compile_options(self): ...
	def data(self)-> numpy.ndarray:		
		'''Document:
		*
		     * get a numpy array which shares the data with the Var.'''
		...
	def dtype(self)-> str:		
		'''Document:
		*
		     * return the data type of the Var.'''
		...
	def grad(self)-> int:		
		'''Document:
		 Jittor Var doesn't have this interface, please change your code as below::
		
		    model = Model()
		    optimizer = SGD(model.parameters())
		    ...
		    optimizer.backward(loss)
		    
		    for p in model.parameters():
		        # prev code:
		        # grad = p.grad
		
		        # change to:
		        grad = p.opt_grad(optimizer)'''
		...
	def ndim(self)-> int:		
		'''Document:
		*
		     * return the number of dimensions.'''
		...
	def requires_grad(self)-> bool:		
		'''Document:
		* 
		     * return True if the Var requires gradient calculation.
		     * @see is_stop_grad'''
		...
	def shape(self)-> Tuple[int]:		
		'''Document:
		* 
		     * return the shape of the Var.'''
		...
	def uncertain_shape(self)-> Tuple[int]: ...
	def view(self, x: Var, shape: Tuple[int])-> Var:		
		'''Document:
		*
		    Returns a tensor with the same data and number of elements as input, but with the specified shape. 
		
		    A single dimension may be -1, in which case it's inferred from the remaining dimensions and the number of elements in input.
		
		    ----------------
		
		    * [in] x:       the input jt.Var
		
		    * [in] shape:   the output shape, an integer array
		
		    ----------------
		
		    Example-1::
		        >>> a = jt.randint(0, 10, shape=(12,))
		        >>> a
		        jt.Var([4 0 8 4 6 3 1 8 1 1 2 2], dtype=int32)
		        >>> jt.reshape(a, (3, 4))
		        jt.Var([[4 0 8 4]
		         [6 3 1 8]
		         [1 1 2 2]], dtype=int32)
		        >>> jt.reshape(a, (-1, 6))
		        jt.Var([[4 0 8 4 6 3]
		         [1 8 1 1 2 2]], dtype=int32)'''
		...
	def permute(self, x: Var, axes: Tuple[int]=())-> Var: ...
	def detach_inplace(self)-> Var:		
		'''Document:
		* 
		     * enable the gradient calculation for the Var.'''
		...
	def astype(self, x: Var, op: str)-> Var: ...
	def half(self, x: Var)-> Var:		
		'''Document:
		*
		    Returns a copy of the input var, casted to float16 (half-precision float).
		
		    ----------------
		
		    * [in] x:   the input jt.Var
		
		    ----------------
		    
		    Example-1::
		        >>> x = jt.rand(3) * 10 
		        >>> x
		        jt.Var([4.093273  2.0086648 8.474352 ], dtype=float32)
		        >>> x.half()
		        jt.Var([4.094 2.008 8.48 ], dtype=float16)
		        >>> jt.half(x)
		        jt.Var([4.094 2.008 8.48 ], dtype=float16)
		        >>> x.float16()
		        jt.Var([4.094 2.008 8.48 ], dtype=float16)
		        >>> jt.float16(x)
		        jt.Var([4.094 2.008 8.48 ], dtype=float16)'''
		...
	def expand_as(self, x: Var, y: Var, dims: Tuple[int]=())-> Var:		
		'''Document:
		*
		    Broadcast ``x`` to the same shape as ``y``.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] y:       the reference jt.Var.
		
		    * [in] dims:    specifies the new dimension in the output shape, an integer array.
		
		    ----------------
		
		    .. note::
		      jt.broadcast_var(x, y, dims) is an alias of jt.broadcast(x, y, dims)
		
		    Example-1::
		        >>> x = jt.randint(0, 10, shape=(2, 2))
		        >>> x
		        jt.Var([[8 1]
		         [7 6]], dtype=int32)
		        >>> y = jt.randint(0, 10, shape=(2, 3, 2))
		        >>> jt.broadcast(x, y, dims=[1])
		        jt.Var([[[8 1]
		          [8 1]
		          [8 1]],
		         [[7 6]
		          [7 6]
		          [7 6]]], dtype=int32)
		        >>> jt.broadcast_var(x, y, dims=[1])
		        jt.Var([[[8 1]
		          [8 1]
		          [8 1]],
		         [[7 6]
		          [7 6]
		          [7 6]]], dtype=int32)'''
		...
class Flags:
	'''A set of flags to configure jittor running behaviors'''
	addr2line_path: str
	'''Path of addr2line. Default: ""'''
	amp_level: int
	'''Auto mixed-precision optimization level, 0: not use fp16, 1-3: preserve level, not use fp16 for now; 4: perfer fp16, but some ops use fp32 e.g. sum,exp; 5: simular with 4, and array op will automatically convert to fp16; 6: all ops prefer fp16. Default: 0'''
	amp_reg: int
	'''Auto mixed-precision control registers, bit 0: prefer 32; bit 1: prefer 16; bit 2: keep reduce type; bit 3 keep white list type; bit 4: array like op prefer too. Default: 0'''
	auto_convert_64_to_32: int
	'''auto convert 64bit numpy array into 32bit jittor array. Default: 1'''
	auto_mixed_precision_level: int
	'''Auto mixed-precision optimization level, 0: not use fp16, 1-3: preserve level, not use fp16 for now; 4: perfer fp16, but some ops use fp32 e.g. sum,exp; 5: simular with 4, and array op will automatically convert to fp16; 6: all ops prefer fp16. Default: 0'''
	cache_path: str
	'''Cache path of jittor. Default: ""'''
	cc_flags: str
	'''Flags of C++ compiler. Default: ""'''
	cc_path: str
	'''Path of C++ compiler. Default: ""'''
	cc_type: str
	'''Type of C++ compiler(clang, icc, g++). Default: ""): Type of C++ compiler(clang, icc, g++'''
	check_graph: int
	'''Unify graph sanity check. Default: 0'''
	compile_options: Any
	'''Override the default loop transfrom options. Default: {}'''
	disable_lock: bool
	'''Disable file lock. Default: 0'''
	enable_tuner: int
	'''Enable tuner. Default: 1'''
	exclude_pass: str
	'''Don't run certain pass. Default: ""'''
	extra_gdb_cmd: str
	'''Extra command pass to GDB, seperate by(;) . Default: ""): Extra command pass to GDB, seperate by(;'''
	gdb_attach: int
	'''gdb attach self process. Default: 0'''
	gdb_path: str
	'''Path of GDB. Default: ""'''
	gopt_disable: int
	'''Disable graph optimizer. Default: 0'''
	has_pybt: int
	'''GDB has pybt or not. Default: 0'''
	jit_search_kernel: int
	'''Jit search for the fastest kernel. Default: 0'''
	jit_search_rerun: int
	'''. Default: 10'''
	jit_search_warmup: int
	'''. Default: 2'''
	jittor_path: str
	'''Source path of jittor. Default: ""'''
	l1_cache_size: int
	'''size of level 1 cache (byte). Default: 32768): size of level 1 cache (byte'''
	lazy_execution: int
	'''Default enabled, if disable, use immediately eager execution rather than lazy execution, This flag makes error message and traceback infomation better. But this flag will raise memory consumption and lower the performance. Default: 1'''
	log_file: str
	'''log to file, mpi env will add $OMPI_COMM_WORLD_RANK suffix. Default: ""'''
	log_op_hash: str
	'''Output compiler pass result of certain hash of op. Default: ""'''
	log_silent: int
	'''The log will be completely silent. Default: 0'''
	log_sync: int
	'''Set log printed synchronously. Default: 1'''
	log_v: int
	'''Verbose level of logging. Default: 0'''
	log_vprefix: str
	'''Verbose level of logging prefix. Default: ""'''
	no_fuse: bool
	'''No fusion optimization for all jittor Var creation. Default: 0'''
	no_grad: bool
	'''No grad for all jittor Var creation. Default: 0'''
	node_order: int
	'''id prior. Default: 0'''
	nvcc_flags: str
	'''Flags of CUDA C++ compiler. Default: ""'''
	nvcc_path: str
	'''Path of CUDA C++ compiler. Default: ""'''
	para_opt_level: int
	'''para_opt_level. Default: 3'''
	profile_memory_enable: int
	'''Enable memory profiler. Default: 0'''
	profiler_enable: int
	'''Enable profiler. Default: 0'''
	profiler_hide_relay: int
	'''Profiler hide relayed op. Default: 0'''
	profiler_record_peek: int
	'''Profiler record peek mem bandwidth. Default: 0'''
	profiler_record_shape: int
	'''Profiler record shape for op. Default: 0'''
	profiler_rerun: int
	'''Profiler rerun. Default: 0'''
	profiler_warmup: int
	'''Profiler warmup. Default: 0'''
	python_path: str
	'''Path of python interpreter. Default: ""'''
	reuse_array: int
	'''try reuse np.array memory into jt.array. Default: 0'''
	rewrite_op: int
	'''Rewrite source file of jit operator or not. Default: 1'''
	stat_allocator_total_alloc_byte: int
	'''Total alloc byte. Default: 0'''
	stat_allocator_total_alloc_call: int
	'''Number of alloc function call. Default: 0'''
	stat_allocator_total_free_byte: int
	'''Total alloc byte. Default: 0'''
	stat_allocator_total_free_call: int
	'''Number of alloc function call. Default: 0'''
	th_mode: int
	'''th mode. Default: 0'''
	trace_depth: int
	'''trace depth for GDB. Default: 10'''
	trace_py_var: int
	'''Trace py stack max depth for debug. Default: 0'''
	trace_var_data: int
	'''Trace py stack max depth for debug. Default: 0'''
	try_use_32bit_index: int
	'''If not overflow, try to use 32 bit type as index type. Default: 0'''
	use_acl: int
	'''Use cuda or not. 1 for trying to use cuda, 2 for forcing to use cuda. Default: 0'''
	use_cuda: int
	'''Use cuda or not. 1 for trying to use cuda, 2 for forcing to use cuda. Default: 0'''
	use_device: int
	'''Use cuda or not. 1 for trying to use cuda, 2 for forcing to use cuda. Default: 0'''
	use_nfef_allocator: int
	'''Enable never free exact fit allocator. Default: 0'''
	use_parallel_op_compiler: int
	'''Number of threads that parallel op comiler used, default 16, set this value to 0 will disable parallel op compiler. Default: 16'''
	use_rocm: int
	'''Use cuda or not. 1 for trying to use cuda, 2 for forcing to use cuda. Default: 0'''
	use_sfrl_allocator: int
	'''Enable sfrl allocator. Default: 1'''
	use_stat_allocator: int
	'''Enable stat allocator. Default: 0'''
	use_temp_allocator: int
	'''Enable temp allocator. Default: 1'''
	use_tensorcore: int
	'''use tensor core. Default: 0'''
flags: Flags
'''Jittor running time flags instance'''
