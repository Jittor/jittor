import jittor_core as jittor_core
import jittor_core as core
from . import autograd as autograd, compile_extern as compile_extern, compiler as compiler, dataset as dataset, distributions as distributions, fft as fft, init as init, linalg as linalg, math_util as math_util, misc as misc, nn as nn, numpy2cupy as numpy2cupy, optim as optim, sparse as sparse
from .benchmarking import BenchmarkResult as BenchmarkResult, benchmark as benchmark
from .compat import contrib as contrib
from .compile_extern import cublas as cublas, cudnn as cudnn, cufft as cufft, curand as curand, cusparse as cusparse, mkl_ops as mkl_ops, mpi as mpi, mpi_ops as mpi_ops
from .compiler import LOG as LOG, compile_custom_op as compile_custom_op, compile_custom_ops as compile_custom_ops, has_cuda as has_cuda
from .linalg import einsum as einsum
from .misc.concatenation import cat as cat, concat as concat
from .nn import attention as attention, baddbmm as baddbmm, bmm as bmm, bmm_transpose as bmm_transpose, matmul as matmul
from .nn.functional.softmax import logsumexp as logsumexp
from .nn.functional.tensor import kron as kron, tensordot as tensordot
from .optim import legacy_schedulers as lr_scheduler

__all__ = ['CTCLoss', 'DumpGraphs', 'ExitHooks', 'Finfo', 'Flags', 'Function', 'GradHooker', 'LOG', 'MemInfo', 'Module', 'NanoString', 'NanoVector', 'RingBuffer', 'Var', 'ZipFile', '__version__', 'abs', 'abs_', 'acos', 'acosh', 'add', 'add_', 'all', 'all_', 'all_equal', 'amax', 'amin', 'amp_flags', 'any', 'any_', 'arange', 'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan', 'arctan2', 'arctanh', 'arg_reduce', 'argmax', 'argmin', 'argsort', 'array', 'array64', 'array_', 'asin', 'asinh', 'atan', 'atan2', 'atanh', 'atleast_1d', 'atleast_2d', 'atleast_3d', 'attention', 'attrs', 'auto_parallel', 'autograd', 'baddbmm', 'bernoulli', 'bfloat16', 'bfloat16_finfo', 'binary', 'binary_dtype_infer', 'bitwise_and', 'bitwise_not', 'bitwise_or', 'bitwise_xor', 'block_diag', 'bmm', 'bmm_transpose', 'bool', 'broadcast', 'broadcast_var', 'candidate', 'cartesian_prod', 'cast', 'cat', 'ceil', 'ceil_int', 'chunk', 'clamp', 'clamp_', 'clean', 'clean_graph', 'cleanup', 'clear_trace_data', 'clone', 'code', 'compile_custom_op', 'compile_custom_ops', 'compile_extern', 'compiler', 'concat', 'conj', 'contiguous', 'contrib', 'copy', 'core', 'cos', 'cosh', 'count_nonzero', 'cpu', 'cross', 'ctc_loss', 'cub_cumsum', 'cublas', 'cuda', 'cudnn', 'cufft', 'cummax', 'cummin', 'cumprod', 'cumsum', 'curand', 'current_device', 'cusparse', 'dataset', 'deg2rad', 'detach', 'device_copy', 'dfs_to_numpy', 'diag', 'diagonal', 'digamma', 'dirty_fix_pytorch_runtime_error', 'display_max_memory_info', 'display_memory_info', 'distributions', 'div', 'divide', 'double', 'dtype', 'dump_all_graphs', 'dump_trace_data', 'einsum', 'empty', 'enable_grad', 'equal', 'erf', 'erf_', 'erfinv', 'erfinv_', 'exp', 'expand', 'expm1', 'fetch', 'fetch_sync', 'fft', 'finfo', 'flag_scope', 'flags', 'flatten', 'flip', 'float', 'float16', 'float32', 'float64', 'float_auto', 'floor', 'floor_divide', 'floor_int', 'format', 'from_torch', 'full', 'full_like', 'fuse_transpose', 'fused_adamw', 'gather', 'gc', 'get_device_count', 'get_len', 'get_max_memory_info', 'get_max_memory_treemap', 'get_mem_info', 'get_seed', 'getitem', 'grad', 'grad_hooker', 'grad_optional', 'gradfunctional', 'graph_check', 'greater', 'greater_equal', 'half', 'has_cuda', 'hash', 'histc', 'hooks', 'hypot', 'igamma', 'iinfo', 'in_mpi', 'index', 'index_add', 'index_add_', 'index_fill', 'index_fill_', 'index_select', 'index_var', 'init', 'int', 'int16', 'int32', 'int64', 'int8', 'is_var', 'isfinite', 'isin', 'isinf', 'isnan', 'isneginf', 'isposinf', 'jittor_core', 'jittor_exit', 'jt_init_subprocess', 'knn', 'kron', 'kthvalue', 'left_shift', 'less', 'less_equal', 'lgamma', 'linalg', 'linspace', 'liveness_info', 'load', 'lock_acquire', 'lock_is_held', 'lock_release', 'log', 'log2', 'log_capture_scope', 'logical_and', 'logical_not', 'logical_or', 'logical_xor', 'logsumexp', 'lr_scheduler', 'make_grid', 'make_module', 'masked_fill', 'math_util', 'matmul', 'max', 'maximum', 'mean', 'median', 'meshgrid', 'migrate_all_to_cpu', 'min', 'minimum', 'misc', 'mkl_ops', 'mod', 'mpi', 'mpi_ops', 'mul', 'multinomial', 'multiply', 'multiply_', 'ne', 'negative', 'new', 'new_empty', 'new_full', 'new_ones', 'new_zeros', 'nms', 'nn', 'no_grad', 'nonzero', 'norm', 'normal', 'normalize', 'not_equal', 'number_of_hold_vars', 'number_of_lived_ops', 'number_of_lived_vars', 'numpy2cupy', 'numpy_code', 'numpy_cumprod', 'numpy_cumsum', 'ones', 'ones_like', 'op_compiler', 'ops', 'optim', 'origin_reshape', 'origin_transpose', 'outer', 'peek', 'peek_s', 'permute', 'pow', 'print_trace', 'print_tree', 'prod', 'product', 'profile_mark', 'profile_scope', 'profiler', 'python_pass_wrapper', 'rad2deg', 'rand', 'rand_like', 'randint', 'randint_like', 'randn', 'randn_like', 'random', 'randperm', 'rank', 'reduce', 'reduce_add', 'reduce_bitwise_and', 'reduce_bitwise_or', 'reduce_bitwise_xor', 'reduce_logical_and', 'reduce_logical_or', 'reduce_logical_xor', 'reduce_maximum', 'reduce_minimum', 'reduce_multiply', 'register_hook', 'reindex', 'reindex_reduce', 'reindex_var', 'reinterpret_view', 'repeat', 'repeat_interleave', 'reshape', 'reuse_np_array', 'right_shift', 'roll', 'round', 'round_int', 'rsqrt', 'safe_clip', 'safe_log', 'safepickle', 'safeunpickle', 'save', 'save_image', 'scatter', 'scatter_', 'scatter_add', 'scatter_add_', 'scatter_reduce', 'searchsorted', 'seed', 'set_device', 'set_global_seed', 'set_lock_fd', 'set_seed', 'setitem', 'sigmoid', 'sigmoid_', 'sin', 'single_log_capture', 'single_process_scope', 'sinh', 'size', 'sort', 'sparse', 'split', 'sqr', 'sqrt', 'sqrt_', 'squeeze', 'stack', 'std', 'sub', 'subtract', 'sum', 'sync', 'sync_all', 't', 'tan', 'tanh', 'tape', 'tape_together', 'tensordot', 'ternary', 'ternary_out_hint', 'tests', 'to', 'to_bool', 'to_device', 'to_float', 'to_int', 'tolist', 'topk', 'transpose', 'tril', 'triu', 'type_as', 'uint16', 'uint32', 'uint64', 'uint8', 'unary', 'unbind', 'unique', 'unique_consecutive', 'unsqueeze', 'var', 'view', 'view_as', 'vtos', 'where', 'world_size', 'wrap_var_addr', 'zeros', 'zeros_like']

__version__: str
gradfunctional = autograd
dtype = NanoString

# Names in __all__ with no definition:
#   CTCLoss
#   DumpGraphs
#   ExitHooks
#   Finfo
#   Flags
#   Function
#   GradHooker
#   MemInfo
#   Module
#   NanoString
#   NanoVector
#   RingBuffer
#   Var
#   ZipFile
#   abs
#   abs_
#   acos
#   acosh
#   add
#   add_
#   all
#   all_
#   all_equal
#   amax
#   amin
#   amp_flags
#   any
#   any_
#   arange
#   arccos
#   arccosh
#   arcsin
#   arcsinh
#   arctan
#   arctan2
#   arctanh
#   arg_reduce
#   argmax
#   argmin
#   argsort
#   array
#   array64
#   array_
#   asin
#   asinh
#   atan
#   atan2
#   atanh
#   atleast_1d
#   atleast_2d
#   atleast_3d
#   attrs
#   auto_parallel
#   bernoulli
#   bfloat16
#   bfloat16_finfo
#   binary
#   binary_dtype_infer
#   bitwise_and
#   bitwise_not
#   bitwise_or
#   bitwise_xor
#   block_diag
#   bool
#   broadcast
#   broadcast_var
#   candidate
#   cartesian_prod
#   cast
#   ceil
#   ceil_int
#   chunk
#   clamp
#   clamp_
#   clean
#   clean_graph
#   cleanup
#   clear_trace_data
#   clone
#   code
#   conj
#   contiguous
#   copy
#   cos
#   cosh
#   count_nonzero
#   cpu
#   cross
#   ctc_loss
#   cub_cumsum
#   cuda
#   cummax
#   cummin
#   cumprod
#   cumsum
#   current_device
#   deg2rad
#   detach
#   device_copy
#   dfs_to_numpy
#   diag
#   diagonal
#   digamma
#   dirty_fix_pytorch_runtime_error
#   display_max_memory_info
#   display_memory_info
#   div
#   divide
#   double
#   dump_all_graphs
#   dump_trace_data
#   empty
#   enable_grad
#   equal
#   erf
#   erf_
#   erfinv
#   erfinv_
#   exp
#   expand
#   expm1
#   fetch
#   fetch_sync
#   finfo
#   flag_scope
#   flags
#   flatten
#   flip
#   float
#   float16
#   float32
#   float64
#   float_auto
#   floor
#   floor_divide
#   floor_int
#   format
#   from_torch
#   full
#   full_like
#   fuse_transpose
#   fused_adamw
#   gather
#   gc
#   get_device_count
#   get_len
#   get_max_memory_info
#   get_max_memory_treemap
#   get_mem_info
#   get_seed
#   getitem
#   grad
#   grad_hooker
#   grad_optional
#   graph_check
#   greater
#   greater_equal
#   half
#   hash
#   histc
#   hooks
#   hypot
#   igamma
#   iinfo
#   in_mpi
#   index
#   index_add
#   index_add_
#   index_fill
#   index_fill_
#   index_select
#   index_var
#   int
#   int16
#   int32
#   int64
#   int8
#   is_var
#   isfinite
#   isin
#   isinf
#   isnan
#   isneginf
#   isposinf
#   jittor_exit
#   jt_init_subprocess
#   knn
#   kthvalue
#   left_shift
#   less
#   less_equal
#   lgamma
#   linspace
#   liveness_info
#   load
#   lock_acquire
#   lock_is_held
#   lock_release
#   log
#   log2
#   log_capture_scope
#   logical_and
#   logical_not
#   logical_or
#   logical_xor
#   make_grid
#   make_module
#   masked_fill
#   max
#   maximum
#   mean
#   median
#   meshgrid
#   migrate_all_to_cpu
#   min
#   minimum
#   mod
#   mul
#   multinomial
#   multiply
#   multiply_
#   ne
#   negative
#   new
#   new_empty
#   new_full
#   new_ones
#   new_zeros
#   nms
#   no_grad
#   nonzero
#   norm
#   normal
#   normalize
#   not_equal
#   number_of_hold_vars
#   number_of_lived_ops
#   number_of_lived_vars
#   numpy_code
#   numpy_cumprod
#   numpy_cumsum
#   ones
#   ones_like
#   op_compiler
#   ops
#   origin_reshape
#   origin_transpose
#   outer
#   peek
#   peek_s
#   permute
#   pow
#   print_trace
#   print_tree
#   prod
#   product
#   profile_mark
#   profile_scope
#   profiler
#   python_pass_wrapper
#   rad2deg
#   rand
#   rand_like
#   randint
#   randint_like
#   randn
#   randn_like
#   random
#   randperm
#   rank
#   reduce
#   reduce_add
#   reduce_bitwise_and
#   reduce_bitwise_or
#   reduce_bitwise_xor
#   reduce_logical_and
#   reduce_logical_or
#   reduce_logical_xor
#   reduce_maximum
#   reduce_minimum
#   reduce_multiply
#   register_hook
#   reindex
#   reindex_reduce
#   reindex_var
#   reinterpret_view
#   repeat
#   repeat_interleave
#   reshape
#   reuse_np_array
#   right_shift
#   roll
#   round
#   round_int
#   rsqrt
#   safe_clip
#   safe_log
#   safepickle
#   safeunpickle
#   save
#   save_image
#   scatter
#   scatter_
#   scatter_add
#   scatter_add_
#   scatter_reduce
#   searchsorted
#   seed
#   set_device
#   set_global_seed
#   set_lock_fd
#   set_seed
#   setitem
#   sigmoid
#   sigmoid_
#   sin
#   single_log_capture
#   single_process_scope
#   sinh
#   size
#   sort
#   split
#   sqr
#   sqrt
#   sqrt_
#   squeeze
#   stack
#   std
#   sub
#   subtract
#   sum
#   sync
#   sync_all
#   t
#   tan
#   tanh
#   tape
#   tape_together
#   ternary
#   ternary_out_hint
#   tests
#   to
#   to_bool
#   to_device
#   to_float
#   to_int
#   tolist
#   topk
#   transpose
#   tril
#   triu
#   type_as
#   uint16
#   uint32
#   uint64
#   uint8
#   unary
#   unbind
#   unique
#   unique_consecutive
#   unsqueeze
#   var
#   view
#   view_as
#   vtos
#   where
#   world_size
#   wrap_var_addr
#   zeros
#   zeros_like
from typing import List, Tuple, Callable, overload
import numpy
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
def sub(x: Var, y: Var)-> Var:
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
def mul(x: Var, y: Var)-> Var:
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
def div(x: Var, y: Var)-> Var:
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
@overload
def code(shape: Tuple[int], dtype: str, inputs: List[Var]={}, cpu_src: str="", cpu_grad_src: List[str]={}, cpu_header: str="", cuda_src: str="", cuda_grad_src: List[str]={}, cuda_header: str="", data={})-> Var:
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

	    Example-6::


	        # This example shows how to pass custom data
	        # into code op kernel without kernel recompiling.
	        # In this example, the data {"x":123} canbe vary
	        # and kernel will not recompile.
	        # NOTE: the data type pass into kernel is float64
	        # cast to int if you want

	        a = jt.code([1], "float32", inputs=[],
	            data = {"x":123},
	            cpu_src="""
	                @out0(0) = data["x"];
	            """).sync()
	        assert a.item() == 123

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
def code(shapes: List[Tuple[int]], dtypes: List[str], inputs: List[Var]={}, cpu_src: str="", cpu_grad_src: List[str]={}, cpu_header: str="", cuda_src: str="", cuda_grad_src: List[str]={}, cuda_header: str="", data={})-> Tuple[Var]:
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

	    Example-6::


	        # This example shows how to pass custom data
	        # into code op kernel without kernel recompiling.
	        # In this example, the data {"x":123} canbe vary
	        # and kernel will not recompile.
	        # NOTE: the data type pass into kernel is float64
	        # cast to int if you want

	        a = jt.code([1], "float32", inputs=[],
	            data = {"x":123},
	            cpu_src="""
	                @out0(0) = data["x"];
	            """).sync()
	        assert a.item() == 123

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
def code(inputs: List[Var], outputs: List[Var], cpu_src: str="", cpu_grad_src: List[str]={}, cpu_header: str="", cuda_src: str="", cuda_grad_src: List[str]={}, cuda_header: str="", data={})-> Tuple[Var]:
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

	    Example-6::


	        # This example shows how to pass custom data
	        # into code op kernel without kernel recompiling.
	        # In this example, the data {"x":123} canbe vary
	        # and kernel will not recompile.
	        # NOTE: the data type pass into kernel is float64
	        # cast to int if you want

	        a = jt.code([1], "float32", inputs=[],
	            data = {"x":123},
	            cpu_src="""
	                @out0(0) = data["x"];
	            """).sync()
	        assert a.item() == 123

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
def tape(x: Var)-> Var:
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
def random(shape: Tuple[int], dtype: str="float32", type: str="uniform")-> Var:
 ...
@overload
def where(cond: Var, dtype: str="int64")-> Tuple[Var]:
	'''Document:
	*
	    Where Operator generate index of true condition.

	    * [in] cond:    condition for index generation

	    * [in] dtype:   type of return indexes; int64 like torch, so an index can
	                    still name an element of a tensor with more than 2**31 of
	                    them, and so it survives arithmetic (Jittor promotes by
	                    byte width, so `index * stride` stays in the index's dtype)

	    * [out] out:  return an array of indexes, same length with number of dims of cond

	    Example::

	        jt.where([[0,0,1],[1,0,0]])
	        # return [jt.Var([0 1], dtype=int64), jt.Var([2 0], dtype=int64)]'''
	...
@overload
def where(cond: Var, x: Var, y: Var)-> Var:
	'''Document:
	*
	     * Condition operator, perform cond ? x : y
	     *'''
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

	        print(jt.index([2,2], 0))
	        # output: [[0,0],[1,1]]
	        print(jt.index([2,2], 1))
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

	        print(jt.index([2,2], 0))
	        # output: [[0,0],[1,1]]
	        print(jt.index([2,2], 1))
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
def fuse_transpose(x: Var, axes: Tuple[int]=())-> Var:
 ...
def fused_adamw(parameters: List[Var], moments: List[Var], variances: List[Var], gradients: List[Var], step: Var, lr: float, beta1: float, beta2: float, weight_decay: float, eps: float)-> Tuple[Var]:
 ...
def array_(args: numpy.ndarray)-> Var:
 ...
def array(obj: float | int | numpy.ndarray | Var)-> Var:
 ...
def empty(shape: Tuple[int], dtype: str="float32")-> Var:
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
def bfloat16(x: Var)-> Var:
	'''Document:
	*
	    Returns a copy of the input var, casted to bfloat16 (brain half-precision float).

	    ----------------

	    * [in] x:   the input jt.Var

	    ----------------

	    Example-1::
	        >>> x = jt.rand(3) * 10
	        >>> x
	        jt.Var([4.093273  2.0086648 8.474352 ], dtype=float32)
	        >>> x.bfloat16()
	        jt.Var([4.094 2.008 8.48 ], dtype=bfloat16)
	        >>> jt.bfloat16(x)
	        jt.Var([4.094 2.008 8.48 ], dtype=bfloat16)'''
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
def conj(x: Var)-> Var:
	'''Document:
	*
	    Returns the complex conjugate of each element. For complex64 inputs this
	    negates the imaginary part (a+bi -> a-bi); for real inputs it is a no-op
	    (identity), matching torch.conj / Tensor.conj semantics.

	    * [in] x: the input jt.Var.

	    ----------------

	    Example-1::
	        >>> x = jt.array(np.array([1+2j, 3-4j], dtype="complex64"))
	        >>> x.conj()
	        jt.Var([1.-2.j 3.+4.j], dtype=complex64)'''
	...
def setitem(x: Var, slices: slice, y: Var, op: str="void")-> Var:
 ...
def fetch(inputs: List[Var], func: Callable)-> Var:
 ...
def transpose(x: Var, axes: Tuple[int]=())-> Var:
 ...
def device_copy(x: Var, device: int)-> Var:
	'''Document:
	*
	    Copy a Var onto another CUDA device -- torch's ``tensor.to("cuda:N")``.
	    Device ``-1`` is the internal host-copy path used by ``tensor.cpu()``;
	    the public ``to_device`` wrapper accepts CUDA indices only.

	    The result lives on ``device`` whatever the input's device is, and later
	    ops on it run there. It is differentiable: the gradient is a copy back to
	    the source's device. Without CUDA it is a plain host copy.'''
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
def getitem(x: Var, slices: slice)-> Var:
 ...
@overload
def getitem(x: Var, slices: slice, _: int)-> Tuple[Var]:
 ...
def ternary(cond: Var, x: Var, y: Var)-> Var:
 ...
def reinterpret_view(x: Var, shape: Tuple[int], dtype: str)-> Var:
	'''Document:
	*
	    Returns a tensor that shares the same storage as input but reinterprets its
	    dtype and shape. The total byte size must stay unchanged.'''
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
def safe_clip(x: Var, left: float=-1e300, right: float=1e300)-> Var:
	'''Document:
	* Safe clip value to a range, and keep
	 the gradient pass thought.

	    * [in] x:   input value
	    * [in] left: float64 clip min value.
	    * [in] right: float64 clip max value.'''
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
def copy(x: Var)-> Var:
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
class Var:
	'''Variable that stores multi-dimensional data.'''
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
	def sub(self, y: Var)-> Var:
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
	def mul(self, y: Var)-> Var:
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
	def div(self, y: Var)-> Var:
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
	@overload
	def code(self, outputs: List[Var], cpu_src: str="", cpu_grad_src: List[str]={}, cpu_header: str="", cuda_src: str="", cuda_grad_src: List[str]={}, cuda_header: str="", data={})-> Tuple[Var]:
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

		    Example-6::


		        # This example shows how to pass custom data
		        # into code op kernel without kernel recompiling.
		        # In this example, the data {"x":123} canbe vary
		        # and kernel will not recompile.
		        # NOTE: the data type pass into kernel is float64
		        # cast to int if you want

		        a = jt.code([1], "float32", inputs=[],
		            data = {"x":123},
		            cpu_src="""
		                @out0(0) = data["x"];
		            """).sync()
		        assert a.item() == 123

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
	def tape(self)-> Var: ...
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
	@overload
	def where(self, dtype: str="int64")-> Tuple[Var]:
		'''Document:
		*
		    Where Operator generate index of true condition.

		    * [in] cond:    condition for index generation

		    * [in] dtype:   type of return indexes; int64 like torch, so an index can
		                    still name an element of a tensor with more than 2**31 of
		                    them, and so it survives arithmetic (Jittor promotes by
		                    byte width, so `index * stride` stays in the index's dtype)

		    * [out] out:  return an array of indexes, same length with number of dims of cond

		    Example::

		        jt.where([[0,0,1],[1,0,0]])
		        # return [jt.Var([0 1], dtype=int64), jt.Var([2 0], dtype=int64)]'''
		...
	@overload
	def where(self, x: Var, y: Var)-> Var:
		'''Document:
		*
		     * Condition operator, perform cond ? x : y
		     *'''
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
	def fuse_transpose(self, axes: Tuple[int]=())-> Var: ...
	def fused_adamw(self, moments: List[Var], variances: List[Var], gradients: List[Var], step: Var, lr: float, beta1: float, beta2: float, weight_decay: float, eps: float)-> Tuple[Var]: ...
	def array(self)-> Var: ...
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
	def bfloat16(self)-> Var:
		'''Document:
		*
		    Returns a copy of the input var, casted to bfloat16 (brain half-precision float).

		    ----------------

		    * [in] x:   the input jt.Var

		    ----------------

		    Example-1::
		        >>> x = jt.rand(3) * 10
		        >>> x
		        jt.Var([4.093273  2.0086648 8.474352 ], dtype=float32)
		        >>> x.bfloat16()
		        jt.Var([4.094 2.008 8.48 ], dtype=bfloat16)
		        >>> jt.bfloat16(x)
		        jt.Var([4.094 2.008 8.48 ], dtype=bfloat16)'''
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
	def conj(self)-> Var:
		'''Document:
		*
		    Returns the complex conjugate of each element. For complex64 inputs this
		    negates the imaginary part (a+bi -> a-bi); for real inputs it is a no-op
		    (identity), matching torch.conj / Tensor.conj semantics.

		    * [in] x: the input jt.Var.

		    ----------------

		    Example-1::
		        >>> x = jt.array(np.array([1+2j, 3-4j], dtype="complex64"))
		        >>> x.conj()
		        jt.Var([1.-2.j 3.+4.j], dtype=complex64)'''
		...
	def setitem(self, slices: slice, y: Var, op: str="void")-> Var: ...
	def fetch(self, func: Callable)-> Var: ...
	def transpose(self, axes: Tuple[int]=())-> Var: ...
	def device_copy(self, device: int)-> Var:
		'''Document:
		*
		    Copy a Var onto another CUDA device -- torch's ``tensor.to("cuda:N")``.
		    Device ``-1`` is the internal host-copy path used by ``tensor.cpu()``;
		    the public ``to_device`` wrapper accepts CUDA indices only.

		    The result lives on ``device`` whatever the input's device is, and later
		    ops on it run there. It is differentiable: the gradient is a copy back to
		    the source's device. Without CUDA it is a plain host copy.'''
		...
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
	def getitem(self, slices: slice)-> Var: ...
	@overload
	def getitem(self, slices: slice, _: int)-> Tuple[Var]: ...
	def ternary(self, x: Var, y: Var)-> Var: ...
	def reinterpret_view(self, shape: Tuple[int], dtype: str)-> Var:
		'''Document:
		*
		    Returns a tensor that shares the same storage as input but reinterprets its
		    dtype and shape. The total byte size must stay unchanged.'''
		...
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
	def safe_clip(self, left: float=-1e300, right: float=1e300)-> Var:
		'''Document:
		* Safe clip value to a range, and keep
		 the gradient pass thought.

		    * [in] x:   input value
		    * [in] left: float64 clip min value.
		    * [in] right: float64 clip max value.'''
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
	def copy(self)-> Var: ...
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
	def sync(self, device_sync: bool=False, weak_sync: bool=True)-> Var: ...
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
	def location(self)-> str: ...
	def migrate_to_cpu(self)-> Var: ...
	def migrate_to_gpu(self)-> Var: ...
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
	def _set_first_order_only(self)-> Var: ...
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
	def release_from_holders(self): ...
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
	def dim(self)-> int:
		'''Document:
		*
		     * return the number of dimensions.'''
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
	def check_cascade_setitem(self, out: Var)-> Var:
		'''Document:
		 check a[x][y] = c'''
		...
	def compile_options(self): ...
	def data(self)-> numpy.ndarray:
		'''Document:
		*
		     * get a numpy array which shares the data with the Var.'''
		...
	def device_id(self)-> int:
		'''Document:
		*
		     * The CUDA device index this Var lives on, or will be computed on; -1
		     * when there is no CUDA device. Host residency is a different question
		     * (see ``location``): a Var migrated to host memory keeps the device it
		     * belongs to and goes back to it.'''
		...
	def device_raw_ptr(self)-> int: ...
	def dtype(self)-> str:
		'''Document:
		*
		     * return the data type of the Var.'''
		...
	def flags(self): ...
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
	def id(self)-> int:
		'''Document:
		*
		     * return id of this Var.'''
		...
	def nbytes(self)-> int:
		'''Document:
		*
		     * return the number of bytes of this Var.'''
		...
	def ndim(self)-> int:
		'''Document:
		*
		     * return the number of dimensions.'''
		...
	def raw_ptr(self)-> int: ...
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
	def var_ptr(self)-> int: ...
	def mpi_all_reduce(self, x: Var, op: str="add")-> Var:
		'''Document:
		*

		    Mpi All Reduce Operator uses the operator [op] to reduce variable [x] in all MPI nodes and broadcast to all MPI nodes.

		    Args:

		    * x: variable to be all reduced.
		    * op: 'sum' or 'add' means sum all [x], 'mean' means average all [x]. Default: 'add'.'''
		...
	def mpi_broadcast(self, x: Var, root: int=0)-> Var:
		'''Document:
		*

		    Mpi Broadcast Operator broadcasts variable [x] in [root] MPI nodes to all MPI nodes.

		    Args:

		    * x: variable to be broadcasted.
		    * root: ID of MPI node to be broadcasted. Default: 0.'''
		...
	def mpi_reduce(self, x: Var, op: str="add", root: int=0)-> Var:
		'''Document:
		*

		    Mpi Reduce Operator uses the operator [op] to reduce variable [x] in all MPI nodes and send to the [root] MPI node.

		    Args:

		    * x: variable to be reduced.
		    * op: 'sum' or 'add' means sum all [x], 'mean' means average all [x]. Default: 'add'.
		    * root: ID of MPI node to output. Default: 0.

		    **The output is meaningful only on [root].** Reduce sends the result to one
		    rank; MPI ignores the receive buffer on every other one. This operator
		    still returns a full-size output on all ranks, filled with zeros off root,
		    so that every rank runs the same graph -- a shape or an alias that varied
		    by rank would make the ranks fuse differently, and such a defect surfaces
		    nowhere near its cause. Zero is a deterministic filler, not a value: reading
		    a non-root output is a bug in the caller, and it is a bug that reproduces
		    the same way every time rather than depending on the allocator.'''
		...
	def ne(self, x: Var, y: Var)-> Var:
		'''Document:
		*
		    Returns ``x != y`` element-wise.

		    This operation is equivalent to ``x != y``.

		    ----------------

		    * [in] x: the first input,  a python number or jt.Var.

		    * [in] y: the second input, a python number or jt.Var.'''
		...
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
	'''Auto mixed-precision control registers, bit 0: prefer 32; bit 1: prefer 16; bit 2: keep reduce type; bit 3 keep white list type; bit 4: array like op prefer too; bit 5, reduce16 intermediate not use 32. Default: 0'''
	auto_convert_64_to_32: int
	'''auto convert 64bit numpy array into 32bit jittor array. Default: 1'''
	auto_flush_ops: int
	'''Pipeline graph construction with device execution on CUDA. Once this many operators have been created since the executor last ran, launch everything pending without waiting for the device, so the device computes while Python keeps building the rest of the step. 0 keeps fully lazy execution. Fusion and dead-code elimination still apply within each launched segment; CPU execution is synchronous and never flushes early. Default: 128'''
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
	cpu_mem_limit: int
	'''cpu_mem_limit. Default: -1'''
	device_id: int
	'''The CUDA device new Vars are placed on, torch's current device. Setting it switches the device in place -- cudaSetDevice plus a handle swap in every library wrapper -- and never restarts the process; the other devices stay usable. Reads -1 only when no CUDA device exists. Default: -1'''
	device_mem_limit: int
	'''device_mem_limit. Default: -1'''
	disable_lock: bool
	'''Disable file lock. Default: 0'''
	enable_tuner: int
	'''Enable tuner. Default: 1'''
	exclude_pass: str
	'''Don't run certain pass. Default: ""'''
	exec_called: int
	'''exec sync called. Default: 0'''
	extra_gdb_cmd: str
	'''Extra command pass to GDB, seperate by(;) . Default: ""): Extra command pass to GDB, seperate by(;'''
	float32_matmul_precision: str
	'''Accumulate precision for float32 matmul and convolution: highest (float32), high (tf32), medium (bfloat16). float16/bfloat16 inputs always accumulate in float32. Default: "highest"): Accumulate precision for float32 matmul and convolution: highest (float32), high (tf32), medium (bfloat16'''
	gdb_attach: int
	'''gdb attach self process. Default: 0'''
	gdb_path: str
	'''Path of GDB. Default: ""'''
	gdb_trace_timeout: int
	'''Seconds to wait for the GDB backtrace child before giving up. Zero or a negative value waits forever. Default: 30'''
	gopt_disable: int
	'''Disable graph optimizer. Default: 0'''
	has_pybt: int
	'''GDB has pybt or not. Default: 0'''
	jit_search_kernel: int
	'''Jit search for the fastest kernel. Default: 0'''
	jit_search_max_candidates: int
	'''Upper bound on the number of candidate combinations a tuner may offer to the jit kernel search. Default: 1024'''
	jit_search_rerun: int
	'''. Default: 10'''
	jit_search_timeout: int
	'''Wall-clock budget in seconds for the jit kernel search, 0 means no limit. The search compiles and times one kernel per combination of the tuner's candidates, so the cost is the product of the per-key choice counts. Default: 0'''
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
	missing_grad_error: int
	'''Raise instead of warning when a target of grad receives no gradient at all and is filled with zeros. Default: 0'''
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
	sfrl_large_block_size_device: int
	'''sfrl_large_block_size, larger will reduce memory shard, only affect device. Default: 5242880'''
	stat_allocator_total_alloc_byte: int
	'''Total alloc byte. Default: 0'''
	stat_allocator_total_alloc_call: int
	'''Number of alloc function call. Default: 0'''
	stat_allocator_total_free_byte: int
	'''Total alloc byte. Default: 0'''
	stat_allocator_total_free_call: int
	'''Number of alloc function call. Default: 0'''
	sync_run: int
	'''Enable per-op-sync or not. Default: 1'''
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
	use_corex: int
	'''Use cuda or not. 1 for trying to use cuda, 2 for forcing to use cuda. Default: 0'''
	use_cuda: int
	'''Use cuda or not. 1 for trying to use cuda, 2 for forcing to use cuda. Default: 0'''
	use_cuda_host_allocator: int
	'''use cuda host allocator for cpu memory globally. Default: 1'''
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
	'''Deprecated, use float32_matmul_precision. Raises the float32 accumulate tier for matmul and convolution: 1=high(tf32), 2 and 3=medium(bfloat16). Default: 0): Deprecated, use float32_matmul_precision. Raises the float32 accumulate tier for matmul and convolution: 1=high(tf32), 2 and 3=medium(bfloat16'''
	use_threading: int
	'''Allow to use python threading with jittor. Default: 0'''
flags: Flags
'''Jittor running time flags instance'''

class RuntimeContext:
	@property
	def sync_run(self) -> int: ...
	@property
	def device_id(self) -> int: ...
	@property
	def use_cuda(self) -> int: ...
	@property
	def cpu_mem_limit(self) -> int: ...
	@property
	def device_mem_limit(self) -> int: ...
	@property
	def node_order(self) -> int: ...
	@property
	def lazy_execution(self) -> int: ...
	@property
	def auto_flush_ops(self) -> int: ...
	@property
	def auto_convert_64_to_32(self) -> int: ...
	@property
	def reuse_array(self) -> int: ...
	@property
	def no_grad(self) -> int: ...
	@property
	def amp_reg(self) -> int: ...
	@property
	def float32_matmul_precision(self) -> str: ...
	@property
	def auto_mixed_precision_level(self) -> int: ...
	@property
	def try_use_32bit_index(self) -> int: ...
	@property
	def no_fuse(self) -> int: ...
	@property
	def gopt_disable(self) -> int: ...
	@property
	def enable_tuner(self) -> int: ...
	@property
	def exec_called(self) -> int: ...
	@property
	def use_threading(self) -> int: ...
	@property
	def use_parallel_op_compiler(self) -> int: ...
	@property
	def profile_memory_enable(self) -> int: ...
	@property
	def profiler_warmup(self) -> int: ...
	@property
	def profiler_enable(self) -> int: ...
	@property
	def profiler_rerun(self) -> int: ...
	@property
	def profiler_record_peek(self) -> int: ...
	@property
	def profiler_record_shape(self) -> int: ...
	@property
	def profiler_hide_relay(self) -> int: ...
	@property
	def check_graph(self) -> int: ...
	@property
	def missing_grad_error(self) -> int: ...
	@property
	def disable_lock(self) -> int: ...
	@property
	def rewrite_op(self) -> int: ...
	@property
	def trace_var_data(self) -> int: ...
	@property
	def log_silent(self) -> int: ...
	@property
	def log_sync(self) -> int: ...
	@property
	def log_v(self) -> int: ...
	def snapshot(self) -> dict[str, int]: ...

class RuntimeState:
	@property
	def sync_run(self) -> int: ...
	@property
	def device_id(self) -> int: ...
	@property
	def use_cuda(self) -> int: ...
	@property
	def cpu_mem_limit(self) -> int: ...
	@property
	def device_mem_limit(self) -> int: ...
	@property
	def node_order(self) -> int: ...
	@property
	def lazy_execution(self) -> int: ...
	@property
	def auto_flush_ops(self) -> int: ...
	@property
	def auto_convert_64_to_32(self) -> int: ...
	@property
	def reuse_array(self) -> int: ...
	@property
	def no_grad(self) -> int: ...
	@property
	def amp_reg(self) -> int: ...
	@property
	def float32_matmul_precision(self) -> str: ...
	@property
	def auto_mixed_precision_level(self) -> int: ...
	@property
	def try_use_32bit_index(self) -> int: ...
	@property
	def no_fuse(self) -> int: ...
	@property
	def gopt_disable(self) -> int: ...
	@property
	def enable_tuner(self) -> int: ...
	@property
	def exec_called(self) -> int: ...
	@property
	def use_threading(self) -> int: ...
	@property
	def use_parallel_op_compiler(self) -> int: ...
	@property
	def profile_memory_enable(self) -> int: ...
	@property
	def profiler_warmup(self) -> int: ...
	@property
	def profiler_enable(self) -> int: ...
	@property
	def profiler_rerun(self) -> int: ...
	@property
	def profiler_record_peek(self) -> int: ...
	@property
	def profiler_record_shape(self) -> int: ...
	@property
	def profiler_hide_relay(self) -> int: ...
	@property
	def check_graph(self) -> int: ...
	@property
	def missing_grad_error(self) -> int: ...
	@property
	def disable_lock(self) -> int: ...
	@property
	def rewrite_op(self) -> int: ...
	@property
	def trace_var_data(self) -> int: ...
	@property
	def log_silent(self) -> int: ...
	@property
	def log_sync(self) -> int: ...
	@property
	def log_v(self) -> int: ...
	@property
	def context(self) -> RuntimeContext: ...

runtime: RuntimeState
'''Read-only view of execution state owned by the native runtime.'''

# Public names whose precise type is not inferred yet.
CTCLoss: Any
DumpGraphs: Any
ExitHooks: Any
Finfo: Any
Function: Any
GradHooker: Any
MemInfo: Any
Module: Any
NanoString: Any
NanoVector: Any
RingBuffer: Any
ZipFile: Any
abs_: Any
add_: Any
all: Any
all_equal: Any
amax: Any
amin: Any
amp_flags: Any
any: Any
arange: Any
arctan2: Any
argmax: Any
argmin: Any
array64: Any
atan2: Any
atleast_1d: Any
atleast_2d: Any
atleast_3d: Any
attrs: Any
auto_parallel: Any
bernoulli: Any
bfloat16_finfo: Any
binary_dtype_infer: Any
block_diag: Any
bool: Any
cartesian_prod: Any
chunk: Any
clamp: Any
clamp_: Any
clean: Any
clean_graph: Any
cleanup: Any
clear_trace_data: Any
contiguous: Any
core: Any
count_nonzero: Any
cpu: Any
cross: Any
ctc_loss: Any
cub_cumsum: Any
cuda: Any
cummax: Any
cummin: Any
cumprod: Any
cumsum: Any
current_device: Any
deg2rad: Any
detach: Any
dfs_to_numpy: Any
diag: Any
diagonal: Any
digamma: Any
dirty_fix_pytorch_runtime_error: Any
display_max_memory_info: Any
display_memory_info: Any
double: Any
dump_all_graphs: Any
dump_trace_data: Any
enable_grad: Any
erf_: Any
erfinv_: Any
expand: Any
expm1: Any
fetch_sync: Any
finfo: Any
flag_scope: Any
flatten: Any
flip: Any
float: Any
float_auto: Any
format: Any
from_torch: Any
full: Any
full_like: Any
gather: Any
gc: Any
get_device_count: Any
get_len: Any
get_max_memory_info: Any
get_max_memory_treemap: Any
get_mem_info: Any
get_seed: Any
grad: Any
grad_hooker: Any
grad_optional: Any
graph_check: Any
half: Any
hash: Any
histc: Any
hooks: Any
hypot: Any
igamma: Any
iinfo: Any
in_mpi: Any
index_add: Any
index_add_: Any
index_fill: Any
index_fill_: Any
index_select: Any
int: Any
is_var: Any
isfinite: Any
isin: Any
isinf: Any
isnan: Any
isneginf: Any
isposinf: Any
jittor_core: Any
jittor_exit: Any
jt_init_subprocess: Any
knn: Any
kthvalue: Any
lgamma: Any
linspace: Any
liveness_info: Any
load: Any
lock_acquire: Any
lock_is_held: Any
lock_release: Any
log2: Any
log_capture_scope: Any
make_grid: Any
make_module: Any
masked_fill: Any
median: Any
meshgrid: Any
migrate_all_to_cpu: Any
multinomial: Any
multiply_: Any
ne: Any
new: Any
new_empty: Any
new_full: Any
new_ones: Any
new_zeros: Any
nms: Any
no_grad: Any
nonzero: Any
norm: Any
normal: Any
normalize: Any
number_of_hold_vars: Any
number_of_lived_ops: Any
number_of_lived_vars: Any
numpy_cumprod: Any
numpy_cumsum: Any
ones: Any
ones_like: Any
op_compiler: Any
ops: Any
origin_reshape: Any
origin_transpose: Any
outer: Any
peek: Any
peek_s: Any
permute: Any
print_trace: Any
print_tree: Any
profile_mark: Any
profile_scope: Any
profiler: Any
python_pass_wrapper: Any
rad2deg: Any
rand: Any
rand_like: Any
randint: Any
randint_like: Any
randn: Any
randn_like: Any
randperm: Any
rank: Any
register_hook: Any
repeat: Any
repeat_interleave: Any
reuse_np_array: Any
roll: Any
rsqrt: Any
safe_log: Any
safepickle: Any
safeunpickle: Any
save: Any
save_image: Any
scatter: Any
scatter_: Any
scatter_add: Any
scatter_add_: Any
scatter_reduce: Any
searchsorted: Any
seed: Any
set_device: Any
set_global_seed: Any
set_lock_fd: Any
set_seed: Any
sigmoid_: Any
single_log_capture: Any
single_process_scope: Any
size: Any
sort: Any
split: Any
sqr: Any
sqrt_: Any
squeeze: Any
stack: Any
std: Any
sync: Any
sync_all: Any
t: Any
tape_together: Any
ternary_out_hint: Any
tests: Any
to: Any
to_bool: Any
to_device: Any
to_float: Any
to_int: Any
tolist: Any
topk: Any
tril: Any
triu: Any
type_as: Any
unbind: Any
unique: Any
unique_consecutive: Any
unsqueeze: Any
var: Any
view: Any
view_as: Any
vtos: Any
world_size: Any
wrap_var_addr: Any
zeros: Any
zeros_like: Any
