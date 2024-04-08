import os
os.environ["FIX_TORCH_ERROR"] = "0"

import jittor as jt
from jittor import *
from typing import Tuple

org_int = int = type(1)
org_float = float = type(1.0)
org_bool = bool = type(True)

import jtorch.compiler

import jtorch_core
from jtorch_core import *

device.__reduce__ = lambda self: (device, (self.type,))
device.__module__ = "jtorch"
jt.jittor_core.device = device

def handle_dtype(args, kw, dtype):
    def convert(x):
        if isinstance(x, jt.Var):
            return x.cast(dtype)
        return x
    if dtype is not None:
        if args is not None:
            if isinstance(args, (tuple,list)):
                args = [ convert(a) for a in args ]
            else:
                args = convert(x)
        if kw is not None:
            kw = { k:convert(v) for k,v in kw.items() }
    return args, kw

def get_args_names(func):
    import inspect
    spec = inspect.getfullargspec(func)
    return spec[0] + spec[4]

def wrapper(func):
    has_dtype = False
    if hasattr(func, "__code__"):
        has_dtype = "dtype" in get_args_names(func)
    def inner(*args, **kw):
        requires_grad = None
        dtype = None
        if "requires_grad" in kw:
            requires_grad = kw["requires_grad"]
            del kw["requires_grad"]
        if not has_dtype and "dtype" in kw:
            dtype = kw["dtype"]
            del kw["dtype"]
        if "device" in kw:
            del kw["device"]
        if 'pin_memory' in kw:
            del kw['pin_memory']
        args, kw = handle_dtype(args, kw, dtype)
        ret = func(*args, **kw)
        if isinstance(ret, jt.Var):
            if requires_grad is not None:
                ret.requires_grad = requires_grad
            if dtype is not None:
                ret.astype(dtype)
        return ret
    return inner
        

import inspect
_wrapper_keys = set(["shape", "start", "size"])
_wrapper_keys.add("x")
for k,v in list(globals().items()):
    if callable(v) and not isinstance(v, type):
        try:
            spec = inspect.getfullargspec(v)
            args_name = spec[0]
            if len(args_name) and args_name[0] in _wrapper_keys:
                globals()[k] = wrapper(v)
            elif spec.varargs in _wrapper_keys:
                globals()[k] = wrapper(v)
        except:
            pass

def empty(*size, dtype=jt.float32, device=None, requires_grad=False):
    if len(size) == 1 and not isinstance(size[0], org_int):
        size = size[0]
    return jt.empty(size, dtype)

Tensor = Var

Tensor.backward = lambda x: jtorch_core.backward(x)
Tensor.grad = property(grad_get, grad_set, grad_del)
Tensor.retains_grad = property(retain_grad_get, retain_grad_set)
def retain_grad(x:Tensor, value:bool=True):
    x.retains_grad = value
    return value
Tensor.retain_grad = retain_grad

Tensor.dim = lambda self: self.ndim
Tensor.ndimension = lambda self: self.ndim
Tensor.nelement = lambda self: self.numel()
Tensor.cuda = lambda self: self
def device_get(x:Tensor):
    return device("cpu") if not jt.has_cuda or not jt.flags.use_cuda else device("cuda")
Tensor.device = property(device_get)

def argmax(x: Var, dim=None, keepdim: bool = False):
    return jt.argmax(x, dim, keepdim)[0]
Tensor.argmax = argmax

def tensor_type(x: Var, dtype=None, **kwargs):
    if dtype:
        return x.astype(dtype)
    else:
        return x.dtype
Tensor.type = tensor_type

def is_floating_point(x: Var):
    return "float" in str(x.dtype)
Tensor.is_floating_point = is_floating_point

from . import autograd
from .autograd import *

def tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False):
    if isinstance(data,list):
        data_list = []
        check = True
        for p in data:
            if isinstance(p, Tensor) and p.numel()==1:
                data_list.append(p.item())
            elif isinstance(p, (org_int,org_float)):
                data_list.append(p)
            else:
                check = False
                break
        if check:
            data = data_list
    return wrapper(array)(data, dtype=dtype, device=device, requires_grad=requires_grad, pin_memory=pin_memory)

# tensor = wrapper(array)
from_numpy = wrapper(array)
strided = None

def mod_zero_grad(self):
    for p in self.parameters():
        p.grad = None
Module.zero_grad = mod_zero_grad

class ModuleMisc:
    def parameters(self):
        return iter(super().parameters())

    def load_state_dict(self, state_dict, strict=False):
        return super().load_state_dict(state_dict)

    def to(self, device=None,dtype=None):
        ''' do nothing but return its self'''
        return self
    def register_parameter(self,name,data):
        self.name = data

    def buffers(self):
        for _, buf in self.named_buffers():
            yield buf

        
def make_module(cls):
    class TMod(ModuleMisc, cls):
        def __init__(self, *args, **kw):
            dtype = None
            if "dtype" in kw:
                dtype = kw["dtype"]
                del kw["dtype"]
            self._dtype = dtype
            with jt.flag_scope(th_mode=0):
                if "device" in kw:
                    del kw["device"]
                super().__init__(*args, **kw)
            for k,v in self.__dict__.items():
                if not k.startswith("_") and isinstance(v, Var) \
                    and v.requires_grad:
                    v.retain_grad()
                if dtype is not None and isinstance(v, Var):
                    v.assign(v.cast(dtype))
        def __call__(self, *args, **kw):
            args, kw = handle_dtype(args, kw, self._dtype)
            # if forward is override by user, call forward
            if self.__class__.forward is not TMod.forward:
                return self.forward(*args, **kw)
            return self.execute(*args, **kw)
        def forward(self, *args, **kw):
            args, kw = handle_dtype(args, kw, self._dtype)
            return self.execute(*args, **kw)
        
        @property
        def training(self):
            if not hasattr(self, "is_train"):
                self.is_train = True
            return self.is_train
        @training.setter
        def training(self, value):
            self.is_train = value

    TMod.__name__ = cls.__name__
    return TMod

import jtorch.cuda
import jtorch.nn
from jtorch.nn import Module, Parameter
import jtorch.optim

from jtorch.utils.dtype import Dtype, get_string_dtype

def frombuffer(buffer: bytearray, 
              *, 
              dtype: Dtype, 
              count: int = -1, 
              offset: int = 0, 
              requires_grad: bool = True) -> Tensor:
    dtype = get_string_dtype(dtype)
    tensor = jt.array(np.frombuffer(buffer, dtype, count=count, offset=offset))
    if requires_grad and tensor.dtype.is_float():
        tensor.requires_grad = True
    return tensor

def conflict_wrapper(origin_func, new_func):
    def wrapper(*args, **kw):
        if jt.flags.th_mode:
            return new_func(*args, **kw)
        else:
            return origin_func(*args, **kw)
    return wrapper

def min(*args, **kw):
    dim = None
    if len(args) >= 2 and isinstance(args[1], org_int):
        dim = args[1]
    elif "dim" in kw and isinstance(kw["dim"], org_int):
        dim = kw["dim"]
    if dim is not None:
        k, v = jt.argmin(*args, **kw)
        return v, k
    elif len(args) == 2 and isinstance(args[1], jt.Var):
        return jt.minimum(args[0], args[1])
    else:
        return jt.min(*args, **kw)
Tensor.min = conflict_wrapper(jt.min, min)

def max(*args, **kw):
    dim = None
    if "dim" in kw:
        x = kw["dim"]
    if len(args) >= 2 and isinstance(args[1], org_int):
        dim = args[1]
    elif "dim" in kw and isinstance(kw["dim"], org_int):
        dim = kw["dim"]
    if dim is not None:
        k, v = jt.argmax(*args, **kw)
        return v, k
    elif len(args) == 2 and isinstance(args[1], jt.Var):
        return jt.maximum(args[0], args[1])
    else:
        return jt.max(*args, **kw)
Tensor.max = conflict_wrapper(jt.max, max)

def argsort(*args, **kw):
    k, v = jt.argsort(*args, **kw)
    return k
Tensor.argsort = conflict_wrapper(jt.argsort, argsort)

LongTensor = jt.int64
FloatTensor = jt.float
HalfTensor = jt.float16
BoolTensor = jt.bool
IntTensor = jt.int32

class JDType:
    def __init__(self, func, str):
        self.func = func
        self.str = str
        self.__name__ = str.split(".")[-1]
    def __call__(self, *args, **kw):
        return self.func(*args, **kw)
    def __str__(self):
        return self.str
    def is_floating_point(self):
        return "float" in str(self.str)

int8 = JDType(jt.int8, "torch.int8")
int16 = JDType(jt.int16, "torch.int16")
int = int32 = JDType(jt.int32, "torch.int32")
long = int64 = JDType(jt.int64, "torch.int64")

half = float16 = JDType(jt.float16, "torch.float16")
float = float32 = JDType(jt.float32, "torch.float32")
double = float64 = JDType(jt.float64, "torch.float64")
bfloat16 = "bfloat16" # TODO
complex64 = "complex64" # TODO
complex128 = "complex128" # TODO
def get_JDtype(dtype):
    if dtype=='float32' or dtype == jt.float32:
        return float32
    elif dtype=='float64' or dtype == jt.float64:
        return float64
    elif dtype=='float16' or dtype == jt.float16:
        return float16
    elif dtype=='int32' or dtype == jt.int32:
        return int32
    elif dtype=='int64' or dtype == jt.int64:
        return int64
    elif dtype=='int16' or dtype == jt.int16:
        return int16
    elif dtype=='int8' or dtype == jt.int8:
        return int8
    else:
        raise Exception("dtype {} not supported".format(dtype))

def load(path,**kwargs):
    def _to_jittor(data):
        if isinstance(data,dict):
            return {k:_to_jittor(d) for k,d in data.items()}
        if isinstance(data,list):
            return [_to_jittor(d) for d in data]
        if isinstance(data,np.ndarray):
            return jt.array(data)
        return data
    data = jt.load(path)
    
    return _to_jittor(data)

def is_tensor(x):
    return isinstance(x, Tensor)

manual_seed = jt.set_global_seed
jt.flags.amp_level = 3
Size = jt.NanoVector

class Generator:
    def __init__(self,*args,**kw) -> None:
        self.seed = None
    def manual_seed(self,seed):
        self.seed = seed



from . import fx


_default_type = "float32"

def get_default_dtype():
    return _default_type
def set_default_dtype(dtype):
    global _default_type
    _default_type = dtype

dtype = JDType

def div(x,y,rounding_mode="floor"):
    assert rounding_mode == "floor"
    z = (x / y)
    if rounding_mode == "floor":
        z = z.floor()    
    if x.dtype == "int32" and (isinstance(y,org_int) or y.dtype == "int32"):
        z = z.int32()
    return z


def randn(*args,**kw):
    wrap_randn = wrapper(jt.randn)
    generator = kw.get('generator',None)
    kw.pop('generator',None)
    if 'layout' in kw:
        del kw['layout']
    if generator is not None and generator.seed is not None:
        jt.set_global_seed(generator.seed)
    return wrap_randn(*args,**kw)

def rand(*args,**kw):
    print("rand")
    wrap_rand = wrapper(jt.rand)
    generator = kw.get('generator',None)
    kw.pop('generator',None)
    if 'layout' in kw:
        del kw['layout']
    if generator is not None and generator.seed is not None:
        jt.set_global_seed(generator.seed)
    return wrap_rand(*args,**kw)



def set_default_tensor_type(t: type or str):
    if isinstance(t, str):
        info = t.split(".")
        if len(info) == 3 and info[1] == 'cuda':
            jt.flags.use_cuda = 1
    #TODO: type


def clamp(x, min=None, max=None):
    return jt.clamp(x, min, max)


def to(x,*args,**kw):
    device = None
    if len(args) == 1:
        device = args[0]
        if isinstance(device, jt.NanoString) or callable(device):
            return jt.to(x,*args,**kw)
        if 'cpu' in str(device):
            args = []
    device = kw.get("device",None)
    if 'cpu' in str(device):
        kw.pop('device',None)
        print("to cpu")
        # print(kw)
    return jt.to(x,*args,**kw)
Tensor.to = conflict_wrapper(jt.to, to)

mm = wrapper(jt.matmul)

def _data_get(x):
    return x

def _data_set(x, value):
    x.assign(value)
    
Tensor.data = property(_data_get, _data_set)
Tensor.layout = None