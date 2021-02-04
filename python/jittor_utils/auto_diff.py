import os
from pathlib import Path
import pickle
import numpy as np
import jittor_utils
from jittor_utils import LOG
import sys

jittor_utils.try_import_jit_utils_core()


has_error = 0

def convert(data):
    if isinstance(data, tuple):
        return tuple( convert(v) for v in data )
    if isinstance(data, list):
        return [ convert(v) for v in data ]
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, dict):
        return {k:convert(data[k]) for k in data}
    if hasattr(data, "numpy"):
        if "Var" in data.__class__.__name__:
            return data.numpy()
        else:
            return data.detach().cpu().numpy()
    return data

rand_hooked = False

def hook_pt_rand(*shape, device=None):
    import torch
    if isinstance(shape, tuple) and len(shape)==1 and isinstance(shape[0], torch.Size):
        shape = tuple(shape[0])
    np.random.seed(0)
    res = torch.from_numpy(np.random.rand(*tuple(shape)).astype("float32"))
    if device is not None:
        return res.to(device)
    return res

def hook_pt_randn(*shape, device=None):
    import torch
    if isinstance(shape, tuple) and len(shape)==1 and isinstance(shape[0], torch.Size):
        shape = tuple(shape[0])
    np.random.seed(0)
    res = torch.from_numpy(np.random.randn(*tuple(shape)).astype("float32"))
    if device is not None:
        return res.to(device)
    return res

def hook_pt_normal(mean, std):
    import torch
    shape = tuple(mean.shape)

    np.random.seed(0)
    return torch.from_numpy(np.random.normal(size=shape).astype("float32")).to(std.device) * std + mean

def hook_jt_rand(shape, dtype="float32", rtype="uniform"):
    import jittor
    np.random.seed(0)
    if rtype == "normal":
        return jittor.array(np.random.normal(size=shape).astype(str(dtype)))
    return jittor.array(np.random.rand(*shape).astype(str(dtype)))

def hook_rand():
    global rand_hooked
    if rand_hooked: return
    rand_hooked = True
    np.random.seed(0)
    if "torch" in sys.modules:
        LOG.i("Hook torch.rand")
        torch = sys.modules["torch"]
        torch.rand = hook_pt_rand
        torch.normal = hook_pt_normal
        torch.randn = hook_pt_randn
        torch.manual_seed(0)
    if "jittor" in sys.modules:
        jittor = sys.modules["jittor"]
        LOG.i("Hook jittor.random")
        jittor.random = hook_jt_rand
        jittor.seed(0)


class Hook:
    def __init__(self, base_name, rtol=5e-2, atol=1e-3):
        if os.environ.get("use_auto_diff", '1') == '0':
            return
        hook_rand()
        self.rid = 0
        self.base_name = base_name
        self.base_path = os.path.join(str(Path.home()), ".cache", "jittor", "auto_diff", base_name)
        os.makedirs(self.base_path, exist_ok=True)
        self.rtol = rtol
        self.atol = atol
        LOG.i("Use cache path:", self.base_path)
        LOG.i(f"rtol:{rtol} atol:{atol}")

    def check_array(self, name, a, b):
        rtol = self.rtol
        atol = self.atol
        global has_error
        err = np.abs(a-b)
        tol = atol + rtol * np.abs(b)
        is_error = np.logical_or( err > tol, (a>=-1e-5)!=(b>=-1e-5))
        index = np.where(is_error)
        assert len(index)>0
        if len(index[0]) == 0:
            return

        has_error += 1
        LOG.w(f"Ndarray <{name}> not match, shape:{a.shape}")
        i = tuple( i[0] for i in index )
        err_rate = is_error.mean()
        LOG.w(f"error index at [{i}], a({a[i]}) b({b[i]}) err({err[i]}) > tol({tol[i]}), err_rate:{err_rate*100:.3f}% amean({a.mean()}) bmean({b.mean()}) astd({a.std()}) bstd({b.std()}) ")
        if err_rate > 0.01:
            LOG.e("!"*10+"Very HIGH err rate"+"!"*10)

    def check(self, name, pre_data, data):
        global has_error
        if pre_data is None and isinstance(data, np.ndarray):
            if (data==0).all():
                LOG.i(f"name {name} is None")
            else:
                LOG.e(f"name {name} is non-zero")
            return
        if type(pre_data) != type(data):
            LOG.e(f"type not match, {pre_data.__class__.__name__}!={data.__class__.__name__}, name: {name}")
            has_error += 1
            return
        if isinstance(pre_data, (list, tuple)):
            if len(pre_data) != len(data):
                has_error += 1
                LOG.e(f"Name <{name}> len not match, {len(pre_data)} != {len(data)}")
            n = max(len(pre_data), len(data))
            for i in range(n):
                a = pre_data[i] if i<len(pre_data) else "None"
                b = data[i] if i<len(data) else "None"
                self.check(name+f".{i}", a, b)
        elif isinstance(pre_data, np.ndarray):
            if pre_data.shape != data.shape: 
                has_error += 1
                LOG.e(f"Ndarray shape <{name}> not match {pre_data.shape} != {data.shape}")
                return
            self.check_array(name, pre_data, data)
        elif isinstance(pre_data, dict):
            if len(pre_data) != len(data): 
                has_error += 1
                LOG.w(f"Dict Name <{name}> len not match, {len(pre_data)} != {len(data)}")
            for k in pre_data:
                pv = pre_data[k]
                if k not in data:
                    has_error += 1
                    msg = f"Key <{k}> not in data, Name <{name}>"
                    if isinstance(pv, np.ndarray):
                        LOG.e(msg)
                    else:
                        LOG.w(msg)
                    continue
                self.check(name+f".{k}", pre_data[k], data[k])
        else:
            if pre_data != data: 
                has_error += 1
                LOG.e(f"Type: {type(pre_data).__name__} Name <{name}> not match {pre_data} != {data}")

    def record(self, name, data, ex_name=""):
        if os.environ.get("use_auto_diff", '1') == '0':
            return
        rid = self.rid
        self.rid += 1
        fpath = os.path.join(self.base_path, f"{rid}.pkl")
        data = convert(data)
        if os.path.isfile(fpath):
            with open(fpath, 'rb') as f:
                pre_name, pre_data = pickle.load(f)
            if pre_name != name: 
                global has_error
                has_error += 1
                LOG.e(f"The {rid} result name not match, {pre_name} != {name}")
                self.rid -= 1
                return
            LOG.i(f"check {rid}:<{ex_name}{name}> ...")
            self.check(ex_name+name, pre_data, data)
        else:
            with open(fpath, 'wb') as f:
                pickle.dump((name, data), f)
            LOG.i(f"save {rid}:<{name}> ok")

    def record_params(self, parameters_dict):
        if os.environ.get("use_auto_diff", '1') == '0':
            return
        rid = self.rid
        self.rid += 1
        global has_error
        pps = {}
        for k, v in parameters_dict.items():
            if k.endswith("num_batches_tracked"):
                continue
            pps[k] = v
        ps = { name:convert(param) for name, param in pps.items() }
        fpath = os.path.join(self.base_path, f"{rid}-params.pkl")
        if os.path.isfile(fpath):
            with open(fpath, 'rb') as f:
                prev_ps = pickle.load(f)
            if len(prev_ps) != len(ps):
                has_error += 1
                LOG.e(f"Params len not match {len(prev_ps)} != {len(ps)}")
            for k in ps:
                a = ps[k]
                if k not in prev_ps:
                    has_error += 1
                    LOG.e(f"prev param <{k}> not found.")
                    continue
                b = prev_ps[k]
                if a.shape != b.shape:
                    has_error += 1
                    LOG.e(f"Params <{k}> shape not match {a.shape} != {b.shape}")
                    continue
                std_a, mean_a = a.std(), a.mean()
                std_b, mean_b = b.std(), b.mean()
                n = a.size
                # law of large number
                std_mean_a = (std_a+std_b)/2 / np.sqrt(n) + 1e-6
                std_std_a = (std_a+std_b)/2 / np.sqrt((n-1)/2) + 1e-6
                x = 4
                if np.abs(mean_a - mean_b) > x * std_mean_a:
                    has_error += 1
                    LOG.e(f"param mean not match, mean_a:{mean_a}, mean_b:{mean_b}, acceptable range:({mean_a - x * std_mean_a}, {mean_a + x * std_mean_a}) name:{k} shape:{a.shape}")
                elif np.abs(std_a - std_b) > x * std_std_a:
                    has_error += 1
                    LOG.e(f"param std not match, std_a:{std_a}, std_b:{std_b}, acceptable range:({std_a - x * std_std_a}, {std_a + x * std_std_a}) name:{k} shape:{a.shape}")
                else:
                    LOG.i(f"check param ok: <{k}>  shape:{a.shape}")
                var = pps[k]
                if hasattr(var, "copy_"):
                    import torch
                    var.data.copy_(torch.from_numpy(b))
                else:
                    var.assign(b)
        else:
            with open(fpath, 'wb') as f:
                pickle.dump(ps, f)
            LOG.i(f"save params ok")

    def hook_function(self, func):
        name = func.__name__
        def new_func(*args, **kw):
            ret = func(*args, **kw)
            self.record(name+".args", args)
            self.record(name+".kw", kw)
            self.record(name+".ret", ret)
            return ret
        return new_func


    def hook_module(self, mod, mod_name=""):
        if os.environ.get("use_auto_diff", '1') == '0':
            return
        if mod_name != "":
            mod_name = "<" + mod_name + ">"
        def forward_hook(self2, input, output, kw=None):
            ex_name = '[' + self2.__class__.__name__ + ']' 
            if "relu" not in self2.__class__.__name__.lower():
                # not test relu, because input may be inplaced
                self.record(self2.__ad_mod_name__+".input", input, ex_name)
            self.record(self2.__ad_mod_name__+".output", output, ex_name)
            if kw is not None:
                self.record(self2.__ad_mod_name__+".kw", kw, ex_name)

        names = []
        for name, module in mod.named_modules():
            name = mod_name + name
            module.__ad_mod_name__ = name
            names.append(name)
            module.register_forward_hook(forward_hook)
            mod_class_name = module.__class__.__name__.lower()
            # make dropout in eval mod and record dropout.p
            if "dropout" in mod_class_name:
                self.record(name+'.p', module.p, "["+mod_class_name+"]")
                module.eval()
        ps = { mod_name+k:v for k, v in mod.state_dict().items() }
        self.record_params(ps)
        self.record("module names", names)

    def hook_optimizer(self, opt, opt_name=""):
        '''
            net = Model()
            opt = optim.SGD(net.parameters(), 0.1)
            hook.hook_optimizer(opt)
        '''
        if os.environ.get("use_auto_diff", '1') == '0':
            return
        origin_step = opt.step
        ex_name = '['+opt.__class__.__name__+']'
        def step_hook(*args, **kw):
            origin_step(*args, **kw)
            self.record(opt_name+".default", opt.defaults, ex_name)
            gid = 0
            n_params = 0
            for pg in opt.param_groups:
                for p in pg["params"]:
                    if hasattr(p, "is_stop_grad"):
                        if p.is_stop_grad():
                            continue
                        n_params += 1
                    else:
                        n_params += 1

            self.record(opt_name+".n_params", n_params, ex_name)

            for pg in opt.param_groups:
                for i, p in reversed(list(enumerate(pg["params"]))):
                    if hasattr(p, "is_stop_grad"):
                        if p.is_stop_grad():
                            continue
                        self.record(f"{opt_name}.grads[{gid}]", pg["grads"][i], "["+p.name()+"]")
                        self.record(f"{opt_name}.params[{gid}]", p, "["+p.name()+"]")
                        gid += 1
                    else:
                        self.record(f"{opt_name}.grads[{gid}]", p.grad)
                        self.record(f"{opt_name}.params[{gid}]", p)
                        gid += 1
        opt.step = step_hook