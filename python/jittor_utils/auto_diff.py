import os
from pathlib import Path
import pickle
import numpy as np
import jittor_utils
from jittor_utils import LOG

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
        data = data.detach()
        if hasattr(data,'cpu'):
            data = data.cpu()
        return data.numpy()
    return data

class Hook:
    def __init__(self, base_name, rtol=5e-2, atol=1e-3):
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
        LOG.e(f"Ndarray <{name}> not match, shape:{a.shape}")
        i = tuple( i[0] for i in index )
        err_rate = is_error.mean()
        LOG.e(f"error index at [{i}], a({a[i]}) b({b[i]}) err({err[i]}) > tol({tol[i]}), err_rate:{err_rate*100:.3f}%")
        if err_rate > 0.01:
            LOG.e("!"*10+"Very HIGH err rate"+"!"*10)

    def check(self, name, pre_data, data):
        global has_error
        assert type(pre_data) == type(data)
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
                LOG.e(f"Ndarray shape <{name}> not match")
                return
            self.check_array(name, pre_data, data)
        elif isinstance(pre_data, dict):
            if len(pre_data) != len(data): 
                has_error += 1
                LOG.e(f"Dict Name <{name}> len not match, {len(pre_data)} != {len(data)}")
            for k in pre_data:
                if k not in data:
                    has_error += 1
                    LOG.e(f"Key <{k}> not in data, Name <{name}>")
                    continue
                self.check(name+f".{k}", pre_data[k], data[k])
        else:
            if pre_data != data: 
                has_error += 1
                LOG.e(f"Type: {type(pre_data).__name__} Name <{name}> not match {pre_data} != {data}")

    def record(self, name, data):
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
                return
            LOG.i(f"check {rid}:<{name}> ...")
            self.check(name, pre_data, data)
        else:
            with open(fpath, 'wb') as f:
                pickle.dump((name, data), f)
            LOG.i(f"save {rid}:<{name}> ok")

    def record_params(self, parameters_dict):
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


    def hook_module(self, mod):
        def forward_hook(self2, input, output):
            if "relu" not in self2.__class__.__name__.lower():
                # not test relu, because input may be inplaced
                self.record(self2.__ad_mod_name__+".input", input)
            self.record(self2.__ad_mod_name__+".output", output)

        names = []
        for name, module in mod.named_modules():
            module.__ad_mod_name__ = name
            names.append(name)
            module.register_forward_hook(forward_hook)
        self.record_params(mod.state_dict())
        self.record("module names", names)


    
    


