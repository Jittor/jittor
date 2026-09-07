"""Native checkpoint and pickle entry points."""

import hashlib
import os
import pickle

from jittor_core import Var

def dfs_to_numpy(x):
    if isinstance(x, list):
        for i in range(len(x)):
            x[i] = dfs_to_numpy(x[i])
    elif isinstance(x, dict):
        for k in x:
            x[k] = dfs_to_numpy(x[k])
    elif isinstance(x, Var):
        return x.numpy()
    return x

def safepickle(obj, path):
    if path.endswith(".pth") or path.endswith(".pt") or path.endswith(".bin"):
        from jittor.serialization.save_pytorch import save_pytorch
        save_pytorch(path, obj)
        return
    # Protocol version 4 was added in Python 3.4. It adds support for very large objects, pickling more kinds of objects, and some data format optimizations.
    # ref: <https://docs.python.org/3/library/pickle.html>
    # obj = dfs_to_numpy(obj)
    s = pickle.dumps(obj, 4)
    checksum = hashlib.sha1(s).digest()
    s += bytes(checksum)
    s += b"HCAJSLHD"
    with open(path, 'wb') as f:
        f.write(s)

def _load_pkl(s, path):
    try:
        return pickle.loads(s)
    except Exception as e:
        msg = str(e)
        msg += f"\nPath: \"{path}\""
        if "trunc" in msg:
            msg += "\nThis file maybe corrupted, please consider remove it" \
                 " and re-download."
        raise RuntimeError(msg)

def _upload(path, url, jk, tdir=""):
    from jittor._core.flags import flags
    tdir = tdir + '/' if tdir != "" else ""
    prefix = f"https://cg.cs.tsinghua.edu.cn/jittor/{tdir}assets"
    if url.startswith("jittorhub://"):
        url = url.replace("jittorhub://", prefix+"/build/checkpoints/")
    assert url.startswith(prefix)
    suffix = url[len(prefix):]
    dir_suffix = "/".join(suffix.split("/")[:-1])
    jkey = flags.cache_path+"/_jkey"
    with open(jkey, 'w') as f:
        f.write(jk)
    assert os.system(f"chmod 600 \"{jkey}\"") == 0
    print(dir_suffix)
    assert os.system(f"s""s""h"f" -i \"{jkey}\" jittor" "@" "166" f".111.68.30 mkdir -p Documents/jittor-blog/{tdir}assets{dir_suffix}") == 0
    assert os.system(f"s""c""p"+f" -i \"{jkey}\" \"{path}\" jittor" "@" "166" f".111.68.30:Documents/jittor-blog/{tdir}assets{suffix}") == 0
    assert os.system(f"s""s""h"f" -i \"{jkey}\" jittor" "@" "166" ".111.68.30 Documents/jittor-blog.git/hooks/post-update") == 0


def safeunpickle(path):
    from jittor import compiler
    if path.startswith("jittorhub://"):
        path = path.replace("jittorhub://", f"https://cg.cs.tsinghua.edu.cn/jittor/assets/build/checkpoints/")
    if path.startswith("https:") or path.startswith("http:"):
        base = path.split("/")[-1]
        fname = os.path.join(compiler.ck_path, base)
        from jittor_utils.misc import download_url_to_local
        download_url_to_local(path, base, compiler.ck_path, None)
        path = fname
        if not (path.endswith(".pth") or path.endswith(".pkl") or path.endswith(".pt")):
            return path
    if path.endswith(".pth") or path.endswith(".pt") or path.endswith(".bin") :
        from jittor.serialization.load_pytorch import load_pytorch
        model_dict = load_pytorch(path)
        return model_dict
    with open(path, "rb") as f:
        s = f.read()
    if not s.endswith(b"HCAJSLHD"):
        return _load_pkl(s, path)
    checksum = s[-28:-8]
    s = s[:-28]
    if hashlib.sha1(s).digest() != checksum:
        raise ValueError("Pickle checksum does not match! path: "+path,
        " This file maybe corrupted, please consider remove it"
        " and re-download.")
    return _load_pkl(s, path)

def load(path: str):
    ''' loads an object from a file.
    '''
    model_dict = safeunpickle(path)
    return model_dict

def save(params_dict, path: str):
    ''' saves the parameter dictionary to a file.

    :param params_dict: parameters to be saved
    :type params_dict: list or dictionary
    :param path: file path
    :type path: str
    '''
    safepickle(params_dict, path)
