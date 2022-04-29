# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import os, sys, shutil
import glob
import jittor_utils as jit_utils

cache_path = os.path.join(jit_utils.home(), ".cache", "jittor")

def callback(func, path, exc_info):
    print(f"remove \"{path}\" failed.")

def rmtree(path):
    if os.path.isdir(path):
        print(f"remove \"{path}\" recursive.")
        shutil.rmtree(path, onerror=callback)

def clean_all():
    rmtree(cache_path)

def clean_core():
    rmtree(cache_path+"/default")
    rmtree(cache_path+"/master")
    fs = glob.glob(cache_path+"/jt*")
    for f in fs: rmtree(f)

def clean_cuda():
    rmtree(cache_path+"/jtcuda")
    rmtree(cache_path+"/cutt")
    rmtree(cache_path+"/cub")
    rmtree(cache_path+"/nccl")

def clean_dataset():
    rmtree(cache_path+"/dataset")

def print_help():
    msg = "|".join(keys)
    print(f"Usage: {sys.executable} -m jittor_utils.clean_cache [{msg}]")
    exit()


keys = [ k[6:] for k in globals() if k.startswith("clean_") ]

if __name__ == "__main__":
    if len(sys.argv)==1:
        print_help()
    else:
        for k in sys.argv[1:]:
            if k not in keys:
                print_help()
            func = globals()["clean_"+k]
            func()