import ctypes
import os
if "CUDA_VISIBLE_DEVICES" in os.environ:
    del os.environ["CUDA_VISIBLE_DEVICES"]
if os.name == 'nt':
    cuda_driver = ctypes.CDLL("nvcuda")
else:
    cuda_driver = ctypes.CDLL("libcuda.so")
driver_version = ctypes.c_int()
r = cuda_driver.cuDriverGetVersion(ctypes.byref(driver_version))
assert r == 0
v = driver_version.value

dcount = ctypes.c_int()
cuda_driver.cuInit(0)
r = cuda_driver.cuDeviceGetCount(ctypes.byref(dcount))

for i in range(dcount.value):
    dev = ctypes.c_void_p()
    major = ctypes.c_int()
    minor = ctypes.c_int()
    assert 0 == cuda_driver.cuDeviceGet(ctypes.byref(dev), i)
    assert 0 == cuda_driver.cuDeviceGetAttribute(ctypes.byref(major), 75, dev)
    assert 0 == cuda_driver.cuDeviceGetAttribute(ctypes.byref(minor), 76, dev)
    print(major.value*10+minor.value)
