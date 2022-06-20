try:
    import fcntl
except ImportError:
    fcntl = None
    try:
        import win32file
        import pywintypes
        _OVERLAPPED = pywintypes.OVERLAPPED()
    except:
        raise Exception("""pywin32 package not found, please install it.
>>> python3.x -m pip install pywin32
If conda is used, please install with command: 
>>> conda install pywin32""")

import os
from jittor_utils import cache_path, LOG

disable_lock = os.environ.get("disable_lock", "0") == "1"

class Lock:   
    def __init__(self, filename):  
        self.handle = open(filename, 'w')
        LOG.v(f'OPEN LOCK path: {filename} PID: {os.getpid()}')
        self.is_locked = False
      
    def lock(self):
        if disable_lock:
            return
        if fcntl:
            fcntl.flock(self.handle, fcntl.LOCK_EX)
        else:
            hfile = win32file._get_osfhandle(self.handle.fileno())
            win32file.LockFileEx(hfile, 2, 0, -0x10000, _OVERLAPPED)
        self.is_locked = True
        LOG.vv(f'LOCK PID: {os.getpid()}')
        
    def unlock(self):
        if disable_lock:
            return
        if fcntl:
            fcntl.flock(self.handle, fcntl.LOCK_UN)
        else:
            hfile = win32file._get_osfhandle(self.handle.fileno())
            win32file.UnlockFileEx(hfile, 0, -0x10000, _OVERLAPPED)
        self.is_locked = False
        LOG.vv(f'UNLOCK PID: {os.getpid()}')
        
    def __del__(self):  
        self.handle.close()


class _base_scope:
    '''base_scope for support @xxx syntax'''
    def __enter__(self): pass
    def __exit__(self, *exc): pass
    def __call__(self, func):
        def inner(*args, **kw):
            with self:
                ret = func(*args, **kw)
            return ret
        return inner

class lock_scope(_base_scope):
    def __enter__(self):
        self.is_locked = jittor_lock.is_locked
        if not self.is_locked:
            jittor_lock.lock()

    def __exit__(self, *exc):
        if not self.is_locked:
            jittor_lock.unlock()

class unlock_scope(_base_scope):
    def __enter__(self):
        self.is_locked = jittor_lock.is_locked
        if self.is_locked:
            jittor_lock.unlock()

    def __exit__(self, *exc):
        if self.is_locked:
            jittor_lock.lock()

lock_path = os.path.abspath(os.path.join(cache_path, "../jittor.lock"))
if not os.path.exists(lock_path):
    LOG.i("Create lock file:", lock_path)
    try:
        os.mknod(lock_path)
    except:
        pass
jittor_lock = Lock(lock_path)
