import fcntl
import os
from jittor_utils import cache_path

class Lock:   
    def __init__(self, filename):  
        self.handle = open(filename, 'w') 
        print(f'Create lock for {filename}, PID {os.getpid()}') 
      
    def lock(self):  
        ret = fcntl.flock(self.handle, fcntl.LOCK_EX)
        print(f'Add lock success {ret}, PID {os.getpid()}')
        
    def unlock(self):  
        ret = fcntl.flock(self.handle, fcntl.LOCK_UN)  
        print(f'Release lock success {ret}, PID {os.getpid()}')
        
    def __del__(self):  
        self.handle.close()

lock_path = os.path.join(cache_path, "../jittor.lock")
if not os.path.exists(lock_path):
    os.mknod(lock_path)
jittor_lock = Lock(lock_path)