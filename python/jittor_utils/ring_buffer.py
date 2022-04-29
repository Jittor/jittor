# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: 
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import multiprocessing as mp
import numpy as np
import ctypes
import random
import pickle
import ctypes

recv_raw_call = 0.0

class RingBufferAllocator:
    def __init__(self, size):
        self.size = size
        self.l = mp.Value(ctypes.c_longlong, 0, lock=False)
        self.r = mp.Value(ctypes.c_longlong, 0, lock=False)
        self.is_full = mp.Value(ctypes.c_bool, False, lock=False)
        self.lock = mp.Lock()
        self.cv = mp.Condition(self.lock)
        
    def __repr__(self):
        l = self.l.value
        r = self.r.value
        is_full = self.is_full.value
        if is_full:
            cap = 0
        else:
            cap = (r - l) / self.size
            if cap<=0: cap += 1
        return f"Buffer(free={cap*100:.3f}% l={l} r={r} size={self.size})"
    
    def alloc_with_lock(self, size):
        with self.lock:
            while True:
                location = self.alloc(size)
                if location is not None: break
                self.cv.wait()
        return location
    
    def free_with_lock(self, size):
        with self.lock:
            location = self.free(size)
            self.cv.notify()
            return location

    def clear(self):
        with self.lock:
            self.l.value = 0
            self.r.value = 0
            self.is_full.value = False
    
    def alloc(self, size):
        if size > self.size:
            raise RuntimeError(f"Buffer size too small {self.size}<{size}")
        l = self.l.value
        r = self.r.value
        is_full = self.is_full.value
        if is_full: return None
        if l == r and l > 0:
            self.l.value = self.r.value = l = r = 0
        # [l, r)
        if r > l:
            freed = r - l
            if freed < size:
                # |----l......r---|
                # |----#########--|
                return None
            # |----l......r---|
            # |----#####------|
            location = l
            self.l.value = l = l + size
        else:
            freed = self.size - l
            if freed < size:
                # |.....r------l...|
                # |------------#######
                if size > r:
                    # |.....r------l...|
                    # |#######-----------
                    return None
                # |.....r------l...|
                # |#####-----------
                if size == r:
                    self.is_full.value = is_full= True
                location = 0
                self.l.value = l = size
            else:
                # |.....r------l...|
                # |------------##--|
                location = l
                if freed == size:
                    self.l.value = l = 0
                else:
                    self.l.value = l = l + size
        if l == r:
            self.is_full.value = is_full = True
        return location
    
    def free(self, size):
        l = self.l.value
        r = self.r.value
        is_full = self.is_full.value
        if size==0: return r
        if is_full:
            self.is_full.value = is_full = False
        elif l == r:
            return None
        location = r
        self.r.value = r = r + size
        if r > self.size:
            location = 0
            self.r.value = r = size
        elif r == self.size:
            self.r.value = r = 0
        return location
                
def str_to_char_array(s, array_len):
    if len(s) > array_len: s = s[:array_len]
    a = np.array(s, dtype='c')
    if len(s) < array_len:
        a = np.pad(a, (0,array_len-len(s)), constant_values=' ')
    return a

def char_array_to_str(a):
    return str(a.tobytes(), 'ascii').strip()

class RingBuffer:
    def __init__(self, buffer):
        self.allocator = RingBufferAllocator(len(buffer))
        self.buffer = buffer

    def clear(self): self.allocator.clear()
            
    def send_int(self, data):
        # int: int64[1]
        #      data
        self.send_raw(np.array([data], dtype='int64'))
    def recv_int(self):
        return int(self.recv_raw(8, (1,), 'int64')[0])
    
    def send_float(self, data):
        # float: float64[1]
        #         data
        self.send_raw(np.array([data], dtype='float64'))
    def recv_float(self):
        return float(self.recv_raw(8, (1,), 'float64')[0])
    
    def send_str(self, data):
        # str: int64[1] char[len]
        #        len     data
        data = np.array(data, dtype='c')
        self.send_int(data.nbytes)
        self.send_raw(data)
    def recv_str(self):
        nbytes = self.recv_int()
        data = self.recv_raw(nbytes, nbytes, 'c')
        return str(data.tostring(), 'ascii')
    
    def send_ndarray(self, data):
        # str: int64[1]  char[8]  int64[1]  int64[slen] char[nbytes]
        #        slen     dtype   nbytes     shape       data
        shape = data.shape
        slen = len(shape)
        self.send_int(slen)
        self.send_fix_len_str(str(data.dtype))
        self.send_int(data.nbytes)
        self.send_raw(np.array(shape, dtype='int64'))
        self.send_raw(data)

    def recv_ndarray(self):
        slen = self.recv_int()
        dtype = self.recv_fix_len_str()
        nbytes = self.recv_int()
        shape = self.recv_raw(slen*8, slen, 'int64')
        data = self.recv_raw(nbytes, shape, dtype)
        return data
    
    def send_tuple(self, data):
        # tuple: int64[1] ....
        #         len
        length = len(data)
        self.send_int(length)
        for a in data:
            self.send(a)
    def recv_tuple(self):
        length = self.recv_int()
        return tuple(self.recv() for i in range(length))
    
    def send_list(self, data):
        # list: int64[1] ....
        #         len
        length = len(data)
        self.send_int(length)
        for a in data:
            self.send(a)

    def recv_list(self):
        length = self.recv_int()
        return [self.recv() for i in range(length)]

    def send_pickle(self, data):
        # pickle: int64[1] char[len]
        #         len     data
        data = pickle.dumps(data)
        data = np.frombuffer(data, dtype='c')
        self.send_int(data.nbytes)
        self.send_raw(data)

    def recv_pickle(self):
        nbytes = self.recv_int()
        data = self.recv_raw(nbytes, nbytes, 'c')
        return pickle.loads(data.tostring())
        
    def __repr__(self):
        return f"{self.allocator}@0x{hex(ctypes.addressof(self.buffer))}"
    
    def send_raw(self, data):
        assert isinstance(data, np.ndarray) # and data.flags.c_contiguous
        with self.allocator.lock:
            location = self.allocator.alloc(data.nbytes)
            while location is None:
                self.allocator.cv.wait()
                location = self.allocator.alloc(data.nbytes)
            window = np.ndarray(shape=data.shape, dtype=data.dtype, 
                                buffer=self.buffer, offset=location)
            window[:] = data
            self.allocator.cv.notify()
            assert window.nbytes == data.nbytes
        
    def recv_raw(self, nbytes, shape, dtype):
        global recv_raw_call
        recv_raw_call += 1
        with self.allocator.lock:
            location = self.allocator.free(nbytes)
            while location is None:
                self.allocator.cv.wait()
                location = self.allocator.free(nbytes)
            data = np.ndarray(shape=shape, dtype=dtype, 
                              buffer=self.buffer, offset=location).copy()
            self.allocator.cv.notify()
            assert data.nbytes == nbytes
            return data
    
    def send_fix_len_str(self, s, array_len=8):
        data = str_to_char_array(s, array_len)
        self.send_raw(data)
    
    def recv_fix_len_str(self, array_len=8):
        data = self.recv_raw(8, 8, 'c')
        return char_array_to_str(data)
    
    def send(self, data):
        ts = type(data).__name__
        send = getattr(self, "send_"+ts, self.send_pickle)
        self.send_fix_len_str(ts)
        send(data)
    
    def recv(self):
        ts = self.recv_fix_len_str()
        recv = getattr(self, "recv_"+ts, self.recv_pickle)
        return recv()    
    
