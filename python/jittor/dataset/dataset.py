# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: 
#     Meng-Hao Guo <guomenghao1997@gmail.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import numpy as np
from urllib import request
import gzip
import pickle
import os
from jittor.dataset.utils import get_random_list, get_order_list, collate_batch, HookTimer
from collections.abc import Sequence, Mapping
import pathlib
from PIL import Image
import multiprocessing as mp
import signal
from jittor_utils import LOG
import jittor as jt
import time
import jittor_utils as jit_utils

dataset_root = os.path.join(jit_utils.home(), ".cache", "jittor", "dataset")
mp_log_v = os.environ.get("mp_log_v", 0) 
mpi = jt.mpi
img_open_hook = HookTimer(Image, "open")
CHECK_MEMORY = int(os.environ.get("CHECK_MEMORY", "0"))

if os.name == "nt":
    from multiprocessing import shared_memory
    class RingBuffer:
        def __init__(self, size, shm=None):
            for i in range(100):
                if (1<<i) >= size: break
            size = 1<<i
            init = False
            if shm is None:
                init = True
                shm = shared_memory.SharedMemory(create=True, size=size+1024)
            rb = jt.core.RingBuffer(size, id(shm.buf), init)
            self.size = size
            self.shm = shm
            self.rb = rb

        def __reduce__(self):
            return (RingBuffer, (self.size, self.shm))
            
        def __del__(self):
            del self.rb
            del self.shm

        def push(self, obj): self.send(obj)
        def pop(self): return self.recv()
        def send(self, obj): self.rb.push(obj)
        def recv(self): return self.rb.pop()
        def clear(self): return self.rb.clear()
        def stop(self): return self.rb.stop()
        def is_stop(self): return self.rb.is_stop()
        def total_pop(self): return self.rb.total_pop()
        def total_push(self): return self.rb.total_push()
        def __repr__(self): return repr(self.rb)
        def keep_numpy_array(self, keep): self.rb.keep_numpy_array(keep)

    jt.RingBuffer = RingBuffer

class Worker:
    def __init__(self, target, args, buffer_size, keep_numpy_array=False):
        self.buffer = jt.RingBuffer(buffer_size)
        self.buffer.keep_numpy_array(keep_numpy_array)

        self.status = mp.Array('f', 5, lock=False)
        self.p = mp.Process(target=target, args=args+(self.buffer,self.status))
        self.p.daemon = True
        self.p.start()

class Dataset(object):
    '''
    Base class for reading data.

    Args::

        [in] batch_size(int): batch size, default 16.
        [in] shuffle(bool): shuffle at each epoch, default False.
        [in] drop_last(bool): if true, the last batch of dataset might smaller than batch_size, default True.
        [in] num_workers(int): number of workers for loading data.
        [in] buffer_size(int): buffer size for each worker in bytes, default(512MB).
        [in] keep_numpy_array(bool): return numpy array rather than jittor array, default(False).
        [in] endless(bool): will this dataset yield data forever, default(False).
    
    Example::

        class YourDataset(Dataset):
            def __init__(self):
                super().__init__()
                self.set_attrs(total_len=1024)

            def __getitem__(self, k):
                return k, k*k

        dataset = YourDataset().set_attrs(batch_size=256, shuffle=True)
        for x, y in dataset:
            ......
    '''
    def __init__(self,
                 batch_size = 16,
                 shuffle = False,
                 drop_last = False,
                 num_workers = 0,
                 buffer_size = 512*1024*1024,
                 stop_grad = True,
                 keep_numpy_array = False,
                 endless = False):
        super().__init__()
        if os.environ.get("DISABLE_MULTIPROCESSING", '0') == '1':
            num_workers = 0
        self.total_len = None
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.buffer_size = buffer_size
        self.stop_grad = stop_grad
        self.keep_numpy_array = keep_numpy_array
        self.endless = endless
        self.epoch_id = 0
        self.sampler = None
        self._disable_workers = False
        self._shuffle_rng = np.random.default_rng(1)

    def __getitem__(self, index):
        raise NotImplementedError

    def __batch_len__(self):
        assert self.total_len >= 0
        assert self.batch_size > 0
        if self.drop_last:
            return self.total_len // self.batch_size
        return (self.total_len-1) // self.batch_size + 1

    def __len__(self):
        return self.__batch_len__()

    def set_attrs(self, **kw):
        ''' 
        You can set attributes of dataset by using set_attrs function, including total_len, batch_size, shuffle, drop_last, num_workers, buffer_size.
        
        Example::

            dataset = YourDataset().set_attrs(batch_size=256, shuffle=True)

        Attrs:

            * batch_size(int): batch size, default 16.
            * total_len(int): total lenght.
            * shuffle(bool): shuffle at each epoch, default False.
            * drop_last(bool): if true, the last batch of dataset might smaller than batch_size, default True.
            * num_workers: number of workers for loading data
            * buffer_size: buffer size for each worker in bytes, default(512MB).
            * stop_grad: stop grad for data, default(True).
        '''
        for k,v in kw.items():
            assert hasattr(self, k), k
            setattr(self, k, v)
        self.reset()
        return self

    def to_jittor(self, batch):
        '''
        Change batch data to jittor array, such as np.ndarray, int, and float.
        '''
        if self.keep_numpy_array: return batch
        if isinstance(batch, jt.Var): return batch
        to_jt = lambda x: jt.array(x).stop_grad() \
            if self.stop_grad else jt.array(x)
        if isinstance(batch, np.ndarray):
            return to_jt(batch)
        if isinstance(batch, dict):
            new_batch = {}
            for k,v in batch.items():
                new_batch[k] = self.to_jittor(v)
            return new_batch
        if not isinstance(batch, (list, tuple)):
            return batch
        new_batch = []
        for a in batch:
            if isinstance(a, np.ndarray):
                new_batch.append(to_jt(a))
            else:
                new_batch.append(self.to_jittor(a))
        return new_batch

    def collate_batch(self, batch):
        '''
        Puts each data field into a tensor with outer dimension batch size.

        Args::

        [in] batch(list): A list of variables, such as jt.var, Image.Image, np.ndarray, int, float, str and so on.

        '''
        return collate_batch(batch)

    def terminate(self):
        '''
        Terminate is used to terminate multi-process worker reading data.
        '''
        if hasattr(self, "workers"):
            for w in self.workers:
                w.p.terminate()
    
    def _worker_main(self, worker_id, buffer, status):
        import jittor_utils
        jittor_utils.cc.init_subprocess()
        jt.jt_init_subprocess()
        seed = jt.get_seed()
        wseed = (seed ^ (worker_id*1167)) ^ 1234
        jt.set_global_seed(wseed)
        # parallel_op_compiler still problematic,
        # it is not work on ubuntu 16.04. but worked on ubuntu 20.04
        # it seems like the static value of parallel compiler
        # is not correctly init.
        jt.flags.use_parallel_op_compiler = 0
        import time
        try:
            gid_obj = self.gid.get_obj()
            gid_lock = self.gid.get_lock()
            start = time.time()
            while True:
                # get id
                with gid_lock:
                    while buffer.is_stop() or self.idqueue.is_stop() or \
                        gid_obj.value >= self.batch_len:
                        self.num_idle.value += 1
                        self.num_idle_c.notify()
                        self.gidc.wait()
                        self.num_idle.value -= 1
                    cid = gid_obj.value
                    batch_index_list = self.index_list_numpy[
                        cid*self.real_batch_size:
                        min(self.real_len, (cid+1)*self.real_batch_size)
                    ].copy()
                    gid_obj.value += 1
                with self.idqueue_lock:
                    self.idqueue.push(worker_id)
                now = time.time()
                other_time = now - start
                start = now

                # load and transform data
                batch = []
                if mp_log_v:
                    print(f"#{worker_id} {os.getpid()} load batch", cid*self.real_batch_size, min(self.real_len, (cid+1)*self.real_batch_size))
                for i in batch_index_list:
                    batch.append(self[i])
                batch = self.collate_batch(batch)
                now = time.time()
                data_time = now - start
                start = now

                # send data to main process
                if mp_log_v:
                    print(f"#{worker_id} {os.getpid()} send", type(batch).__name__, [ type(b).__name__ for b in batch ], buffer)
                try:
                    buffer.send(batch)
                except:
                    if buffer.is_stop():
                        continue
                    raise
                now = time.time()
                send_time = now - start
                start = now
                status[0], status[1], status[2], status[3], status[4] = \
                    other_time, data_time, send_time, \
                    other_time + data_time + send_time, \
                    img_open_hook.duration
                img_open_hook.duration = 0.0
        except:
            import traceback
            line = traceback.format_exc()
            print(line)
            os.kill(os.getppid(), signal.SIGINT)
            exit(0)

    def display_worker_status(self):
        ''' Display dataset worker status, when dataset.num_workers > 0, it will display infomation blow:

.. code-block:: console

        progress:479/5005
        batch(s): 0.302 wait(s):0.000
        recv(s): 0.069  to_jittor(s):0.021
        recv_raw_call: 6720.0
        last 10 workers: [6, 7, 3, 0, 2, 4, 7, 5, 6, 1]
        ID      wait(s) load(s) send(s) total
        #0      0.000   1.340   2.026   3.366   Buffer(free=0.000% l=462425368 r=462425368 size=536870912)
        #1      0.000   1.451   3.607   5.058   Buffer(free=0.000% l=462425368 r=462425368 size=536870912)
        #2      0.000   1.278   1.235   2.513   Buffer(free=0.000% l=462425368 r=462425368 size=536870912)
        #3      0.000   1.426   1.927   3.353   Buffer(free=0.000% l=462425368 r=462425368 size=536870912)
        #4      0.000   1.452   1.074   2.526   Buffer(free=0.000% l=462425368 r=462425368 size=536870912)
        #5      0.000   1.422   3.204   4.625   Buffer(free=0.000% l=462425368 r=462425368 size=536870912)
        #6      0.000   1.445   1.953   3.398   Buffer(free=0.000% l=462425368 r=462425368 size=536870912)
        #7      0.000   1.582   0.507   2.090   Buffer(free=0.000% l=308283552 r=308283552 size=536870912)

Meaning of the outputs:

* progress: dataset loading progress (current/total)
* batch: batch time, exclude data loading time
* wait: time of main proc wait worker proc
* recv: time of recv batch data
* to_jittor: time of batch data to jittor variable
* recv_raw_call: total number of underlying recv_raw called
* last 10 workers: id of last 10 workers which main proc load from.
* table meaning
    * ID: worker id
    * wait: worker wait time
    * open: worker image open time
    * load: worker load time
    * buffer: ring buffer status, such as how many free space, left index, right index, total size(bytes).

Example::
  
  from jittor.dataset import Dataset
  class YourDataset(Dataset):
      pass
  dataset = YourDataset().set_attrs(num_workers=8)
  for x, y in dataset:
      dataset.display_worker_status()
        '''
        if not hasattr(self, "workers"):
            return
        msg = [""]
        msg.append(f"progress:{self.batch_id}/{self.batch_len}")
        msg.append(f"batch(s): {self.batch_time:.3f}\twait(s):{self.wait_time:.3f}")
        msg.append(f"recv(s): {self.recv_time:.3f}\tto_jittor(s):{self.to_jittor_time:.3f}")
        msg.append(f"last 10 workers: {self.last_ids}")
        msg.append(f"ID\twait(s)\topen(s)\tload(s)\tsend(s)\ttotal(s)")
        for i in range(self.num_workers):
            w = self.workers[i]
            s = w.status
            msg.append(f"#{i}\t{s[0]:.3f}\t{s[4]:.3f}\t{s[1]:.3f}\t{s[2]:.3f}\t{s[3]:.3f}\t{w.buffer}")
        LOG.i('\n'.join(msg))

    def _stop_all_workers(self):
        # stop workers
        for w in self.workers:
            w.buffer.stop()
        self.idqueue.stop()
        # wait until all workers idle
        if self.num_idle.value < self.num_workers:
            with self.gid.get_lock():
                self.gid.get_obj().value = self.batch_len
                if mp_log_v:
                    print("idle num", self.num_idle.value)
                while self.num_idle.value < self.num_workers:
                    self.num_idle_c.wait()
                    if mp_log_v:
                        print("idle num", self.num_idle.value)
        # clean workers' buffer
        for w in self.workers:
            w.buffer.clear()
        self.idqueue.clear()
        self.gid_obj.value = 0
            
    def _init_workers(self, index_list):
        jt.migrate_all_to_cpu()
        jt.clean()
        jt.gc()
        self.index_list = mp.Array('i', self.real_len, lock=False)
        workers = []
        # get worker id
        self.idqueue = jt.RingBuffer(2048)
        self.idqueue_lock = mp.Lock()
        # global token index
        self.gid = mp.Value('i', self.batch_len)
        self.gid.value = 0
        # global token index condition
        self.gidc = mp.Condition(self.gid.get_lock())
        # number of idle workers
        self.num_idle = mp.Value('i', 0, lock=False)
        # number of idle workers condition
        self.num_idle_c = mp.Condition(self.gid.get_lock())
        self.index_list_numpy = np.ndarray(dtype='int32', shape=self.real_len, buffer=self.index_list)
        self.index_list_numpy[:] = index_list
        for i in range(self.num_workers):
            w = Worker(target=self._worker_main, args=(i,), 
                       buffer_size=self.buffer_size,
                       keep_numpy_array=self.keep_numpy_array)
            workers.append(w)
        self.workers = workers

    def reset(self):
        if not hasattr(self, "workers"):
            return
        self._stop_all_workers()
        self.terminate()
        del self.index_list
        del self.idmap
        del self.gid
        del self.gidc
        del self.num_idle
        del self.num_idle_c
        del self.workers
        del self.index_list_numpy

    def __del__(self):
        if mp_log_v:
            print("dataset deleted")
        try:
            self.terminate()
        except:
            pass

    def __real_len__(self):
        if self.total_len is None:
            self.total_len = len(self)
        return self.total_len

    def _get_index_list(self):
        if self.total_len is None:
            self.total_len = len(self)
        # maybe rewrite by sampler
        total_len = self.total_len
        if self.sampler:
            index_list = list(self.sampler.__iter__())
            total_len = len(index_list)
            # check is not batch sampler
            if len(index_list):
                assert not isinstance(index_list[0], (list,tuple)), "Batch sampler not support yet."
        elif self.shuffle == False:
            index_list = get_order_list(self.total_len)
        else:
            # using _shuffle_rng to generate multiprocess
            # consist shuffle list
            # index_list = get_random_list(self.total_len)
            index_list = self._shuffle_rng.permutation(range(self.total_len))
        
        # scatter index_list for all mpi process
        # scatter rule:
        #   batch 1   batch 2
        # [........] [........] ...
        #  00011122   00011122
        # if last batch is smaller than world_size
        # pad to world_size
        #  last batch
        # [.] -> [012]
        if jt.in_mpi:
            world_size = mpi.world_size()
            world_rank = mpi.world_rank()
            index_list = np.int32(index_list)
            # TODO: mpi broadcast in subprocess has bug, fix it
            # mpi.broadcast(index_list, 0)

            assert self.batch_size >= world_size, \
                f"Batch size({self.batch_size}) is smaller than MPI world_size({world_size})"
            real_batch_size = (self.batch_size-1) // world_size + 1
            if real_batch_size * world_size != self.batch_size:
                LOG.w("Batch size is not divisible by MPI world size, "
                      "The distributed version may be different from "
                      "the single-process version.")
            fix_batch = total_len // self.batch_size
            last_batch = total_len - fix_batch * self.batch_size
            fix_batch_l = index_list[0:fix_batch*self.batch_size] \
                .reshape(-1,self.batch_size)
            fix_batch_l = fix_batch_l[
                :,real_batch_size*world_rank:real_batch_size*(world_rank+1)]
            real_batch_size = fix_batch_l.shape[1]
            fix_batch_l = fix_batch_l.flatten()
            if not self.drop_last and last_batch > 0:
                last_batch_l = index_list[-last_batch:]
                real_last_batch = (last_batch-1)//world_size+1
                l = real_last_batch * world_rank
                r = l + real_last_batch
                if r > last_batch: r = last_batch
                if l >= r: l = r-1
                index_list = np.concatenate([fix_batch_l, last_batch_l[l:r]])
            else:
                index_list = fix_batch_l

            self.real_len = len(index_list)
            self.real_batch_size = real_batch_size
            assert total_len // self.batch_size == \
                self.real_len // self.real_batch_size, f"Number of batches({total_len // self.batch_size}!={self.real_len // self.real_batch_size}) not match, total_len: {total_len}, batch_size: {self.batch_size}, real_len: {self.real_len}, real_batch_size: {self.real_batch_size}"
        else:
            self.real_len = self.total_len
            self.real_batch_size = self.batch_size
        self.batch_len = self.__batch_len__()
        return index_list

    def _epochs(self):
        if self.endless:
            while True:
                yield
                self.epoch_id += 1
        else:
            yield
        
    def __iter__(self):
        if self._disable_workers:
            self.num_workers = 0
        index_list = self._get_index_list()
        
        if not hasattr(self, "workers") and self.num_workers:
            self._init_workers(index_list)
            self.last_ids = [-1] * 10
        
        if self.num_workers:
            start = time.time()
            self.batch_time = 0
            gid_obj = self.gid.get_obj()
            gid_lock = self.gid.get_lock()

            for _ in self._epochs():
                with gid_lock:
                    if self.num_idle.value:
                        self.gidc.notify_all()

                for i in range(self.batch_len):
                    if self.num_idle.value:
                        with gid_lock:
                            if self.num_idle.value and \
                                gid_obj.value >= self.batch_len:
                                index_list = self._get_index_list()
                                self.index_list_numpy[:] = index_list
                                gid_obj.value = 0
                                self.gidc.notify_all()

                    # get which worker has this batch
                    worker_id = self.idqueue.pop()

                    now = time.time()
                    self.wait_time = now - start
                    start = now

                    self.last_ids[i%10] = worker_id
                    self.batch_id = i
                    w = self.workers[worker_id]
                    if mp_log_v:
                        print(f"#{worker_id} {os.getpid()} recv buffer", w.buffer)
                    batch = w.buffer.recv()

                    now = time.time()
                    self.recv_time = now - start
                    start = now

                    if mp_log_v:
                        print(f"#{worker_id} {os.getpid()} recv", type(batch).__name__, [ type(b).__name__ for b in batch ])
                    batch = self.to_jittor(batch)
                    
                    now = time.time()
                    self.to_jittor_time = now - start
                    start = now

                    yield batch

                    now = time.time()
                    self.batch_time = now - start
                    start = now

                    if CHECK_MEMORY and self.batch_id % CHECK_MEMORY == 0:
                        jt.display_memory_info()
        else:
            for _ in self._epochs():
                self.batch_id = 0
                batch_data = []
                for idx in index_list:
                    batch_data.append(self[int(idx)])
                    if len(batch_data) == self.real_batch_size:
                        batch_data = self.collate_batch(batch_data)
                        tmp = batch_data
                        batch_data = self.to_jittor(batch_data)
                        # breakpoint()
                        yield batch_data
                        self.batch_id += 1
                        if CHECK_MEMORY and self.batch_id % CHECK_MEMORY == 0:
                            jt.display_memory_info()
                        batch_data = []

                # depend on drop_last
                if not self.drop_last and len(batch_data) > 0:
                    batch_data = self.collate_batch(batch_data)
                    batch_data = self.to_jittor(batch_data)
                    self.batch_id += 1
                    yield batch_data


class ImageFolder(Dataset):
    """
    A image classify dataset, load image and label from directory::
    
        * root/label1/img1.png
        * root/label1/img2.png
        * ...
        * root/label2/img1.png
        * root/label2/img2.png
        * ...

    Args::

        [in] root(string): Root directory path.

    Attributes::

        * classes(list): List of the class names.
        * class_to_idx(dict): map from class_name to class_index.
        * imgs(list): List of (image_path, class_index) tuples

    Example::

        train_dir = './data/celebA_train'
        train_loader = ImageFolder(train_dir).set_attrs(batch_size=batch_size, shuffle=True)
        for batch_idx, (x_, target) in enumerate(train_loader):
            ...

    """
    def __init__(self, root, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.classes = sorted([d.name for d in os.scandir(root) if d.is_dir()])
        self.class_to_idx = {v:k for k,v in enumerate(self.classes)}
        self.imgs = []
        image_exts = set(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))
        
        for i, class_name in enumerate(self.classes):
            class_dir = os.path.join(root, class_name)
            for dname, _, fnames in sorted(os.walk(class_dir, followlinks=True)):
                for fname in sorted(fnames):
                    if os.path.splitext(fname)[-1].lower() in image_exts:
                        path = os.path.join(class_dir, fname)
                        self.imgs.append((path, i))
        LOG.i(f"Found {len(self.classes)} classes and {len(self.imgs)} images.")
        self.set_attrs(total_len=len(self.imgs))
        
    def __getitem__(self, k):
        with open(self.imgs[k][0], 'rb') as f:
            img = Image.open(f).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, self.imgs[k][1]

class VarDataset(Dataset):
    """ Dataset using Var directly, TensorDataset is alias of VarDataset, Example::

    import jittor as jt
    from jittor.dataset import VarDataset

    x = jt.array([1,2,3])
    y = jt.array([4,5,6])
    z = jt.array([7,8,9])
    dataset = VarDataset(x, y, z)
    dataset.set_attrs(batch_size=1)

    for a,b,c in dataset:
        print(a,b,c)
    # will print
    #  1,4,7
    #  2,5,8
    #  3,6,9

    """
    def __init__(self, *args):
        super().__init__()
        self.args = args
        self._disable_workers = True
        assert len(args), "At lease one args"
        l = len(args[0])
        for a in args:
            assert l == len(a), "Len should be the same"
        self.set_attrs(total_len=l)

    def __getitem__(self, idx):
        return [ a[idx] for a in self.args ]
        

    def collate_batch(self, batch):
        b = collate_batch(batch)
        for i in range(len(self.args)):
            x = b[i]
            if jt.is_var(self.args[i]) and self.args[i].ndim == 1:
                x.assign(x.squeeze(-1))
        return b

TensorDataset = VarDataset