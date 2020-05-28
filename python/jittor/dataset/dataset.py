# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: 
#     Meng-Hao Guo <guomenghao1997@gmail.com>
#     Dun Liang <randonlang@gmail.com>. 
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import numpy as np
from urllib import request
import gzip
import pickle
import os
from jittor.dataset.utils import get_random_list, get_order_list, collate_batch
from collections.abc import Sequence, Mapping
import pathlib
from PIL import Image
from jittor_utils.ring_buffer import RingBuffer
import multiprocessing as mp
import signal
from jittor_utils import LOG
import jittor as jt

dataset_root = os.path.join(pathlib.Path.home(), ".cache", "jittor", "dataset")
mp_log_v = os.environ.get("mp_log_v", 0) 
mpi = jt.mpi

class Worker:
    def __init__(self, target, args, buffer_size):
        buffer = mp.Array('c', buffer_size, lock=False)
        self.buffer = RingBuffer(buffer)
        self.p = mp.Process(target=target, args=args+(self.buffer,))
        self.p.daemon = True
        self.p.start()

class Dataset(object):
    '''
    base class for reading data
    
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
                 buffer_size = 512*1024*1024):
        super().__init__()
        self.total_len = None
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.buffer_size = buffer_size

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        assert self.total_len >= 0
        assert self.batch_size > 0
        if self.drop_last:
            return self.total_len // self.batch_size
        return (self.total_len-1) // self.batch_size + 1

    def set_attrs(self, **kw):
        '''set attributes of dataset, equivalent to set_attr
        
        Attrs:

            * batch_size(int): batch size, default 16.
            * totol_len(int): totol lenght.
            * shuffle(bool): shuffle at each epoch, default False.
            * drop_last(bool): if true, the last batch of dataset might smaller than batch_size, default True.
            * num_workers: number of workers for loading data
            * buffer_size: buffer size for each worker in bytes, default(512MB).
        '''
        for k,v in kw.items():
            assert hasattr(self, k), k
            setattr(self, k, v)
        return self

    def to_jittor(self, batch):
        if isinstance(batch, np.ndarray):
            return jt.array(batch)
        assert isinstance(batch, Sequence)
        new_batch = []
        for a in batch:
            if isinstance(a, np.ndarray) or \
                isinstance(a, int) or \
                isinstance(a, float):
                new_batch.append(jt.array(a))
            else:
                new_batch.append(a)
        return new_batch

    def collate_batch(self, batch):
        return collate_batch(batch)

    def terminate(self):
        if hasattr(self, "workers"):
            for w in self.workers:
                w.p.terminate()
    
    def _worker_main(self, worker_id, buffer):
        try:
            gid_obj = self.gid.get_obj()
            gid_lock = self.gid.get_lock()
            while True:
                with gid_lock:
                    while gid_obj.value >= self.batch_len:
                        self.num_idle.value += 1
                        self.num_idle_c.notify()
                        self.gidc.wait()
                        self.num_idle.value -= 1
                    cid = gid_obj.value
                    self.idmap[cid] = worker_id
                    gid_obj.value += 1
                    self.gidc.notify()
                batch = []
                if mp_log_v:
                    print(f"#{worker_id} {os.getpid()} load batch", cid*self.real_batch_size, min(self.real_len, (cid+1)*self.real_batch_size))
                for i in range(cid*self.real_batch_size, min(self.real_len, (cid+1)*self.real_batch_size)):
                    batch.append(self[self.index_list[i]])
                batch = self.collate_batch(batch)
                if mp_log_v:
                    print(f"#{worker_id} {os.getpid()} send", type(batch).__name__, [ type(b).__name__ for b in batch ], buffer)
                buffer.send(batch)
        except:
            os.kill(os.getppid(), signal.SIGINT)
            raise

    def _stop_all_workers(self):
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
            
    def _init_workers(self):
        self.index_list = mp.Array('i', self.real_len, lock=False)
        workers = []
        # batch id to worker id
        self.idmap = mp.Array('i', self.batch_len, lock=False)
        # global token index
        self.gid = mp.Value('i', self.batch_len)
        # global token index condition
        self.gidc = mp.Condition(self.gid.get_lock())
        # number of idle workers
        self.num_idle = mp.Value('i', 0, lock=False)
        # number of idle workers condition
        self.num_idle_c = mp.Condition(self.gid.get_lock())
        for i in range(self.num_workers):
            w = Worker(target=self._worker_main, args=(i,), 
                       buffer_size=self.buffer_size)
            workers.append(w)
        self.workers = workers
        self.index_list_numpy = np.ndarray(dtype='int32', shape=self.real_len, buffer=self.index_list)

    def __del__(self):
        if mp_log_v:
            print("dataset deleted")
        self.terminate()

    def __iter__(self):
        if self.shuffle == False:
            index_list = get_order_list(self.total_len)
        else:
            index_list = get_random_list(self.total_len)
        
        # scatter index_list for all mpi process
        # scatter rule:
        #   batch 1   batch 2
        # [........] [........] ...
        #  00011122   00011122
        # if last batch is smaller than world_size
        # pad to world_size
        #  last batch
        # [.] -> [012]
        if mpi:
            world_size = mpi.world_size()
            world_rank = mpi.world_rank()
            index_list = np.int32(index_list)
            mpi.broadcast(index_list, 0)

            assert self.batch_size >= world_size, \
                f"Batch size({self.batch_size}) is smaller than MPI world_size({world_size})"
            real_batch_size = (self.batch_size-1) // world_size + 1
            if real_batch_size * world_size != self.batch_size:
                LOG.w("Batch size is not divisible by MPI world size, "
                      "The distributed version may be different from "
                      "the single-process version.")
            fix_batch = self.total_len // self.batch_size
            last_batch = self.total_len - fix_batch * self.batch_size
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
            assert self.total_len // self.batch_size == \
                self.real_len // self.real_batch_size
        else:
            self.real_len = self.total_len
            self.real_batch_size = self.batch_size
            
        self.batch_len = len(self)
        
        if not hasattr(self, "workers") and self.num_workers:
            self._init_workers()
        
        if self.num_workers:
            self._stop_all_workers()
            self.index_list_numpy[:] = index_list
            gid_obj = self.gid.get_obj()
            gid_lock = self.gid.get_lock()
            with gid_lock:
                gid_obj.value = 0
                self.gidc.notify_all()
            for i in range(self.batch_len):
                # try not get lock first
                if gid_obj.value <= i:
                    with gid_lock:
                        if gid_obj.value <= i:
                            if mp_log_v:
                                print("wait")
                            self.gidc.wait()
                worker_id = self.idmap[i]
                w = self.workers[worker_id]
                if mp_log_v:
                    print(f"#{worker_id} {os.getpid()} recv buffer", w.buffer)
                batch = w.buffer.recv()
                if mp_log_v:
                    print(f"#{worker_id} {os.getpid()} recv", type(batch).__name__, [ type(b).__name__ for b in batch ])
                batch = self.to_jittor(batch)
                yield batch
        else:
            batch_data = []
            for idx in index_list:
                batch_data.append(self[int(idx)])
                if len(batch_data) == self.real_batch_size:
                    batch_data = self.collate_batch(batch_data)
                    batch_data = self.to_jittor(batch_data)
                    yield batch_data
                    batch_data = []

            # depend on drop_last
            if not self.drop_last and len(batch_data) > 0:
                batch_data = self.collate_batch(batch_data)
                batch_data = self.to_jittor(batch_data)
                yield batch_data


class ImageFolder(Dataset):
    """A image classify dataset, load image and label from directory:
    
        * root/label1/img1.png
        * root/label1/img2.png
        * ...
        * root/label2/img1.png
        * root/label2/img2.png
        * ...

    Args:

        * root(string): Root directory path.

    Attributes:

        * classes(list): List of the class names.
        * class_to_idx(dict): map from class_name to class_index.
        * imgs(list): List of (image_path, class_index) tuples
    """
    def __init__(self, root, transform=None):
        # import ipdb; ipdb.set_trace()
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
