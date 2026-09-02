# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: 
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""A hand-run MNIST + MPI example, not a test.

It was called ``test_mpi_in_py.py`` and its only test class is commented out at
the bottom, so pytest collected zero cases from it -- and a file that collects
zero cases is reported exactly like one that passes. Renamed so the tree says
what it is; run it with ``python tests/distributed/mpi_in_py_example.py``.
"""

import unittest
import os, sys
import jittor as jt

from _helpers.child_process import run_mpi_python
import numpy as np
from jittor import nn
from jittor import dataset
mpi = jt.compile_extern.mpi


class Model(nn.Module):
    def __init__(self, input_size):
        self.linear1 = nn.Linear(input_size, 10)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(10, 10)
    def execute(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        return self.linear2(x)

def fork_with_mpi(num_procs=4):
    import sys
    if jt.in_mpi:
        # you can mult other process output
        if jt.rank != 0:
            sys.stdout = open("/dev/null", "w")
        return
    else:
        print(sys.argv)
        # mpirun starts the ranks itself, so none of them inherits this
        # process' sys.path: the helper pins PYTHONPATH for every rank.
        completed = run_mpi_python(num_procs, sys.argv, merge_stderr=True,
                                   launcher="mpirun")
        print(completed.stdout)
        exit(0)

def main():
    mnist = dataset.MNIST()
    model = Model(mnist[0][0].size)
    sgd = jt.optim.SGD(model.parameters(), 1e-3)
    fork_with_mpi()

    for data, label in mnist:
        pred = model(data.reshape(data.shape[0], -1))
        # print(data.shape, label.shape, pred.shape)
        loss = nn.cross_entropy_loss(pred, label)
        sgd.step(loss)
        print(jt.rank, mnist.epoch_id, mnist.batch_id, loss)
        # break



# class TestMpiInPy(unittest.TestCase):
#     def test(self):
#         main()


if __name__ == "__main__":
    # unittest.main()
    main()