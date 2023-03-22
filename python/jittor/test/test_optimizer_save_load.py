import unittest
import jittor as jt
import numpy as np
from jittor import nn
import os


def compare(x, y, shape=4):
    assert((x == y).sum() == shape)


def test_optim(optimzer_type, **kwargs):
    # input

    x = jt.rand(20, 2, 2)
    y1 = []
    y2 = []

    # model & optimizer 1 for save
    linear1 = nn.Linear(2, 2)
    opt = optimzer_type(linear1.parameters(), **kwargs)
    for i in range(10):
        y = linear1(x[i])
        y1.append(y)
        opt.step(y)
    opt_dict = opt.state_dict()
    linear_dict = linear1.state_dict()
    jt.save({'opt': opt_dict, 'linear': linear_dict}, "./optim_test.tar")
    for i in range(10, 20, 1):
        y = linear1(x[i])
        y1.append(y)
        opt.step(y)

    # model & optimizer 2 for load
    linear2 = nn.Linear(2, 2)
    opt2 = optimzer_type(linear2.parameters(), **kwargs)
    opt2_dict = jt.load("./optim_test.tar")
    opt2.load_state_dict(opt2_dict['opt'])
    linear2.load_state_dict(opt2_dict['linear'])
    for i in range(10, 20, 1):
        y = linear2(x[i])
        y2.append(y)
        opt2.step(y)

    for i in range(10):
        compare(y1[10+i], y2[i])


class TestOptimizerSaveLoad(unittest.TestCase):
    def test(self):
        optims = [
            {'opt': jt.nn.SGD, 'kwargs': {'lr': 0.1, 'momentum': 1e-2,
                                          'weight_decay': 1e-2, 'dampening': 1e-3, 'nesterov': True}},
            {'opt': jt.nn.RMSprop, 'kwargs': {'lr': 0.1}},
            {'opt': jt.nn.Adam, 'kwargs': {'lr': 0.1, 'weight_decay': 1e-2}},
            {'opt': jt.nn.AdamW, 'kwargs': {'lr': 0.1, 'weight_decay': 1e-2}},
        ]
        for optim in optims:
            test_optim(optim['opt'], **optim['kwargs'])

    def tearDown(self):
        os.remove("./optim_test.tar")


if __name__ == '__main__':
    unittest.main()
