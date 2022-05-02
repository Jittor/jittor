# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: 
#     Zheng-Ning Liu <lzhengning@gmail.com> 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import jittor.nn as nn
import numpy as np


skip_this_test = False

try:
    jt.dirty_fix_pytorch_runtime_error()
    import torch
    import torch.nn as tnn
except:
    torch = None
    tnn = None
    skip_this_test = True


def check_equal(t_rnn, j_rnn, input, h0, c0):
    j_rnn.load_state_dict(t_rnn.state_dict())

    t_output, (th, tc) = t_rnn(torch.from_numpy(input), 
                              (torch.from_numpy(h0), torch.from_numpy(c0)))
    j_output, (jh, jc) = j_rnn(jt.float32(input), 
                              (jt.float32(h0), jt.float32(c0)))

    assert np.allclose(t_output.detach().numpy(), j_output.data, rtol=1e-03, atol=1e-06)
    assert np.allclose(th.detach().numpy(), jh.data, rtol=1e-03, atol=1e-06)
    assert np.allclose(tc.detach().numpy(), jc.data, rtol=1e-03, atol=1e-06)


@unittest.skipIf(skip_this_test, "No Torch found")
class TestLSTM(unittest.TestCase):
    def test_lstm_cell(self):
        np_h0 = torch.randn(3, 20).numpy()
        np_c0 = torch.randn(3, 20).numpy()

        t_rnn = tnn.LSTMCell(10, 20) 
        input = torch.randn(2, 3, 10)
        h0 = torch.from_numpy(np_h0)
        c0 = torch.from_numpy(np_c0)
        t_output = []
        for i in range(input.size()[0]):
            h0, c0 = t_rnn(input[i], (h0, c0))
            t_output.append(h0)
        t_output = torch.stack(t_output, dim=0)

        j_rnn = nn.LSTMCell(10, 20)
        j_rnn.load_state_dict(t_rnn.state_dict())

        input = jt.float32(input.numpy())
        h0 = jt.float32(np_h0)
        c0 = jt.float32(np_c0)
        j_output = []
        for i in range(input.size()[0]):
            h0, c0 = j_rnn(input[i], (h0, c0))
            j_output.append(h0)
        j_output = jt.stack(j_output, dim=0)

        t_output = t_output.detach().numpy()
        j_output = j_output.data
        assert np.allclose(t_output, j_output, rtol=1e-03, atol=1e-06)

    def test_lstm(self):
        h0 = np.random.rand(1, 2, 20).astype(np.float32)
        c0 = np.random.rand(1, 2, 20).astype(np.float32)
        input = np.random.rand(5, 2, 10).astype(np.float32)

        t_rnn = tnn.LSTM(10, 20) 
        j_rnn = nn.LSTM(10, 20)
        check_equal(t_rnn, j_rnn, input, h0, c0)

        proj_size = 13
        h0 = np.random.rand(1, 2, proj_size).astype(np.float32)
        c0 = np.random.rand(1, 2, 20).astype(np.float32)
        input = np.random.rand(5, 2, 10).astype(np.float32)
        t_rnn = tnn.LSTM(10, 20, proj_size=proj_size) 
        j_rnn = nn.LSTM(10, 20, proj_size=proj_size)
        check_equal(t_rnn, j_rnn, input, h0, c0)

        h0 = np.random.rand(2, 4, 20).astype(np.float32)
        c0 = np.random.rand(2, 4, 20).astype(np.float32)
        input = np.random.rand(5, 4, 10).astype(np.float32)

        t_rnn = tnn.LSTM(10, 20, num_layers=2) 
        j_rnn = nn.LSTM(10, 20, num_layers=2)
        check_equal(t_rnn, j_rnn, input, h0, c0)

        h0 = np.random.rand(2, 4, proj_size).astype(np.float32)
        c0 = np.random.rand(2, 4, 20).astype(np.float32)
        input = np.random.rand(5, 4, 10).astype(np.float32)

        t_rnn = tnn.LSTM(10, 20, num_layers=2, proj_size=proj_size) 
        j_rnn = nn.LSTM(10, 20, num_layers=2, proj_size=proj_size)
        check_equal(t_rnn, j_rnn, input, h0, c0)


if __name__ == "__main__":
    unittest.main()