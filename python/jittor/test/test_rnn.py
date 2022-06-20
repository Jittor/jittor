# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved.
# Maintainers:
#     Zheng-Ning Liu <lzhengning@gmail.com>
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
from unittest.case import skipIf
try:
    import torch
    import torch.nn as tnn
except:
    torch = None
    tnn = None
    skip_this_test = True

import jittor as jt
import jittor.nn as nn
import numpy as np

skip_this_test = False

def check_equal_1(t_rnn, j_rnn, input, h0, dev=None):
    j_rnn.load_state_dict(t_rnn.state_dict())

    if dev:
        t_output, th = t_rnn(torch.from_numpy(input).to(dev), torch.from_numpy(h0).to(dev))

    else:
        t_output, th = t_rnn(torch.from_numpy(input), torch.from_numpy(h0))
    t_output = t_output.detach().cpu().numpy()
    th = th.detach().cpu().numpy()

    j_output, jh = j_rnn(jt.float32(input), jt.float32(h0))
    j_output, jh = j_output.data, jh.data

    np.testing.assert_allclose(t_output, j_output.data, rtol=1e-03, atol=1e-06)
    np.testing.assert_allclose(th, jh.data, rtol=1e-03, atol=1e-06)

def check_equal_2(t_rnn, j_rnn, input, h0, c0, dev=None):
    j_rnn.load_state_dict(t_rnn.state_dict())

    if dev:
        t_output, (th, tc) = t_rnn(torch.from_numpy(input).to(dev),
            (torch.from_numpy(h0).to(dev), torch.from_numpy(c0).to(dev)))
    else:
        t_output, (th, tc) = t_rnn(torch.from_numpy(input).to(dev),
            (torch.from_numpy(h0), torch.from_numpy(c0)))

    j_output, (jh, jc) = j_rnn(jt.float32(input),
                              (jt.float32(h0), jt.float32(c0)))

    np.testing.assert_allclose(t_output.detach().cpu().numpy(), j_output.data, rtol=1e-03, atol=1e-06)
    np.testing.assert_allclose(th.detach().cpu().numpy(), jh.data, rtol=1e-03, atol=1e-06)
    np.testing.assert_allclose(tc.detach().cpu().numpy(), jc.data, rtol=1e-03, atol=1e-06)


@unittest.skipIf(skip_this_test, "No Torch found")
class TestRNN(unittest.TestCase):
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

    def test_rnn_cell(self):
        np_h0 = torch.randn(3, 20).numpy()

        t_rnn = tnn.RNNCell(10, 20)
        input = torch.randn(2, 3, 10)
        h0 = torch.from_numpy(np_h0)
        t_output = []
        for i in range(input.size()[0]):
            h0 = t_rnn(input[i], h0)
            t_output.append(h0)
        t_output = torch.stack(t_output, dim=0)

        j_rnn = nn.RNNCell(10, 20)
        j_rnn.load_state_dict(t_rnn.state_dict())

        input = jt.float32(input.numpy())
        h0 = jt.float32(np_h0)
        j_output = []
        for i in range(input.size()[0]):
            h0 = j_rnn(input[i], h0)
            j_output.append(h0)
        j_output = jt.stack(j_output, dim=0)

        t_output = t_output.detach().numpy()
        j_output = j_output.data
        assert np.allclose(t_output, j_output, rtol=1e-03, atol=1e-06)

    def test_gru_cell(self):
        np_h0 = torch.randn(3, 20).numpy()

        t_rnn = tnn.GRUCell(10, 20)
        input = torch.randn(2, 3, 10)
        h0 = torch.from_numpy(np_h0)
        t_output = []
        for i in range(input.size()[0]):
            h0 = t_rnn(input[i], h0)
            t_output.append(h0)
        t_output = torch.stack(t_output, dim=0)

        j_rnn = nn.GRUCell(10, 20)
        j_rnn.load_state_dict(t_rnn.state_dict())

        input = jt.float32(input.numpy())
        h0 = jt.float32(np_h0)
        j_output = []
        for i in range(input.size()[0]):
            h0 = j_rnn(input[i], h0)
            j_output.append(h0)
        j_output = jt.stack(j_output, dim=0)

        t_output = t_output.detach().numpy()
        j_output = j_output.data
        assert np.allclose(t_output, j_output, rtol=1e-03, atol=1e-06)

    def test_basic_rnn(self):
        h0 = np.random.rand(1, 24, 200).astype(np.float32)
        input = np.random.rand(32, 24, 100).astype(np.float32)

        t_rnn = tnn.RNN(100, 200)
        j_rnn = nn.RNN(100, 200)
        check_equal_1(t_rnn, j_rnn, input, h0)

    def test_multilayer_rnn(self):
        h0 = np.random.rand(4, 4, 200).astype(np.float32)
        input = np.random.rand(5, 4, 100).astype(np.float32)

        t_rnn = tnn.RNN(100, 200, num_layers=4)
        j_rnn = nn.RNN(100, 200, num_layers=4)
        check_equal_1(t_rnn, j_rnn, input, h0)

    def test_bidirectional_rnn(self):
        h0 = np.random.rand(2, 1, 200).astype(np.float32)
        input = np.random.rand(5, 1, 100).astype(np.float32)

        t_rnn = tnn.RNN(100, 200, bidirectional=True)
        j_rnn = nn.RNN(100, 200, bidirectional=True)
        check_equal_1(t_rnn, j_rnn, input, h0)

    def test_no_bias_rnn(self):
        h0 = np.random.rand(4, 4, 200).astype(np.float32)
        input = np.random.rand(5, 4, 100).astype(np.float32)

        t_rnn = tnn.RNN(100, 200, num_layers=2, bidirectional=True, bias=False)
        j_rnn = nn.RNN(100, 200, num_layers=2, bidirectional=True, bias=False)
        check_equal_1(t_rnn, j_rnn, input, h0)

    def test_dropout_rnn(self):
        h0 = np.random.rand(2, 4, 200).astype(np.float32)
        input = np.random.rand(5, 4, 100).astype(np.float32)

        t_rnn = tnn.RNN(100, 200, num_layers=2, dropout=0.5, bias=False)
        j_rnn = nn.RNN(100, 200, num_layers=2, dropout=0.5, bias=False)
        t_rnn.eval()
        j_rnn.eval()
        check_equal_1(t_rnn, j_rnn, input, h0)

    def test_basic_lstm(self):
        h0 = np.random.rand(1, 24, 200).astype(np.float32)
        c0 = np.random.rand(1, 24, 200).astype(np.float32)
        input = np.random.rand(32, 24, 100).astype(np.float32)

        t_rnn = tnn.LSTM(100, 200)
        j_rnn = nn.LSTM(100, 200)
        check_equal_2(t_rnn, j_rnn, input, h0, c0)

    def test_projection_lstm(self):
        proj_size = 13
        h0 = np.random.rand(1, 24, proj_size).astype(np.float32)
        c0 = np.random.rand(1, 24, 200).astype(np.float32)
        input = np.random.rand(32, 24, 100).astype(np.float32)
        t_rnn = tnn.LSTM(100, 200, proj_size=proj_size)
        j_rnn = nn.LSTM(100, 200, proj_size=proj_size)
        check_equal_2(t_rnn, j_rnn, input, h0, c0)

    def test_multilayer_lstm(self):
        h0 = np.random.rand(4, 4, 200).astype(np.float32)
        c0 = np.random.rand(4, 4, 200).astype(np.float32)
        input = np.random.rand(5, 4, 100).astype(np.float32)

        t_rnn = tnn.LSTM(100, 200, num_layers=4)
        j_rnn = nn.LSTM(100, 200, num_layers=4)
        check_equal_2(t_rnn, j_rnn, input, h0, c0)

    def test_multilayer_projection_lstm(self):
        proj_size = 8
        h0 = np.random.rand(2, 4, proj_size).astype(np.float32)
        c0 = np.random.rand(2, 4, 20).astype(np.float32)
        input = np.random.rand(5, 4, 10).astype(np.float32)

        t_rnn = tnn.LSTM(10, 20, num_layers=2, proj_size=proj_size)
        j_rnn = nn.LSTM(10, 20, num_layers=2, proj_size=proj_size)
        check_equal_2(t_rnn, j_rnn, input, h0, c0)

    def test_bidirectional_lstm(self):
        h0 = np.random.rand(2, 1, 200).astype(np.float32)
        c0 = np.random.rand(2, 1, 200).astype(np.float32)
        input = np.random.rand(5, 1, 100).astype(np.float32)

        t_rnn = tnn.LSTM(100, 200, bidirectional=True)
        j_rnn = nn.LSTM(100, 200, bidirectional=True)
        check_equal_2(t_rnn, j_rnn, input, h0, c0)

    def test_bidirectional_projection_lstm(self):
        proj_size = 10
        h0 = np.random.rand(2, 4, proj_size).astype(np.float32)
        c0 = np.random.rand(2, 4, 200).astype(np.float32)
        input = np.random.rand(5, 4, 100).astype(np.float32)

        t_rnn = tnn.LSTM(100, 200, bidirectional=True, proj_size=proj_size)
        j_rnn = nn.LSTM(100, 200, bidirectional=True, proj_size=proj_size)
        check_equal_2(t_rnn, j_rnn, input, h0, c0)

    def test_multilayer_bidirectional_projection_lstm(self):
        h0 = np.random.rand(4, 4, 200).astype(np.float32)
        c0 = np.random.rand(4, 4, 200).astype(np.float32)
        input = np.random.rand(5, 4, 100).astype(np.float32)

        t_rnn = tnn.LSTM(100, 200, num_layers=2, bidirectional=True, bias=False)
        j_rnn = nn.LSTM(100, 200, num_layers=2, bidirectional=True, bias=False)
        check_equal_2(t_rnn, j_rnn, input, h0, c0)

    def test_dropout_lstm(self):
        h0 = np.random.rand(2, 4, 200).astype(np.float32)
        c0 = np.random.rand(2, 4, 200).astype(np.float32)
        input = np.random.rand(5, 4, 100).astype(np.float32)

        t_rnn = tnn.LSTM(100, 200, num_layers=2, dropout=0.5, bias=False)
        j_rnn = nn.LSTM(100, 200, num_layers=2, dropout=0.5, bias=False)
        t_rnn.eval()
        j_rnn.eval()
        check_equal_2(t_rnn, j_rnn, input, h0, c0)

    def test_basic_gru(self):
        h0 = np.random.rand(1, 24, 200).astype(np.float32)
        input = np.random.rand(32, 24, 100).astype(np.float32)

        t_rnn = tnn.GRU(100, 200)
        j_rnn = nn.GRU(100, 200)
        check_equal_1(t_rnn, j_rnn, input, h0)

    def test_multilayer_gru(self):
        h0 = np.random.rand(4, 4, 200).astype(np.float32)
        input = np.random.rand(5, 4, 100).astype(np.float32)

        t_rnn = tnn.GRU(100, 200, num_layers=4)
        j_rnn = nn.GRU(100, 200, num_layers=4)
        check_equal_1(t_rnn, j_rnn, input, h0)

    def test_bidirectional_gru(self):
        h0 = np.random.rand(2, 1, 200).astype(np.float32)
        input = np.random.rand(5, 1, 100).astype(np.float32)

        t_rnn = tnn.GRU(100, 200, bidirectional=True)
        j_rnn = nn.GRU(100, 200, bidirectional=True)
        check_equal_1(t_rnn, j_rnn, input, h0)

    def test_multilayer_bidirectional_gru(self):
        h0 = np.random.rand(4, 4, 200).astype(np.float32)
        input = np.random.rand(5, 4, 100).astype(np.float32)

        t_rnn = tnn.GRU(100, 200, num_layers=2, bidirectional=True, bias=False)
        j_rnn = nn.GRU(100, 200, num_layers=2, bidirectional=True, bias=False)
        check_equal_1(t_rnn, j_rnn, input, h0)

    def test_multilayer_dropout_gru(self):
        h0 = np.random.rand(2, 4, 200).astype(np.float32)
        input = np.random.rand(5, 4, 100).astype(np.float32)

        t_rnn = tnn.GRU(100, 200, num_layers=2, dropout=0.5, bias=False)
        j_rnn = nn.GRU(100, 200, num_layers=2, dropout=0.5, bias=False)
        t_rnn.eval()
        j_rnn.eval()
        check_equal_1(t_rnn, j_rnn, input, h0)

    def test_rnn_default_hx(self):
        input = np.random.rand(32, 24, 12).astype(np.float32)
        h0 = np.zeros((1, 24, 24)).astype(np.float32)

        t_rnn = tnn.RNN(12, 24)
        j_rnn = nn.RNN(12, 24)
        j_rnn.load_state_dict(t_rnn.state_dict())
        t_output, th = t_rnn(torch.from_numpy(input))
        j_output, jh = j_rnn(jt.array(input))

        np.testing.assert_allclose(t_output.detach().cpu().numpy(), j_output.data, rtol=1e-03, atol=1e-06)
        np.testing.assert_allclose(th.detach().cpu().numpy(), jh.data, rtol=1e-03, atol=1e-06)

    def test_lstm_default_hx(self):
        input = np.random.rand(32, 24, 10).astype(np.float32)
        t_rnn = tnn.LSTM(10, 20, num_layers=2, bidirectional=True)
        j_rnn = nn.LSTM(10, 20, num_layers=2, bidirectional=True)
        j_rnn.load_state_dict(t_rnn.state_dict())
        t_output, (th, tc) = t_rnn(torch.from_numpy(input))
        j_output, (jh, jc) = j_rnn(jt.array(input))
        np.testing.assert_allclose(t_output.detach().cpu().numpy(), j_output.data, rtol=1e-03, atol=1e-06)
        np.testing.assert_allclose(th.detach().cpu().numpy(), jh.data, rtol=1e-03, atol=1e-06)
        np.testing.assert_allclose(tc.detach().cpu().numpy(), jc.data, rtol=1e-03, atol=1e-06)

    def test_twobilinear_lstm(self):
        x = jt.rand(5, 4, 10)
        rnn1 = nn.LSTM(10, 20, bidirectional=True)
        out1, _ = rnn1(x)
        rnn2 = nn.LSTM(40, 20, bidirectional=True)
        out2, _ = rnn2(out1)
        target = jt.zeros_like(out2)
        loss = nn.mse_loss(out2, target)

        from jittor import optim
        optimizer = optim.RMSprop(rnn1.parameters())
        optimizer.step(loss)

    @skipIf(not jt.has_cuda, "No Cuda found")
    @jt.flag_scope(use_cuda=1)
    def test_cudnn_rnn(self):
        dev = torch.device('cuda:0')
        t_rnn = tnn.RNN(100, 200, nonlinearity='relu').to(dev)

        j_rnn = nn.RNN(100, 200, nonlinearity='relu')
        j_rnn.train()
        j_rnn.load_state_dict(t_rnn.state_dict())

        h0 = np.random.rand(1, 24, 200).astype(np.float32)
        input = np.random.rand(32, 24, 100).astype(np.float32)

        t_output, th = t_rnn(torch.from_numpy(input).to(dev), 
                             torch.from_numpy(h0).to(dev))
        
        j_output, jh = j_rnn(jt.array(input), jt.array(h0))

        np.testing.assert_allclose(j_output.data, t_output.detach().cpu().numpy())
        np.testing.assert_allclose(jh.data, th.detach().cpu().numpy())

    @skipIf(not jt.has_cuda, "No Cuda found")
    @jt.flag_scope(use_cuda=1)
    def test_cudnn_rnn_train(self):
        dev = torch.device('cuda:0')
        t_rnn = tnn.RNN(32, 64, nonlinearity='relu').to(dev)
        t_optim = torch.optim.SGD(t_rnn.parameters(), lr=1e-3, momentum=0.9)

        j_rnn = nn.RNN(32, 64, nonlinearity='relu')
        j_rnn.load_state_dict(t_rnn.state_dict())
        j_optim = nn.SGD(j_rnn.parameters(), lr=1e-3, momentum=0.9)

        h0 = np.random.rand(1, 4, 64).astype(np.float32)
        input = np.random.rand(12, 4, 32).astype(np.float32)

        for _ in range(10):
            t_optim.zero_grad()
            t_output, th = t_rnn(torch.from_numpy(input).to(dev), torch.from_numpy(h0).to(dev))
            t_loss = (t_output ** 2).sum() + (th ** 2).sum()
            t_loss.backward()
            t_optim.step()

            j_input, jh = jt.array(input), jt.array(h0)
            j_output, jh = j_rnn(j_input, jh)
            j_loss = (j_output ** 2).sum() + (jh ** 2).sum()
            j_optim.step(j_loss)

            np.testing.assert_allclose(t_loss.item(), j_loss.item(), rtol=1e-2)
            np.testing.assert_allclose(t_rnn.bias_hh_l0.detach().cpu().numpy(), j_rnn.bias_hh_l0.data, atol=1e-3, rtol=1e-2)

    @unittest.skipIf(not jt.has_cuda, "No Cuda found")
    @unittest.skipIf(not jt.cudnn, "No Cudnn found")
    @jt.flag_scope(use_cuda=1)
    def test_basic_cudnn_rnn(self):
        dev = torch.device('cuda:0')
        t_rnn = tnn.RNN(100, 200, nonlinearity='relu').to(dev)
        j_rnn = nn.RNN(100, 200, nonlinearity='relu')

        h0 = np.random.rand(1, 24, 200).astype(np.float32)
        input = np.random.rand(32, 24, 100).astype(np.float32)
        check_equal_1(t_rnn, j_rnn, input, h0, dev)

    @unittest.skipIf(not jt.has_cuda, "No Cuda found")
    @unittest.skipIf(not jt.cudnn, "No Cudnn found")
    @jt.flag_scope(use_cuda=1)
    def test_multilayer_cudnn_rnn(self):
        dev = torch.device('cuda:0')
        t_rnn = tnn.RNN(100, 200, num_layers=4, nonlinearity='tanh').to(dev)
        j_rnn = nn.RNN(100, 200, num_layers=4, nonlinearity='tanh')

        h0 = np.random.rand(4, 8, 200).astype(np.float32)
        input = np.random.rand(5, 8, 100).astype(np.float32)
        check_equal_1(t_rnn, j_rnn, input, h0, dev)

    @unittest.skipIf(not jt.has_cuda, "No Cuda found")
    @unittest.skipIf(not jt.cudnn, "No Cudnn found")
    @jt.flag_scope(use_cuda=1)
    def test_bidirectional_cudnn_rnn(self):
        dev = torch.device('cuda:0')
        t_rnn = tnn.RNN(100, 200, bidirectional=True, nonlinearity='tanh').to(dev)
        j_rnn = nn.RNN(100, 200, bidirectional=True, nonlinearity='tanh')

        h0 = np.random.rand(2, 8, 200).astype(np.float32)
        input = np.random.rand(5, 8, 100).astype(np.float32)
        check_equal_1(t_rnn, j_rnn, input, h0, dev)

    @unittest.skipIf(not jt.has_cuda, "No Cuda found")
    @unittest.skipIf(not jt.cudnn, "No Cudnn found")
    @jt.flag_scope(use_cuda=1)
    def test_no_bias_cudnn_rnn(self):
        dev = torch.device('cuda:0')
        t_rnn = tnn.RNN(100, 200, bidirectional=True, bias=False, nonlinearity='tanh').to(dev)
        j_rnn = nn.RNN(100, 200, bidirectional=True, bias=False, nonlinearity='tanh')

        h0 = np.random.rand(2, 8, 200).astype(np.float32)
        input = np.random.rand(5, 8, 100).astype(np.float32)
        check_equal_1(t_rnn, j_rnn, input, h0, dev)

    @unittest.skipIf(not jt.has_cuda, "No Cuda found")
    @unittest.skipIf(not jt.cudnn, "No Cudnn found")
    @jt.flag_scope(use_cuda=1)
    def test_dropout_cudnn_rnn(self):
        dev = torch.device('cuda:0')
        t_rnn = tnn.RNN(100, 200, num_layers=2, dropout=0.5, nonlinearity='tanh').to(dev)
        j_rnn = nn.RNN(100, 200, num_layers=2, dropout=0.5, nonlinearity='tanh')
        t_rnn.eval()
        j_rnn.eval()

        h0 = np.random.rand(2, 8, 200).astype(np.float32)
        input = np.random.rand(5, 8, 100).astype(np.float32)
        check_equal_1(t_rnn, j_rnn, input, h0, dev)

    @unittest.skipIf(not jt.has_cuda, "No Cuda found")
    @unittest.skipIf(not jt.cudnn, "No Cudnn found")
    @jt.flag_scope(use_cuda=1)
    def test_basic_lstm_rnn(self):
        dev = torch.device('cuda:0')
        t_rnn = tnn.LSTM(100, 200).to(dev)
        j_rnn = nn.LSTM(100, 200)

        h0 = np.random.rand(1, 24, 200).astype(np.float32)
        c0 = np.random.rand(1, 24, 200).astype(np.float32)
        input = np.random.rand(32, 24, 100).astype(np.float32)
        check_equal_2(t_rnn, j_rnn, input, h0, c0, dev)

    @unittest.skipIf(not jt.has_cuda, "No Cuda found")
    @unittest.skipIf(not jt.cudnn, "No Cudnn found")
    @jt.flag_scope(use_cuda=1)
    def test_cudnn_rnn_train(self):
        dev = torch.device('cuda:0')
        t_rnn = tnn.RNN(32, 64, nonlinearity='relu').to(dev)
        t_optim = torch.optim.SGD(t_rnn.parameters(), lr=1e-3, momentum=0.9)

        j_rnn = nn.RNN(32, 64, nonlinearity='relu')
        j_rnn.load_state_dict(t_rnn.state_dict())
        j_optim = nn.SGD(j_rnn.parameters(), lr=1e-3, momentum=0.9)

        h0 = np.random.rand(1, 4, 64).astype(np.float32)
        input = np.random.rand(12, 4, 32).astype(np.float32)

        for _ in range(10):
            t_optim.zero_grad()
            t_output, th = t_rnn(torch.from_numpy(input).to(dev), torch.from_numpy(h0).to(dev))
            t_loss = (t_output ** 2).sum() + (th ** 2).sum()
            t_loss.backward()
            t_optim.step()

            j_input, jh = jt.array(input), jt.array(h0)
            j_output, jh = j_rnn(j_input, jh)
            j_loss = (j_output ** 2).sum() + (jh ** 2).sum()
            j_optim.step(j_loss)

            np.testing.assert_allclose(t_loss.item(), j_loss.item(), rtol=1e-4)
            np.testing.assert_allclose(t_rnn.bias_hh_l0.detach().cpu().numpy(), j_rnn.bias_hh_l0.data, atol=1e-4, rtol=1e-4)

    @unittest.skipIf(not jt.has_cuda, "No Cuda found")
    @unittest.skipIf(not jt.cudnn, "No Cudnn found")
    @jt.flag_scope(use_cuda=1)
    def test_cudnn_gru_train(self):
        dev = torch.device('cuda:0')
        t_rnn = tnn.GRU(32, 64).to(dev)
        t_optim = torch.optim.SGD(t_rnn.parameters(), lr=1e-3, momentum=0.9)

        j_rnn = nn.GRU(32, 64)
        j_rnn.load_state_dict(t_rnn.state_dict())
        j_optim = nn.SGD(j_rnn.parameters(), lr=1e-3, momentum=0.9)

        h0 = np.random.rand(1, 4, 64).astype(np.float32)
        input = np.random.rand(12, 4, 32).astype(np.float32)

        for _ in range(10):
            t_optim.zero_grad()
            t_output, th = t_rnn(torch.from_numpy(input).to(dev), torch.from_numpy(h0).to(dev))
            t_loss = (t_output ** 2).sum() + (th ** 2).sum()
            t_loss.backward()
            t_optim.step()

            j_input, jh = jt.array(input), jt.array(h0)
            j_output, jh = j_rnn(j_input, jh)
            j_loss = (j_output ** 2).sum() + (jh ** 2).sum()
            j_optim.step(j_loss)

            np.testing.assert_allclose(t_loss.item(), j_loss.item(), rtol=1e-4)
            np.testing.assert_allclose(t_rnn.bias_hh_l0.detach().cpu().numpy(), j_rnn.bias_hh_l0.data, atol=1e-4, rtol=1e-4)

    @unittest.skipIf(not jt.has_cuda, "No Cuda found")
    @unittest.skipIf(not jt.cudnn, "No Cudnn found")
    @jt.flag_scope(use_cuda=1)
    def test_cudnn_lstm_train(self):
        dev = torch.device('cuda:0')
        t_rnn = tnn.LSTM(32, 64).to(dev)
        t_optim = torch.optim.SGD(t_rnn.parameters(), lr=1e-3, momentum=0.9)

        j_rnn = nn.LSTM(32, 64)
        j_rnn.load_state_dict(t_rnn.state_dict())
        j_optim = nn.SGD(j_rnn.parameters(), lr=1e-3, momentum=0.9)

        h0 = np.random.rand(1, 4, 64).astype(np.float32)
        c0 = np.random.rand(1, 4, 64).astype(np.float32)
        input = np.random.rand(12, 4, 32).astype(np.float32)

        for _ in range(10):
            t_optim.zero_grad()
            t_output, (th, tc) = t_rnn(torch.from_numpy(input).to(dev), 
                (torch.from_numpy(h0).to(dev), torch.from_numpy(c0).to(dev)))
            t_loss = (t_output ** 2).sum() + (th ** 2).sum() + (tc ** 2).sum()
            t_loss.backward()
            t_optim.step()

            j_input, jh0, jc0 = jt.array(input), jt.array(h0), jt.array(c0)
            j_output, (jh, jc) = j_rnn(j_input, (jh0, jc0))
            j_loss = (j_output ** 2).sum() + (jh ** 2).sum() + (jc ** 2).sum()
            j_optim.step(j_loss)

            np.testing.assert_allclose(t_loss.item(), j_loss.item(), rtol=1e-4)
            np.testing.assert_allclose(t_rnn.bias_hh_l0.detach().cpu().numpy(), j_rnn.bias_hh_l0.data, atol=1e-4, rtol=1e-4)

    @unittest.skipIf(not jt.has_cuda, "No Cuda found")
    @unittest.skipIf(not jt.cudnn, "No Cudnn found")
    @jt.flag_scope(use_cuda=1)
    def test_multilayer_bidirectional_cudnn_lstm_train(self):
        dev = torch.device('cuda:0')
        t_rnn = tnn.LSTM(32, 64, num_layers=4, bidirectional=True).to(dev)
        t_optim = torch.optim.SGD(t_rnn.parameters(), lr=1e-3, momentum=0.9)

        j_rnn = nn.LSTM(32, 64, num_layers=4, bidirectional=True)
        j_rnn.load_state_dict(t_rnn.state_dict())
        j_optim = nn.SGD(j_rnn.parameters(), lr=1e-3, momentum=0.9)

        h0 = np.random.rand(8, 4, 64).astype(np.float32)
        c0 = np.random.rand(8, 4, 64).astype(np.float32)
        input = np.random.rand(12, 4, 32).astype(np.float32)

        for _ in range(10):
            t_optim.zero_grad()
            t_output, (th, tc) = t_rnn(torch.from_numpy(input).to(dev), 
                (torch.from_numpy(h0).to(dev), torch.from_numpy(c0).to(dev)))
            t_loss = (t_output ** 2).sum() + (th ** 2).sum() + (tc ** 2).sum()
            t_loss.backward()
            t_optim.step()

            j_input, jh0, jc0 = jt.array(input), jt.array(h0), jt.array(c0)
            j_output, (jh, jc) = j_rnn(j_input, (jh0, jc0))
            j_loss = (j_output ** 2).sum() + (jh ** 2).sum() + (jc ** 2).sum()
            j_optim.step(j_loss)

            np.testing.assert_allclose(t_loss.item(), j_loss.item(), rtol=1e-4)
            np.testing.assert_allclose(t_rnn.bias_hh_l0.detach().cpu().numpy(), j_rnn.bias_hh_l0.data, atol=1e-4, rtol=1e-4)

    @unittest.skipIf(not jt.has_cuda, "No Cuda found")
    @jt.flag_scope(use_cuda=1)
    def test_cudnn_rnn_speed(self):
        from time import time
        iters = 100

        h0 = np.random.rand(1, 128, 256).astype(np.float32)
        input = np.random.rand(128, 128, 128).astype(np.float32)

        dev = torch.device('cuda:0')
        t_rnn = tnn.RNN(128, 256, nonlinearity='relu').to(dev)
        t_optim = torch.optim.SGD(t_rnn.parameters(), lr=1e-3, momentum=0.9)

        t_input = torch.from_numpy(input).to(dev)
        t_h0 = torch.from_numpy(h0).to(dev)

        start_time = time()
        for i in range(iters):
            t_optim.zero_grad()
            t_output, th = t_rnn(t_input, t_h0)
            t_loss = (t_output ** 2).sum() + (th ** 2).sum()
            t_loss.backward()
            t_optim.step()
        print('torch time = ', time() - start_time)        

        j_rnn = nn.RNN(128, 256, nonlinearity='relu')
        j_rnn.load_state_dict(t_rnn.state_dict())
        j_optim = nn.SGD(j_rnn.parameters(), lr=1e-3, momentum=0.9)
        j_input, j_h0 = jt.array(input), jt.array(h0)
        
        start_time = time()
        for i in range(iters):
            j_output, jh = j_rnn(j_input, j_h0)
            j_loss = (j_output ** 2).sum() + (jh ** 2).sum()
            j_optim.step(j_loss)
        jt.sync_all(True)
        print('jittor Cudnn time = ', time() - start_time)

        jt_cudnn, jt.cudnn = jt.cudnn, None
        j_rnn = nn.RNN(128, 256, nonlinearity='relu')
        j_rnn.load_state_dict(t_rnn.state_dict())
        j_optim = nn.SGD(j_rnn.parameters(), lr=1e-3, momentum=0.9)
        start_time = time()
        for i in range(iters):
            j_output, jh = j_rnn(j_input, j_h0)
            j_loss = (j_output ** 2).sum() + (jh ** 2).sum()
            j_optim.step(j_loss)
        jt.sync_all(True)
        print('jittor native time = ', time() - start_time)
        jt.cudnn = jt_cudnn


if __name__ == "__main__":
    unittest.main()