"""Cell-level recurrent neural network implementations."""

import math

import jittor as jt


class LSTMCell(jt.Module):
    ''' A long short-term memory (LSTM) cell.

    :param input_size: The number of expected features in the input
    :type input_size: int

    :param hidden_size: The number of features in the hidden state
    :type hidden_size: int

    :param bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True.
    :type bias: bool, optional

    Example:

    >>> rnn = nn.LSTMCell(10, 20) # (input_size, hidden_size)
    >>> input = jt.randn(2, 3, 10) # (time_steps, batch, input_size)
    >>> hx = jt.randn(3, 20) # (batch, hidden_size)
    >>> cx = jt.randn(3, 20)
    >>> output = []
    >>> for i in range(input.shape[0]):
            hx, cx = rnn(input[i], (hx, cx))
            output.append(hx)
    >>> output = jt.stack(output, dim=0)
    '''
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()

        self.hidden_size = hidden_size
        self.bias = bias

        k = math.sqrt(1 / hidden_size)
        self.weight_ih = jt.nn.init.uniform((4 * hidden_size, input_size), 'float32', -k, k)
        self.weight_hh = jt.nn.init.uniform((4 * hidden_size, hidden_size), 'float32', -k, k)

        if bias:
            self.bias_ih = jt.nn.init.uniform((4 * hidden_size,), 'float32', -k, k)
            self.bias_hh = jt.nn.init.uniform((4 * hidden_size,), 'float32', -k, k)

    def execute(self, input, hx = None):
        if hx is None:
            zeros = jt.zeros((input.shape[0], self.hidden_size), dtype=input.dtype)
            h, c = zeros, zeros
        else:
            h, c = hx

        y = jt.nn.matmul_transpose(input, self.weight_ih) + jt.nn.matmul_transpose(h, self.weight_hh)

        if self.bias:
            y = y + self.bias_ih + self.bias_hh

        i = y[:, :self.hidden_size].sigmoid()
        f = y[:, self.hidden_size : 2 * self.hidden_size].sigmoid()
        g = y[:, 2 * self.hidden_size : 3 * self.hidden_size].tanh()
        o = y[:, 3 * self.hidden_size:].sigmoid()

        c = f * c + i * g
        h = o * c.tanh()

        return h, c


class RNNCell(jt.Module):
    ''' An Elman RNN cell with tanh or ReLU non-linearity.

    :param input_size: The number of expected features in the input
    :type input_size: int

    :param hidden_size: The number of features in the hidden state
    :type hidden_size: int

    :param bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True.
    :type bias: bool, optional

    :param nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'. Default: 'tanh'.
    :type nonlinearity: str, optional

    Example:

    >>> rnn = nn.RNNCell(10, 20)
    >>> input = jt.randn((6, 3, 10))
    >>> hx = jt.randn((3, 20))
    >>> output = []
    >>> for i in range(6):
            hx = rnn(input[i], hx)
            output.append(hx)
    '''
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity = "tanh"):
        super().__init__()

        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity

        k = math.sqrt(1 / hidden_size)
        self.weight_ih = jt.nn.init.uniform((hidden_size, input_size), 'float32', -k, k)
        self.weight_hh = jt.nn.init.uniform((hidden_size, hidden_size), 'float32', -k, k)

        if bias:
            self.bias_ih = jt.nn.init.uniform((hidden_size,), 'float32', -k, k)
            self.bias_hh = jt.nn.init.uniform((hidden_size,), 'float32', -k, k)

    def execute(self, input, hx = None):
        if hx is None:
            hx = jt.zeros((input.shape[0], self.hidden_size), dtype=input.dtype)

        y = jt.nn.matmul_transpose(input, self.weight_ih)+jt.nn.matmul_transpose(hx, self.weight_hh)

        if self.bias:
            y= y + self.bias_ih + self.bias_hh

        if self.nonlinearity == 'tanh':
            y = y.tanh()
        elif self.nonlinearity == 'relu':
            y = jt.nn.relu(y)
        else:
            raise RuntimeError("Unknown nonlinearity: {}".format(self.nonlinearity))

        return y


class GRUCell(jt.Module):
    ''' A gated recurrent unit (GRU) cell.

    :param input_size: The number of expected features in the input
    :type input_size: int

    :param hidden_size: The number of features in the hidden state
    :type hidden_size: int

    :param bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True.
    :type bias: bool, optional

    Example:

    >>> rnn = nn.GRUCell(10, 20)
    >>> input = jt.randn((6, 3, 10))
    >>> hx = jt.randn((3, 20))
    >>> output = []
    >>> for i in range(6):
            hx = rnn(input[i], hx)
            output.append(hx)
    '''
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()

        self.hidden_size = hidden_size
        self.bias = bias

        k = math.sqrt(1 / hidden_size)
        self.weight_ih = jt.nn.init.uniform((3*hidden_size, input_size), 'float32', -k, k)
        self.weight_hh = jt.nn.init.uniform((3*hidden_size, hidden_size), 'float32', -k, k)

        if bias:
            self.bias_ih = jt.nn.init.uniform((3*hidden_size,), 'float32', -k, k)
            self.bias_hh = jt.nn.init.uniform((3*hidden_size,), 'float32', -k, k)

    def execute(self, input, hx = None):
        if hx is None:
            hx = jt.zeros((input.shape[0], self.hidden_size), dtype=input.dtype)

        gi = jt.nn.matmul_transpose(input, self.weight_ih)
        gh = jt.nn.matmul_transpose(hx, self.weight_hh)

        if self.bias:
            gi += self.bias_ih
            gh += self.bias_hh

        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)

        resetgate = jt.sigmoid(i_r + h_r)
        inputgate = jt.sigmoid(i_i + h_i)
        newgate = jt.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hx - newgate)
        return hy
