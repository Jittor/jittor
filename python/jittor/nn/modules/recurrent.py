"""Sequence-level RNN, LSTM, and GRU implementations."""

from .recurrent_base import RNNBase
import jittor as jt


class RNN(RNNBase):
    ''' Applies a multi-layer Elman RNN with tanh ReLU non-linearity to an input sequence.

    :param input_size: The number of expected features in the input.
    :type input_size: int

    :param hidden_size: The number of features in the hidden state.
    :type hidden_size: int

    :param num_layers: Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two RNNs together to form a stacked RNN, with the second RNN taking in outputs of the first RNN and computing the final results. Default: 1
    :type num_layers: int, optinal

    :param nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'. Default: 'tanh'
    :type nonlinearity: str, optional

    :param bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True.
    :type bias: bool, optional

    :param batch_first: If True, then the input and output tensors are provided as (batch, seq, feature). Default: False
    :type bias: bool, optional

    :param dropout: If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer, with dropout probability equal to dropout. Default: 0
    :type dropout: float, optional

    :param bidirectional: If True, becomes a bidirectional RNN. Default: False
    :type bidirectional: bool, optional

    Example:
        >>> rnn = nn.RNN(10, 20, 2)
        >>> input = jt.randn(5, 3, 10)
        >>> h0 = jt.randn(2, 3, 20)
        >>> output, hn = rnn(input, h0)
    '''
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
        nonlinearity: str = 'tanh', bias: bool = True, batch_first: bool = False,
        dropout: float = 0, bidirectional: bool = False) -> None:
        super().__init__('RNN', input_size, hidden_size, num_layers=num_layers,
            bias=bias, batch_first=batch_first, dropout=dropout,
            bidirectional=bidirectional)

        if not nonlinearity in ['tanh', 'relu']:
            raise ValueError('Unrecognized nonlinearity: ' + nonlinearity)
        self.nonlinearity = nonlinearity

    def call_rnn_cell(self, input, hidden, suffix):
        y = jt.nn.matmul_transpose(input, getattr(self, f'weight_ih_{suffix}'))
        y = y + jt.nn.matmul_transpose(hidden, getattr(self, f'weight_hh_{suffix}'))

        if self.bias:
            y = y + getattr(self, f'bias_ih_{suffix}') + getattr(self, f'bias_hh_{suffix}')

        if self.nonlinearity == 'tanh':
            h = jt.tanh(y)
        else:
            h = jt.nn.relu(y)

        return h, h


class LSTM(RNNBase):
    ''' Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

    :param input_size: The number of expected features in the input.
    :type input_size: int

    :param hidden_size: The number of features in the hidden state.
    :type hidden_size: int

    :param num_layers: Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results. Default: 1
    :type num_layers: int, optinal

    :param bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True.
    :type bias: bool, optional

    :param batch_first: If True, then the input and output tensors are provided as (batch, seq, feature). Default: False
    :type bias: bool, optional

    :param dropout: If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to dropout. Default: 0
    :type dropout: float, optional

    :param bidirectional: If True, becomes a bidirectional LSTM. Default: False
    :type bidirectional: bool, optional

    :param proj_size: If > 0, will use LSTM with projections of corresponding size. Default: 0
    :type proj_size: int, optional

    Example:
        >>> rnn = nn.LSTM(10, 20, 2)
        >>> input = jt.randn(5, 3, 10)
        >>> h0 = jt.randn(2, 3, 20)
        >>> c0 = jt.randn(2, 3, 20)
        >>> output, (hn, cn) = rnn(input, (h0, c0))
    '''

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
            batch_first=False, dropout=0, bidirectional=False, proj_size=0):
        super().__init__('LSTM', input_size, hidden_size, num_layers=num_layers,
            bias=bias, batch_first=batch_first, dropout=dropout,
            bidirectional=bidirectional, proj_size=proj_size)

    def call_rnn_cell(self, input, hidden, suffix):
        h, c = hidden
        y = jt.nn.matmul_transpose(input, getattr(self, f'weight_ih_{suffix}'))
        y = y + jt.nn.matmul_transpose(h, getattr(self, f'weight_hh_{suffix}'))

        if self.bias:
            y = y + getattr(self, f'bias_ih_{suffix}') + getattr(self, f'bias_hh_{suffix}')

        i = y[:, :self.hidden_size].sigmoid()
        f = y[:, self.hidden_size : 2 * self.hidden_size].sigmoid()
        g = y[:, 2 * self.hidden_size : 3 * self.hidden_size].tanh()
        o = y[:, 3 * self.hidden_size:].sigmoid()
        c = f * c + i * g
        h = o * c.tanh()

        if self.proj_size > 0:
            h = jt.nn.matmul_transpose(h, getattr(self, f'weight_hr_{suffix}'))

        return h, (h, c)


class GRU(RNNBase):
    ''' Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.

    :param input_size: The number of expected features in the input.
    :type input_size: int

    :param hidden_size: The number of features in the hidden state.
    :type hidden_size: int

    :param num_layers: Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two GRUs together to form a stacked GRU, with the second GRU taking in outputs of the first GRU and computing the final results. Default: 1
    :type num_layers: int, optinal

    :param bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True.
    :type bias: bool, optional

    :param batch_first: If True, then the input and output tensors are provided as (batch, seq, feature). Default: False
    :type bias: bool, optional

    :param dropout: If non-zero, introduces a Dropout layer on the outputs of each GRU layer except the last layer, with dropout probability equal to dropout. Default: 0
    :type dropout: float, optional

    :param bidirectional: If True, becomes a bidirectional GRU. Default: False
    :type bidirectional: bool, optional

    Example:
        >>> rnn = nn.GRU(10, 20, 2)
        >>> input = jt.randn(5, 3, 10)
        >>> h0 = jt.randn(2, 3, 20)
        >>> output, hn = rnn(input, h0)
    '''

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
        bias: bool = True, batch_first: bool = False, dropout: float = 0,
        bidirectional: bool = False) -> None:
        super().__init__('GRU', input_size, hidden_size, num_layers=num_layers,
            bias=bias, batch_first=batch_first, dropout=dropout,
            bidirectional=bidirectional)

    def call_rnn_cell(self, input, hidden, suffix):
        ih = jt.nn.matmul_transpose(input, getattr(self, f'weight_ih_{suffix}'))
        hh = jt.nn.matmul_transpose(hidden, getattr(self, f'weight_hh_{suffix}'))

        if self.bias:
            ih = ih + getattr(self, f'bias_ih_{suffix}')
            hh = hh + getattr(self, f'bias_hh_{suffix}')

        hs = self.hidden_size
        r = (ih[:, :hs] + hh[:, :hs]).sigmoid()
        z = (ih[:, hs: 2 * hs] + hh[:, hs: 2 * hs]).sigmoid()
        n = (ih[:, 2 * hs:] + r * hh[:, 2 * hs:]).tanh()
        h = (1 - z) * n + z * hidden

        return h, h
