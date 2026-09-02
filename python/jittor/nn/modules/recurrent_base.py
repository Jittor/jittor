"""Shared parameter layout, recurrence scheduling, and cuDNN RNN dispatch."""

from abc import abstractmethod
import math

import jittor as jt


class RNNBase(jt.Module):
    def __init__(self, mode: str, input_size: int, hidden_size: int,
            num_layers: int = 1, bias: bool = True, batch_first: bool = False,
            dropout: float = 0, bidirectional: bool = False,
            proj_size: int = 0, nonlinearity: str = None) -> None:
        super().__init__()

        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.proj_size = proj_size
        self.nonlinearity = nonlinearity

        if mode == 'LSTM':
            gate_size = 4 * hidden_size
        elif mode == 'GRU':
            gate_size = 3 * hidden_size
        elif mode == 'RNN':
            gate_size = hidden_size
        else:
            raise ValueError("Unrecognized RNN mode: " + mode)

        num_directions = 1 + bidirectional
        k = math.sqrt(1 / hidden_size)

        def build_unit(name, in_channels, out_channels=None):
            if out_channels is not None:
                shape = (in_channels, out_channels)
            else:
                shape = (in_channels,)
            setattr(self, name, jt.nn.init.uniform(shape, 'float32', -k, k))
            if self.bidirectional:
                setattr(self, name + '_reverse', jt.nn.init.uniform(shape, 'float32', -k, k))

        for layer in range(num_layers):
            if layer == 0:
                build_unit(f'weight_ih_l{layer}', gate_size, input_size)
            else:
                if proj_size > 0:
                    build_unit(f'weight_ih_l{layer}', gate_size, num_directions * proj_size)
                else:
                    build_unit(f'weight_ih_l{layer}', gate_size, num_directions * hidden_size)

            if proj_size > 0:
                build_unit(f'weight_hh_l{layer}', gate_size, proj_size)
                build_unit(f'weight_hr_l{layer}', proj_size, hidden_size)
            else:
                build_unit(f'weight_hh_l{layer}', gate_size, hidden_size)

            if bias:
                build_unit(f'bias_ih_l{layer}', gate_size)
                build_unit(f'bias_hh_l{layer}', gate_size)

    def _cudnn_flatten_weights(self, cudnn_mode):
        def copy_to_flatten_weight(param_name, offset_idx, num_gates):
            def copy_to(param_name, offset_idx, idx):
                cur_offset = self._cudnn_weight_offset[offset_idx]
                param = getattr(self, param_name)
                param = param[self.hidden_size * idx: self.hidden_size * (idx + 1)]
                ft_weight[cur_offset:cur_offset + param.numel()] = param.flatten()

            if self.bias:
                for idx in range(num_gates):
                    copy_to('weight' + param_name, offset_idx + idx * 2, idx)
                    copy_to('bias' + param_name, offset_idx + idx * 2 + 1, idx)
                return num_gates * 2
            else:
                for idx in range(num_gates):
                    copy_to('weight' + param_name, offset_idx + idx, idx)
                return num_gates

        if jt.flags.use_cuda and jt.cudnn and jt.compiler.is_cuda:
            # cuDNN describes the whole RNN with one data type, so the flat
            # weight has to be in the parameters' dtype -- and the offsets
            # cuDNN reports are element indices into a buffer of that dtype,
            # so the query depends on it too. Both were pinned to float32,
            # which is why a half RNN could not be built at all.
            # str(): NanoString does not compare against None, and the flat
            # layout only depends on the dtype's name.
            dtype = str(self.weight_ih_l0.dtype)
            if getattr(self, '_cudnn_weight_dtype', None) != dtype:
                offset_array = jt.cudnn.cudnn_rnn_weight_offset(
                    cudnn_mode,
                    self.input_size,
                    self.hidden_size,
                    self.num_layers,
                    self.proj_size,
                    self.bias,
                    self.bidirectional,
                    dtype
                )
                self._cudnn_weight_size = offset_array[0]
                self._cudnn_weight_offset = offset_array[1:]
                self._cudnn_weight_dtype = dtype

            num_gates = {
                "RNN": 1, "LSTM": 4, "GRU": 3
            }[self.mode]
            ft_weight = jt.zeros(self._cudnn_weight_size, dtype=dtype)

            cnt = 0
            for layer in range(self.num_layers):
                suffix = ''
                cnt += copy_to_flatten_weight(f'_ih_l{layer}' + suffix, cnt, num_gates)
                cnt += copy_to_flatten_weight(f'_hh_l{layer}' + suffix, cnt, num_gates)
                if self.bidirectional:
                    suffix = '_reverse'
                    cnt += copy_to_flatten_weight(f'_ih_l{layer}' + suffix, cnt, num_gates)
                    cnt += copy_to_flatten_weight(f'_hh_l{layer}' + suffix, cnt, num_gates)
            return ft_weight
        else:
            raise RuntimeError("Not Cudnn found")

    @abstractmethod
    def call_rnn_cell(self, input, hidden, suffix):
        pass

    def call_rnn_sequence(self, input, hidden, suffix):
        if 'reverse' in suffix:
            input = input[::-1]

        output = []
        for s in range(input.shape[0]):
            out, hidden = self.call_rnn_cell(input[s], hidden, suffix)
            output.append(out)

        if 'reverse' in suffix:
            output = output[::-1]
        output = jt.stack(output, dim=0)

        return output, hidden

    def _execute_cudnn_rnn(self, input, hx):
        cudnn_mode = {
            ('RNN', 'tanh'): 'tanh',
            ('RNN', 'relu'): 'relu',
            ('LSTM', None): 'lstm',
            ('GRU', None): 'gru'
        }[(self.mode, self.nonlinearity)]
        ft_weight = self._cudnn_flatten_weights(cudnn_mode)

        if self.mode == 'LSTM':
            ret = jt.cudnn.ops.cudnn_rnn(input, hx[0], hx[1], ft_weight,
                cudnn_mode, self.input_size, self.hidden_size, self.num_layers, 0,
                self.dropout, self.bias, self.bidirectional, self.is_training()
            )
            return ret[0], (ret[1], ret[2])
        else:
            ret = jt.cudnn.ops.cudnn_rnn(input, hx, ft_weight,
                cudnn_mode, self.input_size, self.hidden_size, self.num_layers, 0,
                self.dropout, self.bias, self.bidirectional, self.is_training()
            )
            return ret[0], ret[1]

    def execute(self, input, hx=None):
        if self.batch_first:
            input = input.permute(1, 0, 2)

        num_directions = 2 if self.bidirectional else 1

        if hx is None:
            if self.mode in ['RNN', 'GRU']:
                hx = jt.zeros((num_directions * self.num_layers, input.shape[1], self.hidden_size), dtype=input.dtype)
            elif self.mode == 'LSTM':
                hx = (jt.zeros((num_directions * self.num_layers, input.shape[1], self.hidden_size), dtype=input.dtype),
                      jt.zeros((num_directions * self.num_layers, input.shape[1], self.hidden_size), dtype=input.dtype))

        if jt.flags.use_cuda and jt.cudnn and self.proj_size == 0 and jt.compiler.is_cuda:
            output, hidden_n = self._execute_cudnn_rnn(input, hx)
            # batch_first: the input was permuted to (seq,batch,feat) above; permute the
            # output back to (batch,seq,feat) to match torch (and jittor's own docstring).
            if self.batch_first:
                output = output.permute(1, 0, 2)
            return output, hidden_n
        else:
            hidden_n = []

            for l in range(self.num_layers):
                output = []

                if isinstance(hx, tuple):
                    hidden = [h[l * num_directions] for h in hx]
                else:
                    hidden = hx[l * num_directions]

                output, _hidden = self.call_rnn_sequence(input, hidden, f'l{l}')
                hidden_n.append(_hidden)

                if self.bidirectional:
                    if isinstance(hx, tuple):
                        hidden = [h[l * num_directions + 1] for h in hx]
                    else:
                        hidden = hx[l * num_directions + 1]

                    output_b, _hidden = self.call_rnn_sequence(input, hidden, f'l{l}_reverse')
                    output = jt.concat([output, output_b], dim=-1)
                    hidden_n.append(_hidden)

                if self.dropout > 0:
                    input = jt.nn.dropout(output, p=self.dropout)
                else:
                    input = output

            if isinstance(hx, tuple):
                hidden_n = tuple(jt.stack(hn, dim=0) for hn in zip(*hidden_n))
            else:
                hidden_n = jt.stack(hidden_n, dim=0)

            # batch_first: permute output back to (batch, seq, feat) -- it was computed in
            # (seq, batch, feat) from the permuted input. h_n/c_n stay (layers*dirs, batch,
            # hidden) regardless, matching torch.
            if self.batch_first:
                output = output.permute(1, 0, 2)
            return output, hidden_n
