"""Stable sequence-packing compatibility objects."""

import builtins as _builtins_rnn
import collections as _collections_rnn
import jittor as _jt

def _rnn_lengths_to_list(lengths):
    if isinstance(lengths, _jt.Var):
        lengths = lengths.numpy()
    if hasattr(lengths, "tolist"):
        lengths = lengths.tolist()
    if isinstance(lengths, (_builtins_rnn.int, _builtins_rnn.float)):
        lengths = [lengths]
    return [_builtins_rnn.int(x) for x in list(lengths)]

def _rnn_index_tensor(x, order, batch_first):
    order = _rnn_lengths_to_list(order)
    if not order:
        return x
    if batch_first:
        return _jt.stack([x[i] for i in order], dim=0)
    return _jt.stack([x[:, i] for i in order], dim=1)

def _rnn_pad_sequence(sequences, batch_first=False, padding_value=0.0):
    seqs = list(sequences)
    if not seqs:
        raise ValueError("pad_sequence expects a non-empty sequence list")
    max_len = _builtins_rnn.max(_builtins_rnn.int(s.shape[0]) for s in seqs)
    trailing = tuple(seqs[0].shape[1:])
    padded = []
    for s in seqs:
        pad_len = max_len - _builtins_rnn.int(s.shape[0])
        if pad_len > 0:
            pad = _jt.ones((pad_len,) + trailing, dtype=s.dtype) * padding_value
            s = _jt.concat([s, pad], dim=0)
        padded.append(s)
    out = _jt.stack(padded, dim=0)
    return out if batch_first else out.transpose(0, 1)

_PackedSequenceBase = _collections_rnn.namedtuple(
    "PackedSequence", ("data", "batch_sizes", "sorted_indices", "unsorted_indices"))

class PackedSequence(_PackedSequenceBase):
    __slots__ = ()

    def __new__(cls, data, batch_sizes=None, sorted_indices=None, unsorted_indices=None):
        return _PackedSequenceBase.__new__(cls, data, batch_sizes, sorted_indices, unsorted_indices)

    def to(self, *args, **kwargs):
        data = self.data.to(*args, **kwargs) if hasattr(self.data, "to") else self.data
        return type(self)(data, self.batch_sizes, self.sorted_indices, self.unsorted_indices)

    cuda = to
    cpu = to

def pack_padded_sequence(input, lengths, batch_first=False, enforce_sorted=True):
    lengths_list = _rnn_lengths_to_list(lengths)
    if not enforce_sorted:
        order = sorted(range(len(lengths_list)), key=lambda i: lengths_list[i], reverse=True)
        unsorted = [0] * len(order)
        for sorted_pos, original_pos in enumerate(order):
            unsorted[original_pos] = sorted_pos
        input = _rnn_index_tensor(input, order, batch_first)
        lengths_list = [lengths_list[i] for i in order]
        sorted_indices = _jt.array(order).int64()
        unsorted_indices = _jt.array(unsorted).int64()
    else:
        sorted_indices = None
        unsorted_indices = None

    max_len = lengths_list[0] if lengths_list else 0
    pieces = []
    batch_sizes = []
    for t in range(max_len):
        active = _builtins_rnn.sum(1 for n in lengths_list if n > t)
        if active <= 0:
            break
        batch_sizes.append(active)
        if batch_first:
            pieces.append(input[:active, t])
        else:
            pieces.append(input[t, :active])
    if pieces:
        data = _jt.concat(pieces, dim=0)
    else:
        trailing = tuple(input.shape[2:])
        data = _jt.ones((0,) + trailing, dtype=input.dtype)
    return PackedSequence(data, _jt.array(batch_sizes).int64(), sorted_indices, unsorted_indices)

def pad_packed_sequence(sequence, batch_first=False, padding_value=0.0, total_length=None):
    if not isinstance(sequence, PackedSequence):
        return sequence, None
    batch_sizes = _rnn_lengths_to_list(sequence.batch_sizes)
    max_len = len(batch_sizes)
    batch_size = _builtins_rnn.max(batch_sizes) if batch_sizes else 0
    data = sequence.data
    trailing = tuple(data.shape[1:])
    steps = []
    offset = 0
    for active in batch_sizes:
        step = data[offset:offset + active]
        offset += active
        if active < batch_size:
            pad = _jt.ones((batch_size - active,) + trailing, dtype=data.dtype) * padding_value
            step = _jt.concat([step, pad], dim=0)
        steps.append(step)
    if steps:
        out = _jt.stack(steps, dim=0)
    else:
        out = _jt.ones((0, batch_size) + trailing, dtype=data.dtype) * padding_value
    if total_length is not None:
        total_length = _builtins_rnn.int(total_length)
        if total_length < max_len:
            raise ValueError("total_length must be at least the packed sequence length")
        if total_length > max_len:
            pad = _jt.ones((total_length - max_len, batch_size) + trailing, dtype=data.dtype) * padding_value
            out = _jt.concat([out, pad], dim=0)
    lengths_list = [_builtins_rnn.sum(1 for n in batch_sizes if n > i) for i in range(batch_size)]
    if sequence.unsorted_indices is not None:
        out = _rnn_index_tensor(out, sequence.unsorted_indices, batch_first=False)
        order = _rnn_lengths_to_list(sequence.unsorted_indices)
        lengths_list = [lengths_list[i] for i in order]
    if batch_first:
        out = out.transpose(0, 1)
    return out, _jt.array(lengths_list).int64()
