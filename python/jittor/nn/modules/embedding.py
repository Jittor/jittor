"""Stateful embedding modules exposed through :mod:`jittor.nn`."""

import jittor as jt


class Embedding(jt.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        padding_idx=None,
        dtype="float32",
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
        _weight=None,
        _freeze=False,
        device=None,
    ):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx < 0:
                padding_idx += num_embeddings
            if padding_idx < 0 or padding_idx >= num_embeddings:
                raise AssertionError("padding_idx must be within num_embeddings")
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        if dtype is None:
            dtype = "float32"
        elif not isinstance(dtype, str):
            dtype = str(dtype).replace("torch.", "") or "float32"
        if _weight is not None:
            self.weight = _weight if isinstance(_weight, jt.Var) else jt.array(_weight)
        else:
            self.weight = jt.init.gauss([self.num_embeddings, self.embedding_dim], dtype)
            if padding_idx is not None:
                self.weight[padding_idx] = 0
        if _freeze:
            self.weight = self.weight.stop_grad()

    def execute(self, x):
        return jt.nn.embedding(
            x,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

    def reset_parameters(self):
        weight = jt.init.gauss(
            [self.num_embeddings, self.embedding_dim], self.weight.dtype
        )
        if self.padding_idx is not None:
            weight[self.padding_idx] = 0
        self.weight.update(weight)


class EmbeddingBag(jt.Module):
    def __init__(self, num_embeddings, embedding_dim, mode="mean", dtype="float32"):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.mode = mode
        self.weight = jt.init.gauss([num_embeddings, embedding_dim], dtype)

    def execute(self, input, offsets=None, per_sample_weights=None):
        return jt.nn.embedding_bag(
            input,
            self.weight,
            offsets,
            self.mode,
            per_sample_weights,
        )


__all__ = ["Embedding", "EmbeddingBag"]
