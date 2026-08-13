"""General tensor algebra operations exposed through :mod:`jittor.nn`."""

import numpy as np

import jittor as jt


# reference: https://github.com/pytorch/pytorch/blob/8ea5b572a63b1acc538a9fc8d3862c73739116e8/torch/functional.py#L1258
def tensordot(a, b, dims=2):
    r"""Returns a contraction of a and b over multiple dimensions.

    :attr:`tensordot` implements a generalized matrix product.

    Args:
      a (Tensor): Left tensor to contract
      b (Tensor): Right tensor to contract
      dims (int or Tuple[List[int], List[int]] or List[List[int]] containing two lists or Tensor): number of dimensions to
         contract or explicit lists of dimensions for :attr:`a` and
         :attr:`b` respectively

    When called with a non-negative integer argument :attr:`dims` = :math:`d`, and
    the number of dimensions of :attr:`a` and :attr:`b` is :math:`m` and :math:`n`,
    respectively, :func:`tensordot` computes

    .. math::
        r_{i_0,...,i_{m-d}, i_d,...,i_n}
          = \\sum_{k_0,...,k_{d-1}} a_{i_0,...,i_{m-d},k_0,...,k_{d-1}} \\times b_{k_0,...,k_{d-1}, i_d,...,i_n}.

    When called with :attr:`dims` of the list form, the given dimensions will be contracted
    in place of the last :math:`d` of :attr:`a` and the first :math:`d` of :math:`b`. The sizes
    in these dimensions must match.

    """
    if not isinstance(dims, (tuple, list, int)):
        raise RuntimeError(
            "tensordot expects dims to be int or "
            + "Tuple[List[int], List[int]] or "
            + "List[List[int]] containing two lists, but got "
            + f"dims={dims}"
        )

    dims_a, dims_b = [], []

    if isinstance(dims, (tuple, list)):
        dims_a, dims_b = dims

    if isinstance(dims, (int)):
        if dims < 0:
            raise RuntimeError(f"tensordot expects dims >= 0, but got dims={dims}")
        if dims > min(len(a.shape), len(b.shape)):
            raise RuntimeError(f"tensordot expects dims < ndim_a or ndim_b, but got dims={dims}")
        dims_a = list(range(len(a.shape) - dims, len(a.shape)))
        dims_b = list(range(dims))

    # reference: https://github.com/pytorch/pytorch/blob/8ea5b572a63b1acc538a9fc8d3862c73739116e8/aten/src/ATen/native/Linear.cpp#L769
    def __tensordot_native(input1: jt.Var, input2: jt.Var, dims1, dims2):
        if not isinstance(dims1, (list, tuple)):
            raise RuntimeError(
                "tensordot expects dims1 to be List[Int], but got dims={}".format(dims1)
            )
        if not isinstance(dims2, (list, tuple)):
            raise RuntimeError(
                "tensordot expects dims2 to be List[Int], but got dims={}".format(dims2)
            )
        dims1 = list(dims1)
        dims2 = list(dims2)
        if len(dims1) != len(dims2):
            raise RuntimeError("both dimension lists should have the same length")
        if input1.dtype != input2.dtype:
            raise RuntimeError("both inputs should have the same dtype")
        t1 = input1
        t2 = input2
        csize = 1
        input1_bitmap = np.zeros(len(input1.shape), dtype="bool")
        input2_bitmap = np.zeros(len(input2.shape), dtype="bool")
        for i in range(len(dims1)):
            s1 = input1.shape[dims1[i]]
            s2 = input2.shape[dims2[i]]
            input1_bitmap[dims1] = True
            input2_bitmap[dims2] = True
            if s2 == 1:  # broadcasted dimensions can be summed right away
                t1 = t1.sum(dims1[i], keepdims=True)
            elif s1 == 1:
                t2 = t2.sum(dims2[i], keepdims=True)
            else:
                if s1 != s2:
                    raise RuntimeError(
                        "contracted dimensions need to match, but first has size {}, in dim {}, and second has size {}".format(
                            s1, i, s2
                        )
                    )
                csize *= s1

        p1, p2 = [], []  # p1, p2: input permutations
        rsizes = []
        size1, size2 = 1, 1  #  number of non-contracted elements
        for i in range(len(input1.shape)):
            if not input1_bitmap[i]:
                p1.append(i)
                size1 *= t1.shape[i]
                rsizes.append(t1.shape[i])
        p1 += dims1
        p2 += dims2
        for i in range(len(input2.shape)):
            if not input2_bitmap[i]:
                p2.append(i)
                size2 *= t2.shape[i]
                rsizes.append(t2.shape[i])

        # permute and reshape for matrix multiplication
        t1 = t1.permute(p1).reshape((size1, csize))
        t2 = t2.permute(p2).reshape((csize, size2))
        # multiply and reshape to target size
        # Jittor represents a scalar result as a one-element Var rather than a
        # zero-rank tensor; reshape accepts no empty shape, so preserve that
        # established native convention when every dimension is contracted.
        return jt.nn.matmul(t1, t2).reshape(rsizes if rsizes else (1,))

    return __tensordot_native(a, b, dims_a, dims_b)


# reference: https://github.com/pytorch/pytorch/blob/5ed3b70d09a4ab2a5be4becfda9dd0d3e3227c39/aten/src/ATen/native/LinearAlgebra.cpp#L3375
def kron(a: jt.Var, b: jt.Var):
    a_dim, b_dim = len(a.shape), len(b.shape)
    max_dim = max(a_dim, b_dim)
    pad_a, pad_b = max_dim - a_dim, max_dim - b_dim
    a_reshape, b_reshape = [], []
    result_reshape = []
    for i in range(max_dim):
        a_2i_shape = a.shape[i - pad_a] if i >= pad_a else 1
        b_2i1_shape = b.shape[i - pad_b] if i >= pad_b else 1
        a_reshape.append(a_2i_shape)
        a_reshape.append(1)
        b_reshape.append(1)
        b_reshape.append(b_2i1_shape)
        result_reshape.append(a_2i_shape * b_2i1_shape)
    a = a.reshape(a_reshape)
    b = b.reshape(b_reshape)
    return (a * b).reshape(result_reshape)


def one_hot(x: jt.Var, num_classes: int = -1) -> jt.Var:
    """Returns the one_hot encoding of inputs.

    :param x: class values of any shape
    :type x: jt.Var with bool or integer dtype

    :param num_classes: Total number of classes. If set to -1, the number of classes will be inferred as one greater than the largest class value in the input tensor.
    :type num_classes: int, optional

    :return: a Var with one more dimension with 1 values at the index
    of last dimension indicated by the input, and 0 everywhere else.
    :rtype: jt.Var

    .. note::
        if the values in x are greater than num_class or less than 0,
        the returned one_hot will be all zeros.

    Example:
        >>> jt.nn.one_hot(jt.arange(5) % 3)
            jt.Var([[1 0 0]
                [0 1 0]
                [0 0 1]
                [1 0 0]
                [0 1 0]], dtype=int32)
        >>> jt.nn.one_hot(jt.arange(5) % 3, num_classes=5)
            jt.Var([[1 0 0 0 0]
                [0 1 0 0 0]
                [0 0 1 0 0]
                [1 0 0 0 0]
                [0 1 0 0 0]], dtype=int32)
        >>> jt.nn.one_hot(jt.arange(6).reshape(3,2) % 3)
            jt.Var([[[1 0 0]
                [0 1 0]]

                [[0 0 1]
                [1 0 0]]

                [[0 1 0]
                [0 0 1]]], dtype=int32)
    """

    assert x.dtype in [
        jt.bool,
        jt.int8,
        jt.int16,
        jt.int32,
        jt.int64,
        jt.uint8,
        jt.uint16,
        jt.uint32,
        jt.uint64,
    ]
    if num_classes == -1:
        num_classes = x.max().item() + 1

    N = len(x.shape)
    indices = ["i" + str(i) for i in range(N)]
    y = jt.ones_like(x).reindex(
        x.shape + [num_classes],
        indices,
        extras=[x],
        overflow_conditions=[f"i{N} != @e0({','.join(indices)})"],
        overflow_value=0,
    )
    return y


__all__ = ["kron", "one_hot", "tensordot"]
