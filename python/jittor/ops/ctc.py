"""Ctc tensor operations."""

from .._core.function import Function
from .._core.module import Module
from .. import _arg_policy

class _CTCLossFunction(Function):
    def execute(self, log_probs, targets, input_lengths, target_lengths, blank=0, zero_infinity=False):
        from jittor.backends.cuda.kernels.misc import ctc as _cuda_ctc
        import jittor as jt
        self.blank = blank
        T, N, C = log_probs.shape
        _N, S = targets.shape
        assert _N == N
        log_alpha = jt.full([T,N,S*2+1], -1e30)
        result = jt.empty((N,))
        jt.code([log_probs, targets, input_lengths, target_lengths], [log_alpha, result], cpu_src=f"""
            constexpr int blank = {blank};
            for (int i=0; i<in0_shape1; i++) {{
                int input_len = @in2(i);
                int target_len = @in3(i);
                @out0(0,i,0) = @in0(0,i,blank);
                if (target_len)
                    @out0(0,i,1) = @in0(0,i,@in1(i,0));
                for (int j=1; j<input_len; j++)
                    for (int k=0; k<target_len*2+1; k++) {{
                        int target = k%2 ? @in1(i,k/2) : blank;
                        int target_2 = target;
                        if (k>1 && k%2) target_2 = @in1(i,k/2-1);
                        out_type l1 = @out0(j-1,i,k);
                        out_type l2 = -1e30;
                        if (k>0) l2 = @out0(j-1,i,k-1);
                        out_type l3 = -1e30;
                        if (k>1 && target_2 != target)
                            l3 = @out0(j-1,i,k-2);
                        out_type m = std::max(l1, std::max(l2, l3));
                        @out0(j,i,k) = std::log(
                            std::exp(l1-m) +
                            std::exp(l2-m) +
                            std::exp(l3-m)
                        ) + m + @in0(j,i,target);
                    }}
                if (input_len==0)
                    @out1(i) = @out0(0,i,0);
                else {{
                    out_type l1 = @out0(input_len-1, i, target_len*2);
                    out_type l2 = -1e30;
                    if (target_len)
                        l2 = @out0(input_len-1, i, target_len*2-1);
                    out_type m = std::max(l1, l2);
                    out_type log_likelihood = std::log(std::exp(l1-m)+std::exp(l2-m))+m;
                    @out1(i) = -log_likelihood;
                }}
            }}
        """, cuda_src=_cuda_ctc.forward_source(blank))
        self.saved_var = [log_probs, targets, input_lengths, target_lengths, log_alpha, result]
        return result

    def grad(self, dout):
        from jittor.backends.cuda.kernels.misc import ctc as _cuda_ctc
        import jittor as jt
        blank = self.blank
        inputs = self.saved_var + [dout]
        dlog_probs = jt.zeros_like(inputs[0])
        dlog_alpha = jt.zeros_like(inputs[4])
        jt.code(inputs, [dlog_probs, dlog_alpha], cpu_src=f"""
            constexpr int blank = {blank};
            for (int i=0; i<in0_shape1; i++) {{
                int input_len = @in2(i);
                int target_len = @in3(i);
                if (input_len==0)
                    // write out1 --> read in6
                    // out1(i) = out0(0,i,0);
                    @out1(0,i,0) = @in6(i);
                else {{
                    out_type l1 = @in4(input_len-1, i, target_len*2);
                    out_type l2 = -1e30;
                    if (target_len)
                        l2 = @in4(input_len-1, i, target_len*2-1);
                    out_type m = std::max(l1, l2);
                    // out_type log_likelihood = std::log(std::exp(l1-m)+std::exp(l2-m))+m;
                    // out1(i) = -log_likelihood;
                    out_type l1_exp = std::exp(l1-m);
                    out_type l2_exp = std::exp(l2-m);
                    out_type sumexp = l1_exp + l2_exp;

                    out_type dlog_likelihood = -@in6(i);
                    out_type dl1 = dlog_likelihood * l1_exp / sumexp;
                    out_type dl2 = dlog_likelihood * l2_exp / sumexp;

                    @out1(input_len-1, i, target_len*2) = dl1;
                    if (target_len)
                        @out1(input_len-1, i, target_len*2-1) = dl2;
                }}
                for (int j=input_len-1; j>0; j--)
                    for (int k=0; k<target_len*2+1; k++) {{
                        int target = k%2 ? @in1(i,k/2) : blank;
                        int target_2 = target;
                        if (k>1 && k%2) target_2 = @in1(i,k/2-1);
                        out_type l1 = @in4(j-1,i,k);
                        out_type l2 = -1e30;
                        if (k>0) l2 = @in4(j-1,i,k-1);
                        out_type l3 = -1e30;
                        if (k>1 && target_2 != target)
                            l3 = @in4(j-1,i,k-2);
                        out_type m = std::max(l1, std::max(l2, l3));
                        out_type l1_exp = std::exp(l1-m);
                        out_type l2_exp = std::exp(l2-m);
                        out_type l3_exp = std::exp(l3-m);
                        out_type sumexp = l1_exp + l2_exp + l3_exp;
                        out_type dalpha = @out1(j,i,k);

                        @out0(j,i,target) += dalpha;

                        @out1(j-1,i,k) += dalpha * l1_exp / sumexp;
                        if (k>0)
                            @out1(j-1,i,k-1) += dalpha * l2_exp / sumexp;
                        if (k>1 && target_2 != target)
                            @out1(j-1,i,k-2) += dalpha * l3_exp / sumexp;
                    }}
                // read in0 -> white out0
                // write out0 ->read out1
                // out0(0,i,0) = in0(0,i,blank);
                @out0(0,i,blank) += @out1(0,i,0);
                if (target_len)
                    @out0(0,i,@in1(i,0)) += @out1(0,i,1);
            }}
        """, cuda_src=_cuda_ctc.backward_source(blank))
        return (dlog_probs,)


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, reduction='mean', zero_infinity=False):
    '''The Connectionist Temporal Classification loss.


    Reference:
        A. Graves et al.: Connectionist Temporal Classification:
        Labelling Unsegmented Sequence Data with Recurrent Neural Networks:
        https://www.cs.toronto.edu/~graves/icml_2006.pdf

    Input:

        log_probs: shape is [T, N, C], T is the sequence length, N is the batch size, C is the class number.
        targets: shape is [N, S], N is the batch size, S is the target sequence length, element should between [0,C).
        input_lengths: shape is [N], which represents the length of input, element should between [0,T].
        target_lengths: shape is N, which represents the length of target, element should between [0,S].
        blank (int, default 0): blank label index
        reduction (string): reduce batch loss,
            if reduction is none, it will return (N,) array,
            if reduction is mean or sum, it will return one scalar
        zero_infinity (bool, default False):
            NOT IMPLEMENTED -- passing True raises NotImplementedError. torch
            zeroes infinite losses and their gradients; this implementation
            does not, and silently accepting the flag would leave inf/nan in
            the reduced loss.

    Example:

        import jittor as jt
        T = 50      # Input sequence length
        C = 20      # Number of classes (including blank)
        N = 16      # Batch size
        S = 30      # Target sequence length of longest target in batch (padding length)
        S_min = 10  # Minimum target length, for demonstration purposes

        input = jt.randn(T, N, C).log_softmax(2)
        # Initialize random batch of targets (0 = blank, 1:C = classes)
        target = jt.randint(low=1, high=C, shape=(N, S), dtype=jt.int)

        input_lengths = jt.full((N,), T, dtype=jt.int)
        target_lengths = jt.randint(low=S_min, high=S+1, shape=(N,), dtype=jt.int)
        loss = jt.ctc_loss(input, target, input_lengths, target_lengths)

        dinput = jt.grad(loss, input)

    '''
    import jittor as jt
    if zero_infinity:
        _arg_policy.unsupported(
            "jittor.ctc_loss", "zero_infinity", zero_infinity,
            "infinite losses (a target longer than the input) and their "
            "gradients are kept as inf instead of being zeroed, so the reduced "
            "loss and every gradient in the batch become inf/nan")
    result = jt.misc._CTCLossFunction.apply(
        log_probs, targets, input_lengths, target_lengths, blank,
        zero_infinity,
    )
    if reduction=="mean":
        return result.mean()
    elif reduction=="sum":
        return result.sum()
    assert reduction=="none"
    return result


class CTCLoss(Module):
    '''The Connectionist Temporal Classification loss.


    Reference:
        A. Graves et al.: Connectionist Temporal Classification:
        Labelling Unsegmented Sequence Data with Recurrent Neural Networks:
        https://www.cs.toronto.edu/~graves/icml_2006.pdf


    Args:

        blank (int, default 0): blank label index
        reduction (string): reduce batch loss,
            if reduction is none, it will return (N,) array,
            if reduction is mean or sum, it will return one scalar
        zero_infinity (bool, default False):
            NOT IMPLEMENTED -- passing True raises NotImplementedError. torch
            zeroes infinite losses and their gradients; this implementation
            does not, and silently accepting the flag would leave inf/nan in
            the reduced loss.

    Input:

        log_probs: shape is [T, N, C], T is the sequence length, N is the batch size, C is the class number.
        targets: shape is [N, S], N is the batch size, S is the target sequence length, element should between [0,C).
        input_lengths: shape is [N], which represents the length of input, element should between [0,T].
        target_lengths: shape is N, which represents the length of target, element should between [0,S].

    Example:

        import jittor as jt
        T = 50      # Input sequence length
        C = 20      # Number of classes (including blank)
        N = 16      # Batch size
        S = 30      # Target sequence length of longest target in batch (padding length)
        S_min = 10  # Minimum target length, for demonstration purposes

        input = jt.randn(T, N, C).log_softmax(2)
        # Initialize random batch of targets (0 = blank, 1:C = classes)
        target = jt.randint(low=1, high=C, shape=(N, S), dtype=jt.int)

        input_lengths = jt.full((N,), T, dtype=jt.int)
        target_lengths = jt.randint(low=S_min, high=S+1, shape=(N,), dtype=jt.int)
        ctc_loss = jt.CTCLoss()
        loss = ctc_loss(input, target, input_lengths, target_lengths)

        dinput = jt.grad(loss, input)

    '''
    def __init__(self, blank=0, reduction='mean', zero_infinity=False):
        self.blank = blank
        self.reduction = reduction
        self.zero_infinity = zero_infinity

    def execute(self, log_probs, targets, input_lengths, target_lengths):
        import jittor as jt
        return jt.misc.ctc_loss(
            log_probs, targets, input_lengths, target_lengths, self.blank,
            self.reduction, self.zero_infinity,
        )
