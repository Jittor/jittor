"""CUDA source builders for the native CTC forward and backward recurrence."""


def forward_source(blank):
    return (f"""
        __global__ void kernel(@ARGS_DEF) {{
            @PRECALC;
            constexpr int blank = {blank};
            for (int i=blockIdx.x; i<in0_shape1; i+=gridDim.x) {{
                int input_len = @in2(i);
                int target_len = @in3(i);
                @out0(0,i,0) = @in0(0,i,blank);
                if (target_len)
                    @out0(0,i,1) = @in0(0,i,@in1(i,0));
                for (int j=1; j<input_len; j++)
                    for (int k=threadIdx.x; k-threadIdx.x<target_len*2+1; k+=blockDim.x) {{
                        __syncthreads();
                        if (k>=target_len*2+1)
                            continue;
                        int target = k%2 ? @in1(i,k/2) : blank;
                        int target_2 = target;
                        if (k>1 && k%2) target_2 = @in1(i,k/2-1);
                        out_type l1 = @out0(j-1,i,k);
                        out_type l2 = -1e30;
                        if (k>0) l2 = @out0(j-1,i,k-1);
                        out_type l3 = -1e30;
                        if (k>1 && target_2 != target)
                            l3 = @out0(j-1,i,k-2);
                        out_type m = ::max(l1, ::max(l2, l3));
                        @out0(j,i,k) = ::log(
                            ::exp(l1-m) +
                            ::exp(l2-m) +
                            ::exp(l3-m)
                        ) + m + @in0(j,i,target);
                    }}
                 __syncthreads();
                if (input_len==0)
                    @out1(i) = @out0(0,i,0);
                else {{
                    out_type l1 = @out0(input_len-1, i, target_len*2);
                    out_type l2 = -1e30;
                    if (target_len)
                        l2 = @out0(input_len-1, i, target_len*2-1);
                    out_type m = ::max(l1, l2);
                    out_type log_likelihood = ::log(::exp(l1-m)+::exp(l2-m))+m;
                    @out1(i) = -log_likelihood;
                }}
            }}
        }}
        kernel<<<std::min(in0_shape1, 1024), std::min(in1_shape1*2+1, 1024)>>>(@ARGS);
        """
    )


def backward_source(blank):
    return (f"""
        __global__ void kernel(@ARGS_DEF) {{
            @PRECALC;
            constexpr int blank = {blank};
            for (int i=blockIdx.x; i<in0_shape1; i+=gridDim.x) {{
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
                    out_type m = ::max(l1, l2);
                    // out_type log_likelihood = ::log(::exp(l1-m)+::exp(l2-m))+m;
                    // out1(i) = -log_likelihood;
                    out_type l1_exp = ::exp(l1-m);
                    out_type l2_exp = ::exp(l2-m);
                    out_type sumexp = l1_exp + l2_exp;

                    out_type dlog_likelihood = -@in6(i);
                    out_type dl1 = dlog_likelihood * l1_exp / sumexp;
                    out_type dl2 = dlog_likelihood * l2_exp / sumexp;

                    @out1(input_len-1, i, target_len*2) = dl1;
                    if (target_len)
                        @out1(input_len-1, i, target_len*2-1) = dl2;
                }}
                for (int j=input_len-1; j>0; j--)
                    for (int k=threadIdx.x; k-threadIdx.x<target_len*2+1; k+=blockDim.x) {{
                        __syncthreads();
                        if (k>=target_len*2+1)
                            continue;
                        int target = k%2 ? @in1(i,k/2) : blank;
                        int target_2 = target;
                        if (k>1 && k%2) target_2 = @in1(i,k/2-1);
                        out_type l1 = @in4(j-1,i,k);
                        out_type l2 = -1e30;
                        if (k>0) l2 = @in4(j-1,i,k-1);
                        out_type l3 = -1e30;
                        if (k>1 && target_2 != target)
                            l3 = @in4(j-1,i,k-2);
                        out_type m = ::max(l1, ::max(l2, l3));
                        out_type l1_exp = ::exp(l1-m);
                        out_type l2_exp = ::exp(l2-m);
                        out_type l3_exp = ::exp(l3-m);
                        out_type sumexp = l1_exp + l2_exp + l3_exp;
                        out_type dalpha = @out1(j,i,k);

                        atomicAdd(&@out0(j,i,target), dalpha);

                        atomicAdd(&@out1(j-1,i,k), dalpha * l1_exp / sumexp);
                        if (k>0)
                            atomicAdd(&@out1(j-1,i,k-1), dalpha * l2_exp / sumexp);
                        if (k>1 && target_2 != target)
                            atomicAdd(&@out1(j-1,i,k-2), dalpha * l3_exp / sumexp);
                    }}
                // read in0 -> white out0
                // write out0 ->read out1
                // out0(0,i,0) = in0(0,i,blank);
                __syncthreads();
                if (threadIdx.x==0) {{
                    @out0(0,i,blank) += @out1(0,i,0);
                    if (target_len)
                        @out0(0,i,@in1(i,0)) += @out1(0,i,1);
                }}
            }}
        }}
        kernel<<<std::min(in0_shape1, 1024), std::min(in1_shape1*2+1, 1024)>>>(@ARGS);
        """
    )
