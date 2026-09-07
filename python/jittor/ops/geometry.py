"""Geometry tensor operations."""


def knn(unknown, known, k):
    ''' find k neighbors for unknown array from known array

    Args:

        unknown (var): shape [b, n, c]
        known (var): shape [b, m, c]
        k (int)

    '''
    import jittor as jt
    b, n, c = unknown.shape
    _, m, _ = known.shape
    dists2 = jt.empty((b, n, k), dtype="float")
    idx = jt.empty((b, n, k), dtype="int")
    src = '''
__inline_static__
@python.jittor.auto_parallel(2, block_num=256)
void knn_kernel(int b, int batch_index, int n, int index, int m,
                        const float *__restrict__ unknown,
                        const float *__restrict__ known,
                        float *__restrict__ dist2,
                        int *__restrict__ idx) {

#define K %s
    unknown += batch_index * n * 3;
    known += batch_index * m * 3;
    dist2 += batch_index * n * K;
    idx += batch_index * n * K;
    int j = index;
    {
        float ux = unknown[j * 3 + 0];
        float uy = unknown[j * 3 + 1];
        float uz = unknown[j * 3 + 2];

        float tmp_dist[K];
        int tmp_idx[K];
        #pragma unroll
        for (int i=0; i<K; i++) tmp_dist[i] = 1e30;
        for (int k = 0; k < m; ++k) {
            float x = known[k * 3 + 0];
            float y = known[k * 3 + 1];
            float z = known[k * 3 + 2];
            float d = (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);

            int first = -1;
            #pragma unroll
            for (int i=0; i<K; i++)
                if (first == -1 && d<tmp_dist[i])
                    first = i;
            if (first == -1) continue;
            #pragma unroll
            for (int i=0; i<K; i++)
                if (K-1-i > first) {
                    tmp_dist[K-1-i] = tmp_dist[K-2-i];
                    tmp_idx[K-1-i] = tmp_idx[K-2-i];
                }
            tmp_dist[first] = d;
            tmp_idx[first] = k;
        }
        #pragma unroll
        for (int i=0; i<K; i++) {
            dist2[j * K + i] = tmp_dist[i];
            idx[j * K + i] = tmp_idx[i];
        }
    }
}
    knn_kernel(in0->shape[0], 0, in0->shape[1], 0, in1->shape[1], in0_p, in1_p, out0_p, out1_p);
    ''' % k
    return jt.code([unknown, known], [dists2, idx],
    cpu_src=src,
    cuda_src=src)


def cross(input, other, dim=-1):
    r'''
    Returns the cross product of vectors in dimension dim of input and other.

    the cross product can be calculated by (a1,a2,a3) x (b1,b2,b3) = (a2b3-a3b2, a3b1-a1b3, a1b2-a2b1)

    input and other must have the same size, and the size of their dim dimension should be 3.

    If dim is not given, it defaults to the first dimension found with the size 3.

    Args:

        input (Tensor) – the input tensor.

        other (Tensor) – the second input tensor

        dim (int, optional) – the dimension to take the cross-product in.

        out (Tensor, optional) – the output tensor.

    Example:

        >>> input = jt.random((6,3))

        >>> other = jt.random((6,3))

        >>> jt.cross(input, other, dim=1)
        [[-0.42732686  0.6827885  -0.49206433]
        [ 0.4651107   0.27036983 -0.5580432 ]
        [-0.31933784  0.10543461  0.09676848]
        [-0.58346975 -0.21417202  0.55176204]
        [-0.40861478  0.01496297  0.38638002]
        [ 0.18393655 -0.04907863 -0.17928357]]

        >>> jt.cross(input, other)
        [[-0.42732686  0.6827885  -0.49206433]
        [ 0.4651107   0.27036983 -0.5580432 ]
        [-0.31933784  0.10543461  0.09676848]
        [-0.58346975 -0.21417202  0.55176204]
        [-0.40861478  0.01496297  0.38638002]
        [ 0.18393655 -0.04907863 -0.17928357]]
    '''
    import jittor as jt
    assert input.shape==other.shape, "input shape and other shape must be same"
    if dim < 0: dim += len(input.shape)
    assert input.shape[dim] == 3, "input dim shape must be 3"
    a1 = input[(slice(None,),)*dim+(1,)]*other[(slice(None,),)*dim+(2,)]-input[(slice(None,),)*dim+(2,)]*other[(slice(None,),)*dim+(1,)]
    a2 = input[(slice(None,),)*dim+(2,)]*other[(slice(None,),)*dim+(0,)]-input[(slice(None,),)*dim+(0,)]*other[(slice(None,),)*dim+(2,)]
    a3 = input[(slice(None,),)*dim+(0,)]*other[(slice(None,),)*dim+(1,)]-input[(slice(None,),)*dim+(1,)]*other[(slice(None,),)*dim+(0,)]
    return jt.concat([a1.unsqueeze(dim),a2.unsqueeze(dim),a3.unsqueeze(dim)], dim=dim)


def nms(dets,thresh):
    '''
      dets jt.array [x1,y1,x2,y2,score]
      x(:,0)->x1,x(:,1)->y1,x(:,2)->x2,x(:,3)->y2,x(:,4)->score
    '''
    import jittor as jt
    threshold = str(thresh)
    order = jt.argsort(dets[:,4],descending=True)[0]
    dets = dets[order]
    s_1 = '(@x(j,2)-@x(j,0)+1)*(@x(j,3)-@x(j,1)+1)'
    s_2 = '(@x(i,2)-@x(i,0)+1)*(@x(i,3)-@x(i,1)+1)'
    s_inter_w = 'max((Tx)0,min(@x(j,2),@x(i,2))-max(@x(j,0),@x(i,0))+1)'
    s_inter_h = 'max((Tx)0,min(@x(j,3),@x(i,3))-max(@x(j,1),@x(i,1))+1)'
    s_inter = s_inter_h+'*'+s_inter_w
    iou = s_inter + '/(' + s_1 +'+' + s_2 + '-' + s_inter + ')'
    fail_cond = iou+'>'+threshold
    selected = jt.candidate(dets, fail_cond)
    return order[selected]
