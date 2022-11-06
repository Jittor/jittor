import jittor as jt


def ccl_3d(data_3d):
    ''' 
    3D connected component labelling, original code from https://github.com/DanielPlayne/playne-equivalence-algorithm
    Args:
        [in]param data_3d: binary three-dimensional vector
            type data_3d: jittor array

    Returns:
        [out]result : labeled three-dimensional vector

    Example:
    >>> import jittor as jt
    >>> jt.flags.use_cuda = 1
    >>> data_3d = jt.zeros((10, 11, 12), dtype=jt.uint32)
    >>> data_3d[2:4, :, :] = 1
    >>> data_3d[5:7, :, :] = 1
    >>> result = ccl_3d(data_3d)
    >>> print(result[:, 0, 0])
    >>> print(
        jt.unique(result, return_counts=True, return_inverse=True)[0],
        jt.unique(result, return_counts=True, return_inverse=True)[2])
    '''

    data_3d = data_3d.astype(jt.uint32)
    cX = data_3d.shape[0]
    cY = data_3d.shape[1]
    cZ = data_3d.shape[2]
    changed = jt.ones([1], dtype=jt.uint32)
    data_3d_copy = data_3d.copy()
    data_3d = data_3d.reshape(cX * cY * cZ)
    result = jt.code(data_3d.shape,
                     data_3d.dtype, [data_3d, changed],
                     cuda_header='''
                    @alias(g_image, in0)
                    @alias(g_labels, out)
                    ''',
                     cuda_src=r'''
                    __global__ void init_labels(@ARGS_DEF, const int cX, const int cY, const int cZ, const int pX, const int pY) {
                        @PRECALC
                        // Calculate index
                        const unsigned int ix = (blockIdx.x * blockDim.x) + threadIdx.x;
                        const unsigned int iy = (blockIdx.y * blockDim.y) + threadIdx.y;
                        const unsigned int iz = (blockIdx.z * blockDim.z) + threadIdx.z;

                        if((ix < cX) && (iy < cY) && (iz < cZ)) {
                            const unsigned char pzyx = @g_image(iz*pY + iy*pX + ix);
                            
                            // Neighbour Connections
                            const bool nzm1yx   = (iz > 0) ? (pzyx == @g_image((iz-1)*pY +  iy   *pX + ix  )) : false;
                            const bool nzym1x   = (iy > 0) ? (pzyx == @g_image( iz   *pY + (iy-1)*pX + ix  )) : false;
                            const bool nzyxm1   = (ix > 0) ? (pzyx == @g_image( iz   *pY +  iy   *pX + ix-1)) : false;

                            // Label
                            unsigned int label;

                            // Initialise Label
                            label = (nzyxm1) ? (    iz*pY +     iy*pX + ix-1) : (iz*pY + iy*pX + ix);
                            label = (nzym1x) ? (    iz*pY + (iy-1)*pX +   ix) : label;
                            label = (nzm1yx) ? ((iz-1)*pY +     iy*pX +   ix) : label;
                            // Write to Global Memory
                            @g_labels(iz*pY + iy*pX + ix) = label;
                        }
                    }

                    __device__ __inline__ unsigned int find_root(@ARGS_DEF, unsigned int label) {
                        // Resolve Label
                        unsigned int next = @g_labels(label);

                        // Follow chain
                        while(label != next) {
                            // Move to next
                            label = next;
                            next = @g_labels(label);
                        }

                        // Return label
                        return label;
                    }

                    __global__ void resolve_labels(@ARGS_DEF, const int cX, const int cY, const int cZ, const int pX, const int pY) {
                        @PRECALC
                        // Calculate index
                        const unsigned int id = ((blockIdx.z * blockDim.z) + threadIdx.z) * pY +
                                                ((blockIdx.y * blockDim.y) + threadIdx.y) * pX +
                                                ((blockIdx.x * blockDim.x) + threadIdx.x);
                        
                        // Check Thread Range
                        if(id < cX*cY*cZ) {
                            // Resolve Label
                            @g_labels(id) = find_root(@ARGS, @g_labels(id));
                        }
                    }

                    __global__ void label_equivalence(@ARGS_DEF, const int cX, const int cY, const int cZ, const int pX, const int pY) {
                        @PRECALC
                        // Calculate index
                        const unsigned int ix = (blockIdx.x * blockDim.x) + threadIdx.x;
                        const unsigned int iy = (blockIdx.y * blockDim.y) + threadIdx.y;
                        const unsigned int iz = (blockIdx.z * blockDim.z) + threadIdx.z;

                        // Check Thread Range
                        if((ix < cX) && (iy < cY) && (iz < cZ)) {
                            // Get image and label values
                            const unsigned char pzyx  = @g_image(iz*pY + iy*pX + ix);
                            
                            // Neighbouring indexes
                            const unsigned int xm1 = ix-1;
                            const unsigned int xp1 = ix+1;
                            const unsigned int ym1 = iy-1;
                            const unsigned int yp1 = iy+1;
                            const unsigned int zm1 = iz-1;
                            const unsigned int zp1 = iz+1;

                            // Get neighbour labels
                            const unsigned int lzm1yx = (iz > 0)    ? @g_labels(zm1*pY +  iy*pX +  ix) : 0;
                            const unsigned int lzym1x = (iy > 0)    ? @g_labels( iz*pY + ym1*pX +  ix) : 0;
                            const unsigned int lzyxm1 = (ix > 0)    ? @g_labels( iz*pY +  iy*pX + xm1) : 0;
                            const unsigned int lzyx   =               @g_labels( iz*pY +  iy*pX +  ix);
                            const unsigned int lzyxp1 = (ix < cX-1) ? @g_labels( iz*pY +  iy*pX + xp1) : 0;
                            const unsigned int lzyp1x = (iy < cY-1) ? @g_labels( iz*pY + yp1*pX +  ix) : 0;
                            const unsigned int lzp1yx = (iz < cZ-1) ? @g_labels(zp1*pY +  iy*pX +  ix) : 0;

                            const bool nzm1yx = (iz > 0)    ? (pzyx == @g_image(zm1*pY +  iy*pX +  ix)) : false;
                            const bool nzym1x = (iy > 0)    ? (pzyx == @g_image( iz*pY + ym1*pX +  ix)) : false;
                            const bool nzyxm1 = (ix > 0)    ? (pzyx == @g_image( iz*pY +  iy*pX + xm1)) : false;
                            const bool nzyxp1 = (ix < cX-1) ? (pzyx == @g_image( iz*pY +  iy*pX + xp1)) : false;
                            const bool nzyp1x = (iy < cY-1) ? (pzyx == @g_image( iz*pY + yp1*pX +  ix)) : false;
                            const bool nzp1yx = (iz < cZ-1) ? (pzyx == @g_image(zp1*pY +  iy*pX +  ix)) : false;

                            // Lowest label
                            unsigned int label = lzyx;

                            // Find lowest neighbouring label
                            label = ((nzm1yx) && (lzm1yx < label)) ? lzm1yx : label;
                            label = ((nzym1x) && (lzym1x < label)) ? lzym1x : label;
                            label = ((nzyxm1) && (lzyxm1 < label)) ? lzyxm1 : label;
                            label = ((nzyxp1) && (lzyxp1 < label)) ? lzyxp1 : label;
                            label = ((nzyp1x) && (lzyp1x < label)) ? lzyp1x : label;
                            label = ((nzp1yx) && (lzp1yx < label)) ? lzp1yx : label;

                            // If labels are different, resolve them
                            if(label < lzyx) {
                                // Update label
                                // Nonatomic write may overwrite another label but on average seems to give faster results
                                @g_labels(lzyx) = label;

                                // Record the change
                                @in1(0) = 1;
                            }
                        }
                    }
                    ''' + f'''
                    dim3 block(32, 4, 4);
                    const int cX= {cX};
                    const int cY= {cY};
                    const int cZ= {cZ};
                    const int pX= cX;
                    const int pY= cX*cY;''' + '''
                    dim3 grid(ceil(cX/(float)block.x), ceil(cY/(float)block.y), ceil(cZ/(float)block.z));
                    
                    // Initialise labels
                    init_labels <<< grid, block >>>(@ARGS, cX, cY, cZ, pX, pY);
                    
                    // Resolve the labels
                    resolve_labels <<< grid, block >>>(@ARGS, cX, cY, cZ, pX, pY);

                    // Changed Flag
                    int32 changed = 1;
                    
                    // While labels have changed
                    while(changed) {
                        // Copy changed to device
                        cudaMemsetAsync(in1_p, 0, 4);
                        
                        // Label image
                        label_equivalence <<< grid, block >>>(@ARGS, cX, cY, cZ, pX, pY);

                        // Copy changed flag to host
                        cudaMemcpy(&changed, in1_p, sizeof(int32), cudaMemcpyDeviceToHost);
                        
                        // Resolve the labels
                        resolve_labels <<< grid, block>>>(@ARGS, cX, cY, cZ, pX, pY);
                    }
                    ''')
    result = result.reshape((cX, cY, cZ)) * data_3d_copy
    value = jt.unique(result)
    value = value[value != 0]

    map_result = jt.zeros((int(value.max().numpy()[0]) + 1), dtype=jt.uint32)
    map_result[value] = jt.index(value.shape)[0] + 1
    result = map_result[result]

    return result
