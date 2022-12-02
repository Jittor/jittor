import jittor as jt


def ccl_2d(data_2d):
    ''' 
    2D connected component labelling, original code from https://github.com/DanielPlayne/playne-equivalence-algorithm
    Args:
        [in]param data_2d: binary two-dimensional vector
            type data_2d: jittor array

    Returns:
        [out]result: labeled two-dimensional vector

    Example:
    >>> import jittor as jt
    >>> jt.flags.use_cuda = 1
    >>> import cv2
    >>> import numpy as np
    >>> img = cv2.imread('testImg.png', 0)
    >>> a = img.mean()
    >>> img[img <= a] = 0
    >>> img[img > a] = 1
    >>> img = jt.Var(img)

    >>> result = ccl_2d(img)
    >>> print(jt.unique(result, return_counts=True, return_inverse=True)[0], jt.unique(result, return_counts=True, return_inverse=True)[2])
    >>> cv2.imwrite('testImg_result.png', result.numpy().astype(np.uint8) * 50)
    '''

    data_2d = data_2d.astype(jt.uint32)
    cY = data_2d.shape[0]
    cX = data_2d.shape[1]
    data_2d_copy = data_2d.clone()
    changed = jt.ones([1], dtype=jt.uint32)
    data_2d = data_2d.reshape(cX * cY)
    result = jt.code(data_2d.shape,
                     data_2d.dtype, [data_2d, changed],
                     cuda_header='''
                    @alias(g_image, in0)
                    @alias(g_labels, out)
                    ''',
                     cuda_src=r'''
                    __global__ void init_labels(@ARGS_DEF, const int cX, const int cY) {
                        @PRECALC
                        // Calculate index
                        const unsigned int ix = (blockIdx.x * blockDim.x) + threadIdx.x;
                        const unsigned int iy = (blockIdx.y * blockDim.y) + threadIdx.y;
                        @g_labels(iy*cX + ix) = iy*cX + ix;
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

                    __global__ void resolve_labels(@ARGS_DEF, const int cX, const int cY) {
                        @PRECALC
                        // Calculate index
                        const unsigned int id = ((blockIdx.y * blockDim.y) + threadIdx.y) * cX +
                                                ((blockIdx.x * blockDim.x) + threadIdx.x);
                        
                        // Check Thread Range
                        if(id < cX*cY) {
                            // Resolve Label
                            @g_labels(id) = find_root(@ARGS, @g_labels(id));
                        }
                    }

                    __global__ void label_equivalence(@ARGS_DEF, const int cX, const int cY) {
                        @PRECALC
                        // Calculate index
                        const unsigned int ix = (blockIdx.x * blockDim.x) + threadIdx.x;
                        const unsigned int iy = (blockIdx.y * blockDim.y) + threadIdx.y;

                        // Check Thread Range
                        if((ix < cX) && (iy < cY)) {
                            // Get image and label values
                            const unsigned char cyx   = @g_image( iy*cX + ix);
                            
                            // Get neighbour labels
                            const unsigned int lym1x = (iy > 0)    ? @g_labels((iy-1)*cX +   ix) : 0;
                            const unsigned int lyxm1 = (ix > 0)    ? @g_labels(iy    *cX + ix-1) : 0;
                            const unsigned int lyx   =               @g_labels(iy    *cX +   ix);
                            const unsigned int lyxp1 = (ix < cX-1) ? @g_labels(iy    *cX + ix+1) : 0;
                            const unsigned int lyp1x = (iy < cY-1) ? @g_labels((iy+1)*cX +   ix) : 0;

                            const unsigned int lym1xm1 = (iy > 0    && ix > 0   )    ? @g_labels((iy-1)*cX +   ix-1) : 0;
                            const unsigned int lym1xp1 = (iy > 0    && ix < cX-1)    ? @g_labels((iy-1)*cX +   ix+1) : 0;
                            const unsigned int lyp1xm1 = (iy < cY-1 && ix > 0   )    ? @g_labels((iy+1)*cX +   ix-1) : 0;
                            const unsigned int lyp1xp1 = (iy < cY-1 && ix < cX-1)    ? @g_labels((iy+1)*cX +   ix+1) : 0;

                            const bool nym1x = (iy > 0)    ? (cyx == (@g_image((iy-1)*cX +   ix))) : false;
                            const bool nyxm1 = (ix > 0)    ? (cyx == (@g_image(iy    *cX + ix-1))) : false;
                            const bool nyxp1 = (ix < cX-1) ? (cyx == (@g_image(iy    *cX + ix+1))) : false;
                            const bool nyp1x = (iy > cY-1) ? (cyx == (@g_image((iy+1)*cX +   ix))) : false;

                            const bool nym1xm1 = (iy > 0    && ix > 0   )    ? (cyx == (@g_image((iy-1)*cX +   ix-1))) : false;
                            const bool nym1xp1 = (iy > 0    && ix < cX-1)    ? (cyx == (@g_image((iy-1)*cX +   ix+1))) : false;
                            const bool nyp1xm1 = (iy < cY-1 && ix > 0   )    ? (cyx == (@g_image((iy+1)*cX +   ix-1))) : false;
                            const bool nyp1xp1 = (iy < cY-1 && ix < cX-1)    ? (cyx == (@g_image((iy+1)*cX +   ix+1))) : false;

                            // Lowest label
                            unsigned int label = lyx;

                            // Find lowest neighbouring label
                            label = ((nym1x) && (lym1x < label)) ? lym1x : label;
                            label = ((nyxm1) && (lyxm1 < label)) ? lyxm1 : label;
                            label = ((nyxp1) && (lyxp1 < label)) ? lyxp1 : label;
                            label = ((nyp1x) && (lyp1x < label)) ? lyp1x : label;

                            label = ((nym1xm1) && (lym1xm1 < label)) ? lym1xm1 : label;
                            label = ((nym1xp1) && (lym1xp1 < label)) ? lym1xp1 : label;
                            label = ((nyp1xm1) && (lyp1xm1 < label)) ? lyp1xm1 : label;
                            label = ((nyp1xp1) && (lyp1xp1 < label)) ? lyp1xp1 : label;

                            // If labels are different, resolve them
                            if(label < lyx) {
                                // Update label
                                // Nonatomic write may overwrite another label but on average seems to give faster results
                                @g_labels(lyx) = label;

                                // Record the change
                                @in1(0) = 1;
                            }
                        }
                    }
                    ''' + f'''
                    dim3 block(32, 32);
                    const int cX= {cX};
                    const int cY= {cY};''' + '''
                    dim3 grid(ceil(cX/(float)block.x), ceil(cY/(float)block.y));
                    dim3 resolve_block(32, 32);
                    dim3 resolve_grid(ceil(cX/(float)resolve_block.x), ceil(cY/(float)resolve_block.y));
                    
                    // Initialise labels
                    init_labels <<< grid, block >>>(@ARGS, cX, cY);
                    
                    // Resolve the labels
                    resolve_labels <<< resolve_grid, resolve_block >>>(@ARGS, cX, cY);

                    // Changed Flag
                    int32 changed = 1;
                    
                    // While labels have changed
                    while(changed) {
                        // Copy changed to device
                        cudaMemsetAsync(in1_p, 0, 4);
                        
                        // Label image
                        label_equivalence <<< grid, block >>>(@ARGS, cX, cY);

                        // Copy changed flag to host
                        cudaMemcpy(&changed, in1_p, sizeof(int32), cudaMemcpyDeviceToHost);
                        
                        // Resolve the labels
                        resolve_labels <<< resolve_grid, resolve_block>>>(@ARGS, cX, cY);
                    }
                    ''')
    result = result.reshape((cY, cX)) * data_2d_copy
    value = jt.unique(result)
    value = value[value != 0]

    map_result = jt.zeros((int(value.max().numpy()[0]) + 1), dtype=jt.uint32)
    map_result[value] = jt.index(value.shape)[0] + 1
    result = map_result[result]

    return result
