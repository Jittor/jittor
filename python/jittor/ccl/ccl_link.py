import jittor as jt


def ccl_link(score_map, link_map, result_comp_area_thresh=6):
    """
    Find components in score map and link them with link map, original code from https://github.com/DanielPlayne/playne-equivalence-algorithm.
    Args:
        [in]param score_map: binary two-dimensional vector
            type score_map: jittor array
        [in]param link_map: two-dimensional vector with 8 channels
            type link_map: jittor array
        [in]param result_comp_area_thresh: threshold of component area
            type result_comp_area_thresh: int
    Returns:
        [out]result: labeled two-dimensional vector
    Example:
    >>> import jittor as jt
    >>> jt.flags.use_cuda = 1
    >>> import cv2
    >>> import numpy as np
    >>> score_map = jt.Var(np.load("score_map.npy"))
    >>> link_map = jt.Var(np.load("link_map.npy"))
    >>> score_map = score_map >= 0.5
    >>> link_map = link_map >= 0.8
    >>> for i in range(8):
    >>>     link_map[:, :, i] = link_map[:, :, i] & score_map
    
    >>> result = ccl_link(score_map, link_map)
    >>> cv2.imwrite('pixellink.png', result.numpy().astype(np.uint8) * 50)
    """
    score_map = score_map.astype(jt.uint32)
    link_map = link_map.astype(jt.uint32)
    cY = score_map.shape[0]
    cX = score_map.shape[1]
    changed = jt.ones([1], dtype=jt.uint32)
    score_map = score_map.reshape(cX * cY)
    result = jt.code(score_map.shape,
                     score_map.dtype, [score_map, link_map, changed],
                     cuda_header='''
                    @alias(score_map, in0)
                    @alias(link_map, in1)
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
                            const unsigned char cyx   = @score_map( iy*cX + ix);
                            
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
                            bool nym1x, nyxm1, nyxp1, nyp1x, nym1xm1, nym1xp1, nyp1xm1, nyp1xp1;
                            if(cyx) {
                                nym1x = (iy > 0)    ? ((cyx == (@score_map((iy-1)*cX +   ix))) && (@link_map(iy, ix, 6) || @link_map(iy-1,   ix, 7))) : false; // up
                                nyxm1 = (ix > 0)    ? ((cyx == (@score_map(iy    *cX + ix-1))) && (@link_map(iy, ix, 0) || @link_map(iy-1, ix-1, 3))) : false; // left
                                nyxp1 = (ix < cX-1) ? ((cyx == (@score_map(iy    *cX + ix+1))) && (@link_map(iy, ix, 3) || @link_map(iy,   ix+1, 0))) : false; // right
                                nyp1x = (iy > cY-1) ? ((cyx == (@score_map((iy+1)*cX +   ix))) && (@link_map(iy, ix, 7) || @link_map(iy+1,   ix, 6))) : false; // down

                                nym1xm1 = (iy > 0    && ix > 0   )    ? ((cyx == (@score_map((iy-1)*cX +   ix-1))) && (@link_map(iy, ix, 2) || @link_map(iy-1,  ix-1, 4))) : false; // up-left
                                nym1xp1 = (iy > 0    && ix < cX-1)    ? ((cyx == (@score_map((iy-1)*cX +   ix+1))) && (@link_map(iy, ix, 5) || @link_map(iy-1,  ix+1, 1))) : false; // up-right
                                nyp1xm1 = (iy < cY-1 && ix > 0   )    ? ((cyx == (@score_map((iy+1)*cX +   ix-1))) && (@link_map(iy, ix, 1) || @link_map(iy+1,  ix-1, 5))) : false; // down-left
                                nyp1xp1 = (iy < cY-1 && ix < cX-1)    ? ((cyx == (@score_map((iy+1)*cX +   ix+1))) && (@link_map(iy, ix, 4) || @link_map(iy+1,  ix+1, 2))) : false; // down-right
                            }
                            else {
                                nym1x = (iy > 0)    ? (cyx == (@score_map((iy-1)*cX +   ix))) : false; // up
                                nyxm1 = (ix > 0)    ? (cyx == (@score_map(iy    *cX + ix-1))) : false; // left
                                nyxp1 = (ix < cX-1) ? (cyx == (@score_map(iy    *cX + ix+1))) : false; // right
                                nyp1x = (iy > cY-1) ? (cyx == (@score_map((iy+1)*cX +   ix))) : false; // down

                                nym1xm1 = (iy > 0    && ix > 0   )    ? (cyx == (@score_map((iy-1)*cX +   ix-1))) : false; // up-left
                                nym1xp1 = (iy > 0    && ix < cX-1)    ? (cyx == (@score_map((iy-1)*cX +   ix+1))) : false; // up-right
                                nyp1xm1 = (iy < cY-1 && ix > 0   )    ? (cyx == (@score_map((iy+1)*cX +   ix-1))) : false; // down-left
                                nyp1xp1 = (iy < cY-1 && ix < cX-1)    ? (cyx == (@score_map((iy+1)*cX +   ix+1))) : false; // down-right
                            }

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
                                @in2(0) = 1;
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
                        cudaMemsetAsync(in2_p, 0, 4);
                        
                        // Label image
                        label_equivalence <<< grid, block >>>(@ARGS, cX, cY);

                        // Copy changed flag to host
                        cudaMemcpy(&changed, in2_p, sizeof(int32), cudaMemcpyDeviceToHost);
                        
                        // Resolve the labels
                        resolve_labels <<< resolve_grid, resolve_block >>>(@ARGS, cX, cY);
                    }
                    ''')

    result = result.reshape((cY, cX))

    value, _, cnt = jt.unique(result, return_inverse=True, return_counts=True)
    value = (cnt > result_comp_area_thresh) * value
    value = value[value != 0]

    map_result = jt.zeros((int(value.max().numpy()[0]) + 1), dtype=jt.uint32)
    map_result[value] = jt.index(value.shape)[0] + 1
    result = map_result[result]

    return result
