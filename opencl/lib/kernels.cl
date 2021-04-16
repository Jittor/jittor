
// =================================================================================================
// Project: 
// Exploring the performance of general matrix-multiplication on an NVIDIA Tesla K40m GPU.
//
// File information:
// Institution.... SURFsara <www.surfsara.nl>
// Author......... Cedric Nugteren <cedric.nugteren@surfsara.nl>
// Changed at..... 2014-11-06
// License........ MIT license
// Tab-size....... 4 spaces
// Line length.... 100 characters
//
// =================================================================================================
//
// Matrices in column-major format
// A: K columns, M rows
// B: N columns, K rows
// C: N columns, M rows
//                         
//                   N     
//                o-----o  
//                |     |  
//              K | [B] |  
//                |     |  
//                o-----o  
//        K          N     
//    o-------o   o-----o  
//  M |  [A]  | M | [C] |  
//    |       |   |     |  
//    o-------o   o-----o  
//                         
//
// C-code for column-major matrix multiplication with alpha=1 and beta=0:
//
// for (int m=0; m<M; m++) {
//     for (int n=0; n<N; n++) {
//         float acc = 0.0f;
//         for (int k=0; k<K; k++) {
//             acc += A[k*M + m] * B[n*K + k];
//         }
//         C[n*M + m] = acc;
//     }
// }
//
// =================================================================================================

// Data-widths
#if WIDTH == 1
    typedef float floatX;
#elif WIDTH == 2
    typedef float2 floatX;
#elif WIDTH == 4
    typedef float4 floatX;
#elif WIDTH == 8
    typedef float8 floatX;
#endif

// =================================================================================================
#if KERNEL == 1

// First naive implementation
__kernel void myGEMM1(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C) {
    
    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)

    // Compute a single element (loop over K)
    float acc = 0.0f;
    for (int k=0; k<K; k++) {
        acc += A[k*M + globalRow] * B[globalCol*K + k];
    }

    // Store the result
    C[globalCol*M + globalRow] = acc;
}

#endif
// =================================================================================================
#if KERNEL == 2

// Tiled and coalesced version
__kernel void myGEMM2(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C) {
    
    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS)
    const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)

    // Local memory to fit a tile of TS*TS elements of A and B
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];

    // Initialise the accumulation register
    float acc = 0.0f;
    
    // Loop over all tiles
    const int numTiles = K/TS;
    for (int t=0; t<numTiles; t++) {

        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        Asub[col][row] = A[tiledCol*M + globalRow];
        Bsub[col][row] = B[globalCol*K + tiledRow];

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            acc += Asub[k][row] * Bsub[col][k];
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final result in C
    C[globalCol*M + globalRow] = acc;
}

#endif
// =================================================================================================
#if KERNEL == 3

// Increased the amount of work-per-thread by a factor WPT
__kernel void myGEMM3(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C) {
    
    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS/WPT == RTS)
    const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)

    // Local memory to fit a tile of TS*TS elements of A and B
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];

    // Initialise the accumulation registers
    float acc[WPT];
    for (int w=0; w<WPT; w++) {
        acc[w] = 0.0f;
    }
    
    // Loop over all tiles
    const int numTiles = K/TS;
    for (int t=0; t<numTiles; t++) {

        // Load one tile of A and B into local memory
        for (int w=0; w<WPT; w++) {
            const int tiledRow = TS*t + row;
            const int tiledCol = TS*t + col;
            Asub[col + w*RTS][row] = A[(tiledCol + w*RTS)*M + globalRow];
            Bsub[col + w*RTS][row] = B[(globalCol + w*RTS)*K + tiledRow];
        }

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            for (int w=0; w<WPT; w++) {
                acc[w] += Asub[k][row] * Bsub[col + w*RTS][k];
            }
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final results in C
    for (int w=0; w<WPT; w++) {
        C[(globalCol + w*RTS)*M + globalRow] = acc[w];
    }
}

#endif
// =================================================================================================
#if KERNEL == 4

// Use wider data types
__kernel void myGEMM4(const int M, const int N, const int K,
                      const __global floatX* A,
                      const __global floatX* B,
                      __global floatX* C) {

    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS/WIDTH)
    const int col = get_local_id(1); // Local col ID (max: TS)
    const int globalRow = (TS/WIDTH)*get_group_id(0) + row; // Row ID of C (0..M/WIDTH)
    const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)

    // Local memory to fit a tile of TS*TS elements of A and B
    __local floatX Asub[TS][TS/WIDTH];
    __local floatX Bsub[TS][TS/WIDTH];

    // Initialise the accumulation registers
    #if WIDTH == 1
        floatX acc = 0.0f;
    #elif WIDTH == 2
        floatX acc = { 0.0f, 0.0f };
    #elif WIDTH == 4
        floatX acc = { 0.0f, 0.0f, 0.0f, 0.0f };
    #elif WIDTH == 8
        floatX acc = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    #endif
    
    // Loop over all tiles
    const int numTiles = K/TS;
    for (int tile=0; tile<numTiles; tile++) {

        // Load one tile of A and B into local memory
        const int tiledRow = (TS/WIDTH)*tile + row;
        const int tiledCol = TS*tile + col;
        Asub[col][row] = A[tiledCol*(M/WIDTH) + globalRow];
        Bsub[col][row] = B[globalCol*(K/WIDTH) + tiledRow];

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform the computation for a single tile
        floatX vecA, vecB;
        float valB;
        for (int k=0; k<TS/WIDTH; k++) {
            vecB = Bsub[col][k];
            for (int w=0; w<WIDTH; w++) {
                vecA = Asub[WIDTH*k + w][row];
                #if WIDTH == 1
                    valB = vecB;
                    acc += vecA * valB;
                #elif WIDTH == 2
                    switch (w) {
                        case 0: valB = vecB.x; break;
                        case 1: valB = vecB.y; break;
                    }
                    acc.x += vecA.x * valB;
                    acc.y += vecA.y * valB;
                #elif WIDTH == 4
                    switch (w) {
                        case 0: valB = vecB.x; break;
                        case 1: valB = vecB.y; break;
                        case 2: valB = vecB.z; break;
                        case 3: valB = vecB.w; break;
                    }
                    acc.x += vecA.x * valB;
                    acc.y += vecA.y * valB;
                    acc.z += vecA.z * valB;
                    acc.w += vecA.w * valB;
                #elif WIDTH == 8
                    switch (w) {
                        case 0: valB = vecB.s0; break;
                        case 1: valB = vecB.s1; break;
                        case 2: valB = vecB.s2; break;
                        case 3: valB = vecB.s3; break;
                        case 4: valB = vecB.s4; break;
                        case 5: valB = vecB.s5; break;
                        case 6: valB = vecB.s6; break;
                        case 7: valB = vecB.s7; break;
                    }
                    acc.s0 += vecA.s0 * valB;
                    acc.s1 += vecA.s1 * valB;
                    acc.s2 += vecA.s2 * valB;
                    acc.s3 += vecA.s3 * valB;
                    acc.s4 += vecA.s4 * valB;
                    acc.s5 += vecA.s5 * valB;
                    acc.s6 += vecA.s6 * valB;
                    acc.s7 += vecA.s7 * valB;
                #endif
            }
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final results in C
    C[globalCol*(M/WIDTH) + globalRow] = acc;
}

#endif
// =================================================================================================
#if KERNEL == 5

// Pre-transpose the input matrix B and use rectangular tiles
__kernel void myGEMM5(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C) {

    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS/WPT == RTS)
    const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)

    // Local memory to fit a tile of A and B
    __local float Asub[TSDK][TS];
    __local float Bsub[TS][TSDK+2];

    // Initialise the accumulation registers
    float acc[WPT];
    for (int w=0; w<WPT; w++) {
        acc[w] = 0.0f;
    }
    
    // Loop over all tiles
    const int numTiles = K/TSDK;
    for (int t=0; t<numTiles; t++) {

        // Load one tile of A and B into local memory
        for (int l=0; l<LPT; l++) {
            const int tiledIndex = TSDK*t + col + l*RTS;
            int indexA = (tiledIndex)*M + TS*get_group_id(0) + row;
            int indexB = (tiledIndex)*N + TS*get_group_id(1) + row;
            Asub[col + l*RTS][row] = A[indexA];
            Bsub[row][col + l*RTS] = B[indexB];
        }

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform the computation for a single tile
        for (int k=0; k<TSDK; k++) {
            for (int w=0; w<WPT; w++) {
                acc[w] += Asub[k][row] * Bsub[col + w*RTS][k];
            }
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final results in C
    for (int w=0; w<WPT; w++) {
        C[(globalCol + w*RTS)*M + globalRow] = acc[w];
    }
}

#endif
// =================================================================================================
#if KERNEL == 6

// Use 2D register blocking (further increase in work per thread)
__kernel void myGEMM6(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C) {

    // Thread identifiers
    const int tidm = get_local_id(0); // Local row ID (max: TSM/WPTM == RTSM)
    const int tidn = get_local_id(1); // Local col ID (max: TSN/WPTN == RTSN)
    const int offsetM = TSM*get_group_id(0); // Work-group offset
    const int offsetN = TSN*get_group_id(1); // Work-group offset

    // Local memory to fit a tile of A and B
    __local float Asub[TSK][TSM];
    __local float Bsub[TSN][TSK+2];

    // Allocate register space
    float Areg;
    float Breg[WPTN];
    float acc[WPTM][WPTN];

    // Initialise the accumulation registers
    #pragma unroll
    for (int wm=0; wm<WPTM; wm++) {
        #pragma unroll
        for (int wn=0; wn<WPTN; wn++) {
            acc[wm][wn] = 0.0f;
        }
    }
    
    // Loop over all tiles
    const int numTiles = K/TSK;
    int t=0;
    do {

        // Load one tile of A and B into local memory
        #pragma unroll
        for (int la=0; la<LPTA; la++) {
            int tid = tidn*RTSM + tidm;
            int id = la*RTSN*RTSM + tid;
            int row = MOD2(id,TSM);
            int col = DIV2(id,TSM);
            int tiledIndex = TSK*t + col;
            Asub[col][row] = A[tiledIndex*M + offsetM + row];
            Bsub[row][col] = B[tiledIndex*N + offsetN + row];
        }

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Loop over the values of a single tile
        for (int k=0; k<TSK; k++) {

            // Cache the values of Bsub in registers
            #pragma unroll
            for (int wn=0; wn<WPTN; wn++) {
                int col = tidn + wn*RTSN;
                Breg[wn] = Bsub[col][k];
            }

            // Perform the computation
            #pragma unroll
            for (int wm=0; wm<WPTM; wm++) {
                int row = tidm + wm*RTSM;
                Areg = Asub[k][row];
                #pragma unroll
                for (int wn=0; wn<WPTN; wn++) {
                    acc[wm][wn] += Areg * Breg[wn];
                }
            }
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);

        // Next tile
        t++;
    } while (t<numTiles);

    // Store the final results in C
    #pragma unroll
    for (int wm=0; wm<WPTM; wm++) {
        int globalRow = offsetM + tidm + wm*RTSM;
        #pragma unroll
        for (int wn=0; wn<WPTN; wn++) {
            int globalCol = offsetN + tidn + wn*RTSN;
            C[globalCol*M + globalRow] = acc[wm][wn];
        }
    }
}

#endif
// =================================================================================================
#if KERNEL == 7

// Wider loads combined with 2D register blocking
__kernel void myGEMM7(const int M, const int N, const int K,
                      const __global floatX* A,
                      const __global floatX* B,
                      __global float* C) {

    // Thread identifiers
    const int tidm = get_local_id(0); // Local row ID (max: TSM/WPTM == RTSM)
    const int tidn = get_local_id(1); // Local col ID (max: TSN/WPTN == RTSN)
    const int offsetM = TSM*get_group_id(0); // Work-group offset
    const int offsetN = TSN*get_group_id(1); // Work-group offset

    // Local memory to fit a tile of A and B
    __local float Asub[TSK][TSM];
    __local float Bsub[TSK][TSN];

    // Allocate register space
    float Areg;
    float Breg[WPTN];
    float acc[WPTM][WPTN];

    // Initialise the accumulation registers
    #pragma unroll
    for (int wm=0; wm<WPTM; wm++) {
        #pragma unroll
        for (int wn=0; wn<WPTN; wn++) {
            acc[wm][wn] = 0.0f;
        }
    }
    
    // Loop over all tiles
    const int numTiles = K/TSK;
    int t=0;
    do {

        // Load one tile of A and B into local memory
        #pragma unroll
        for (int la=0; la<LPTA/WIDTH; la++) {
            int tid = tidn*RTSM + tidm;
            int id = la*RTSN*RTSM + tid;
            int row = MOD2(id,TSM/WIDTH);
            int col = DIV2(id,TSM/WIDTH);

            // Load the values (wide vector load)
            int tiledIndex = TSK*t + col;
            floatX vecA = A[tiledIndex*(M/WIDTH) + offsetM/WIDTH + row];
            floatX vecB = B[tiledIndex*(N/WIDTH) + offsetN/WIDTH + row];

            // Store the loaded vectors into local memory
            #if WIDTH == 1
                Asub[col][row] = vecA;
            #elif WIDTH == 2
                Asub[col][WIDTH*row + 0] = vecA.x;
                Asub[col][WIDTH*row + 1] = vecA.y;
            #elif WIDTH == 4
                Asub[col][WIDTH*row + 0] = vecA.x;
                Asub[col][WIDTH*row + 1] = vecA.y;
                Asub[col][WIDTH*row + 2] = vecA.z;
                Asub[col][WIDTH*row + 3] = vecA.w;
            #endif
            #if WIDTH == 1
                Bsub[col][row] = vecB;
            #elif WIDTH == 2
                Bsub[col][WIDTH*row + 0] = vecB.x;
                Bsub[col][WIDTH*row + 1] = vecB.y;
            #elif WIDTH == 4
                Bsub[col][WIDTH*row + 0] = vecB.x;
                Bsub[col][WIDTH*row + 1] = vecB.y;
                Bsub[col][WIDTH*row + 2] = vecB.z;
                Bsub[col][WIDTH*row + 3] = vecB.w;
            #endif
        }

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Loop over the values of a single tile
        #pragma unroll
        for (int k=0; k<TSK; k++) {

            // Cache the values of Bsub in registers
            #pragma unroll
            for (int wn=0; wn<WPTN; wn++) {
                int col = tidn + wn*RTSN;
                Breg[wn] = Bsub[k][col];
            }

            // Perform the computation
            #pragma unroll
            for (int wm=0; wm<WPTM; wm++) {
                int row = tidm + wm*RTSM;
                Areg = Asub[k][row];
                #pragma unroll
                for (int wn=0; wn<WPTN; wn++) {
                    acc[wm][wn] += Areg * Breg[wn];
                }
            }
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);

        // Next tile
        t++;
    } while (t<numTiles);

    // Store the final results in C
    #pragma unroll
    for (int wm=0; wm<WPTM; wm++) {
        int globalRow = offsetM + tidm + wm*RTSM;
        #pragma unroll
        for (int wn=0; wn<WPTN; wn++) {
            int globalCol = offsetN + tidn + wn*RTSN;
            C[globalCol*M + globalRow] = acc[wm][wn];
        }
    }
}

#endif
// =================================================================================================
#if KERNEL == 8

// CUDA and Kepler-specific optimisations (LDG and warp-shuffle)
__kernel void myGEMM8(const int M, const int N, const int K,
                      const __global floatX* A,
                      const __global floatX* B,
                      __global float* C) {

    // Thread identifiers
    const int tidm = get_local_id(0); // Local row ID (max: TSM/WPTM == RTSM)
    const int tidn = get_local_id(1); // Local col ID (max: TSN/WPTN == RTSN)
    const int offsetM = TSM*get_group_id(0); // Work-group offset
    const int offsetN = TSN*get_group_id(1); // Work-group offset

    // Local memory to fit a tile of A and B
    __local float Asub[TSK][TSM];
    __local float Bsub[TSK][TSN];

    // Allocate register space
    float Areg;
    float Breg[WPTN];
    float acc[WPTM][WPTN];

    // Initialise the accumulation registers
    #pragma unroll
    for (int wm=0; wm<WPTM; wm++) {
        #pragma unroll
        for (int wn=0; wn<WPTN; wn++) {
            acc[wm][wn] = 0.0f;
        }
    }
    
    // Loop over all tiles
    const int numTiles = K/TSK;
    int t=0;
    do {

        // Load one tile of A and B into local memory
        #pragma unroll
        for (int la=0; la<LPTA/WIDTH; la++) {
            int tid = tidn*RTSM + tidm;
            int id = la*RTSN*RTSM + tid;
            int row = MOD2(id,TSM/WIDTH);
            int col = DIV2(id,TSM/WIDTH);

            // Load the values (wide vector load)
            int tiledIndex = TSK*t + col;
            int indexA = tiledIndex*(M/WIDTH) + offsetM/WIDTH + row;
            int indexB = tiledIndex*(N/WIDTH) + offsetN/WIDTH + row;
            #ifdef USE_LDG
                floatX vecA = __ldg(&A[indexA]);
                floatX vecB = __ldg(&B[indexB]);
            #else
                floatX vecA = A[indexA];
                floatX vecB = B[indexB];
            #endif

            // Store the loaded vectors into local memory
            #if WIDTH == 1
                Asub[col][row] = vecA;
            #elif WIDTH == 2
                Asub[col][WIDTH*row + 0] = vecA.x;
                Asub[col][WIDTH*row + 1] = vecA.y;
            #elif WIDTH == 4
                Asub[col][WIDTH*row + 0] = vecA.x;
                Asub[col][WIDTH*row + 1] = vecA.y;
                Asub[col][WIDTH*row + 2] = vecA.z;
                Asub[col][WIDTH*row + 3] = vecA.w;
            #endif
            #if WIDTH == 1
                Bsub[col][row] = vecB;
            #elif WIDTH == 2
                Bsub[col][WIDTH*row + 0] = vecB.x;
                Bsub[col][WIDTH*row + 1] = vecB.y;
            #elif WIDTH == 4
                Bsub[col][WIDTH*row + 0] = vecB.x;
                Bsub[col][WIDTH*row + 1] = vecB.y;
                Bsub[col][WIDTH*row + 2] = vecB.z;
                Bsub[col][WIDTH*row + 3] = vecB.w;
            #endif
        }

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Loop over the values of a single tile
        #pragma unroll
        for (int k=0; k<TSK; k++) {

            // Cache the values of Bsub in registers
            #ifdef USE_SHUFFLE
                int col = tidn + (tidm % WPTN)*RTSN;
                float val = Bsub[k][col];
                #pragma unroll
                for (int wn=0; wn<WPTN; wn++) {
                    Breg[wn] = __shfl(val, wn, WPTN);
                }
            #else
                #pragma unroll
                for (int wn=0; wn<WPTN; wn++) {
                    int col = tidn + wn*RTSN;
                    Breg[wn] = Bsub[k][col];
                }
            #endif

            /*// Cache the values of Asub in registers
            #ifdef USE_SHUFFLE
                for (int wn=0; wn<WPTN; wn+=(32/RTSM)) {
                    int type = tidn % (32/RTSM);
                    int row = tidm + (wn+type)*RTSM;
                    float val = Asub[k][row];
                    Areg[wn] = __shfl_up(val, RTSM);
                    Areg[wn+1] = __shfl_down(val, RTSM);
                }
            #endif */

            // Perform the computation
            #pragma unroll
            for (int wm=0; wm<WPTM; wm++) {
                int row = tidm + wm*RTSM;
                Areg = Asub[k][row];
                #pragma unroll
                for (int wn=0; wn<WPTN; wn++) {
                    acc[wm][wn] += Areg * Breg[wn];
                }
            }
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);

        // Next tile
        t++;
    } while (t<numTiles);

    // Store the final results in C
    #pragma unroll
    for (int wm=0; wm<WPTM; wm++) {
        int globalRow = offsetM + tidm + wm*RTSM;
        #pragma unroll
        for (int wn=0; wn<WPTN; wn++) {
            int globalCol = offsetN + tidn + wn*RTSN;
            C[globalCol*M + globalRow] = acc[wm][wn];
        }
    }
}

#endif
// =================================================================================================
#if KERNEL == 9

// With pre-fetching
__kernel void myGEMM9(const int M, const int N, const int K,
                      const __global floatX* A,
                      const __global floatX* B,
                      __global float* C) {

    // Thread identifiers
    const int tidm = get_local_id(0); // Local row ID (max: TSM/WPTM == RTSM)
    const int tidn = get_local_id(1); // Local col ID (max: TSN/WPTN == RTSN)
    const int offsetM = TSM*get_group_id(0); // Work-group offset
    const int offsetN = TSN*get_group_id(1); // Work-group offset

    // Local memory to fit two tiles of A and B
    __local float Asub[2][TSK*TSM];
    __local float Bsub[2][TSK*TSN];

    // Allocate register space
    float Areg;
    float Breg[WPTN];
    float acc[WPTM][WPTN];

    // Initialise the accumulation registers
    #pragma unroll
    for (int wm=0; wm<WPTM; wm++) {
        #pragma unroll
        for (int wn=0; wn<WPTN; wn++) {
            acc[wm][wn] = 0.0f;
        }
    }

    // Load the first tile of A and B into local memory
    #pragma unroll
    for (int la=0; la<LPTA/WIDTH; la++) {
        int tid = tidn*RTSM + tidm;
        int id = la*RTSN*RTSM + tid;
        int row = MOD2(id,TSM/WIDTH);
        int col = DIV2(id,TSM/WIDTH);

        // Load the values (wide vector load)
        int tiledIndex = TSK*0 + col;
        int indexA = tiledIndex*(M/WIDTH) + offsetM/WIDTH + row;
        int indexB = tiledIndex*(N/WIDTH) + offsetN/WIDTH + row;
        #ifdef USE_LDG
            floatX vecA = __ldg(&A[indexA]);
            floatX vecB = __ldg(&B[indexB]);
        #else
            floatX vecA = A[indexA];
            floatX vecB = B[indexB];
        #endif

        // Store the loaded vectors into local memory
        #if WIDTH == 1
            Asub[0][col*TSM + row] = vecA;
        #elif WIDTH == 2
            Asub[0][col*TSM + WIDTH*row + 0] = vecA.x;
            Asub[0][col*TSM + WIDTH*row + 1] = vecA.y;
        #elif WIDTH == 4
            Asub[0][col*TSM + WIDTH*row + 0] = vecA.x;
            Asub[0][col*TSM + WIDTH*row + 1] = vecA.y;
            Asub[0][col*TSM + WIDTH*row + 2] = vecA.z;
            Asub[0][col*TSM + WIDTH*row + 3] = vecA.w;
        #endif
        #if WIDTH == 1
            Bsub[0][col*TSN + row] = vecB;
        #elif WIDTH == 2
            Bsub[0][col*TSN + WIDTH*row + 0] = vecB.x;
            Bsub[0][col*TSN + WIDTH*row + 1] = vecB.y;
        #elif WIDTH == 4
            Bsub[0][col*TSN + WIDTH*row + 0] = vecB.x;
            Bsub[0][col*TSN + WIDTH*row + 1] = vecB.y;
            Bsub[0][col*TSN + WIDTH*row + 2] = vecB.z;
            Bsub[0][col*TSN + WIDTH*row + 3] = vecB.w;
        #endif
    }

    // Synchronise
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Loop over all tiles
    const int numTiles = K/TSK;
    int t=0;
    do {

        // Load the next tile of A and B into local memory
        int tt = t + 1;
        if (tt < numTiles) {
            #pragma unroll
            for (int la=0; la<LPTA/WIDTH; la++) {
                int tid = tidn*RTSM + tidm;
                int id = la*RTSN*RTSM + tid;
                int row = MOD2(id,TSM/WIDTH);
                int col = DIV2(id,TSM/WIDTH);

                // Load the values (wide vector load)
                int tiledIndex = TSK*tt + col;
                int indexA = tiledIndex*(M/WIDTH) + offsetM/WIDTH + row;
                int indexB = tiledIndex*(N/WIDTH) + offsetN/WIDTH + row;
                #ifdef USE_LDG
                    floatX vecA = __ldg(&A[indexA]);
                    floatX vecB = __ldg(&B[indexB]);
                #else
                    floatX vecA = A[indexA];
                    floatX vecB = B[indexB];
                #endif

                // Store the loaded vectors into local memory
                #if WIDTH == 1
                    Asub[tt%2][col*TSM + row] = vecA;
                #elif WIDTH == 2
                    Asub[tt%2][col*TSM + WIDTH*row + 0] = vecA.x;
                    Asub[tt%2][col*TSM + WIDTH*row + 1] = vecA.y;
                #elif WIDTH == 4
                    Asub[tt%2][col*TSM + WIDTH*row + 0] = vecA.x;
                    Asub[tt%2][col*TSM + WIDTH*row + 1] = vecA.y;
                    Asub[tt%2][col*TSM + WIDTH*row + 2] = vecA.z;
                    Asub[tt%2][col*TSM + WIDTH*row + 3] = vecA.w;
                #endif
                #if WIDTH == 1
                    Bsub[tt%2][col*TSN + row] = vecB;
                #elif WIDTH == 2
                    Bsub[tt%2][col*TSN + WIDTH*row + 0] = vecB.x;
                    Bsub[tt%2][col*TSN + WIDTH*row + 1] = vecB.y;
                #elif WIDTH == 4
                    Bsub[tt%2][col*TSN + WIDTH*row + 0] = vecB.x;
                    Bsub[tt%2][col*TSN + WIDTH*row + 1] = vecB.y;
                    Bsub[tt%2][col*TSN + WIDTH*row + 2] = vecB.z;
                    Bsub[tt%2][col*TSN + WIDTH*row + 3] = vecB.w;
                #endif
            }
        }

        // Loop over the values of a single tile
        #pragma unroll
        for (int k=0; k<TSK; k++) {

            // Cache the values of Bsub in registers
            #pragma unroll
            for (int wn=0; wn<WPTN; wn++) {
                int col = tidn + wn*RTSN;
                Breg[wn] = Bsub[t%2][k*TSN + col];
            }

            // Perform the computation
            #pragma unroll
            for (int wm=0; wm<WPTM; wm++) {
                int row = tidm + wm*RTSM;
                Areg = Asub[t%2][k*TSM + row];
                #pragma unroll
                for (int wn=0; wn<WPTN; wn++) {
                    acc[wm][wn] += Areg * Breg[wn];
                }
            }
        }

        // Synchronise
        barrier(CLK_LOCAL_MEM_FENCE);

        // Next tile
        t++;
    } while (t<numTiles);

    // Store the final results in C
    #pragma unroll
    for (int wm=0; wm<WPTM; wm++) {
        int globalRow = offsetM + tidm + wm*RTSM;
        #pragma unroll
        for (int wn=0; wn<WPTN; wn++) {
            int globalCol = offsetN + tidn + wn*RTSN;
            C[globalCol*M + globalRow] = acc[wm][wn];
        }
    }
}

#endif
// =================================================================================================
#if KERNEL == 10

#define BK TSK
#define BN TSN
#define BM TSM
#define TX RTSM
#define TY RTSN
#define RX WPTM
#define RY WPTN

// With support for incomplete tiles and arbitrary input/output matrix sizes
__kernel void myGEMM10(const int M, const int N, const int K,
                       const __global floatX* A,
                       const __global floatX* B,
                       __global float* C) {

    // Thread identifiers
    const int tidm = get_local_id(0); // Local row ID (max: TSM/WPTM == RTSM)
    const int tidn = get_local_id(1); // Local col ID (max: TSN/WPTN == RTSN)
    const int gidm = get_group_id(0); // Work-group ID
    const int gidn = get_group_id(1); // Work-group ID
    const int tid = tidn*RTSM + tidm; // Global thread ID (max RTSM*RTSN)

    // Local memory to fit two tiles of A and B
    __local float Asub[2][TSK*TSM];
    __local float Bsub[2][TSK*TSN];

    // Allocate register space
    float Areg;
    float Breg[WPTN];
    float acc[WPTM][WPTN];

    // Initialise the accumulation registers
    #pragma unroll
    for (int wm=0; wm<WPTM; wm++) {
        #pragma unroll
        for (int wn=0; wn<WPTN; wn++) {
            acc[wm][wn] = 0.0f;
        }
    }

    // Tile A
    #pragma unroll
    for (int la=0; la<LPTA/WIDTH; la++) {
        int id = la*RTSN*RTSM + tid;
        int row = MOD2(id,TSM/WIDTH);
        int col = DIV2(id,TSM/WIDTH);

        // Load the value (wide vector load)
        int tiledIndex = TSK*0 + col;
        int indexA = tiledIndex*(M/WIDTH) + gidm*(TSM/WIDTH) + row;
        #ifdef USE_LDG
            floatX vecA = __ldg(&A[indexA]);
        #else
            floatX vecA = A[indexA];
        #endif

        // Store the loaded vector into local memory
        #if WIDTH == 1
            Asub[0][col*TSM + row] = vecA;
        #elif WIDTH == 2
            Asub[0][col*TSM + WIDTH*row + 0] = vecA.x;
            Asub[0][col*TSM + WIDTH*row + 1] = vecA.y;
        #elif WIDTH == 4
            Asub[0][col*TSM + WIDTH*row + 0] = vecA.x;
            Asub[0][col*TSM + WIDTH*row + 1] = vecA.y;
            Asub[0][col*TSM + WIDTH*row + 2] = vecA.z;
            Asub[0][col*TSM + WIDTH*row + 3] = vecA.w;
        #endif
    }

    // Tile B
    #pragma unroll
    for (int lb=0; lb<LPTB/WIDTH; lb++) {
        int id = lb*RTSN*RTSM + tid;
        int row = MOD2(id,TSN/WIDTH);
        int col = DIV2(id,TSN/WIDTH);

        // Load the value (wide vector load)
        int tiledIndex = TSK*0 + col;
        int indexB = tiledIndex*(N/WIDTH) + gidn*(TSN/WIDTH) + row;
        #ifdef USE_LDG
            floatX vecB = __ldg(&B[indexB]);
        #else
            floatX vecB = B[indexB];
        #endif

        // Store the loaded vector into local memory
        #if WIDTH == 1
            Bsub[0][col*TSN + row] = vecB;
        #elif WIDTH == 2
            Bsub[0][col*TSN + WIDTH*row + 0] = vecB.x;
            Bsub[0][col*TSN + WIDTH*row + 1] = vecB.y;
        #elif WIDTH == 4
            Bsub[0][col*TSN + WIDTH*row + 0] = vecB.x;
            Bsub[0][col*TSN + WIDTH*row + 1] = vecB.y;
            Bsub[0][col*TSN + WIDTH*row + 2] = vecB.z;
            Bsub[0][col*TSN + WIDTH*row + 3] = vecB.w;
        #endif
    }
    
    // Loop over all tiles
    const int numTiles = K/TSK;
    int t=0;
    do {

        // Synchronise
        barrier(CLK_LOCAL_MEM_FENCE);

        // Load the next tile of A and B into local memory
        int tt = t + 1;
        if (tt < numTiles) {

            // Tile A
            #pragma unroll
            for (int la=0; la<LPTA/WIDTH; la++) {
                int id = la*RTSN*RTSM + tid;
                int row = MOD2(id,TSM/WIDTH);
                int col = DIV2(id,TSM/WIDTH);

                // Load the value (wide vector load)
                int tiledIndex = TSK*tt + col;
                int indexA = tiledIndex*(M/WIDTH) + gidm*(TSM/WIDTH) + row;
                #ifdef USE_LDG
                    floatX vecA = __ldg(&A[indexA]);
                #else
                    floatX vecA = A[indexA];
                #endif

                // Store the loaded vector into local memory
                #if WIDTH == 1
                    Asub[tt%2][col*TSM + row] = vecA;
                #elif WIDTH == 2
                    Asub[tt%2][col*TSM + WIDTH*row + 0] = vecA.x;
                    Asub[tt%2][col*TSM + WIDTH*row + 1] = vecA.y;
                #elif WIDTH == 4
                    Asub[tt%2][col*TSM + WIDTH*row + 0] = vecA.x;
                    Asub[tt%2][col*TSM + WIDTH*row + 1] = vecA.y;
                    Asub[tt%2][col*TSM + WIDTH*row + 2] = vecA.z;
                    Asub[tt%2][col*TSM + WIDTH*row + 3] = vecA.w;
                #endif
            }

            // Tile B
            #pragma unroll
            for (int lb=0; lb<LPTB/WIDTH; lb++) {
                int id = lb*RTSN*RTSM + tid;
                int row = MOD2(id,TSN/WIDTH);
                int col = DIV2(id,TSN/WIDTH);

                // Load the value (wide vector load)
                int tiledIndex = TSK*tt + col;
                int indexB = tiledIndex*(N/WIDTH) + gidn*(TSN/WIDTH) + row;
                #ifdef USE_LDG
                    floatX vecB = __ldg(&B[indexB]);
                #else
                    floatX vecB = B[indexB];
                #endif

                // Store the loaded vector into local memory
                #if WIDTH == 1
                    Bsub[tt%2][col*TSN + row] = vecB;
                #elif WIDTH == 2
                    Bsub[tt%2][col*TSN + WIDTH*row + 0] = vecB.x;
                    Bsub[tt%2][col*TSN + WIDTH*row + 1] = vecB.y;
                #elif WIDTH == 4
                    Bsub[tt%2][col*TSN + WIDTH*row + 0] = vecB.x;
                    Bsub[tt%2][col*TSN + WIDTH*row + 1] = vecB.y;
                    Bsub[tt%2][col*TSN + WIDTH*row + 2] = vecB.z;
                    Bsub[tt%2][col*TSN + WIDTH*row + 3] = vecB.w;
                #endif
            }
        }

        // Loop over the values of a single tile
        #pragma unroll
        for (int k=0; k<TSK; k++) {

            // Cache the values of Bsub in registers
            #pragma unroll
            for (int wn=0; wn<WPTN; wn++) {
                int col = tidn + wn*RTSN;
                Breg[wn] = Bsub[t%2][k*TSN + col];
            }

            // Perform the computation
            #pragma unroll
            for (int wm=0; wm<WPTM; wm++) {
                int row = tidm + wm*RTSM;
                Areg = Asub[t%2][k*TSM + row];
                #pragma unroll
                for (int wn=0; wn<WPTN; wn++) {
                    acc[wm][wn] += Areg * Breg[wn];
                }
            }
        }

        // Next tile
        t++;
    } while (t<numTiles);

    // Store the final results in C
    #pragma unroll
    for (int wm=0; wm<WPTM; wm++) {
        int globalRow = gidm*TSM + tidm + wm*RTSM;
        #pragma unroll
        for (int wn=0; wn<WPTN; wn++) {
            int globalCol = gidn*TSN + tidn + wn*RTSN;
            C[globalCol*M + globalRow] = acc[wm][wn];
        }
    }
}

#endif
// =================================================================================================
#if KERNEL == 11

// Typedefs for clBlas-mimic kernel (myGEMM11)
#if RX == 2
    typedef float2 floatA;
    typedef float2 floatC;
#elif RX == 4
    typedef float4 floatA;
    typedef float4 floatC;
#elif RX == 8
    typedef float8 floatA;
    typedef float8 floatC;
#endif
#if RK == 2
    typedef float2 floatB;
#elif RK == 4
    typedef float4 floatB;
#elif RK == 8
    typedef float8 floatB;
#endif

// Mimic clBlas (4x8 register tiling with vector data-types)
__kernel void myGEMM11(const int M, const int N, const int K,
                       const __global floatA* restrict A,
                       const __global floatB* restrict B,
                       __global floatC* C) {
    
    // Allocate register space
    float aReg[RK][RX];
    float bReg[RY][RK];
    float acc[RY][RX];

    // Initialise the accumulation registers
    #pragma unroll
    for (int y=0; y<RY; y++) {
        for (int x=0; x<RX; x++) {
            acc[y][x] = 0.0;
        }
    }

    // Loop over all tiles
    const int numTiles = K/RK;
    for (int t=0; t<numTiles; t++) {

        // Load a tile of A and B into register memory
        #pragma unroll
        for (int y=0; y<RY; y++) {

            // Load the data
            floatA aVec = A[(RK*t + y)*(M/RX) + get_global_id(0)];
            floatB bVec = B[(RY*get_global_id(1) + y)*numTiles + t];

            // Store the vector of A into registers
            #if RX == 2
                aReg[y][0] = aVec.x;
                aReg[y][1] = aVec.y;
            #elif RX == 4
                aReg[y][0] = aVec.x;
                aReg[y][1] = aVec.y;
                aReg[y][2] = aVec.z;
                aReg[y][3] = aVec.w;
            #elif RX == 8
                aReg[y][0] = aVec.s0;
                aReg[y][1] = aVec.s1;
                aReg[y][2] = aVec.s2;
                aReg[y][3] = aVec.s3;
                aReg[y][4] = aVec.s4;
                aReg[y][5] = aVec.s5;
                aReg[y][6] = aVec.s6;
                aReg[y][7] = aVec.s7;
            #endif

            // Store the vector of B into registers
            #if RK == 2
                bReg[y][0] = bVec.x;
                bReg[y][1] = bVec.y;
            #elif RK == 4
                bReg[y][0] = bVec.x;
                bReg[y][1] = bVec.y;
                bReg[y][2] = bVec.z;
                bReg[y][3] = bVec.w;
            #elif RK == 8
                bReg[y][0] = bVec.s0;
                bReg[y][1] = bVec.s1;
                bReg[y][2] = bVec.s2;
                bReg[y][3] = bVec.s3;
                bReg[y][4] = bVec.s4;
                bReg[y][5] = bVec.s5;
                bReg[y][6] = bVec.s6;
                bReg[y][7] = bVec.s7;
            #endif
        }

        // Perform the computations
        #pragma unroll
        for (int k=0; k<RK; k++) {
            #pragma unroll
            for (int y=0; y<RY; y++) {
                #pragma unroll
                for (int x=0; x<RX; x++) {
                    acc[y][x] += aReg[k][x] * bReg[y][k];
                }
            }
        }
    }

    // Store the final results in C
    #pragma unroll
    for (int y=0; y<RY; y++) {
        floatC accVec;
        #if RX == 2
            accVec.x = acc[y][0];
            accVec.y = acc[y][1];
        #elif RX == 4
            accVec.x = acc[y][0];
            accVec.y = acc[y][1];
            accVec.z = acc[y][2];
            accVec.w = acc[y][3];
        #elif RX == 8
            accVec.s0 = acc[y][0];
            accVec.s1 = acc[y][1];
            accVec.s2 = acc[y][2];
            accVec.s3 = acc[y][3];
            accVec.s4 = acc[y][4];
            accVec.s5 = acc[y][5];
            accVec.s6 = acc[y][6];
            accVec.s7 = acc[y][7];
        #endif
        C[(y + RY*get_global_id(1)) * (M/RX) + get_global_id(0)] = accVec;
    }
}

#endif
// =================================================================================================

// Simple transpose kernel for a P * Q matrix
__kernel void transpose(const int P, const int Q,
                        const __global float* input,
                        __global float* output) {
    
    // Thread identifiers
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    const int ID0 = get_group_id(0)*TRANSPOSEX + tx; // 0..P
    const int ID1 = get_group_id(1)*TRANSPOSEY + ty; // 0..Q

    // Set-up the local memory for shuffling
    __local float buffer[TRANSPOSEX][TRANSPOSEY];

    // Swap the x and y coordinates to perform the rotation (coalesced)
    if (ID0 < P && ID1 < Q) {
        buffer[ty][tx] = input[ID1*P + ID0];
    }

    // Synchronise all threads
    barrier(CLK_LOCAL_MEM_FENCE);

    // We don't have to swap the x and y thread indices here,
    // because that's already done in the local memory
    const int newID0 = get_group_id(1)*TRANSPOSEY + tx;
    const int newID1 = get_group_id(0)*TRANSPOSEX + ty;

    // Store the transposed result (coalesced)
    if (newID0 < Q && newID1 < P) {
        output[newID1*Q + newID0] = buffer[tx][ty];
    }
}

// =================================================================================================

// Pad the P * Q matrix with zeroes to form a P_XL * Q_XL matrix
__kernel void paddingAddZeroes(const int P, const int Q,
                               const __global float* input,
                               const int P_XL, const int Q_XL,
                               __global float* output) {
    
    // Thread identifiers
    const int tx = get_group_id(0)*PADDINGX + get_local_id(0); // 0..P_XL in blocks of PADDINGX
    const int ty = get_group_id(1)*PADDINGY + get_local_id(1); // 0..Q_XL in blocks of PADDINGY

    // Check whether we are within bounds of the XL matrix
    if (tx < P_XL && ty < Q_XL) {

        // Copy the input or pad a zero
        float value;
        if (tx < P && ty < Q) {
            value = input[ty*P + tx];
        }
        else {
            value = 0.0f;
        }

        // Store the result
        output[ty*P_XL + tx] = value;
    }
}

// =================================================================================================

// Remove padded values from a P_XL * Q_XL matrix to form a P * Q matrix
__kernel void paddingRemoveZeroes(const int P_XL, const int Q_XL,
                                  const __global float* input,
                                  const int P, const int Q,
                                  __global float* output) {
    
    // Thread identifiers
    const int tx = get_group_id(0)*PADDINGX + get_local_id(0); // 0..P in blocks of PADDINGX
    const int ty = get_group_id(1)*PADDINGY + get_local_id(1); // 0..Q in blocks of PADDINGY


    // Only store the result if within P * Q bounds
    if (tx < P && ty < Q) {
        output[ty*P + tx] = input[ty*P_XL + tx];
    }
}

// =================================================================================================
