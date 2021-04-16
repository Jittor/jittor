
// =================================================================================================
// Project: 
// Exploring the performance of general matrix-multiplication on an NVIDIA Tesla K40m GPU.
//
// File information:
// Institution.... SURFsara <www.surfsara.nl>
// Author......... Cedric Nugteren <cedric.nugteren@surfsara.nl>
// Changed at..... 2014-11-07
// License........ MIT license
// Tab-size....... 4 spaces
// Line length.... 100 characters
//
// =================================================================================================

// Select a kernel
#define KERNEL 8

// Constants for kernels 1 -- 5
#define TS 32                        // The square-root of the 2D tile-size (== work-group dims)

// Constants for kernels 3, 5
#define WPT 8                        // The amount of work-per-thread, i.e. the thread-coarsening factor
#define RTS (TS/WPT)                 // The reduced tile-size in one dimension

// Constants for kernels 4, 7 -- 10
#define WIDTH 4                      // The vector-width (in number of floats)

// Constants for kernel 5
#define TSDK 16                      // The tile-size in dimension K (for kernel 5 only)
#define LPT ((TSDK*WPT)/(TS))        // The amount of loads-per-thread (assume TSN==TSM)

// Constants for kernels 6 -- 10
#define TSM 128                      // The tile-size in dimension M
#define TSN 128                      // The tile-size in dimension N
#define TSK 16                       // The tile-size in dimension K
#define WPTM 8                       // The amount of work-per-thread in dimension M
#define WPTN 8                       // The amount of work-per-thread in dimension N
#define RTSM (TSM/WPTM)              // The reduced tile-size in dimension M (== number of threads)
#define RTSN (TSN/WPTN)              // The reduced tile-size in dimension N (== number of threads)
#define LPTA ((TSK*WPTM*WPTN)/(TSN)) // The amount of loads-per-thread for A
#define LPTB ((TSK*WPTM*WPTN)/(TSM)) // The amount of loads-per-thread for B

// Constraints on settings for kernels 6 -- 10
// Note: TSM/WPTM has to be integer
// Note: TSN/WPTN has to be integer
// Note: TSM/WIDTH has to be integer
// Note: TSN/WIDTH has to be integer
// Note: (TSK*WPTM*WPTN)/(TSN*WIDTH) has to be integer
// Note: (TSK*WPTM*WPTN)/(TSM*WIDTH) has to be integer

// Constants for kernel 11 (mimicing clBlas)
#define THREADSX 8
#define THREADSY 8
#define RX 8
#define RY 4
#define RK (RY)

// Constants for the supporting transpose kernel
#define TRANSPOSEX 16
#define TRANSPOSEY 16

// Constants for the supporting padding kernels
#define PADDINGX 16
#define PADDINGY 16

// Macros for host and kernel code
#define MIN(a,b) ((a) > (b)) ? (b) : (a)
#define MAX(a,b) ((a) > (b)) ? (a) : (b)
#define CEIL_DIV(x,y) (((x) + (y) - 1) / (y))
#define MOD2(x,y) ((x) % (y))
#define DIV2(x,y) ((x) / (y))

// =================================================================================================
