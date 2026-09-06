#ifdef JIT_cuda
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "type/cuda_atomic.h"
#include "indexing_codegen.h"

// The common indexing body is expanded before KernelIR builds the CUDA kernel.
// Colliding scatter reductions need raw-IEEE atomics; its storage is copied
// verbatim and never transformed to the ordered-integer reduction encoding.
#define indexing_backend_copy() if (op != ip) checkCudaErrors(cudaMemcpyAsync(op, ip, out->size, cudaMemcpyDeviceToDevice, 0));
#define indexing_backend_void() op[iid] = (Ti)dp[did]
#define indexing_backend_add() atomicAdd(&op[iid], (Ti)dp[did])
#define indexing_backend_maximum() cuda_atomic_max_rmw(&op[iid], (Ti)dp[did])
#define indexing_backend_minimum() cuda_atomic_min_rmw(&op[iid], (Ti)dp[did])
#define indexing_backend_multiply() cuda_atomic_mul(&op[iid], (Ti)dp[did])
// Conditional evaluation receives JIT definitions, not the local macro table.
// Select by OP explicitly instead of testing locally defined macro names.
#define indexing_backend_update() @if(@strcmp(@OP,void)==0, @expand_macro(indexing_backend_void), @if(@strcmp(@OP,add)==0, @expand_macro(indexing_backend_add), @if(@strcmp(@OP,maximum)==0, @expand_macro(indexing_backend_maximum), @if(@strcmp(@OP,minimum)==0, @expand_macro(indexing_backend_minimum), @if(@strcmp(@OP,multiply)==0, @expand_macro(indexing_backend_multiply), op[iid] = @expand_op(@OP, @Ti, op[iid], @Ti, dp[did], @Td))))));
#endif
