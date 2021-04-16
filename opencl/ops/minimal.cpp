
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
// Compilation example:
// g++ -O3 -I$OPENCL_DIR/include minimal.cpp -o minimal -lOpenCL
// 
// =================================================================================================

// Includes
#include <stdio.h>
#include <sys/time.h>
#include <CL/cl.h>
#include "opencl_matmul_helper.h"
// =================================================================================================

// Repeat all kernels multiple times to get an average timing result
#define NUM_RUNS 1

// Size of the matrices - K, M, N (squared)
#define SIZE 64

// Threadblock sizes (e.g. for kernels myGEMM1 or myGEMM2)
#define TS 1

// =================================================================================================

// Set the kernel as a string (better to do this in a separate file though)
namespace jittor{
const char *kernelstring =
    "__kernel void myGEMM1(const int M, const int N, const int K,"
    "                      const __global float* A,"
    "                      const __global float* B,"
    "                      __global float* C) {"
    "    const int globalRow = get_global_id(0);"
    "    const int globalCol = get_global_id(1);"
    "    float acc = 0.0f;"
    "    for (int k=0; k<K; k++) {"
    "        acc += A[k*M + globalRow] * B[globalCol*K + k];"
    "    }"
    "    C[globalCol*M + globalRow] = acc;"
    "}";

// =================================================================================================

// Matrix-multiplication using a custom OpenCL SGEMM kernel.
void myclblas(float* A, float* B, float* C,
              int K, int M, int N) {
    // Timers
    struct timeval Tvalue;
    struct timezone dummy;
    /*
    for(int i=0;i<K*N;i++){
        printf("%.3f\n",B[i]);
    }
    */
    // int K = SIZE;
    // int M = SIZE;
    // int N = SIZE;
    // Configure the OpenCL environment
    printf(">>> Initializing OpenCL...\n");
    cl_platform_id platform = 0;
    clGetPlatformIDs(1, &platform, NULL);
    cl_device_id device = 0;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);
    char deviceName[1024];
    clGetDeviceInfo(device, CL_DEVICE_NAME, 1024, deviceName, NULL);
    cl_event event = NULL;

    // Compile the kernel
    cl_program program = clCreateProgramWithSource(context, 1, &kernelstring, NULL, NULL);
    clBuildProgram(program, 0, NULL, "", NULL, NULL);
    // Check for compilation errors
    size_t logSize;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
    char* messages = (char*)malloc((1+logSize)*sizeof(char));
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, messages, NULL);
    messages[logSize] = '\0';
    if (logSize > 10) { printf(">>> Compiler message: %s\n", messages); }
    free(messages);

    // Prepare OpenCL memory objects
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY,  M*K*sizeof(float), NULL, NULL);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY,  K*N*sizeof(float), NULL, NULL);
    cl_mem bufC = clCreateBuffer(context, CL_MEM_READ_WRITE, M*N*sizeof(float), NULL, NULL);

    // Copy matrices to the GPU
    clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, M*K*sizeof(float), A, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, K*N*sizeof(float), B, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufC, CL_TRUE, 0, M*N*sizeof(float), C, 0, NULL, NULL);

    // Configure the myGEMM kernel and set its arguments
    cl_kernel kernel = clCreateKernel(program, "myGEMM1", NULL);
    clSetKernelArg(kernel, 0, sizeof(int), (void*)&M);
    clSetKernelArg(kernel, 1, sizeof(int), (void*)&N);
    clSetKernelArg(kernel, 2, sizeof(int), (void*)&K);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&bufA);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&bufB);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&bufC);

    // Start the timed loop
    printf(">>> Starting %d myGEMM runs...\n", NUM_RUNS);
    gettimeofday(&Tvalue, &dummy);
    double starttime = (double)Tvalue.tv_sec + 1.0e-6*((double)Tvalue.tv_usec);
    for (int r=0; r<NUM_RUNS; r++) {

        // Run the myGEMM kernel
        const size_t local[2] = { (size_t)TS, (size_t)TS };
        const size_t global[2] = { (size_t)M, (size_t)N };
        clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, &event);

        // Wait for calculations to be finished
        clWaitForEvents(1, &event);
    }
    
    // End the timed loop
    gettimeofday(&Tvalue, &dummy);
    double endtime = (double)Tvalue.tv_sec + 1.0e-6*((double)Tvalue.tv_usec);
    double runtime = (endtime - starttime) / (double)NUM_RUNS;
    double gflop = ((long)K * (long)M * (long)N * 2) / (1000*1000*1000);
    printf(">>> Done: took %.3lf seconds per run, %.1lf GFLOPS\n", runtime, gflop/runtime);

    // Copy the output matrix C back to the CPU memory
    clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, M*N*sizeof(float), C, 0, NULL, NULL);
    for(int i=0;i<M*N;i++){
        printf("%.3f\n",C[i]);
    }
    // Free the OpenCL memory objects
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);

    // Clean-up OpenCL 
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
}
}
// =================================================================================================
