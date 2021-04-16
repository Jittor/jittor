
// =================================================================================================
// Project: 
// Exploring the performance of general matrix-multiplication on an NVIDIA Tesla K40m GPU.
//
// File information:
// Institution.... SURFsara <www.surfsara.nl>
// Author......... Cedric Nugteren <cedric.nugteren@surfsara.nl>
// Changed at..... 2014-11-17
// License........ MIT license
// Tab-size....... 4 spaces
// Line length.... 100 characters
//
// =================================================================================================

#include <CL/cl.h>
#include "opencl_matmul_helper.h"
#include <errno.h>
#include <unistd.h>
// Set the locations of the OpenCL kernel files
#define CL_INCLUDE_FILE "settings.h"
#define CL_KERNEL_FILE "kernels.cl"
// Determine the location where to output the PTX code
#define CL_PTX_FILE "bin/myGEMM.cl.ptx"

// Define OpenCL compiler options, such as "-cl-nv-maxrregcount=127"
#define COMPILER_OPTIONS ""

namespace jittor{
// Forward declaration of the OpenCL error checking function
void checkError(cl_int error, int line);

// =================================================================================================
// =================================================================================================

// Matrix-multiplication using a custom OpenCL SGEMM kernel. This function also copies the input
// matrices to the GPU, runs SGEMM, and copies the output matrix back to the CPU.
void myclblas(float* A, float* B, float* C,
              int K, int M, int N) {

    // In case of myGEMM10, compute matrix sizes K, M, N as rounded-up to form complete tiles
    #if KERNEL == 10
        int K_XL = CEIL_DIV(K, TSK) * TSK;
        int M_XL = CEIL_DIV(M, TSM) * TSM;
        int N_XL = CEIL_DIV(N, TSN) * TSN;
    #else
        int K_XL = K;
        int M_XL = M;
        int N_XL = N;
    #endif

    // Define OpenCL variables
    cl_int err;
    cl_platform_id platform = 0;
    cl_device_id device = 0;
    cl_device_id devices[MAX_NUM_DEVICES];
    cl_uint numDevices = 0;
    cl_context_properties props[3] = {CL_CONTEXT_PLATFORM, 0, 0};
    cl_context context = 0;
    cl_command_queue queue = 0;
    cl_event event = NULL;
    cl_program program = NULL;
    char deviceName[MAX_DEVICE_NAME];

    // Configure the OpenCL environment
    err = clGetPlatformIDs(1, &platform, NULL);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
    device = devices[CURRENT_DEVICE];
    props[1] = (cl_context_properties)platform;
    context = clCreateContext(props, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, MAX_DEVICE_NAME, deviceName, NULL);
    checkError(err,__LINE__);
    //printf("## %d devices, running on %d: '%s'\n", numDevices, CURRENT_DEVICE, deviceName);

    // Read the kernel file from disk
    long sizeHeader, sizeSource;
    char* header = readKernelFile(CL_INCLUDE_FILE, &sizeHeader);
    char* source = readKernelFile(CL_KERNEL_FILE, &sizeSource);
    long size = 2 + sizeHeader + sizeSource;
    char* code = (char*)malloc(size*sizeof(char));
    for (int c=0; c<size; c++) { code[c] = '\0'; }
    strcat(code, header);
    strcat(code, source);
    const char* constCode = code;
    free(header);
    free(source);

    // Compile the kernel file
    program = clCreateProgramWithSource(context, 1, &constCode, NULL, &err);
    checkError(err,__LINE__);
    err = clBuildProgram(program, 0, NULL, COMPILER_OPTIONS, NULL, NULL);

    // Check for compilation errors
    size_t logSize;
    err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
    checkError(err,__LINE__);
    char* messages = (char*)malloc((1+logSize)*sizeof(char));
    err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, messages, NULL);
    checkError(err,__LINE__);
    messages[logSize] = '\0';
    //if (logSize > 10) { printf("## Compiler message: %s\n", messages); }
    free(messages);

    // Retrieve the PTX code from the OpenCL compiler and output it to disk
    size_t binSize;
    err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binSize, NULL);
    checkError(err,__LINE__);
    unsigned char *bin = (unsigned char *)malloc(binSize);
    err = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char *), &bin, NULL);
    checkError(err,__LINE__);
    FILE* file = fopen(CL_PTX_FILE, "wb");
    fwrite(bin, sizeof(char), binSize, file);
    fclose(file);
    free(bin);

    // Prepare OpenCL memory objects
    cl_mem bufA    = clCreateBuffer(context, CL_MEM_READ_ONLY,  M*K*sizeof(*A), NULL, &err);
    cl_mem bufB    = clCreateBuffer(context, CL_MEM_READ_ONLY,  K*N*sizeof(*B), NULL, &err);
    cl_mem bufB_TR = clCreateBuffer(context, CL_MEM_READ_ONLY,  N*K*sizeof(*B), NULL, &err);
    cl_mem bufC    = clCreateBuffer(context, CL_MEM_READ_WRITE, M*N*sizeof(*C), NULL, &err);
    checkError(err,__LINE__);

    // Copy matrices to the GPU (also C to erase the results of the previous run)
    err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, M*K*sizeof(*A), A, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, K*N*sizeof(*B), B, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufC, CL_TRUE, 0, M*N*sizeof(*C), C, 0, NULL, NULL);
    checkError(err,__LINE__);

    // Create extra objects for rounded-up sizes (only needed in case of myGEMM10)
    cl_mem bufA_XL    = clCreateBuffer(context, CL_MEM_READ_ONLY,  M_XL*K_XL*sizeof(*A), NULL, &err);
    cl_mem bufB_TR_XL = clCreateBuffer(context, CL_MEM_READ_ONLY,  N_XL*K_XL*sizeof(*B), NULL, &err);
    cl_mem bufC_XL    = clCreateBuffer(context, CL_MEM_READ_WRITE, M_XL*N_XL*sizeof(*C), NULL, &err);
    checkError(err,__LINE__);

    // Configure the myGEMM kernel
    char kernelname[100];
    sprintf(kernelname, "myGEMM%d", KERNEL);
    cl_kernel kernel1 = clCreateKernel(program, kernelname, &err);
    checkError(err,__LINE__);

    // Set the arguments of the myGEMM kernel
    #if KERNEL == 10
        err = clSetKernelArg(kernel1, 0, sizeof(int), (void*)&M_XL);
        err = clSetKernelArg(kernel1, 1, sizeof(int), (void*)&N_XL);
        err = clSetKernelArg(kernel1, 2, sizeof(int), (void*)&K_XL);
        err = clSetKernelArg(kernel1, 3, sizeof(cl_mem), (void*)&bufA_XL);
        err = clSetKernelArg(kernel1, 4, sizeof(cl_mem), (void*)&bufB_TR_XL);
        err = clSetKernelArg(kernel1, 5, sizeof(cl_mem), (void*)&bufC_XL);
    #else
        err = clSetKernelArg(kernel1, 0, sizeof(int), (void*)&M);
        err = clSetKernelArg(kernel1, 1, sizeof(int), (void*)&N);
        err = clSetKernelArg(kernel1, 2, sizeof(int), (void*)&K);
        err = clSetKernelArg(kernel1, 3, sizeof(cl_mem), (void*)&bufA);
        #if KERNEL == 5 || KERNEL == 6 || KERNEL == 7 || KERNEL == 8 || KERNEL == 9
            err = clSetKernelArg(kernel1, 4, sizeof(cl_mem), (void*)&bufB_TR);
        #else
            err = clSetKernelArg(kernel1, 4, sizeof(cl_mem), (void*)&bufB);
        #endif
        err = clSetKernelArg(kernel1, 5, sizeof(cl_mem), (void*)&bufC);
    #endif
    checkError(err,__LINE__);

    // Configure the supporting transpose kernel and set its arguments (only for certain myGEMMs)
    #if KERNEL == 5 || KERNEL == 6 || KERNEL == 7 || KERNEL == 8 || KERNEL == 9 || KERNEL == 10
        cl_kernel kernel2 = clCreateKernel(program, "transpose", &err);
        checkError(err,__LINE__);
        err = clSetKernelArg(kernel2, 0, sizeof(int), (void*)&K);
        err = clSetKernelArg(kernel2, 1, sizeof(int), (void*)&N);
        err = clSetKernelArg(kernel2, 2, sizeof(cl_mem), (void*)&bufB);
        err = clSetKernelArg(kernel2, 3, sizeof(cl_mem), (void*)&bufB_TR);
        checkError(err,__LINE__);
        const size_t tLocal[2] = { TRANSPOSEX, TRANSPOSEY };
        const size_t tGlobal[2] = { (size_t)K, (size_t)N };
    #endif

    // Configure the supporting padding kernels and set their arguments (only for myGEMM10)
    #if KERNEL == 10
        cl_kernel kernel3a = clCreateKernel(program, "paddingAddZeroes", &err);
        checkError(err,__LINE__);
        err = clSetKernelArg(kernel3a, 0, sizeof(int), (void*)&M);
        err = clSetKernelArg(kernel3a, 1, sizeof(int), (void*)&K);
        err = clSetKernelArg(kernel3a, 2, sizeof(cl_mem), (void*)&bufA);
        err = clSetKernelArg(kernel3a, 3, sizeof(int), (void*)&M_XL);
        err = clSetKernelArg(kernel3a, 4, sizeof(int), (void*)&K_XL);
        err = clSetKernelArg(kernel3a, 5, sizeof(cl_mem), (void*)&bufA_XL);
        checkError(err,__LINE__);
        cl_kernel kernel3b = clCreateKernel(program, "paddingAddZeroes", &err);
        checkError(err,__LINE__);
        err = clSetKernelArg(kernel3b, 0, sizeof(int), (void*)&N);
        err = clSetKernelArg(kernel3b, 1, sizeof(int), (void*)&K);
        err = clSetKernelArg(kernel3b, 2, sizeof(cl_mem), (void*)&bufB_TR);
        err = clSetKernelArg(kernel3b, 3, sizeof(int), (void*)&N_XL);
        err = clSetKernelArg(kernel3b, 4, sizeof(int), (void*)&K_XL);
        err = clSetKernelArg(kernel3b, 5, sizeof(cl_mem), (void*)&bufB_TR_XL);
        checkError(err,__LINE__);
        cl_kernel kernel3c = clCreateKernel(program, "paddingRemoveZeroes", &err);
        checkError(err,__LINE__);
        err = clSetKernelArg(kernel3c, 0, sizeof(int), (void*)&M_XL);
        err = clSetKernelArg(kernel3c, 1, sizeof(int), (void*)&N_XL);
        err = clSetKernelArg(kernel3c, 2, sizeof(cl_mem), (void*)&bufC_XL);
        err = clSetKernelArg(kernel3c, 3, sizeof(int), (void*)&M);
        err = clSetKernelArg(kernel3c, 4, sizeof(int), (void*)&N);
        err = clSetKernelArg(kernel3c, 5, sizeof(cl_mem), (void*)&bufC);
        checkError(err,__LINE__);
        const size_t pLocal[2] = { PADDINGX, PADDINGY };
        const size_t pAGlobal[2] = { (size_t)M_XL, (size_t)K_XL };
        const size_t pBGlobal[2] = { (size_t)N_XL, (size_t)K_XL };
        const size_t pCGlobal[2] = { (size_t)M, (size_t)N };
    #endif

    // Configure the thread/work-group dimensions of the myGEMM kernel
    #if KERNEL == 1 || KERNEL == 2
        const size_t local[2] = { TS, TS };
        const size_t global[2] = { (size_t)M, (size_t)N };
    #elif KERNEL == 3 || KERNEL == 5
        const size_t local[2] = { TS, TS/WPT };
        const size_t global[2] = { (size_t)M, (size_t)(N/WPT) };
    #elif KERNEL == 4
        const size_t local[2] = { TS/WIDTH, TS };
        const size_t global[2] = { (size_t)(M/WIDTH), (size_t)N };
    #elif KERNEL == 6 || KERNEL == 7 || KERNEL == 8 || KERNEL == 9
        const size_t local[2] = { TSM/WPTM, TSN/WPTN };
        const size_t global[2] = { (size_t)(M/WPTM), (size_t)(N/WPTN) };
    #elif KERNEL == 10
        const size_t local[2] = { TSM/WPTM, TSN/WPTN };
        const size_t global[2] = { (size_t)(M_XL/WPTM), (size_t)(N_XL/WPTN) };
    #elif KERNEL == 11
        const size_t local[2] = { THREADSX, THREADSY };
        const size_t global[2] = { (size_t)(M/RX), (size_t)(N/RY) };
    #endif

    // Start the timed loop
    // double startTime = opencl_timer();
    for (int r=0; r<NUM_RUNS; r++) {

        // Run the transpose kernel first
        #if KERNEL == 5 || KERNEL == 6 || KERNEL == 7 || KERNEL == 8 || KERNEL == 9 || KERNEL == 10
            err = clEnqueueNDRangeKernel(queue, kernel2, 2, NULL, tGlobal, tLocal, 0, NULL, &event);
        #endif

        // Make the inputs extra large with padded zeros
        #if KERNEL == 10
            err = clEnqueueNDRangeKernel(queue, kernel3a, 2, NULL, pAGlobal, pLocal, 0, NULL, &event);
            err = clEnqueueNDRangeKernel(queue, kernel3b, 2, NULL, pBGlobal, pLocal, 0, NULL, &event);
        #endif

        // Run the myGEMM kernel
        err = clEnqueueNDRangeKernel(queue, kernel1, 2, NULL, global, local, 0, NULL, &event);

        // Remove padded zeroes from the larger output
        #if KERNEL == 10
            err = clEnqueueNDRangeKernel(queue, kernel3c, 2, NULL, pCGlobal, pLocal, 0, NULL, &event);
        #endif

        // Wait for calculations to be finished
        checkError(err,__LINE__);
        err = clWaitForEvents(1, &event);
    }

    // End the timed loop
    // timers[timerID].t += (timer() - startTime) / (double)NUM_RUNS;
    // timers[timerID].kf += ((long)K * (long)M * (long)N * 2) / 1000;

    // Copy the output matrix C back to the CPU memory
    err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, M*N*sizeof(*C), C, 0, NULL, NULL);
    checkError(err,__LINE__);

    // Free the memory objects
    free(code);
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufB_TR);
    clReleaseMemObject(bufC);
    clReleaseMemObject(bufA_XL);
    clReleaseMemObject(bufB_TR_XL);
    clReleaseMemObject(bufC_XL);

    // Clean-up OpenCL 
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseProgram(program);
    clReleaseKernel(kernel1);
    #if KERNEL == 5 || KERNEL == 6 || KERNEL == 7 || KERNEL == 8 || KERNEL == 9 || KERNEL == 10
        clReleaseKernel(kernel2);
    #endif
    #if KERNEL == 10
        clReleaseKernel(kernel3a);
        clReleaseKernel(kernel3b);
        clReleaseKernel(kernel3c);
    #endif
}

// =================================================================================================

// Print an error message to screen (only if it occurs)
void checkError(cl_int error, int line) {
    if (error != CL_SUCCESS) {
        switch (error) {
            case CL_DEVICE_NOT_FOUND:                 printf("-- Error at %d:  Device not found.\n", line); break;
            case CL_DEVICE_NOT_AVAILABLE:             printf("-- Error at %d:  Device not available\n", line); break;
            case CL_COMPILER_NOT_AVAILABLE:           printf("-- Error at %d:  Compiler not available\n", line); break;
            case CL_MEM_OBJECT_ALLOCATION_FAILURE:    printf("-- Error at %d:  Memory object allocation failure\n", line); break;
            case CL_OUT_OF_RESOURCES:                 printf("-- Error at %d:  Out of resources\n", line); break;
            case CL_OUT_OF_HOST_MEMORY:               printf("-- Error at %d:  Out of host memory\n", line); break;
            case CL_PROFILING_INFO_NOT_AVAILABLE:     printf("-- Error at %d:  Profiling information not available\n", line); break;
            case CL_MEM_COPY_OVERLAP:                 printf("-- Error at %d:  Memory copy overlap\n", line); break;
            case CL_IMAGE_FORMAT_MISMATCH:            printf("-- Error at %d:  Image format mismatch\n", line); break;
            case CL_IMAGE_FORMAT_NOT_SUPPORTED:       printf("-- Error at %d:  Image format not supported\n", line); break;
            case CL_BUILD_PROGRAM_FAILURE:            printf("-- Error at %d:  Program build failure\n", line); break;
            case CL_MAP_FAILURE:                      printf("-- Error at %d:  Map failure\n", line); break;
            case CL_INVALID_VALUE:                    printf("-- Error at %d:  Invalid value\n", line); break;
            case CL_INVALID_DEVICE_TYPE:              printf("-- Error at %d:  Invalid device type\n", line); break;
            case CL_INVALID_PLATFORM:                 printf("-- Error at %d:  Invalid platform\n", line); break;
            case CL_INVALID_DEVICE:                   printf("-- Error at %d:  Invalid device\n", line); break;
            case CL_INVALID_CONTEXT:                  printf("-- Error at %d:  Invalid context\n", line); break;
            case CL_INVALID_QUEUE_PROPERTIES:         printf("-- Error at %d:  Invalid queue properties\n", line); break;
            case CL_INVALID_COMMAND_QUEUE:            printf("-- Error at %d:  Invalid command queue\n", line); break;
            case CL_INVALID_HOST_PTR:                 printf("-- Error at %d:  Invalid host pointer\n", line); break;
            case CL_INVALID_MEM_OBJECT:               printf("-- Error at %d:  Invalid memory object\n", line); break;
            case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:  printf("-- Error at %d:  Invalid image format descriptor\n", line); break;
            case CL_INVALID_IMAGE_SIZE:               printf("-- Error at %d:  Invalid image size\n", line); break;
            case CL_INVALID_SAMPLER:                  printf("-- Error at %d:  Invalid sampler\n", line); break;
            case CL_INVALID_BINARY:                   printf("-- Error at %d:  Invalid binary\n", line); break;
            case CL_INVALID_BUILD_OPTIONS:            printf("-- Error at %d:  Invalid build options\n", line); break;
            case CL_INVALID_PROGRAM:                  printf("-- Error at %d:  Invalid program\n", line); break;
            case CL_INVALID_PROGRAM_EXECUTABLE:       printf("-- Error at %d:  Invalid program executable\n", line); break;
            case CL_INVALID_KERNEL_NAME:              printf("-- Error at %d:  Invalid kernel name\n", line); break;
            case CL_INVALID_KERNEL_DEFINITION:        printf("-- Error at %d:  Invalid kernel definition\n", line); break;
            case CL_INVALID_KERNEL:                   printf("-- Error at %d:  Invalid kernel\n", line); break;
            case CL_INVALID_ARG_INDEX:                printf("-- Error at %d:  Invalid argument index\n", line); break;
            case CL_INVALID_ARG_VALUE:                printf("-- Error at %d:  Invalid argument value\n", line); break;
            case CL_INVALID_ARG_SIZE:                 printf("-- Error at %d:  Invalid argument size\n", line); break;
            case CL_INVALID_KERNEL_ARGS:              printf("-- Error at %d:  Invalid kernel arguments\n", line); break;
            case CL_INVALID_WORK_DIMENSION:           printf("-- Error at %d:  Invalid work dimensionsension\n", line); break;
            case CL_INVALID_WORK_GROUP_SIZE:          printf("-- Error at %d:  Invalid work group size\n", line); break;
            case CL_INVALID_WORK_ITEM_SIZE:           printf("-- Error at %d:  Invalid work item size\n", line); break;
            case CL_INVALID_GLOBAL_OFFSET:            printf("-- Error at %d:  Invalid global offset\n", line); break;
            case CL_INVALID_EVENT_WAIT_LIST:          printf("-- Error at %d:  Invalid event wait list\n", line); break;
            case CL_INVALID_EVENT:                    printf("-- Error at %d:  Invalid event\n", line); break;
            case CL_INVALID_OPERATION:                printf("-- Error at %d:  Invalid operation\n", line); break;
            case CL_INVALID_GL_OBJECT:                printf("-- Error at %d:  Invalid OpenGL object\n", line); break;
            case CL_INVALID_BUFFER_SIZE:              printf("-- Error at %d:  Invalid buffer size\n", line); break;
            case CL_INVALID_MIP_LEVEL:                printf("-- Error at %d:  Invalid mip-map level\n", line); break;
            case -1024:                               printf("-- Error at %d:  *clBLAS* Functionality is not implemented\n", line); break;
            case -1023:                               printf("-- Error at %d:  *clBLAS* Library is not initialized yet\n", line); break;
            case -1022:                               printf("-- Error at %d:  *clBLAS* Matrix A is not a valid memory object\n", line); break;
            case -1021:                               printf("-- Error at %d:  *clBLAS* Matrix B is not a valid memory object\n", line); break;
            case -1020:                               printf("-- Error at %d:  *clBLAS* Matrix C is not a valid memory object\n", line); break;
            case -1019:                               printf("-- Error at %d:  *clBLAS* Vector X is not a valid memory object\n", line); break;
            case -1018:                               printf("-- Error at %d:  *clBLAS* Vector Y is not a valid memory object\n", line); break;
            case -1017:                               printf("-- Error at %d:  *clBLAS* An input dimension (M,N,K) is invalid\n", line); break;
            case -1016:                               printf("-- Error at %d:  *clBLAS* Leading dimension A must not be less than the size of the first dimension\n", line); break;
            case -1015:                               printf("-- Error at %d:  *clBLAS* Leading dimension B must not be less than the size of the second dimension\n", line); break;
            case -1014:                               printf("-- Error at %d:  *clBLAS* Leading dimension C must not be less than the size of the third dimension\n", line); break;
            case -1013:                               printf("-- Error at %d:  *clBLAS* The increment for a vector X must not be 0\n", line); break;
            case -1012:                               printf("-- Error at %d:  *clBLAS* The increment for a vector Y must not be 0\n", line); break;
            case -1011:                               printf("-- Error at %d:  *clBLAS* The memory object for Matrix A is too small\n", line); break;
            case -1010:                               printf("-- Error at %d:  *clBLAS* The memory object for Matrix B is too small\n", line); break;
            case -1009:                               printf("-- Error at %d:  *clBLAS* The memory object for Matrix C is too small\n", line); break;
            case -1008:                               printf("-- Error at %d:  *clBLAS* The memory object for Vector X is too small\n", line); break;
            case -1007:                               printf("-- Error at %d:  *clBLAS* The memory object for Vector Y is too small\n", line); break;
            case -1001:                               printf("-- Error at %d:  Code -1001: no GPU available?\n", line); break;
            default:                                  printf("-- Error at %d:  Unknown with code %d\n", line, error);
        }
        exit(1);
    }
}

// =================================================================================================
// Timer function: Measure the current time
double opencl_timer(void) {
    struct timeval Tvalue;
    struct timezone dummy;
    gettimeofday(&Tvalue, &dummy);
    double etime = (double)Tvalue.tv_sec + 1.0e-6*((double)Tvalue.tv_usec);
    return etime;
    //return omp_get_wtime();
}

// Timer function: Get the execution time
double opencl_wtime(profile_t timer) {
    return (timer.t);
}

// Timer function: Get the GFLOPS number
double opencl_gflops(profile_t timer) {
    return ((double)timer.kf/(1000.0*1000.0)) / (timer.t);
}

// Load an OpenCL kernel from file
char* readKernelFile(const char* filename, long* _size) {

    // Open the file
    errno = 0;
    char mypwd[200];
    char* test;
    test = getcwd(mypwd, 150);
    if(test == 0){}
    printf("%s\n",mypwd);
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("-- Error opening file %s 1111\n", filename);
        printf("error is %d\n",errno);
        exit(1);
    }

    // Get its size
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    rewind(file);

    // Read the kernel code as a string
    char* source = (char *)malloc((size+1)*sizeof(char));
    size_t f_l = fread(source, 1, size*sizeof(char), file);
    if(f_l < 0)
        exit(-1);
    printf("length is %d\n",(int)f_l);
    source[size] = '\0';
    fclose(file);

    // Save the size and return the source string
    *_size = (size+1);
    return source;
}
}