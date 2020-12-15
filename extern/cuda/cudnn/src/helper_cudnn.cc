#include <cudnn.h>
#include "helper_cuda.h"

const char *_cudaGetErrorEnum(cudnnStatus_t error) {
    return cudnnGetErrorString(error);
}