#include <cudnn.h>
#include "utils/log.h"
#include "helper_cuda.h"

const char *_cudaGetErrorEnum(cudnnStatus_t error) {
    return cudnnGetErrorString(error);
}