// ***************************************************************
// Copyright (c) 2023 Jittor.
// All Rights Reserved. 
// Maintainers:
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <immintrin.h>
#include <unistd.h>
#include <stdint.h>
#include <stdio.h>

#include "mpi_wrapper.h"
#include "common.h"
#include "ops/array_op.h"

char jt_mpi_err_buffer[MPI_MAX_ERROR_STRING];

void throw_mpi_error(int result, 
    char const *const func, const char *const file, int const line) {
    int resultlen;
    MPI_Error_string(result, jt_mpi_err_buffer, &resultlen);
    LOGf << "MPI error at " >> file >> ":" >> line << "code="
        >> result >> '(' >> jt_mpi_err_buffer >> ')' << func;
}

namespace jittor {

MPI_Datatype MPI_HALF;
MPI_Op MPI_HALF_ADD;

void HalfAdd(void* invec, void* inoutvec, int* len, MPI_Datatype* type) {
    // return;
    short* in = (short*)invec;
    short* inout = (short*)inoutvec;

    int i = 0;
    for (; i < (*len / 16) * 16; i += 16) {
        // 将半精度浮点数转换为单精度浮点数
        __m256 in1 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(in + i)));
        __m256 in2 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(inout + i)));

        // 执行向量加法
        __m256 out = _mm256_add_ps(in1, in2);

        // 将单精度浮点数转换回半精度浮点数，并存储结果
        _mm_storeu_si128((__m128i*)(inout + i), _mm256_cvtps_ph(out, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    }

    // 处理剩余的半精度浮点数
    for (; i < *len; i++) {
        // 将半精度浮点数转换为单精度浮点数
        __m128 in1 = _mm_cvtph_ps(_mm_set1_epi16(*(in + i)));
        __m128 in2 = _mm_cvtph_ps(_mm_set1_epi16(*(inout + i)));

        // 执行向量加法
        __m128 out = _mm_add_ps(in1, in2);

        // 将单精度浮点数转换回半精度浮点数，并存储结果
        *(inout + i) = _mm_cvtps_ph(out, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)[0];
    }
}


int mpi_world_size = 1;
int mpi_world_rank = 0;
int mpi_local_size = 1;
int mpi_local_rank = 0;
bool inside_mpi = false;
bool mpi_enabled = false;
bool use_device_mpi = false;

int _mpi_world_size() {
    return mpi_enabled ? mpi_world_size : 1;
}

int _mpi_world_rank() {
    return mpi_enabled ? mpi_world_rank : 0;
}

int _mpi_local_rank() {
    return mpi_enabled ? mpi_local_rank : 0;
}

void _mpi_broadcast(ArrayArgs&& args, int root) {
    if (!mpi_enabled) return;
    int64 size = args.dtype.dsize();
    for (auto j : args.shape)
        size *= j;
    MPI_CHECK(MPI_Bcast((void *)args.ptr, size, MPI_BYTE, root, MPI_COMM_WORLD));
}

static uint64_t getHostHash(const char* string) {
    // Based on DJB2, result = result * 33 + char
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++){
        result = ((result << 5) + result) + string[c];
    }
    return result;
}


static void getHostName(char* hostname, int maxlen) {
    gethostname(hostname, maxlen);
    for (int i=0; i< maxlen; i++) {
        if (hostname[i] == '.') {
            hostname[i] = '\0';
            return;
        }
    }
}

struct mpi_initer {

mpi_initer() {
    inside_mpi = !!getenv("OMPI_COMM_WORLD_SIZE");
    if (!inside_mpi) return;
    mpi_enabled = true;
    LOGvv << "MPI init...";
    MPI_CHECK(MPI_Init(NULL, NULL));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_world_rank));

    //calculating localRank based on hostname which is used in selecting a GPU
    uint64_t hostHashs[mpi_world_rank];
    char hostname[1024];
    getHostName(hostname, 1024);
    hostHashs[mpi_world_rank] = getHostHash(hostname);
    MPI_CHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
    mpi_local_rank = 0;
    for (int p=0; p<mpi_world_size; p++) {
        if (p == mpi_world_rank) break;
        if (hostHashs[p] == hostHashs[mpi_world_rank]) mpi_local_rank++;
    }
    mpi_local_size = 0;
    for (int p=0; p<mpi_world_size; p++) {
        if (hostHashs[p] == hostHashs[mpi_world_rank]) mpi_local_size++;
    }
    LOGv << "MPI init finished: local" << mpi_local_rank
        << "size" << mpi_local_size
        << "global" << mpi_world_rank
        << "size" << mpi_world_size;
        
    // init mpi half type
    MPI_Type_contiguous(1, MPI_SHORT, &MPI_HALF);
    MPI_Type_commit(&MPI_HALF);
    MPI_Op_create(HalfAdd, /* commute= */1, &MPI_HALF_ADD);

}

~mpi_initer() {
    if (!inside_mpi) return;
    MPI_Type_free(&MPI_HALF);
    MPI_Op_free(&MPI_HALF_ADD);
    MPI_CHECK(MPI_Finalize());
}

};

static mpi_initer mpi_init;


void var_broadcast(VarHolder* x, int root) {
    Var* v = x->var;
    ASSERT(v->mem_ptr && !v->allocator->is_cuda());
    MPI_Bcast(v->mem_ptr, v->size/8, MPI_DOUBLE, root, MPI_COMM_WORLD);
}

void var_reduce(VarHolder* x, int root) {
    Var* v = x->var;
    ASSERT(v->mem_ptr && !v->allocator->is_cuda());
    MPI_Datatype dtype;
    MPI_Op op;
    if (v->dtype() == ns_float16)
        dtype = MPI_HALF, op = MPI_HALF_ADD;
    else if (v->dtype() == ns_float32)
        dtype = MPI_FLOAT, op = MPI_SUM;
    else if (v->dtype() == ns_float64)
        dtype = MPI_DOUBLE, op = MPI_SUM;
    else if (v->dtype() == ns_int32)
        dtype = MPI_INT, op = MPI_SUM;
    else if (v->dtype() == ns_int64)
        dtype = MPI_LONG_LONG_INT, op = MPI_SUM;
    else if (v->dtype() == ns_uint8)
        dtype = MPI_UNSIGNED_CHAR, op = MPI_SUM;
    else
        LOGf << "Not supported dtype" << v->dtype();
    // LOGir << mpi_world_rank << "reduce" << v;
    if (mpi_world_rank == root)
        MPI_Reduce(MPI_IN_PLACE, v->mem_ptr, v->num, dtype, op, root, MPI_COMM_WORLD);
    else
        MPI_Reduce(v->mem_ptr, nullptr, v->num, dtype, op, root, MPI_COMM_WORLD);
}

void var_all_reduce(VarHolder* x) {
    Var* v = x->var;
    // LOGir << "nccl_all_reduce" << v;
    ASSERT(v->mem_ptr && !v->allocator->is_cuda());
    // LOGir << "is_cuda" << v->allocator->is_cuda() << v->mem_ptr << v->num;
    // for (int i=0; i<100; i++) LOGir << "???" << v;
    if (v->dtype() == ns_float16)
        MPI_Allreduce(MPI_IN_PLACE, v->mem_ptr, v->num, MPI_HALF, MPI_HALF_ADD, MPI_COMM_WORLD);
    else if (v->dtype() == ns_float32)
        MPI_Allreduce(MPI_IN_PLACE, v->mem_ptr, v->num, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    else if (v->dtype() == ns_float64)
        MPI_Allreduce(MPI_IN_PLACE, v->mem_ptr, v->num, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    else if (v->dtype() == ns_int32)
        MPI_Allreduce(MPI_IN_PLACE, v->mem_ptr, v->num, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    else if (v->dtype() == ns_int64)
        MPI_Allreduce(MPI_IN_PLACE, v->mem_ptr, v->num, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
    else if (v->dtype() == ns_uint8)
        MPI_Allreduce(MPI_IN_PLACE, v->mem_ptr, v->num, MPI_UNSIGNED_CHAR, MPI_SUM, MPI_COMM_WORLD);
    else
        LOGf << "Not supported dtype" << v->dtype();
}


} // jittor