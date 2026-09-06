/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 ******************************************************************************/

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cub/device/device_radix_sort.cuh>

struct CubTestPair {
    float key;
    int value;

    bool operator<(const CubTestPair& other) const {
        return key < other.key;
    }
};

static int cub_test_entry(int argc, char** argv) {
    int num_items = 150;
    for (int i = 0; i < argc; ++i) {
        if (std::strncmp(argv[i], "--n=", 4) == 0)
            num_items = std::atoi(argv[i] + 4);
    }
    if (num_items <= 0) {
        std::fprintf(stderr, "cub_test requires a positive item count\n");
        return 1;
    }

    std::vector<float> keys(num_items);
    std::vector<int> values(num_items);
    std::vector<CubTestPair> reference(num_items);
    std::vector<float> actual_keys(num_items);
    std::vector<int> actual_values(num_items);
    for (int i = 0; i < num_items; ++i) {
        // 37 is coprime with both maintained test sizes, so every key is unique.
        keys[i] = static_cast<float>((static_cast<long long>(i) * 37) % num_items)
            - static_cast<float>(num_items) / 2.0f;
        values[i] = i;
        reference[i] = {keys[i], values[i]};
    }
    std::stable_sort(reference.begin(), reference.end());

    float* device_keys_in = nullptr;
    float* device_keys_out = nullptr;
    int* device_values_in = nullptr;
    int* device_values_out = nullptr;
    void* temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    int result = 0;

#define CUB_TEST_CHECK(expression) do { \
    cudaError_t status = (expression); \
    if (status != cudaSuccess) { \
        std::fprintf(stderr, "CUB test CUDA error at %s: %s\n", \
            #expression, cudaGetErrorString(status)); \
        result = 1; \
        goto cleanup; \
    } \
} while (0)

    CUB_TEST_CHECK(cudaMalloc(&device_keys_in, sizeof(float) * num_items));
    CUB_TEST_CHECK(cudaMalloc(&device_keys_out, sizeof(float) * num_items));
    CUB_TEST_CHECK(cudaMalloc(&device_values_in, sizeof(int) * num_items));
    CUB_TEST_CHECK(cudaMalloc(&device_values_out, sizeof(int) * num_items));
    CUB_TEST_CHECK(cudaMemcpy(
        device_keys_in, keys.data(), sizeof(float) * num_items,
        cudaMemcpyHostToDevice));
    CUB_TEST_CHECK(cudaMemcpy(
        device_values_in, values.data(), sizeof(int) * num_items,
        cudaMemcpyHostToDevice));
    CUB_TEST_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, temp_storage_bytes, device_keys_in, device_keys_out,
        device_values_in, device_values_out, num_items));
    CUB_TEST_CHECK(cudaMalloc(&temp_storage, temp_storage_bytes));
    CUB_TEST_CHECK(cub::DeviceRadixSort::SortPairs(
        temp_storage, temp_storage_bytes, device_keys_in, device_keys_out,
        device_values_in, device_values_out, num_items));
    CUB_TEST_CHECK(cudaMemcpy(
        actual_keys.data(), device_keys_out, sizeof(float) * num_items,
        cudaMemcpyDeviceToHost));
    CUB_TEST_CHECK(cudaMemcpy(
        actual_values.data(), device_values_out, sizeof(int) * num_items,
        cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_items; ++i) {
        if (actual_keys[i] != reference[i].key ||
            actual_values[i] != reference[i].value) {
            std::fprintf(stderr,
                "CUB radix sort mismatch at %d: (%g, %d) != (%g, %d)\n",
                i, actual_keys[i], actual_values[i],
                reference[i].key, reference[i].value);
            result = 1;
            break;
        }
    }

cleanup:
    if (temp_storage) cudaFree(temp_storage);
    if (device_values_out) cudaFree(device_values_out);
    if (device_values_in) cudaFree(device_values_in);
    if (device_keys_out) cudaFree(device_keys_out);
    if (device_keys_in) cudaFree(device_keys_in);
#undef CUB_TEST_CHECK
    return result;
}
