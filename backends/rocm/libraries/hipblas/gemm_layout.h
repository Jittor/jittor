#pragma once

#include <algorithm>
#include <climits>
#include <cstdint>
#include <stdexcept>

namespace jittor {

struct HipblasGemmLayout {
    int rows, columns, inner, lda, ldb, ldc;
};

inline HipblasGemmLayout hipblas_gemm_layout(
    int64_t a_rows, int64_t a_columns, int64_t b_rows, int64_t b_columns,
    bool trans_a, bool trans_b) {
    for (auto size : {a_rows, a_columns, b_rows, b_columns})
        if (size < 0 || size > INT_MAX)
            throw std::invalid_argument("hipBLAS matmul dimensions must fit nonnegative int32");
    const int rows = trans_a ? a_columns : a_rows;
    const int inner = trans_a ? a_rows : a_columns;
    const int columns = trans_b ? b_rows : b_columns;
    if (inner != (trans_b ? b_columns : b_rows))
        throw std::invalid_argument("hipBLAS matmul inner dimensions must match");
    // Row-major C=A*B is column-major C^T=B^T*A^T, with operands exchanged.
    return {rows, columns, inner, std::max(1, int(b_columns)),
            std::max(1, int(a_columns)), std::max(1, columns)};
}

} // namespace jittor
