#pragma once
#include "core/var.h"

namespace jittor {

inline void cusparse_check_spmm_metadata(Var* output, Var* dense, Var* values,
    Var* columns, Var* rows, int a_rows, int a_columns, bool transpose_a,
    bool transpose_b, bool csr) {
    USER_CHECK(a_rows >= 0 && a_columns >= 0) << "cuSPARSE matrix dimensions must be nonnegative";
    USER_CHECK(dense->shape.size() == 2 && output->shape.size() == 2)
        << "cuSPARSE dense input and output must have rank two";
    USER_CHECK(values->shape.size() == 1 && columns->shape.size() == 1 && rows->shape.size() == 1)
        << "cuSPARSE values and index arrays must have rank one";
    USER_CHECK(columns->dtype() == ns_int32 || columns->dtype() == ns_int64)
        << "cuSPARSE indices require int32 or int64";
    USER_CHECK(rows->dtype() == columns->dtype()) << "cuSPARSE row and column index dtypes must match";
    USER_CHECK(columns->num == values->num) << "cuSPARSE column indices and values lengths must match";
    USER_CHECK(rows->num == (csr ? int64(a_rows)+1 : values->num))
        << "cuSPARSE row index/offset length does not match the sparse matrix";
    const int64 m = transpose_a ? a_columns : a_rows;
    const int64 k = transpose_a ? a_rows : a_columns;
    const int64 b_rows = dense->shape[transpose_b ? 1 : 0];
    const int64 n = dense->shape[transpose_b ? 0 : 1];
    USER_CHECK(k == b_rows) << "cuSPARSE inner matrix dimensions must match";
    USER_CHECK(output->shape[0] == m && output->shape[1] == n)
        << "cuSPARSE output shape does not match the requested matrix product";
}

} // namespace jittor
