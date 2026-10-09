// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/common.h"

namespace jittor {

// What a code operator's key carries in place of its source: this prefix and
// a digest of the header and source. A kernel carries kilobytes of CUDA text,
// and every execution of the operator copied and hashed all of it at least
// twice to find its compiled kernel -- 0.4 us of host time per kilobyte, 2-6 us
// an operator for the kernels of a BERT-base layer. The token is looked up from
// one pass of a fast hash and one comparison, and names the same kernel; the
// same text gives the same token in every process, so compiled kernels on disk
// are found again.
constexpr const char* code_source_prefix = "@jtsrc:";

// The token standing for `header` and `src`, interning them on first sight.
const string& code_source_token(const string& header, const string& src);

// The key tail -- header, lifted kernels, "«CODE:", the rest -- a token
// stands for, or nullptr if this process has not seen it.
const string* code_source_tail(const string& token);

// Whether `value` is such a token.
inline bool is_code_source_token(const string& value) {
    return value.compare(0, 7, code_source_prefix) == 0;
}

} // jittor
