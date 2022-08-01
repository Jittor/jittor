// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved.
// Maintainers: Zheng-Ning Liu <lzhengning@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "rocm_translator.h"
#include "utils/str_utils.h"

namespace jittor {

void rocm_translate(string& src) {
    auto tokens = token_split(src);

    for (int i = 0; i < tokens.size(); ++i) {
        auto &token = tokens[i];

        if (token == "cub::CountingInputIterator") {
            token = "rocprim::counting_iterator";
        } else if (token == "cub::DeviceSelect::Flagged") {
            token_replace(tokens, i, "cub::DeviceSelect::Flagged($1,$2,$3,$4,$5,$6,$7)", "rocprim::select($1,$2,$3,$4,$5,$6,$7)", false);
        } else if (token == "cub::TransformInputIterator") {
            token_replace(tokens, i, "cub::TransformInputIterator<$1,$2,$3>", "rocprim::transform_iterator<$3,$2,$1>");
        } else if (token == "__shfl_sync") {
            token = "__shfl";
        } else if (token == "__shfl_up_sync") {
            token = "__shfl_up";
        }
    }
    
    src = join(tokens, "");
}

}