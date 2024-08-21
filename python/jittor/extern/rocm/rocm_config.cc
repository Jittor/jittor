// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved.
// Maintainers: Zheng-Ning Liu <lzhengning@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "rocm_config.h"
#include "utils/str_utils.h"

namespace jittor
{
    void rocm_config(const string &name, string &src)
    {
        int pos;
        string error_token_substring = "benchmark = false;";
        string error_token_0 = "if (fwd_algo_cache.size()>=max_cache_size) " + error_token_substring;
        string error_token_1 = "if (bwdw_algo_cache.size()>=max_cache_size) " + error_token_substring;
        string error_token_2 = "if (bwdx_algo_cache.size()>=max_cache_size) " + error_token_substring;
        if ((pos = src.find(error_token_0)) != string::npos) {
            src.erase(pos, error_token_0.size());
        }
        if ((pos = src.find(error_token_1)) != string::npos) {
            src.erase(pos, error_token_1.size());
        }
        if ((pos = src.find(error_token_2)) != string::npos) {
            src.erase(pos, error_token_2.size());
        }

        string use_cub_where = "cub_where && (ndim>1 || std::abs(cond->num)>4096)";
        if ((pos = src.find(use_cub_where)) != string::npos) {
            src.replace(pos, use_cub_where.size(), "cub_where");
        }

        string enable_rocm = "HIP enabled";
        if ((pos = src.find(enable_rocm)) != string::npos) {
            src.replace(pos, enable_rocm.size(), "ROCm enabled");
        }
    }
}

