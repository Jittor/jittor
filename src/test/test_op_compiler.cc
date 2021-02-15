// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <regex>
#include "op_compiler.h"

namespace jittor {

JIT_TEST(regex) {
    std::string s(R"(
        asdas
void adasd 
asdads XxxXxxOp::jit_run() {
    xxxx
})");
    std::regex e(R"([^]*\s(\S*Op)::jit_run[^]*)");
    std::smatch cm;

    // std::regex_match ( s, cm, e, std::regex_constants::match_default );
    std::regex_match ( s, cm, e);

    CHECK(cm.size()==2);
    CHECK(cm[1]=="XxxXxxOp");
}

} // jittor
