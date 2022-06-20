// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <functional>

using namespace std;

void test_main();

void expect_error(function<void()> func);

int main() {
    try {
        test_main();
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
        return 1;
    }
}