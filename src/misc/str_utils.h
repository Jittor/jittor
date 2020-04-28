// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"

namespace jittor {

// a: main string
// b: pattern string
// start: start index(include)
// equal: match reach the end
// end: end index(exclude)
bool startswith(const string& a, const string& b, uint start=0, bool equal=false, uint end=0);

// a: main string
// b: pattern string
bool endswith(const string& a, const string& b);

// s: main string
// sep: pattern string for split
// max_split: maximun split number(include)
vector<string> split(const string& s, const string& sep, int max_split=0);

string strip(const string& s);

} // jittor