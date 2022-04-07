// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
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

string format(const string& s, const vector<string>& v);

string replace(const string& a, const string& b, const string& c);

string join(const vector<string>& vs, const string& x);

vector<string> token_split(const string& s, bool exclude_comments=false);

int token_replace(vector<string>& tokens, int i, const string& src, const string& dst, bool match_whitespace=true);

string token_replace(const string& s, const string& src, const string& dst);

string token_replace_all(const string& s, const string& src, const string& dst);
} // jittor