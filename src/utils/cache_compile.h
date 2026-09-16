// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/common.h"

namespace jittor {
namespace jit_compiler {

string read_all(const string& fname);
void write(const string& fname, const string& src);
bool file_exist(const string& fname);
string join(string a, string b);
bool cache_compile(string cmd, const string& cache_path="", const string& jittor_path="");

// What cache_compile() would decide, without deciding it.
//
// cache_compile() answers "does this product need rebuilding?" by comparing
// the key the command would produce now against the key recorded in
// <output>.key. That comparison is pure reading, but it used to be reachable
// only from inside the build lock, so every cache *hit* paid for the lock as
// if it were a build. This runs the same comparison on its own.
struct CacheProbe {
    // The -o argument of the command: the product it would write.
    string output_name;
    // Where the key of that product is recorded, i.e. output_name + ".key".
    string key_name;
    // The key this command would produce right now.
    string cache_key;
    // The recorded key already equals cache_key: nothing to build.
    bool up_to_date = false;
};

// `in_memory_input_name` names one input of `cmd` whose content the caller
// holds in `in_memory_input_content` and has not written to disk yet. It is
// hashed from the string, which is what hashing the file it is about to
// become would give -- so a caller that only wants to know whether a build is
// needed does not have to perform the write first. Pass an empty name to read
// every input from disk, exactly as cache_compile() does.
CacheProbe cache_compile_probe(string cmd,
                               const string& in_memory_input_name = "",
                               const string& in_memory_input_content = "");

} // jit_compiler
} // jittor