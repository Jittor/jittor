#pragma once
#include "common.h"
#include "runtime/backend.h"
#include "jit_key.h"

namespace jittor {

struct Op;
struct OpDef;

constexpr uint32 OpBackendCpu = 1;
constexpr uint32 OpBackendAccelerator = 2;
constexpr uint32 OpBackendAny = OpBackendCpu | OpBackendAccelerator;

struct Codegen {
    string source_path, extra_flags;
    vector<pair<string, uint64>> var_members;
    void (*fragment)(Op*, JK&) = nullptr;
    void (*prepare)(Op*, JK&) = nullptr;
    void (*optimize)(Op*, string&) = nullptr;
};

struct Kernel {
    void (*native)(Op*) = nullptr;
    void (*jit)(Op*, JK&) = nullptr;
    jit_op_entry_t (*compile)(Op*) = nullptr;
    bool fallback_only = false;
};

struct OpImplementation {
    Kernel kernel;
    Codegen codegen;
};

EXTERN_LIB void prepare_registered_codegen(Op* op, JK& key);
EXTERN_LIB void execute_registered_jit(Op* op, JK& key);
EXTERN_LIB jit_op_entry_t compile_registered_source(Op* op);
EXTERN_LIB shared_ptr<const OpDef> get_op_definition(const string& name, bool required = true);

} // namespace jittor
