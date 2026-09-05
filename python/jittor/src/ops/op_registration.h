#pragma once
#include "op.h"
#include "var.h"
#include "ops/op_register.h"
#include <type_traits>

namespace jittor {

template<class T>
void register_op_definition(OpDef definition, uint32 backend_mask = T::backend_mask) {
    definition.codegen.fragment = [](Op* op, JK& key) {
        static_cast<T*>(op)->T::jit_prepare(key);
    };
    definition.codegen.prepare = prepare_registered_codegen;
    definition.codegen.optimize = [](Op* op, string& source) {
        static_cast<T*>(op)->T::compile_optimize(source);
    };
    Kernel kernel;
    // Graph-only operators deliberately inherit both no-op hooks. A class
    // with JIT preparation but no native implementation must not fall through
    // to the old empty Op::run if its preparation unexpectedly produces no key.
    constexpr bool native_override = !std::is_same<decltype(&T::run), decltype(&Op::run)>::value;
    constexpr bool has_jit = !std::is_same<decltype(&T::jit_prepare), decltype(&Op::jit_prepare)>::value;
    if (native_override || !has_jit)
        kernel.native = [](Op* op) { static_cast<T*>(op)->T::run(); };
    kernel.jit = execute_registered_jit;
    if (backend_mask & OpBackendCpu)
        definition.implementations.emplace(BackendId::Cpu, OpImplementation{kernel, definition.codegen});
    if (backend_mask & OpBackendAccelerator)
        definition.implementations.emplace(accelerator_backend_id(), OpImplementation{kernel, definition.codegen});
    op_registe(definition);
}

} // namespace jittor
