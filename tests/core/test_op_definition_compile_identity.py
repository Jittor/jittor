"""Replacing codegen must not reuse another pinned definition's JIT binary."""

import numpy as np


_PROBE = r"""
#pragma once
#include "op.h"
#include "var.h"
#include "var_holder.h"
#include "ops/op_register.h"
#include <stdexcept>
namespace jittor {
namespace {
OpDef code_original, fused_original, binary_original;
bool code_changed = false, fused_changed = false, binary_changed = false;
int fused_compiles = 0;
void code_override(Op* op, string& source) {
    auto original = code_original.implementations.at(BackendId::Cpu).codegen.optimize;
    if (original) original(op, source);
    source.insert(0, "#define DISPATCH_REPLACEMENT_OFFSET 7\n");
}
void observe_fused_compile(Op* op, string& source) {
    ++fused_compiles;
    auto original = fused_original.implementations.at(BackendId::Cpu).codegen.optimize;
    if (original) original(op, source);
}
void unchanged_binary_fragment(Op* op, JK& key) {
    binary_original.implementations.at(BackendId::Cpu).codegen.fragment(op, key);
}
}
// @pyjt(compile_identity_replace)
void compile_identity_replace(const string& name) {
    OpDef replacement = get_op_info(name);
    if (name == "code") {
        if (code_changed) throw std::runtime_error("code already replaced");
        code_original = replacement;
        replacement.implementations.at(BackendId::Cpu).codegen.optimize = code_override;
        code_changed = true;
    } else if (name == "fused") {
        if (fused_changed) throw std::runtime_error("fused already replaced");
        fused_original = replacement;
        replacement.implementations.at(BackendId::Cpu).codegen.optimize = observe_fused_compile;
        fused_compiles = 0;
        fused_changed = true;
    } else if (name == "binary") {
        if (binary_changed) throw std::runtime_error("binary already replaced");
        binary_original = replacement;
        replacement.implementations.at(BackendId::Cpu).codegen.fragment = unchanged_binary_fragment;
        binary_changed = true;
    } else throw std::runtime_error("unknown compile identity probe operator");
    op_registe(replacement);
}
// @pyjt(compile_identity_restore)
void compile_identity_restore() {
    if (code_changed) { op_registe(code_original); code_changed = false; }
    if (fused_changed) { op_registe(fused_original); fused_changed = false; }
    if (binary_changed) { op_registe(binary_original); binary_changed = false; }
}
// @pyjt(compile_identity_key)
string compile_identity_key(VarHolder* value) {
    return value->var->input()->get_jit_key(get_jk());
}
// @pyjt(compile_identity_fused_compiles)
int compile_identity_fused_compiles() { return fused_compiles; }
// @pyjt(compile_identity_generations)
vector<string> compile_identity_generations() {
    NativeOpRegistry registry;
    registry.register_op({"identity_probe", "identity_probe.cc", ""});
    auto first = registry.definition("identity_probe");
    registry.register_op(*first);
    auto second = registry.definition("identity_probe");
    registry.unregister("identity_probe");
    registry.register_op(*first);
    auto third = registry.definition("identity_probe");
    NativeOpRegistry independent;
    independent.register_op(*first);
    auto other_initial = independent.definition("identity_probe");
    independent.register_op(*first);
    return {first->compile_identity, second->compile_identity, third->compile_identity,
            other_initial->compile_identity, independent.definition("identity_probe")->compile_identity};
}
}
"""


def _probe(jt):
    import jittor_utils

    return jittor_utils.compile_module(_PROBE, jt.compiler.cc_flags)


def _code(jt, source):
    return jt.code(source.shape, source.dtype, [source], cpu_header="""
        #ifndef DISPATCH_REPLACEMENT_OFFSET
        #define DISPATCH_REPLACEMENT_OFFSET 2
        #endif
        """, cpu_src="""
        for (int i = 0; i < in0_shape0; ++i)
            @out(i) = @in0(i) + DISPATCH_REPLACEMENT_OFFSET;
        """)


def test_codegen_replacement_keeps_old_graph_binary_identity():
    import jittor as jt

    probe = _probe(jt)
    data = np.arange(29, dtype=np.float32)
    with jt.flag_scope(use_cuda=0, lazy_execution=1, auto_flush_ops=0,
                       use_parallel_op_compiler=0):
        jt.sync_all(True)
        source = jt.array(data).sync()
        old_graph = _code(jt, source)
        old_key = probe.compile_identity_key(old_graph)
        probe.compile_identity_replace("code")
        try:
            new_graph = _code(jt, source)
            new_key = probe.compile_identity_key(new_graph)
            assert new_key != old_key
            assert probe.compile_identity_key(old_graph) == old_key
            # Compile the replacement first so shared keys would also corrupt
            # the pending old graph, not only bypass the replacement callback.
            np.testing.assert_array_equal(new_graph.numpy(), data + 7)
            np.testing.assert_array_equal(old_graph.numpy(), data + 2)
        finally:
            try:
                jt.sync_all(True)
            finally:
                probe.compile_identity_restore()


def test_initial_identity_is_stable_but_replacements_and_reregistration_are_distinct():
    import jittor as jt

    first, second, third, other_initial, other_replaced = _probe(jt).compile_identity_generations()
    assert first == other_initial == ""
    assert second and third and other_replaced
    assert len({second, third, other_replaced}) == 3


def test_fused_cache_identity_includes_its_definition_and_child_definitions():
    import jittor as jt

    probe = _probe(jt)
    data = np.arange(263, dtype=np.float32) / 8
    expected = (data + 2) * 3
    with jt.flag_scope(use_cuda=0, lazy_execution=1, auto_flush_ops=0,
                       use_parallel_op_compiler=0):
        jt.sync_all(True)
        source = jt.array(data).sync()
        np.testing.assert_array_equal(((source + 2) * 3).numpy(), expected)
        probe.compile_identity_replace("fused")
        try:
            np.testing.assert_array_equal(((source + 2) * 3).numpy(), expected)
            first_compiles = probe.compile_identity_fused_compiles()
            assert first_compiles > 0
            probe.compile_identity_replace("binary")
            np.testing.assert_array_equal(((source + 2) * 3).numpy(), expected)
            assert probe.compile_identity_fused_compiles() > first_compiles
        finally:
            try:
                jt.sync_all(True)
            finally:
                probe.compile_identity_restore()
